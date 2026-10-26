"""Predeclared acute evidence interventions, not retrained causal ablations.

All variants use the full D checkpoint's locked validation threshold. No test
access and no further threshold selection. Distribution shift is a limitation.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from deepblink_torch.cli import predict
from deepblink_torch.evaluation import match_coordinates
from deepblink_torch.preprocessing import prepare_image
from optimize_v2 import OUT, decode_grid, load_development, write_csv, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "xpu"], default="xpu")
    parser.add_argument("--model", choices=["deeploci_evidence", "deeploci_feature_evidence"], default="deeploci_evidence")
    args = parser.parse_args()
    folder = OUT / "02_architecture" / args.model / "seed_42"
    val = json.loads((folder / "validation_v2/validation_summary.json").read_text())
    threshold = val["threshold"]
    torch.set_num_threads(4)
    model, _ = predict._load_model(folder / "best_model.pt", torch.device(args.device))
    data = load_development("valid")
    names = ("full", "fixed_cell_center", "zero_image_evidence", "zero_entire_residual")
    totals = {name: dict(tp=0, fp=0, fn=0) for name in names}
    rows = []
    correction_magnitudes = []
    for raw, shape, sid, truth in zip(data["x"], data["original_shape"], data["sample_id"], data["y"]):
        h, w = map(int, shape)
        image = torch.from_numpy(prepare_image(raw[:h, :w], 512)[None, None]).to(args.device)
        mask = torch.zeros_like(image)
        mask[..., :h, :w] = 1
        features = model._forward_features(image)
        logits = model.logit_conv(features)
        offsets = torch.sigmoid(logits[:, 1:])
        projected = model.feature_projection(features)
        full_evidence = model.evidence_reader(image, offsets, mask)
        fixed_evidence = model.evidence_reader(image, torch.full_like(offsets, .5), mask)
        evidence = dict(full=full_evidence, fixed_cell_center=fixed_evidence,
                        zero_image_evidence=torch.zeros_like(full_evidence))
        grids = {}
        for name in names:
            correction = (torch.zeros_like(logits[:, :1]) if name == "zero_entire_residual" else
                          model.evidence_fusion(torch.cat([projected, evidence[name]], 1)))
            occupancy = torch.sigmoid(logits[:, :1] + correction)
            grids[name] = torch.cat([occupancy, offsets], 1).permute(0, 2, 3, 1)[0].cpu().numpy()
            if name == "full":
                correction_magnitudes.append(float(correction[..., :h//4, :w//4].abs().mean()))
            coords, _ = decode_grid(grids[name], shape, threshold)
            match = match_coordinates(coords, truth, 3.)
            row = dict(sample_id=str(sid), intervention=name, tp=match.tp, fp=match.fp, fn=match.fn)
            rows.append(row)
            for key in ("tp", "fp", "fn"):
                totals[name][key] += row[key]
        direct = model(image, valid_mask=mask)[0].cpu().numpy()
        np.testing.assert_allclose(grids["full"], direct, rtol=1e-5, atol=1e-6)
    results = []
    for name in names:
        tp, fp, fn = [totals[name][k] for k in ("tp", "fp", "fn")]
        results.append(dict(intervention=name, threshold=threshold, tp=tp, fp=fp, fn=fn,
                       precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                       f1=2*tp/(2*tp+fp+fn)))
    agrees = (totals["full"]["tp"], totals["full"]["fp"], totals["full"]["fn"]) == (val["tp"], val["fp"], val["fn"])
    if args.device == "xpu":
        assert agrees
    destination = folder / ("mechanism_probes" if args.device == "xpu" else "mechanism_probes_cpu")
    write_csv(destination / "intervention_summary.csv", results)
    write_csv(destination / "intervention_per_image.csv", rows)
    write_json(destination / "probe_metadata.json", dict(validation_only=True, retrained_ablations=False,
               device=args.device, precision="float32", full_counts_agree_with_primary_xpu=agrees,
               thresholds_recalibrated_for_interventions=False,
               mean_image_mean_abs_logit_correction=float(np.mean(correction_magnitudes)),
               interpretation="Acute dependence only; distribution shift possible; cannot establish structural attribution"))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
