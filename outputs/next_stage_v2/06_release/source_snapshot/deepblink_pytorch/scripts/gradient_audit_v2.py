"""Train-split gradient diagnostics for the D occupancy-to-offset path.

Describes one trained model at fixed training examples, not causal evidence
that gradient conflict produced its validation performance.
"""
import numpy as np
import argparse
import torch
from deepblink_torch.cli import predict
from deepblink_torch.data_protocol import get_prediction_matrix
from deepblink_torch.losses import dice_loss, rmse_loss
from deepblink_torch.preprocessing import prepare_image
from optimize_v2 import OUT, load_development, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["deeploci_evidence", "deeploci_feature_evidence"], default="deeploci_evidence")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = OUT / "02_architecture" / args.model / "seed_42"
    model, _ = predict._load_model(folder / "best_model.pt", torch.device("cpu"))
    data = load_development("train")
    indices = np.random.default_rng(20261025).choice(len(data["x"]), 24, replace=False)
    rows = []
    per_image = []
    for index in indices:
        raw, shape, truth, sid = [data[key][index] for key in ("x", "original_shape", "y", "sample_id")]
        h, w = map(int, shape)
        image = torch.from_numpy(prepare_image(raw[:h, :w], 512)[None, None])
        mask = torch.zeros_like(image)
        mask[..., :h, :w] = 1
        target = torch.from_numpy(get_prediction_matrix(truth, 512, 4)[None]).float()
        target[:, h//4:, :, 0] = -1
        target[:, :, w//4:, 0] = -1
        with torch.no_grad():
            features = model._forward_features(image)
            logits = model.logit_conv(features)
            projected = model.feature_projection(features)
        logits = logits.detach().requires_grad_(True)
        offsets = torch.sigmoid(logits[:, 1:])
        evidence = model.evidence_reader(image, offsets, mask)
        correction = model.evidence_fusion(torch.cat([projected, evidence], 1))
        p = torch.sigmoid(logits[:, :1] + correction)
        prediction = torch.cat([p, offsets], 1).permute(0, 2, 3, 1)
        gd = torch.autograd.grad(dice_loss(target, prediction), offsets, retain_graph=True)[0]
        gr = torch.autograd.grad(2*rmse_loss(target, prediction), offsets)[0]
        gd = gd[0].permute(1, 2, 0).detach().numpy()
        gr = gr[0].permute(1, 2, 0).detach().numpy()
        positives = target[0, ..., 0].numpy() > .5
        valid = target[0, ..., 0].numpy() >= 0
        dnorm, rnorm = np.linalg.norm(gd, axis=-1), np.linalg.norm(gr, axis=-1)
        for y, x in np.argwhere(positives):
            denominator = float(dnorm[y, x]*rnorm[y, x])
            cosine = float(np.dot(gd[y, x], gr[y, x])/denominator) if denominator > 1e-20 else None
            rows.append(dict(sample_id=str(sid), cell_y=int(y), cell_x=int(x),
                             dice_offset_gradient_norm=float(dnorm[y, x]),
                             two_rmse_offset_gradient_norm=float(rnorm[y, x]), cosine=cosine))
        per_image.append(dict(sample_id=str(sid), n_positive_cells=int(positives.sum()),
                              dice_grad_l2_valid=float(np.linalg.norm(gd[valid])),
                              dice_grad_l2_negative=float(np.linalg.norm(gd[valid & ~positives])),
                              two_rmse_grad_l2=float(np.linalg.norm(gr[valid]))))
    cosines = [row["cosine"] for row in rows if row["cosine"] is not None]
    ratios = [row["dice_offset_gradient_norm"]/row["two_rmse_offset_gradient_norm"] for row in rows
              if row["two_rmse_offset_gradient_norm"] > 1e-12]
    summary = dict(subset="train", n_images=24, sample_selection_seed=20261025,
                   checkpoint=f"{args.model} seed42 minimum validation loss", n_positive_cells=len(rows),
                   n_nonzero_gradient_pairs=len(cosines), negative_cosine_fraction=float(np.mean(np.asarray(cosines)<0)),
                   median_cosine=float(np.median(cosines)),
                   median_dice_to_two_rmse_gradient_norm_ratio=float(np.median(ratios)),
                   interpretation="Conditional head-path gradients only, not parameter-space gradients or causal performance attribution",
                   additional_training=False, labels_changed=False, test_access=False)
    destination = folder / "gradient_audit"
    write_csv(destination / "per_positive_cell.csv", rows)
    write_csv(destination / "per_image.csv", per_image)
    write_json(destination / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
