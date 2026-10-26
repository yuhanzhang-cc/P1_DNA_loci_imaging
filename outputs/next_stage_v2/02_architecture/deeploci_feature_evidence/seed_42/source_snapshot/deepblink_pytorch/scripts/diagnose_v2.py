"""Descriptive development errors and predeclared mechanism proxies.

Unmatched predictions are annotation-relative FP, not proof of biology or
annotation completeness. Human annotation review remains a separate task.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from error_context_diagnostics import _extract_patch_features, _extract_patch
from optimize_v2 import OUT, decode_grid, load_development, write_csv, write_json
from deepblink_torch.evaluation import match_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUT / "00_protocol/reference_seed_42")
    parser.add_argument("--output", type=Path, default=OUT / "08_innovation/development_audit")
    args = parser.parse_args()
    dev = load_development("valid")
    training = load_development("train")
    prereg_path = OUT / "08_innovation/mechanism_preregistration.json"
    if prereg_path.exists():
        prereg = json.loads(prereg_path.read_text())
    else:
        zscores = []
        for raw, truth, shape in zip(training["x"], training["y"], training["original_shape"]):
            h, w = map(int, shape)
            for y, x in truth:
                z = _extract_patch_features(raw[:h, :w], y, x, 10)["local_zscore"]
                if np.isfinite(z):
                    zscores.append(z)
        prereg = dict(weak_truth_local_zscore_q25=float(np.quantile(zscores, .25)),
                      cutoff_source="176 training images, finite truth-centered 21x21 patches",
                      n_training_truth=len(zscores), off_peak_fp_cutoff_px=3.,
                      point_radius_px=3., patch_radius_px=10,
                      primary="validation point micro F1 at registered validation threshold",
                      mechanism=["off_peak_annotation_relative_fp_count", "weak_truth_recall",
                                 "annotation_relative_fp_with_probability_gt_0_9"],
                      falsification="No F1 gain, unchanged target errors, or weak recall tradeoff; compare simple head and center-fixed ablation",
                      no_claim_of_background_segmentation_ground_truth=True,
                      human_blinded_review_status="pending", test_evaluated=False)
        write_json(prereg_path, prereg)
    threshold = json.loads((args.predictions / "validation_summary.json").read_text())["threshold"]
    with np.load(args.predictions / "prediction_grids.npz") as archive:
        if list(archive["sample_id"]) != list(dev["sample_id"]):
            raise ValueError("Prediction sample order differs from frozen validation.")
        grids = archive["grids"]
    events, patches = [], []
    weak_tp = weak_n = off_peak = high_confidence_fp = 0
    for raw, truth, shape, sid, grid in zip(dev["x"], dev["y"], dev["original_shape"], dev["sample_id"], grids):
        h, w = map(int, shape)
        image = raw[:h, :w]
        coords, scores = decode_grid(grid, shape, threshold)
        match = match_coordinates(coords, truth, 3.)
        matched_pred, matched_true = set(match.matched_pred_indices), set(match.matched_true_indices)
        for i, (y, x) in enumerate(truth):
            features = _extract_patch_features(image, y, x, 10)
            weak = features["local_zscore"] < prereg["weak_truth_local_zscore_q25"]
            weak_n += int(weak)
            weak_tp += int(weak and i in matched_true)
        candidates = [("TP" if i in matched_pred else "FP", y, x, float(scores[i]))
                      for i, (y, x) in enumerate(coords)]
        candidates += [("FN", *truth[i], None) for i in match.unmatched_true_indices]
        for status, y, x, score in candidates:
            features = _extract_patch_features(image, y, x, 10)
            off_peak += int(status == "FP" and features["peak_offset_px"] > 3.)
            high_confidence_fp += int(status == "FP" and score > .9)
            row = dict(sample_id=str(sid), status=status, y=float(y), x=float(x),
                       probability=score, **features)
            row = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
            events.append(row)
            patches.append((row, _extract_patch(image, y, x, 10)))
    summary = dict(n_validation_images=22, threshold=threshold, weak_truth_count=weak_n,
                   weak_truth_tp=weak_tp, weak_truth_recall=weak_tp/weak_n if weak_n else None,
                   off_peak_fp_count=off_peak, high_confidence_fp_count=high_confidence_fp,
                   test_evaluated=False, human_blinded_review_status="pending", events={})
    for status in ("TP", "FP", "FN"):
        rows = [r for r in events if r["status"] == status]
        summary["events"][status] = dict(count=len(rows), medians={
            k: float(np.median([r[k] for r in rows if r[k] is not None]))
            for k in ("local_zscore", "peak_offset_px", "peak_component_size_px")})
    write_csv(args.output / "point_events.csv", events)
    write_csv(args.output / "annotation_uncertainty_queue.csv", [dict(**r, reviewer_status="unreviewed",
              reason="unmatched prediction requires image and annotation review") for r in events if r["status"] == "FP"])
    write_json(args.output / "error_summary.json", summary)
    fig, axes = plt.subplots(3, 8, figsize=(16, 6), dpi=150)
    for row_index, status in enumerate(("TP", "FP", "FN")):
        chosen = [(row, patch) for row, patch in patches if row["status"] == status]
        chosen.sort(key=lambda item: (-(item[0]["probability"] or 0), item[0]["sample_id"]))
        for ax, (row, patch) in zip(axes[row_index], chosen[:8]):
            lo, hi = np.quantile(patch, [.01, .99])
            ax.imshow(patch, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"{row['sample_id']} {status}\nz={row['local_zscore']:.2f}", fontsize=8)
            ax.axis("off")
        for ax in axes[row_index, len(chosen[:8]):]:
            ax.axis("off")
    fig.suptitle("Development validation patches; descriptive examples, annotation review pending")
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "development_patches.png")
    plt.close(fig)
    pool = [(str(sid), subset) for subset, data in (("train", training), ("valid", dev)) for sid in data["sample_id"]]
    rng = np.random.default_rng(20261025)
    selected = rng.choice(len(pool), 40, replace=False)
    queue = OUT / "08_innovation/blinded_annotation_review_queue.csv"
    if not queue.exists():
        write_csv(queue, [dict(sample_id=pool[i][0], split=pool[i][1], review_status="pending",
                  method_identity="hidden", reviewer="",
                  notes="No relabeling or exclusion authorized by automated diagnostics") for i in selected])
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
