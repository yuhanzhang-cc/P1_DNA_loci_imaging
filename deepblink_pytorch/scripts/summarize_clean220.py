"""Post-training comparison and review bundle; never selects a model on test."""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import tifffile

from run_clean220 import OLD, OUT, evaluate_one, write_csv, write_json
from error_context_diagnostics import _extract_patch_features
from deepblink_torch.cli.train import _plot_training_curves


def main():
    results = json.loads((OUT/"comparison.json").read_text())
    config = json.loads((OUT/"experiment_config.json").read_text())
    device = config["device"]
    torch.set_num_threads(config["threads"])
    # Same frozen weights + the new canvas isolates a preprocessing component.
    derivative = OUT/"historical_fixed_canvas/weights_fixed_canvas.pt"
    if not derivative.exists():
        derivative.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(OLD/"best_model_clean_113.pt", map_location="cpu", weights_only=False)
        checkpoint["input_size"] = 512
        checkpoint["preprocessing"] = "normalize_original_then_zero_pad_right_bottom"
        checkpoint["derivative_note"] = "Unchanged historical weights; fixed canvas only, no retraining."
        torch.save(checkpoint, derivative)
    valid = evaluate_one(derivative, OUT/"dna_clean_220.npz",
                         derivative.parent/"validation", "valid", sweep=True, device=device)
    p = valid["probability"]
    write_json(derivative.parent/"threshold_lock.json", dict(probability=p, selected_on="valid"))
    results = [r for r in results if r["model"] != "historical_fixed_canvas"]
    for cohort in ["old113", "new107"]:
        test = evaluate_one(derivative, OUT/f"test_{cohort}.npz",
                            derivative.parent/f"test_{cohort}", "test", p, device=device)
        results.append(dict(model="historical_fixed_canvas", cohort=cohort, probability=p,
                            n_images=test["n_images"], tp=test["micro_tp"], fp=test["micro_fp"],
                            fn=test["micro_fn"], precision=test["micro_precision"],
                            recall=test["micro_recall"], f1=test["micro_f1"],
                            median_error_px=test["median_error_px"]))
    write_csv(OUT/"comparison.csv", results)
    write_json(OUT/"comparison.json", results)
    histories = {}
    for seed in config["seeds"]:
        name = f"seed_{seed}"
        with (OUT/name/"training_metrics.csv").open() as handle:
            histories[name] = list(csv.DictReader(handle))
        _plot_training_curves(OUT/name/"training_curves.png", histories[name])
    best_rows = {name:min(rows, key=lambda r:float(r["val_loss"])) for name,rows in histories.items()}
    selected = min(best_rows, key=lambda name:float(best_rows[name]["val_loss"]))
    lock = json.loads((OUT/selected/"threshold_lock.json").read_text())
    checkpoint = torch.load(OUT/selected/"best_model.pt", map_location="cpu", weights_only=False)
    checkpoint["calibrated_probability"] = lock["probability"]
    checkpoint["selection_rule"] = "minimum validation loss among seeds; threshold selected on validation"
    torch.save(checkpoint, OUT/"recommended_model.pt")
    write_json(OUT/"recommended_model.json", dict(selected_seed=selected,
               epoch=int(best_rows[selected]["epoch"]), validation_loss=float(best_rows[selected]["val_loss"]),
               probability=lock["probability"], input_size=512,
               model="UNetDeepBlink", parameter_count=1258723,
               inference_note="Pass --probability explicitly; checkpoint stores calibrated_probability for provenance."))
    manifest = {r["sample_id"]:r for r in csv.DictReader((OUT/"frozen_split.csv").open())}
    strata = []
    for model in [*histories, "historical_baseline", "historical_fixed_canvas"]:
        for cohort in ["old113", "new107"]:
            with (OUT/model/f"test_{cohort}/summary/per_image_metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            for dimension in ["size", "count"]:
                groups = defaultdict(list)
                for row in rows:
                    info = manifest[row["sample_id"]]
                    n = int(info["n_points"])
                    group = info["height"] if dimension == "size" else "1-5" if n <= 5 else "6-20" if n <= 20 else ">20"
                    groups[group].append(row)
                for group,items in groups.items():
                    tp,fp,fn = [sum(float(r[k]) for r in items) for k in ["tp", "fp", "fn"]]
                    strata.append(dict(model=model, cohort=cohort, dimension=dimension, group=group,
                                       n_images=len(items), tp=tp, fp=fp, fn=fn,
                                       precision=tp/(tp+fp) if tp+fp else 0,
                                       recall=tp/(tp+fn) if tp+fn else 0,
                                       f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))
    write_csv(OUT/"stratified_metrics.csv", strata)
    selected_dir = OUT/selected
    (OUT/"review_overlays").mkdir(exist_ok=True)
    with (selected_dir/"test_new107/summary/per_image_metrics.csv").open() as handle:
        cases = sorted(csv.DictReader(handle), key=lambda r:float(r["f1"]))[:5]
    for case in cases:
        shutil.copy2(selected_dir/"test_new107/match_overlays"/(case["sample_id"]+".png"), OUT/"review_overlays")
    means = {}
    for cohort in ["old113", "new107"]:
        values = [r["f1"] for r in results if r["model"] in histories and r["cohort"] == cohort]
        means[cohort] = dict(mean_f1=float(np.mean(values)), sd_f1=float(np.std(values, ddof=1)),
                             min_f1=min(values), max_f1=max(values))
    write_json(OUT/"replicate_summary.json", means)
    events = []
    for model in [selected, "historical_baseline"]:
        for match_path in (OUT/model/"test_new107/matches").glob("*.csv"):
            image = tifffile.imread(OUT.parents[1]/"data/raw/tif"/(match_path.stem+".tif"))
            with match_path.open() as handle:
                for row in csv.DictReader(handle):
                    status = row["match_status"].upper()
                    prefix = "true" if status == "FN" else "pred"
                    y, x = float(row[f"{prefix}_y_px"]), float(row[f"{prefix}_x_px"])
                    features = _extract_patch_features(image, y, x, radius=10)
                    events.append(dict(model=model, sample_id=match_path.stem, event=status,
                                       confidence=float(row["pred_p"]) if row["pred_p"] else float("nan"),
                                       **features))
    write_csv(OUT/"error_features_new20.csv", events)
    error_summary = {}
    for model in [selected, "historical_baseline"]:
        error_summary[model] = {}
        for event in ["TP", "FP", "FN"]:
            rows = [r for r in events if r["model"] == model and r["event"] == event]
            stats = {"n":len(rows)}
            for field in ["confidence", "local_zscore", "local_contrast", "peak_sharpness"]:
                values = [r[field] for r in rows if np.isfinite(r[field])]
                stats["median_"+field] = float(np.median(values)) if values else None
            error_summary[model][event] = stats
    write_json(OUT/"error_summary_new20.json", error_summary)
    report = ["# clean220 training results", "",
              "Raw files were not modified. Excluded sample_230 and sample_286.",
              "220 images / 5076 points. Train 166, validation 22, test 32 (historical 12 + new 20).",
              "Fixed 512 canvas; normalize original pixels, zero-pad right/bottom, ignore padded loss cells.",
              "Architecture and loss retained: UNetDeepBlink (1,258,723 parameters), Dice + 2*RMSE.",
              f"Selected {selected}, epoch {best_rows[selected]['epoch']}, by minimum validation loss only.",
              f"Validation-locked probability: {lock['probability']}.", "",
              "| Model | Test cohort | Precision | Recall | F1 | TP | FP | FN | Median error px |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        report.append(f"| {r['model']} | {r['cohort']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['tp']:.0f} | {r['fp']:.0f} | {r['fn']:.0f} | {r['median_error_px']:.4f} |")
    report += ["", "Replicate summaries:", "```json", json.dumps(means, indent=2), "```", "",
               "Interpretation boundaries:",
               "- Old12 is a reused historical regression set, not blind validation.",
               "- New20 is image-held-out, not an external biological cohort. Acquisition/batch independence is not fully established.",
               "- Native historical inference and fixed-canvas historical inference use identical weights; their difference is preprocessing plus validation calibration.",
               "- New training also changes data coverage, masking and numerical backend. Performance differences alone do not isolate a biological mechanism or a causal contribution from data size.",
               "- Localization errors are conditional on successful matches and must be interpreted with recall.",
               "- Three-seed SD describes training variability, not uncertainty from limited test images.",
               "- Six grid collisions remain in training; evaluation retains all original truth coordinates.", "",
               "Artifacts: recommended_model.pt/json, frozen_split.csv, data_audit.json, experiment_config.json,",
               "environment.json, source_snapshot/, run.log, per-seed training_metrics.csv/training_curves.png,",
               "validation threshold tables, test predictions/matches/overlays, stratified_metrics.csv and review_overlays/."]
    (OUT/"results_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("SELECTED", selected, "REPLICATES", means)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
