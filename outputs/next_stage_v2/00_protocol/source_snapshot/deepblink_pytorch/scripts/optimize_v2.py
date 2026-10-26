"""Development-only v2 calibration and bounded architecture screening.

Never loads or evaluates test arrays. References reuse historical weights.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from deepblink_torch.cli import predict, train
from deepblink_torch.evaluation import match_coordinates, metrics_from_match
from deepblink_torch.preprocessing import prepare_image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/next_stage_v2"
ARCHIVE = ROOT / "outputs/261024_clean220_random/dna_clean220_random.npz"
THRESHOLDS = [0.] + [float(x) for x in np.linspace(.025, .975, 29)] + [1.]
PROTOCOL = "pooled220_methods_v2"

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_development(subset="valid"):
    if subset not in ("train", "valid"):
        raise ValueError("Optimization is restricted to train/valid, never test.")
    with np.load(ARCHIVE, allow_pickle=True) as archive:
        return {key: archive[f"{key}_{subset}"] for key in
                ("x", "y", "sample_id", "original_shape")}

def decode_grid(grid, shape, threshold, cell_size=4):
    # The boundary candidate explicitly keeps numerical sigmoid zeros too.
    keep = grid[..., 0] >= 0 if threshold == 0 else grid[..., 0] > threshold
    rows, cols = np.nonzero(keep)
    scores = grid[rows, cols, 0]
    coords = (np.column_stack([rows, cols]) + grid[rows, cols, 1:]) * cell_size
    inside = ((coords >= 0).all(axis=1) &
              (coords[:, 0] < shape[0]) & (coords[:, 1] < shape[1]))
    return coords[inside].astype(np.float32), scores[inside].astype(np.float32)

@torch.no_grad()
def infer_development(checkpoint, destination, subset="valid", device="xpu"):
    data = load_development(subset)
    model, cell_size = predict._load_model(checkpoint, torch.device(device))
    if cell_size != 4:
        raise ValueError("Registered screen uses cell_size=4.")
    torch.set_num_threads(4)
    grids = []
    for raw, shape in zip(data["x"], data["original_shape"]):
        height, width = map(int, shape)
        image = prepare_image(raw[:height, :width], model.input_size)
        tensor = torch.from_numpy(image[None, None]).to(device)
        if getattr(model, "requires_valid_mask", False):
            mask = torch.zeros_like(tensor)
            mask[..., :height, :width] = 1
            grid = model(tensor, valid_mask=mask)
        else:
            grid = model(tensor)
        grid = grid[0].float().cpu().numpy()
        if not np.isfinite(grid).all():
            raise FloatingPointError("Nonfinite inference output.")
        grids.append(grid)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "prediction_grids.npz", grids=np.stack(grids),
                        sample_id=data["sample_id"], original_shape=data["original_shape"])
    del model
    if device == "xpu":
        torch.xpu.empty_cache()
    return data, grids

def calibrate(checkpoint, destination, device="xpu"):
    identity = dict(protocol_id=PROTOCOL, checkpoint_sha256=sha256(checkpoint),
                    matcher_sha256=sha256(ROOT / "deepblink_pytorch/deepblink_torch/evaluation.py"),
                    thresholds=THRESHOLDS, inference_precision="float32", subset="valid",
                    boundary_rule="threshold_0_keeps_all_including_numerical_zeros")
    summary_path = destination / "validation_summary.json"
    if summary_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached["identity"] != identity:
            raise ValueError(f"Cached calibration differs: {destination}")
        return cached
    data, grids = infer_development(checkpoint, destination, device=device)
    sweep, per_threshold = [], []
    for threshold in THRESHOLDS:
        rows = []
        for sid, shape, truth, grid in zip(data["sample_id"], data["original_shape"], data["y"], grids):
            coords, _ = decode_grid(grid, shape, threshold)
            row = metrics_from_match(match_coordinates(coords, truth, 3.))
            row = {k: (None if not np.isfinite(v) else v) for k, v in row.items()}
            rows.append(dict(sample_id=str(sid), height=int(shape[0]), width=int(shape[1]),
                             n_true=len(truth), n_pred=len(coords), **row))
        tp, fp, fn = [sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn")]
        precision = tp / (tp + fp) if tp + fp else 0.
        recall = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.
        sweep.append(dict(threshold=threshold, tp=tp, fp=fp, fn=fn,
                          precision=precision, recall=recall, f1=f1))
        per_threshold.append(rows)
    index = max(range(len(sweep)), key=lambda i: (sweep[i]["f1"], sweep[i]["precision"], sweep[i]["threshold"]))
    best = sweep[index]
    coverage = []
    for sid, shape, grid in zip(data["sample_id"], data["original_shape"], grids):
        _, scores = decode_grid(grid, shape, -1.)
        coverage.append(dict(sample_id=str(sid), valid_candidates=len(scores),
                             scores_gt_0_5=int((scores > .5).sum()),
                             scores_gt_0_975=int((scores > .975).sum()),
                             score_quantiles=np.quantile(scores, [0, .5, .9, .99, .999, 1]).tolist()))
    summary = dict(identity=identity, **best, n_images=len(grids), n_truth=sum(map(len, data["y"])),
                   macro_f1=float(np.mean([r["f1"] for r in per_threshold[index]])),
                   count_mae=float(np.mean([abs(r["n_pred"]-r["n_true"]) for r in per_threshold[index]])),
                   selection="maximum validation point micro F1; ties higher precision then threshold",
                   legacy_precision_floor=False, test_evaluated=False)
    write_csv(destination / "threshold_metrics.csv", sweep)
    write_csv(destination / "per_image_metrics.csv", per_threshold[index])
    write_json(destination / "score_coverage.json", coverage)
    write_json(summary_path, summary)
    return summary

def lock_protocol():
    path = OUT / "00_protocol/lock.json"
    value = dict(protocol_id=PROTOCOL, status="locked_for_initial_development_screen",
                 split_manifest_sha256=sha256(ROOT / "outputs/261024_clean220_random/frozen_split.csv"),
                 archive_sha256=sha256(ARCHIVE), radius_px=3.,
                 matching="maximum_cardinality_then_minimum_distance",
                 thresholds=THRESHOLDS, threshold_operator="strict_greater_than_except_0_keep_all",
                 keep_all=0., drop_all=1., inference_precision="float32",
                 seed=42, epochs=100, batch_size=8, lr=.001, weight_decay=0.,
                 loss="dice_plus_2_rmse", augmentation=False,
                 screening_device="xpu", amp_training="bfloat16", channels_last_training=True,
                 checkpoint="minimum_validation_loss", config_selection="validation_point_micro_f1",
                 test_access_for_optimization=False,
                 test_status="previously_observed_internal_benchmark_not_blind")
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Protocol differs from frozen lock.")
    write_json(path, value)
    (OUT / "execution_pid.txt").write_text(str(os.getpid()))
    return value

def snapshot(destination):
    files = list((ROOT / "deepblink_pytorch/deepblink_torch").rglob("*.py")) + [Path(__file__)]
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = destination / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256(source)
    write_json(destination / "source_hashes.json", hashes)
    write_json(destination / "environment.json", dict(python=sys.version, torch=torch.__version__,
               device=torch.xpu.get_device_name(0) if torch.xpu.is_available() else "cpu"))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reference", "screen"])
    parser.add_argument("--models", nargs="+", default=["deeploci_evidence", "deeploci_local"])
    parser.add_argument("--device", default="xpu", choices=["xpu", "cpu"])
    args = parser.parse_args(argv)
    lock_protocol()
    if args.action == "reference":
        snapshot(OUT / "00_protocol")
        summaries = []
        for seed in (42, 43, 44):
            ckpt = ROOT / f"outputs/261024_clean220_random/seed_{seed}/best_model.pt"
            result = calibrate(ckpt, OUT / f"00_protocol/reference_seed_{seed}", args.device)
            summaries.append(dict(model="deepblink", seed=seed, **{k: result[k] for k in
                                  ("f1", "precision", "recall", "threshold", "tp", "fp", "fn")}))
            print(json.dumps(summaries[-1]), flush=True)
        write_csv(OUT / "00_protocol/reference_validation.csv", summaries)
        return
    if len(set(args.models)) != len(args.models) or len(args.models) > 4:
        raise ValueError("At most four unique registered architecture candidates.")
    for model_name in args.models:
        if model_name not in ("deeploci_evidence", "deeploci_local"):
            raise ValueError("Unregistered architecture candidate.")
        destination = OUT / "02_architecture" / model_name / "seed_42"
        status_path = destination / "status.json"
        ckpt = destination / "best_model.pt"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["status"] == "complete":
                print(f"Already complete: {model_name}", flush=True)
                continue
            raise RuntimeError(f"Incomplete attempt needs audit before rerunning: {destination}")
        training_args = train.parse_args([
            "--data", str(ARCHIVE), "--checkpoint", str(ckpt), "--model", model_name,
            "--epochs", "100", "--batch-size", "8", "--lr", ".001", "--seed", "42",
            "--device", args.device, "--threads", "4", "--channels-last", "--development-only"] +
            (["--amp"] if args.device == "xpu" else []))
        snapshot(destination)
        write_json(destination / "registered_config.json", dict(protocol_id=PROTOCOL, args=vars(training_args),
                    n_train=176, n_valid=22, updates_per_epoch=22, total_updates=2200,
                    role="core_hypothesis" if model_name == "deeploci_evidence" else "simple_alternative_control",
                    test_evaluated=False))
        started = time.time()
        write_json(status_path, dict(status="running", pid=os.getpid(), started_unix=started))
        try:
            train.run_training(training_args)
            result = calibrate(ckpt, destination / "validation_v2", args.device)
            write_json(status_path, dict(status="complete", elapsed_seconds=time.time()-started,
                       checkpoint_sha256=sha256(ckpt), validation_f1=result["f1"], test_evaluated=False))
            print(json.dumps(dict(model=model_name, validation=result)), flush=True)
        except Exception as error:
            write_json(status_path, dict(status="failed", error=repr(error), test_evaluated=False))
            raise

if __name__ == "__main__":
    main()
