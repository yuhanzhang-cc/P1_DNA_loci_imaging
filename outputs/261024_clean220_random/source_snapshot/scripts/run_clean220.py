"""Frozen clean220 experiment: audit, prepare, train, validation calibration, test.

Run from the repository root with the existing dna_pytorch environment.
No raw files are changed. Historical test images remain a regression set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import torch

from deepblink_torch.cli import create, evaluate, train
from deepblink_torch.utils.pixel_size import read_pixel_size_or_default


ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"sample_230", "sample_286"}
OUT = ROOT / "outputs" / "261019_clean220"
OLD = ROOT / "outputs" / "260628test"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OLD / "dna_clean_113.split.csv").open() as handle:
        old = {row["sample_id"]: row["subset"] for row in csv.DictReader(handle)}
    raw = ROOT / "data" / "raw"
    with (raw / "logs" / "annoation_record.csv").open(encoding="utf-8-sig") as handle:
        logs = {Path(row["raw_tif_name"]).stem: row for row in csv.DictReader(handle)}
    images = {p.stem: p for p in (raw / "tif").glob("*.tif") if p.stem not in EXCLUDED}
    labels = {p.stem: p for p in (raw / "csv").glob("*.csv") if p.stem not in EXCLUDED}
    if set(images) != set(labels) or len(images) != 220 or not set(old) <= set(images):
        raise ValueError("Expected exactly 220 paired images, including all clean113 samples.")
    manifest = []
    hashes = set()
    for sample in sorted(images):
        image = tifffile.imread(images[sample])
        if image.shape not in [(256, 256), (512, 512)] or not np.isfinite(image).all() or image.std() == 0:
            raise ValueError(f"Invalid image: {sample}")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate image content: {sample}")
        hashes.add(digest)
        coords = create._read_coords(labels[sample])
        with labels[sample].open(encoding="utf-8-sig") as handle:
            descriptor = list(csv.reader(handle))[3]
        pixel_units = "(pixel)" in descriptor
        pixel_size = read_pixel_size_or_default(images[sample])
        if not pixel_units and pixel_size.source == "default_1.0":
            raise ValueError(f"Missing micron calibration: {sample}")
        if not pixel_units:
            coords = create._convert_coords_to_pixels(coords, pixel_size)
        if not len(coords) or not np.isfinite(coords).all() or np.any(coords < 0) or np.any(coords >= image.shape):
            raise ValueError(f"Invalid coordinates: {sample}")
        if len(np.unique(coords, axis=0)) != len(coords):
            raise ValueError(f"Duplicate coordinates: {sample}")
        log = logs[sample]
        source_group = log["source_path"] + "|" + log["source_file_name"]
        collisions = len(coords) - len(np.unique(np.floor(coords).astype(int)//4, axis=0))
        manifest.append(dict(sample_id=sample, subset=old.get(sample, ""),
                             cohort="old113" if sample in old else "new107",
                             height=image.shape[0], width=image.shape[1], n_points=len(coords),
                             grid_collision_loss=collisions, coordinate_unit="pixel" if pixel_units else "micron",
                             pixel_size_x=pixel_size.x, pixel_size_y=pixel_size.y,
                             calibration_source=pixel_size.source, source_group=source_group,
                             image_sha256=hashlib.sha256(images[sample].read_bytes()).hexdigest(),
                             csv_sha256=hashlib.sha256(labels[sample].read_bytes()).hexdigest()))
    # Freeze BEFORE any new-model inference. Stratify new holdout by image size.
    rng = random.Random(20261019)
    for size in [256, 512]:
        candidates = [row for row in manifest if row["cohort"] == "new107" and row["height"] == size]
        rng.shuffle(candidates)
        n_test = 16 if size == 256 else 4
        n_valid = 9 if size == 256 else 2
        for i, row in enumerate(candidates):
            row["subset"] = "test" if i < n_test else "valid" if i < n_test+n_valid else "train"
    groups = {}
    for row in manifest:
        previous = groups.setdefault(row["source_group"], row["subset"])
        if previous != row["subset"]:
            raise ValueError("Same source image/series crosses splits; revise grouping before training.")
    frozen = OUT / "frozen_split.csv"
    if frozen.exists():
        with frozen.open(encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        for before, after in zip(existing, manifest, strict=True):
            if any(str(after[k]) != before[k] for k in before):
                raise ValueError("Frozen manifest changed; refusing to overwrite.")
    else:
        write_csv(frozen, manifest)
    summary = dict(n_images=len(manifest), n_points=sum(r["n_points"] for r in manifest),
                   excluded=sorted(EXCLUDED), split_counts=dict(Counter(r["subset"] for r in manifest)),
                   cohort_splits=dict(Counter(r["cohort"]+"/"+r["subset"] for r in manifest)),
                   grid_collisions=[{k:r[k] for k in ["sample_id", "n_points", "grid_collision_loss"]}
                                    for r in manifest if r["grid_collision_loss"]],
                   grouping="source_path + source_file_name; biological independence is not established",
                   split_seed=20261019, input_size=512,
                   preprocessing="normalize original pixels; constant zero padding right/bottom; ignore padded loss cells")
    write_json(OUT / "data_audit.json", summary)
    archive = OUT / "dna_clean_220.npz"
    if not archive.exists():
        create.run_create(create.parse_args([
            "--images", str(raw/"tif"), "--labels", str(raw/"csv"), "--output", str(archive),
            "--input-size", "512", "--exclude-samples", ",".join(sorted(EXCLUDED)),
            "--split-manifest", str(frozen), "--seed", "20261019"]))
    # Truthful separate reporting for historical regression and unseen new holdout.
    with np.load(archive, allow_pickle=True) as data:
        for cohort in ["old113", "new107"]:
            mask = np.array([(str(s) in old) == (cohort == "old113") for s in data["sample_id_test"]])
            filtered = {k:(v[mask] if k.endswith("_test") and v.ndim and len(v) == len(mask) else v)
                        for k,v in ((k,data[k]) for k in data.files)}
            path = OUT / f"test_{cohort}.npz"
            if not path.exists():
                np.savez(path, **filtered)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (OUT/"code_changes.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    write_json(OUT/"environment.json", dict(python=sys.version, torch=torch.__version__,
               cuda=torch.cuda.is_available(), platform=platform.platform(), revision=revision,
               xpu_available=torch.xpu.is_available(),
               xpu_name=torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
               source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROOT/"deepblink_pytorch/deepblink_torch").rglob("*.py")}))
    snapshot = OUT / "source_snapshot"
    for p in list((ROOT/"deepblink_pytorch/deepblink_torch").rglob("*.py")) + [Path(__file__)]:
        target = snapshot / p.relative_to(ROOT/"deepblink_pytorch")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    print(json.dumps(summary, indent=2), flush=True)
    return archive


def evaluate_one(checkpoint, archive, output, subset, probability=None, sweep=False, device="cpu"):
    if (output/"summary/metrics_summary.json").exists():
        return json.loads((output/"summary/metrics_summary.json").read_text())
    argv = ["--data", str(archive), "--images", str(ROOT/"data/raw/tif"),
            "--model", str(checkpoint), "--output", str(output), "--subset", subset,
            "--device", device, "--match-radius-px", "3"]
    if probability is not None:
        argv += ["--probability", str(probability)]
    if sweep:
        argv += ["--probability-sweep", ",".join(f"{p:.2f}" for p in np.arange(.1, .951, .05))]
    evaluate.run_test(evaluate.parse_args(argv))
    return json.loads((output/"summary/metrics_summary.json").read_text())


def run(args):
    archive = prepare()
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    config = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  seeds=args.seeds, threads=args.threads, device=device, amp=args.amp,
                  channels_last=args.channels_last, model="deepblink", loss="dice_rmse",
                  checkpoint_selection="minimum validation loss, independently per seed",
                  threshold_selection="validation F1 with precision >= validation precision at p=0.5",
                  primary_endpoint="micro F1 at radius 3 px on new107 held-out 20 images",
                  historical_endpoint="old113 12-image test; previously reused, regression only")
    config_path = OUT/"experiment_config.json"
    if list(OUT.glob("seed_*/training_complete.json")) and json.loads(config_path.read_text()) != config:
        raise ValueError("Completed training exists with a different configuration.")
    write_json(config_path, config)
    if args.prepare_only:
        return
    for seed in args.seeds:
        folder = OUT/f"seed_{seed}"
        checkpoint = folder/"best_model.pt"
        finished = folder/"training_complete.json"
        if not finished.exists():
            argv = ["--data", str(archive), "--checkpoint", str(checkpoint), "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--seed", str(seed),
                    "--threads", str(args.threads), "--device", device]
            if args.amp:
                argv += ["--amp"]
            if args.channels_last:
                argv += ["--channels-last"]
            started = time.time()
            train.run_training(train.parse_args(argv))
            write_json(finished, dict(seconds=time.time()-started, epochs=args.epochs))
    # Calibration uses validation only; test scores never select checkpoint or seed.
    models = [(f"seed_{seed}", OUT/f"seed_{seed}/best_model.pt") for seed in args.seeds]
    models.append(("historical_baseline", OLD/"best_model_clean_113.pt"))
    results = []
    for name, checkpoint in models:
        folder = OUT/name
        valid = evaluate_one(checkpoint, archive, folder/"validation", "valid", sweep=True, device=device)
        probability = valid["probability"]
        write_json(folder/"threshold_lock.json", dict(probability=probability, selected_on="valid",
                   rule=config["threshold_selection"], locked_before_test=True))
        for cohort in ["old113", "new107"]:
            test = evaluate_one(checkpoint, OUT/f"test_{cohort}.npz", folder/f"test_{cohort}", "test", probability, device=device)
            results.append(dict(model=name, cohort=cohort, probability=probability,
                                n_images=test["n_images"], tp=test["micro_tp"], fp=test["micro_fp"],
                                fn=test["micro_fn"], precision=test["micro_precision"], recall=test["micro_recall"],
                                f1=test["micro_f1"], median_error_px=test["median_error_px"]))
    # Reproduce the historical operating point separately from current calibration.
    evaluate_one(OLD/"best_model_clean_113.pt", OLD/"dna_clean_113.npz",
                 OUT/"historical_baseline/reproduction_p04", "test", .4, device=device)
    write_csv(OUT/"comparison.csv", results)
    write_json(OUT/"comparison.json", results)
    print("FINAL COMPARISON", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    run(parser.parse_args())
