"""Train a new 220-image version on one pooled random 80/10/10 split.

The earlier source manifest supplies the audited inventory, never assignments.
Historical 113-model scores refer only to its own historical test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from deepblink_torch.cli import create, train
from run_clean220 import ROOT, OLD, evaluate_one, write_csv, write_json


OUT = ROOT / "outputs/261024_clean220_random"
INVENTORY = ROOT / "outputs/261019_clean220/frozen_split.csv"
EXCLUDED = {"sample_230", "sample_286"}


def pooled_split(inventory, seed):
    """Randomly assign all samples together, without origin or old assignments."""
    rows = [{k:v for k,v in row.items() if k not in ("cohort", "subset")}
            for row in inventory]
    rows.sort(key=lambda row: row["sample_id"])
    if len(rows) != 220 or len({row["sample_id"] for row in rows}) != 220:
        raise ValueError("Expected exactly 220 unique samples.")
    if len({row["source_group"] for row in rows}) != len(rows):
        raise ValueError("Repeated source groups require grouped splitting before training.")
    shuffled = list(range(len(rows)))
    random.Random(seed).shuffle(shuffled)
    for position, index in enumerate(shuffled):
        rows[index]["subset"] = "train" if position < 176 else "valid" if position < 198 else "test"
    return rows


def prepare(split_seed):
    with INVENTORY.open(encoding="utf-8") as handle:
        inventory = list(csv.DictReader(handle))
    rows = pooled_split(inventory, split_seed)
    raw = ROOT / "data/raw"
    for folder, suffix in [("tif", ".tif"), ("csv", ".csv")]:
        names = {p.stem for p in (raw/folder).glob("*"+suffix)} - EXCLUDED
        if names != {row["sample_id"] for row in rows}:
            raise ValueError("Current raw inventory differs from the authorized 220 samples.")
    for row in rows:
        for folder, suffix, field in [("tif", ".tif", "image_sha256"), ("csv", ".csv", "csv_sha256")]:
            path = raw/folder/(row["sample_id"]+suffix)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[field]:
                raise ValueError(f"Raw file changed since audit: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT/"frozen_split.csv"
    if frozen.exists():
        with frozen.open(encoding="utf-8") as handle:
            if list(csv.DictReader(handle)) != rows:
                raise ValueError("Existing frozen split differs; refusing to replace it.")
    else:
        write_csv(frozen, rows)
    archive = OUT/"dna_clean220_random.npz"
    if not archive.exists():
        create.run_create(create.parse_args([
            "--images", str(raw/"tif"), "--labels", str(raw/"csv"),
            "--output", str(archive), "--input-size", "512", "--split-manifest", str(frozen),
            "--exclude-samples", ",".join(sorted(EXCLUDED)), "--seed", str(split_seed)]))
    audit = dict(n_images=len(rows), n_points=sum(int(r["n_points"]) for r in rows),
                 excluded=sorted(EXCLUDED), split_seed=split_seed,
                 split_method="pooled random image split; origin not used; all source groups unique",
                 raw_hashes_verified=440,
                 splits={s:dict(images=sum(r["subset"]==s for r in rows),
                                points=sum(int(r["n_points"]) for r in rows if r["subset"]==s),
                                sizes=dict(Counter(r["height"] for r in rows if r["subset"]==s)))
                         for s in ["train", "valid", "test"]},
                 grid_collisions=[{k:r[k] for k in ["sample_id", "subset", "grid_collision_loss"]}
                                  for r in rows if int(r["grid_collision_loss"])])
    write_json(OUT/"data_audit.json", audit)
    return archive, rows


def snapshot():
    files = list((ROOT/"deepblink_pytorch/deepblink_torch").rglob("*.py"))
    files += [Path(__file__), Path(__file__).with_name("run_clean220.py")]
    for path in files:
        dest = OUT/"source_snapshot"/path.relative_to(ROOT/"deepblink_pytorch")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write_json(OUT/"environment.json", dict(python=sys.version, torch=torch.__version__,
               xpu=torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
               git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    (OUT/"code_changes.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))


def summarize(config, rows, selection):
    results = []
    for seed in config["seeds"]:
        name = f"seed_{seed}"
        summary = json.loads((OUT/name/"test/summary/metrics_summary.json").read_text())
        checkpoint = torch.load(OUT/name/"best_model.pt", map_location="cpu", weights_only=False)
        results.append(dict(model=name, best_epoch=checkpoint["epoch"], validation_loss=checkpoint["best_val_loss"],
                            probability=summary["probability"], n_images=summary["n_images"],
                            precision=summary["micro_precision"], recall=summary["micro_recall"],
                            f1=summary["micro_f1"], macro_f1=summary["macro_f1"],
                            tp=summary["micro_tp"], fp=summary["micro_fp"], fn=summary["micro_fn"],
                            median_error_px=summary["median_error_px"]))
    write_csv(OUT/"seed_comparison.csv", results)
    write_json(OUT/"seed_comparison.json", results)
    f1s = [r["f1"] for r in results]
    replicate = dict(mean_f1=float(np.mean(f1s)), sd_f1=float(np.std(f1s, ddof=1)),
                     min_f1=min(f1s), max_f1=max(f1s))
    write_json(OUT/"replicate_summary.json", replicate)
    selected = selection["selected_model"]
    checkpoint = torch.load(OUT/selected/"best_model.pt", map_location="cpu", weights_only=False)
    checkpoint["calibrated_probability"] = selection["thresholds"][selected]
    checkpoint["selection_rule"] = "minimum validation loss among seeds; validation-only threshold"
    torch.save(checkpoint, OUT/"selected_model.pt")
    write_json(OUT/"selected_model.json", dict(**selection, epoch=checkpoint["epoch"],
               input_size=512, probability=checkpoint["calibrated_probability"],
               inference_note="Pass --probability explicitly; default CLI probability is 0.5."))
    # Historical scores remain on their original test. Do not score historical
    # weights on the new random test: those weights may have trained on its images.
    history = json.loads((OLD/"evaluation_final/summary/metrics_summary.json").read_text())
    selected_result = next(r for r in results if r["model"] == selected)
    versions = [dict(version="113_historical", n_total=113, n_test=12,
                     precision=history["micro_precision"], recall=history["micro_recall"], f1=history["micro_f1"],
                     interpretation="historical own split; test-sweep threshold; not a matched test comparison"),
                dict(version="220_pooled_random", n_total=220, n_test=22,
                     precision=selected_result["precision"], recall=selected_result["recall"], f1=selected_result["f1"],
                     interpretation="new random split; validation-only selection; different test difficulty")]
    write_csv(OUT/"version_comparison.csv", versions)
    write_json(OUT/"version_comparison.json", versions)
    metadata = {r["sample_id"]:r for r in rows}
    strata = []
    for result in results:
        with (OUT/result["model"]/"test/summary/per_image_metrics.csv").open() as handle:
            per_image = list(csv.DictReader(handle))
        for size in ["256", "512"]:
            subset = [r for r in per_image if metadata[r["sample_id"]]["height"] == size]
            if not subset:
                continue
            tp,fp,fn = [sum(float(r[k]) for r in subset) for k in ["tp", "fp", "fn"]]
            strata.append(dict(model=result["model"], size=size, n_images=len(subset), tp=tp, fp=fp, fn=fn,
                               precision=tp/(tp+fp) if tp+fp else 0,
                               recall=tp/(tp+fn) if tp+fn else 0,
                               f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))
    write_csv(OUT/"size_metrics.csv", strata)
    report = ["# clean220 pooled-random version", "",
              "All 220 samples were pooled before random 176/22/22 assignment. Origin was not used.",
              f"Split seed: {config['split_seed']}. Training seeds: {config['seeds']}.",
              f"Selected {selected} by validation loss before test inference; probability {checkpoint['calibrated_probability']}.",
              "", "| Seed | Epoch | P | R | Micro F1 | Macro F1 | TP | FP | FN | Median error px |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        report.append(f"| {r['model']} | {r['best_epoch']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['macro_f1']:.4f} | {r['tp']:.0f} | {r['fp']:.0f} | {r['fn']:.0f} | {r['median_error_px']:.4f} |")
    report += ["", f"Three-seed F1: {replicate['mean_f1']:.4f} +/- {replicate['sd_f1']:.4f} (sample SD).",
               "", "Historical 113 F1: 0.8488 on its own original 12-image test, with test-sweep threshold.",
               "The new random test is different; numerical version differences do not isolate data-size effects.",
               "The new test was not used for this round's fitting/checkpoint/threshold selection; some images were analyzed in earlier repository experiments.",
               "These are image-held-out results, not independent biological/external validation.",
               "SD across seeds describes training variation, not uncertainty due to the 22 test images.",
               "Images are normalized on original pixels and padded right/bottom to 512. Padded cells are ignored by the loss.",
               "Original truth coordinates are retained at evaluation, including any grid collisions.",
               "Raw files and prior experiment folders are unchanged. See frozen_split.csv, data_audit.json, source_snapshot/ and run.log."]
    (OUT/"results_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("FINAL RESULTS", json.dumps(results, indent=2), "REPLICATES", replicate, flush=True)


def run(args):
    archive, rows = prepare(args.split_seed)
    config = dict(split_seed=args.split_seed, seeds=args.seeds, epochs=100, batch_size=8,
                  lr=.001, device="xpu", amp=True, channels_last=True, threads=4,
                  architecture="UNetDeepBlink", loss="dice_rmse", split_counts=[176,22,22],
                  selection="minimum validation loss; validation sweep with p=0.5 precision floor",
                  test_policy="one pooled random test; no origin-specific evaluation")
    config_path = OUT/"experiment_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Experiment configuration changed; use a distinct experiment directory.")
    write_json(config_path, config)
    if not (OUT/"environment.json").exists():
        snapshot()
    if args.prepare_only:
        print(json.dumps(json.loads((OUT/"data_audit.json").read_text()), indent=2), flush=True)
        return
    if not torch.xpu.is_available():
        raise RuntimeError("The verified isolated Intel XPU environment is required for this experiment.")
    torch.set_num_threads(4)
    for seed in args.seeds:
        folder = OUT/f"seed_{seed}"
        if (folder/"training_complete.json").exists():
            continue
        started = time.time()
        train.run_training(train.parse_args([
            "--data", str(archive), "--checkpoint", str(folder/"best_model.pt"),
            "--epochs", "100", "--batch-size", "8", "--lr", ".001", "--seed", str(seed),
            "--threads", "4", "--device", "xpu", "--amp", "--channels-last"]))
        write_json(folder/"training_complete.json", dict(epochs=100, seconds=time.time()-started))
    thresholds, validation_losses = {}, {}
    for seed in args.seeds:
        name = f"seed_{seed}"
        checkpoint = OUT/name/"best_model.pt"
        valid = evaluate_one(checkpoint, archive, OUT/name/"validation", "valid", sweep=True, device="xpu")
        thresholds[name] = valid["probability"]
        validation_losses[name] = torch.load(checkpoint, map_location="cpu", weights_only=False)["best_val_loss"]
        write_json(OUT/name/"threshold_lock.json", dict(probability=valid["probability"], selected_on="valid"))
    selection = dict(selected_model=min(validation_losses, key=validation_losses.get),
                     validation_losses=validation_losses, thresholds=thresholds,
                     rule="minimum validation loss; thresholds selected only on validation")
    selection_path = OUT/"model_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Previously locked model selection changed.")
    if not selection_path.exists():
        write_json(selection_path, selection)
    for seed in args.seeds:
        name = f"seed_{seed}"
        evaluate_one(OUT/name/"best_model.pt", archive, OUT/name/"test", "test",
                     thresholds[name], device="xpu")
    summarize(config, rows, selection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20261024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    run(parser.parse_args())
