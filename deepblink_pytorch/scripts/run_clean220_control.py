"""Old-data-only versus expanded-data training on identical pooled holdouts.

Reuse the completed, verified expanded arm. Train the 89-image old-only arm
from scratch with the same recipe, seeds, validation and test images. The test
has already been observed: this is a controlled internal comparison, not a new
blind evaluation. Run from the repository root in the isolated XPU environment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from deepblink_torch.cli import train
from run_clean220 import ROOT, OLD, evaluate_one, write_csv, write_json


EXPANDED = ROOT / "outputs/261024_clean220_random"
OUT = ROOT / "outputs/261025_clean220_control"
SEEDS = [42, 43, 44]


def filter_training_arrays(arrays, old_ids):
    """Filter every training-aligned array; preserve all holdout arrays exactly."""
    ids = [str(value) for value in arrays["sample_id_train"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated training sample IDs.")
    holdout = {str(value) for split in ("valid", "test")
               for value in arrays["sample_id_" + split]}
    if set(ids) & holdout:
        raise ValueError("Training and holdout samples overlap.")
    mask = np.array([value in old_ids for value in ids], dtype=bool)
    if not mask.any() or mask.all():
        raise ValueError("Expected both old and added training samples.")
    result = {}
    for key, value in arrays.items():
        if key.endswith("_train"):
            if not value.ndim or len(value) != len(mask):
                raise ValueError(f"Training metadata is not aligned: {key}")
            result[key] = value[mask]
        else:
            result[key] = value.copy()
    n = sum(len(result["sample_id_" + split]) for split in ("train", "valid", "test"))
    result["train_ratio"] = np.array(int(mask.sum()) / n, dtype=np.float32)
    result["valid_ratio"] = np.array(len(result["sample_id_valid"]) / n, dtype=np.float32)
    result["split_method"] = np.array("old_only_train_shared_pooled_holdouts")
    return result, mask


def paired_image_bootstrap(control, expanded, *, n_resamples=10000, seed=20261025):
    """CI for the across-seed mean micro-F1 difference, resampling images.

    Inputs are (seed, image, [TP, FP, FN]); use identical sampled image indices
    for both arms and all seeds. Seeds are fixed replicates, never 66 independent
    images. The interval conditions on this split, these models and image-level
    exchangeability; it does not establish biological/batch independence.
    """
    control, expanded = np.asarray(control, dtype=float), np.asarray(expanded, dtype=float)
    if control.shape != expanded.shape or control.ndim != 3 or control.shape[-1] != 3:
        raise ValueError("Counts must have matching (seed, image, 3) shapes.")
    if not np.isfinite(control).all() or not np.isfinite(expanded).all():
        raise ValueError("Non-finite counts.")
    if np.any(control < 0) or np.any(expanded < 0):
        raise ValueError("Negative counts.")
    if not np.array_equal(control[..., 0] + control[..., 2], expanded[..., 0] + expanded[..., 2]):
        raise ValueError("Paired images must have the same truth counts.")
    if control.shape[1] < 2 or n_resamples < 1:
        raise ValueError("Need at least two images and one resample.")

    def f1(counts):
        denominator = 2 * counts[..., 0] + counts[..., 1] + counts[..., 2]
        return np.divide(2 * counts[..., 0], denominator,
                         out=np.zeros_like(denominator), where=denominator > 0)

    point = float(np.mean(f1(expanded.sum(axis=1)) - f1(control.sum(axis=1))))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, control.shape[1], size=(n_resamples, control.shape[1]))
    boot_control = f1(control[:, indices, :].sum(axis=2))
    boot_expanded = f1(expanded[:, indices, :].sum(axis=2))
    differences = np.mean(boot_expanded - boot_control, axis=0)
    return dict(delta_mean_micro_f1=point,
                percentile_95_ci=[float(x) for x in np.quantile(differences, [.025, .975])],
                n_resamples=n_resamples, seed=seed, n_images=control.shape[1],
                n_fixed_training_seeds=control.shape[0],
                method="paired image-cluster percentile bootstrap; shared image resamples across seeds and arms",
                limitation="Conditional on trained models and one already observed split; image exchangeability assumed, biological/batch independence not established; no p-value.")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen configuration changed: {path}")
    else:
        write_json(path, value)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    old_ids = {r["sample_id"] for r in read_rows(OLD / "dna_clean_113.split.csv")}
    rows = read_rows(EXPANDED / "frozen_split.csv")
    assert len(old_ids) == 113 and len(rows) == 220
    raw_hashes = {}
    for row in rows:
        for folder, suffix, field in (("tif", ".tif", "image_sha256"), ("csv", ".csv", "csv_sha256")):
            path = ROOT / "data/raw" / folder / (row["sample_id"] + suffix)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row[field]:
                raise ValueError(f"Raw input changed: {path}")
            raw_hashes[str(path.relative_to(ROOT))] = digest
    original_config = json.loads((EXPANDED / "experiment_config.json").read_text())
    expected = dict(seeds=SEEDS, epochs=100, batch_size=8, lr=.001, device="xpu",
                    amp=True, channels_last=True, threads=4,
                    architecture="UNetDeepBlink", loss="dice_rmse")
    for key, value in expected.items():
        if original_config[key] != value:
            raise ValueError(f"Expanded recipe differs: {key}")
    environment = json.loads((EXPANDED / "environment.json").read_text())
    core_files_checked = 0
    for relative, digest in environment["source_sha256"].items():
        if "deepblink_torch" in Path(relative).parts:
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Core code changed since expanded training: {relative}")
            core_files_checked += 1
    if not core_files_checked:
        raise ValueError("Expanded source snapshot contains no core modules.")
    if environment["torch"] != torch.__version__:
        raise ValueError("Torch version differs from the expanded arm.")
    for seed in SEEDS:
        metrics = read_rows(EXPANDED / f"seed_{seed}/training_metrics.csv")
        if [int(r["epoch"]) for r in metrics] != list(range(1, 101)):
            raise ValueError("Expanded training is incomplete.")
        checkpoint = torch.load(EXPANDED / f"seed_{seed}/best_model.pt", map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        for key in ("epochs", "batch_size", "lr", "device", "amp", "channels_last", "threads"):
            if args[key] != expected[key]:
                raise ValueError(f"Expanded checkpoint arguments differ: {seed}/{key}")
        if checkpoint["seed"] != seed:
            raise ValueError("Expanded checkpoint seed differs.")
        del checkpoint
    archive = OUT / "dna_old_only_shared_holdouts.npz"
    with np.load(EXPANDED / "dna_clean220_random.npz", allow_pickle=True) as data:
        arrays, mask = filter_training_arrays({k: data[k] for k in data.files}, old_ids)
    if mask.sum() != 89:
        raise ValueError("Expected 89 old training images in the existing pooled split.")
    if not archive.exists():
        np.savez(archive, **arrays)
    with np.load(archive, allow_pickle=True) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("Saved archive keys differ.")
        for key, value in arrays.items():
            a = saved[key]
            equal = (a.shape == value.shape and a.dtype == value.dtype)
            if equal:
                equal = (all(np.array_equal(x, y) for x, y in zip(a, value))
                         if value.dtype == object else np.array_equal(a, value))
            if not equal:
                raise ValueError(f"Saved archive differs: {key}")
    assignments = [dict(**row, arm_role=("old_only_train" if row["sample_id"] in old_ids
                                       else "excluded_added_train") if row["subset"] == "train"
                        else "shared_" + row["subset"]) for row in rows]
    manifest = OUT / "comparison_manifest.csv"
    if manifest.exists():
        if read_rows(manifest) != assignments:
            raise ValueError("Comparison manifest changed.")
    else:
        write_csv(manifest, assignments)
    config = dict(**expected, date="2026-10-25", expanded_directory=str(EXPANDED.relative_to(ROOT)),
                  control_train_images=89, expanded_train_images=176, added_train_images=87,
                  valid_images=22, test_images=22, match_radius_px=3,
                  checkpoint_rule="minimum validation loss, separately per seed",
                  threshold_rule="validation F1 with precision >= validation precision at p=0.5",
                  selected_seed_rule="minimum validation loss among seeds, separately per arm",
                  primary_endpoint="expanded minus control mean test micro-F1 across the three paired seeds",
                  uncertainty="10000 paired image-cluster bootstrap resamples, seed 20261025; training seeds held fixed",
                  test_policy="already observed pooled test; no threshold/seed selection on test",
                  budget_note="100 epochs and batch size 8 in both arms; 12 versus 22 optimizer updates per epoch (1200 vs 2200 total). This measures the same-recipe data expansion effect, not a fixed-update data-only effect.")
    freeze_json(OUT / "experiment_config.json", config)
    freeze_json(OUT / "input_hashes.json", raw_hashes)
    audit = dict(raw_files_verified=len(raw_hashes), holdout_arrays_identical=True,
                 counts={split: dict(images=len(arrays["sample_id_" + split]),
                                     truth_points=sum(len(x) for x in arrays["y_" + split]))
                         for split in ("train", "valid", "test")},
                 old_training_ids=[str(x) for x in arrays["sample_id_train"]],
                 excluded_added_training_ids=[r["sample_id"] for r in rows
                                              if r["subset"] == "train" and r["sample_id"] not in old_ids],
                 core_code_matches_expanded_snapshot=True, core_files_checked=core_files_checked)
    write_json(OUT / "data_audit.json", audit)
    if not (OUT / "environment.json").exists():
        files = list((ROOT / "deepblink_pytorch/deepblink_torch").rglob("*.py"))
        files += [Path(__file__), Path(__file__).with_name("run_clean220.py")]
        for path in files:
            dest = OUT / "source_snapshot" / path.relative_to(ROOT / "deepblink_pytorch")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        write_json(OUT / "environment.json", dict(python=sys.version, torch=torch.__version__,
                   xpu=torch.xpu.get_device_name(0) if torch.xpu.is_available() else None,
                   source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                   git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        (OUT / "code_changes.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=ROOT))
    print("DATA AUDIT", json.dumps(audit), flush=True)
    return archive, config


@contextmanager
def temporary_awake():
    """A thread-scoped Windows request, automatically cleared on process exit."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_ulong]
    kernel.SetThreadExecutionState.restype = ctypes.c_ulong
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError(ctypes.get_last_error())
    print("Temporary system-awake request active; display sleep allowed.", flush=True)
    try:
        yield
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        print("Temporary system-awake request released.", flush=True)


def summarize(config, selection):
    results, counts, per_image = [], {}, {}
    image_ids = sorted(r["sample_id"] for r in read_rows(EXPANDED / "frozen_split.csv") if r["subset"] == "test")
    for arm, folder in (("old_only_89", OUT), ("expanded_176", EXPANDED)):
        counts[arm] = []
        for seed in SEEDS:
            path = folder / f"seed_{seed}"
            summary = json.loads((path / "test/summary/metrics_summary.json").read_text())
            checkpoint = torch.load(path / "best_model.pt", map_location="cpu", weights_only=False)
            results.append(dict(arm=arm, seed=seed, best_epoch=checkpoint["epoch"],
                                validation_loss=checkpoint["best_val_loss"], probability=summary["probability"],
                                precision=summary["micro_precision"], recall=summary["micro_recall"],
                                f1=summary["micro_f1"], macro_f1=summary["macro_f1"],
                                tp=summary["micro_tp"], fp=summary["micro_fp"], fn=summary["micro_fn"],
                                median_error_px=summary["median_error_px"]))
            data = {r["sample_id"]: r for r in read_rows(path / "test/summary/per_image_metrics.csv")}
            if set(data) != set(image_ids):
                raise ValueError("Test image IDs differ between arms.")
            counts[arm].append([[float(data[i][k]) for k in ("tp", "fp", "fn")] for i in image_ids])
            for i in image_ids:
                per_image[arm, seed, i] = data[i]
            del checkpoint
    write_csv(OUT / "arm_seed_metrics.csv", results)
    write_json(OUT / "arm_seed_metrics.json", results)
    arm_summary, deltas = {}, []
    for arm in counts:
        group = [r for r in results if r["arm"] == arm]
        arm_summary[arm] = {key: dict(mean=float(np.mean([r[key] for r in group])),
                                      sd=float(np.std([r[key] for r in group], ddof=1)))
                            for key in ("precision", "recall", "f1", "macro_f1", "median_error_px")}
    for seed in SEEDS:
        control = next(r for r in results if r["arm"] == "old_only_89" and r["seed"] == seed)
        expanded = next(r for r in results if r["arm"] == "expanded_176" and r["seed"] == seed)
        deltas.append(dict(seed=seed, **{"delta_" + k: expanded[k] - control[k]
                                       for k in ("precision", "recall", "f1", "tp", "fp", "fn")}))
    write_csv(OUT / "paired_seed_deltas.csv", deltas)
    bootstrap = paired_image_bootstrap(counts["old_only_89"], counts["expanded_176"])
    write_json(OUT / "paired_bootstrap.json", bootstrap)
    write_json(OUT / "arm_summary.json", arm_summary)
    image_deltas = []
    for seed in SEEDS:
        for i in image_ids:
            a, b = per_image["old_only_89", seed, i], per_image["expanded_176", seed, i]
            image_deltas.append(dict(seed=seed, sample_id=i, n_true=int(a["n_true"]),
                                     control_f1=float(a["f1"]), expanded_f1=float(b["f1"]),
                                     delta_f1=float(b["f1"]) - float(a["f1"]),
                                     delta_fp=float(b["fp"]) - float(a["fp"]),
                                     delta_fn=float(b["fn"]) - float(a["fn"])))
    write_csv(OUT / "paired_image_deltas.csv", image_deltas)
    checkpoint = torch.load(OUT / selection["selected_model"] / "best_model.pt", map_location="cpu", weights_only=False)
    checkpoint["calibrated_probability"] = selection["thresholds"][selection["selected_model"]]
    checkpoint["selection_rule"] = config["selected_seed_rule"]
    torch.save(checkpoint, OUT / "selected_control_model.pt")
    write_json(OUT / "selected_control_model.json", dict(**selection, epoch=checkpoint["epoch"],
               probability=checkpoint["calibrated_probability"], input_size=512,
               inference_note="Pass --probability explicitly; CLI default remains 0.5."))
    lines = ["# Same-holdout data expansion comparison", "",
             "Old-only: 89 training images. Expanded: 176 (same 89 plus 87 added). Shared validation/test: 22/22.",
             "Same code, XPU environment, architecture, loss, batch size, learning rate, 100 epochs and paired seeds.",
             "100 epochs entail 1200 vs 2200 optimizer steps: this is a same-recipe expansion comparison.",
             "The expanded arm is reused after verifying raw hashes, core-code hashes and training arguments.",
             "The test was previously observed; this is internal controlled evidence, not independent blind validation.",
             "", "| Arm | Seed | Best epoch | Threshold | P | R | F1 | TP | FP | FN |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['best_epoch']} | {r['probability']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['tp']:.0f} | {r['fp']:.0f} | {r['fn']:.0f} |")
    lines += ["", "Across-seed summaries: " + json.dumps(arm_summary), "",
              "Paired uncertainty: " + json.dumps(bootstrap), "",
              "No test-based threshold or model selection. No outlier deletion. No biological independence claim.",
              "Image bootstrap conditions on these three fitted seed pairs and assumes image exchangeability.",
              "The training set changes in size, origin and size composition; this does not isolate pure sample count.",
              "Historical 113 published F1 is deliberately not used as the control endpoint."]
    (OUT / "results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("FINAL COMPARISON", json.dumps(arm_summary), json.dumps(bootstrap), flush=True)


def run(args):
    archive, config = prepare()
    if args.prepare_only:
        return
    if not torch.xpu.is_available():
        raise RuntimeError("Use the verified isolated Intel XPU environment.")
    torch.set_num_threads(4)
    with temporary_awake():
        for seed in SEEDS:
            folder = OUT / f"seed_{seed}"
            finished = folder / "training_complete.json"
            if finished.exists():
                continue
            # Partial training is never silently overwritten or called resumed.
            if (folder / "training_metrics.csv").exists():
                raise RuntimeError(f"Partial run needs explicit investigation before restart: {folder}")
            started = time.time()
            train.run_training(train.parse_args([
                "--data", str(archive), "--checkpoint", str(folder / "best_model.pt"),
                "--epochs", "100", "--batch-size", "8", "--lr", ".001", "--seed", str(seed),
                "--threads", "4", "--device", "xpu", "--amp", "--channels-last"]))
            write_json(finished, dict(epochs=100, seconds=time.time() - started))
        losses, thresholds = {}, {}
        for seed in SEEDS:
            name = f"seed_{seed}"
            checkpoint = OUT / name / "best_model.pt"
            valid = evaluate_one(checkpoint, archive, OUT / name / "validation", "valid", sweep=True, device="xpu")
            thresholds[name] = valid["probability"]
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            losses[name] = ckpt["best_val_loss"]
            del ckpt
            freeze_json(OUT / name / "threshold_lock.json", dict(probability=valid["probability"], selected_on="valid"))
        selection = dict(selected_model=min(losses, key=losses.get), validation_losses=losses,
                         thresholds=thresholds, rule="minimum validation loss; validation-only thresholds")
        freeze_json(OUT / "model_selection.json", selection)
        for seed in SEEDS:
            name = f"seed_{seed}"
            evaluate_one(OUT / name / "best_model.pt", archive, OUT / name / "test", "test", thresholds[name], device="xpu")
        summarize(config, selection)
    write_json(OUT / "experiment_complete.json", dict(seeds=SEEDS, epochs_per_seed=100,
               data_audit="data_audit.json", results="arm_seed_metrics.csv", uncertainty="paired_bootstrap.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
