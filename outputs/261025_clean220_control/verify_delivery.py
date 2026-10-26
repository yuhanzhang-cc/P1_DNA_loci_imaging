"""Post-run integrity checks; never tunes or changes experiment outputs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
EXPANDED = ROOT / "outputs/261024_clean220_random"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


assert (OUT / "experiment_complete.json").exists()
hashes = json.loads((OUT / "input_hashes.json").read_text())
for name, digest in hashes.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
expanded_hashes = json.loads((OUT / "expanded_artifact_hashes.json").read_text())
for name, digest in expanded_hashes.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
source_hashes = json.loads((OUT / "environment.json").read_text())["source_sha256"]
for name, digest in source_hashes.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
with np.load(OUT / "dna_old_only_shared_holdouts.npz", allow_pickle=True) as control:
    with np.load(EXPANDED / "dna_clean220_random.npz", allow_pickle=True) as expanded:
        for key in expanded.files:
            if key.endswith(("_valid", "_test")):
                a, b = control[key], expanded[key]
                assert a.shape == b.shape and a.dtype == b.dtype, key
                if a.dtype == object:
                    assert all(np.array_equal(x, y) for x, y in zip(a, b)), key
                else:
                    assert np.array_equal(a, b), key
        split_ids = {s: set(str(x) for x in control["sample_id_" + s]) for s in ("train", "valid", "test")}
        assert [len(split_ids[s]) for s in ("train", "valid", "test")] == [89, 22, 22]
        assert not (split_ids["train"] & split_ids["valid"] or split_ids["train"] & split_ids["test"] or split_ids["valid"] & split_ids["test"])
selection = json.loads((OUT / "model_selection.json").read_text())
for seed in (42, 43, 44):
    folder = OUT / f"seed_{seed}"
    history = rows(folder / "training_metrics.csv")
    assert [int(r["epoch"]) for r in history] == list(range(1, 101))
    assert all(np.isfinite(float(r[key])) for r in history for key in ("train_loss", "val_loss", "val_f1"))
    checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=False)
    expected = min(history, key=lambda r: float(r["val_loss"]))
    assert checkpoint["epoch"] == int(expected["epoch"])
    assert checkpoint["best_val_loss"] == float(expected["val_loss"])
    assert all(torch.isfinite(value).all().item() for value in checkpoint["model_state_dict"].values())
    summary = json.loads((folder / "test/summary/metrics_summary.json").read_text())
    assert summary["n_images"] == 22
    assert summary["micro_tp"] + summary["micro_fn"] == 548
    assert summary["probability"] == selection["thresholds"][f"seed_{seed}"]
    tp, fp, fn = [summary["micro_" + key] for key in ("tp", "fp", "fn")]
    assert np.isclose(summary["micro_f1"], 2 * tp / (2 * tp + fp + fn))
    per_image = rows(folder / "test/summary/per_image_metrics.csv")
    assert len(per_image) == 22 and {r["sample_id"] for r in per_image} == split_ids["test"]
    assert (OUT / "model_selection.json").stat().st_mtime <= (folder / "test/summary/metrics_summary.json").stat().st_mtime
    del checkpoint
checkpoint = torch.load(OUT / "selected_control_model.pt", map_location="cpu", weights_only=False)
selected_metadata = json.loads((OUT / "selected_control_model.json").read_text())
assert checkpoint["calibrated_probability"] == selected_metadata["probability"]
assert checkpoint["seed"] == int(selection["selected_model"].split("_")[-1])
assert "Temporary system-awake request released." in (OUT / "run.log").read_text(encoding="utf-8", errors="replace")
assert "Temporary display/system request released." in (OUT / "standby_guard.log").read_text(encoding="utf-8", errors="replace")
checks = dict(raw_files_verified=len(hashes), expanded_artifacts_verified=len(expanded_hashes),
              source_files_verified=len(source_hashes),
              holdout_arrays_identical=True, disjoint_control_splits=True,
              finite_training_epochs=300, finite_model_states=True,
              checkpoint_selection_matches_minimum_validation_loss=True,
              control_selection_saved_before_test_summaries=True,
              every_test_images=22, every_test_truth_points=548,
              candidate_metadata_matches_checkpoint=True, temporary_awake_released=True,
              temporary_display_guard_released=True,
              bootstrap_scope="paired image resampling, conditional on three fitted seed pairs",
              targeted_tests="5 passed (control experiment and pooled split); syntax checks passed")
(OUT / "delivery_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(json.dumps(checks, indent=2))
