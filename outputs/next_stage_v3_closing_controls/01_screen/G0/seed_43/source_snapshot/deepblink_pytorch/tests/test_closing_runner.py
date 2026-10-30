"""Budget, pairing, freeze, ETA and singleton checks; never launch training."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("closing_runner_test", scripts / "train_closing_controls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change", ["seventh_run", "other_seed", "shorter_training", "test_access", "new_threshold_epochs"])
def test_protocol_rejects_unauthorized_changes(runner, change):
    plan = runner.read_json(runner.PLAN)
    runner.validate_plan(plan)
    if change == "seventh_run":
        plan["max_new_training_tasks"] = 7
    elif change == "other_seed":
        plan["authorized_targets"][-1][1] = 45
    elif change == "shorter_training":
        plan["epochs_per_task"] = 50
    elif change == "test_access":
        plan["test_arrays_loaded"] = True
    else:
        plan["checkpoint_epochs"].append(1)
    with pytest.raises(ValueError):
        runner.validate_plan(plan)


def test_pairing_detects_recipe_change(runner):
    reference = runner.read_json(runner.PARENTS[0] / "01_screen/R0/seed_42/registered_config.json")
    current = copy.deepcopy(reference)
    current["model"] = "deeploci_grid_control"
    current["protocol_id"] = runner.STUDY
    runner.check_recipe(current, reference)
    current["batch_size"] = 4
    with pytest.raises(ValueError, match="recipe"):
        runner.check_recipe(current, reference)


def test_same_seed_orders_match_all_three_archived_references(runner):
    import optimize_v3 as base
    with base.np.load(base.ARCHIVE, allow_pickle=True) as archive:
        ids = archive["sample_id_train"]
    for seed in (42, 43, 44):
        actual = runner.expected_order(base, ids, seed)
        assert actual == (runner.PARENTS[0] / f"01_screen/R0/seed_{seed}/sample_order.csv").read_bytes()


def test_eta_uses_actual_work_and_completed_state(runner):
    tasks = [dict(status="pending", epoch=0, elapsed_seconds=None)] * 6
    assert runner.estimate_remaining(tasks, now=1000, default_seconds_per_task=2600) == 15600
    tasks = [dict(status="complete", epoch=100, elapsed_seconds=2000),
             dict(status="running", epoch=10, first_epoch=1, started=800, elapsed_seconds=None)]
    assert runner.estimate_remaining(tasks, now=1000, default_seconds_per_task=2600) == 1800
    assert runner.estimate_remaining([dict(status="complete", epoch=100, elapsed_seconds=2000)], 1000, 2600) == 0


def test_document_missing_markers_is_not_overwritten(runner, tmp_path, monkeypatch):
    doc = tmp_path / "progress.md"
    doc.write_text("User changed document", encoding="utf-8")
    monkeypatch.setattr(runner, "DOC", doc)
    with pytest.raises(ValueError, match="markers"):
        runner.update_doc({})
    assert doc.read_text() == "User changed document"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows worker lock")
def test_second_worker_cannot_acquire_lock(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path / "study")
    with runner.single_worker():
        with pytest.raises(OSError):
            with runner.single_worker():
                pytest.fail("Duplicate worker acquired lock")


def test_lock_rejects_changed_sources(runner, tmp_path, monkeypatch):
    from types import SimpleNamespace
    out = tmp_path / "study"
    (out / "00_protocol").mkdir(parents=True)
    lock = dict(study_id=runner.STUDY, authorized_targets=[list(t) for t in runner.TARGETS],
                max_new_training_tasks=6, source_hashes={"original": "digest"})
    runner.write_json(out / "00_protocol/lock.json", lock)
    monkeypatch.setattr(runner, "OUT", out)
    with pytest.raises(ValueError, match="sources changed"):
        runner.validate_lock(SimpleNamespace(source_hashes=lambda: {"modified": "digest"}))
