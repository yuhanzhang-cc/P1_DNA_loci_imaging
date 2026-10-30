import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch

from deepblink_torch.cli import predict
from deepblink_torch.experiment_log import ExperimentLog
from deepblink_torch.experiment_log import sha256, write_json
from deepblink_torch.networks.closing_controls import UNetDeepLociGridControl, UNetDeepLociGlobalQuery
from deepblink_torch.networks.detail_route import UNetDeepLociDetailRoute
from deepblink_torch.networks.unet import UNetDeepBlink
from deepblink_torch.losses import combined_dice_rmse_loss
from deepblink_torch.round2_protocol import sweep_candidate_counts, THRESHOLDS_V3
from deepblink_torch.evaluation import match_coordinates
from deepblink_torch.spotiflow_bridge import (
    export_development, evaluate_checkpoints, save_candidates, read_candidates,
    spotiflow_candidates, validate_candidates)


@pytest.mark.parametrize("name,branch", [("deeploci_grid_control", "grid_projection"),
                                          ("deeploci_global_query", "query")])
def test_control_initial_equivalence_gradients_and_reload(tmp_path, name, branch):
    torch.manual_seed(42)
    base = UNetDeepBlink().eval()
    torch.manual_seed(42)
    model = predict._build_model_for_checkpoint(name, cell_size=4, device=torch.device("cpu")).eval()
    x = torch.randn(2, 1, 32, 32)
    mask = torch.ones_like(x)
    mask[1, :, 16:, :] = 0
    mask[1, :, :, 16:] = 0
    with torch.no_grad():
        assert torch.equal(base(x), model(x, mask))
        assert torch.isfinite(model(x, torch.zeros_like(mask))).all()
    target = torch.zeros(2, 8, 8, 3)
    target[:, 2, 2] = torch.tensor([1., .2, .7])
    target[1, 4:, :, 0] = -1
    target[1, :, 4:, 0] = -1
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = combined_dice_rmse_loss(target, model(x, mask))
        loss.backward()
        optimizer.step()
    parameters = list(getattr(model, branch).parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert any(torch.count_nonzero(p.grad) for p in parameters)
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(model_name=name, cell_size=4, input_size=512, model_state_dict=model.state_dict()), checkpoint)
    restored, cell = predict._load_model(checkpoint, torch.device("cpu"))
    restored.eval()
    with torch.no_grad():
        assert cell == 4 and torch.equal(restored(x, mask), model(x, mask))


def test_registered_parameter_budgets():
    count = lambda model: sum(p.numel() for p in model.parameters())
    base = count(UNetDeepBlink())
    assert count(UNetDeepLociGridControl()) - base == 6781
    assert count(UNetDeepLociDetailRoute()) - base == 6627
    assert count(UNetDeepLociGlobalQuery()) == count(UNetDeepLociDetailRoute())


def test_global_query_excludes_padding_and_is_shared():
    grid = torch.ones(2, 128, 8, 8)
    grid[:, :, 4:, :] = 1000
    mask = torch.ones(2, 1, 32, 32)
    mask[:, :, 16:, :] = 0
    summary = UNetDeepLociGlobalQuery.global_grid(grid, mask)
    assert summary.shape == (2, 128, 1, 1) and torch.equal(summary, torch.ones_like(summary))
    assert torch.count_nonzero(UNetDeepLociGlobalQuery.global_grid(grid, torch.zeros_like(mask))) == 0
    model = UNetDeepLociGlobalQuery()
    seen = []
    hook = model.query.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    model._detail_summary(torch.randn(2, 8, 32, 32), grid, mask)
    hook.remove()
    assert torch.equal(seen[0], summary.expand_as(grid))


def toy_archive(tmp_path):
    archive = tmp_path / "data.npz"
    values = {}
    rows = []
    for subset, n in [("train", 2), ("valid", 1)]:
        ids = [f"{subset}_{i}" for i in range(n)]
        images = np.arange(n * 16 * 16, dtype=np.float32).reshape(n, 16, 16)
        truths = np.tile(np.array([[[2.25, 5.75]]], dtype=np.float32), (n, 1, 1))
        values.update({f"x_{subset}": images, f"y_{subset}": truths,
                       f"sample_id_{subset}": ids, f"original_shape_{subset}": [[8, 16]] * n})
        rows += [dict(sample_id=s, subset=subset, height=8, width=16, n_points=1) for s in ids]
    # Deliberately no test arrays: exporter must never ask for them.
    np.savez(archive, **values)
    split = tmp_path / "split.csv"
    with split.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return archive, split, values


def test_export_round_trip_and_no_overwrite(tmp_path):
    archive, split, values = toy_archive(tmp_path)
    out = tmp_path / "export"
    rows = export_development(archive, split, out, frozen=False)
    assert len(rows) == 3 and not (out / "test").exists()
    assert tifffile.imread(out / "train/train_0.tif").shape == (8, 16)
    assert np.array_equal(tifffile.imread(out / "train/train_0.tif"), values["x_train"][0, :8])
    assert (out / "val/valid_0.csv").read_text().splitlines()[0] == "y,x"
    with pytest.raises(FileExistsError):
        export_development(archive, split, out, frozen=False)


def test_export_rejects_wrong_frozen_inputs(tmp_path):
    archive, split, _ = toy_archive(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        export_development(archive, split, tmp_path / "out")


def test_logging_preserves_failure(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        with ExperimentLog(output, {"seed": 42}):
            raise RuntimeError("synthetic failure")
    assert json.loads((output / "status.json").read_text())["event"] == "failed"
    assert "synthetic failure" in (output / "failure.txt").read_text()


def test_spotiflow_adapter_has_no_score_prefilter():
    class Fake:
        def predict(self, image, **kwargs):
            assert kwargs["prob_thresh"] == 0 and kwargs["subpix"]
            return np.array([[1., 2.], [-.1, 2.]]), SimpleNamespace(prob=np.array([.01, .9]))
    coords, scores = spotiflow_candidates(Fake(), np.zeros((8, 16)), "cpu")
    assert coords.tolist() == [[1., 2.]] and scores[0] == np.float32(.01)


def test_shared_evaluator_matching_and_candidate_round_trip(tmp_path):
    data = dict(sample_id=["a"], original_shape=np.array([[16, 16]]), y=[np.array([[2., 3.], [4., 5.]])])
    candidates = [(np.array([[2., 3.], [4., 5.], [15., 15.]], dtype=np.float32), np.array([.2, .9, .01], dtype=np.float32))]
    path = tmp_path / "candidates.npz"
    save_candidates(path, candidates, data["sample_id"], data["original_shape"])
    loaded = read_candidates(path, data)
    counts = sweep_candidate_counts(*loaded[0], data["y"][0])
    for threshold, count in zip(THRESHOLDS_V3, counts):
        scores = loaded[0][1]
        keep = np.ones(len(scores), dtype=bool) if threshold == 0 else scores > threshold
        match = match_coordinates(loaded[0][0][keep], data["y"][0], 3.)
        assert count[:3] == (match.tp, match.fp, match.fn)
    rows, selected, images = evaluate_checkpoints([(10, loaded), (5, loaded)], data)
    assert len(rows) == 62 and selected["f1"] == 1 and selected["epoch"] == 5
    assert images[0]["tp"] == 2
    data["sample_id"] = ["wrong"]
    with pytest.raises(ValueError, match="IDs"):
        read_candidates(path, data)


@pytest.mark.parametrize("scores", [[float("nan")], [1.1], [-.1], []])
def test_bad_candidate_scores_rejected(scores):
    with pytest.raises(ValueError):
        validate_candidates([[1., 2.]], scores, (8, 8))


@pytest.mark.parametrize("fault", [None, "mixed_seed", "changed_candidates"])
def test_evaluation_cli_logs_and_integrity(tmp_path, monkeypatch, fault):
    import importlib.util
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_spotiflow.py"
    spec = importlib.util.spec_from_file_location("closing_evaluation_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive, _, values = toy_archive(tmp_path)
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "ARCHIVE_SHA256", sha256(archive))
    runs = []
    for epoch in (5, 10):
        run = tmp_path / str(epoch)
        run.mkdir()
        save_candidates(run / "candidates.npz", [(values["y_valid"][0], np.array([.2]))],
                        values["sample_id_valid"], values["original_shape_valid"])
        write_json(run / "config.json", dict(archive_sha256=sha256(archive), epoch=epoch,
                   run_id="synthetic", seed=43 if fault == "mixed_seed" and epoch == 10 else 42,
                   pretrained=None, device="cpu"))
        write_json(run / "status.json", dict(event="completed"))
        write_json(run / "candidate_integrity.json", dict(
                   sha256="wrong" if fault == "changed_candidates" else sha256(run / "candidates.npz")))
        runs.append(str(run))
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", [str(script), "evaluate", "--runs", *runs, "--output", str(output)])
    if fault:
        with pytest.raises(ValueError):
            module.main()
        assert json.loads((output / "status.json").read_text())["event"] == "failed"
    else:
        module.main()
        selected = json.loads((output / "selection.json").read_text())
        assert selected["epoch"] == 5 and selected["f1"] == 1.
        assert selected["checkpoint_count"] == 2 and not selected["full_20_checkpoint_budget"]
        assert json.loads((output / "status.json").read_text())["event"] == "completed"
