import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from deepblink_torch.round2_protocol import (
    THRESHOLDS_V3, decode_candidates, dense_heatmap_loss, encode_dense_targets,
    local_peak_mask, sweep_candidate_counts, threshold_mask, training_orders)
from deepblink_torch.evaluation import match_coordinates


def test_dense_integer_fractional_edge_targets_and_gaussian():
    target = encode_dense_targets([[0, 0], [15.8, 15.1]], (16, 16), 32)
    assert target[0, 0, 0] == target[15, 15, 0] == 1
    np.testing.assert_allclose(target[15, 15, 1:3], [.8, .1], atol=1e-6)
    assert target[0, 1, 0] == pytest.approx(np.exp(-.5))
    assert target[0, 4, 0] == 0
    assert np.all(target[16:, :, 0] == -1)
    with pytest.raises(ValueError, match="collision"):
        encode_dense_targets([[1.1, 1.2], [1.5, 1.8]], (16, 16), 32)


def test_dense_loss_ignores_padding_and_regresses_exact_zero_offsets():
    target = torch.from_numpy(encode_dense_targets([[0., 0.]], (16, 16), 32))[None]
    prediction = torch.zeros(1, 32, 32, 3, requires_grad=True)
    with torch.no_grad(): prediction[..., 1:] = .2
    loss = dense_heatmap_loss(target, prediction)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, 16:]) == 0
    assert torch.count_nonzero(prediction.grad[0, 0, 0, 1:]) == 2
    cropped_loss = dense_heatmap_loss(target[:, :16, :16], prediction.detach()[:, :16, :16])
    torch.testing.assert_close(loss.detach(), cropped_loss)


def test_dense_peak_ties_crop_and_score_boundaries():
    scores = np.ones((8, 8), dtype=np.float32)
    keep = local_peak_mask(scores)
    assert keep.sum() == 1 and keep[0, 0]
    pred = np.zeros((16, 16, 3), dtype=np.float32)
    pred[..., 0] = -99
    pred[3, 4] = [99, .25, .75]
    pred[12, 12] = [100, .25, .75]
    coords, values = decode_candidates(pred, (8, 8), dense=True)
    np.testing.assert_allclose(coords[values > .9], [[3.25, 4.75]])
    assert threshold_mask(np.array([0., 1.]), 0).all()
    assert not threshold_mask(np.array([0., 1.]), 1).any()


@pytest.mark.parametrize("seed", range(10))
def test_accelerated_threshold_matcher_preserves_full_match_counts(seed):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 30, (12, 2)).astype(np.float32)
    coords = np.r_[truth+rng.normal(0, 1, truth.shape), rng.uniform(0, 100, (120, 2))].astype(np.float32)
    scores = rng.choice(np.array([0., .01, .5, .99, 1.], dtype=np.float32), len(coords))
    for threshold, row in zip(THRESHOLDS_V3, sweep_candidate_counts(coords, scores, truth)):
        expected = match_coordinates(coords[threshold_mask(scores, threshold)], truth, 3)
        assert row[:3] == (expected.tp, expected.fp, expected.fn)
        assert len(row[3]) == expected.tp


def test_orders_are_independent_of_model_initialization_and_grid_is_fixed():
    torch.manual_seed(900)
    first = training_orders(176, 100, 42)
    torch.randn(20000)
    assert first == training_orders(176, 100, 42)
    assert first != training_orders(176, 100, 43)
    assert all(sorted(order) == list(range(176)) for order in first)
    assert len(THRESHOLDS_V3) == 31
    assert all(a < b for a, b in zip(THRESHOLDS_V3, THRESHOLDS_V3[1:]))


def test_native_and_point_checkpoint_selection_and_exports(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("round2_runner_test", scripts / "optimize_v3.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [dict(epoch=5, f1=.8, precision=.8, threshold=.5),
                dict(epoch=10, f1=.8, precision=.8, threshold=.9)]
        assert max(rows, key=module.selection_key)["epoch"] == 5
        data = dict(sample_id=np.array(["synthetic"]), original_shape=np.array([[16, 16]]),
                    y=[np.array([[3., 4.]], dtype=np.float32)], weak=[np.array([True])])
        candidates = [(np.array([[3.1, 4.2]], dtype=np.float32), np.array([.9], dtype=np.float32))]
        selected = dict(threshold=.5, tp=1)
        result = module.export_selected(tmp_path, candidates, data, selected)
        assert result["tp_median_error_px"] == pytest.approx(5**.5/10, abs=1e-5)
        module.save_candidates(tmp_path / "c.npz", candidates, data)
        restored = module.load_candidates(tmp_path / "c.npz")
        np.testing.assert_array_equal(restored[0][0], candidates[0][0])
    finally:
        sys.path.remove(str(scripts))


def test_development_corpus_never_reads_test_arrays(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("round2_dev_boundary_test", scripts / "optimize_v3.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        accessed = []
        class FakeArchive:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __getitem__(self, key):
                accessed.append(key)
                assert not key.endswith("_test")
                prefix = key.rsplit("_", 1)[0]
                if prefix == "x": return np.zeros((1, 512, 512), dtype=np.uint16)
                if prefix == "y": return [np.empty((0, 2), dtype=np.float32)]
                if prefix == "sample_id": return np.array(["synthetic"])
                if prefix == "original_shape": return np.array([[256, 256]])
                raise AssertionError(key)
        monkeypatch.setattr(module.np, "load", lambda *args, **kwargs: FakeArchive())
        with pytest.raises(ValueError, match="split sizes"):
            module.DevelopmentCorpus()
        assert len(accessed) == 8
    finally:
        sys.path.remove(str(scripts))
