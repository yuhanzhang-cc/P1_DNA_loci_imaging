"""Serialization must not turn an accepted subpixel point into H or W."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit

from deepblink_torch.round2_protocol import decode_candidates, local_peak_mask

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from research_v4.evaluation import save_candidates, load_candidates

spec = importlib.util.spec_from_file_location("h0_boundary_recovery_test", SCRIPTS / "recover_h0_boundary.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def legacy_decode(prediction, shape, *, dense=False, cell_size=4):
    """Pre-fix behavior, retained here only to reproduce the reported defect."""
    if dense:
        scores = expit(prediction[:shape[0], :shape[1], 0])
        rows, cols = np.nonzero(local_peak_mask(scores))
        coords = np.column_stack([rows, cols]) + prediction[rows, cols, 1:3]
        values = scores[rows, cols]
    else:
        rows, cols = np.indices(prediction.shape[:2])
        rows, cols = rows.ravel(), cols.ravel()
        coords = (np.column_stack([rows, cols]) + prediction[rows, cols, 1:3]) * cell_size
        values = prediction[rows, cols, 0]
    keep = (coords >= 0).all(1) & (coords < shape).all(1)
    return coords[keep].astype(np.float32), values[keep].astype(np.float32)


@pytest.mark.parametrize("shape", [(256, 512), (512, 256)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dense", [False, True])
def test_float32_roundout_is_removed_before_serialization(tmp_path, shape, axis, dense):
    cell = 1 if dense else 4
    prediction = np.zeros((512 // cell, 512 // cell, 3), np.float32)
    prediction[..., 0] = -10 if dense else .01
    location = [20, 30]
    location[axis] = shape[axis] // cell - 1
    prediction[tuple(location)] = [10 if dense else .9, .25, .25]
    prediction[tuple(location)][axis + 1] = np.nextafter(np.float32(1), np.float32(0))
    old = legacy_decode(prediction, shape, dense=dense, cell_size=cell)
    assert np.any(old[0] >= shape), "Fixture must reproduce the original failure"
    data = dict(original_shape=np.asarray([shape]), sample_id=np.asarray(["synthetic_edge"]))
    with pytest.raises(ValueError, match="out of bounds"):
        save_candidates(tmp_path / "old.npz", [old], data)
    corrected = decode_candidates(prediction, shape, dense=dense, cell_size=cell)
    assert corrected[0].dtype == corrected[1].dtype == np.float32
    assert np.all(corrected[0] >= 0) and np.all(corrected[0] < shape)
    # No clipping or score reassignment: every retained candidate is byte-exact.
    keep = (old[0] < shape).all(1)
    np.testing.assert_array_equal(corrected[0], old[0][keep])
    np.testing.assert_array_equal(corrected[1], old[1][keep])
    events = []
    patched = recovery.overlay_decoder(legacy_decode, events.append)(prediction, shape, dense=dense, cell_size=cell)
    np.testing.assert_array_equal(patched[0], corrected[0])
    np.testing.assert_array_equal(patched[1], corrected[1])
    assert sum(e["removed"] for e in events) == 1
    save_candidates(tmp_path / "corrected.npz", [corrected], data)
    restored = load_candidates(tmp_path / "corrected.npz", data)[0]
    np.testing.assert_array_equal(restored[0], corrected[0])
    np.testing.assert_array_equal(restored[1], corrected[1])


@pytest.mark.parametrize("dense", [False, True])
def test_normal_zero_and_true_outside_offsets_are_unchanged(dense):
    rng = np.random.default_rng(43)
    cell = 1 if dense else 4
    prediction = rng.uniform(.05, .95, (128, 128, 3)).astype(np.float32)
    prediction[0, 0, 1:] = 0
    prediction[100, 100, 1:] = 2
    shape = (100 * cell, 110 * cell)
    old = legacy_decode(prediction, shape, dense=dense, cell_size=cell)
    new = decode_candidates(prediction, shape, dense=dense, cell_size=cell)
    np.testing.assert_array_equal(new[0], old[0])
    np.testing.assert_array_equal(new[1], old[1])


@pytest.mark.parametrize("bad", [[-1e-6, 1], [256.01, 1], [np.nan, 1], [np.inf, 1]])
def test_overlay_rejects_other_coordinate_errors(bad):
    with pytest.raises(ValueError):
        recovery.filter_serialized_bounds(np.asarray([bad], np.float32), np.asarray([.9], np.float32), (256, 256))


def test_empty_candidates_and_genuine_nonfinite_prediction():
    coords, scores, dropped = recovery.filter_serialized_bounds(np.empty((0, 2), np.float32), np.empty(0, np.float32), (256, 256))
    assert coords.shape == (0, 2) and scores.shape == dropped.shape == (0,)
    prediction = np.zeros((4, 4, 3), np.float32)
    prediction[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        decode_candidates(prediction, (16, 16))


def test_protected_result_mutation_is_rejected(tmp_path, monkeypatch):
    from research_v4.common import sha256
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    model = tmp_path / "completed.pt"
    model.write_bytes(b"synthetic integrity fixture")
    record = dict(patch_id=recovery.PATCH_ID, overlay_sha256=sha256(recovery.__file__),
                  protected_files={"completed.pt": sha256(model)})
    recovery.verify_protected(record)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="completed result changed"):
        recovery.verify_protected(record)
