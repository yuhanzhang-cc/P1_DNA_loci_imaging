"""Protect the shared holdouts and paired comparison estimand."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_clean220_control import filter_training_arrays, paired_image_bootstrap


def test_control_filters_all_training_metadata_and_preserves_holdouts():
    arrays = dict(sample_id_train=np.array(["a", "b", "c"]),
                  x_train=np.arange(12).reshape(3, 2, 2),
                  original_shape_train=np.array([[2, 2], [1, 1], [2, 2]]),
                  sample_id_valid=np.array(["v"]), sample_id_test=np.array(["t"]),
                  y_valid=np.array([[[1., 2.]]]), x_test=np.arange(4).reshape(1, 2, 2),
                  input_size=np.array(2))
    result, mask = filter_training_arrays(arrays, {"a", "c", "v", "t"})
    assert mask.tolist() == [True, False, True]
    assert result["sample_id_train"].tolist() == ["a", "c"]
    np.testing.assert_array_equal(result["x_train"], arrays["x_train"][[0, 2]])
    np.testing.assert_array_equal(result["original_shape_train"], arrays["original_shape_train"][[0, 2]])
    for key in ("sample_id_valid", "sample_id_test", "y_valid", "x_test", "input_size"):
        np.testing.assert_array_equal(result[key], arrays[key])
    arrays["sample_id_test"] = np.array(["a"])
    with pytest.raises(ValueError, match="overlap"):
        filter_training_arrays(arrays, {"a", "c"})


def test_bootstrap_respects_image_pairing_and_fixed_seeds():
    a = np.array([[[2, 1, 1], [6, 2, 2]], [[1, 2, 2], [7, 0, 1]]])
    result = paired_image_bootstrap(a, a, n_resamples=200)
    assert result["delta_mean_micro_f1"] == 0
    assert result["percentile_95_ci"] == [0., 0.]
    b = a.copy()
    b[..., 0] += 1
    b[..., 2] -= 1
    improved = paired_image_bootstrap(a, b, n_resamples=200)
    assert improved["delta_mean_micro_f1"] > 0
    assert improved["percentile_95_ci"][0] > 0
    assert improved == paired_image_bootstrap(a, b, n_resamples=200)
    reverse = paired_image_bootstrap(b, a, n_resamples=200)
    assert reverse["delta_mean_micro_f1"] == pytest.approx(-improved["delta_mean_micro_f1"])
    b[0, 0, 0] += 1
    with pytest.raises(ValueError, match="truth counts"):
        paired_image_bootstrap(a, b)


def test_endpoint_is_micro_f1_not_the_mean_image_f1():
    # One missed point in a tiny image and 90/100 in a large image.
    # Fixing the tiny image gives a small micro-F1 gain, not a 0.5 macro gain.
    control = np.array([[[0, 0, 1], [90, 0, 10]]])
    expanded = np.array([[[1, 0, 0], [90, 0, 10]]])
    result = paired_image_bootstrap(control, expanded, n_resamples=100)
    expected = 182 / 192 - 180 / 191
    assert result["delta_mean_micro_f1"] == pytest.approx(expected)
