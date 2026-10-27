"""Optimization split boundaries, decoding endpoints, and calibration budget."""
from pathlib import Path
import sys
import numpy as np
import pytest
from deepblink_torch.datasets.spots import SpotsDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from optimize_v2 import THRESHOLDS, decode_grid, load_development


def test_development_dataset_does_not_require_test_arrays(tmp_path):
    path = tmp_path / "development.npz"
    coords = np.empty(1, dtype=object)
    coords[0] = np.array([[2., 3.]], dtype=np.float32)
    np.savez(path, x_train=np.ones((1, 16, 16)), y_train=coords,
             x_valid=np.ones((1, 16, 16)), y_valid=coords)
    data = SpotsDataset(path, 4, development_only=True)
    assert len(data) == 1 and len(data.x_test) == 0
    with pytest.raises(ValueError, match="test split"):
        SpotsDataset(path, 4, subset="test", development_only=True)
    with pytest.raises(ValueError, match="must be present"):
        SpotsDataset(path, 4)
    with pytest.raises(ValueError, match="never test"):
        load_development("test")


def test_threshold_endpoints_include_numerical_zeros_and_exclude_padding():
    assert len(THRESHOLDS) == len(set(THRESHOLDS)) == 31
    grid = np.zeros((2, 2, 3), dtype=np.float32)
    grid[0, 1, 0] = 1.
    coords, _ = decode_grid(grid, (4, 8), 0.)
    assert len(coords) == 2
    assert len(decode_grid(grid, (4, 8), 1.)[0]) == 0
