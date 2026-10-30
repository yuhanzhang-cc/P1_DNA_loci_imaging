"""Sigmoid offsets can equal one; scores must stay with source cells."""
import numpy as np
from deepblink_torch.cli import evaluate, predict


def test_duplicate_boundary_coordinates_keep_two_distinct_source_scores():
    grid = np.zeros((2, 2, 3), dtype=np.float32)
    grid[0, 0] = [.8, 1., 1.]
    grid[1, 1] = [.9, 0., 0.]
    result = predict._decode_with_scores(grid, cell_size=4, probability=.5, orig_h=8, orig_w=8)
    np.testing.assert_array_equal(result[:, :2], [[4., 4.], [4., 4.]])
    np.testing.assert_allclose(result[:, 2], [.8, .9])
    np.testing.assert_array_equal(result, evaluate._decode_prediction_with_probability(
        grid, cell_size=4, probability=.5, orig_h=8, orig_w=8))


def test_decoder_empty_output_and_padded_boundaries():
    grid = np.zeros((2, 2, 3), dtype=np.float32)
    assert predict._decode_with_scores(grid, cell_size=4, probability=.5, orig_h=8, orig_w=8).shape == (0, 3)
    grid[1, 1] = [.9, 1., 1.]
    assert predict._decode_with_scores(grid, cell_size=4, probability=.5, orig_h=8, orig_w=8).shape == (0, 3)
