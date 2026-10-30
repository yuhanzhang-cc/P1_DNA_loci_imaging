"""Detection matching must prioritize true positives over shorter distances."""
import numpy as np
import pytest

from deepblink_torch.evaluation import match_coordinates


def test_two_long_valid_matches_beat_one_zero_distance_match():
    truth = np.array([[0., 0.], [0., 3.]])
    pred = np.array([[0., 0.], [0., -3.]])
    match = match_coordinates(pred, truth, 3.)
    assert (match.tp, match.fp, match.fn) == (2, 0, 0)
    assert sorted(match.matched_distances) == [3., 3.]


def test_dense_candidates_and_empty_inputs():
    pred = np.column_stack([np.arange(16384), np.zeros(16384)])
    match = match_coordinates(pred, np.array([[17., 0.], [29., 0.]]), 0.)
    assert (match.tp, match.fp, match.fn) == (2, 16382, 0)
    assert match_coordinates(pred[:0], pred[:2], 3).fn == 2
    assert match_coordinates(pred[:2], pred[:0], 3).fp == 2
    with pytest.raises(ValueError, match="finite"):
        match_coordinates(np.array([[np.nan, 0]]), pred[:2], 3)
