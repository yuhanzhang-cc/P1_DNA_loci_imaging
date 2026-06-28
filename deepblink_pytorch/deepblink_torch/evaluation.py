"""Detection evaluation helpers for deepblink_torch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchResult:
    """One-to-one match result for a predicted and true coordinate set."""

    tp: int
    fp: int
    fn: int
    matched_pred_indices: np.ndarray
    matched_true_indices: np.ndarray
    matched_distances: np.ndarray
    unmatched_pred_indices: np.ndarray
    unmatched_true_indices: np.ndarray


def _as_coords(coords: np.ndarray | list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float32)
    if array.size == 0:
        return array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Coordinates must have shape (N, 2), got {array.shape}.")
    return array


def _safe_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1


def _nanmean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float32)
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def _nanmedian_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float32)
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmedian(array))


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate with NumPy 1.x/2.x compatibility."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def match_coordinates(pred_coords: np.ndarray, true_coords: np.ndarray, radius_px: float) -> MatchResult:
    """Match predicted and true coordinates using a Hungarian assignment.

    Pairs farther than ``radius_px`` are treated as unmatched.
    """
    if radius_px < 0:
        raise ValueError(f"radius_px must be >= 0, got {radius_px}.")

    pred = _as_coords(pred_coords)
    true = _as_coords(true_coords)
    n_pred = len(pred)
    n_true = len(true)

    if n_pred == 0 and n_true == 0:
        empty = np.empty((0,), dtype=np.int32)
        empty_f = np.empty((0,), dtype=np.float32)
        return MatchResult(0, 0, 0, empty, empty, empty_f, empty, empty)

    size = max(n_pred, n_true)
    dummy_cost = float(radius_px) + 1.0
    cost = np.full((size, size), dummy_cost, dtype=np.float64)

    distances = None
    if n_pred > 0 and n_true > 0:
        distances = np.linalg.norm(true[:, None, :] - pred[None, :, :], axis=2)
        cost[:n_true, :n_pred] = np.where(distances <= radius_px, distances, dummy_cost)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pred_indices: list[int] = []
    matched_true_indices: list[int] = []
    matched_distances: list[float] = []

    for row, col in zip(row_ind, col_ind):
        if row < n_true and col < n_pred and distances is not None:
            distance = float(distances[row, col])
            if distance <= radius_px:
                matched_true_indices.append(row)
                matched_pred_indices.append(col)
                matched_distances.append(distance)

    matched_true = np.asarray(matched_true_indices, dtype=np.int32)
    matched_pred = np.asarray(matched_pred_indices, dtype=np.int32)
    matched_distances_arr = np.asarray(matched_distances, dtype=np.float32)

    unmatched_true = np.asarray(
        sorted(set(range(n_true)) - set(matched_true_indices)),
        dtype=np.int32,
    )
    unmatched_pred = np.asarray(
        sorted(set(range(n_pred)) - set(matched_pred_indices)),
        dtype=np.int32,
    )

    tp = int(len(matched_true_indices))
    fp = int(n_pred - tp)
    fn = int(n_true - tp)

    return MatchResult(
        tp=tp,
        fp=fp,
        fn=fn,
        matched_pred_indices=matched_pred,
        matched_true_indices=matched_true,
        matched_distances=matched_distances_arr,
        unmatched_pred_indices=unmatched_pred,
        unmatched_true_indices=unmatched_true,
    )


def metrics_from_match(match: MatchResult) -> dict[str, float]:
    """Convert a match result to scalar metrics."""
    precision, recall, f1 = _safe_precision_recall_f1(match.tp, match.fp, match.fn)

    if len(match.matched_distances) == 0:
        if match.tp == 0 and match.fp == 0 and match.fn == 0:
            mean_error = 0.0
            median_error = 0.0
            rmse = 0.0
        else:
            mean_error = float("nan")
            median_error = float("nan")
            rmse = float("nan")
    else:
        mean_error = float(np.mean(match.matched_distances))
        median_error = float(np.median(match.matched_distances))
        rmse = float(np.sqrt(np.mean(np.square(match.matched_distances))))

    return {
        "tp": float(match.tp),
        "fp": float(match.fp),
        "fn": float(match.fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_error_px": mean_error,
        "median_error_px": median_error,
        "rmse_px": rmse,
    }


def f1_curve(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    max_distance_px: float = 3.0,
    n_cutoffs: int = 50,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute F1 across a range of distance cutoffs and integrate it."""
    if max_distance_px <= 0:
        raise ValueError(f"max_distance_px must be > 0, got {max_distance_px}.")
    if n_cutoffs < 2:
        raise ValueError(f"n_cutoffs must be >= 2, got {n_cutoffs}.")

    cutoffs = np.linspace(0.0, float(max_distance_px), num=n_cutoffs, dtype=np.float32)
    f1_scores = []
    pred = _as_coords(pred_coords)
    true = _as_coords(true_coords)

    for cutoff in cutoffs:
        match = match_coordinates(pred, true, float(cutoff))
        _, _, f1 = _safe_precision_recall_f1(match.tp, match.fp, match.fn)
        f1_scores.append(f1)

    f1_scores_array = np.asarray(f1_scores, dtype=np.float32)
    f1_integral = _trapezoid_integral(f1_scores_array, cutoffs) / float(max_distance_px)
    return cutoffs, f1_scores_array, f1_integral


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-image metrics into summary metrics."""
    if not rows:
        return {}

    total_tp = sum(int(row["tp"]) for row in rows)
    total_fp = sum(int(row["fp"]) for row in rows)
    total_fn = sum(int(row["fn"]) for row in rows)
    precision, recall, f1 = _safe_precision_recall_f1(total_tp, total_fp, total_fn)

    summary = {
        "micro_tp": float(total_tp),
        "micro_fp": float(total_fp),
        "micro_fn": float(total_fn),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(f1),
        "macro_precision": _nanmean_or_nan([row["precision"] for row in rows]),
        "macro_recall": _nanmean_or_nan([row["recall"] for row in rows]),
        "macro_f1": _nanmean_or_nan([row["f1"] for row in rows]),
        "mean_f1_integral": _nanmean_or_nan([row["f1_integral"] for row in rows]),
        "median_f1_integral": _nanmedian_or_nan([row["f1_integral"] for row in rows]),
        "mean_rmse_px": _nanmean_or_nan([row["rmse_px"] for row in rows]),
        "median_rmse_px": _nanmedian_or_nan([row["rmse_px"] for row in rows]),
        "mean_mean_error_px": _nanmean_or_nan([row["mean_error_px"] for row in rows]),
        "median_mean_error_px": _nanmedian_or_nan([row["mean_error_px"] for row in rows]),
        "n_images": float(len(rows)),
    }
    return summary
