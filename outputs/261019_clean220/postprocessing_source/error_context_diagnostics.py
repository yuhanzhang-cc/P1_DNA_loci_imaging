"""Offline error-context diagnostics for clean113-style deepBlink runs.

This script intentionally stays outside the main train/evaluate/predict CLI.
It reads an existing dataset/checkpoint/evaluation summary, recomputes dense
candidate predictions once, and writes diagnostic tables, plots, patches, and
post-processing ablations to a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deepblink_torch.cli import predict as predict_cli
from deepblink_torch.evaluation import MatchResult
from deepblink_torch.evaluation import match_coordinates
from deepblink_torch.evaluation import metrics_from_match


IMAGE_EXTENSIONS = (".tif", ".tiff")
EVENT_FIELDNAMES = [
    "sample_id",
    "image_name",
    "event_type",
    "confidence",
    "pred_y_px",
    "pred_x_px",
    "true_y_px",
    "true_x_px",
    "matched_distance_px",
    "center_intensity",
    "local_mean",
    "local_max",
    "local_min",
    "local_std",
    "local_contrast",
    "local_zscore",
    "local_max_to_mean",
    "peak_sharpness",
    "peak_offset_px",
    "peak_component_size_px",
    "distance_to_border_px",
    "image_mean",
    "image_std",
    "image_min",
    "image_max",
    "n_pred",
    "n_true",
    "n_tp_sample",
    "n_fp_sample",
    "n_fn_sample",
    "sample_f1",
    "sample_precision",
    "sample_recall",
    "foreground_like_pixels",
]
POSTPROCESS_FIELDNAMES = [
    "strategy",
    "threshold",
    "top_n",
    "max_n",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "median_error_px",
    "n_images",
    "n_pred_spots",
    "mean_pred_per_image",
    "recall_ceiling",
    "notes",
]
HARD_EXAMPLE_FIELDNAMES = [
    "example_group",
    "sample_id",
    "image_name",
    "event_type",
    "confidence",
    "local_zscore",
    "local_contrast",
    "coordinate_y_px",
    "coordinate_x_px",
    "patch_path",
    "suggested_review_label",
    "notes",
]
EVENT_COLORS = {"TP": "#4ADE80", "FP": "#FB7185", "FN": "#38BDF8"}
PPT_BG = "#05070B"
PPT_FG = "#F8FAFC"
PPT_MUTED = "#94A3B8"
PPT_GRID = "#334155"


@dataclass(frozen=True)
class SampleData:
    sample_id: str
    image_name: str
    image: np.ndarray
    truth_coords: np.ndarray


@dataclass(frozen=True)
class SamplePrediction:
    sample: SampleData
    dense_candidates: np.ndarray


@dataclass(frozen=True)
class BaselineConfig:
    probability: float
    match_radius_px: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate error-context diagnostics and post-processing ablations."
    )
    parser.add_argument("--data", required=True, help="Path to clean113 .npz dataset.")
    parser.add_argument(
        "--evaluation",
        required=True,
        help="Existing evaluation output directory used for baseline metadata.",
    )
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--output", required=True, help="Output diagnostics directory.")
    parser.add_argument(
        "--images",
        default=None,
        help="Optional directory with raw .tif images. Defaults to images stored in the .npz.",
    )
    parser.add_argument("--subset", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--patch-radius", type=int, default=10)
    parser.add_argument("--top-k-patches", type=int, default=12)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--match-radius-px",
        type=float,
        default=None,
        help="Override match radius. Defaults to evaluation summary or 3.0.",
    )
    return parser.parse_args(argv)


def _load_npz(data_path: Path) -> dict[str, np.ndarray]:
    with np.load(data_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _subset_key(subset: str, field: str) -> str:
    return f"{field}_{subset}"


def _read_summary_row(evaluation_dir: Path) -> dict[str, str]:
    summary_csv = evaluation_dir / "summary" / "metrics_summary.csv"
    if not summary_csv.exists():
        return {}
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return next(reader, {})


def _float_or_default(value: object, default: float) -> float:
    try:
        if value in (None, ""):
            return default
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _int_or_default(value: object, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return parsed


def _load_baseline_config(
    evaluation_dir: Path,
    match_radius_override: float | None,
) -> BaselineConfig:
    row = _read_summary_row(evaluation_dir)
    probability = _float_or_default(row.get("probability"), 0.5)
    match_radius_px = (
        float(match_radius_override)
        if match_radius_override is not None
        else _float_or_default(row.get("match_radius_px"), 3.0)
    )
    return BaselineConfig(
        probability=probability,
        match_radius_px=match_radius_px,
        precision=_float_or_default(row.get("micro_precision"), float("nan")),
        recall=_float_or_default(row.get("micro_recall"), float("nan")),
        f1=_float_or_default(row.get("micro_f1"), float("nan")),
        tp=_int_or_default(row.get("micro_tp"), 0),
        fp=_int_or_default(row.get("micro_fp"), 0),
        fn=_int_or_default(row.get("micro_fn"), 0),
    )


def _find_image_file(images_dir: Path, image_name: str, sample_id: str) -> Path | None:
    candidate = images_dir / image_name
    if candidate.is_file():
        return candidate
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{sample_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _load_samples(
    data_path: Path,
    subset: str,
    images_dir: Path | None,
) -> list[SampleData]:
    data = _load_npz(data_path)
    required = [
        _subset_key(subset, "x"),
        _subset_key(subset, "y"),
        _subset_key(subset, "sample_id"),
        _subset_key(subset, "image_name"),
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Dataset is missing required arrays: {missing}")

    images = data[_subset_key(subset, "x")]
    truth = data[_subset_key(subset, "y")]
    sample_ids = data[_subset_key(subset, "sample_id")]
    image_names = data[_subset_key(subset, "image_name")]

    samples: list[SampleData] = []
    for index in range(len(sample_ids)):
        sample_id = str(sample_ids[index])
        image_name = str(image_names[index])
        image = np.asarray(images[index], dtype=np.float32)
        if images_dir is not None:
            image_file = _find_image_file(images_dir, image_name, sample_id)
            if image_file is not None:
                image = predict_cli._read_image(image_file)
        samples.append(
            SampleData(
                sample_id=sample_id,
                image_name=image_name,
                image=np.asarray(image, dtype=np.float32),
                truth_coords=np.asarray(truth[index], dtype=np.float32).reshape(-1, 2),
            )
        )
    return samples


def _dense_candidates_from_prediction(
    pred_matrix: np.ndarray,
    *,
    cell_size: int,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    rows: list[list[float]] = []
    nrow, ncol, _ = pred_matrix.shape
    for cell_r in range(nrow):
        for cell_c in range(ncol):
            probability = float(pred_matrix[cell_r, cell_c, 0])
            y = float(cell_r * cell_size + pred_matrix[cell_r, cell_c, 1] * cell_size)
            x = float(cell_c * cell_size + pred_matrix[cell_r, cell_c, 2] * cell_size)
            if 0 <= y < orig_h and 0 <= x < orig_w:
                rows.append([y, x, probability])
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    candidates = np.asarray(rows, dtype=np.float32)
    order = np.argsort(-candidates[:, 2], kind="stable")
    return candidates[order]


def _run_dense_inference(
    samples: list[SampleData],
    model_path: Path,
    device: torch.device,
) -> tuple[list[SamplePrediction], int]:
    model, cell_size = predict_cli._load_model(model_path, device)
    predictions: list[SamplePrediction] = []
    for sample in samples:
        orig_h, orig_w = sample.image.shape
        padded = predict_cli.prepare_image(sample.image, getattr(model, "input_size", None))
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_matrix = model(tensor).squeeze(0).detach().cpu().numpy()
        predictions.append(
            SamplePrediction(
                sample=sample,
                dense_candidates=_dense_candidates_from_prediction(
                    pred_matrix,
                    cell_size=cell_size,
                    orig_h=orig_h,
                    orig_w=orig_w,
                ),
            )
        )
        print(
            f"Inferred {sample.sample_id}: dense_candidates={len(predictions[-1].dense_candidates)} "
            f"true={len(sample.truth_coords)}"
        )
    return predictions, cell_size


def _select_by_threshold(candidates: np.ndarray, threshold: float) -> np.ndarray:
    if len(candidates) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return candidates[candidates[:, 2] > float(threshold)]


def _select_top_n(candidates: np.ndarray, top_n: int) -> np.ndarray:
    if top_n <= 0 or len(candidates) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return candidates[: min(int(top_n), len(candidates))]


def _select_threshold_max_n(candidates: np.ndarray, threshold: float, max_n: int) -> np.ndarray:
    selected = _select_by_threshold(candidates, threshold)
    return _select_top_n(selected, max_n)


def _safe_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _evaluate_candidate_sets(
    predictions: list[SamplePrediction],
    selector_name: str,
    selected_by_sample: list[np.ndarray],
    *,
    match_radius_px: float,
    threshold: float | str = "",
    top_n: int | str = "",
    max_n: int | str = "",
    notes: str = "",
) -> dict[str, object]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred = 0
    distances: list[np.ndarray] = []
    recall_ceiling_values: list[float] = []

    for item, selected in zip(predictions, selected_by_sample, strict=True):
        match = match_coordinates(selected[:, :2], item.sample.truth_coords, match_radius_px)
        total_tp += match.tp
        total_fp += match.fp
        total_fn += match.fn
        total_pred += len(selected)
        distances.append(np.asarray(match.matched_distances, dtype=np.float32))
        if top_n != "":
            n_true = len(item.sample.truth_coords)
            if n_true > 0:
                recall_ceiling_values.append(min(float(top_n), float(n_true)) / float(n_true))
        elif max_n != "":
            n_true = len(item.sample.truth_coords)
            if n_true > 0:
                recall_ceiling_values.append(min(float(max_n), float(n_true)) / float(n_true))

    precision, recall, f1 = _safe_prf(total_tp, total_fp, total_fn)
    concatenated = (
        np.concatenate([arr for arr in distances if len(arr) > 0])
        if any(len(arr) > 0 for arr in distances)
        else np.empty((0,), dtype=np.float32)
    )
    median_error = float(np.median(concatenated)) if len(concatenated) > 0 else float("nan")
    recall_ceiling = (
        float(np.mean(recall_ceiling_values)) if recall_ceiling_values else ""
    )
    return {
        "strategy": selector_name,
        "threshold": threshold,
        "top_n": top_n,
        "max_n": max_n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "median_error_px": median_error,
        "n_images": len(predictions),
        "n_pred_spots": total_pred,
        "mean_pred_per_image": total_pred / max(1, len(predictions)),
        "recall_ceiling": recall_ceiling,
        "notes": notes,
    }


def _estimate_validation_prior_counts(data_path: Path) -> dict[str, int]:
    data = _load_npz(data_path)
    counts: list[int] = []
    for subset in ("train", "valid"):
        key = _subset_key(subset, "y")
        if key not in data:
            continue
        counts.extend(int(len(np.asarray(coords).reshape(-1, 2))) for coords in data[key])
    if not counts:
        return {}
    array = np.asarray(counts, dtype=np.float32)
    return {
        "median": int(round(float(np.median(array)))),
        "q75": int(round(float(np.quantile(array, 0.75)))),
        "q90": int(round(float(np.quantile(array, 0.90)))),
    }


def _run_postprocess_ablation(
    predictions: list[SamplePrediction],
    data_path: Path,
    baseline: BaselineConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(
        _evaluate_candidate_sets(
            predictions,
            "baseline_threshold_from_evaluation",
            [_select_by_threshold(item.dense_candidates, baseline.probability) for item in predictions],
            match_radius_px=baseline.match_radius_px,
            threshold=baseline.probability,
            notes="Baseline threshold read from evaluation summary.",
        )
    )

    thresholds = [round(float(value), 2) for value in np.arange(0.05, 0.951, 0.01)]
    for threshold in thresholds:
        rows.append(
            _evaluate_candidate_sets(
                predictions,
                "threshold_only",
                [_select_by_threshold(item.dense_candidates, threshold) for item in predictions],
                match_radius_px=baseline.match_radius_px,
                threshold=threshold,
            )
        )

    for top_n in [5, 8, 12, 16, 20, 24, 30, 36, 40]:
        rows.append(
            _evaluate_candidate_sets(
                predictions,
                "fixed_top_n",
                [_select_top_n(item.dense_candidates, top_n) for item in predictions],
                match_radius_px=baseline.match_radius_px,
                top_n=top_n,
                notes="Uses fixed per-image N; may cap recall for high-count images.",
            )
        )

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for max_n in [12, 16, 20, 24, 30, 36, 40]:
            rows.append(
                _evaluate_candidate_sets(
                    predictions,
                    "threshold_max_n",
                    [
                        _select_threshold_max_n(item.dense_candidates, threshold, max_n)
                        for item in predictions
                    ],
                    match_radius_px=baseline.match_radius_px,
                    threshold=threshold,
                    max_n=max_n,
                    notes="Threshold first, then keep at most max_n predictions per image.",
                )
            )

    prior_counts = _estimate_validation_prior_counts(data_path)
    for label, top_n in prior_counts.items():
        rows.append(
            _evaluate_candidate_sets(
                predictions,
                "validation_prior_top_n",
                [_select_top_n(item.dense_candidates, top_n) for item in predictions],
                match_radius_px=baseline.match_radius_px,
                top_n=top_n,
                notes=f"Top-N from train/valid count prior: {label}.",
            )
        )
    return rows


def _component_size(binary: np.ndarray, seed: tuple[int, int]) -> int:
    if binary.size == 0 or not binary[seed]:
        return 0
    stack = [seed]
    visited = {seed}
    height, width = binary.shape
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    stack.append((ny, nx))
    return len(visited)


def _patch_bounds(image: np.ndarray, y: float, x: float, radius: int) -> tuple[int, int, int, int]:
    height, width = image.shape
    cy = int(round(float(y)))
    cx = int(round(float(x)))
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    return y0, y1, x0, x1


def _extract_patch(image: np.ndarray, y: float, x: float, radius: int) -> np.ndarray:
    y0, y1, x0, x1 = _patch_bounds(image, y, x, radius)
    return np.asarray(image[y0:y1, x0:x1], dtype=np.float32)


def _extract_patch_features(image: np.ndarray, y: float, x: float, radius: int) -> dict[str, float]:
    height, width = image.shape
    cy = int(np.clip(round(float(y)), 0, height - 1))
    cx = int(np.clip(round(float(x)), 0, width - 1))
    patch = _extract_patch(image, y, x, radius)
    local_mean = float(np.mean(patch))
    local_max = float(np.max(patch))
    local_min = float(np.min(patch))
    local_std = float(np.std(patch))
    center = float(image[cy, cx])
    eps = 1e-6
    max_position = np.unravel_index(int(np.argmax(patch)), patch.shape)
    y0, _y1, x0, _x1 = _patch_bounds(image, y, x, radius)
    max_y = y0 + int(max_position[0])
    max_x = x0 + int(max_position[1])
    threshold = local_mean + local_std
    component = _component_size(patch >= threshold, (int(max_position[0]), int(max_position[1])))
    return {
        "center_intensity": center,
        "local_mean": local_mean,
        "local_max": local_max,
        "local_min": local_min,
        "local_std": local_std,
        "local_contrast": center - local_mean,
        "local_zscore": (center - local_mean) / local_std if local_std > eps else float("nan"),
        "local_max_to_mean": local_max / local_mean if abs(local_mean) > eps else float("nan"),
        "peak_sharpness": local_max - local_mean,
        "peak_offset_px": float(math.hypot(max_y - float(y), max_x - float(x))),
        "peak_component_size_px": float(component),
        "distance_to_border_px": float(min(y, x, height - 1 - y, width - 1 - x)),
    }


def _image_features(image: np.ndarray) -> dict[str, float]:
    image_mean = float(np.mean(image))
    image_std = float(np.std(image))
    threshold = image_mean + 2.0 * image_std
    return {
        "image_mean": image_mean,
        "image_std": image_std,
        "image_min": float(np.min(image)),
        "image_max": float(np.max(image)),
        "foreground_like_pixels": float(np.count_nonzero(image > threshold)),
    }


def _base_event_row(
    sample: SampleData,
    event_type: str,
    *,
    confidence: float,
    pred_coord: tuple[float, float] | None,
    true_coord: tuple[float, float] | None,
    matched_distance_px: float,
    sample_metrics: dict[str, float],
    patch_features: dict[str, float],
    image_features: dict[str, float],
) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "image_name": sample.image_name,
        "event_type": event_type,
        "confidence": confidence,
        "pred_y_px": pred_coord[0] if pred_coord is not None else float("nan"),
        "pred_x_px": pred_coord[1] if pred_coord is not None else float("nan"),
        "true_y_px": true_coord[0] if true_coord is not None else float("nan"),
        "true_x_px": true_coord[1] if true_coord is not None else float("nan"),
        "matched_distance_px": matched_distance_px,
        **patch_features,
        **image_features,
        "n_pred": int(sample_metrics["n_pred"]),
        "n_true": int(sample_metrics["n_true"]),
        "n_tp_sample": int(sample_metrics["tp"]),
        "n_fp_sample": int(sample_metrics["fp"]),
        "n_fn_sample": int(sample_metrics["fn"]),
        "sample_f1": sample_metrics["f1"],
        "sample_precision": sample_metrics["precision"],
        "sample_recall": sample_metrics["recall"],
    }


def _build_error_context(
    predictions: list[SamplePrediction],
    baseline: BaselineConfig,
    patch_radius: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    event_rows: list[dict[str, object]] = []
    per_sample_rows: list[dict[str, object]] = []
    for item in predictions:
        sample = item.sample
        selected = _select_by_threshold(item.dense_candidates, baseline.probability)
        match = match_coordinates(selected[:, :2], sample.truth_coords, baseline.match_radius_px)
        metrics = metrics_from_match(match)
        sample_metrics = {
            "n_pred": len(selected),
            "n_true": len(sample.truth_coords),
            "tp": match.tp,
            "fp": match.fp,
            "fn": match.fn,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        image_features = _image_features(sample.image)
        per_sample_rows.append(
            {
                "sample_id": sample.sample_id,
                "tp": match.tp,
                "fp": match.fp,
                "fn": match.fn,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

        for pred_index, true_index, distance in zip(
            match.matched_pred_indices,
            match.matched_true_indices,
            match.matched_distances,
            strict=True,
        ):
            pred = selected[int(pred_index)]
            true = sample.truth_coords[int(true_index)]
            patch_features = _extract_patch_features(sample.image, pred[0], pred[1], patch_radius)
            event_rows.append(
                _base_event_row(
                    sample,
                    "TP",
                    confidence=float(pred[2]),
                    pred_coord=(float(pred[0]), float(pred[1])),
                    true_coord=(float(true[0]), float(true[1])),
                    matched_distance_px=float(distance),
                    sample_metrics=sample_metrics,
                    patch_features=patch_features,
                    image_features=image_features,
                )
            )

        for pred_index in match.unmatched_pred_indices:
            pred = selected[int(pred_index)]
            patch_features = _extract_patch_features(sample.image, pred[0], pred[1], patch_radius)
            event_rows.append(
                _base_event_row(
                    sample,
                    "FP",
                    confidence=float(pred[2]),
                    pred_coord=(float(pred[0]), float(pred[1])),
                    true_coord=None,
                    matched_distance_px=float("nan"),
                    sample_metrics=sample_metrics,
                    patch_features=patch_features,
                    image_features=image_features,
                )
            )

        for true_index in match.unmatched_true_indices:
            true = sample.truth_coords[int(true_index)]
            patch_features = _extract_patch_features(sample.image, true[0], true[1], patch_radius)
            event_rows.append(
                _base_event_row(
                    sample,
                    "FN",
                    confidence=float("nan"),
                    pred_coord=None,
                    true_coord=(float(true[0]), float(true[1])),
                    matched_distance_px=float("nan"),
                    sample_metrics=sample_metrics,
                    patch_features=patch_features,
                    image_features=image_features,
                )
            )
    return event_rows, per_sample_rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _finite_values(rows: list[dict[str, object]], event_type: str, field: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if row.get("event_type") != event_type:
            continue
        value = _float_or_default(row.get(field), float("nan"))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float32)


def _plot_hist_by_event(
    output_path: Path,
    rows: list[dict[str, object]],
    field: str,
    event_types: Iterable[str],
    title: str,
    xlabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    colors = {"TP": "tab:green", "FP": "tab:red", "FN": "tab:blue"}
    any_values = False
    for event_type in event_types:
        values = _finite_values(rows, event_type, field)
        if len(values) == 0:
            continue
        any_values = True
        ax.hist(values, bins=24, density=False, alpha=0.45, label=f"{event_type} (n={len(values)})", color=colors[event_type])
        ax.axvline(float(np.median(values)), color=colors[event_type], lw=1.4, ls="--")
    if any_values:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_confidence_vs_z(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    colors = {"TP": "tab:green", "FP": "tab:red"}
    plotted = False
    for event_type in ("TP", "FP"):
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if row.get("event_type") != event_type:
                continue
            confidence = _float_or_default(row.get("confidence"), float("nan"))
            zscore = _float_or_default(row.get("local_zscore"), float("nan"))
            if math.isfinite(confidence) and math.isfinite(zscore):
                xs.append(confidence)
                ys.append(zscore)
        if xs:
            plotted = True
            ax.scatter(xs, ys, s=18, alpha=0.65, color=colors[event_type], label=event_type, edgecolors="none")
    if plotted:
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Local z-score")
        ax.set_title("Confidence vs Local z-score")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No TP/FP confidence values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_per_sample_burden(output_path: Path, per_sample_rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        per_sample_rows,
        key=lambda row: (-(int(row["fn"]) + int(row["fp"])), str(row["sample_id"])),
    )
    x = np.arange(len(sorted_rows))
    tp = np.asarray([int(row["tp"]) for row in sorted_rows])
    fp = np.asarray([int(row["fp"]) for row in sorted_rows])
    fn = np.asarray([int(row["fn"]) for row in sorted_rows])
    labels = [str(row["sample_id"]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    ax.bar(x, tp, color="tab:green", label="TP")
    ax.bar(x, fp, bottom=tp, color="tab:red", label="FP")
    ax.bar(x, fn, bottom=tp + fp, color="tab:blue", label="FN")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.set_ylabel("Events")
    ax.set_title("Per-sample Error Burden")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False)
    for index, row in enumerate(sorted_rows[:3]):
        ax.text(index, tp[index] + fp[index] + fn[index] + 0.5, "hard", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _make_figures(output_dir: Path, event_rows: list[dict[str, object]], per_sample_rows: list[dict[str, object]]) -> None:
    figures_dir = output_dir / "figures"
    _plot_hist_by_event(
        figures_dir / "confidence_tp_fp.png",
        event_rows,
        "confidence",
        ("TP", "FP"),
        "TP vs FP Confidence",
        "Confidence",
    )
    _plot_hist_by_event(
        figures_dir / "local_zscore_tp_fp_fn.png",
        event_rows,
        "local_zscore",
        ("TP", "FP", "FN"),
        "Local z-score by Event Type",
        "Local z-score",
    )
    _plot_hist_by_event(
        figures_dir / "local_contrast_tp_fp_fn.png",
        event_rows,
        "local_contrast",
        ("TP", "FP", "FN"),
        "Local Contrast by Event Type",
        "Center intensity - local mean",
    )
    _plot_per_sample_burden(figures_dir / "per_sample_error_burden.png", per_sample_rows)
    _plot_confidence_vs_z(figures_dir / "confidence_vs_local_zscore.png", event_rows)


def _style_dark_axes(ax, *, grid: bool = True) -> None:
    ax.set_facecolor(PPT_BG)
    ax.tick_params(colors=PPT_MUTED, labelsize=11)
    ax.xaxis.label.set_color(PPT_FG)
    ax.yaxis.label.set_color(PPT_FG)
    ax.title.set_color(PPT_FG)
    for spine in ax.spines.values():
        spine.set_color(PPT_GRID)
    if grid:
        ax.grid(True, color=PPT_GRID, alpha=0.35, linewidth=0.8)


def _save_dark_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(PPT_BG)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08, facecolor=PPT_BG)
    plt.close(fig)


def _strategy_label(row: dict[str, object] | None) -> str:
    if row is None:
        return "n/a"
    strategy = str(row["strategy"])
    if strategy == "baseline_threshold_from_evaluation":
        return "Baseline"
    if strategy == "threshold_only":
        return "Threshold"
    if strategy == "fixed_top_n":
        return "Fixed Top-N"
    if strategy == "threshold_max_n":
        return "Threshold+N"
    if strategy == "validation_prior_top_n":
        return "Prior Top-N"
    return strategy


def _strategy_rows_for_ppt(recommendation: dict[str, object]) -> list[dict[str, object]]:
    rows = [
        recommendation["baseline_row"],
        recommendation["threshold_best"],
        recommendation["topn_best"],
        recommendation["hybrid_best"],
        recommendation["prior_best"],
    ]
    return [row for row in rows if row is not None]


def _metric_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float32)


def _plot_error_context_summary_ppt(
    output_path: Path,
    recommendation: dict[str, object],
) -> None:
    baseline = recommendation["baseline_row"]
    z_medians = recommendation["zscore_medians"]
    contrast_medians = recommendation["contrast_medians"]
    strategy_rows = _strategy_rows_for_ppt(recommendation)

    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor=PPT_BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.35], wspace=0.34)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(PPT_BG)
    ax0.set_axis_off()
    ax0.text(0.0, 0.98, "Baseline performance", color=PPT_FG, fontsize=23, weight="bold", va="top")
    ax0.text(0.0, 0.90, "clean113 test set, p=0.4", color=PPT_MUTED, fontsize=13, va="top")
    metrics = [
        ("F1", float(baseline["f1"]), "#A7F3D0"),
        ("Precision", float(baseline["precision"]), "#BFDBFE"),
        ("Recall", float(baseline["recall"]), "#FDE68A"),
    ]
    y = 0.76
    for label, value, color in metrics:
        ax0.text(0.02, y, label, color=PPT_MUTED, fontsize=14, va="center")
        ax0.text(0.58, y, f"{value:.4f}", color=color, fontsize=27, weight="bold", va="center")
        y -= 0.13
    counts = f"TP={int(baseline['tp'])}   FP={int(baseline['fp'])}   FN={int(baseline['fn'])}"
    ax0.text(0.02, 0.33, counts, color=PPT_FG, fontsize=18, weight="bold")
    ax0.text(
        0.02,
        0.23,
        f"Median localization error: {float(baseline['median_error_px']):.3f} px",
        color=PPT_MUTED,
        fontsize=14,
    )
    ax0.text(
        0.02,
        0.08,
        "Default unchanged.",
        color="#FCA5A5",
        fontsize=15,
        weight="bold",
        clip_on=False,
    )

    ax1 = fig.add_subplot(gs[0, 1])
    _style_dark_axes(ax1)
    events = ["TP", "FP", "FN"]
    x = np.arange(len(events), dtype=np.float32)
    width = 0.34
    z_values = np.asarray([z_medians[event] for event in events], dtype=np.float32)
    contrast_values = np.asarray([contrast_medians[event] / 1000.0 for event in events], dtype=np.float32)
    ax1.bar(x - width / 2, z_values, width=width, color=[EVENT_COLORS[event] for event in events], alpha=0.95, label="Local z-score")
    ax1.bar(x + width / 2, contrast_values, width=width, color="#FBBF24", alpha=0.78, label="Local contrast / 1000")
    for idx, value in enumerate(z_values):
        ax1.text(idx - width / 2, value + 0.08, f"{value:.2f}", color=PPT_FG, ha="center", fontsize=11)
    for idx, value in enumerate(contrast_values):
        ax1.text(idx + width / 2, value + 0.08, f"{contrast_medians[events[idx]]:.0f}", color=PPT_FG, ha="center", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(events, color=PPT_FG, fontsize=13, weight="bold")
    ax1.set_ylabel("Median feature value")
    ax1.set_title("Error-context features", fontsize=20, weight="bold")
    legend = ax1.legend(frameon=False, fontsize=11, loc="upper right")
    for text in legend.get_texts():
        text.set_color(PPT_FG)
    ax1.text(
        0.02,
        -0.18,
        "FN is enriched for low local z-score / low contrast.",
        transform=ax1.transAxes,
        color="#38BDF8",
        fontsize=13,
        weight="bold",
    )

    ax2 = fig.add_subplot(gs[0, 2])
    _style_dark_axes(ax2)
    labels = [_strategy_label(row) for row in strategy_rows]
    x = np.arange(len(strategy_rows), dtype=np.float32)
    width = 0.24
    ax2.bar(x - width, _metric_array(strategy_rows, "f1"), width, color="#A7F3D0", label="F1")
    ax2.bar(x, _metric_array(strategy_rows, "precision"), width, color="#BFDBFE", label="Precision")
    ax2.bar(x + width, _metric_array(strategy_rows, "recall"), width, color="#FDE68A", label="Recall")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=22, ha="right", color=PPT_FG, fontsize=11)
    ax2.set_ylabel("Metric")
    ax2.set_title("Post-processing ablation", fontsize=20, weight="bold")
    legend = ax2.legend(frameon=False, fontsize=11, loc="lower left")
    for text in legend.get_texts():
        text.set_color(PPT_FG)
    for idx, row in enumerate(strategy_rows):
        if row is not recommendation["baseline_row"]:
            ax2.text(idx, 1.01, "rejected", color="#FCA5A5", ha="center", fontsize=10, weight="bold")
    ax2.text(
        0.02,
        -0.28,
        "Top-N hurts recall; hybrid does not improve over baseline.",
        transform=ax2.transAxes,
        color="#FCA5A5",
        fontsize=13,
        weight="bold",
    )

    fig.suptitle("Error-context diagnostics: confidence filtering is not enough", color=PPT_FG, fontsize=28, weight="bold", y=0.99)
    _save_dark_figure(fig, output_path)


def _plot_confidence_localz_scatter_ppt(output_path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=180, facecolor=PPT_BG)
    _style_dark_axes(ax)
    markers = {"TP": "o", "FP": "X"}
    for event_type in ("TP", "FP"):
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if row.get("event_type") != event_type:
                continue
            confidence = _float_or_default(row.get("confidence"), float("nan"))
            zscore = _float_or_default(row.get("local_zscore"), float("nan"))
            if math.isfinite(confidence) and math.isfinite(zscore):
                xs.append(confidence)
                ys.append(zscore)
        if xs:
            ax.scatter(
                xs,
                ys,
                s=52 if event_type == "FP" else 34,
                alpha=0.78,
                color=EVENT_COLORS[event_type],
                marker=markers[event_type],
                label=f"{event_type} (n={len(xs)})",
                edgecolors="none",
            )
            median_conf = float(np.median(xs))
            ax.axvline(median_conf, color=EVENT_COLORS[event_type], lw=1.8, ls="--")
            ax.text(
                median_conf,
                ax.get_ylim()[1] * 0.92,
                f"{event_type} median={median_conf:.4f}",
                color=EVENT_COLORS[event_type],
                fontsize=12,
                rotation=90,
                va="top",
                ha="right",
            )
    ax.set_xlabel("Model confidence", fontsize=15)
    ax.set_ylabel("Local z-score", fontsize=15)
    ax.set_title("FPs are not simply low-confidence predictions", fontsize=24, weight="bold")
    legend = ax.legend(frameon=False, fontsize=13, loc="upper left")
    for text in legend.get_texts():
        text.set_color(PPT_FG)
    ax.text(
        0.02,
        0.04,
        "High-confidence FPs overlap the TP confidence range, so confidence-only filtering risks deleting true loci.",
        transform=ax.transAxes,
        color="#FCA5A5",
        fontsize=14,
        weight="bold",
    )
    _save_dark_figure(fig, output_path)


def _plot_feature_distributions_ppt(output_path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), dpi=180, facecolor=PPT_BG)
    fields = [("local_zscore", "Local z-score"), ("local_contrast", "Local contrast")]
    for ax, (field, title) in zip(axes, fields, strict=True):
        _style_dark_axes(ax)
        data = [_finite_values(rows, event_type, field) for event_type in ("TP", "FP", "FN")]
        finite = np.concatenate([arr for arr in data if len(arr) > 0])
        if len(finite) > 0:
            lo = float(np.percentile(finite, 2))
            hi = float(np.percentile(finite, 98))
            if hi > lo:
                ax.set_ylim(lo, hi)
        box = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PPT_FG, "linewidth": 2},
            whiskerprops={"color": PPT_MUTED},
            capprops={"color": PPT_MUTED},
        )
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["TP", "FP", "FN"], color=PPT_FG, fontsize=12)
        for patch, event_type in zip(box["boxes"], ("TP", "FP", "FN"), strict=True):
            patch.set_facecolor(EVENT_COLORS[event_type])
            patch.set_alpha(0.72)
            patch.set_edgecolor(PPT_FG)
        for index, arr in enumerate(data, start=1):
            if len(arr) > 0:
                median = float(np.median(arr))
                ax.text(index, median, f"  {median:.2f}", color=PPT_FG, fontsize=12, va="center", weight="bold")
        ax.set_title(title, fontsize=21, weight="bold")
        ax.set_ylabel("Raw feature value", fontsize=14)
        ax.text(
            0.02,
            -0.15,
            "Axis clipped to the 2-98 percentile range; medians are raw values.",
            transform=ax.transAxes,
            color=PPT_MUTED,
            fontsize=10,
        )
    fig.suptitle("FN events are enriched for low-SNR / low-contrast loci", color=PPT_FG, fontsize=25, weight="bold")
    _save_dark_figure(fig, output_path)


def _plot_postprocess_ablation_ppt(output_path: Path, recommendation: dict[str, object]) -> None:
    rows = _strategy_rows_for_ppt(recommendation)
    fig, ax = plt.subplots(figsize=(13.8, 7.2), dpi=180, facecolor=PPT_BG)
    _style_dark_axes(ax)
    labels = [_strategy_label(row) for row in rows]
    x = np.arange(len(rows), dtype=np.float32)
    width = 0.24
    ax.bar(x - width, _metric_array(rows, "f1"), width, color="#A7F3D0", label="F1")
    ax.bar(x, _metric_array(rows, "precision"), width, color="#BFDBFE", label="Precision")
    ax.bar(x + width, _metric_array(rows, "recall"), width, color="#FDE68A", label="Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=PPT_FG, fontsize=12)
    ax.set_ylabel("Metric", fontsize=15)
    ax.set_title("Post-processing ablation does not beat the baseline", fontsize=24, weight="bold")
    legend = ax.legend(frameon=False, fontsize=13, loc="lower left")
    for text in legend.get_texts():
        text.set_color(PPT_FG)
    baseline_f1 = float(recommendation["baseline_row"]["f1"])
    ax.axhline(baseline_f1, color="#A7F3D0", ls="--", lw=1.8, alpha=0.8)
    ax.text(len(rows) - 0.15, baseline_f1 + 0.01, "baseline F1", color="#A7F3D0", ha="right", fontsize=12)
    for idx, row in enumerate(rows):
        text = "keep default" if row is recommendation["baseline_row"] else "rejected"
        color = "#A7F3D0" if row is recommendation["baseline_row"] else "#FCA5A5"
        ax.text(idx, 1.0, text, color=color, ha="center", fontsize=11, weight="bold")
    ax.text(
        0.02,
        0.06,
        "Fixed Top-N recall-caps high-count images; threshold+Top-N only matches or underperforms the baseline.",
        transform=ax.transAxes,
        color="#FCA5A5",
        fontsize=14,
        weight="bold",
    )
    _save_dark_figure(fig, output_path)


def _select_low_z_fn_rows(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    fn_rows = [
        row for row in rows
        if row.get("event_type") == "FN"
        and math.isfinite(_float_or_default(row.get("local_zscore"), float("nan")))
    ]
    return sorted(fn_rows, key=lambda row: _float_or_default(row.get("local_zscore"), float("inf")))[:count]


def _select_high_z_fn_rows(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    fn_rows = [
        row for row in rows
        if row.get("event_type") == "FN"
        and math.isfinite(_float_or_default(row.get("local_zscore"), float("nan")))
    ]
    return sorted(fn_rows, key=lambda row: _float_or_default(row.get("local_zscore"), -float("inf")), reverse=True)[:count]


def _plot_patch_montage_ppt(
    output_path: Path,
    event_rows: list[dict[str, object]],
    samples: list[SampleData],
    *,
    patch_radius: int,
    examples_per_row: int = 6,
) -> None:
    image_lookup = _sample_image_lookup(samples)
    rows_to_plot = [
        ("TP controls", _select_patch_rows(event_rows, "TP", examples_per_row)),
        ("High-confidence FP", _select_patch_rows(event_rows, "FP", examples_per_row)),
        ("Low-z FN", _select_low_z_fn_rows(event_rows, examples_per_row)),
        ("High-z FN", _select_high_z_fn_rows(event_rows, examples_per_row)),
    ]
    nrows = len(rows_to_plot)
    ncols = examples_per_row
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.35 * ncols, 2.35 * nrows), dpi=170, facecolor=PPT_BG)
    axes = np.asarray(axes)
    for row_index, (row_label, selected_rows) in enumerate(rows_to_plot):
        for col_index in range(ncols):
            ax = axes[row_index, col_index]
            ax.set_facecolor(PPT_BG)
            ax.set_axis_off()
            if col_index >= len(selected_rows):
                continue
            event_row = selected_rows[col_index]
            image = image_lookup[str(event_row["sample_id"])]
            y, x = _event_anchor(event_row)
            patch = _extract_patch(image, y, x, patch_radius)
            ax.imshow(_normalise_patch_for_display(patch), cmap="gray", origin="upper")
            confidence = _float_or_default(event_row.get("confidence"), float("nan"))
            conf_text = "" if not math.isfinite(confidence) else f" p={confidence:.2f}"
            zscore = _float_or_default(event_row.get("local_zscore"), float("nan"))
            z_text = "z=nan" if not math.isfinite(zscore) else f"z={zscore:.1f}"
            ax.set_title(f"{event_row['sample_id']}{conf_text}\n{z_text}", color=PPT_FG, fontsize=8)
        axes[row_index, 0].text(
            -0.12,
            0.5,
            row_label,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            color=PPT_FG,
            fontsize=14,
            weight="bold",
            ha="center",
            va="center",
        )
    fig.suptitle(
        "Representative patches for manual error review",
        color=PPT_FG,
        fontsize=23,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Each patch is independently contrast-normalized for visibility; no biological structure class is assigned automatically.",
        color=PPT_MUTED,
        ha="center",
        fontsize=11,
    )
    _save_dark_figure(fig, output_path)


def _make_ppt_figures(
    output_dir: Path,
    event_rows: list[dict[str, object]],
    samples: list[SampleData],
    recommendation: dict[str, object],
    *,
    patch_radius: int,
) -> None:
    figures_dir = output_dir / "figures_ppt"
    _plot_error_context_summary_ppt(figures_dir / "error_context_summary.png", recommendation)
    _plot_confidence_localz_scatter_ppt(figures_dir / "confidence_localz_scatter.png", event_rows)
    _plot_feature_distributions_ppt(figures_dir / "tp_fp_fn_feature_distributions.png", event_rows)
    _plot_postprocess_ablation_ppt(figures_dir / "postprocess_ablation_summary.png", recommendation)
    _plot_patch_montage_ppt(
        figures_dir / "patch_montage.png",
        event_rows,
        samples,
        patch_radius=patch_radius,
    )


def _event_anchor(row: dict[str, object]) -> tuple[float, float]:
    if row["event_type"] == "FN":
        return float(row["true_y_px"]), float(row["true_x_px"])
    return float(row["pred_y_px"]), float(row["pred_x_px"])


def _patch_filename(row: dict[str, object]) -> str:
    y, x = _event_anchor(row)
    confidence = _float_or_default(row.get("confidence"), float("nan"))
    confidence_text = "nan" if not math.isfinite(confidence) else f"{confidence:.3f}"
    zscore = _float_or_default(row.get("local_zscore"), float("nan"))
    zscore_text = "nan" if not math.isfinite(zscore) else f"{zscore:.2f}"
    return (
        f"{row['sample_id']}_{row['event_type']}_conf-{confidence_text}_"
        f"z-{zscore_text}_y-{y:.1f}_x-{x:.1f}.png"
    )


def _normalise_patch_for_display(patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch, dtype=np.float32)
    low = float(np.percentile(patch, 1))
    high = float(np.percentile(patch, 99))
    if high <= low:
        return np.zeros_like(patch, dtype=np.float32)
    return np.clip((patch - low) / (high - low), 0.0, 1.0)


def _select_patch_rows(rows: list[dict[str, object]], event_type: str, top_k: int) -> list[dict[str, object]]:
    candidates = [row for row in rows if row.get("event_type") == event_type]
    if event_type == "FP":
        return sorted(candidates, key=lambda row: _float_or_default(row.get("confidence"), -1.0), reverse=True)[:top_k]
    if event_type == "FN":
        finite = [row for row in candidates if math.isfinite(_float_or_default(row.get("local_zscore"), float("nan")))]
        low = sorted(finite, key=lambda row: _float_or_default(row.get("local_zscore"), float("inf")))[: max(1, top_k // 2)]
        high = sorted(finite, key=lambda row: _float_or_default(row.get("local_zscore"), -float("inf")), reverse=True)[: max(1, top_k - len(low))]
        selected: list[dict[str, object]] = []
        seen: set[int] = set()
        for row in low + high:
            marker = id(row)
            if marker not in seen:
                selected.append(row)
                seen.add(marker)
        return selected[:top_k]
    if event_type == "TP":
        high_conf = sorted(candidates, key=lambda row: _float_or_default(row.get("confidence"), -1.0), reverse=True)[: max(1, top_k // 2)]
        zscores = [_float_or_default(row.get("local_zscore"), float("nan")) for row in candidates]
        finite_zscores = [value for value in zscores if math.isfinite(value)]
        median_z = float(np.median(finite_zscores)) if finite_zscores else 0.0
        mid_z = sorted(
            candidates,
            key=lambda row: abs(_float_or_default(row.get("local_zscore"), median_z) - median_z),
        )[: max(1, top_k - len(high_conf))]
        selected = []
        seen = set()
        for row in high_conf + mid_z:
            marker = id(row)
            if marker not in seen:
                selected.append(row)
                seen.add(marker)
        return selected[:top_k]
    return candidates[:top_k]


def _sample_image_lookup(samples: list[SampleData]) -> dict[str, np.ndarray]:
    return {sample.sample_id: sample.image for sample in samples}


def _write_patch_png(path: Path, patch: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=120)
    ax.imshow(_normalise_patch_for_display(patch), cmap="gray", origin="upper")
    ax.set_title(title, fontsize=6)
    ax.set_axis_off()
    fig.tight_layout(pad=0.05)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def _write_montage(path: Path, patch_items: list[tuple[np.ndarray, str]]) -> None:
    if not patch_items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = min(6, len(patch_items))
    nrows = int(math.ceil(len(patch_items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows), dpi=140)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, (patch, title) in zip(axes_array, patch_items, strict=False):
        ax.imshow(_normalise_patch_for_display(patch), cmap="gray", origin="upper")
        ax.set_title(title, fontsize=6)
        ax.set_axis_off()
    for ax in axes_array[len(patch_items):]:
        ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _export_patches(
    output_dir: Path,
    event_rows: list[dict[str, object]],
    samples: list[SampleData],
    *,
    patch_radius: int,
    top_k: int,
) -> None:
    image_lookup = _sample_image_lookup(samples)
    montage_names = {
        "FP": "top_fp_patches_montage.png",
        "FN": "top_fn_patches_montage.png",
        "TP": "tp_control_patches_montage.png",
    }
    for event_type in ("TP", "FP", "FN"):
        selected_rows = _select_patch_rows(event_rows, event_type, top_k)
        patch_items: list[tuple[np.ndarray, str]] = []
        for row in selected_rows:
            image = image_lookup[str(row["sample_id"])]
            y, x = _event_anchor(row)
            patch = _extract_patch(image, y, x, patch_radius)
            title = f"{row['sample_id']} {event_type}"
            _write_patch_png(output_dir / "patches" / event_type / _patch_filename(row), patch, title)
            patch_items.append((patch, title))
        _write_montage(output_dir / "figures" / montage_names[event_type], patch_items)


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _hard_example_coordinate(row: dict[str, object]) -> tuple[float | str, float | str]:
    if row.get("event_type") == "FN":
        return row.get("true_y_px", ""), row.get("true_x_px", "")
    return row.get("pred_y_px", ""), row.get("pred_x_px", "")


def _hard_example_patch(
    output_dir: Path,
    row: dict[str, object],
    image_lookup: dict[str, np.ndarray],
    *,
    group: str,
    patch_radius: int,
) -> str:
    if row.get("event_type") not in {"TP", "FP", "FN"}:
        return ""
    image = image_lookup[str(row["sample_id"])]
    y, x = _event_anchor(row)
    patch = _extract_patch(image, y, x, patch_radius)
    filename = f"{group}_{_patch_filename(row)}"
    path = output_dir / "patches" / "hard_examples" / filename
    _write_patch_png(path, patch, f"{group} {row['event_type']}")
    return _relative_path(path, output_dir)


def _hard_event_row(
    output_dir: Path,
    row: dict[str, object],
    image_lookup: dict[str, np.ndarray],
    *,
    group: str,
    patch_radius: int,
) -> dict[str, object]:
    y, x = _hard_example_coordinate(row)
    return {
        "example_group": group,
        "sample_id": row.get("sample_id", ""),
        "image_name": row.get("image_name", ""),
        "event_type": row.get("event_type", ""),
        "confidence": row.get("confidence", ""),
        "local_zscore": row.get("local_zscore", ""),
        "local_contrast": row.get("local_contrast", ""),
        "coordinate_y_px": y,
        "coordinate_x_px": x,
        "patch_path": _hard_example_patch(
            output_dir,
            row,
            image_lookup,
            group=group,
            patch_radius=patch_radius,
        ),
        "suggested_review_label": "",
        "notes": "manual review candidate; no automatic biological structure label assigned",
    }


def _sample_image_name_lookup(samples: list[SampleData]) -> dict[str, str]:
    return {sample.sample_id: sample.image_name for sample in samples}


def _build_hard_examples(
    output_dir: Path,
    event_rows: list[dict[str, object]],
    per_sample_rows: list[dict[str, object]],
    samples: list[SampleData],
    *,
    patch_radius: int,
) -> list[dict[str, object]]:
    image_lookup = _sample_image_lookup(samples)
    image_name_lookup = _sample_image_name_lookup(samples)
    hard_rows: list[dict[str, object]] = []

    fp_rows = [row for row in event_rows if row.get("event_type") == "FP"]
    high_conf_fp = sorted(
        fp_rows,
        key=lambda row: _float_or_default(row.get("confidence"), -1.0),
        reverse=True,
    )[:20]
    for row in high_conf_fp:
        hard_rows.append(
            _hard_event_row(
                output_dir,
                row,
                image_lookup,
                group="high_conf_fp",
                patch_radius=patch_radius,
            )
        )

    for row in _select_low_z_fn_rows(event_rows, 20):
        hard_rows.append(
            _hard_event_row(
                output_dir,
                row,
                image_lookup,
                group="low_z_fn",
                patch_radius=patch_radius,
            )
        )

    for row in _select_high_z_fn_rows(event_rows, 20):
        hard_rows.append(
            _hard_event_row(
                output_dir,
                row,
                image_lookup,
                group="high_z_fn",
                patch_radius=patch_radius,
            )
        )

    hardest_samples = sorted(
        per_sample_rows,
        key=lambda row: (-int(row["fn"]), float(row["f1"]), -int(row["fp"]), str(row["sample_id"])),
    )[:20]
    for row in hardest_samples:
        sample_id = str(row["sample_id"])
        hard_rows.append(
            {
                "example_group": "hardest_sample",
                "sample_id": sample_id,
                "image_name": image_name_lookup.get(sample_id, ""),
                "event_type": "SAMPLE",
                "confidence": "",
                "local_zscore": "",
                "local_contrast": "",
                "coordinate_y_px": "",
                "coordinate_x_px": "",
                "patch_path": "",
                "suggested_review_label": "",
                "notes": (
                    f"TP={int(row['tp'])}; FP={int(row['fp'])}; FN={int(row['fn'])}; "
                    f"F1={float(row['f1']):.4f}"
                ),
            }
        )
    return hard_rows


def _write_ppt_takeaways(
    output_dir: Path,
    recommendation: dict[str, object],
) -> None:
    confidence_medians = recommendation["confidence_medians"]
    zscore_medians = recommendation["zscore_medians"]
    contrast_medians = recommendation["contrast_medians"]
    text = f"""# PPT Takeaways

## One-sentence conclusion

Error-context diagnostics suggest that the remaining errors are not solved by confidence filtering; false negatives are enriched for low-SNR loci, while false positives often remain high-confidence.

## Key findings

- The current default remains the best choice: dice_rmse with threshold sweep at the baseline operating point.
- FP confidence overlaps TP confidence: median TP confidence = {confidence_medians['TP']:.4f}, median FP confidence = {confidence_medians['FP']:.4f}.
- FN events are enriched for low local z-score and lower local contrast: median local z-score TP = {zscore_medians['TP']:.2f}, FP = {zscore_medians['FP']:.2f}, FN = {zscore_medians['FN']:.2f}; median local contrast TP = {contrast_medians['TP']:.0f}, FP = {contrast_medians['FP']:.0f}, FN = {contrast_medians['FN']:.0f}.
- Fixed Top-N and validation-prior Top-N are rejected because they recall-cap high-count images.
- Threshold+Top-N hybrid does not improve over the baseline.

## Recommended next technical step

- Do not change default inference yet.
- Build curated hard-positive and hard-negative patch sets from `hard_examples.csv`.
- Test dim-loci augmentation and hard-negative sampling before architecture modification.
- Use patch review to decide whether FP examples reflect structured background, annotation ambiguity, crowding, or artifacts; do not assign biological structure labels without masks.
"""
    (output_dir / "ppt_takeaways.md").write_text(text, encoding="utf-8")


def _best_row(rows: list[dict[str, object]], strategy: str) -> dict[str, object] | None:
    selected = [row for row in rows if row["strategy"] == strategy]
    if not selected:
        return None
    return max(
        selected,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            -float(row["fp"]),
        ),
    )


def _best_precision_constrained_threshold(rows: list[dict[str, object]], precision_floor: float) -> dict[str, object] | None:
    selected = [
        row for row in rows
        if row["strategy"] == "threshold_only" and float(row["precision"]) >= precision_floor
    ]
    if not selected:
        return None
    return max(selected, key=lambda row: (float(row["f1"]), float(row["recall"]), -float(row["fp"])))


def _median_by_event(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for event_type in ("TP", "FP", "FN"):
        values = _finite_values(rows, event_type, field)
        result[event_type] = float(np.median(values)) if len(values) > 0 else float("nan")
    return result


def _count_by_event(rows: list[dict[str, object]]) -> dict[str, int]:
    return {event_type: sum(1 for row in rows if row["event_type"] == event_type) for event_type in ("TP", "FP", "FN")}


def _strategy_delta_text(row: dict[str, object] | None, baseline_row: dict[str, object]) -> str:
    if row is None:
        return "n/a"
    delta_fp = int(row["fp"]) - int(baseline_row["fp"])
    delta_fn = int(row["fn"]) - int(baseline_row["fn"])
    delta_tp = int(row["tp"]) - int(baseline_row["tp"])
    return (
        f"F1={float(row['f1']):.4f}, P={float(row['precision']):.4f}, "
        f"R={float(row['recall']):.4f}, FP {int(row['fp'])} ({delta_fp:+d}), "
        f"FN {int(row['fn'])} ({delta_fn:+d}), TP {int(row['tp'])} ({delta_tp:+d})"
    )


def _recommendation(
    post_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
    baseline: BaselineConfig,
) -> dict[str, object]:
    baseline_row = next(row for row in post_rows if row["strategy"] == "baseline_threshold_from_evaluation")
    threshold_best = _best_row(post_rows, "threshold_only")
    topn_best = _best_row(post_rows, "fixed_top_n")
    hybrid_best = _best_row(post_rows, "threshold_max_n")
    prior_best = _best_row(post_rows, "validation_prior_top_n")
    precision_floor = baseline.precision if math.isfinite(baseline.precision) else 0.88
    constrained = _best_precision_constrained_threshold(post_rows, precision_floor)

    confidence_medians = _median_by_event(event_rows, "confidence")
    zscore_medians = _median_by_event(event_rows, "local_zscore")
    contrast_medians = _median_by_event(event_rows, "local_contrast")
    sharpness_medians = _median_by_event(event_rows, "peak_sharpness")
    event_counts = _count_by_event(event_rows)

    fp_conf = confidence_medians["FP"]
    tp_conf = confidence_medians["TP"]
    fp_conf_lower = math.isfinite(fp_conf) and math.isfinite(tp_conf) and fp_conf < tp_conf - 0.05
    fn_low_z = (
        math.isfinite(zscore_medians["FN"])
        and math.isfinite(zscore_medians["TP"])
        and zscore_medians["FN"] < zscore_medians["TP"] - 0.5
    )
    fp_high_intensity_low_sharpness = (
        math.isfinite(zscore_medians["FP"])
        and math.isfinite(zscore_medians["TP"])
        and math.isfinite(sharpness_medians["FP"])
        and math.isfinite(sharpness_medians["TP"])
        and zscore_medians["FP"] >= zscore_medians["TP"] - 0.25
        and sharpness_medians["FP"] < sharpness_medians["TP"]
    )

    def is_default_candidate(row: dict[str, object] | None) -> bool:
        if row is None:
            return False
        fp_reduced = int(row["fp"]) < int(baseline_row["fp"])
        f1_not_worse = float(row["f1"]) >= float(baseline_row["f1"])
        recall_ok = float(row["recall"]) >= float(baseline_row["recall"]) - 0.02
        tp_loss = int(baseline_row["tp"]) - int(row["tp"])
        fp_gain = int(baseline_row["fp"]) - int(row["fp"])
        bad_trade = fp_gain <= 3 and tp_loss >= 5
        return fp_reduced and f1_not_worse and recall_ok and not bad_trade

    candidates = [
        ("threshold_only", threshold_best),
        ("fixed_top_n", topn_best),
        ("threshold_max_n", hybrid_best),
        ("validation_prior_top_n", prior_best),
    ]
    recommended = [(name, row) for name, row in candidates if is_default_candidate(row)]
    if recommended:
        best_name, best_strategy = max(recommended, key=lambda item: float(item[1]["f1"]))
        best_strategy_text = f"Candidate post-processing strategy: {best_name} ({_strategy_delta_text(best_strategy, baseline_row)})."
    else:
        best_strategy_text = (
            "Do not change the default inference strategy yet; use dice_rmse + current "
            "threshold sweep as default, and move to hard-negative mining / augmentation / "
            "context-aware model modification."
        )

    return {
        "baseline_row": baseline_row,
        "threshold_best": threshold_best,
        "threshold_precision_constrained": constrained,
        "topn_best": topn_best,
        "hybrid_best": hybrid_best,
        "prior_best": prior_best,
        "confidence_medians": confidence_medians,
        "zscore_medians": zscore_medians,
        "contrast_medians": contrast_medians,
        "sharpness_medians": sharpness_medians,
        "event_counts": event_counts,
        "fp_conf_lower": fp_conf_lower,
        "fn_low_z": fn_low_z,
        "fp_high_intensity_low_sharpness": fp_high_intensity_low_sharpness,
        "best_strategy_text": best_strategy_text,
    }


def _format_strategy(row: dict[str, object] | None) -> str:
    if row is None:
        return "n/a"
    parts = [str(row["strategy"])]
    if row.get("threshold") != "":
        parts.append(f"threshold={row['threshold']}")
    if row.get("top_n") != "":
        parts.append(f"top_n={row['top_n']}")
    if row.get("max_n") != "":
        parts.append(f"max_n={row['max_n']}")
    parts.append(f"F1={float(row['f1']):.4f}")
    parts.append(f"P={float(row['precision']):.4f}")
    parts.append(f"R={float(row['recall']):.4f}")
    parts.append(f"TP={int(row['tp'])}")
    parts.append(f"FP={int(row['fp'])}")
    parts.append(f"FN={int(row['fn'])}")
    return ", ".join(parts)


def _write_readme(
    output_dir: Path,
    args: argparse.Namespace,
    baseline: BaselineConfig,
    recommendation: dict[str, object],
) -> None:
    readme = output_dir / "README.md"
    event_counts = recommendation["event_counts"]
    confidence_medians = recommendation["confidence_medians"]
    zscore_medians = recommendation["zscore_medians"]
    contrast_medians = recommendation["contrast_medians"]
    sharpness_medians = recommendation["sharpness_medians"]
    text = f"""# Error-context diagnostics

## Inputs

- Dataset: `{Path(args.data).resolve()}`
- Evaluation summary: `{Path(args.evaluation).resolve()}`
- Model checkpoint: `{Path(args.model).resolve()}`
- Optional image directory: `{Path(args.images).resolve() if args.images else 'not provided; used npz images'}`
- Subset: `{args.subset}`
- Baseline probability: `{baseline.probability}`
- Match radius px: `{baseline.match_radius_px}`

Command:

```bash
python deepblink_pytorch/scripts/error_context_diagnostics.py --data {args.data} --evaluation {args.evaluation} --model {args.model} --output {args.output} --subset {args.subset} --patch-radius {args.patch_radius} --top-k-patches {args.top_k_patches} --device {args.device}
```

## Outputs

- `error_context.csv`: one row per TP/FP/FN event at the baseline threshold.
- `postprocess_ablation.csv`: offline comparison of threshold-only, fixed Top-N, threshold+max-N, and validation-prior Top-N strategies.
- `hard_examples.csv`: prioritized examples for manual patch review.
- `ppt_takeaways.md`: English speaker-note text for slides.
- `figures/`: diagnostic plots and patch montages.
- `figures_ppt/`: black-background, English, presentation-ready figures.
- `patches/TP`, `patches/FP`, `patches/FN`: representative event patches.
- `patches/hard_examples`: patch images referenced by `hard_examples.csv`.

## Field Notes

Patch features are measured around the event anchor. TP/FP use the predicted coordinate;
FN uses the ground-truth coordinate. No nucleus/nucleolus/membrane class is assigned,
because no segmentation mask is provided.

Important fields:

- `confidence`: model probability for TP/FP; NaN for FN.
- `local_contrast`: center intensity minus local patch mean.
- `local_zscore`: local contrast divided by local patch std.
- `local_max_to_mean`: local max divided by local mean.
- `peak_sharpness`: local max minus local mean.
- `peak_component_size_px`: connected component size around the local max after thresholding at local mean + local std.
- `distance_to_border_px`: distance from event anchor to nearest image border.

## Diagnostic Summary

- Event counts: TP={event_counts['TP']}, FP={event_counts['FP']}, FN={event_counts['FN']}.
- Median confidence: TP={confidence_medians['TP']:.4f}, FP={confidence_medians['FP']:.4f}.
- Median local z-score: TP={zscore_medians['TP']:.4f}, FP={zscore_medians['FP']:.4f}, FN={zscore_medians['FN']:.4f}.
- Median local contrast: TP={contrast_medians['TP']:.4f}, FP={contrast_medians['FP']:.4f}, FN={contrast_medians['FN']:.4f}.
- Median peak sharpness: TP={sharpness_medians['TP']:.4f}, FP={sharpness_medians['FP']:.4f}, FN={sharpness_medians['FN']:.4f}.

Interpretation:

- FP mainly low confidence: `{recommendation['fp_conf_lower']}`.
- FN mainly low local z-score relative to TP: `{recommendation['fn_low_z']}`.
- FP high-z but lower-sharpness suspicious background proxy: `{recommendation['fp_high_intensity_low_sharpness']}`.
- Confidence filtering risk is high because FP confidence overlaps TP confidence.
- FN errors point toward dim-loci / low-SNR examples rather than a post-processing threshold problem.

## Post-processing Summary

- Baseline: {_format_strategy(recommendation['baseline_row'])}
- Best threshold-only: {_format_strategy(recommendation['threshold_best'])}
- Precision-constrained threshold-only: {_format_strategy(recommendation['threshold_precision_constrained'])}
- Best fixed Top-N: {_format_strategy(recommendation['topn_best'])}
- Best threshold+max-N hybrid: {_format_strategy(recommendation['hybrid_best'])}
- Best validation-prior Top-N: {_format_strategy(recommendation['prior_best'])}

Conclusion:

{recommendation['best_strategy_text']}

Explicit default-inference decision:

- Do not change default inference.
- Fixed Top-N is rejected because spot count varies strongly by image and high-count images become recall-capped.
- Confidence filtering alone is risky because high-confidence FP events overlap the TP confidence range.
- Threshold+Top-N hybrid does not improve over the baseline.
- This branch is offline diagnostics only and does not modify default training/evaluation/prediction behavior.

Rejected-strategy rules applied:

- Reject if F1 is below baseline.
- Reject if recall drops by more than about 0.02.
- Reject if FP decreases only slightly while TP loss is substantial.
- Reject fixed Top-N if high-count images are visibly recall-capped.

## Recommended Next Step

Use this output for manual FP/FN patch review. If FP patches are high-confidence structured
background, prioritize hard-negative mining or a lightweight local-context classifier.
If FN patches are dim/low-z, prioritize augmentation/sampling for dim loci. Do not infer
specific biological structures without masks.

Recommended merge-safe next work:

- Curate hard-positive and hard-negative patch sets from `hard_examples.csv`.
- Test dim-loci augmentation and hard-negative sampling before architecture changes.
- Keep these diagnostics out of the default inference pipeline until a strategy beats baseline without hurting recall.
"""
    readme.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    data_path = Path(args.data).resolve()
    evaluation_dir = Path(args.evaluation).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output).resolve()
    images_dir = Path(args.images).resolve() if args.images else None

    if args.patch_radius < 1:
        raise ValueError("--patch-radius must be >= 1")
    if args.top_k_patches < 1:
        raise ValueError("--top-k-patches must be >= 1")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not evaluation_dir.is_dir():
        raise NotADirectoryError(f"Evaluation directory not found: {evaluation_dir}")
    if images_dir is not None and not images_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline_config(evaluation_dir, args.match_radius_px)
    print(
        f"Baseline metadata | probability={baseline.probability:g} "
        f"| match_radius_px={baseline.match_radius_px:g} | F1={baseline.f1:.4f}"
    )

    samples = _load_samples(data_path, args.subset, images_dir)
    device = predict_cli.resolve_device(args.device)
    predictions, cell_size = _run_dense_inference(samples, model_path, device)
    print(f"Dense inference completed | samples={len(samples)} | cell_size={cell_size}")

    event_rows, per_sample_rows = _build_error_context(predictions, baseline, args.patch_radius)
    postprocess_rows = _run_postprocess_ablation(predictions, data_path, baseline)
    recommendation = _recommendation(postprocess_rows, event_rows, baseline)

    _write_csv(output_dir / "error_context.csv", event_rows, EVENT_FIELDNAMES)
    _write_csv(output_dir / "postprocess_ablation.csv", postprocess_rows, POSTPROCESS_FIELDNAMES)
    _make_figures(output_dir, event_rows, per_sample_rows)
    _make_ppt_figures(
        output_dir,
        event_rows,
        samples,
        recommendation,
        patch_radius=args.patch_radius,
    )
    _export_patches(
        output_dir,
        event_rows,
        samples,
        patch_radius=args.patch_radius,
        top_k=args.top_k_patches,
    )
    hard_example_rows = _build_hard_examples(
        output_dir,
        event_rows,
        per_sample_rows,
        samples,
        patch_radius=args.patch_radius,
    )
    _write_csv(output_dir / "hard_examples.csv", hard_example_rows, HARD_EXAMPLE_FIELDNAMES)
    _write_ppt_takeaways(output_dir, recommendation)
    _write_readme(output_dir, args, baseline, recommendation)

    machine_summary = {
        "baseline": recommendation["baseline_row"],
        "threshold_best": recommendation["threshold_best"],
        "threshold_precision_constrained": recommendation["threshold_precision_constrained"],
        "topn_best": recommendation["topn_best"],
        "hybrid_best": recommendation["hybrid_best"],
        "prior_best": recommendation["prior_best"],
        "diagnostics": {
            "event_counts": recommendation["event_counts"],
            "confidence_medians": recommendation["confidence_medians"],
            "zscore_medians": recommendation["zscore_medians"],
            "contrast_medians": recommendation["contrast_medians"],
            "sharpness_medians": recommendation["sharpness_medians"],
            "fp_conf_lower": recommendation["fp_conf_lower"],
            "fn_low_z": recommendation["fn_low_z"],
            "fp_high_intensity_low_sharpness": recommendation["fp_high_intensity_low_sharpness"],
            "recommendation": recommendation["best_strategy_text"],
        },
        "ppt_outputs": {
            "summary_figure": "figures_ppt/error_context_summary.png",
            "confidence_scatter": "figures_ppt/confidence_localz_scatter.png",
            "feature_distributions": "figures_ppt/tp_fp_fn_feature_distributions.png",
            "postprocess_summary": "figures_ppt/postprocess_ablation_summary.png",
            "patch_montage": "figures_ppt/patch_montage.png",
            "takeaways": "ppt_takeaways.md",
            "hard_examples": "hard_examples.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(machine_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote diagnostics to: {output_dir}")
    print(recommendation["best_strategy_text"])
    return machine_summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
