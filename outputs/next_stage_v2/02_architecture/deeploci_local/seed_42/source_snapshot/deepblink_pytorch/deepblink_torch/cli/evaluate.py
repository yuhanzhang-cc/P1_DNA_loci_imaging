"""Test-set evaluation command for deepblink_torch CLI."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..evaluation import MatchResult
from ..evaluation import aggregate_metric_rows
from ..evaluation import f1_curve
from ..evaluation import match_coordinates
from ..evaluation import metrics_from_match
from . import overlay as overlay_cli
from . import predict as predict_cli


SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = (".tif", ".tiff")
DEFAULT_MATCH_RADIUS_PX = 3.0
DEFAULT_N_CUTOFFS = 50
DEFAULT_PROBABILITY = 0.5


@dataclass(frozen=True)
class EvalSample:
    """One evaluation sample with split metadata and ground truth coordinates."""

    sample_id: str
    image_name: str
    label_name: str
    source_index: int
    truth_coords_px: np.ndarray


@dataclass(frozen=True)
class EvalPrediction:
    """One sample after model inference, before threshold-specific decoding."""

    sample: EvalSample
    image_path: Path
    image: np.ndarray
    pred: np.ndarray
    orig_h: int
    orig_w: int
    truth_coords_px: np.ndarray


def add_test_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add evaluation arguments to an argparse parser."""
    parser.add_argument(
        "--data",
        required=True,
        help="Path to deepBlink .npz dataset with split metadata.",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing the raw images referenced by the dataset archive.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=SPLITS,
        help="Dataset subset to evaluate.",
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=None,
        help=(
            "Probability threshold for coordinate decoding. When omitted and "
            "--probability-sweep is provided, the best sweep threshold is used "
            f"for predictions and overlays; otherwise defaults to {DEFAULT_PROBABILITY}."
        ),
    )
    parser.add_argument(
        "--probability-sweep",
        default=None,
        help=(
            "Optional comma-separated probability thresholds to summarize in "
            "summary/threshold_metrics.csv without re-running model inference."
        ),
    )
    parser.add_argument(
        "--match-radius-px",
        type=float,
        default=DEFAULT_MATCH_RADIUS_PX,
        help="Matching radius in pixels used for precision, recall, and F1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "xpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--overlay-mode",
        type=str,
        default="side-by-side",
        choices=overlay_cli.MODES,
        help="Overlay rendering mode for per-image PNGs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Generate extra diagnostic plots: the legacy F1 curve, error CDF, "
            "and localization-error-vs-confidence scatter plot."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a model on a deepBlink split.")
    add_test_arguments(parser)
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    """Resolve runtime device, falling back to CPU if CUDA is unavailable."""
    if requested == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("Intel XPU requested but unavailable in this environment.")
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda but CUDA is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def _load_npz_archive(data_path: Path) -> dict[str, np.ndarray]:
    with np.load(data_path, allow_pickle=True) as data:
        return {name: data[name] for name in data.files}


def _subset_key(subset: str, field: str) -> str:
    return f"{field}_{subset}"


def _require_split_metadata(data: dict[str, np.ndarray], subset: str) -> None:
    required = [
        f"x_{subset}",
        f"y_{subset}",
        f"sample_id_{subset}",
        f"image_name_{subset}",
        f"label_name_{subset}",
        f"source_index_{subset}",
    ]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(
            "Dataset archive is missing split metadata needed for evaluation: "
            f"{missing}. Re-run create to generate a new .npz."
        )


def _load_eval_samples(data_path: Path, subset: str) -> list[EvalSample]:
    data = _load_npz_archive(data_path)
    _require_split_metadata(data, subset)

    sample_ids = data[_subset_key(subset, "sample_id")]
    image_names = data[_subset_key(subset, "image_name")]
    label_names = data[_subset_key(subset, "label_name")]
    source_indices = data[_subset_key(subset, "source_index")]
    truth_coords = data[_subset_key(subset, "y")]

    if not (
        len(sample_ids)
        == len(image_names)
        == len(label_names)
        == len(source_indices)
        == len(truth_coords)
    ):
        raise ValueError(f"Split metadata arrays for subset={subset} do not have matching lengths.")

    samples: list[EvalSample] = []
    for index in range(len(sample_ids)):
        samples.append(
            EvalSample(
                sample_id=str(sample_ids[index]),
                image_name=str(image_names[index]),
                label_name=str(label_names[index]),
                source_index=int(source_indices[index]),
                truth_coords_px=np.asarray(truth_coords[index], dtype=np.float32),
            )
        )
    return samples


def _find_image_file(images_dir: Path, image_name: str, sample_id: str) -> Path:
    candidate = images_dir / image_name
    if candidate.is_file():
        return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{sample_id}{ext}"
        if candidate.is_file():
            return candidate

    glob_matches = [
        path
        for path in sorted(images_dir.glob(f"{sample_id}.*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(glob_matches) == 1:
        return glob_matches[0]

    raise FileNotFoundError(
        f"Could not locate image for sample_id={sample_id!r}, image_name={image_name!r} in {images_dir}."
    )


def _as_relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _parse_probability_sweep(raw_value: str | None) -> list[float]:
    if raw_value is None:
        return []

    tokens = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    if not tokens:
        return []

    thresholds: list[float] = []
    for token in tokens:
        try:
            threshold = float(token)
        except ValueError as exc:
            raise ValueError(
                f"--probability-sweep values must be floats in [0, 1], got {token!r}."
            ) from exc
        if not (0 <= threshold <= 1):
            raise ValueError(
                f"--probability-sweep values must be in [0, 1], got {threshold}."
            )
        if threshold not in thresholds:
            thresholds.append(threshold)
    return thresholds


def _thresholds_with_reference(
    thresholds: list[float],
    reference_probability: float = DEFAULT_PROBABILITY,
    extra_thresholds: list[float] | None = None,
) -> list[float]:
    if not thresholds:
        return []
    unique = {float(threshold) for threshold in thresholds}
    unique.add(float(reference_probability))
    for threshold in extra_thresholds or []:
        unique.add(float(threshold))
    return sorted(unique)


def _f1_curve_limit(match_radius_px: float) -> float:
    return float(match_radius_px) if match_radius_px > 0 else 0.1


def _decode_prediction_with_probability(
    pred: np.ndarray,
    *,
    cell_size: int,
    probability: float,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    coords = predict_cli.get_coordinate_list(
        pred,
        cell_size=cell_size,
        probability=probability,
    )
    if len(coords) > 0:
        in_bounds = (coords[:, 0] < orig_h) & (coords[:, 1] < orig_w)
        coords = coords[in_bounds]
    return predict_cli._coords_with_probabilities(pred, coords, cell_size=cell_size)


def _match_overlay_coords(
    pred_coords_with_p: np.ndarray,
    true_coords_px: np.ndarray,
    match: MatchResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_coords = np.asarray(pred_coords_with_p, dtype=np.float32)
    true_coords = np.asarray(true_coords_px, dtype=np.float32)

    tp_coords = (
        pred_coords[match.matched_pred_indices, :2]
        if len(match.matched_pred_indices) > 0
        else np.empty((0, 2), dtype=np.float32)
    )
    fp_coords = (
        pred_coords[match.unmatched_pred_indices, :2]
        if len(match.unmatched_pred_indices) > 0
        else np.empty((0, 2), dtype=np.float32)
    )
    fn_coords = (
        true_coords[match.unmatched_true_indices]
        if len(match.unmatched_true_indices) > 0
        else np.empty((0, 2), dtype=np.float32)
    )
    return tp_coords, fp_coords, fn_coords


def _build_match_rows(
    pred_coords_with_p: np.ndarray,
    true_coords_px: np.ndarray,
    match: MatchResult,
) -> list[dict[str, object]]:
    pred_coords_with_p = np.asarray(pred_coords_with_p, dtype=np.float32)
    true_coords_px = np.asarray(true_coords_px, dtype=np.float32)

    rows: list[dict[str, object]] = []
    for pred_index, true_index, distance in zip(
        match.matched_pred_indices,
        match.matched_true_indices,
        match.matched_distances,
        strict=True,
    ):
        rows.append(
            {
                "match_status": "tp",
                "pred_index": int(pred_index),
                "true_index": int(true_index),
                "pred_y_px": float(pred_coords_with_p[pred_index, 0]),
                "pred_x_px": float(pred_coords_with_p[pred_index, 1]),
                "pred_p": float(pred_coords_with_p[pred_index, 2]),
                "true_y_px": float(true_coords_px[true_index, 0]),
                "true_x_px": float(true_coords_px[true_index, 1]),
                "distance_px": float(distance),
            }
        )

    for pred_index in match.unmatched_pred_indices:
        rows.append(
            {
                "match_status": "fp",
                "pred_index": int(pred_index),
                "true_index": "",
                "pred_y_px": float(pred_coords_with_p[pred_index, 0]),
                "pred_x_px": float(pred_coords_with_p[pred_index, 1]),
                "pred_p": float(pred_coords_with_p[pred_index, 2]),
                "true_y_px": "",
                "true_x_px": "",
                "distance_px": "",
            }
        )

    for true_index in match.unmatched_true_indices:
        rows.append(
            {
                "match_status": "fn",
                "pred_index": "",
                "true_index": int(true_index),
                "pred_y_px": "",
                "pred_x_px": "",
                "pred_p": "",
                "true_y_px": float(true_coords_px[true_index, 0]),
                "true_x_px": float(true_coords_px[true_index, 1]),
                "distance_px": "",
            }
        )

    return rows


def _write_rows_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_precision_recall_curve(
    output_path: Path,
    recall: np.ndarray,
    precision: np.ndarray,
    average_precision: float,
) -> None:
    overlay_cli._require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = overlay_cli.plt.subplots(figsize=(6, 4), dpi=150)
    if len(recall) > 0:
        ax.step(recall, precision, where="post", color="tab:blue", lw=2, label="PR curve")
        marker_step = max(1, len(recall) // 20)
        ax.scatter(
            recall[::marker_step],
            precision[::marker_step],
            s=14,
            color="tab:blue",
            alpha=0.35,
            edgecolors="none",
        )
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.grid(True, alpha=0.25)
        ap_text = f"AP = {average_precision:.4f}" if np.isfinite(average_precision) else "AP = n/a"
        ax.text(
            0.98,
            0.98,
            ap_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    else:
        ax.text(0.5, 0.5, "No evaluation predictions", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    overlay_cli.plt.close(fig)


def _plot_sorted_per_image_f1(output_path: Path, per_image_rows: list[dict[str, object]]) -> None:
    overlay_cli._require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = overlay_cli.plt.subplots(figsize=(12, 4.5), dpi=150)
    if per_image_rows:
        sorted_rows = sorted(per_image_rows, key=lambda row: (float(row["f1"]), str(row["sample_id"])))
        x = np.arange(len(sorted_rows), dtype=np.int32)
        f1_values = np.asarray([float(row["f1"]) for row in sorted_rows], dtype=np.float32)
        ax.bar(x, f1_values, color="tab:purple", alpha=0.85, width=0.82)
        ax.plot(x, f1_values, color="tab:purple", lw=1.2, alpha=0.9)
        ax.axhline(float(np.mean(f1_values)), color="tab:gray", ls="--", lw=1, label="mean F1")
        ax.axhline(float(np.median(f1_values)), color="tab:orange", ls=":", lw=1.2, label="median F1")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Test images sorted by F1")
        ax.set_ylabel("F1")
        ax.set_title("Per-image F1 scores")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper left")

        label_step = 1 if len(sorted_rows) <= 24 else max(1, len(sorted_rows) // 12)
        tick_positions = x[::label_step]
        tick_labels = [str(sorted_rows[index]["sample_id"]) for index in range(0, len(sorted_rows), label_step)]
        if tick_positions[-1] != x[-1]:
            tick_positions = np.append(tick_positions, x[-1])
            tick_labels.append(str(sorted_rows[-1]["sample_id"]))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=75, ha="right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No evaluation samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    overlay_cli.plt.close(fig)


def _plot_error_vs_confidence(
    output_path: Path,
    matched_confidences: np.ndarray,
    matched_distances: np.ndarray,
) -> None:
    overlay_cli._require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = overlay_cli.plt.subplots(figsize=(6, 4), dpi=150)
    if len(matched_confidences) > 0:
        ax.scatter(
            matched_confidences,
            matched_distances,
            s=18,
            alpha=0.6,
            color="tab:green",
            edgecolors="none",
        )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Localization error (px)")
        ax.set_title("Localization Error vs Confidence")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.0, 1.05)
    else:
        ax.text(0.5, 0.5, "No matched detections", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    overlay_cli.plt.close(fig)


def _compute_pr_curve(
    sample_predictions: list[tuple[np.ndarray, np.ndarray]],
    match_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    total_true = int(sum(len(truth_coords_px) for _, truth_coords_px in sample_predictions))
    if total_true == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            float("nan"),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    matched_truth_flags = [np.zeros(len(truth_coords_px), dtype=bool) for _, truth_coords_px in sample_predictions]
    flat_predictions: list[tuple[float, int, int]] = []
    for sample_index, (pred_coords_with_p, _truth_coords_px) in enumerate(sample_predictions):
        for pred_index, row in enumerate(np.asarray(pred_coords_with_p, dtype=np.float32)):
            flat_predictions.append((float(row[2]), sample_index, pred_index))

    if not flat_predictions:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            0.0,
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    flat_predictions.sort(key=lambda item: item[0], reverse=True)

    true_positive = 0
    false_positive = 0
    precision_points: list[float] = []
    recall_points: list[float] = []
    matched_confidences: list[float] = []
    matched_distances: list[float] = []

    for score, sample_index, pred_index in flat_predictions:
        pred_coords_with_p, truth_coords_px = sample_predictions[sample_index]
        truth_coords = np.asarray(truth_coords_px, dtype=np.float32)
        if len(truth_coords) == 0:
            false_positive += 1
        else:
            used_mask = matched_truth_flags[sample_index]
            available_indices = np.where(~used_mask)[0]
            if len(available_indices) == 0:
                false_positive += 1
            else:
                pred_xy = np.asarray(pred_coords_with_p[pred_index, :2], dtype=np.float32)
                distances = np.linalg.norm(truth_coords[available_indices] - pred_xy[None, :], axis=1)
                best_available = int(np.argmin(distances))
                best_distance = float(distances[best_available])
                if best_distance <= match_radius_px:
                    best_true_index = int(available_indices[best_available])
                    used_mask[best_true_index] = True
                    true_positive += 1
                    matched_confidences.append(float(score))
                    matched_distances.append(best_distance)
                else:
                    false_positive += 1

        precision_points.append(true_positive / max(1, true_positive + false_positive))
        recall_points.append(true_positive / total_true)

    recall_curve = np.concatenate(([0.0], np.asarray(recall_points, dtype=np.float32)))
    precision_curve = np.concatenate(([1.0], np.asarray(precision_points, dtype=np.float32)))
    precision_envelope = np.maximum.accumulate(precision_curve[::-1])[::-1]
    average_precision = float(np.sum(np.diff(recall_curve) * precision_envelope[1:]))
    return (
        np.asarray(recall_points, dtype=np.float32),
        np.asarray(precision_points, dtype=np.float32),
        average_precision,
        np.asarray(matched_confidences, dtype=np.float32),
        np.asarray(matched_distances, dtype=np.float32),
    )


def _summarize_threshold_rows(
    threshold_rows: dict[float, list[dict[str, float]]],
    threshold_distances: dict[float, list[np.ndarray]],
    *,
    match_radius_px: float,
    n_images: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for threshold in threshold_rows:
        metric_rows = threshold_rows[threshold]
        summary = aggregate_metric_rows(metric_rows)
        distances = (
            np.concatenate([arr for arr in threshold_distances[threshold] if len(arr) > 0])
            if any(len(arr) > 0 for arr in threshold_distances[threshold])
            else np.empty((0,), dtype=np.float32)
        )
        total_true = int(summary["micro_tp"] + summary["micro_fn"])
        total_pred = int(summary["micro_tp"] + summary["micro_fp"])
        if len(distances) > 0:
            mean_error_px = float(np.mean(distances))
            median_error_px = float(np.median(distances))
            rmse_error_px = float(np.sqrt(np.mean(np.square(distances))))
        else:
            mean_error_px = float("nan")
            median_error_px = float("nan")
            rmse_error_px = float("nan")

        row: dict[str, object] = {
            "probability": float(threshold),
            "match_radius_px": float(match_radius_px),
            "n_images": int(n_images),
            "n_true_spots": total_true,
            "n_pred_spots": total_pred,
            "n_matched_spots": int(len(distances)),
            "mean_error_px": mean_error_px,
            "median_error_px": median_error_px,
            "rmse_error_px": rmse_error_px,
        }
        for key, value in summary.items():
            if key != "n_images":
                row[key] = value
        rows.append(row)

    return rows


def _select_best_threshold_from_sweep(
    threshold_metric_rows: list[dict[str, object]],
    *,
    reference_probability: float = DEFAULT_PROBABILITY,
) -> tuple[float, dict[str, object], float, float]:
    if not threshold_metric_rows:
        raise ValueError("threshold_metric_rows must not be empty.")

    reference_row = min(
        threshold_metric_rows,
        key=lambda row: abs(float(row["probability"]) - reference_probability),
    )
    reference_precision = float(reference_row["micro_precision"])
    precision_floor = reference_precision

    eligible_rows = [
        row
        for row in threshold_metric_rows
        if float(row["micro_precision"]) + 1e-12 >= precision_floor
    ]
    if not eligible_rows:
        eligible_rows = [reference_row]

    best_row = max(
        eligible_rows,
        key=lambda row: (
            float(row["micro_f1"]),
            float(row["micro_precision"]),
            float(row["micro_recall"]),
            float(row["probability"]),
        ),
    )
    return (
        float(best_row["probability"]),
        best_row,
        float(reference_row["probability"]),
        reference_precision,
    )


def _plot_f1_curve(
    output_path: Path,
    cutoffs_px: np.ndarray,
    all_f1_scores: list[np.ndarray],
    match_radius_px: float,
) -> None:
    overlay_cli._require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = overlay_cli.plt.subplots(figsize=(6, 4), dpi=150)
    if all_f1_scores:
        stacked = np.stack(all_f1_scores, axis=0)
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)
        ax.plot(cutoffs_px, mean_curve, color="tab:blue", lw=2, label="mean F1")
        ax.fill_between(
            cutoffs_px,
            np.clip(mean_curve - std_curve, 0.0, 1.0),
            np.clip(mean_curve + std_curve, 0.0, 1.0),
            color="tab:blue",
            alpha=0.18,
            label="mean +/- std",
        )
        ax.axvline(match_radius_px, color="tab:red", ls="--", lw=1, label="match radius")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(float(cutoffs_px[0]), float(cutoffs_px[-1]))
        ax.set_xlabel("Distance cutoff (px)")
        ax.set_ylabel("F1")
        ax.set_title("F1 vs distance cutoff")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No evaluation samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    overlay_cli.plt.close(fig)


def _plot_error_cdf(output_path: Path, error_distances: np.ndarray) -> None:
    overlay_cli._require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = overlay_cli.plt.subplots(figsize=(6, 4), dpi=150)
    if len(error_distances) > 0:
        sorted_errors = np.sort(error_distances)
        cdf = np.arange(1, len(sorted_errors) + 1, dtype=np.float32) / float(len(sorted_errors))
        ax.plot(sorted_errors, cdf, color="tab:green", lw=2)
        ax.axvline(float(np.median(sorted_errors)), color="tab:green", ls="--", lw=1)
        ax.set_xlabel("Localization error (px)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Matched localization error CDF")
    else:
        ax.text(0.5, 0.5, "No matched coordinates", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    overlay_cli.plt.close(fig)


def _summarize_row(row: dict[str, object]) -> dict[str, object]:
    summarized: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, Path):
            summarized[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            summarized[key] = value.item()
        else:
            summarized[key] = value
    return summarized


def run_test(args: argparse.Namespace) -> Path:
    """Evaluate the checkpoint on one dataset split and write summary artifacts."""
    data_path = Path(args.data).resolve()
    images_dir = Path(args.images).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output).resolve()
    raw_probability = getattr(args, "probability", None)
    probability_was_specified = raw_probability is not None
    requested_probability = (
        DEFAULT_PROBABILITY if raw_probability is None else float(raw_probability)
    )

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not (0 <= requested_probability <= 1):
        raise ValueError(f"--probability must be in [0, 1], got {requested_probability}")
    if args.match_radius_px < 0:
        raise ValueError(f"--match-radius-px must be >= 0, got {args.match_radius_px}")
    probability_sweep = _parse_probability_sweep(getattr(args, "probability_sweep", None))
    sweep_thresholds = _thresholds_with_reference(
        probability_sweep,
        extra_thresholds=[requested_probability] if probability_was_specified else None,
    )

    samples = _load_eval_samples(data_path, args.subset)
    if not samples:
        raise ValueError(f"Split {args.subset!r} contains no samples.")

    device = resolve_device(args.device)
    model, cell_size = predict_cli._load_model(model_path, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary"
    predictions_dir = output_dir / "predictions"
    matches_dir = output_dir / "matches"
    overlays_dir = output_dir / "overlays"
    match_overlays_dir = output_dir / "match_overlays"
    figures_dir = output_dir / "figures"

    eval_predictions: list[EvalPrediction] = []
    per_image_rows: list[dict[str, object]] = []
    sample_predictions: list[tuple[np.ndarray, np.ndarray]] = []
    all_f1_scores: list[np.ndarray] = []
    all_error_distances: list[np.ndarray] = []
    threshold_rows: dict[float, list[dict[str, float]]] = {
        threshold: [] for threshold in sweep_thresholds
    }
    threshold_distances: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in sweep_thresholds
    }

    print(
        f"Evaluating split={args.subset} | device={device} | data={data_path} "
        f"| model={model_path.name} | samples={len(samples)} | "
        f"probability={requested_probability} | match_radius_px={args.match_radius_px}"
    )

    for sample in samples:
        image_path = _find_image_file(images_dir, sample.image_name, sample.sample_id)
        image = predict_cli._read_image(image_path)
        orig_h, orig_w = image.shape
        padded = predict_cli.prepare_image(image, getattr(model, "input_size", None))

        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = predict_cli._forward_with_geometry(model, tensor, orig_h, orig_w).squeeze(0).detach().cpu().numpy()

        truth_coords_px = np.asarray(sample.truth_coords_px, dtype=np.float32)
        eval_predictions.append(
            EvalPrediction(
                sample=sample,
                image_path=image_path,
                image=image,
                pred=pred,
                orig_h=orig_h,
                orig_w=orig_w,
                truth_coords_px=truth_coords_px,
            )
        )

        for threshold in sweep_thresholds:
            sweep_coords_with_p = _decode_prediction_with_probability(
                pred,
                cell_size=cell_size,
                probability=threshold,
                orig_h=orig_h,
                orig_w=orig_w,
            )
            sweep_match = match_coordinates(
                sweep_coords_with_p[:, :2],
                truth_coords_px,
                args.match_radius_px,
            )
            sweep_metrics = metrics_from_match(sweep_match)
            _sweep_cutoffs, _sweep_f1_scores, sweep_f1_integral = f1_curve(
                sweep_coords_with_p[:, :2],
                truth_coords_px,
                max_distance_px=_f1_curve_limit(float(args.match_radius_px)),
                n_cutoffs=DEFAULT_N_CUTOFFS,
            )
            sweep_metrics["f1_integral"] = sweep_f1_integral
            threshold_rows[threshold].append(sweep_metrics)
            threshold_distances[threshold].append(
                np.asarray(sweep_match.matched_distances, dtype=np.float32)
            )

        print(f"Inferred {sample.sample_id} | true={len(truth_coords_px)}")

    threshold_metric_rows: list[dict[str, object]] = []
    selected_probability = requested_probability
    probability_source = "specified" if probability_was_specified else "default"
    best_probability_from_sweep: float | str = ""
    best_sweep_micro_precision: float | str = ""
    best_sweep_micro_recall: float | str = ""
    best_sweep_micro_f1: float | str = ""
    precision_reference_probability: float | str = ""
    precision_reference: float | str = ""

    if sweep_thresholds:
        threshold_metric_rows = _summarize_threshold_rows(
            threshold_rows,
            threshold_distances,
            match_radius_px=float(args.match_radius_px),
            n_images=len(samples),
        )
        (
            best_probability,
            best_threshold_row,
            reference_probability,
            reference_precision,
        ) = _select_best_threshold_from_sweep(threshold_metric_rows)
        best_probability_from_sweep = best_probability
        best_sweep_micro_precision = float(best_threshold_row["micro_precision"])
        best_sweep_micro_recall = float(best_threshold_row["micro_recall"])
        best_sweep_micro_f1 = float(best_threshold_row["micro_f1"])
        precision_reference_probability = reference_probability
        precision_reference = reference_precision

        if probability_was_specified:
            selected_probability = requested_probability
            probability_source = "specified"
            print(
                f"Best threshold from sweep: p={best_probability:g}, but "
                f"--probability p={requested_probability:g} was specified; "
                "using the specified threshold for predictions and overlays."
            )
        else:
            selected_probability = best_probability
            probability_source = "sweep_best_f1_precision_constrained"
            print(
                f"Best threshold from sweep: p={best_probability:g}, using it for overlays "
                f"(precision floor from p={reference_probability:g}: {reference_precision:.4f})."
            )

        for row in threshold_metric_rows:
            is_selected = abs(float(row["probability"]) - selected_probability) <= 1e-12
            row["selected_for_outputs"] = is_selected
            if is_selected:
                row["selection_reason"] = probability_source
            else:
                row["selection_reason"] = ""

    print(
        f"Using probability={selected_probability:g} for predictions, matches, and overlays "
        f"| source={probability_source}"
    )

    for item in eval_predictions:
        sample = item.sample
        coords_with_p = _decode_prediction_with_probability(
            item.pred,
            cell_size=cell_size,
            probability=selected_probability,
            orig_h=item.orig_h,
            orig_w=item.orig_w,
        )

        prediction_csv = predictions_dir / f"{sample.sample_id}.csv"
        predict_cli._write_csv(prediction_csv, coords_with_p, output_unit="px")
        predict_cli._write_prediction_metadata(
            csv_path=prediction_csv,
            image_path=item.image_path,
            model_path=model_path,
            output_unit="px",
        )

        truth_coords_px = item.truth_coords_px
        match = match_coordinates(coords_with_p[:, :2], truth_coords_px, args.match_radius_px)
        metrics = metrics_from_match(match)
        _cutoffs_px, f1_scores, f1_integral = f1_curve(
            coords_with_p[:, :2],
            truth_coords_px,
            max_distance_px=_f1_curve_limit(float(args.match_radius_px)),
            n_cutoffs=DEFAULT_N_CUTOFFS,
        )
        all_f1_scores.append(f1_scores)
        all_error_distances.append(np.asarray(match.matched_distances, dtype=np.float32))

        match_csv = matches_dir / f"{sample.sample_id}.csv"
        match_rows = _build_match_rows(coords_with_p, truth_coords_px, match)
        _write_rows_csv(
            match_csv,
            match_rows,
            fieldnames=[
                "match_status",
                "pred_index",
                "true_index",
                "pred_y_px",
                "pred_x_px",
                "pred_p",
                "true_y_px",
                "true_x_px",
                "distance_px",
            ],
        )

        footer_text = overlay_cli._build_footer_text(
            image_name=item.image_path.name,
            model_name=model_path.name,
        )
        overlay_png = overlays_dir / f"{sample.sample_id}.png"
        overlay_cli._save_overlay_png(
            image=item.image,
            coords=coords_with_p[:, :2],
            output_path=overlay_png,
            mode=args.overlay_mode,
            footer_text=footer_text,
        )
        tp_coords, fp_coords, fn_coords = _match_overlay_coords(
            coords_with_p,
            truth_coords_px,
            match,
        )
        match_overlay_png = match_overlays_dir / f"{sample.sample_id}.png"
        overlay_cli._save_match_overlay_png(
            image=item.image,
            tp_coords=tp_coords,
            fp_coords=fp_coords,
            fn_coords=fn_coords,
            output_path=match_overlay_png,
            mode=args.overlay_mode,
            footer_text=footer_text,
        )

        sample_predictions.append((coords_with_p, truth_coords_px))

        per_image_row = {
            "sample_id": sample.sample_id,
            "image_name": sample.image_name,
            "label_name": sample.label_name,
            "source_index": sample.source_index,
            "n_true": int(len(truth_coords_px)),
            "n_pred": int(len(coords_with_p)),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mean_error_px": metrics["mean_error_px"],
            "median_error_px": metrics["median_error_px"],
            "rmse_px": metrics["rmse_px"],
            "f1_integral": f1_integral,
            "prediction_csv": _as_relative_path(prediction_csv, output_dir),
            "match_csv": _as_relative_path(match_csv, output_dir),
            "overlay_png": _as_relative_path(overlay_png, output_dir),
            "match_overlay_png": _as_relative_path(match_overlay_png, output_dir),
        }
        per_image_rows.append(per_image_row)

        print(
            f"Evaluated {sample.sample_id} | true={len(truth_coords_px)} | pred={len(coords_with_p)} "
            f"| tp={metrics['tp']:.0f} | fp={metrics['fp']:.0f} | fn={metrics['fn']:.0f} "
            f"| f1={metrics['f1']:.4f} | median_error_px={metrics['median_error_px']:.4f} "
            f"| probability={selected_probability:g}"
        )

    concatenated_errors = (
        np.concatenate([arr for arr in all_error_distances if len(arr) > 0])
        if any(len(arr) > 0 for arr in all_error_distances)
        else np.empty((0,), dtype=np.float32)
    )
    summary = aggregate_metric_rows(per_image_rows)
    total_true = int(summary["micro_tp"] + summary["micro_fn"])
    total_pred = int(summary["micro_tp"] + summary["micro_fp"])
    if len(concatenated_errors) > 0:
        mean_error_px = float(np.mean(concatenated_errors))
        median_error_px = float(np.median(concatenated_errors))
        rmse_error_px = float(np.sqrt(np.mean(np.square(concatenated_errors))))
    else:
        mean_error_px = float("nan")
        median_error_px = float("nan")
        rmse_error_px = float("nan")
    summary.update(
        {
            "subset": args.subset,
            "data_file_name": data_path.name,
            "model_file_name": model_path.name,
            "match_radius_px": float(args.match_radius_px),
            "probability": float(selected_probability),
            "probability_source": probability_source,
            "requested_probability": (
                float(requested_probability) if probability_was_specified else ""
            ),
            "best_probability_from_sweep": best_probability_from_sweep,
            "best_sweep_micro_precision": best_sweep_micro_precision,
            "best_sweep_micro_recall": best_sweep_micro_recall,
            "best_sweep_micro_f1": best_sweep_micro_f1,
            "precision_reference_probability": precision_reference_probability,
            "precision_reference": precision_reference,
            "device": str(device),
            "cell_size": int(cell_size),
            "n_true_spots": total_true,
            "n_pred_spots": total_pred,
            "n_matched_spots": int(len(concatenated_errors)),
            "mean_error_px": mean_error_px,
            "median_error_px": median_error_px,
            "rmse_error_px": rmse_error_px,
            "n_images": int(len(samples)),
        }
    )

    recall, precision, average_precision, matched_confidences, matched_distances = _compute_pr_curve(
        sample_predictions,
        float(args.match_radius_px),
    )
    summary["average_precision"] = average_precision

    summary_csv = summary_dir / "metrics_summary.csv"
    summary_json = summary_dir / "metrics_summary.json"
    per_image_csv = summary_dir / "per_image_metrics.csv"
    threshold_metrics_csv = summary_dir / "threshold_metrics.csv"

    _write_rows_csv(summary_csv, [_summarize_row(summary)], fieldnames=list(summary.keys()))
    _write_json(summary_json, _summarize_row(summary))
    _write_rows_csv(
        per_image_csv,
        [_summarize_row(row) for row in per_image_rows],
        fieldnames=list(per_image_rows[0].keys()),
    )
    if threshold_metric_rows:
        _write_rows_csv(
            threshold_metrics_csv,
            [_summarize_row(row) for row in threshold_metric_rows],
            fieldnames=list(threshold_metric_rows[0].keys()),
        )

    _plot_precision_recall_curve(figures_dir / "pr_curve.png", recall, precision, average_precision)
    _plot_sorted_per_image_f1(figures_dir / "per_image_f1_boxplot.png", per_image_rows)

    if args.verbose:
        cutoffs_px = np.linspace(
            0.0,
            float(args.match_radius_px) if args.match_radius_px > 0 else 0.1,
            num=DEFAULT_N_CUTOFFS,
            dtype=np.float32,
        )
        _plot_f1_curve(figures_dir / "f1_curve.png", cutoffs_px, all_f1_scores, float(args.match_radius_px))
        _plot_error_cdf(figures_dir / "error_cdf.png", concatenated_errors)
        _plot_error_vs_confidence(
            figures_dir / "localization_error_vs_confidence.png",
            matched_confidences,
            matched_distances,
        )

    print(
        f"Summary | precision={summary['micro_precision']:.4f} | recall={summary['micro_recall']:.4f} "
        f"| F1={summary['micro_f1']:.4f} | median_error_px={summary['median_error_px']:.4f} "
        f"| mean_f1_integral={summary['mean_f1_integral']:.4f} | AP={summary['average_precision']:.4f} "
        f"| tp={summary['micro_tp']:.0f} "
        f"| fp={summary['micro_fp']:.0f} | fn={summary['micro_fn']:.0f}"
    )
    print(
        f"Outputs | summary={summary_dir} | predictions={predictions_dir} | matches={matches_dir} "
        f"| overlays={overlays_dir} | match_overlays={match_overlays_dir} | figures={figures_dir}"
    )
    if probability_sweep:
        print(f"Threshold sweep saved to: {threshold_metrics_csv}")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for evaluation."""
    args = parse_args(argv)
    run_test(args)


if __name__ == "__main__":
    main()
