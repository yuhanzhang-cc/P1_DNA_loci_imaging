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


@dataclass(frozen=True)
class EvalSample:
    """One evaluation sample with split metadata and ground truth coordinates."""

    sample_id: str
    image_name: str
    label_name: str
    source_index: int
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
        default=0.5,
        help="Probability threshold for coordinate decoding.",
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
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--overlay-mode",
        type=str,
        default="side-by-side",
        choices=overlay_cli.MODES,
        help="Overlay rendering mode for per-image PNGs.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a model on a deepBlink split.")
    add_test_arguments(parser)
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    """Resolve runtime device, falling back to CPU if CUDA is unavailable."""
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

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not (0 <= args.probability <= 1):
        raise ValueError(f"--probability must be in [0, 1], got {args.probability}")
    if args.match_radius_px < 0:
        raise ValueError(f"--match-radius-px must be >= 0, got {args.match_radius_px}")

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
    figures_dir = output_dir / "figures"

    per_image_rows: list[dict[str, object]] = []
    all_f1_scores: list[np.ndarray] = []
    all_error_distances: list[np.ndarray] = []

    print(
        f"Evaluating split={args.subset} | device={device} | data={data_path} "
        f"| model={model_path.name} | samples={len(samples)} | "
        f"probability={args.probability} | match_radius_px={args.match_radius_px}"
    )

    for sample in samples:
        image_path = _find_image_file(images_dir, sample.image_name, sample.sample_id)
        image = predict_cli._read_image(image_path)
        normalized = predict_cli._normalize_image(image)
        padded, (orig_h, orig_w) = predict_cli._pad_to_square_power2(normalized)

        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).squeeze(0).detach().cpu().numpy()

        coords = predict_cli.get_coordinate_list(
            pred,
            cell_size=cell_size,
            probability=args.probability,
        )
        if len(coords) > 0:
            in_bounds = (coords[:, 0] < orig_h) & (coords[:, 1] < orig_w)
            coords = coords[in_bounds]
        coords_with_p = predict_cli._coords_with_probabilities(pred, coords, cell_size=cell_size)

        prediction_csv = predictions_dir / f"{sample.sample_id}.csv"
        predict_cli._write_csv(prediction_csv, coords_with_p, output_unit="px")
        predict_cli._write_prediction_metadata(
            csv_path=prediction_csv,
            image_path=image_path,
            model_path=model_path,
            output_unit="px",
        )

        truth_coords_px = np.asarray(sample.truth_coords_px, dtype=np.float32)
        match = match_coordinates(coords_with_p[:, :2], truth_coords_px, args.match_radius_px)
        metrics = metrics_from_match(match)
        cutoff_limit = float(args.match_radius_px) if args.match_radius_px > 0 else 0.1
        _cutoffs_px, f1_scores, f1_integral = f1_curve(
            coords_with_p[:, :2],
            truth_coords_px,
            max_distance_px=cutoff_limit,
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
            image_name=image_path.name,
            model_name=model_path.name,
        )
        overlay_png = overlays_dir / f"{sample.sample_id}.png"
        overlay_cli._save_overlay_png(
            image=image,
            coords=coords_with_p[:, :2],
            output_path=overlay_png,
            mode=args.overlay_mode,
            footer_text=footer_text,
        )

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
        }
        per_image_rows.append(per_image_row)

        print(
            f"Evaluated {sample.sample_id} | true={len(truth_coords_px)} | pred={len(coords_with_p)} "
            f"| tp={metrics['tp']:.0f} | fp={metrics['fp']:.0f} | fn={metrics['fn']:.0f} "
            f"| f1={metrics['f1']:.4f} | median_error_px={metrics['median_error_px']:.4f}"
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
            "probability": float(args.probability),
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

    summary_csv = summary_dir / "metrics_summary.csv"
    summary_json = summary_dir / "metrics_summary.json"
    per_image_csv = summary_dir / "per_image_metrics.csv"

    _write_rows_csv(summary_csv, [_summarize_row(summary)], fieldnames=list(summary.keys()))
    _write_json(summary_json, _summarize_row(summary))
    _write_rows_csv(
        per_image_csv,
        [_summarize_row(row) for row in per_image_rows],
        fieldnames=list(per_image_rows[0].keys()),
    )

    cutoffs_px = np.linspace(
        0.0,
        float(args.match_radius_px) if args.match_radius_px > 0 else 0.1,
        num=DEFAULT_N_CUTOFFS,
        dtype=np.float32,
    )
    _plot_f1_curve(figures_dir / "f1_curve.png", cutoffs_px, all_f1_scores, float(args.match_radius_px))
    _plot_error_cdf(figures_dir / "error_cdf.png", concatenated_errors)

    print(
        f"Summary | precision={summary['micro_precision']:.4f} | recall={summary['micro_recall']:.4f} "
        f"| F1={summary['micro_f1']:.4f} | median_error_px={summary['median_error_px']:.4f} "
        f"| mean_f1_integral={summary['mean_f1_integral']:.4f} | tp={summary['micro_tp']:.0f} "
        f"| fp={summary['micro_fp']:.0f} | fn={summary['micro_fn']:.0f}"
    )
    print(
        f"Outputs | summary={summary_dir} | predictions={predictions_dir} | matches={matches_dir} "
        f"| overlays={overlays_dir} | figures={figures_dir}"
    )
    return output_dir


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for evaluation."""
    args = parse_args(argv)
    run_test(args)


if __name__ == "__main__":
    main()
