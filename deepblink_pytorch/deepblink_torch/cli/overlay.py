"""Overlay visualization command for deepblink_torch CLI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard
    matplotlib = None
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - import guard
    tifffile = None
    TIFF_IMPORT_ERROR = exc
else:
    TIFF_IMPORT_ERROR = None

IMAGE_EXTENSIONS = (".tif", ".tiff")
MODES = ("side-by-side", "single")


def add_overlay_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add overlay arguments to an argparse parser."""
    parser.add_argument(
        "--images",
        required=True,
        help="Path to one image file or a directory with .tif/.tiff images.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "Path to one prediction CSV or a directory of per-image CSV files. "
            "CSV coordinates must be pixel columns from predict output. "
            "Matching .meta.json sidecars are used for PNG footers when present."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG file or output directory.",
    )
    parser.add_argument(
        "--mode",
        default="side-by-side",
        choices=MODES,
        help="Overlay rendering mode.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse overlay CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Render deepblink_torch prediction CSV files as PNG overlays."
    )
    add_overlay_arguments(parser)
    return parser.parse_args(argv)


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for overlay visualization. "
            "Install with `pip install matplotlib`."
        ) from MATPLOTLIB_IMPORT_ERROR


def _read_image(image_path: Path) -> np.ndarray:
    if tifffile is None:
        raise ImportError(
            "tifffile is required to read TIFF images. Install with `pip install tifffile`."
        ) from TIFF_IMPORT_ERROR

    image = tifffile.imread(str(image_path))
    if image.ndim > 2:
        image = np.squeeze(image)
        if image.ndim != 2:
            raise ValueError(
                f"Only 2D grayscale images are supported, got shape {image.shape} "
                f"for {image_path}."
            )
    return np.asarray(image)


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input image path not found: {path}")

    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(sorted(path.glob(f"*{ext}")))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {path}.")
    return files


def _resolve_csv_columns(fieldnames: list[str]) -> tuple[str, str]:
    normalized = {name.strip().lower(): name for name in fieldnames}
    if "y [um]" in normalized or "x [um]" in normalized:
        raise ValueError(
            "Overlay expects pixel-coordinate CSV files. Re-run predict with "
            "`--output-unit px` before rendering overlays."
        )

    for y_name, x_name in (("y", "x"), ("y [px]", "x [px]")):
        if y_name in normalized and x_name in normalized:
            return normalized[y_name], normalized[x_name]

    raise ValueError("CSV must contain pixel coordinate columns: y [px],x [px].")


def _read_coords(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return np.empty((0, 2), dtype=np.float32)

        y_col, x_col = _resolve_csv_columns(reader.fieldnames)
        coords: list[list[float]] = []
        for row in reader:
            if row.get(y_col) in (None, "") or row.get(x_col) in (None, ""):
                continue
            coords.append([float(row[y_col]), float(row[x_col])])

    if not coords:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def _pair_images_and_csvs(images_path: Path, csv_path: Path) -> list[tuple[Path, Path]]:
    image_files = _collect_images(images_path)

    if csv_path.is_file():
        if len(image_files) != 1:
            raise ValueError(
                "A single --csv file can only be used with a single --images file. "
                "Use a CSV directory for image-directory overlays."
            )
        return [(image_files[0], csv_path)]

    if not csv_path.is_dir():
        raise FileNotFoundError(f"Prediction CSV path not found: {csv_path}")

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for image_file in image_files:
        prediction_csv = csv_path / f"{image_file.stem}.csv"
        if not prediction_csv.exists():
            missing.append(prediction_csv.name)
            continue
        pairs.append((image_file, prediction_csv))

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            "Missing prediction CSV files for some images. "
            f"Examples: {preview}"
        )

    return pairs


def _resolve_output_png(output_path: Path, image_path: Path, n_pairs: int) -> Path:
    if n_pairs == 1 and output_path.suffix.lower() == ".png":
        return output_path
    return output_path / f"{image_path.stem}_overlay.png"


def _prediction_metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def _read_prediction_metadata(csv_path: Path) -> dict[str, str]:
    metadata_path = _prediction_metadata_path(csv_path)
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid prediction metadata in {metadata_path}.")
    return {str(key): str(value) for key, value in metadata.items()}


def _build_footer_text(image_name: str, model_name: str | None) -> str:
    model_label = model_name or "unknown model"
    footer = f"image: {image_name} | model: {model_label}"
    return textwrap.fill(footer, width=96)


def _draw_overlay_axes(ax, image: np.ndarray, coords: np.ndarray) -> None:
    ax.imshow(image, cmap="gray", origin="upper")
    if len(coords) > 0:
        ax.scatter(
            coords[:, 1],
            coords[:, 0],
            s=18,
            c="red",
            marker="o",
            linewidths=0,
        )
    ax.set_axis_off()


def _save_overlay_png(
    image: np.ndarray,
    coords: np.ndarray,
    output_path: Path,
    mode: str,
    footer_text: str,
) -> None:
    _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "single":
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        _draw_overlay_axes(ax, image, coords)
    elif mode == "side-by-side":
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
        axes[0].imshow(image, cmap="gray", origin="upper")
        axes[0].set_title("Image")
        axes[0].set_axis_off()
        _draw_overlay_axes(axes[1], image, coords)
        axes[1].set_title("Prediction")
    else:
        raise ValueError(f"Unsupported overlay mode: {mode}")

    fig.subplots_adjust(bottom=0.14)
    fig.text(0.5, 0.03, footer_text, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.2)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_overlay(args: argparse.Namespace) -> None:
    """Render overlay PNG files from prediction CSV files."""
    images_path = Path(args.images).resolve()
    csv_path = Path(args.csv).resolve()
    output_path = Path(args.output).resolve()

    pairs = _pair_images_and_csvs(images_path, csv_path)
    if len(pairs) > 1 and output_path.suffix.lower() == ".png":
        raise ValueError("--output must be a directory when rendering multiple images.")

    for image_file, prediction_csv in pairs:
        image = _read_image(image_file)
        coords = _read_coords(prediction_csv)
        metadata = _read_prediction_metadata(prediction_csv)
        footer_text = _build_footer_text(
            image_name=metadata.get("image_file_name", image_file.name),
            model_name=metadata.get("model_file_name"),
        )
        png_path = _resolve_output_png(output_path, image_file, n_pairs=len(pairs))
        _save_overlay_png(image, coords, png_path, mode=args.mode, footer_text=footer_text)
        print(
            f"Saved overlay: {png_path} | image={image_file.name} "
            f"| csv={prediction_csv.name} | points={len(coords)} | mode={args.mode}"
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for overlay visualization."""
    args = parse_args(argv)
    run_overlay(args)


if __name__ == "__main__":
    main()
