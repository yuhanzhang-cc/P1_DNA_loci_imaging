"""Dataset creation command for deepblink_torch CLI."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - import guard
    tifffile = None
    TIFF_IMPORT_ERROR = exc
else:
    TIFF_IMPORT_ERROR = None

IMAGE_EXTENSIONS = (".tif", ".tiff")


def add_create_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add dataset creation arguments to an argparse parser."""
    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing input image files (.tif/.tiff).",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help=(
            "Directory containing per-image CSV labels with matching file stem. "
            "Supported columns: (r,c), (y,x), (Y,X), (POSITION_Y,POSITION_X)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz path.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split shuffling.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse dataset creation arguments."""
    parser = argparse.ArgumentParser(
        description="Create deepBlink-compatible .npz from images and CSV labels."
    )
    add_create_arguments(parser)
    return parser.parse_args(argv)


def _resolve_csv_columns(fieldnames: list[str]) -> tuple[str, str]:
    candidates = [
        ("r", "c"),
        ("y", "x"),
        ("Y", "X"),
        ("POSITION_Y", "POSITION_X"),
    ]
    for row_col, col_col in candidates:
        if row_col in fieldnames and col_col in fieldnames:
            return row_col, col_col
    raise ValueError(
        "CSV must contain one of column pairs: "
        "(r,c), (y,x), (Y,X), (POSITION_Y,POSITION_X)."
    )


def _read_coords(csv_path: Path) -> np.ndarray:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return np.empty((0, 2), dtype=np.float32)
        row_col, col_col = _resolve_csv_columns(reader.fieldnames)

        coords: list[list[float]] = []
        for item in reader:
            if item.get(row_col) in (None, "") or item.get(col_col) in (None, ""):
                continue
            coords.append([float(item[row_col]), float(item[col_col])])

    if not coords:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def _read_image(image_path: Path) -> np.ndarray:
    if tifffile is None:
        raise ImportError(
            "tifffile is required for dataset creation. "
            "Install with `pip install tifffile`."
        ) from TIFF_IMPORT_ERROR

    image = tifffile.imread(str(image_path))
    if image.ndim > 2:
        image = np.squeeze(image)
        if image.ndim != 2:
            raise ValueError(
                f"Only 2D grayscale images are supported, got shape {image.shape} "
                f"for {image_path}."
            )
    return image.astype(np.float32, copy=False)


def _collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(sorted(images_dir.glob(f"*{ext}")))
    image_files = sorted(image_files)

    if not image_files:
        raise FileNotFoundError(f"No image files found in {images_dir}.")

    pairs: list[tuple[Path, Path]] = []
    missing = []
    for image_path in image_files:
        csv_path = labels_dir / f"{image_path.stem}.csv"
        if not csv_path.exists():
            missing.append(csv_path.name)
            continue
        pairs.append((image_path, csv_path))

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            "Missing label CSV files for some images. "
            f"Examples: {preview}"
        )
    return pairs


def _validate_image_shapes(images: list[np.ndarray]) -> None:
    first_shape = images[0].shape
    if any(img.shape != first_shape for img in images):
        raise ValueError("All images must have the same shape.")
    if first_shape[0] != first_shape[1]:
        raise ValueError(
            f"Images must be square for deepBlink training, got shape {first_shape}."
        )


def _split_indices(
    n_samples: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if n_samples < 3:
        raise ValueError("Need at least 3 image-label pairs to create train/valid/test splits.")
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError(
            "train_ratio and valid_ratio must be > 0 and train_ratio + valid_ratio < 1."
        )

    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)

    n_train = max(1, int(round(n_samples * train_ratio)))
    n_valid = max(1, int(round(n_samples * valid_ratio)))
    n_test = n_samples - n_train - n_valid
    if n_test < 1:
        n_test = 1
        if n_train > n_valid:
            n_train -= 1
        else:
            n_valid -= 1

    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid :]
    return train_idx, valid_idx, test_idx


def _gather_subset(
    images: list[np.ndarray],
    coords: list[np.ndarray],
    subset_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([images[i] for i in subset_idx], dtype=np.float32)
    y = np.asarray([coords[i] for i in subset_idx], dtype=object)
    return x, y


def run_create(args: argparse.Namespace) -> Path:
    """Create .npz dataset archive and return output path."""
    images_dir = Path(args.images).resolve()
    labels_dir = Path(args.labels).resolve()
    output_path = Path(args.output).resolve()

    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels directory not found: {labels_dir}")

    pairs = _collect_pairs(images_dir, labels_dir)
    images = [_read_image(image_path) for image_path, _ in pairs]
    coords = [_read_coords(csv_path) for _, csv_path in pairs]
    _validate_image_shapes(images)

    train_idx, valid_idx, test_idx = _split_indices(
        n_samples=len(pairs),
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    x_train, y_train = _gather_subset(images, coords, train_idx)
    x_valid, y_valid = _gather_subset(images, coords, valid_idx)
    x_test, y_test = _gather_subset(images, coords, test_idx)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
    )
    print(
        f"Saved dataset: {output_path} | "
        f"train={len(x_train)}, valid={len(x_valid)}, test={len(x_test)}"
    )
    return output_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for dataset creation."""
    args = parse_args(argv)
    run_create(args)


if __name__ == "__main__":
    main()

