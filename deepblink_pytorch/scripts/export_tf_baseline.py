"""Export deepBlink TensorFlow baseline predictions as CSV files.

This script assumes the original TensorFlow-based deepBlink package is already
installed and importable in the current environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from deepblink.inference import predict
from deepblink.io import load_image
from deepblink.io import load_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export TensorFlow deepBlink baseline predictions."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a TensorFlow deepBlink .h5 model.",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing input .tif images.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory for CSV files. Defaults to "
            "artifacts/tf_baseline relative to this script."
        ),
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=0.5,
        help="Probability threshold used for prediction export.",
    )
    return parser.parse_args()


def resolve_output_dir(output: str | None) -> Path:
    """Return the output directory for baseline CSV files."""
    if output is not None:
        outdir = Path(output).resolve()
    else:
        outdir = (Path(__file__).resolve().parents[1] / "artifacts" / "tf_baseline")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def iter_tif_files(image_dir: Path) -> list[Path]:
    """Return all tif and tiff files under the image directory."""
    tif_files = sorted(image_dir.glob("*.tif"))
    tif_files.extend(sorted(image_dir.glob("*.tiff")))
    tif_files = sorted(tif_files)
    if not tif_files:
        raise FileNotFoundError(f"No .tif or .tiff files found in {image_dir}.")
    return tif_files


def export_predictions(
    model_path: Path, image_dir: Path, output_dir: Path, probability: float
) -> None:
    """Run batch inference and export CSV files."""
    model = tf.keras.models.load_model(str(model_path), compile=False)
    image_files = iter_tif_files(image_dir)

    for image_file in image_files:
        image = load_image(str(image_file), is_rgb=False)
        coords = predict(image, model, probability=probability)
        frame = pd.DataFrame(coords, columns=["y", "x", "p"])
        output_file = output_dir / f"{image_file.stem}.csv"
        frame.to_csv(output_file, index=False)
        print(f"Saved {output_file}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    model_path = Path(args.model).resolve()
    image_dir = Path(args.images).resolve()
    output_dir = resolve_output_dir(args.output)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    export_predictions(
        model_path=model_path,
        image_dir=image_dir,
        output_dir=output_dir,
        probability=args.probability,
    )


if __name__ == "__main__":
    main()
