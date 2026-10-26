"""Prediction command for deepblink_torch CLI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from deepblink_torch import UNetDeepBlink
from deepblink_torch import UNetDeepLociContext
from deepblink_torch import get_coordinate_list
from deepblink_torch.data_protocol import next_power
from ..utils.pixel_size import PixelSize
from ..utils.pixel_size import read_pixel_size_or_default
from ..preprocessing import prepare_image
from ..networks.evidence import UNetDeepLociEvidence, UNetDeepLociLocal
from ..networks.feature_evidence import UNetDeepLociFeatureEvidence
from ..networks.masked_se import UNetDeepBlinkMaskedSE

try:
    import tifffile
except ImportError as exc:  # pragma: no cover - import guard
    tifffile = None
    TIFF_IMPORT_ERROR = exc
else:
    TIFF_IMPORT_ERROR = None

IMAGE_EXTENSIONS = (".tif", ".tiff")
MODEL_CHOICES = ("deepblink", "deeploci_context", "deeploci_evidence", "deeploci_local",
                 "deeploci_feature_evidence", "deepblink_masked_se")


def add_predict_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add prediction arguments to an argparse parser."""
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument(
        "--images",
        required=True,
        help="Path to one image file or a directory with .tif/.tiff images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file or output directory (if --images is a directory).",
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=0.5,
        help="Probability threshold for coordinate decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "xpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--output-unit",
        type=str,
        default="px",
        choices=["px", "micron"],
        help="Coordinate unit written to CSV columns.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse prediction CLI arguments."""
    parser = argparse.ArgumentParser(description="Predict spots with deepblink_torch.")
    add_predict_arguments(parser)
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    """Resolve runtime device, falling back to CPU if CUDA is unavailable."""
    if requested == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("Intel XPU requested but unavailable in this environment.")
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda but CUDA is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def _read_image(image_path: Path) -> np.ndarray:
    if tifffile is None:
        raise ImportError(
            "tifffile is required for prediction. Install with `pip install tifffile`."
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


def _normalize_image(image: np.ndarray) -> np.ndarray:
    std = float(np.std(image))
    if std == 0:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - float(np.mean(image))) / std).astype(np.float32, copy=False)


def _pad_to_square_power2(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape
    target = next_power(max(height, width))
    pad_width = ((0, target - height), (0, target - width))
    padded = np.pad(image, pad_width, mode="reflect").astype(np.float32, copy=False)
    return padded, (height, width)


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


def _normalise_model_name(model_name: object) -> str:
    normalized = str(model_name or "deepblink")
    if normalized not in MODEL_CHOICES:
        raise ValueError(
            f"Unsupported checkpoint model_name={normalized!r}. "
            f"Expected one of {MODEL_CHOICES}."
        )
    return normalized


def _checkpoint_model_name(checkpoint: object) -> str:
    if not isinstance(checkpoint, dict):
        return "deepblink"

    raw_model_name = checkpoint.get("model_name")
    if raw_model_name is None:
        model_config = checkpoint.get("model_config")
        if isinstance(model_config, dict):
            raw_model_name = model_config.get("model_name")
    if raw_model_name is None:
        architecture_config = checkpoint.get("architecture_config")
        if isinstance(architecture_config, dict):
            raw_model_name = architecture_config.get("model_name")
    if raw_model_name is None:
        args = checkpoint.get("args")
        if isinstance(args, dict):
            raw_model_name = args.get("model")
    return _normalise_model_name(raw_model_name or "deepblink")


def _build_model_for_checkpoint(
    model_name: str,
    *,
    cell_size: int,
    device: torch.device,
) -> torch.nn.Module:
    if model_name == "deepblink":
        return UNetDeepBlink(cell_size=cell_size).to(device)
    if model_name == "deeploci_context":
        return UNetDeepLociContext(cell_size=cell_size).to(device)
    if model_name == "deeploci_evidence":
        return UNetDeepLociEvidence(cell_size=cell_size).to(device)
    if model_name == "deeploci_local":
        return UNetDeepLociLocal(cell_size=cell_size).to(device)
    if model_name == "deeploci_feature_evidence":
        return UNetDeepLociFeatureEvidence(cell_size=cell_size).to(device)
    if model_name == "deepblink_masked_se":
        return UNetDeepBlinkMaskedSE(cell_size=cell_size).to(device)
    raise ValueError(f"Unsupported checkpoint model_name={model_name!r}.")


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cell_size = int(checkpoint.get("cell_size", 4)) if isinstance(checkpoint, dict) else 4
    model_name = _checkpoint_model_name(checkpoint)

    model = _build_model_for_checkpoint(
        model_name,
        cell_size=cell_size,
        device=device,
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.input_size = checkpoint.get("input_size") if isinstance(checkpoint, dict) else None
    return model, cell_size


def _forward_with_geometry(model, tensor, orig_h, orig_w):
    if getattr(model, "requires_valid_mask", False):
        mask = torch.zeros_like(tensor)
        mask[..., :orig_h, :orig_w] = 1
        return model(tensor, valid_mask=mask)
    return model(tensor)


def _coords_with_probabilities(
    pred_matrix: np.ndarray,
    coords: np.ndarray,
    cell_size: int,
) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.float32)

    result = np.zeros((len(coords), 3), dtype=np.float32)
    result[:, :2] = coords

    max_r = pred_matrix.shape[0] - 1
    max_c = pred_matrix.shape[1] - 1
    for idx, (r, c) in enumerate(coords):
        cell_r = min(max_r, int(np.floor(r)) // cell_size)
        cell_c = min(max_c, int(np.floor(c)) // cell_size)
        result[idx, 2] = pred_matrix[cell_r, cell_c, 0]
    return result


def _decode_with_scores(pred_matrix, *, cell_size, probability, orig_h, orig_w):
    """Keep source grid scores when offsets round to a neighboring boundary.

    A coordinate alone cannot distinguish offset=1 in one cell from offset=0
    in its neighbor. Preserve the selected cell indices through decoding.
    """
    rows, cols = np.nonzero(pred_matrix[..., 0] > probability)
    coords = (np.column_stack([rows, cols]) + pred_matrix[rows, cols, 1:]) * cell_size
    scores = pred_matrix[rows, cols, 0]
    inside = ((coords >= 0).all(axis=1) & (coords[:, 0] < orig_h) & (coords[:, 1] < orig_w))
    return np.column_stack([coords[inside], scores[inside]]).astype(np.float32)


def _convert_output_unit(coords_with_p: np.ndarray, pixel_size: PixelSize) -> np.ndarray:
    if coords_with_p.size == 0:
        return coords_with_p

    converted = coords_with_p.copy()
    converted[:, 0] *= float(pixel_size.y)
    converted[:, 1] *= float(pixel_size.x)
    return converted


def _write_csv(output_path: Path, coords_with_p: np.ndarray, output_unit: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if output_unit == "micron":
            writer.writerow(["y [um]", "x [um]", "p"])
        else:
            writer.writerow(["y [px]", "x [px]", "p"])
        for row in coords_with_p:
            writer.writerow([float(row[0]), float(row[1]), float(row[2])])


def _write_prediction_metadata(
    csv_path: Path,
    image_path: Path,
    model_path: Path,
    output_unit: str,
) -> None:
    metadata_path = csv_path.with_suffix(".meta.json")
    metadata = {
        "csv_file_name": csv_path.name,
        "image_file_name": image_path.name,
        "model_file_name": model_path.name,
        "output_unit": output_unit,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_predict(args: argparse.Namespace) -> None:
    """Run inference on one or many images."""
    model_path = Path(args.model).resolve()
    image_path = Path(args.images).resolve()
    output_path = Path(args.output).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not (0 <= args.probability <= 1):
        raise ValueError(f"--probability must be in [0, 1], got {args.probability}")

    device = resolve_device(args.device)
    model, cell_size = _load_model(model_path, device)
    image_files = _collect_images(image_path)

    output_is_dir = image_path.is_dir()
    if output_is_dir:
        output_path.mkdir(parents=True, exist_ok=True)
    elif output_path.exists() and output_path.is_dir():
        output_is_dir = True

    for file_path in image_files:
        image = _read_image(file_path)
        orig_h, orig_w = image.shape
        padded = prepare_image(image, getattr(model, "input_size", None))

        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = _forward_with_geometry(model, tensor, orig_h, orig_w).squeeze(0).detach().cpu().numpy()

        coords_with_p = _decode_with_scores(pred, cell_size=cell_size, probability=args.probability,
                                            orig_h=orig_h, orig_w=orig_w)
        if args.output_unit == "micron":
            pixel_size = read_pixel_size_or_default(file_path)
            coords_with_p = _convert_output_unit(coords_with_p, pixel_size)

        if output_is_dir:
            csv_path = output_path / f"{file_path.stem}.csv"
        else:
            csv_path = output_path
        _write_csv(csv_path, coords_with_p, output_unit=args.output_unit)
        _write_prediction_metadata(
            csv_path=csv_path,
            image_path=file_path,
            model_path=model_path,
            output_unit=args.output_unit,
        )
        print(f"Saved predictions: {csv_path} | output_unit={args.output_unit}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for prediction."""
    args = parse_args(argv)
    run_predict(args)


if __name__ == "__main__":
    main()
