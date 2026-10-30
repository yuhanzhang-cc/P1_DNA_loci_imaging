"""Shared image preparation; fixed canvas is opt-in for new checkpoints."""

from __future__ import annotations

import numpy as np

from .data_protocol import next_power


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Image contains non-finite values.")
    std = float(image.std())
    if std == 0:
        return np.zeros_like(image)
    return ((image - float(image.mean())) / std).astype(np.float32)


def prepare_image(image: np.ndarray, input_size: int | None = None) -> np.ndarray:
    """Normalize original pixels, then pad right/bottom without rescaling."""
    normalized = normalize_image(image)
    height, width = normalized.shape
    if input_size is None:
        target = next_power(max(height, width))
        return np.pad(normalized, ((0, target-height), (0, target-width)), mode="reflect")
    if input_size < max(height, width) or input_size != next_power(input_size):
        raise ValueError("input_size must be a power of two covering the original image.")
    return np.pad(normalized, ((0, input_size-height), (0, input_size-width)), mode="constant")
