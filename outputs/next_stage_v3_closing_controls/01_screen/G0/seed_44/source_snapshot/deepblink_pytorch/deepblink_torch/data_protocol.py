"""Numpy-only data protocol utilities for the PyTorch deepBlink migration.

This module mirrors the coordinate <-> prediction-matrix contract used by the
original TensorFlow deepBlink implementation in ``deepblink/data.py``.

The core representation is a 3-channel prediction matrix with layout::

    [..., 0] -> p      : object presence / probability for a grid cell
    [..., 1] -> r_rel  : relative row offset inside the cell
    [..., 2] -> c_rel  : relative column offset inside the cell

Coordinates are always represented internally as ``(r, c)`` in absolute pixel
space.

Notes on behavioral compatibility with deepBlink
-----------------------------------------------
- Grid assignment uses ``floor(coord) // cell_size``.
- Coordinates on a cell boundary belong to the cell to the "right/bottom" of
  that boundary, matching the original ``np.floor`` based logic.
- Relative offsets are normalized by ``cell_size``.
- Decoding uses a default threshold of ``0.5`` when ``probability is None``.
- When multiple coordinates fall into the same cell during encoding, later
  coordinates overwrite earlier ones, matching the original implementation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Union

import numpy as np


ArrayLikeImageSize = Union[int, Sequence[int], np.ndarray]


def next_power(x: int) -> int:
    """Return the next power of 2 greater than or equal to ``x``.

    This matches the behavior of ``deepblink.data.next_power(x, k=2)`` for the
    default base-2 case used throughout deepBlink inference padding.

    Note that this follows the original deepBlink implementation exactly:
    powers are generated starting from ``2 ** 1``. As a result,
    ``next_power(1)`` returns ``2`` rather than ``1``.

    Examples
    --------
    - ``next_power(1) -> 2``
    - ``next_power(5) -> 8``
    - ``next_power(16) -> 16``

    Parameters
    ----------
    x:
        Positive integer input.

    Returns
    -------
    int
        The smallest power of 2 such that ``power >= x``.

    Raises
    ------
    ValueError
        If ``x`` is smaller than 1.
    """
    if x < 1:
        raise ValueError(f"x must be >= 1, but got {x}.")

    y, power = 0, 1
    while y < x:
        y = 2 ** power
        power += 1
    return y


def _resolve_image_shape(image_size: ArrayLikeImageSize) -> tuple[int, int]:
    """Normalize image size input to ``(height, width)``.

    The original deepBlink implementation exposes separate ``image_size`` and
    optional ``size_c`` arguments. For the PyTorch migration, we accept either:

    - a scalar integer: square image of shape ``(image_size, image_size)``
    - a 2-item sequence / array: explicit ``(height, width)``

    Parameters
    ----------
    image_size:
        Either an integer side length or a 2-item shape-like object.

    Returns
    -------
    tuple[int, int]
        ``(height, width)``

    Raises
    ------
    ValueError
        If the input cannot be interpreted as image shape information.
    """
    if isinstance(image_size, (int, np.integer)):
        size = int(image_size)
        if size < 1:
            raise ValueError(f"image_size must be >= 1, but got {image_size}.")
        return size, size

    if isinstance(image_size, np.ndarray):
        values = image_size.tolist()
    else:
        values = list(image_size)

    if len(values) != 2:
        raise ValueError(
            "image_size must be either an integer or a 2-item sequence "
            f"(height, width), but got {image_size!r}."
        )

    height, width = int(values[0]), int(values[1])
    if height < 1 or width < 1:
        raise ValueError(
            f"image dimensions must be >= 1, but got ({height}, {width})."
        )
    return height, width


def get_prediction_matrix(
    coords: Union[np.ndarray, Iterable[Sequence[float]]],
    image_size: ArrayLikeImageSize,
    cell_size: int = 4,
) -> np.ndarray:
    """Encode absolute coordinates into a deepBlink prediction matrix.

    Each input coordinate ``(r, c)`` is assigned to exactly one grid cell. The
    output tensor stores, per cell:

    - channel 0: object presence indicator ``p`` (1 if occupied, else 0)
    - channel 1: relative row offset ``r_rel`` in the cell
    - channel 2: relative column offset ``c_rel`` in the cell

    The encoding formula is identical to the original deepBlink logic::

        cell_r = min(nrow - 1, floor(r) // cell_size)
        cell_c = min(ncol - 1, floor(c) // cell_size)

        r_rel = (r - cell_r * cell_size) / cell_size
        c_rel = (c - cell_c * cell_size) / cell_size

    This means:
    - grid ownership is decided using ``floor``
    - coordinates on an exact cell boundary belong to the next cell
      (except at the image edge, where indices are clipped)
    - later coordinates overwrite earlier coordinates if they map to the same
      cell, matching the original implementation

    Parameters
    ----------
    coords:
        Absolute pixel coordinates with shape ``(N, 2)`` in ``(r, c)`` order.
    image_size:
        Image size. Either:
        - integer for square images, or
        - ``(height, width)`` sequence for rectangular images.
    cell_size:
        Size of one grid cell in pixels.

    Returns
    -------
    np.ndarray
        Prediction matrix of shape ``(ceil(H / cell_size), ceil(W / cell_size), 3)``.

    Raises
    ------
    ValueError
        If ``cell_size`` is invalid or coordinate input shape is invalid.
    """
    if cell_size < 1:
        raise ValueError(f"cell_size must be >= 1, but got {cell_size}.")

    coords_array = np.asarray(coords, dtype=np.float32)
    if coords_array.size == 0:
        coords_array = coords_array.reshape(0, 2)

    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(
            "coords must have shape (N, 2) in (r, c) order, "
            f"but got shape {coords_array.shape}."
        )

    height, width = _resolve_image_shape(image_size)
    nrow = math.ceil(height / cell_size)
    ncol = math.ceil(width / cell_size)

    prediction_matrix = np.zeros((nrow, ncol, 3), dtype=np.float32)

    for r, c in coords_array:
        cell_r = min(nrow - 1, int(np.floor(r)) // cell_size)
        cell_c = min(ncol - 1, int(np.floor(c)) // cell_size)

        relative_r = (r - cell_r * cell_size) / cell_size
        relative_c = (c - cell_c * cell_size) / cell_size

        prediction_matrix[cell_r, cell_c] = (1.0, relative_r, relative_c)

    return prediction_matrix


def absolute_coordinate(
    pred_matrix: np.ndarray,
    cell_size: int = 4,
    probability: float | None = None,
) -> np.ndarray:
    """Decode a deepBlink prediction matrix into absolute ``(r, c)`` coordinates.

    This function is the inverse of ``get_prediction_matrix`` at the protocol
    level. It scans the presence/probability channel and converts all cells with
    ``p > threshold`` back into absolute coordinates.

    The threshold behavior matches original deepBlink inference semantics:
    - if ``probability is None``, use a default threshold of ``0.5``
    - otherwise use the provided value directly

    Decoding follows the original formula::

        r = cell_r * cell_size + r_rel * cell_size
        c = cell_c * cell_size + c_rel * cell_size

    Parameters
    ----------
    pred_matrix:
        Prediction matrix with shape ``(H_grid, W_grid, 3)``.
    cell_size:
        Size of one grid cell in pixels.
    probability:
        Presence threshold. ``None`` defaults to ``0.5``.

    Returns
    -------
    np.ndarray
        Absolute coordinates with shape ``(N, 2)`` in ``(r, c)`` order. The
        dtype is ``float32``.

    Raises
    ------
    ValueError
        If ``pred_matrix`` does not have shape ``(H_grid, W_grid, 3)`` or if
        ``cell_size`` is invalid.
    """
    if cell_size < 1:
        raise ValueError(f"cell_size must be >= 1, but got {cell_size}.")

    matrix = np.asarray(pred_matrix, dtype=np.float32)
    if matrix.ndim != 3:
        raise ValueError("pred_matrix must have shape (H_grid, W_grid, 3).")
    if matrix.shape[2] != 3:
        raise ValueError("pred_matrix must have depth 3 in the last dimension.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            "pred_matrix spatial dimensions must be >= 1, "
            f"but got {matrix.shape[:2]}."
        )

    threshold = 0.5 if probability is None else probability

    coords_r = []
    coords_c = []

    matrix_r, matrix_c = np.where(matrix[..., 0] > threshold, 1, 0).nonzero()
    for r_idx, c_idx in zip(matrix_r, matrix_c):
        grid_r = r_idx * cell_size
        grid_c = c_idx * cell_size
        spot_r = matrix[r_idx, c_idx, 1]
        spot_c = matrix[r_idx, c_idx, 2]

        coord_abs_r = grid_r + spot_r * cell_size
        coord_abs_c = grid_c + spot_c * cell_size

        coords_r.append(coord_abs_r)
        coords_c.append(coord_abs_c)

    if not coords_r:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray([coords_r, coords_c], dtype=np.float32).T


def get_coordinate_list(
    pred: np.ndarray,
    cell_size: int = 4,
    probability: float | None = None,
) -> np.ndarray:
    """Return decoded coordinates from a prediction matrix.

    This is a thin compatibility wrapper around ``absolute_coordinate``. It is
    intentionally kept separate because the original deepBlink code exposes both
    a helper for coordinate reconstruction and a public decoding function.

    Parameters
    ----------
    pred:
        Prediction matrix with shape ``(H_grid, W_grid, 3)``.
    cell_size:
        Size of one grid cell in pixels.
    probability:
        Presence threshold. ``None`` defaults to ``0.5``.

    Returns
    -------
    np.ndarray
        Absolute coordinates with shape ``(N, 2)`` in ``(r, c)`` order.
    """
    return absolute_coordinate(
        pred_matrix=pred,
        cell_size=cell_size,
        probability=probability,
    )
