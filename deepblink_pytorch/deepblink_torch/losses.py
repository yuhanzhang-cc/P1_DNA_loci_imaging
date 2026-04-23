"""Loss functions for the PyTorch deepBlink migration.

This module mirrors the TensorFlow deepBlink loss protocol while operating on
PyTorch tensors with channel-last layout:

- ``(B, H, W, 3)``

The three channels are interpreted as:

- channel 0: probability / occupancy
- channel 1: relative row offset
- channel 2: relative column offset

Notes
-----
- ``dice_loss`` is applied to the probability channel only.
- ``rmse_loss`` is applied to the offset channels only.
- ``combined_dice_rmse_loss`` follows the original deepBlink weighting:
  ``dice_loss + 2 * rmse_loss``.
"""

from __future__ import annotations

import torch


def _validate_prediction_shape(y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    """Validate that tensors follow the deepBlink channel-last protocol."""
    if y_true.ndim != 4 or y_pred.ndim != 4:
        raise ValueError(
            "y_true and y_pred must have shape (B, H, W, 3), "
            f"but got y_true={tuple(y_true.shape)}, y_pred={tuple(y_pred.shape)}."
        )
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes, "
            f"but got y_true={tuple(y_true.shape)}, y_pred={tuple(y_pred.shape)}."
        )
    if y_true.shape[-1] != 3:
        raise ValueError(
            "The last dimension must have size 3 for (p, r_rel, c_rel), "
            f"but got {tuple(y_true.shape)}."
        )


def dice_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute the Dice coefficient on channel-last tensors.

    This matches the original deepBlink implementation:

    ``Dice = (2 * intersection + smooth) / (sum(y_true) + sum(y_pred) + smooth)``

    Parameters
    ----------
    y_true:
        Ground-truth probability mask, typically ``y_true[..., 0]``.
    y_pred:
        Predicted probability mask, typically ``y_pred[..., 0]``.
    smooth:
        Constant added to numerator and denominator to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Scalar Dice score tensor.
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth
    )
    return dice


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Dice loss on the probability channel.

    Parameters
    ----------
    y_true:
        Ground-truth tensor with shape ``(B, H, W, 3)``.
    y_pred:
        Predicted tensor with shape ``(B, H, W, 3)``.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor equal to ``1 - dice_score`` on ``[..., 0]``.
    """
    _validate_prediction_shape(y_true, y_pred)
    return 1.0 - dice_score(y_true[..., 0], y_pred[..., 0])


def rmse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute deepBlink-style RMSE over relative offsets.

    This mirrors the original TensorFlow implementation in
    ``deepblink.losses.rmse``:

    1. Select offset channels ``[..., 1:]``.
    2. Build a mask by checking where the true offsets are exactly zero.
    3. Zero out both true and predicted offsets at masked positions.
    4. Count positive cells via ``count_nonzero(sum(y_true_offsets, axis=-1))``.
    5. Compute the square root of the average summed squared displacement.

    The intended semantics are:
    - regression is only evaluated for cells containing a true spot
    - background cells do not contribute to offset regression

    Parameters
    ----------
    y_true:
        Ground-truth tensor with shape ``(B, H, W, 3)``.
    y_pred:
        Predicted tensor with shape ``(B, H, W, 3)``.

    Returns
    -------
    torch.Tensor
        Scalar RMSE tensor.
    """
    _validate_prediction_shape(y_true, y_pred)

    y_true_offsets = y_true[..., 1:]
    y_pred_offsets = y_pred[..., 1:]

    comparison = torch.eq(y_true_offsets, torch.tensor(0.0, dtype=y_true.dtype, device=y_true.device))

    y_true_masked = torch.where(comparison, torch.zeros_like(y_true_offsets), y_true_offsets)
    y_pred_masked = torch.where(comparison, torch.zeros_like(y_pred_offsets), y_pred_offsets)

    sum_rc_coords = torch.sum(y_true_offsets, dim=-1)
    n_true_spots = torch.count_nonzero(sum_rc_coords).to(y_true.dtype)

    squared_displacement_xy_summed = torch.sum(
        torch.square(y_true_masked - y_pred_masked),
        dim=-1,
    )
    rmse_value = torch.sqrt(
        torch.sum(squared_displacement_xy_summed)
        / (n_true_spots + torch.finfo(y_true.dtype).eps)
    )
    return rmse_value


def combined_dice_rmse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:
    """Combine Dice loss and RMSE loss using the original deepBlink weighting.

    Parameters
    ----------
    y_true:
        Ground-truth tensor with shape ``(B, H, W, 3)``.
    y_pred:
        Predicted tensor with shape ``(B, H, W, 3)``.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor equal to:
        ``dice_loss(y_true, y_pred) + 2 * rmse_loss(y_true, y_pred)``.
    """
    return dice_loss(y_true, y_pred) + rmse_loss(y_true, y_pred) * 2.0
