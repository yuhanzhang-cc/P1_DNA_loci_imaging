from __future__ import annotations

import torch

from deepblink_torch import combined_dice_focal_rmse_loss
from deepblink_torch import combined_dice_rmse_loss
from deepblink_torch import combined_focal_rmse_loss
from deepblink_torch import focal_loss
from deepblink_torch import rmse_loss


def _targets() -> torch.Tensor:
    targets = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    targets[0, 1, 2] = torch.tensor([1.0, 0.25, 0.50])
    targets[1, 3, 0] = torch.tensor([1.0, 0.75, 0.25])
    return targets


def _predictions() -> torch.Tensor:
    preds = torch.full((2, 4, 4, 3), 0.20, dtype=torch.float32)
    preds[..., 1:] = 0.0
    preds[0, 1, 2] = torch.tensor([0.80, 0.30, 0.45])
    preds[1, 3, 0] = torch.tensor([0.70, 0.65, 0.35])
    return preds.requires_grad_()


def test_loss_variants_are_finite_and_backward() -> None:
    targets = _targets()
    loss_functions = [
        combined_dice_rmse_loss,
        combined_focal_rmse_loss,
        combined_dice_focal_rmse_loss,
    ]

    for loss_function in loss_functions:
        preds = _predictions()
        loss = loss_function(targets, preds)

        assert torch.isfinite(loss)
        loss.backward()
        assert preds.grad is not None
        assert torch.all(torch.isfinite(preds.grad))


def test_focal_loss_uses_probability_channel_only() -> None:
    targets = _targets()
    preds = _predictions().detach()
    preds_with_changed_offsets = preds.clone()
    preds_with_changed_offsets[..., 1:] = 99.0

    assert torch.isclose(
        focal_loss(targets, preds),
        focal_loss(targets, preds_with_changed_offsets),
    )


def test_rmse_ignores_background_offsets() -> None:
    targets = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    targets[0, 0, 0] = torch.tensor([1.0, 0.25, 0.50])

    preds = torch.full((1, 2, 2, 3), 0.40, dtype=torch.float32)
    preds[0, 0, 0, 1:] = torch.tensor([0.25, 0.50])

    preds_with_background_offsets = preds.clone()
    preds_with_background_offsets[0, 1, 1, 1:] = torch.tensor([99.0, 99.0])

    assert torch.isclose(rmse_loss(targets, preds), rmse_loss(targets, preds_with_background_offsets))
    assert torch.isclose(
        combined_focal_rmse_loss(targets, preds),
        combined_focal_rmse_loss(targets, preds_with_background_offsets),
    )


def test_rmse_penalizes_positive_cell_offsets() -> None:
    targets = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    targets[0, 0, 0] = torch.tensor([1.0, 0.25, 0.50])

    preds_good = torch.full((1, 2, 2, 3), 0.40, dtype=torch.float32)
    preds_good[0, 0, 0, 1:] = torch.tensor([0.25, 0.50])
    preds_bad = preds_good.clone()
    preds_bad[0, 0, 0, 1:] = torch.tensor([0.95, 0.05])

    assert rmse_loss(targets, preds_bad) > rmse_loss(targets, preds_good)
