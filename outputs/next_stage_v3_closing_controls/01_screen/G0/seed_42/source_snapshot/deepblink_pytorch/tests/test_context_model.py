from __future__ import annotations

from pathlib import Path

import torch

from deepblink_torch import UNetDeepBlink
from deepblink_torch import UNetDeepLociContext
from deepblink_torch import combined_dice_rmse_loss
from deepblink_torch.cli import predict as predict_cli


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def test_context_model_matches_baseline_output_shape() -> None:
    inputs = torch.randn((2, 1, 32, 32), dtype=torch.float32)
    baseline = UNetDeepBlink(cell_size=4).eval()
    context = UNetDeepLociContext(cell_size=4).eval()

    with torch.no_grad():
        baseline_output = baseline(inputs)
        context_output = context(inputs)

    assert baseline_output.shape == (2, 8, 8, 3)
    assert context_output.shape == baseline_output.shape
    assert torch.all((context_output >= 0.0) & (context_output <= 1.0))


def test_context_model_backward_with_baseline_loss_is_finite() -> None:
    model = UNetDeepLociContext(cell_size=4)
    inputs = torch.randn((1, 1, 32, 32), dtype=torch.float32)
    predictions = model(inputs)
    targets = torch.zeros_like(predictions)
    targets[0, 2, 3] = torch.tensor([1.0, 0.25, 0.75])

    loss = combined_dice_rmse_loss(targets, predictions)

    assert torch.isfinite(loss)
    loss.backward()
    finite_grads = [
        torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert finite_grads
    assert all(bool(is_finite) for is_finite in finite_grads)


def test_context_model_adds_only_lightweight_parameters() -> None:
    baseline_params = _parameter_count(UNetDeepBlink(cell_size=4))
    context_params = _parameter_count(UNetDeepLociContext(cell_size=4))

    assert baseline_params == 1_258_723
    assert context_params == 1_304_771
    assert context_params - baseline_params == 46_048


def test_load_model_keeps_old_checkpoint_compatible(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "old_best_model.pt"
    baseline = UNetDeepBlink(cell_size=4)
    torch.save(
        {
            "model_state_dict": baseline.state_dict(),
            "cell_size": 4,
        },
        checkpoint_path,
    )

    loaded, cell_size = predict_cli._load_model(checkpoint_path, torch.device("cpu"))

    assert type(loaded) is UNetDeepBlink
    assert cell_size == 4


def test_load_model_uses_checkpoint_model_name_for_context(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "context_best_model.pt"
    context = UNetDeepLociContext(cell_size=4)
    torch.save(
        {
            "model_state_dict": context.state_dict(),
            "cell_size": 4,
            "model_name": "deeploci_context",
        },
        checkpoint_path,
    )

    loaded, cell_size = predict_cli._load_model(checkpoint_path, torch.device("cpu"))

    assert type(loaded) is UNetDeepLociContext
    assert cell_size == 4
