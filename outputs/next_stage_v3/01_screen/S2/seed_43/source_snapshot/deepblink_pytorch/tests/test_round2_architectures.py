import numpy as np
import pytest
import torch

from deepblink_torch.networks.unet import UNetDeepBlink
from deepblink_torch.networks.detail_route import UNetDeepLociDetailRoute, phase_features
from deepblink_torch.networks.adaptive_context import UNetDeepLociAdaptiveContext, ExcludedCenterContext
from deepblink_torch.networks.dense_heatmap import UNetDeepLociHeatmap
from deepblink_torch.cli import predict
from deepblink_torch.losses import combined_dice_rmse_loss


def test_phase_rearrangement_preserves_location_and_channel():
    image = torch.arange(2*8*8).reshape(1, 2, 8, 8).float()
    phases = phase_features(image)
    for iy in range(4):
        for ix in range(4):
            assert torch.equal(phases[0, iy*4+ix], image[0, :, iy::4, ix::4])


@pytest.mark.parametrize("factory", [
    lambda: UNetDeepLociDetailRoute(selection="attention"),
    lambda: UNetDeepLociDetailRoute(selection="projection"),
    lambda: UNetDeepLociAdaptiveContext(adaptive=True),
    lambda: UNetDeepLociAdaptiveContext(adaptive=False)])
def test_zero_residual_matches_baseline_and_empty_mask_is_finite(factory):
    torch.manual_seed(13)
    base = UNetDeepBlink().eval()
    torch.manual_seed(13)
    model = factory().eval()
    for name, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    image = torch.randn(1, 1, 32, 32)
    mask = torch.zeros_like(image)
    mask[..., :16, :16] = 1
    with torch.no_grad():
        assert torch.equal(base(image), model(image, valid_mask=mask))
        assert torch.isfinite(model(image, valid_mask=torch.zeros_like(image))).all()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    target = torch.zeros(1, 8, 8, 3)
    target[0, 2, 2] = torch.tensor([1., .2, .7])
    for _ in range(2):
        optimizer.zero_grad()
        loss = combined_dice_rmse_loss(target, model(image, valid_mask=mask))
        loss.backward()
        optimizer.step()
    branch = model.detail_projection if hasattr(model, "detail_projection") else model.context_projection
    assert branch.weight.grad is not None
    assert torch.isfinite(branch.weight.grad).all() and torch.count_nonzero(branch.weight.grad) > 0


def test_matched_parameter_controls_and_registered_size_goals():
    count = lambda model: sum(p.numel() for p in model.parameters())
    base = count(UNetDeepBlink())
    r1, s1 = [count(UNetDeepLociDetailRoute(selection=s)) for s in ("attention", "projection")]
    r2, s2 = [count(UNetDeepLociAdaptiveContext(adaptive=s)) for s in (True, False)]
    assert abs((r1-base)-(s1-base))/(r1-base) < .10
    assert r2 == s2
    assert r1/base <= 1.05 and r2/base <= 1.10


def test_center_exclusion_and_no_support_have_defined_gradients():
    context = ExcludedCenterContext(2, 5)
    feature = torch.randn(1, 2, 16, 16, requires_grad=True)
    output, _ = context(feature, torch.ones(1, 1, 16, 16))
    output.square().mean().backward()
    assert torch.count_nonzero(context.weight.grad[..., 1:4, 1:4]) == 0
    empty, coverage = context(feature, torch.zeros(1, 1, 16, 16))
    assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0
    assert torch.count_nonzero(coverage) == 0


@pytest.mark.parametrize("name,cell_size", [
    ("deeploci_detail", 4), ("deeploci_detail_simple", 4), ("deeploci_heatmap", 1),
    ("deeploci_adaptive_context", 4), ("deeploci_fixed_context", 4)])
def test_prediction_checkpoint_round_trip(tmp_path, name, cell_size):
    model = predict._build_model_for_checkpoint(name, cell_size=cell_size, device=torch.device("cpu")).eval()
    path = tmp_path / "model.pt"
    torch.save(dict(model_name=name, cell_size=cell_size, input_size=512, model_state_dict=model.state_dict()), path)
    restored, restored_cell = predict._load_model(path, torch.device("cpu"))
    assert restored_cell == cell_size
    image = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        actual = predict._forward_with_geometry(restored, image, 16, 16)
        expected = predict._forward_with_geometry(model, image, 16, 16)
    assert torch.equal(actual, expected)


def test_dense_reference_removes_unused_second_encoder():
    model = UNetDeepLociHeatmap()
    assert not hasattr(model, "encoders_v2") and not hasattr(model, "logit_block")
    assert model(torch.zeros(1, 1, 32, 32)).shape == (1, 32, 32, 3)
