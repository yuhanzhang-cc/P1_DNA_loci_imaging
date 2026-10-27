"""Geometry, gradients, compatibility, and zero-residual reference checks."""
import numpy as np
import pytest
import torch
from deepblink_torch.networks.unet import UNetDeepBlink
from deepblink_torch.networks.evidence import RadialEvidence, UNetDeepLociEvidence, UNetDeepLociLocal
from deepblink_torch.cli import train, predict


def test_zero_residual_reproduces_backbone_and_loads_checkpoint(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base = UNetDeepBlink(dropout=0).eval()
    model = UNetDeepLociEvidence(dropout=0).eval()
    result = model.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith(("evidence_reader.", "feature_projection.", "evidence_fusion.")) for k in result.missing_keys)
    image = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        expected = base(image)
        actual = model(image, valid_mask=torch.ones_like(image))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    path = tmp_path / "model.pt"
    torch.save(dict(model_state_dict=model.state_dict(), model_name="deeploci_evidence",
                    cell_size=4, input_size=32), path)
    loaded, _ = predict._load_model(path, torch.device("cpu"))
    with torch.no_grad():
        torch.testing.assert_close(predict._forward_with_geometry(loaded, image, 32, 32), actual)
    with pytest.raises(ValueError, match="explicit"):
        loaded(image)


def test_reader_excludes_padding_and_has_offset_gradients():
    torch.manual_seed(17)
    image = torch.randn(1, 1, 32, 32)
    mask = torch.zeros_like(image)
    mask[..., :16, :16] = 1
    other = image.clone()
    other[..., 16:, :] = 1000
    other[..., :, 16:] = -1000
    offsets = torch.full((1, 2, 8, 8), .4, requires_grad=True)
    reader = RadialEvidence()
    output = reader(image, offsets, mask)
    torch.testing.assert_close(output, reader(other, offsets, mask), rtol=0, atol=0)
    assert output.shape == (1, 12, 8, 8)
    output.square().mean().backward()
    assert torch.isfinite(offsets.grad).all() and offsets.grad.abs().sum() > 0


def test_training_mask_uses_only_geometry():
    class Capture(torch.nn.Module):
        requires_valid_mask = True
        def forward(self, image, valid_mask):
            return valid_mask
    targets = torch.zeros(1, 8, 8, 3)
    targets[:, 4:, :, 0] = -1
    image = torch.zeros(1, 1, 32, 32)
    a = train._forward_training(Capture(), image, targets)
    targets[:, :4, :, 0] = 1
    b = train._forward_training(Capture(), image, targets)
    torch.testing.assert_close(a, b)
    assert a.sum() == 512


@pytest.mark.parametrize("name", ["deeploci_local", "deeploci_evidence"])
def test_new_architecture_backward_and_metadata(name):
    torch.set_num_threads(2)
    model = train._build_model(name, cell_size=4, device=torch.device("cpu"))
    image = torch.randn(2, 1, 32, 32)
    targets = torch.zeros(2, 8, 8, 3)
    targets[:, 2, 2] = torch.tensor([1, .4, .6])
    output = train._forward_training(model, image, targets)
    assert output.shape == targets.shape
    loss = train.combined_dice_rmse_loss(targets, output)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert train._architecture_config(name, model)["model_name"] == name
