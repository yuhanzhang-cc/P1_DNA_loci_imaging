"""D2 feature sampling and A geometry-only statistics and compatibility."""
import torch
import pytest
from deepblink_torch.cli import train, predict
from deepblink_torch.networks.unet import UNetDeepBlink
from deepblink_torch.networks.feature_evidence import LearnedRadialEvidence, UNetDeepLociFeatureEvidence
from deepblink_torch.networks.masked_se import GeometricSqueezeBlock, UNetDeepBlinkMaskedSE
from deepblink_torch.networks._networks import SqueezeBlock
from deepblink_torch.losses import combined_dice_rmse_loss


def test_masked_se_gate_excludes_arbitrary_invalid_features():
    torch.manual_seed(42)
    source = SqueezeBlock(8)
    masked = GeometricSqueezeBlock(source)
    features = torch.randn(1, 8, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., :8, :8] = 1
    altered = features.clone()
    altered[..., 8:, :] = 10000
    altered[..., :, 8:] = -10000
    torch.testing.assert_close(masked.weights_for(features, mask), masked.weights_for(altered, mask), rtol=0, atol=0)
    full = torch.ones_like(mask)
    masked.valid_mask = full
    torch.testing.assert_close(masked(features), source(features), rtol=1e-5, atol=1e-6)


def test_all_valid_masked_model_preserves_parameters_and_reference():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base = UNetDeepBlink(dropout=0).eval()
    torch.manual_seed(42)
    model = UNetDeepBlinkMaskedSE(dropout=0).eval()
    assert set(model.state_dict()) == set(base.state_dict())
    for key in base.state_dict():
        torch.testing.assert_close(base.state_dict()[key], model.state_dict()[key], rtol=0, atol=0)
    image = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(image, valid_mask=torch.ones_like(image)), base(image), rtol=1e-5, atol=1e-6)
    assert all(block.valid_mask is None for block in model.modules() if isinstance(block, GeometricSqueezeBlock))
    assert sum(p.numel() for p in model.parameters()) == 1258723


def test_learned_feature_contrast_constant_field_and_padding():
    reader = LearnedRadialEvidence()
    fields = torch.full((1, 8, 32, 32), 3.)
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., :16, :16] = 1
    offsets = torch.full((1, 2, 8, 8), .4)
    original = reader.read_features(fields, offsets, mask)
    assert original.shape == (1, 51, 8, 8)
    torch.testing.assert_close(original[:, :24], torch.zeros_like(original[:, :24]), atol=1e-5, rtol=0)
    fields[..., 16:, :] = 1000
    fields[..., :, 16:] = -1000
    torch.testing.assert_close(original, reader.read_features(fields, offsets, mask), rtol=0, atol=0)


def test_feature_model_zero_residual_and_learnable_embedding():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base = UNetDeepBlink(dropout=0).eval()
    model = UNetDeepLociFeatureEvidence(dropout=0).eval()
    loaded = model.load_state_dict(base.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    image = torch.randn(1, 1, 32, 32)
    mask = torch.ones_like(image)
    with torch.no_grad():
        torch.testing.assert_close(model(image, valid_mask=mask), base(image), rtol=0, atol=0)
        model.evidence_fusion[-1].weight.fill_(.02)
    target = torch.zeros(1, 8, 8, 3)
    target[:, 3, 3] = torch.tensor([1., .4, .6])
    loss = combined_dice_rmse_loss(target, model(image, valid_mask=mask))
    loss.backward()
    grads = [p.grad for p in model.evidence_reader.embedding.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize("name", ["deeploci_evidence", "deeploci_local", "deeploci_feature_evidence", "deepblink_masked_se"])
def test_registered_checkpoint_roundtrip_backward_and_geometry(name, tmp_path):
    torch.set_num_threads(2)
    model = train._build_model(name, cell_size=4, device=torch.device("cpu")).eval()
    image = torch.randn(2, 1, 32, 32)
    target = torch.zeros(2, 8, 8, 3)
    target[:, 2, 2] = torch.tensor([1., .4, .6])
    target[0, 4:, :, 0] = -1
    target[0, :, 4:, 0] = -1
    output = train._forward_training(model, image, target)
    combined_dice_rmse_loss(target, output).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    metadata = train._architecture_config(name, model)
    assert metadata["model_name"] == name
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(model_name=name, model_state_dict=model.state_dict(), cell_size=4,
                    input_size=32, architecture_config=metadata), checkpoint)
    restored, _ = predict._load_model(checkpoint, torch.device("cpu"))
    with torch.no_grad():
        actual = predict._forward_with_geometry(restored, image[:1], 16, 16)
    torch.testing.assert_close(actual, output[:1], rtol=1e-5, atol=1e-6)
