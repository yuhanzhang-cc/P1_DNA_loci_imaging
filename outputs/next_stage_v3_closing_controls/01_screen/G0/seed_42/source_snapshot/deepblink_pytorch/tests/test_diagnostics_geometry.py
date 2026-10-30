"""Legacy diagnostics must recover original dimensions and pass geometry."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import error_context_diagnostics as diagnostics


def test_archive_fallback_crops_before_normalization(tmp_path):
    archive = tmp_path / "data.npz"
    labels = np.empty(1, dtype=object)
    labels[0] = np.array([[2., 3.]], dtype=np.float32)
    raw = np.zeros((1, 16, 16), dtype=np.float32)
    raw[0, :8, :8] = np.arange(64).reshape(8, 8)
    np.savez(archive, x_valid=raw, y_valid=labels, sample_id_valid=["sample_a"],
             image_name_valid=["sample_a.tif"], original_shape_valid=[[8, 8]])
    samples = diagnostics._load_samples(archive, "valid", None)
    assert samples[0].image.shape == (8, 8)
    np.testing.assert_array_equal(samples[0].image, raw[0, :8, :8])


def test_dense_inference_passes_original_geometry(monkeypatch):
    class MaskModel(torch.nn.Module):
        requires_valid_mask = True
        input_size = 16
        def forward(self, image, valid_mask):
            assert image.shape == valid_mask.shape == (1, 1, 16, 16)
            assert valid_mask.sum() == 64
            torch.testing.assert_close(image[..., :8, :8].mean(), torch.tensor(0.), atol=1e-6, rtol=0)
            result = torch.full((1, 4, 4, 3), .5)
            result[..., 0] = .9
            return result
    monkeypatch.setattr(diagnostics.predict_cli, "_load_model", lambda *args: (MaskModel(), 4))
    sample = diagnostics.SampleData("sample_a", "sample_a.tif", np.arange(64,dtype=np.float32).reshape(8,8),
                                    np.array([[2.,3.]],dtype=np.float32))
    predictions, _ = diagnostics._run_dense_inference([sample], Path("unused.pt"), torch.device("cpu"))
    assert len(predictions[0].dense_candidates) == 4
    assert (predictions[0].dense_candidates[:, :2] < 8).all()
