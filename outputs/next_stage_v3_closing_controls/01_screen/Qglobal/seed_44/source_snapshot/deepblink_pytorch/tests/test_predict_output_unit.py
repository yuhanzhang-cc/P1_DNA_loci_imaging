import json

import numpy as np

from deepblink_torch.cli.predict import _convert_output_unit
from deepblink_torch.cli.predict import _write_csv
from deepblink_torch.cli.predict import _write_prediction_metadata
from deepblink_torch.utils.pixel_size import PixelSize


def test_output_unit_px_keeps_coordinates_unchanged() -> None:
    coords_with_p = np.asarray([[10.5, 12.25, 0.9]], dtype=np.float32)
    pixel_size = PixelSize(x=1.0, y=1.0, source="default_1.0")
    converted = _convert_output_unit(coords_with_p, pixel_size)

    np.testing.assert_allclose(converted, coords_with_p)


def test_output_unit_micron_scales_y_and_x_independently() -> None:
    coords_with_p = np.asarray([[10.0, 12.0, 0.9]], dtype=np.float32)
    pixel_size = PixelSize(x=0.2, y=0.5, source="imagej")
    converted = _convert_output_unit(coords_with_p, pixel_size)

    np.testing.assert_allclose(converted, np.asarray([[5.0, 2.4, 0.9]], dtype=np.float32))


def test_write_csv_labels_micron_columns(tmp_path) -> None:
    csv_path = tmp_path / "predictions.csv"
    coords_with_p = np.asarray([[5.0, 2.4, 0.9]], dtype=np.float32)

    _write_csv(csv_path, coords_with_p, output_unit="micron")

    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "y [um],x [um],p"


def test_write_csv_labels_px_columns(tmp_path) -> None:
    csv_path = tmp_path / "predictions.csv"
    coords_with_p = np.asarray([[10.5, 12.25, 0.9]], dtype=np.float32)

    _write_csv(csv_path, coords_with_p, output_unit="px")

    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "y [px],x [px],p"


def test_write_prediction_metadata_sidecar(tmp_path) -> None:
    csv_path = tmp_path / "predictions.csv"
    image_path = tmp_path / "sample_1.tif"
    model_path = tmp_path / "best_model.pt"

    _write_prediction_metadata(
        csv_path=csv_path,
        image_path=image_path,
        model_path=model_path,
        output_unit="px",
    )

    metadata_path = csv_path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "csv_file_name": "predictions.csv",
        "image_file_name": "sample_1.tif",
        "model_file_name": "best_model.pt",
        "output_unit": "px",
    }
