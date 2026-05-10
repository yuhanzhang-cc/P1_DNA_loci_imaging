from pathlib import Path

import numpy as np
import tifffile

from deepblink_torch.cli.overlay import _build_footer_text
from deepblink_torch.cli.overlay import _pair_images_and_csvs
from deepblink_torch.cli.overlay import _read_prediction_metadata
from deepblink_torch.cli.overlay import _read_coords


def test_overlay_pairs_images_and_csvs_by_stem(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    csv_dir = tmp_path / "predictions"
    images_dir.mkdir()
    csv_dir.mkdir()

    tifffile.imwrite(images_dir / "sample_1.tif", np.zeros((4, 4), dtype=np.uint8))
    (csv_dir / "sample_1.csv").write_text(
        "y [px],x [px],p\n1,2,0.9\n",
        encoding="utf-8",
    )

    pairs = _pair_images_and_csvs(images_dir, csv_dir)

    assert pairs == [(images_dir / "sample_1.tif", csv_dir / "sample_1.csv")]
    coords = _read_coords(csv_dir / "sample_1.csv")
    np.testing.assert_allclose(coords, np.asarray([[1.0, 2.0]], dtype=np.float32))


def test_overlay_rejects_micron_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample_1.csv"
    csv_path.write_text("y [um],x [um],p\n1.0,2.0,0.9\n", encoding="utf-8")

    try:
        _read_coords(csv_path)
    except ValueError as exc:
        assert "pixel-coordinate CSV" in str(exc)
        return

    raise AssertionError("Expected micron CSV to be rejected.")


def test_overlay_uses_prediction_metadata_for_footer(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample_1.csv"
    metadata_path = csv_path.with_suffix(".meta.json")
    metadata_path.write_text(
        '{"image_file_name": "sample_1.tif", "model_file_name": "best_model.pt"}',
        encoding="utf-8",
    )

    metadata = _read_prediction_metadata(csv_path)
    footer = _build_footer_text(
        image_name=metadata["image_file_name"],
        model_name=metadata["model_file_name"],
    )

    assert "sample_1.tif" in footer
    assert "best_model.pt" in footer
