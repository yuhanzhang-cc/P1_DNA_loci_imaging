from pathlib import Path
import math

from deepblink_torch.utils.pixel_size import PixelSizeReadError
from deepblink_torch.utils.pixel_size import read_pixel_size_from_image
from deepblink_torch.utils.pixel_size import read_pixel_size_or_default


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIF_DIR = PROJECT_ROOT / "data" / "raw" / "tif"


def test_read_pixel_size_from_imagej_tif() -> None:
    pixel_size = read_pixel_size_from_image(TIF_DIR / "sample_4.tif")
    assert pixel_size.source == "imagej"
    assert math.isclose(pixel_size.x, 0.133000018, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(pixel_size.y, 0.133000018, rel_tol=1e-6, abs_tol=1e-6)


def test_read_pixel_size_missing_metadata_raises() -> None:
    try:
        read_pixel_size_from_image(TIF_DIR / "sample_100.tif")
    except PixelSizeReadError:
        return
    raise AssertionError("Expected PixelSizeReadError for missing metadata.")


def test_read_pixel_size_or_default_falls_back_to_one() -> None:
    pixel_size = read_pixel_size_or_default(TIF_DIR / "sample_100.tif")
    assert pixel_size.source == "default_1.0"
    assert math.isclose(pixel_size.x, 1.0)
    assert math.isclose(pixel_size.y, 1.0)
