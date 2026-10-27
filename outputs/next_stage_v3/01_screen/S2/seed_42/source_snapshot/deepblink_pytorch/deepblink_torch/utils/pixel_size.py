"""Read physical pixel sizes from image metadata."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tifffile


@dataclass(frozen=True)
class PixelSize:
    """Physical pixel size in microns per pixel."""

    x: float
    y: float
    source: str


class PixelSizeReadError(ValueError):
    """Raised when an image has no usable physical pixel-size metadata."""


UNIT_TO_MICRON = {
    "um": 1.0,
    "\xb5m": 1.0,
    "\u03bcm": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "micrometre": 1.0,
    "micrometres": 1.0,
    "nm": 0.001,
    "nanometer": 0.001,
    "nanometers": 0.001,
    "nanometre": 0.001,
    "nanometres": 0.001,
    "mm": 1000.0,
    "millimeter": 1000.0,
    "millimeters": 1000.0,
    "millimetre": 1000.0,
    "millimetres": 1000.0,
}


def _rational_to_float(value: Any) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])

    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator is not None:
        return float(numerator) / float(denominator)

    return float(value)


def _parse_description(description: str | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if not description:
        return values

    for line in description.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()
    return values


def _unit_to_micron(unit: str | None) -> float | None:
    if unit is None:
        return None
    return UNIT_TO_MICRON.get(unit.strip().lower())


def _standard_resolution_unit_to_micron(unit_value: Any) -> float | None:
    name = getattr(unit_value, "name", str(unit_value)).strip().lower()
    value = getattr(unit_value, "value", unit_value)

    if value == 2 or name in {"inch", "in"}:
        return 25400.0
    if value == 3 or name in {"centimeter", "centimetre", "cm"}:
        return 10000.0
    return None


def _pixel_size_from_resolution(
    x_resolution: Any,
    y_resolution: Any,
    micron_per_unit: float,
) -> tuple[float, float]:
    x_pixels_per_unit = _rational_to_float(x_resolution)
    y_pixels_per_unit = _rational_to_float(y_resolution)
    if x_pixels_per_unit <= 0 or y_pixels_per_unit <= 0:
        raise PixelSizeReadError("resolution tags must be positive")
    return micron_per_unit / x_pixels_per_unit, micron_per_unit / y_pixels_per_unit


def _read_imagej_pixel_size(tif: tifffile.TiffFile) -> PixelSize | None:
    page = tif.pages[0]
    tags = page.tags
    x_resolution = tags.get("XResolution")
    y_resolution = tags.get("YResolution")
    if x_resolution is None or y_resolution is None:
        return None

    description = page.description or ""
    description_values = _parse_description(description)
    imagej_metadata = tif.imagej_metadata or {}
    unit = imagej_metadata.get("unit") or description_values.get("unit")
    micron_per_unit = _unit_to_micron(unit)

    if micron_per_unit is None:
        return None
    if not tif.is_imagej and "imagej" not in description.lower():
        return None

    x_size, y_size = _pixel_size_from_resolution(
        x_resolution.value,
        y_resolution.value,
        micron_per_unit,
    )
    return PixelSize(x=x_size, y=y_size, source="imagej")


def _read_ome_pixel_size(ome_xml: str | None) -> PixelSize | None:
    if not ome_xml or "PhysicalSize" not in ome_xml:
        return None

    root = ET.fromstring(ome_xml)
    pixels = next((element for element in root.iter() if element.tag.endswith("Pixels")), None)
    if pixels is None:
        return None

    x_value = pixels.attrib.get("PhysicalSizeX")
    y_value = pixels.attrib.get("PhysicalSizeY")
    if x_value is None or y_value is None:
        return None

    x_scale = _unit_to_micron(pixels.attrib.get("PhysicalSizeXUnit", "um"))
    y_scale = _unit_to_micron(pixels.attrib.get("PhysicalSizeYUnit", "um"))
    if x_scale is None or y_scale is None:
        return None

    return PixelSize(
        x=float(x_value) * x_scale,
        y=float(y_value) * y_scale,
        source="ome",
    )


def _read_standard_tiff_pixel_size(page: tifffile.TiffPage) -> PixelSize | None:
    tags = page.tags
    x_resolution = tags.get("XResolution")
    y_resolution = tags.get("YResolution")
    resolution_unit = tags.get("ResolutionUnit")
    if x_resolution is None or y_resolution is None or resolution_unit is None:
        return None

    micron_per_unit = _standard_resolution_unit_to_micron(resolution_unit.value)
    if micron_per_unit is None:
        return None

    x_size, y_size = _pixel_size_from_resolution(
        x_resolution.value,
        y_resolution.value,
        micron_per_unit,
    )
    return PixelSize(x=x_size, y=y_size, source="tiff-resolution")


def read_pixel_size_from_image(image_path: str | Path) -> PixelSize:
    """Read ``(x, y)`` physical pixel size in microns per pixel.

    The reader intentionally does not guess when metadata are absent. Callers
    that want project-specific defaults should catch ``PixelSizeReadError``.
    """

    path = Path(image_path)
    with tifffile.TiffFile(str(path)) as tif:
        imagej_result = _read_imagej_pixel_size(tif)
        if imagej_result is not None:
            return imagej_result

        ome_result = _read_ome_pixel_size(getattr(tif, "ome_metadata", None))
        if ome_result is not None:
            return ome_result

        tiff_result = _read_standard_tiff_pixel_size(tif.pages[0])
        if tiff_result is not None:
            return tiff_result

    raise PixelSizeReadError(f"No usable pixel-size metadata found in {path}.")


def read_pixel_size_or_default(
    image_path: str | Path,
    default: float = 1.0,
) -> PixelSize:
    """Read pixel size, falling back to ``default`` microns per pixel."""

    try:
        return read_pixel_size_from_image(image_path)
    except Exception as exc:
        print(
            f"Warning: could not read pixel size from {Path(image_path).name}; "
            f"using {default} um/px. Reason: {exc}"
        )
        return PixelSize(x=float(default), y=float(default), source="default_1.0")
