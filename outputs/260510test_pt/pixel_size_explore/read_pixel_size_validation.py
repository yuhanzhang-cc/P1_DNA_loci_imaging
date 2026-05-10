from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import tifffile


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TIF_DIR = REPO_ROOT / "data" / "raw" / "tif"
ANNOTATION_RECORD = REPO_ROOT / "data" / "raw" / "logs" / "annoation_record.csv"
OUTPUT_CSV = SCRIPT_DIR / "read_pixel_size_result.csv"


UNIT_TO_UM = {
    "um": 1.0,
    "\u00b5m": 1.0,
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


def rational_to_float(value: Any) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        return float(numerator) / float(denominator)
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator is not None:
        return float(numerator) / float(denominator)
    return float(value)


def parse_description(description: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not description:
        return parsed
    for line in description.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def unit_scale_um(unit: str | None) -> float | None:
    if not unit:
        return None
    return UNIT_TO_UM.get(unit.strip().lower())


def standard_resolution_unit_scale_um(unit_value: Any) -> float | None:
    name = getattr(unit_value, "name", str(unit_value)).strip().lower()
    value = getattr(unit_value, "value", unit_value)

    if value == 2 or name in {"inch", "in"}:
        return 25400.0
    if value == 3 or name in {"centimeter", "centimetre", "cm"}:
        return 10000.0
    return None


def read_ome_pixel_size_um(ome_xml: str | None) -> tuple[float, float] | None:
    if not ome_xml or "PhysicalSize" not in ome_xml:
        return None
    root = ET.fromstring(ome_xml)
    pixels = next(
        (element for element in root.iter() if element.tag.endswith("Pixels")),
        None,
    )
    if pixels is None:
        return None

    x_value = pixels.attrib.get("PhysicalSizeX")
    y_value = pixels.attrib.get("PhysicalSizeY")
    if x_value is None or y_value is None:
        return None

    x_unit = pixels.attrib.get("PhysicalSizeXUnit", "um")
    y_unit = pixels.attrib.get("PhysicalSizeYUnit", "um")
    x_scale = unit_scale_um(x_unit)
    y_scale = unit_scale_um(y_unit)
    if x_scale is None or y_scale is None:
        return None

    return float(x_value) * x_scale, float(y_value) * y_scale


def pixel_size_from_resolution(
    x_resolution: Any,
    y_resolution: Any,
    unit_scale: float,
) -> tuple[float, float]:
    x_pixels_per_unit = rational_to_float(x_resolution)
    y_pixels_per_unit = rational_to_float(y_resolution)
    if x_pixels_per_unit <= 0 or y_pixels_per_unit <= 0:
        raise ValueError("resolution tags must be positive")
    return unit_scale / x_pixels_per_unit, unit_scale / y_pixels_per_unit


def read_pixel_size_from_image(image_path: Path) -> tuple[float, float, str]:
    with tifffile.TiffFile(str(image_path)) as tif:
        page = tif.pages[0]

        ome_result = read_ome_pixel_size_um(getattr(tif, "ome_metadata", None))
        if ome_result is not None:
            return ome_result[0], ome_result[1], "ome-xml"

        tags = page.tags
        x_resolution = tags.get("XResolution")
        y_resolution = tags.get("YResolution")
        if x_resolution is None or y_resolution is None:
            raise ValueError("missing XResolution or YResolution TIFF tag")

        description = page.description or ""
        description_values = parse_description(description)
        imagej_metadata = tif.imagej_metadata or {}

        imagej_unit = imagej_metadata.get("unit") or description_values.get("unit")
        imagej_scale = unit_scale_um(imagej_unit)
        if imagej_scale is not None and (tif.is_imagej or "imagej" in description.lower()):
            x_size, y_size = pixel_size_from_resolution(
                x_resolution.value,
                y_resolution.value,
                imagej_scale,
            )
            return x_size, y_size, "imagej-resolution"

        resolution_unit = tags.get("ResolutionUnit")
        if resolution_unit is not None:
            standard_scale = standard_resolution_unit_scale_um(resolution_unit.value)
            if standard_scale is not None:
                x_size, y_size = pixel_size_from_resolution(
                    x_resolution.value,
                    y_resolution.value,
                    standard_scale,
                )
                return x_size, y_size, "standard-tiff-resolution"

        raise ValueError("no usable physical unit for TIFF resolution tags")


def load_annotation_record() -> dict[str, tuple[float, float]]:
    records: dict[str, tuple[float, float]] = {}
    with ANNOTATION_RECORD.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("raw_tif_name")
            x_value = row.get("micron_per_pixel_x")
            y_value = row.get("micron_per_pixel_y")
            if not filename or not x_value or not y_value:
                continue
            records[filename] = (float(x_value), float(y_value))
    return records


def matches_reference(
    observed: tuple[float, float],
    expected: tuple[float, float],
) -> bool:
    return math.isclose(observed[0], expected[0], rel_tol=5e-3, abs_tol=1e-4) and math.isclose(
        observed[1],
        expected[1],
        rel_tol=5e-3,
        abs_tol=1e-4,
    )


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.9g}"


def main() -> None:
    records = load_annotation_record()
    tif_files = sorted(TIF_DIR.glob("*.tif"))

    rows: list[dict[str, str]] = []
    notes: list[str] = []
    methods: Counter[str] = Counter()

    for image_path in tif_files:
        pixel_size_x: float | None = None
        pixel_size_y: float | None = None
        status = "fail"

        try:
            pixel_size_x, pixel_size_y, method = read_pixel_size_from_image(image_path)
            methods[method] += 1
            status = "success"

            reference = records.get(image_path.name)
            if reference is None:
                status = "warning"
                notes.append(f"{image_path.name}: read from image, but no annotation record row")
            elif not matches_reference((pixel_size_x, pixel_size_y), reference):
                status = "warning"
                notes.append(
                    f"{image_path.name}: image metadata=({pixel_size_x:.9g}, {pixel_size_y:.9g}) "
                    f"annotation_record=({reference[0]:.9g}, {reference[1]:.9g})"
                )
        except Exception as exc:
            notes.append(f"{image_path.name}: fail: {exc}")

        rows.append(
            {
                "filename": image_path.name,
                "pixel_size_x": format_float(pixel_size_x),
                "pixel_size_y": format_float(pixel_size_y),
                "status": status,
            }
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "pixel_size_x", "pixel_size_y", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    statuses = Counter(row["status"] for row in rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(
        "Status counts: "
        f"success={statuses['success']} warning={statuses['warning']} fail={statuses['fail']} "
        f"total={len(rows)}"
    )
    if methods:
        print("Read methods: " + ", ".join(f"{name}={count}" for name, count in sorted(methods.items())))
    if notes:
        print("Notes for non-success rows:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
