#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


LOG_HEADER = [
    "date",
    "source_path",
    "source_file_name",
    "frame",
    "z_axis",
    "channel",
    "raw_tif_name",
    "raw_csv_name",
    "raw_png_name",
    "x_pixel",
    "y_pixel",
    "micron_per_pixel_x",
    "micron_per_pixel_y",
    "estimated_object_diameter",
    "quality_threshold",
]

SPOT_HEADER_ROWS = [
    [
        "LABEL",
        "ID",
        "TRACK_ID",
        "QUALITY",
        "POSITION_X",
        "POSITION_Y",
        "POSITION_Z",
        "POSITION_T",
        "FRAME",
        "RADIUS",
        "VISIBILITY",
    ],
    [
        "Label",
        "Spot ID",
        "Track ID",
        "Quality",
        "X",
        "Y",
        "Z",
        "T",
        "Frame",
        "Radius",
        "Visibility",
    ],
    [
        "Label",
        "Spot ID",
        "Track ID",
        "Quality",
        "X",
        "Y",
        "Z",
        "T",
        "Frame",
        "R",
        "Visibility",
    ],
]

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".nd2", ".mvd2"}


@dataclass
class SidecarRecord:
    path: Path
    data: dict[str, str]

    @property
    def xml_path(self) -> Path:
        return Path(self.data["xml_path"])

    @property
    def tif_name(self) -> str:
        return self.data["raw_tif_name"]

    @property
    def csv_name(self) -> str:
        return self.data["raw_csv_name"]

    @property
    def png_name(self) -> str:
        return self.data["raw_png_name"]

    @property
    def x_pixel(self) -> str:
        return self.data["x_pixel"]

    @property
    def y_pixel(self) -> str:
        return self.data["y_pixel"]

    @property
    def micron_per_pixel_x(self) -> str:
        return self.data["micron_per_pixel_x"]

    @property
    def micron_per_pixel_y(self) -> str:
        return self.data["micron_per_pixel_y"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utilities for the semi-automatic Fiji/TrackMate annotation workflow."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to the current repository.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "build-manifest",
        help="Scan a source directory and write an ordered manifest of annotatable files.",
    )
    manifest_parser.add_argument("input_root", type=Path, help="Directory to scan.")
    manifest_parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Defaults to data/raw/logs/annotation_state/source_manifest_<timestamp>.csv",
    )

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Convert TrackMate XML sidecars into final CSV exports and append the main annotation log.",
    )
    finalize_parser.add_argument(
        "--sidecar-dir",
        type=Path,
        help="Directory containing *.meta.txt files. Defaults to data/raw/logs/annotation_state.",
    )
    finalize_parser.add_argument(
        "--log-csv",
        type=Path,
        help="Main annotation log CSV. Defaults to data/raw/logs/annoation_record.csv.",
    )
    finalize_parser.add_argument(
        "--csv-dir",
        type=Path,
        help="Directory for exported spot CSV files. Defaults to data/raw/csv.",
    )
    finalize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing files.",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show the next sample number and any pending sidecars that still need XML files.",
    )
    status_parser.add_argument(
        "--sidecar-dir",
        type=Path,
        help="Directory containing *.meta.txt files. Defaults to data/raw/logs/annotation_state.",
    )
    status_parser.add_argument(
        "--log-csv",
        type=Path,
        help="Main annotation log CSV. Defaults to data/raw/logs/annoation_record.csv.",
    )
    return parser.parse_args()


def repo_path(repo_root: Path, relative: str) -> Path:
    return repo_root / relative


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_log_header(log_path: Path) -> None:
    if log_path.exists():
        return
    ensure_parent(log_path)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_HEADER)


def read_existing_log_entries(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    with log_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["raw_tif_name"]
            for row in reader
            if row.get("raw_tif_name")
        }


def find_next_sample_number(repo_root: Path) -> int:
    tif_dir = repo_path(repo_root, "data/raw/tif")
    max_seen = 0
    if tif_dir.exists():
        for path in tif_dir.glob("sample_*.tif"):
            stem = path.stem
            try:
                max_seen = max(max_seen, int(stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return max_seen + 1


def build_manifest(input_root: Path, output_path: Path) -> int:
    rows = []
    for path in sorted(input_root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        rows.append(
            {
                "order": len(rows) + 1,
                "source_path": str(path),
                "source_file_name_hint": path.name,
                "file_type": path.suffix.lower(),
                "notes": (
                    "For .mvd2, source_file_name must be filled with the Fiji image title for each series."
                    if path.suffix.lower() == ".mvd2"
                    else ""
                ),
            }
        )

    ensure_parent(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["order", "source_path", "source_file_name_hint", "file_type", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_sidecar(path: Path) -> SidecarRecord:
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                raise ValueError(f"Invalid sidecar line in {path}: {raw_line.rstrip()}")
            data[key] = value
    required = {
        "date",
        "source_path",
        "source_file_name",
        "frame",
        "z_axis",
        "channel",
        "raw_tif_name",
        "raw_csv_name",
        "raw_png_name",
        "x_pixel",
        "y_pixel",
        "micron_per_pixel_x",
        "micron_per_pixel_y",
        "xml_path",
    }
    defaultable = {
        "x_pixel",
        "y_pixel",
        "micron_per_pixel_x",
        "micron_per_pixel_y",
    }
    missing = sorted((required - defaultable) - data.keys())
    if missing:
        raise ValueError(f"Missing required keys in {path}: {', '.join(missing)}")
    for key in defaultable:
        data.setdefault(key, "")
    return SidecarRecord(path=path, data=data)


def iter_sidecars(sidecar_dir: Path) -> Iterable[SidecarRecord]:
    if not sidecar_dir.exists():
        return []
    sidecars = []
    for path in sorted(sidecar_dir.glob("sample_*.meta.txt")):
        sidecars.append(parse_sidecar(path))
    return sidecars


def spot_rows_from_xml(xml_path: Path) -> tuple[list[list[str]], dict[str, str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    model = root.find("Model")
    spatial_units = "um"
    time_units = "sec"
    if model is not None:
        spatial_units = model.attrib.get("spatialunits", spatial_units)
        time_units = model.attrib.get("timeunits", time_units)
    spatial_units = normalize_spatial_unit(spatial_units)

    detector_settings = root.find("./Settings/DetectorSettings")
    threshold = ""
    diameter = ""
    if detector_settings is not None:
        threshold = detector_settings.attrib.get("THRESHOLD", "")
        radius_value = detector_settings.attrib.get("RADIUS", "")
        if radius_value:
            diameter = str(float(radius_value) * 2.0)

    all_spots = root.find("./Model/AllSpots")
    rows: list[list[str]] = []
    if all_spots is not None:
        for spots_in_frame in all_spots.findall("SpotsInFrame"):
            for spot in spots_in_frame.findall("Spot"):
                rows.append(
                    [
                        spot.attrib.get("name", ""),
                        spot.attrib.get("ID", ""),
                        "",
                        spot.attrib.get("QUALITY", ""),
                        spot.attrib.get("POSITION_X", ""),
                        spot.attrib.get("POSITION_Y", ""),
                        spot.attrib.get("POSITION_Z", ""),
                        spot.attrib.get("POSITION_T", ""),
                        spot.attrib.get("FRAME", ""),
                        spot.attrib.get("RADIUS", ""),
                        spot.attrib.get("VISIBILITY", ""),
                    ]
                )

    units_row = [
        "",
        "",
        "",
        "(quality)",
        f"({spatial_units})",
        f"({spatial_units})",
        f"({spatial_units})",
        f"({time_units})",
        "",
        f"({spatial_units})",
        "",
    ]
    header_rows = [*SPOT_HEADER_ROWS, units_row]
    metadata = {
        "estimated_object_diameter": diameter,
        "quality_threshold": threshold,
    }
    return header_rows + rows, metadata


def write_spot_csv(csv_path: Path, rows: list[list[str]]) -> None:
    ensure_parent(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def append_log_row(log_path: Path, record: SidecarRecord, metadata: dict[str, str]) -> None:
    ensure_log_header(log_path)
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                record.data["date"],
                record.data["source_path"],
                record.data["source_file_name"],
                record.data["frame"],
                record.data["z_axis"],
                record.data["channel"],
                record.data["raw_tif_name"],
                record.data["raw_csv_name"],
                record.data["raw_png_name"],
                record.x_pixel,
                record.y_pixel,
                record.micron_per_pixel_x,
                record.micron_per_pixel_y,
                metadata.get("estimated_object_diameter", ""),
                metadata.get("quality_threshold", ""),
            ]
        )


def normalize_spatial_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized in {"µm", "um", "micrometer", "micrometers", "micron", "microns"}:
        return "micron"
    return unit


def finalize_records(
    sidecar_dir: Path,
    log_path: Path,
    csv_dir: Path,
    dry_run: bool,
) -> int:
    ensure_log_header(log_path)
    existing_entries = read_existing_log_entries(log_path)
    processed = 0
    errors = 0

    for record in iter_sidecars(sidecar_dir):
        if record.tif_name in existing_entries:
            continue
        if not record.xml_path.exists():
            print(f"SKIP {record.tif_name}: missing XML {record.xml_path}", file=sys.stderr)
            continue

        try:
            rows, metadata = spot_rows_from_xml(record.xml_path)
            csv_path = csv_dir / record.csv_name
            if dry_run:
                print(
                    f"DRY-RUN {record.tif_name}: would write {csv_path} and append {log_path}"
                )
            else:
                write_spot_csv(csv_path, rows)
                append_log_row(log_path, record, metadata)
            processed += 1
        except Exception as exc:  # pragma: no cover - defensive CLI error path
            errors += 1
            print(f"ERROR {record.path}: {exc}", file=sys.stderr)

    if errors:
        raise SystemExit(1)
    return processed


def show_status(sidecar_dir: Path, log_path: Path, repo_root: Path) -> None:
    ensure_log_header(log_path)
    logged = read_existing_log_entries(log_path)
    pending = []
    for record in iter_sidecars(sidecar_dir):
        if record.tif_name in logged:
            continue
        pending.append(record)

    print(f"next_sample_number=sample_{find_next_sample_number(repo_root)}")
    print(f"pending_sidecars={len(pending)}")
    for record in pending:
        xml_state = "ready" if record.xml_path.exists() else "missing_xml"
        print(f"{record.tif_name}\t{record.data['source_path']}\t{xml_state}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    if args.command == "build-manifest":
        output = args.output
        if output is None:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            output = repo_path(
                repo_root,
                f"data/raw/logs/annotation_state/source_manifest_{timestamp}.csv",
            )
        count = build_manifest(args.input_root.resolve(), output.resolve())
        print(f"Wrote {count} entries to {output}")
        return

    if args.command == "finalize":
        sidecar_dir = (
            args.sidecar_dir.resolve()
            if args.sidecar_dir
            else repo_path(repo_root, "data/raw/logs/annotation_state")
        )
        log_path = (
            args.log_csv.resolve()
            if args.log_csv
            else repo_path(repo_root, "data/raw/logs/annoation_record.csv")
        )
        csv_dir = (
            args.csv_dir.resolve()
            if args.csv_dir
            else repo_path(repo_root, "data/raw/csv")
        )
        processed = finalize_records(
            sidecar_dir=sidecar_dir,
            log_path=log_path,
            csv_dir=csv_dir,
            dry_run=args.dry_run,
        )
        print(f"Processed {processed} record(s).")
        return

    if args.command == "status":
        sidecar_dir = (
            args.sidecar_dir.resolve()
            if args.sidecar_dir
            else repo_path(repo_root, "data/raw/logs/annotation_state")
        )
        log_path = (
            args.log_csv.resolve()
            if args.log_csv
            else repo_path(repo_root, "data/raw/logs/annoation_record.csv")
        )
        show_status(sidecar_dir=sidecar_dir, log_path=log_path, repo_root=repo_root)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
