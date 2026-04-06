#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_TIF_DIR = REPO_ROOT / "data/raw/tif"
RAW_PNG_DIR = REPO_ROOT / "data/raw/png"
RAW_CSV_DIR = REPO_ROOT / "data/raw/csv"
TRACKMATE_XML_DIR = REPO_ROOT / "data/raw/logs/trackmate_xml"
ANNOTATION_STATE_DIR = REPO_ROOT / "data/raw/logs/annotation_state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete the highest-numbered sample_N files across tif/png/csv/xml/meta "
            "when an in-progress annotation should be discarded."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which files would be deleted.",
    )
    return parser.parse_args()


def latest_sample_base(tif_dir: Path) -> str | None:
    max_number = -1
    max_base = None
    for path in tif_dir.glob("sample_*.tif"):
        stem = path.stem
        try:
            number = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if number > max_number:
            max_number = number
            max_base = stem
    return max_base


def sample_related_paths(sample_base: str) -> list[Path]:
    return [
        RAW_TIF_DIR / f"{sample_base}.tif",
        RAW_PNG_DIR / f"{sample_base}.png",
        RAW_CSV_DIR / f"{sample_base}.csv",
        TRACKMATE_XML_DIR / f"{sample_base}.xml",
        ANNOTATION_STATE_DIR / f"{sample_base}.meta.txt",
    ]


def main() -> None:
    args = parse_args()

    sample_base = latest_sample_base(RAW_TIF_DIR)
    if sample_base is None:
        print("No sample_N.tif files found in data/raw/tif.")
        return

    targets = sample_related_paths(sample_base)
    existing_targets = [path for path in targets if path.exists()]

    if not existing_targets:
        print(f"No files found for {sample_base}.")
        return

    action = "Would delete" if args.dry_run else "Deleting"
    print(f"{action} files for {sample_base}:")
    for path in existing_targets:
        print(path)

    if args.dry_run:
        return

    for path in existing_targets:
        path.unlink()

    print(f"Deleted {len(existing_targets)} file(s) for {sample_base}.")


if __name__ == "__main__":
    main()
