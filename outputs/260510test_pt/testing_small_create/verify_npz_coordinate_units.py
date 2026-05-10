from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
NPZ_PATH = BASE / "small_test.npz"
CSV_DIR = BASE.parent / "testing_small_csv"
REPORT_PATH = BASE / "coordinate_unit_verification.csv"


def read_csv_coords(path: Path) -> np.ndarray:
    """Return CSV TrackMate coordinates as [Y, X] in microns."""
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                x = float(row["POSITION_X"])
                y = float(row["POSITION_Y"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append([y, x])
    return np.asarray(rows, dtype=np.float64)


def arrays_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-3) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=0.0, atol=atol))


def main() -> None:
    csv_coords = {path.name: read_csv_coords(path) for path in sorted(CSV_DIR.glob("*.csv"))}
    report_rows: list[dict[str, object]] = []

    with np.load(NPZ_PATH, allow_pickle=True) as npz:
        for split in ("train", "valid", "test"):
            labels = npz[f"y_{split}"]
            pixel_x = np.asarray(npz[f"pixel_size_x_{split}"], dtype=np.float64)
            pixel_y = np.asarray(npz[f"pixel_size_y_{split}"], dtype=np.float64)

            for index, label_array in enumerate(labels):
                npz_coords = np.asarray(label_array, dtype=np.float64)
                matched_name = ""
                matched_mode = ""
                max_abs_diff_px = np.nan
                max_abs_diff_um = np.nan
                min_ratio_to_um = np.nan
                max_ratio_to_um = np.nan

                for csv_name, coords_um in csv_coords.items():
                    expected_px = coords_um.copy()
                    expected_px[:, 0] /= pixel_y[index]
                    expected_px[:, 1] /= pixel_x[index]

                    if arrays_close(npz_coords, expected_px):
                        matched_name = csv_name
                        matched_mode = "pixel"
                        max_abs_diff_px = float(np.max(np.abs(npz_coords - expected_px)))
                        max_abs_diff_um = float(np.max(np.abs(npz_coords - coords_um)))
                        nonzero_um = coords_um != 0
                        ratios = npz_coords[nonzero_um] / coords_um[nonzero_um]
                        min_ratio_to_um = float(np.min(ratios))
                        max_ratio_to_um = float(np.max(ratios))
                        break

                    if arrays_close(npz_coords, coords_um):
                        matched_name = csv_name
                        matched_mode = "micron"
                        max_abs_diff_px = float(np.max(np.abs(npz_coords - expected_px)))
                        max_abs_diff_um = float(np.max(np.abs(npz_coords - coords_um)))
                        min_ratio_to_um = 1.0
                        max_ratio_to_um = 1.0
                        break

                report_rows.append(
                    {
                        "split": split,
                        "index": index,
                        "matched_csv": matched_name,
                        "matched_unit": matched_mode,
                        "pixel_size_x": float(pixel_x[index]),
                        "pixel_size_y": float(pixel_y[index]),
                        "spot_count": int(npz_coords.shape[0]),
                        "max_abs_diff_vs_expected_pixel": max_abs_diff_px,
                        "max_abs_diff_vs_raw_micron": max_abs_diff_um,
                        "min_npz_over_raw_micron": min_ratio_to_um,
                        "max_npz_over_raw_micron": max_ratio_to_um,
                    }
                )

    fieldnames = list(report_rows[0].keys())
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    for row in report_rows:
        print(
            f"{row['split']}[{row['index']}] {row['matched_csv']}: "
            f"matched_unit={row['matched_unit']}, "
            f"pixel_size=({row['pixel_size_x']:.9g}, {row['pixel_size_y']:.9g}), "
            f"ratio_to_raw_um={row['min_npz_over_raw_micron']:.6g}.."
            f"{row['max_npz_over_raw_micron']:.6g}, "
            f"max_diff_pixel={row['max_abs_diff_vs_expected_pixel']:.6g}"
        )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
