from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "artifacts" / "tf_baseline"


def test_tf_baseline_directory_exists() -> None:
    assert BASELINE_DIR.exists()
    assert BASELINE_DIR.is_dir()


def test_tf_baseline_contains_csv_files() -> None:
    csv_files = sorted(BASELINE_DIR.glob("*.csv"))
    assert csv_files, "Expected at least one baseline CSV file in artifacts/tf_baseline."


def test_tf_baseline_csv_has_y_x_p_columns() -> None:
    csv_files = sorted(BASELINE_DIR.glob("*.csv"))
    assert csv_files, "Expected at least one baseline CSV file in artifacts/tf_baseline."

    for csv_file in csv_files:
        frame = pd.read_csv(csv_file)
        assert list(frame.columns) == ["y", "x", "p"], (
            f"{csv_file} must contain exactly three columns: y, x, p."
        )
