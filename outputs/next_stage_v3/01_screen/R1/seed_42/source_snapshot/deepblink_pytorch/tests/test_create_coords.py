from pathlib import Path

import numpy as np

from deepblink_torch.cli.create import _read_coords


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV_DIR = PROJECT_ROOT / "data" / "raw" / "csv"


def test_read_coords_tracks_position_columns_from_real_trackmate_csv() -> None:
    coords = _read_coords(RAW_CSV_DIR / "sample_1.csv")

    assert coords.shape[1] == 2
    np.testing.assert_allclose(
        coords[0],
        np.asarray([38.69241441303001, 5.391913130323854], dtype=np.float32),
    )


def test_read_coords_skips_trackmate_header_rows_and_reads_first_data_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS,VISIBILITY",
                "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius,Visibility",
                "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R,Visibility",
                ",,,(quality),(micron),(micron),(micron),(sec),,(micron),",
                "ID1,1,,88.0,5.5,38.5,0.0,0.0,0,0.7,1",
            ]
        ),
        encoding="utf-8",
    )

    coords = _read_coords(csv_path)

    np.testing.assert_allclose(coords, np.asarray([[38.5, 5.5]], dtype=np.float32))


def test_read_coords_keeps_plain_y_x_csv_compatible(tmp_path: Path) -> None:
    csv_path = tmp_path / "plain.csv"
    csv_path.write_text("y,x,p\n10.5,12.25,0.9\n", encoding="utf-8")

    coords = _read_coords(csv_path)

    np.testing.assert_allclose(coords, np.asarray([[10.5, 12.25]], dtype=np.float32))
