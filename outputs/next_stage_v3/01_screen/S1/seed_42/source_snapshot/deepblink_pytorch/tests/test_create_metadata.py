from pathlib import Path
import shutil

import numpy as np

from deepblink_torch.cli.create import run_create


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TIF_DIR = PROJECT_ROOT / "data" / "raw" / "tif"
RAW_CSV_DIR = PROJECT_ROOT / "data" / "raw" / "csv"


def test_create_writes_split_metadata_and_manifest(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    for stem in ["sample_1", "sample_10", "sample_100"]:
        shutil.copy2(RAW_TIF_DIR / f"{stem}.tif", images_dir / f"{stem}.tif")
        shutil.copy2(RAW_CSV_DIR / f"{stem}.csv", labels_dir / f"{stem}.csv")

    output_path = tmp_path / "dataset.npz"
    run_create(
        type(
            "Args",
            (),
            {
                "images": str(images_dir),
                "labels": str(labels_dir),
                "output": str(output_path),
                "train_ratio": 0.8,
                "valid_ratio": 0.1,
                "seed": 42,
            },
        )()
    )

    with np.load(output_path, allow_pickle=True) as data:
        assert "image_name_train" in data.files
        assert "label_name_train" in data.files
        assert "sample_id_train" in data.files
        assert "split_seed" in data.files
        assert "train_ratio" in data.files
        assert "valid_ratio" in data.files
        assert set(data["image_name_train"].tolist()).issubset(
            {"sample_1.tif", "sample_10.tif", "sample_100.tif"}
        )

    manifest_path = output_path.with_suffix(".split.csv")
    assert manifest_path.exists()
    manifest_head = manifest_path.read_text(encoding="utf-8").splitlines()[0]
    assert manifest_head == (
        "subset,subset_index,source_index,sample_id,image_name,label_name,"
        "pixel_size_x,pixel_size_y,pixel_size_source"
    )
