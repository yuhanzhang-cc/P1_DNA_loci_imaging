import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import tifffile

from deepblink_torch.cli import evaluate
from deepblink_torch.evaluation import match_coordinates
from deepblink_torch.evaluation import metrics_from_match


class _DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch = x.shape[0]
        output = torch.zeros((batch, 1, 2, 3), dtype=x.dtype, device=x.device)
        output[:, 0, 0, 0] = 0.9
        output[:, 0, 0, 1] = 0.25
        output[:, 0, 0, 2] = 0.25
        output[:, 0, 1, 0] = 0.4
        output[:, 0, 1, 1] = 0.25
        output[:, 0, 1, 2] = 0.25
        return output


def test_match_coordinates_and_metrics_are_consistent() -> None:
    pred = np.asarray([[1.0, 1.0], [10.0, 10.0]], dtype=np.float32)
    true = np.asarray([[1.2, 1.1], [20.0, 20.0]], dtype=np.float32)

    match = match_coordinates(pred, true, radius_px=1.0)
    metrics = metrics_from_match(match)

    assert match.tp == 1
    assert match.fp == 1
    assert match.fn == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_run_test_writes_summary_predictions_and_visuals(tmp_path: Path, monkeypatch) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    output_dir = tmp_path / "evaluation"
    data_path = tmp_path / "dataset.npz"
    model_path = tmp_path / "best_model.pt"
    model_path.write_text("dummy checkpoint", encoding="utf-8")

    image = np.zeros((4, 8), dtype=np.uint8)
    image[1, 1] = 255
    image[1, 5] = 255
    tifffile.imwrite(images_dir / "sample_1.tif", image)

    truth = np.asarray([np.asarray([[1.0, 1.0], [1.0, 5.0]], dtype=np.float32)], dtype=object)
    np.savez(
        data_path,
        x_test=np.asarray([image], dtype=np.float32),
        y_test=truth,
        sample_id_test=np.asarray(["sample_1"], dtype="<U32"),
        image_name_test=np.asarray(["sample_1.tif"], dtype="<U64"),
        label_name_test=np.asarray(["sample_1.csv"], dtype="<U64"),
        source_index_test=np.asarray([0], dtype=np.int32),
    )

    monkeypatch.setattr(
        evaluate.predict_cli,
        "_load_model",
        lambda checkpoint_path, device: (_DummyModel(), 4),
    )

    args = type(
        "Args",
        (),
        {
            "data": str(data_path),
            "images": str(images_dir),
            "model": str(model_path),
            "output": str(output_dir),
            "subset": "test",
            "probability": None,
            "match_radius_px": 3.0,
            "device": "cpu",
            "overlay_mode": "single",
            "verbose": False,
            "probability_sweep": "0.3,0.5",
        },
    )()

    result_dir = evaluate.run_test(args)

    assert result_dir == output_dir
    assert (output_dir / "summary" / "metrics_summary.csv").exists()
    assert (output_dir / "summary" / "metrics_summary.json").exists()
    assert (output_dir / "summary" / "per_image_metrics.csv").exists()
    assert (output_dir / "predictions" / "sample_1.csv").exists()
    assert (output_dir / "predictions" / "sample_1.meta.json").exists()
    assert (output_dir / "matches" / "sample_1.csv").exists()
    assert (output_dir / "overlays" / "sample_1.png").exists()
    assert (output_dir / "match_overlays" / "sample_1.png").exists()
    assert (output_dir / "figures" / "pr_curve.png").exists()
    assert (output_dir / "figures" / "per_image_f1_boxplot.png").exists()
    assert (output_dir / "summary" / "threshold_metrics.csv").exists()
    assert not (output_dir / "figures" / "f1_curve.png").exists()
    assert not (output_dir / "figures" / "error_cdf.png").exists()
    assert not (output_dir / "figures" / "localization_error_vs_confidence.png").exists()

    summary_path = output_dir / "summary" / "metrics_summary.csv"
    summary_head = summary_path.read_text(encoding="utf-8").splitlines()[0]
    assert "micro_precision" in summary_head
    assert "micro_recall" in summary_head
    assert "micro_f1" in summary_head
    assert "median_error_px" in summary_head
    assert "mean_f1_integral" in summary_head
    assert "average_precision" in summary_head

    summary_json = json.loads((output_dir / "summary" / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary_json["probability"] == 0.3
    assert summary_json["probability_source"] == "sweep_best_f1_precision_constrained"
    assert summary_json["best_probability_from_sweep"] == 0.3
    assert summary_json["n_matched_spots"] == 2
    assert math.isfinite(summary_json["median_error_px"])
    assert math.isfinite(summary_json["mean_f1_integral"])

    prediction_lines = (output_dir / "predictions" / "sample_1.csv").read_text(encoding="utf-8").splitlines()
    assert len(prediction_lines) == 3

    per_image_lines = (output_dir / "summary" / "per_image_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert "sample_1" in per_image_lines[1]
    assert "predictions/sample_1.csv" in per_image_lines[1]
    assert "overlays/sample_1.png" in per_image_lines[1]
    assert "match_overlays/sample_1.png" in per_image_lines[1]

    with (output_dir / "summary" / "threshold_metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        threshold_rows = list(csv.DictReader(handle))
    assert [row["probability"] for row in threshold_rows] == ["0.3", "0.5"]
    assert threshold_rows[0]["micro_f1"] == "1.0"
    assert threshold_rows[0]["selected_for_outputs"] == "True"
    assert threshold_rows[0]["selection_reason"] == "sweep_best_f1_precision_constrained"
    assert threshold_rows[1]["selected_for_outputs"] == "False"
    assert float(threshold_rows[1]["micro_f1"]) < 1.0


def test_run_test_explicit_probability_overrides_sweep_for_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    output_dir = tmp_path / "evaluation"
    data_path = tmp_path / "dataset.npz"
    model_path = tmp_path / "best_model.pt"
    model_path.write_text("dummy checkpoint", encoding="utf-8")

    image = np.zeros((4, 8), dtype=np.uint8)
    image[1, 1] = 255
    image[1, 5] = 255
    tifffile.imwrite(images_dir / "sample_1.tif", image)

    truth = np.asarray([np.asarray([[1.0, 1.0], [1.0, 5.0]], dtype=np.float32)], dtype=object)
    np.savez(
        data_path,
        x_test=np.asarray([image], dtype=np.float32),
        y_test=truth,
        sample_id_test=np.asarray(["sample_1"], dtype="<U32"),
        image_name_test=np.asarray(["sample_1.tif"], dtype="<U64"),
        label_name_test=np.asarray(["sample_1.csv"], dtype="<U64"),
        source_index_test=np.asarray([0], dtype=np.int32),
    )

    monkeypatch.setattr(
        evaluate.predict_cli,
        "_load_model",
        lambda checkpoint_path, device: (_DummyModel(), 4),
    )

    args = type(
        "Args",
        (),
        {
            "data": str(data_path),
            "images": str(images_dir),
            "model": str(model_path),
            "output": str(output_dir),
            "subset": "test",
            "probability": 0.5,
            "match_radius_px": 3.0,
            "device": "cpu",
            "overlay_mode": "single",
            "verbose": False,
            "probability_sweep": "0.3,0.5",
        },
    )()

    evaluate.run_test(args)

    summary_json = json.loads((output_dir / "summary" / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary_json["probability"] == 0.5
    assert summary_json["probability_source"] == "specified"
    assert summary_json["best_probability_from_sweep"] == 0.3

    prediction_lines = (output_dir / "predictions" / "sample_1.csv").read_text(encoding="utf-8").splitlines()
    assert len(prediction_lines) == 2

    with (output_dir / "summary" / "threshold_metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        threshold_rows = list(csv.DictReader(handle))
    selected_rows = [row for row in threshold_rows if row["selected_for_outputs"] == "True"]
    assert len(selected_rows) == 1
    assert selected_rows[0]["probability"] == "0.5"
    assert selected_rows[0]["selection_reason"] == "specified"


def test_run_test_verbose_generates_legacy_diagnostics(tmp_path: Path, monkeypatch) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    output_dir = tmp_path / "evaluation_verbose"
    data_path = tmp_path / "dataset.npz"
    model_path = tmp_path / "best_model.pt"
    model_path.write_text("dummy checkpoint", encoding="utf-8")

    image = np.zeros((4, 8), dtype=np.uint8)
    image[1, 1] = 255
    image[1, 5] = 255
    tifffile.imwrite(images_dir / "sample_1.tif", image)

    truth = np.asarray([np.asarray([[1.0, 1.0], [1.0, 5.0]], dtype=np.float32)], dtype=object)
    np.savez(
        data_path,
        x_test=np.asarray([image], dtype=np.float32),
        y_test=truth,
        sample_id_test=np.asarray(["sample_1"], dtype="<U32"),
        image_name_test=np.asarray(["sample_1.tif"], dtype="<U64"),
        label_name_test=np.asarray(["sample_1.csv"], dtype="<U64"),
        source_index_test=np.asarray([0], dtype=np.int32),
    )

    monkeypatch.setattr(
        evaluate.predict_cli,
        "_load_model",
        lambda checkpoint_path, device: (_DummyModel(), 4),
    )

    args = type(
        "Args",
        (),
        {
            "data": str(data_path),
            "images": str(images_dir),
            "model": str(model_path),
            "output": str(output_dir),
            "subset": "test",
            "probability": 0.5,
            "match_radius_px": 3.0,
            "device": "cpu",
            "overlay_mode": "single",
            "verbose": True,
            "probability_sweep": None,
        },
    )()

    evaluate.run_test(args)

    assert (output_dir / "figures" / "pr_curve.png").exists()
    assert (output_dir / "figures" / "per_image_f1_boxplot.png").exists()
    assert (output_dir / "figures" / "f1_curve.png").exists()
    assert (output_dir / "figures" / "error_cdf.png").exists()
    assert (output_dir / "figures" / "localization_error_vs_confidence.png").exists()
    assert not (output_dir / "summary" / "threshold_metrics.csv").exists()
