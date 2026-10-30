from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "error_context_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("error_context_diagnostics", SCRIPT_PATH)
assert SPEC is not None
diagnostics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = diagnostics
SPEC.loader.exec_module(diagnostics)


def test_patch_features_include_common_context_fields() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 10.0
    image[4, 5] = 5.0

    features = diagnostics._extract_patch_features(image, y=4.0, x=4.0, radius=2)

    assert features["center_intensity"] == 10.0
    assert features["local_max"] == 10.0
    assert features["local_std"] > 0.0
    assert features["local_contrast"] > 0.0
    assert features["local_zscore"] > 0.0
    assert features["peak_component_size_px"] >= 1.0
    assert features["distance_to_border_px"] == 4.0


def test_error_context_builds_tp_fp_fn_rows() -> None:
    sample = diagnostics.SampleData(
        sample_id="sample_1",
        image_name="sample_1.tif",
        image=np.zeros((16, 16), dtype=np.float32),
        truth_coords=np.asarray([[4.0, 4.0], [12.0, 12.0]], dtype=np.float32),
    )
    prediction = diagnostics.SamplePrediction(
        sample=sample,
        dense_candidates=np.asarray(
            [
                [4.2, 4.1, 0.9],
                [2.0, 13.0, 0.8],
                [12.0, 12.0, 0.1],
            ],
            dtype=np.float32,
        ),
    )
    baseline = diagnostics.BaselineConfig(
        probability=0.5,
        match_radius_px=1.0,
        precision=float("nan"),
        recall=float("nan"),
        f1=float("nan"),
        tp=0,
        fp=0,
        fn=0,
    )

    rows, per_sample = diagnostics._build_error_context([prediction], baseline, patch_radius=2)

    assert [row["event_type"] for row in rows] == ["TP", "FP", "FN"]
    assert np.isclose(rows[0]["confidence"], 0.9)
    assert np.isnan(rows[1]["true_y_px"])
    assert np.isnan(rows[2]["confidence"])
    assert per_sample[0]["tp"] == 1
    assert per_sample[0]["fp"] == 1
    assert per_sample[0]["fn"] == 1


def test_postprocess_ablation_includes_expected_strategy_families(tmp_path: Path) -> None:
    train_truth = np.asarray(
        [
            np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
            np.asarray([[3.0, 3.0]], dtype=np.float32),
        ],
        dtype=object,
    )
    valid_truth = np.asarray(
        [np.asarray([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]], dtype=np.float32)],
        dtype=object,
    )
    data_path = tmp_path / "dataset.npz"
    np.savez(
        data_path,
        x_train=np.zeros((2, 8, 8), dtype=np.float32),
        y_train=train_truth,
        x_valid=np.zeros((1, 8, 8), dtype=np.float32),
        y_valid=valid_truth,
    )
    sample = diagnostics.SampleData(
        sample_id="sample_1",
        image_name="sample_1.tif",
        image=np.zeros((8, 8), dtype=np.float32),
        truth_coords=np.asarray([[1.0, 1.0], [6.0, 6.0]], dtype=np.float32),
    )
    prediction = diagnostics.SamplePrediction(
        sample=sample,
        dense_candidates=np.asarray(
            [
                [1.0, 1.0, 0.95],
                [6.0, 6.0, 0.90],
                [2.0, 6.0, 0.80],
            ],
            dtype=np.float32,
        ),
    )
    baseline = diagnostics.BaselineConfig(
        probability=0.5,
        match_radius_px=1.0,
        precision=2 / 3,
        recall=1.0,
        f1=0.8,
        tp=2,
        fp=1,
        fn=0,
    )

    rows = diagnostics._run_postprocess_ablation([prediction], data_path, baseline)
    strategies = {row["strategy"] for row in rows}

    assert "baseline_threshold_from_evaluation" in strategies
    assert "threshold_only" in strategies
    assert "fixed_top_n" in strategies
    assert "threshold_max_n" in strategies
    assert "validation_prior_top_n" in strategies
    assert any(row["strategy"] == "threshold_only" and row["threshold"] == 0.81 for row in rows)
    assert any(row["strategy"] == "fixed_top_n" and row["top_n"] == 5 for row in rows)


def test_ppt_figures_and_hard_examples_can_be_generated(tmp_path: Path) -> None:
    sample = diagnostics.SampleData(
        sample_id="sample_1",
        image_name="sample_1.tif",
        image=np.eye(24, dtype=np.float32) * 10.0,
        truth_coords=np.asarray([[6.0, 6.0], [18.0, 18.0]], dtype=np.float32),
    )
    event_rows = [
        {
            "sample_id": "sample_1",
            "image_name": "sample_1.tif",
            "event_type": "TP",
            "confidence": 0.95,
            "pred_y_px": 6.0,
            "pred_x_px": 6.0,
            "true_y_px": 6.1,
            "true_x_px": 6.1,
            "local_zscore": 3.0,
            "local_contrast": 10.0,
            "peak_sharpness": 12.0,
        },
        {
            "sample_id": "sample_1",
            "image_name": "sample_1.tif",
            "event_type": "FP",
            "confidence": 0.91,
            "pred_y_px": 4.0,
            "pred_x_px": 18.0,
            "true_y_px": float("nan"),
            "true_x_px": float("nan"),
            "local_zscore": 2.0,
            "local_contrast": 4.0,
            "peak_sharpness": 5.0,
        },
        {
            "sample_id": "sample_1",
            "image_name": "sample_1.tif",
            "event_type": "FN",
            "confidence": float("nan"),
            "pred_y_px": float("nan"),
            "pred_x_px": float("nan"),
            "true_y_px": 18.0,
            "true_x_px": 18.0,
            "local_zscore": 1.0,
            "local_contrast": 2.0,
            "peak_sharpness": 4.0,
        },
    ]
    baseline_row = {
        "strategy": "baseline_threshold_from_evaluation",
        "threshold": 0.4,
        "top_n": "",
        "max_n": "",
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "median_error_px": 0.1,
    }
    recommendation = {
        "baseline_row": baseline_row,
        "threshold_best": baseline_row,
        "topn_best": {**baseline_row, "strategy": "fixed_top_n", "top_n": 5, "precision": 0.4, "recall": 0.4, "f1": 0.4},
        "hybrid_best": {**baseline_row, "strategy": "threshold_max_n", "max_n": 40},
        "prior_best": {**baseline_row, "strategy": "validation_prior_top_n", "top_n": 6, "precision": 0.3, "recall": 0.3, "f1": 0.3},
        "threshold_precision_constrained": baseline_row,
        "confidence_medians": {"TP": 0.95, "FP": 0.91, "FN": float("nan")},
        "zscore_medians": {"TP": 3.0, "FP": 2.0, "FN": 1.0},
        "contrast_medians": {"TP": 10.0, "FP": 4.0, "FN": 2.0},
        "sharpness_medians": {"TP": 12.0, "FP": 5.0, "FN": 4.0},
        "event_counts": {"TP": 1, "FP": 1, "FN": 1},
        "fp_conf_lower": False,
        "fn_low_z": True,
        "fp_high_intensity_low_sharpness": False,
        "best_strategy_text": "Do not change the default inference strategy yet.",
    }

    diagnostics._make_ppt_figures(
        tmp_path,
        event_rows,
        [sample],
        recommendation,
        patch_radius=3,
    )
    hard_rows = diagnostics._build_hard_examples(
        tmp_path,
        event_rows,
        [{"sample_id": "sample_1", "tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}],
        [sample],
        patch_radius=3,
    )
    diagnostics._write_ppt_takeaways(tmp_path, recommendation)

    assert (tmp_path / "figures_ppt" / "error_context_summary.png").exists()
    assert (tmp_path / "figures_ppt" / "patch_montage.png").exists()
    assert (tmp_path / "ppt_takeaways.md").exists()
    assert hard_rows
    patch_paths = [row["patch_path"] for row in hard_rows if row["patch_path"]]
    assert patch_paths
    assert (tmp_path / patch_paths[0]).exists()
