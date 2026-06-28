from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deepblink_torch.cli import train as train_cli


class _DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def to(self, device: torch.device) -> _DummyModel:  # type: ignore[override]
        return self


def test_default_checkpoint_path_uses_experiment_directory(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    data_path = tmp_path / "dataset.npz"
    data_path.write_text("dummy", encoding="utf-8")

    cwd = tmp_path / "outputs" / "exp_001" / "evaluation" / "summary"
    cwd.mkdir(parents=True)
    monkeypatch.chdir(cwd)

    checkpoint_path = train_cli._default_checkpoint_path(data_path, project_root=project_root)

    assert checkpoint_path == project_root / "outputs" / "exp_001" / "training_modification" / "best_model.pt"


def test_default_checkpoint_path_falls_back_to_timestamp(tmp_path: Path, monkeypatch) -> None:
    class _FixedDatetime:
        @classmethod
        def now(cls):
            class _FixedNow:
                @staticmethod
                def strftime(format_string: str) -> str:
                    return "20260628_123456"

            return _FixedNow()

    project_root = tmp_path
    data_path = tmp_path / "dataset.npz"
    data_path.write_text("dummy", encoding="utf-8")

    cwd = tmp_path / "workspace"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(train_cli, "datetime", _FixedDatetime)

    checkpoint_path = train_cli._default_checkpoint_path(data_path, project_root=project_root)

    assert checkpoint_path == project_root / "outputs" / "training_20260628_123456" / "training_modification" / "best_model.pt"


def test_train_parser_accepts_loss_variants() -> None:
    for loss_name in train_cli.LOSS_CHOICES:
        args = train_cli.parse_args(["--data", "dataset.npz", "--loss", loss_name])

        assert args.loss == loss_name


def test_run_training_places_artifacts_next_to_explicit_checkpoint(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "dataset.npz"
    data_path.write_text("dummy", encoding="utf-8")
    checkpoint_path = tmp_path / "runs" / "exp_a" / "best_model.pt"

    monkeypatch.setattr(train_cli, "build_dataloaders", lambda *args, **kwargs: (object(), object()))
    monkeypatch.setattr(train_cli, "UNetDeepBlink", _DummyModel)
    monkeypatch.setattr(train_cli, "train_one_epoch", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(train_cli, "validate", lambda *args, **kwargs: (2.0, 3.0, 4.0, 0.5, 0.6, 0.7))

    captured: dict[str, Path | list[dict[str, object]]] = {}

    def fake_write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
        captured["csv_path"] = path
        captured["rows"] = rows

    def fake_plot_training_curves(path: Path, history: list[dict[str, object]]) -> None:
        captured["png_path"] = path
        captured["history"] = history

    monkeypatch.setattr(train_cli, "_write_rows_csv", fake_write_rows_csv)
    monkeypatch.setattr(train_cli, "_plot_training_curves", fake_plot_training_curves)

    args = argparse.Namespace(
        data=str(data_path),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        device="cpu",
        checkpoint=str(checkpoint_path),
    )

    result = train_cli.run_training(args)

    assert result == checkpoint_path.resolve()
    assert checkpoint_path.exists()
    assert captured["csv_path"] == checkpoint_path.resolve().parent / "training_metrics.csv"
    assert captured["png_path"] == checkpoint_path.resolve().parent / "training_curves.png"
    assert isinstance(captured["rows"], list)
    assert captured["rows"][0]["loss_name"] == "dice_rmse"
    assert captured["rows"][0]["focal_alpha"] == 0.25
    assert captured["rows"][0]["focal_gamma"] == 2.0
    assert captured["rows"][0]["rmse_weight"] == 2.0
    assert captured["rows"][0]["val_rmse"] == 4.0

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["loss_name"] == "dice_rmse"
    assert checkpoint["focal_alpha"] == 0.25
    assert checkpoint["focal_gamma"] == 2.0
    assert checkpoint["rmse_weight"] == 2.0
    assert checkpoint["loss_config"] == {
        "loss_name": "dice_rmse",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "rmse_weight": 2.0,
    }
