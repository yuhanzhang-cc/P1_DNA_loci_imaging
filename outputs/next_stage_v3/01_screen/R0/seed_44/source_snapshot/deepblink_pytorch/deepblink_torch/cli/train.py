"""Train command for deepblink_torch CLI."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from deepblink_torch import DEFAULT_RMSE_WEIGHT
from deepblink_torch import SpotsDataset
from deepblink_torch import UNetDeepBlink
from deepblink_torch import UNetDeepLociContext
from deepblink_torch import combined_dice_focal_rmse_loss
from deepblink_torch import combined_dice_rmse_loss
from deepblink_torch import combined_focal_rmse_loss
from deepblink_torch import dice_loss
from deepblink_torch import rmse_loss
from deepblink_torch.networks.evidence import UNetDeepLociEvidence, UNetDeepLociLocal
from deepblink_torch.networks.feature_evidence import UNetDeepLociFeatureEvidence
from deepblink_torch.networks.masked_se import UNetDeepBlinkMaskedSE


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LOSS_CHOICES = ("dice_rmse", "focal_rmse", "dice_focal_rmse")
MODEL_CHOICES = ("deepblink", "deeploci_context", "deeploci_evidence", "deeploci_local",
                 "deeploci_feature_evidence", "deepblink_masked_se")


def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add training arguments to an argparse parser."""
    parser.add_argument(
        "--data",
        required=True,
        help="Path to deepBlink .npz dataset containing train/valid/test splits.",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Use bfloat16 autocast on CUDA/XPU; losses remain float32."
    )
    parser.add_argument(
        "--development-only", action="store_true", help="Load train/valid arrays only; never read test arrays."
    )
    parser.add_argument(
        "--channels-last", action="store_true", help="Use channels-last convolution memory layout."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Python, NumPy and PyTorch training seed."
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="PyTorch CPU intra-op threads."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for train/valid DataLoaders.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "xpu"],
        help="Training device.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Output checkpoint path. Defaults to "
            "./outputs/<experiment>/training_modification/best_model.pt."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=LOSS_CHOICES,
        default="dice_rmse",
        help="Training loss composition. Defaults to the baseline dice_rmse.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="deepblink",
        help="Model architecture. Defaults to the baseline deepblink model.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal loss positive-class alpha. Used by focal loss variants.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss focusing gamma. Used by focal loss variants.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train UNetDeepBlink with PyTorch.")
    add_train_arguments(parser)
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    """Resolve runtime device, falling back to CPU if CUDA is unavailable."""
    if requested == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("Intel XPU requested but unavailable in this environment.")
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda but CUDA is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def build_dataloaders(
    data_path: Path,
    batch_size: int,
    cell_size: int = 4,
    development_only: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train/valid DataLoaders from one deepBlink npz file."""
    train_dataset = SpotsDataset(
        name=data_path,
        cell_size=cell_size,
        subset="train",
        development_only=development_only,
    )
    valid_dataset = SpotsDataset(
        name=data_path,
        cell_size=cell_size,
        subset="valid",
        development_only=development_only,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader


def _normalise_loss_name(loss_name: str) -> str:
    if loss_name not in LOSS_CHOICES:
        raise ValueError(f"--loss must be one of {LOSS_CHOICES}, but got {loss_name!r}.")
    return loss_name


def _normalise_model_name(model_name: str) -> str:
    if model_name not in MODEL_CHOICES:
        raise ValueError(
            f"--model must be one of {MODEL_CHOICES}, but got {model_name!r}."
        )
    return model_name


def _count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _build_model(
    model_name: str,
    *,
    cell_size: int,
    device: torch.device,
) -> torch.nn.Module:
    model_name = _normalise_model_name(model_name)
    if model_name == "deepblink":
        return UNetDeepBlink(cell_size=cell_size).to(device)
    if model_name == "deeploci_context":
        return UNetDeepLociContext(cell_size=cell_size).to(device)
    if model_name == "deeploci_evidence":
        return UNetDeepLociEvidence(cell_size=cell_size).to(device)
    if model_name == "deeploci_local":
        return UNetDeepLociLocal(cell_size=cell_size).to(device)
    if model_name == "deeploci_feature_evidence":
        return UNetDeepLociFeatureEvidence(cell_size=cell_size).to(device)
    if model_name == "deepblink_masked_se":
        return UNetDeepBlinkMaskedSE(cell_size=cell_size).to(device)
    raise ValueError(f"Unsupported model: {model_name}")


def _architecture_config(model_name: str, model: torch.nn.Module) -> dict[str, object]:
    config: dict[str, object] = {
        "model_name": model_name,
        "cell_size": int(getattr(model, "cell_size", 4)),
        "output_layout": "channel_last",
        "output_channels": 3,
        "occupancy_channel": 0,
        "offset_channels": [1, 2],
        "backbone": "UNetDeepBlink",
        "block": str(getattr(model, "block_name", "convolutional")),
        "filters": int(getattr(model, "filters", 5)),
        "ndown": int(getattr(model, "ndown", 2)),
    }
    if model_name == "deeploci_feature_evidence":
        config.update(head="offset_conditioned_learned_feature_contrast_residual",
                      radii_original_px=[1.5, 3., 6.], angular_samples=8,
                      image_embedding_channels=8, image_embedding_convs=2,
                      evidence_channels=51, feature_channels=16, fusion_channels=32,
                      valid_mask="explicit_geometric_region", auxiliary_losses=[])
    elif model_name == "deepblink_masked_se":
        config.update(head="shared_1x1_3_channel", offset_head="shared_1x1_conv",
                      squeeze_pooling="geometric_mask_weighted_mean",
                      valid_mask="explicit_geometric_region")
    elif model_name == "deeploci_evidence":
        config.update(head="offset_conditioned_radial_image_evidence_residual",
                      radii_original_px=[1.5, 3., 6.], angular_samples=8,
                      evidence_channels=12, feature_channels=16, fusion_channels=32,
                      valid_mask="explicit_geometric_region", auxiliary_losses=[])
    elif model_name == "deeploci_local":
        config.update(head="occupancy_3x3_hidden16_relu_1x1", offset_head="1x1_conv")
    elif model_name == "deeploci_context":
        config.update(
            {
                "head": "split_context_occupancy_offset",
                "occupancy_context": {
                    "conv3x3_channels": int(getattr(model, "context_channels", 32)),
                    "dilated_conv3x3_channels": int(
                        getattr(model, "context_channels", 32)
                    ),
                    "dilation": int(getattr(model, "context_dilation", 2)),
                    "activation": "relu",
                },
                "offset_head": "1x1_conv",
            }
        )
    else:
        config.update(
            {
                "head": "shared_1x1_3_channel",
                "occupancy_context": None,
                "offset_head": "shared_1x1_conv",
            }
        )
    return config


def _build_loss_config(args: argparse.Namespace) -> dict[str, object]:
    loss_name = _normalise_loss_name(getattr(args, "loss", "dice_rmse"))
    focal_alpha = float(getattr(args, "focal_alpha", 0.25))
    focal_gamma = float(getattr(args, "focal_gamma", 2.0))
    if not 0.0 <= focal_alpha <= 1.0:
        raise ValueError(f"--focal-alpha must be in [0, 1], but got {focal_alpha}")
    if focal_gamma < 0.0:
        raise ValueError(f"--focal-gamma must be >= 0, but got {focal_gamma}")
    return {
        "loss_name": loss_name,
        "focal_alpha": focal_alpha,
        "focal_gamma": focal_gamma,
        "rmse_weight": DEFAULT_RMSE_WEIGHT,
    }


def _build_loss_function(loss_config: dict[str, object]) -> LossFunction:
    loss_name = str(loss_config["loss_name"])
    focal_alpha = float(loss_config["focal_alpha"])
    focal_gamma = float(loss_config["focal_gamma"])
    rmse_weight = float(loss_config["rmse_weight"])

    if loss_name == "dice_rmse":
        return combined_dice_rmse_loss
    if loss_name == "focal_rmse":
        return lambda y_true, y_pred: combined_focal_rmse_loss(
            y_true,
            y_pred,
            alpha=focal_alpha,
            gamma=focal_gamma,
            rmse_weight=rmse_weight,
        )
    if loss_name == "dice_focal_rmse":
        return lambda y_true, y_pred: combined_dice_focal_rmse_loss(
            y_true,
            y_pred,
            alpha=focal_alpha,
            gamma=focal_gamma,
            rmse_weight=rmse_weight,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def _describe_loss(loss_config: dict[str, object]) -> str:
    loss_name = str(loss_config["loss_name"])
    rmse_weight = float(loss_config["rmse_weight"])
    if loss_name == "dice_rmse":
        return f"dice + {rmse_weight:g}*rmse"
    if loss_name == "focal_rmse":
        return (
            f"focal(alpha={float(loss_config['focal_alpha']):g}, "
            f"gamma={float(loss_config['focal_gamma']):g}) + {rmse_weight:g}*rmse"
        )
    return (
        f"dice + focal(alpha={float(loss_config['focal_alpha']):g}, "
        f"gamma={float(loss_config['focal_gamma']):g}) + {rmse_weight:g}*rmse"
    )


def _forward_training(model, images, targets):
    """Derive geometry solely from ignored cells, never positive locations."""
    if getattr(model, "requires_valid_mask", False):
        mask = torch.nn.functional.interpolate(
            (targets[..., 0] >= 0).float().unsqueeze(1), size=images.shape[-2:], mode="nearest")
        return model(images, valid_mask=mask)
    return model(images)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: LossFunction = combined_dice_rmse_loss,
) -> float:
    """Run one training epoch and return average train loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if getattr(model, "channels_last_input", False):
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(getattr(model, "amp_enabled", False))):
            preds = _forward_training(model, images, targets)
        preds = preds.float()
        loss = loss_fn(targets, preds)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; checkpoint is not updated.")
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: LossFunction = combined_dice_rmse_loss,
) -> tuple[float, float, float, float, float, float]:
    """Run validation and return average losses plus cell-level classification metrics."""
    model.eval()
    total_combined = 0.0
    total_dice = 0.0
    total_rmse = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if getattr(model, "channels_last_input", False):
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(getattr(model, "amp_enabled", False))):
            preds = _forward_training(model, images, targets)
        preds = preds.float()
        combined = loss_fn(targets, preds)
        dice = dice_loss(targets, preds)
        rmse = rmse_loss(targets, preds)
        valid_cells = targets[..., 0] >= 0
        predicted_positive = (preds[..., 0] > 0.5) & valid_cells
        true_positive = targets[..., 0] > 0.5
        total_tp += int(torch.count_nonzero(predicted_positive & true_positive).item())
        total_fp += int(torch.count_nonzero(predicted_positive & ~true_positive).item())
        total_fn += int(torch.count_nonzero(~predicted_positive & true_positive).item())

        total_combined += combined.item()
        total_dice += dice.item()
        total_rmse += rmse.item()
        n_batches += 1

    denom = max(1, n_batches)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return (
        total_combined / denom,
        total_dice / denom,
        total_rmse / denom,
        float(precision),
        float(recall),
        float(f1),
    )


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_training_curves(output_path: Path, history: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    val_f1 = [float(row["val_f1"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    axes[0].plot(epochs, train_loss, color="tab:blue", lw=1.8, label="train_loss")
    axes[0].plot(epochs, val_loss, color="tab:red", lw=1.8, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, val_f1, color="tab:green", lw=1.8, label="val_f1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_title("Validation grid F1 (p=0.5)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _infer_experiment_name(reference_paths: list[Path]) -> str | None:
    """Infer the current experiment name from a cwd or data path."""
    for reference_path in reference_paths:
        resolved = reference_path.resolve()
        parts = resolved.parts
        if "outputs" in parts:
            outputs_index = parts.index("outputs")
            if outputs_index + 1 < len(parts):
                return parts[outputs_index + 1]
    return None


def _default_checkpoint_path(data_path: Path, project_root: Path | None = None) -> Path:
    """Resolve the default checkpoint path for a training run."""
    root = project_root or Path(__file__).resolve().parents[2]
    experiment_name = _infer_experiment_name([Path.cwd(), data_path])
    if experiment_name is None:
        experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return root / "outputs" / experiment_name / "training_modification" / "best_model.pt"


def run_training(args: argparse.Namespace) -> Path:
    """Run end-to-end training and return best checkpoint path."""
    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, but got {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, but got {args.batch_size}")
    if args.lr <= 0:
        raise ValueError(f"--lr must be > 0, but got {args.lr}")

    loss_config = _build_loss_config(args)
    loss_fn = _build_loss_function(loss_config)
    model_name = _normalise_model_name(getattr(args, "model", "deepblink"))
    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(getattr(args, "threads", 4)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = resolve_device(args.device)
    train_loader, valid_loader = build_dataloaders(
        data_path=data_path,
        batch_size=args.batch_size,
        cell_size=4,
        development_only=bool(getattr(args, "development_only", False)),
    )

    model = _build_model(model_name, cell_size=4, device=device)
    model.amp_enabled = bool(getattr(args, "amp", False))
    model.channels_last_input = bool(getattr(args, "channels_last", False))
    if model.amp_enabled and device.type not in ("cuda", "xpu"):
        raise ValueError("--amp requires CUDA or XPU.")
    if model.channels_last_input:
        model = model.to(memory_format=torch.channels_last)
    parameter_count = _count_parameters(model)
    architecture_config = _architecture_config(model_name, model)
    architecture_config_json = json.dumps(architecture_config, sort_keys=True)
    input_size = getattr(getattr(train_loader, "dataset", None), "input_size", None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.checkpoint is None:
        checkpoint_path = _default_checkpoint_path(data_path)
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = checkpoint_path.parent / "training_metrics.csv"
    curves_png_path = checkpoint_path.parent / "training_curves.png"

    best_val_loss = float("inf")
    history: list[dict[str, object]] = []

    print(
        f"Start training | device={device} | data={data_path} | epochs={args.epochs} "
        f"| batch_size={args.batch_size} | lr={args.lr} | cell_size=4"
    )
    print(f"Selected model: {model_name} | parameters={parameter_count:,}")
    print(
        f"Selected loss: {loss_config['loss_name']} | "
        f"composition={_describe_loss(loss_config)}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
        )
        val_loss, val_dice_loss, val_rmse_loss, val_precision, val_recall, val_f1 = validate(
            model=model,
            loader=valid_loader,
            device=device,
            loss_fn=loss_fn,
        )

        history.append(
            {
                "epoch": epoch,
                "seed": seed,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "model_name": model_name,
                "parameter_count": parameter_count,
                "architecture_config": architecture_config_json,
                "loss_name": loss_config["loss_name"],
                "focal_alpha": loss_config["focal_alpha"],
                "focal_gamma": loss_config["focal_gamma"],
                "rmse_weight": loss_config["rmse_weight"],
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_dice_loss": float(val_dice_loss),
                "val_rmse": float(val_rmse_loss),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "val_f1": float(val_f1),
            }
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "cell_size": 4,
                    "input_size": input_size,
                    "preprocessing": "normalize_original_then_zero_pad_right_bottom" if input_size else "legacy",
                    "seed": seed,
                    "model_name": model_name,
                    "architecture_config": architecture_config,
                    "parameter_count": parameter_count,
                    "loss_name": loss_config["loss_name"],
                    "focal_alpha": loss_config["focal_alpha"],
                    "focal_gamma": loss_config["focal_gamma"],
                    "rmse_weight": loss_config["rmse_weight"],
                    "loss_config": loss_config,
                    "args": vars(args),
                },
                checkpoint_path,
            )

        # Persist progress even if a long training job is interrupted.
        _write_rows_csv(metrics_csv_path, history)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_f1={val_f1:.6f} "
            f"val_dice_loss={val_dice_loss:.6f} "
            f"val_rmse={val_rmse_loss:.6f} "
            f"{'(best)' if is_best else ''}"
        )

    _write_rows_csv(metrics_csv_path, history)
    _plot_training_curves(curves_png_path, history)

    print(f"Training metrics saved to: {metrics_csv_path}")
    print(f"Training curves saved to: {curves_png_path}")
    print(f"Training completed. Best checkpoint: {checkpoint_path}")
    return checkpoint_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for training."""
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
