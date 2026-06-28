"""Train command for deepblink_torch CLI."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from deepblink_torch import SpotsDataset
from deepblink_torch import UNetDeepBlink
from deepblink_torch import combined_dice_rmse_loss
from deepblink_torch import dice_loss
from deepblink_torch import rmse_loss


def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add training arguments to an argparse parser."""
    parser.add_argument(
        "--data",
        required=True,
        help="Path to deepBlink .npz dataset containing train/valid/test splits.",
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
        choices=["cuda", "cpu"],
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
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train UNetDeepBlink with PyTorch.")
    add_train_arguments(parser)
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    """Resolve runtime device, falling back to CPU if CUDA is unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda but CUDA is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested)


def build_dataloaders(
    data_path: Path,
    batch_size: int,
    cell_size: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train/valid DataLoaders from one deepBlink npz file."""
    train_dataset = SpotsDataset(
        name=data_path,
        cell_size=cell_size,
        subset="train",
    )
    valid_dataset = SpotsDataset(
        name=data_path,
        cell_size=cell_size,
        subset="valid",
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


def train_one_epoch(
    model: UNetDeepBlink,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average train loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = combined_dice_rmse_loss(targets, preds)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: UNetDeepBlink,
    loader: DataLoader,
    device: torch.device,
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

        preds = model(images)
        combined = combined_dice_rmse_loss(targets, preds)
        dice = dice_loss(targets, preds)
        rmse = rmse_loss(targets, preds)
        predicted_positive = preds[..., 0] > 0.5
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
    axes[1].set_title("Validation F1")
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

    device = resolve_device(args.device)
    train_loader, valid_loader = build_dataloaders(
        data_path=data_path,
        batch_size=args.batch_size,
        cell_size=4,
    )

    model = UNetDeepBlink(cell_size=4).to(device)
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

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_dice_loss, val_rmse_loss, val_precision, val_recall, val_f1 = validate(
            model=model,
            loader=valid_loader,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
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
                    "args": vars(args),
                },
                checkpoint_path,
            )

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
