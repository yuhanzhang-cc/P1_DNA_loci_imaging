"""Train command for deepblink_torch CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

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
            "artifacts/checkpoints/best_model.pt relative to project root."
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
) -> tuple[float, float, float]:
    """Run validation and return average (combined, dice, rmse) losses."""
    model.eval()
    total_combined = 0.0
    total_dice = 0.0
    total_rmse = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images)
        combined = combined_dice_rmse_loss(targets, preds)
        dice = dice_loss(targets, preds)
        rmse = rmse_loss(targets, preds)

        total_combined += combined.item()
        total_dice += dice.item()
        total_rmse += rmse.item()
        n_batches += 1

    denom = max(1, n_batches)
    return total_combined / denom, total_dice / denom, total_rmse / denom


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
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_path = project_root / "artifacts" / "checkpoints" / "best_model.pt"
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

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
        val_loss, val_dice_loss, val_rmse_loss = validate(
            model=model,
            loader=valid_loader,
            device=device,
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
            f"val_dice_loss={val_dice_loss:.6f} "
            f"val_rmse_loss={val_rmse_loss:.6f} "
            f"{'(best)' if is_best else ''}"
        )

    print(f"Training completed. Best checkpoint: {checkpoint_path}")
    return checkpoint_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for training."""
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
