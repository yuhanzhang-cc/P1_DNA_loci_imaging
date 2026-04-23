"""Main CLI entry point for deepblink_torch."""

from __future__ import annotations

import argparse

from . import create
from . import predict
from . import train


def build_parser() -> argparse.ArgumentParser:
    """Build top-level CLI parser with subcommands."""
    parser = argparse.ArgumentParser(prog="deepblink-torch", description="deepblink_torch CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create .npz dataset from images + CSV labels.")
    create.add_create_arguments(create_parser)

    train_parser = subparsers.add_parser("train", help="Train UNetDeepBlink model.")
    train.add_train_arguments(train_parser)

    predict_parser = subparsers.add_parser("predict", help="Run model inference on image(s).")
    predict.add_predict_arguments(predict_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "create":
        create.run_create(args)
        return
    if args.command == "train":
        train.run_training(args)
        return
    if args.command == "predict":
        predict.run_predict(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

