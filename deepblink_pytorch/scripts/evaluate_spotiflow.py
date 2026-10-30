"""Infer unfiltered validation candidates or evaluate registered checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np

from deepblink_torch.experiment_log import ExperimentLog, sha256, write_json
from deepblink_torch.spotiflow_bridge import (
    ARCHIVE_SHA256, evaluate_checkpoints, load_validation, read_candidates,
    save_candidates, spotiflow_candidates, write_csv)

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "outputs/261024_clean220_random/dna_clean220_random.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    infer = sub.add_parser("infer")
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--pretrained", choices=["general"])
    source.add_argument("--model-folder", type=Path)
    infer.add_argument("--epoch", type=int, required=True)
    infer.add_argument("--seed", type=int, choices=[42, 43, 44])
    infer.add_argument("--run-id", required=True, help="Shared identity of one training run, across its snapshots")
    infer.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    infer.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--runs", type=Path, nargs="+", required=True,
                          help="Inference output directories, one per registered checkpoint")
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Frozen input archive changed.")
    config = dict(vars(args), archive_sha256=ARCHIVE_SHA256, test_arrays_loaded=False)
    config = {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, list) else v
              for k, v in config.items()}
    with ExperimentLog(args.output, config) as log:
        data = load_validation(ARCHIVE)
        if args.action == "infer":
            from importlib.metadata import version
            from spotiflow.model import Spotiflow
            if args.pretrained and args.epoch != 0:
                raise ValueError("Pretrained zero-shot inference requires epoch=0.")
            if not args.pretrained and args.epoch not in range(5, 101, 5):
                raise ValueError("Fine-tuned snapshots must use epochs 5..100, every 5 epochs.")
            if not args.pretrained and args.seed is None:
                raise ValueError("Fine-tuned inference requires its training seed.")
            log.event("runtime", spotiflow=version("spotiflow"), device=args.device)
            model = (Spotiflow.from_pretrained(args.pretrained) if args.pretrained
                     else Spotiflow.from_folder(str(args.model_folder)))
            if args.model_folder:
                write_json(log.output / "model_hashes.json", {
                    p.relative_to(args.model_folder).as_posix(): sha256(p)
                    for p in sorted(args.model_folder.rglob("*")) if p.is_file()})
            candidates = []
            with np.load(ARCHIVE, allow_pickle=True) as archive:
                images = archive["x_valid"]
                for image, shape, sample in zip(images, data["original_shape"], data["sample_id"]):
                    h, w = map(int, shape)
                    candidates.append(spotiflow_candidates(model, image[:h, :w], args.device))
                    log.event("image_complete", sample_id=str(sample), candidates=len(candidates[-1][0]))
            save_candidates(log.output / "candidates.npz", candidates, data["sample_id"], data["original_shape"])
            write_json(log.output / "candidate_integrity.json", dict(
                sha256=sha256(log.output / "candidates.npz"), normalization="Spotiflow auto",
                prob_thresh=0., min_distance=1, exclude_border=False, subpix=True))
        else:
            checkpoints = []
            identities = set()
            for run in args.runs:
                metadata = json.loads((run / "config.json").read_text(encoding="utf-8"))
                status = json.loads((run / "status.json").read_text(encoding="utf-8"))
                if metadata.get("archive_sha256") != ARCHIVE_SHA256 or status.get("event") != "completed":
                    raise ValueError("Inference did not complete against the frozen archive.")
                identities.add((metadata.get("run_id"), metadata.get("seed"), metadata.get("pretrained"), metadata.get("device")))
                integrity = json.loads((run / "candidate_integrity.json").read_text(encoding="utf-8"))
                if integrity["sha256"] != sha256(run / "candidates.npz"):
                    raise ValueError("Saved candidates changed after inference.")
                checkpoints.append((int(metadata["epoch"]), read_candidates(run / "candidates.npz", data)))
            if len(identities) != 1:
                raise ValueError("Do not mix models, seeds, devices, or training runs during checkpoint selection.")
            rows, selected, per_image = evaluate_checkpoints(checkpoints, data)
            write_csv(log.output / "threshold_metrics.csv", rows, list(rows[0]))
            write_csv(log.output / "per_image_metrics.csv", per_image, list(per_image[0]))
            write_json(log.output / "selection.json", dict(**selected, radius_px=3,
                       checkpoint_count=len(checkpoints), threshold_count=31,
                       full_20_checkpoint_budget=len(checkpoints) == 20, development_validation_only=True))
            log.event("evaluation_complete", **selected)


if __name__ == "__main__":
    main()
