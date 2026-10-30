"""Native reader + pretrained inference + one synthetic CPU fine-tuning epoch."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np

from deepblink_torch.experiment_log import ExperimentLog, write_json
from deepblink_torch.spotiflow_bridge import spotiflow_candidates

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/closing_preparation/spotiflow_data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/closing_preparation/spotiflow_smoke")
    args = parser.parse_args()
    config = dict(spotiflow=version("spotiflow"), device="cpu", test_arrays_loaded=False,
                  performance_experiment=False, training="one synthetic epoch only")
    with ExperimentLog(args.output, config) as log:
        with (log.output / "runtime.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
            import torch
            import lightning.pytorch as pl
            import tifffile
            from spotiflow.model import Spotiflow
            from spotiflow.model.config import SpotiflowTrainingConfig
            from spotiflow.utils import get_data

            torch.set_num_threads(4)
            pl.seed_everything(42, workers=True)
            train_x, train_y, val_x, val_y = get_data(args.data, normalize=False, include_test=False)
            if (len(train_x), len(val_x)) != (176, 22):
                raise ValueError("Native Spotiflow reader changed split counts.")
            for folder, images, truths in [("train", train_x, train_y), ("val", val_x, val_y)]:
                paths = sorted((args.data / folder).glob("*.tif"))
                for path, image, truth in zip(paths, images, truths):
                    csv_truth = np.loadtxt(path.with_suffix(".csv"), delimiter=",", skiprows=1, ndmin=2)
                    if not np.array_equal(tifffile.imread(path), image) or not np.allclose(csv_truth, truth):
                        raise ValueError("Native reader changed pixels/coordinates or file order.")
            log.event("native_reader_verified", train_images=176, valid_images=22)
            model = Spotiflow.from_pretrained("general", inference_mode=False,
                                              cache_dir=ROOT / ".venv/spotiflow_models")
            before = time.perf_counter()
            candidates = []
            for size in (256, 512):
                index = next(i for i, image in enumerate(train_x) if image.shape == (size, size))
                coords, scores = spotiflow_candidates(model, train_x[index], "cpu")
                candidates.append(dict(shape=[size, size], candidates=len(coords),
                                       score_min=float(scores.min()) if len(scores) else None))
            infer_seconds = time.perf_counter() - before
            log.event("pretrained_inference_verified", images=2, seconds=infer_seconds)
            # Synthetic images exercise backward/optimizer/checkpoint APIs only.
            yy, xx = np.mgrid[:64, :64]
            synthetic = [np.exp(-((yy-y)**2 + (xx-x)**2)/4.).astype(np.float32)
                         for y, x in [(20., 30.), (35., 25.)]]
            spots = [np.array([[20., 30.]], dtype=np.float32), np.array([[35., 25.]], dtype=np.float32)]
            training = SpotiflowTrainingConfig(crop_size=64, batch_size=1, num_train_samples=1,
                                               num_epochs=1, optimize_threshold=False)
            before = time.perf_counter()
            model.fit(synthetic[:1], spots[:1], synthetic[1:], spots[1:], augment_train=False,
                      save_dir=log.output / "synthetic_model", train_config=training,
                      device="cpu", logger="tensorboard", num_workers=0)
            restored = Spotiflow.from_folder(log.output / "synthetic_model")
            spotiflow_candidates(restored, synthetic[0], "cpu")
            fit_seconds = time.perf_counter() - before
            write_json(log.output / "smoke_results.json", dict(
                native_reader_pairs=198, pretrained_training_images=candidates,
                inference_seconds=infer_seconds, synthetic_fit_seconds=fit_seconds,
                native_fit_and_reload_verified=True, test_evaluated=False,
                scientific_performance_measured=False, torch=torch.__version__))
            log.event("synthetic_fit_reload_verified", seconds=fit_seconds)


if __name__ == "__main__":
    main()
