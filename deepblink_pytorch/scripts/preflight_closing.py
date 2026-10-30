"""Two optimizer steps on two training images, never a performance experiment."""
import argparse
import gc
from pathlib import Path
import time

import numpy as np
import torch

from deepblink_torch.cli import predict
from deepblink_torch.data_protocol import get_prediction_matrix
from deepblink_torch.experiment_log import ExperimentLog, sha256, write_json
from deepblink_torch.losses import combined_dice_rmse_loss
from deepblink_torch.networks.unet import UNetDeepBlink
from deepblink_torch.preprocessing import prepare_image
from deepblink_torch.spotiflow_bridge import ARCHIVE_SHA256

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "outputs/261024_clean220_random/dna_clean220_random.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "xpu"], default="xpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/closing_preparation/control_smoke")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise ValueError("Frozen input archive changed.")
    config = dict(device=args.device, seed=42, optimizer_steps=2, test_arrays_loaded=False,
                  performance_experiment=False, archive_sha256=ARCHIVE_SHA256)
    with ExperimentLog(args.output, config) as log:
        log.event("runtime", torch=torch.__version__, device=args.device,
                  device_name=torch.xpu.get_device_name() if args.device == "xpu" else "CPU")
        write_json(log.output / "source_hashes.json", {
            p.relative_to(ROOT).as_posix(): sha256(p) for p in
            sorted((ROOT / "deepblink_pytorch/deepblink_torch").rglob("*.py"))
        } | {Path(__file__).relative_to(ROOT).as_posix(): sha256(__file__)})
        with np.load(ARCHIVE, allow_pickle=True) as archive:
            shapes = archive["original_shape_train"]
            indices = [int(np.flatnonzero(np.all(shapes == size, axis=1))[0]) for size in (256, 512)]
            ids = archive["sample_id_train"][indices].tolist()
            raw, truth = archive["x_train"][indices], archive["y_train"][indices]
            images, targets, masks = [], [], []
            for image, coords, (h, w) in zip(raw, truth, shapes[indices]):
                h, w = int(h), int(w)
                images.append(prepare_image(image[:h, :w], 512))
                target = get_prediction_matrix(coords, 512, 4).astype(np.float32)
                target[h//4:, :, 0] = -1
                target[:, w//4:, 0] = -1
                targets.append(target)
                mask = np.zeros((1, 512, 512), dtype=np.float32)
                mask[:, :h, :w] = 1
                masks.append(mask)
        x = torch.from_numpy(np.stack(images)[:, None]).to(args.device).contiguous(memory_format=torch.channels_last)
        y = torch.from_numpy(np.stack(targets)).to(args.device)
        mask = torch.from_numpy(np.stack(masks)).to(args.device)
        results = []
        base_count = sum(p.numel() for p in UNetDeepBlink().parameters())
        for name, branch in [("deeploci_grid_control", "grid_projection"), ("deeploci_global_query", "query")]:
            torch.manual_seed(42)
            if args.device == "xpu":
                torch.xpu.manual_seed_all(42)
            model = predict._build_model_for_checkpoint(name, cell_size=4, device=torch.device(args.device))
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
            start, losses = time.perf_counter(), []
            for step in range(2):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "xpu"):
                    pred = model(x, valid_mask=mask)
                loss = combined_dice_rmse_loss(y, pred.float())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite smoke loss.")
                loss.backward()
                branch_params = list(getattr(model, branch).parameters())
                if step == 1 and not any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) for p in branch_params):
                    raise ValueError("Control branch did not receive a finite nonzero gradient.")
                optimizer.step()
                losses.append(float(loss.detach()))
                log.event("optimizer_step", model=name, step=step + 1, loss=losses[-1])
            model.eval()
            with torch.no_grad():
                prediction = model(x, mask)
                if prediction.shape != (2, 128, 128, 3) or not torch.isfinite(prediction).all():
                    raise ValueError("Invalid mixed-size forward output.")
            del prediction
            model.cpu().eval()
            checkpoint = log.output / f"{name}.pt"
            torch.save(dict(model_name=name, cell_size=4, input_size=512, model_state_dict=model.state_dict()), checkpoint)
            restored, _ = predict._load_model(checkpoint, torch.device("cpu"))
            restored.eval()
            # Small fixed tensor isolates serialization from accelerator roundoff.
            probe = torch.zeros(1, 1, 32, 32)
            with torch.no_grad():
                expected = model(probe, torch.ones_like(probe))
                actual = restored(probe, torch.ones_like(probe))
            if not torch.equal(expected, actual):
                raise ValueError("CPU weight reload changed predictions.")
            results.append(dict(model=name, parameters=sum(p.numel() for p in model.parameters()),
                                added_parameters=sum(p.numel() for p in model.parameters()) - base_count,
                                losses=losses, seconds=time.perf_counter() - start,
                                cpu_reload_max_abs=float((expected - actual).abs().max())))
            del optimizer, model, restored
            gc.collect()
            if args.device == "xpu":
                torch.xpu.empty_cache()
        write_json(log.output / "smoke_results.json", dict(training_sample_ids=ids, results=results,
                   test_evaluated=False, full_training_started=False))


if __name__ == "__main__":
    main()
