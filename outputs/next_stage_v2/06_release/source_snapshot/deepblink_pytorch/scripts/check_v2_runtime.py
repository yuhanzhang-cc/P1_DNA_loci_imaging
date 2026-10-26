"""One full-size XPU training update per architecture, not an accuracy trial."""
import json
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from deepblink_torch.cli import train
from optimize_v2 import OUT, load_development, write_json
from deepblink_torch.preprocessing import prepare_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["deeploci_evidence", "deeploci_local"])
    parser.add_argument("--output-name", default="xpu_runtime_check.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = load_development("train")
    images, targets = [], []
    from deepblink_torch.data_protocol import get_prediction_matrix
    indices = [i for size in (256, 512) for i in np.flatnonzero(data["original_shape"][:, 0] == size)[:4]]
    for index in indices:
        raw, coords, shape = [data[key][index] for key in ("x", "y", "original_shape")]
        h, w = map(int, shape)
        images.append(torch.from_numpy(prepare_image(raw[:h, :w], 512)))
        target = torch.from_numpy(get_prediction_matrix(coords, 512, 4)).float()
        target[h//4:, :, 0] = -1
        target[:, w//4:, 0] = -1
        targets.append(target)
    image = torch.stack(images)[:, None].to("xpu").contiguous(memory_format=torch.channels_last)
    target = torch.stack(targets).to("xpu")
    rows = []
    for name in args.models:
        torch.manual_seed(42)
        torch.xpu.manual_seed_all(42)
        model = train._build_model(name, cell_size=4, device=torch.device("xpu"))
        model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        started = time.perf_counter()
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            pred = train._forward_training(model, image, target)
        loss = train.combined_dice_rmse_loss(target, pred.float())
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all())
                                                   for p in model.parameters() if p.grad is not None)
        if not finite:
            raise FloatingPointError(name)
        optimizer.step()
        torch.xpu.synchronize()
        row = dict(model=name, loss=float(loss.detach()), parameters=train._count_parameters(model),
                   full_batch_shape=list(image.shape), backward_finite=finite,
                   seconds_including_first_compile=time.perf_counter()-started,
                   amp="bfloat16", evidence_sampling_precision="float32")
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer, pred, loss
        torch.xpu.empty_cache()
    write_json(OUT / "00_protocol" / args.output_name, rows)


if __name__ == "__main__":
    main()
