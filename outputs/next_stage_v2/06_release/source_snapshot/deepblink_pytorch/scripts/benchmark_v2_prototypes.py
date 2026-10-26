"""Forward-only resource check on development images; no test labels or scores."""
import time
import numpy as np
import torch
from deepblink_torch.cli import predict
from deepblink_torch.preprocessing import prepare_image
from optimize_v2 import OUT, ROOT, load_development, write_csv, write_json


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    dev = load_development("valid")
    examples = {}
    for raw, shape, sid in zip(dev["x"], dev["original_shape"], dev["sample_id"]):
        h, w = map(int, shape)
        if h not in examples:
            examples[h] = (raw[:h, :w], str(sid))
    checkpoints = {"deepblink": ROOT / "outputs/261024_clean220_random/seed_42/best_model.pt"}
    checkpoints.update({name: OUT / "02_architecture" / name / "seed_42/best_model.pt"
                        for name in ("deeploci_evidence", "deeploci_local", "deeploci_feature_evidence", "deepblink_masked_se")})
    rows = []
    for name, path in checkpoints.items():
        model, _ = predict._load_model(path, torch.device("xpu"))
        for size, (raw, sid) in sorted(examples.items()):
            tensor = torch.from_numpy(prepare_image(raw, 512)[None, None]).to("xpu")
            for _ in range(5):
                predict._forward_with_geometry(model, tensor, size, size)
            torch.xpu.synchronize()
            torch.xpu.reset_peak_memory_stats()
            seconds = []
            for _ in range(20):
                torch.xpu.synchronize()
                started = time.perf_counter()
                prediction = predict._forward_with_geometry(model, tensor, size, size)
                torch.xpu.synchronize()
                seconds.append(time.perf_counter()-started)
                del prediction
            rows.append(dict(model=name, raw_height=size, canvas_height=512, development_sample_id=sid,
                             median_ms=1000*float(np.median(seconds)),
                             p25_ms=1000*float(np.quantile(seconds, .25)),
                             p75_ms=1000*float(np.quantile(seconds, .75)),
                             torch_peak_allocated_MiB=torch.xpu.max_memory_allocated()/2**20,
                             warmup=5, repeats=20, batch=1, precision="float32", scope="forward_only"))
        del model, tensor
        torch.xpu.empty_cache()
    write_csv(OUT / "02_architecture/runtime_benchmark.csv", rows)
    write_json(OUT / "02_architecture/runtime_benchmark_metadata.json",
               dict(device=torch.xpu.get_device_name(0), torch=torch.__version__, threads=4,
                    excludes="disk IO and normalization and decoding and overlays",
                    synchronized=True, channels_last=False, test_access=False,
                    limitation="Initial local resource estimate; shared host and two development FOVs only"))
    print(rows, flush=True)


if __name__ == "__main__":
    main()
