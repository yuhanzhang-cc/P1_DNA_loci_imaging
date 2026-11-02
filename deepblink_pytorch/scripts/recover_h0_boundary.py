"""Audited evaluation-only overlay for the sealed v4 H0 epoch-60 incident.

Run with the original CUDA venv. Never overwrite files listed in bundle_manifest.
The original training identity is retained; the effective decoder is recorded in
a separate numerical-evaluation addendum that is included in the result archive.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deepblink_pytorch/scripts"))
# Set the original CUDA environment before importing torch.
import cloud_recipe_train as runner

PATCH_ID = "h0-float32-fov-20261102-v1"
MANIFEST_SHA256 = "d40e993fca53e5478b7d2988992ffd8344cb8803af95eac984166543651c5b90"
OUTPUT = ROOT / "outputs/research_v4_cuda"
QUEUE = OUTPUT / "baseline_h0"
TRIAL = QUEUE / "H0_cuda_reference/seed_43"
ADDENDUM = QUEUE / "numerical_decode_addendum.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def filter_serialized_bounds(coords, scores, shape):
    """Drop only exact upper-edge float32 round-out; reject other invalid data."""
    import numpy as np
    shape = np.asarray(shape)
    if shape.shape != (2,) or not np.all(shape > 0) or not np.all(shape == shape.astype(int)):
        raise ValueError("Invalid original image shape")
    if coords.dtype != np.float32 or scores.dtype != np.float32:
        raise ValueError("Original decoder must return float32")
    if coords.shape != (len(scores), 2) or not np.isfinite(coords).all() or not np.isfinite(scores).all():
        raise ValueError("Malformed/nonfinite candidates")
    if np.any(coords < 0) or np.any(coords > shape):
        raise ValueError("Unexpected candidate error; not an upper-edge round-out")
    keep = (coords < shape).all(1)
    return coords[keep], scores[keep], ~keep


def overlay_decoder(original, on_drop):
    def decode(prediction, shape, **kwargs):
        coords, scores = original(prediction, shape, **kwargs)
        kept_coords, kept_scores, dropped = filter_serialized_bounds(coords, scores, shape)
        if dropped.any():
            on_drop(dict(shape=list(map(int, shape)), coordinates=coords[dropped].tolist(),
                         scores=scores[dropped].tolist(), removed=int(dropped.sum())))
        return kept_coords, kept_scores
    return decode


def verify_environment():
    import torch
    from research_v4.common import read_json, sha256
    if sha256(ROOT / "bundle_manifest.json") != MANIFEST_SHA256:
        raise ValueError("Recovery is only for the original ce2d1821ceec training bundle")
    _, _, corpus = runner.verify_bundle()
    protocol = read_json(QUEUE / "protocol.json")
    if not torch.cuda.is_available():
        raise RuntimeError("Recovery requires the original CUDA device; no CPU fallback")
    if (protocol["torch"] != str(torch.__version__) or protocol["gpu"] != torch.cuda.get_device_name(0)
            or protocol["manifest_sha256"] != MANIFEST_SHA256 or protocol["device"] != "cuda"):
        raise ValueError("Original H0 GPU/environment identity changed")
    runner.check_complete(OUTPUT, "baseline_r0")
    return corpus


def verify_protected(record):
    from research_v4.common import sha256
    if record["patch_id"] != PATCH_ID or record["overlay_sha256"] != sha256(Path(__file__)):
        raise ValueError("Recovery overlay changed after registration")
    for relative, digest in record["protected_files"].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT) or sha256(path) != digest:
            raise ValueError("Previously completed result changed: " + relative)


def audit(corpus):
    import torch
    from research_v4 import engine
    from research_v4.common import dump, read_json, sha256
    from research_v4.evaluation import load_candidates
    if ADDENDUM.exists():
        record = read_json(ADDENDUM)
        verify_protected(record)
        if sha256(QUEUE / "boundary_overlay_source.py") != record["overlay_sha256"]:
            raise ValueError("Registered recovery source changed")
        print("Existing recovery addendum and completed results verified", flush=True)
        return record
    runner.check_complete(OUTPUT, "recipe")
    if read_json(QUEUE / "H0_cuda_reference/seed_42/status.json")["status"] != "complete":
        raise ValueError("Expected completed H0 seed 42")
    recovery = torch.load(TRIAL / "recovery.pt", map_location="cpu", weights_only=True)
    config = read_json(TRIAL / "registered_config.json")
    if (recovery["config"] != config or config["key"] != "H0" or config["seed"] != 43
            or config["engineering_only"] or config["source_lock"] != MANIFEST_SHA256
            or recovery["epoch"] != 59 or recovery["updates"] != 1298):
        raise ValueError("Not the expected epoch-59 H0 recovery point; inspect before proceeding")
    if len(recovery["history"]) != 59 or [r["epoch"] for r in recovery["snapshots"]] != list(range(5, 60, 5)):
        raise ValueError("Recovery history/snapshot budget differs")
    if read_json(TRIAL / "snapshots.json") != recovery["snapshots"]:
        raise ValueError("Snapshot registry differs from committed recovery")
    protected = {}
    # All R0 files and the completed H0 trial remain byte-identical after recovery.
    for folder in (OUTPUT / "baseline_r0", QUEUE / "H0_cuda_reference/seed_42"):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = sha256(path)
    valid = corpus.splits["valid"]
    for folder in (QUEUE / "H0_cuda_reference/seed_42", TRIAL):
        for item in read_json(folder / "snapshots.json"):
            snapshot = folder / f"epoch_{item['epoch']:03d}"
            if sha256(snapshot / "model.pt") != item["weight_sha256"] or sha256(snapshot / "candidates.npz") != item["candidates_sha256"]:
                raise ValueError("Committed snapshot hash mismatch")
            for (coords, scores), shape in zip(load_candidates(snapshot / "candidates.npz", valid), valid["original_shape"]):
                _, _, dropped = filter_serialized_bounds(coords, scores, shape)
                if dropped.any():
                    raise ValueError("Previously accepted candidates need a broader audit")
            for path in snapshot.iterdir():
                if path.is_file():
                    protected[path.relative_to(ROOT).as_posix()] = sha256(path)
    for name in ("registered_config.json", "task_protocol.json", "initialization.json"):
        path = TRIAL / name
        protected[path.relative_to(ROOT).as_posix()] = sha256(path)
    failed = TRIAL / "epoch_060/model.pt"
    payload = torch.load(failed, map_location="cpu", weights_only=True)
    if payload["epoch"] != 60 or payload["seed"] != 43 or payload["model_name"] != engine.MODELS["H0"]:
        raise ValueError("Unexpected failed epoch-60 checkpoint")
    events = []
    engine.configure_device("cuda")
    with engine.preserve_rng():
        model = engine.make_model("H0", "cuda")
        model.load_state_dict(payload["model_state_dict"])
        if engine.model_hash(model) != payload["model_state_sha256"]:
            raise ValueError("Failed model tensor hash differs")
        original = engine.decode_candidates
        index = [0]
        def decode(prediction, shape, **kwargs):
            coords, scores = original(prediction, shape, **kwargs)
            kept_coords, kept_scores, dropped = filter_serialized_bounds(coords, scores, shape)
            if dropped.any():
                events.append(dict(sample_id=str(valid["sample_id"][index[0]]), shape=list(map(int, shape)),
                                   coordinates=coords[dropped].tolist(), scores=scores[dropped].tolist()))
            index[0] += 1
            return kept_coords, kept_scores
        engine.decode_candidates = decode
        try:
            engine.infer_candidates(model, valid, "cuda")
        finally:
            engine.decode_candidates = original
    if not events:
        raise ValueError("Failed snapshot did not reproduce the boundary issue; stop for investigation")
    record = dict(patch_id=PATCH_ID, created_utc=utc_now(), original_manifest_sha256=MANIFEST_SHA256,
                  overlay_sha256=sha256(Path(__file__)),
                  correction="Filter strict half-open FOV in serialized float32; no clipping, score or label changes",
                  training_recipe_changed=False, test_arrays_loaded=False, last_committed_epoch=59,
                  recovery_sha256=sha256(TRIAL / "recovery.pt"), failed_model_sha256=sha256(failed),
                  expected_epoch60_tensor_sha256=payload["model_state_sha256"],
                  actual_cuda_reproduction=events, protected_files=protected)
    # The original packer includes this source and the addendum in the results.
    source = QUEUE / "boundary_overlay_source.py"
    if source.exists() and sha256(source) != record["overlay_sha256"]:
        raise ValueError("A different recovery source is already registered")
    source.write_bytes(Path(__file__).read_bytes())
    dump(ADDENDUM, record)
    print(json.dumps(dict(audit_passed=True, reproduced_images=len(events), recovery_epoch=59,
                          next_seed=43, patch_id=PATCH_ID)), flush=True)
    return record


def resume(record):
    from research_v4 import engine
    from research_v4.common import dump, read_json, sha256
    verify_protected(record)
    import torch
    recovery = torch.load(TRIAL / "recovery.pt", map_location="cpu", weights_only=True)
    if recovery["epoch"] < 59:
        raise ValueError("Recovery checkpoint moved backwards")
    if recovery["epoch"] == 59 and sha256(TRIAL / "recovery.pt") != record["recovery_sha256"]:
        raise ValueError("Original epoch-59 recovery checkpoint changed")
    if recovery["epoch"] > 59 and not (QUEUE / "epoch60_replay_check.json").exists():
        raise ValueError("An unverified runner advanced the recovery checkpoint")
    original_decode, original_save = engine.decode_candidates, engine.save_torch
    log = QUEUE / "boundary_roundout_events.jsonl"
    context = {}
    def on_drop(event):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(utc=utc_now(), patch_id=PATCH_ID, **context, **event)) + "\n")
    def save(path, payload):
        if Path(path).name == "model.pt":
            context.update(seed=payload["seed"], epoch=payload["epoch"])
            if payload["seed"] == 43 and payload["epoch"] == 60:
                if payload["model_state_sha256"] != record["expected_epoch60_tensor_sha256"]:
                    raise ValueError("Replayed epoch-60 tensors changed; stop for investigation")
                dump(QUEUE / "epoch60_replay_check.json", dict(exact_tensor_replay=True,
                     tensor_sha256=payload["model_state_sha256"], utc=utc_now()))
        return original_save(path, payload)
    engine.decode_candidates = overlay_decoder(original_decode, on_drop)
    engine.save_torch = save
    previous_argv = sys.argv
    try:
        sys.argv = [str(runner.__file__), "--train", "--queue", "baseline_h0", "--device", "cuda", "--resume"]
        runner.main()
    finally:
        sys.argv = previous_argv
        engine.decode_candidates, engine.save_torch = original_decode, original_save
    verify_protected(record)
    replay = read_json(QUEUE / "epoch60_replay_check.json")
    if not replay["exact_tensor_replay"]:
        raise ValueError("Missing exact replay verification")
    if not any(sha256(p) == record["failed_model_sha256"] for p in (TRIAL / "orphaned_snapshots").glob("epoch_060_*/model.pt")):
        raise ValueError("Original failed snapshot was not preserved")
    dump(QUEUE / "boundary_recovery_complete.json", dict(patch_id=PATCH_ID, completed_utc=utc_now(),
         completed_results_byte_identical=True, epoch60_tensor_exact=True, test_evaluated=False))
    subprocess.run([sys.executable, str(ROOT / "deepblink_pytorch/scripts/cloud_job_control.py"),
                    "pack", "--queue", "baseline_h0"], cwd=ROOT, check=True)


def main():
    from filelock import FileLock
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--start", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.start:
        # Audit synchronously before detaching. The child also rechecks all gates.
        subprocess.run([sys.executable, str(Path(__file__)), "--audit-only"], cwd=ROOT, check=True)
        log = OUTPUT / "baseline_h0_boundary_recovery.log"
        with log.open("ab", buffering=0) as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--resume"], cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"Started audited H0 recovery PID {process.pid}; log: {log}")
        return
    with FileLock(str(OUTPUT / ".h0_boundary_recovery.lock"), timeout=0):
        # Avoid auditing while an original runner is updating snapshots.
        with FileLock(str(OUTPUT / ".gpu_runner.lock"), timeout=0):
            corpus = verify_environment()
            record = audit(corpus)
        if args.resume:
            resume(record)


if __name__ == "__main__":
    main()
