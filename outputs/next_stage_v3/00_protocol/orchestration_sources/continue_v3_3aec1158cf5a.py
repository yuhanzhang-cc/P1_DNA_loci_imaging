"""Durable bounded orchestration after an already running v3 phase-one job.

No hyperparameters are changed here. The trainer independently enforces every
model/seed/budget gate. A failed trial stops orchestration for investigation.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/next_stage_v3"
PYTHON = ROOT / ".venv/clean220_xpu/Scripts/python.exe"
TRAINER = ROOT / "deepblink_pytorch/scripts/optimize_v3.py"
STOP = OUT / "STOP_REQUESTED"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def process_alive(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def check_source():
    configs = list((OUT / "01_screen").glob("*/seed_*/registered_config.json"))
    if not configs:
        raise ValueError("No authorized phase-one configuration exists.")
    for path in configs:
        config = read_json(path)
        if config["protocol_id"] != "pooled220_methods_v3" or config["test_arrays_loaded"]:
            raise ValueError("Unexpected protocol or test access.")
        for name, digest in config["source_hashes"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError("Training source changed: " + name)
    return len(configs)


def status(stage, **extra):
    write_json(OUT / "controller_status.json", dict(stage=stage, pid=os.getpid(),
                 updated_unix=time.time(), source_sha256=sha256(Path(__file__)), **extra))
    print(stage, json.dumps(extra), flush=True)


def stopped():
    if STOP.exists():
        raise InterruptedError("Explicit STOP_REQUESTED marker exists.")


def run_command(arguments, log_name):
    stopped()
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with (OUT / log_name).open("a", encoding="utf-8") as log:
        child = subprocess.Popen([str(PYTHON), *map(str, arguments)], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        status("running_stage", child_pid=child.pid, command=list(map(str, arguments)), log=log_name)
        while child.poll() is None:
            if STOP.exists():
                child.terminate()
                child.wait(timeout=30)
                raise InterruptedError("Requested stop; the current epoch recovery file was retained.")
            time.sleep(10)
            status("running_stage", child_pid=child.pid, command=list(map(str, arguments)), log=log_name)
        if child.returncode:
            raise RuntimeError(f"Stage exited {child.returncode}; inspect {log_name}")


def run_training(models, seeds, log_name, resume=False):
    current = check_source()
    missing = sum(not (OUT / "01_screen" / key / f"seed_{seed}/registered_config.json").exists()
                  for key in models for seed in seeds)
    if current + missing > 12:
        raise ValueError("The twelve-task budget would be exceeded.")
    run_command([TRAINER, "train", "--models", *models, "--seeds", *seeds, "--device", "xpu"] +
                (["--resume"] if resume else []), log_name)


def finish_phase1_after_infrastructure_loss():
    """Continue only a dead process's unchanged, checkpointed configuration."""
    for key in ("R0", "H0", "R1", "S1"):
        stopped()
        folder = OUT / "01_screen" / key / "seed_42"
        path = folder / "status.json"
        if path.exists():
            previous = read_json(path)
            if previous["status"] == "complete":
                continue
            # Explicit interrupts, nonfinite losses and assertions require an
            # agent audit. Do not retry them as infrastructure failures.
            if previous["status"] != "running" or process_alive(previous["pid"]):
                raise RuntimeError(f"{key} needs investigation before recovery: {previous}")
            if not (folder / "resume.pt").exists():
                raise RuntimeError(f"{key} has no completed-epoch recovery point.")
            check_source()
            write_json(OUT / f"infrastructure_recovery_{key}_seed42.json",
                        dict(previous_status=previous, same_configuration=True,
                             reason="external phase-one process ended without final status", time_unix=time.time()))
            run_training([key], [42], "train_phase1_recovery.log", resume=True)
        else:
            run_training([key], [42], "train_phase1_continued.log")


def verify_completed():
    import numpy as np
    import torch
    torch.set_num_threads(4)
    configs = sorted((OUT / "01_screen").glob("*/seed_*/registered_config.json"))
    assert 4 <= len(configs) <= 12
    checks, orders = [], {}
    for path in configs:
        folder, config = path.parent, read_json(path)
        assert read_json(folder / "status.json")["status"] == "complete"
        history = read_rows(folder / "training_metrics.csv")
        assert [int(r["epoch"]) for r in history] == list(range(1, 101))
        assert all(math.isfinite(float(r[k])) for r in history for k in ("train_loss", "val_loss", "seconds"))
        assert int(history[-1]["optimizer_updates"]) == 2200
        selection = read_json(folder / "selection.json")
        assert selection["epoch_threshold_pairs"] == 620 and selection["checkpoint_candidates"] == 20
        pool = []
        for epoch in range(5, 101, 5):
            snapshot = folder / f"epoch_{epoch:03d}"
            values = read_json(snapshot / "threshold_metrics.json")
            assert len(values) == 31
            assert [r["threshold"] for r in values] == read_json(OUT / "00_protocol/lock.json")["thresholds"]
            assert all(r["tp"] + r["fn"] == 372 for r in values)
            pool.extend(dict(epoch=epoch, **r) for r in values)
            with np.load(snapshot / "candidates.npz", allow_pickle=False) as arrays:
                assert len(arrays["sample_id"]) == 22 and len(arrays["offsets"]) == 23
                assert np.isfinite(arrays["coordinates"]).all() and np.isfinite(arrays["scores"]).all()
                assert ((arrays["scores"] >= 0) & (arrays["scores"] <= 1)).all()
        expected = max(pool, key=lambda r: (r["f1"], r["precision"], -r["epoch"], r["threshold"]))
        assert all(selection["primary"][k] == expected[k] for k in ("epoch", "threshold", "f1", "tp", "fp", "fn"))
        weight_hashes = {}
        for filename, role in [("best_model.pt", "primary"), ("native_selected_model.pt", "native_secondary")]:
            weight = folder / filename
            payload = torch.load(weight, map_location="cpu", weights_only=False)
            assert payload["epoch"] == selection[role]["epoch"] and payload["input_size"] == 512
            assert all(torch.isfinite(t).all() for t in payload["model_state_dict"].values())
            weight_hashes[filename] = sha256(weight)
            assert weight_hashes[filename] == sha256(folder / f"epoch_{payload['epoch']:03d}/model.pt")
        order_hash = sha256(folder / "sample_order.csv")
        if config["seed"] in orders:
            assert orders[config["seed"]] == order_hash
        orders[config["seed"]] = order_hash
        checks.append(dict(model=config["key"], seed=config["seed"], epochs=100,
                   primary=selection["primary"], native_secondary=selection["native_secondary"],
                   weight_sha256=weight_hashes, same_seed_order_sha256=order_hash))
    check_source()
    manifest = read_rows(ROOT / "outputs/261024_clean220_random/frozen_split.csv")
    raw_checked = 0
    for row in manifest:
        for sub, suffix, key in [("tif", ".tif", "image_sha256"), ("csv", ".csv", "csv_sha256")]:
            assert sha256(ROOT / "data/raw" / sub / (row["sample_id"]+suffix)) == row[key]
            raw_checked += 1
    lock = read_json(OUT / "00_protocol/lock.json")
    assert sha256(ROOT / "outputs/261024_clean220_random/dna_clean220_random.npz") == lock["archive_sha256"]
    result = dict(status="complete", trials=checks, training_tasks=len(configs), epochs=100*len(configs),
                   raw_hashes_verified=raw_checked, test_arrays_read=False, independent_validation=False,
                   blinded_annotation_review="pending", architecture_novelty="unconfirmed",
                   targeted_tests_preflight=63, selection_optimism="validation used for 620 registered selections per run")
    comparisons = []
    selected_route = read_json(OUT / "decision.json")["selected_route"]
    if selected_route:
        indexed = {(r["model"], r["seed"]): r["primary"]["f1"] for r in checks}
        seeds = [42, 43, 44]
        if all((selected_route, seed) in indexed and ("R0", seed) in indexed for seed in seeds):
            deltas = [indexed[selected_route, seed]-indexed["R0", seed] for seed in seeds]
            comparisons.append(dict(route=selected_route, paired_seed_validation_deltas=deltas,
                  mean_delta=statistics.mean(deltas), sample_sd_delta=statistics.stdev(deltas),
                  positive_seeds=sum(d > 0 for d in deltas), not_a_significance_test=True))
    result["repeat_descriptive_comparisons"] = comparisons
    write_json(OUT / "delivery_checks.json", result)
    lines = ["# 本轮执行完成记录", "", f"已完成{len(configs)}次训练，共{100*len(configs)}个名义epoch；预定预算上限12次。",
             "", "完整分数与机制工作点见阶段汇报.md、validation_comparison.csv和decision.json。所有分数来自开发验证集；当前没有新的内部测试或独立验证。",
             "", "验收通过：每次100轮、20个快照、620个epoch/阈值组合；最佳和原生损失选择权重CPU参数有限；候选坐标/得分有限；同种子样本顺序一致；440个原始文件哈希及输入NPZ未变。",
             "", "人工盲法标注复核及架构新颖性仍未确认。仅有开发分数或常见构件组合不能自动通过核心创新关口。训练和校准的变化须与结构差异分开归因。"]
    (OUT / "本轮完成记录.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-pid", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    count = check_source()
    expected = [OUT / f"01_screen/{key}/seed_42/status.json" for key in ("R0", "H0", "R1", "S1")]
    if args.check_only:
        print(json.dumps(dict(source_valid=True, configured_tasks=count, max_tasks=12,
                             phase1_alive=process_alive(args.phase1_pid), launches_training=False)))
        return
    status("waiting_for_phase1", phase1_pid=args.phase1_pid)
    while process_alive(args.phase1_pid):
        stopped()
        time.sleep(15)
        status("waiting_for_phase1", phase1_pid=args.phase1_pid)
    if not all(p.exists() and read_json(p)["status"] == "complete" for p in expected):
        finish_phase1_after_infrastructure_loss()
    run_command([TRAINER, "report"], "controller_report.log")
    decision = read_json(OUT / "decision.json")
    if decision["r2_stage_enabled"]:
        write_json(OUT / "conditional_context_activation.json", dict(reason=decision["r2_rationale"],
                     first_stage_decision=decision, no_new_hyperparameters=True, time_unix=time.time()))
        run_training(["R2", "S2"], [42], "train_phase2.log")
    run_command([TRAINER, "report"], "controller_report.log")
    decision = read_json(OUT / "decision.json")
    route = decision["selected_route"]
    if route:
        control = {"R1": "S1", "R2": "S2"}[route]
        run_training(["R0", route, control], [43, 44], "train_repeats.log")
    run_command([TRAINER, "report"], "controller_report.log")
    run_command([ROOT / "deepblink_pytorch/scripts/analyze_v3.py"], "analysis.log")
    status("final_validation")
    result = verify_completed()
    status("complete", training_tasks=result["training_tasks"], epochs=result["epochs"],
           independent_validation=False, test_evaluated=False)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("stopped" if isinstance(error, InterruptedError) else "needs_investigation", error=repr(error))
        raise
