"""Six bounded ablations using the byte-unchanged v3 training implementation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo

from complete_detail_v3 import read_json, read_rows, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/next_stage_v3_closing_controls"
PLAN = ROOT / "docs/next_stage_plan_20261025/round2_20261026/closing_protocol_20261027.json"
DOC = PLAN.with_name("closing_training_20261027.md")
PARENTS = (ROOT / "outputs/next_stage_v3", ROOT / "outputs/next_stage_v3_detail_completion")
STUDY = "pooled220_v3_closing_controls"
TARGETS = (("G0", 42), ("Qglobal", 42), ("G0", 43), ("Qglobal", 43), ("G0", 44), ("Qglobal", 44))
NAMES = dict(G0="deeploci_grid_control", Qglobal="deeploci_global_query")
BEGIN, END = "<!-- CLOSING_TRAINING_STATUS_BEGIN -->", "<!-- CLOSING_TRAINING_STATUS_END -->"
STATE, MUTEX, STOP_HEARTBEAT = {}, threading.Lock(), threading.Event()
RECIPE_FIELDS = ("epochs", "batch_size", "lr", "weight_decay", "device", "training_amp", "threads",
                 "channels_last", "train_images", "validation_images", "updates", "loss", "sampler", "test_arrays_loaded")


def timezone():
    return ZoneInfo.from_file(io.BytesIO((OUT / "00_protocol/America_New_York.tzif").read_bytes()),
                              key="America/New_York")


def local_time(stamp=None):
    return datetime.fromtimestamp(time.time() if stamp is None else stamp, timezone()).isoformat(timespec="seconds")


def validate_plan(plan):
    if plan["study_id"] != STUDY or plan["authorized_targets"] != [list(t) for t in TARGETS] or plan["models"] != NAMES:
        raise ValueError("Unregistered model or seed.")
    expected = dict(max_new_training_tasks=6, max_new_epochs=600, epochs_per_task=100,
                    updates_per_task=2200, batch_size=8, optimizer="Adam", learning_rate=.001,
                    weight_decay=0., device="xpu", training_amp="bfloat16", threads=4,
                    train_images=176, validation_images=22, test_arrays_loaded=False,
                    extra_hyperparameter_search=False, early_stopping=False,
                    checkpoint_epochs=list(range(5, 101, 5)), matching_radius_px=3.)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("The six-run recipe or budget differs from the authorized design.")


def check_recipe(current, reference):
    if any(current.get(k) != reference.get(k) for k in RECIPE_FIELDS):
        raise ValueError("Paired reference recipe differs.")


def reference_files():
    files = []
    for parent in PARENTS:
        files.extend(parent / name for name in ("00_protocol/lock.json", "delivery_checks.json", "validation_comparison.csv"))
        for folder in sorted((parent / "01_screen").glob("*/seed_*")):
            files.extend(folder / name for name in ("registered_config.json", "status.json", "selection.json",
                         "sample_order.csv", "best_model.pt", "native_selected_model.pt"))
    return files


def configure():
    import optimize_v3 as base
    if not getattr(base, "_closing_configured", False):
        original_sources = base.source_hashes

        def sources():
            extra = [Path(__file__), PLAN]
            extra += [Path(__file__).with_name(n) for n in ("complete_detail_v3.py", "continue_v3.py", "analyze_v3.py")]
            return original_sources() | {p.relative_to(ROOT).as_posix(): sha256(p) for p in extra}

        base.OUT, base.PROTOCOL, base.NAMES = OUT, STUDY, dict(NAMES)
        base.source_hashes = sources
        base.lock_protocol = lambda: validate_lock(base)
        base._closing_configured = True
    base.torch.set_num_threads(4)
    return base


def validate_lock(base):
    lock = read_json(OUT / "00_protocol/lock.json")
    validate_plan(read_json(PLAN))
    if lock["study_id"] != STUDY or lock["authorized_targets"] != [list(t) for t in TARGETS] or lock["max_new_training_tasks"] != 6:
        raise ValueError("Invalid locked budget.")
    if base.source_hashes() != lock["source_hashes"]:
        raise ValueError("Frozen sources changed; do not edit active training sources.")
    for name, digest in {**lock["reference_artifact_hashes"], **lock["preflight_artifact_hashes"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Protected artifact changed: " + name)
    if sha256(base.ARCHIVE) != lock["archive_sha256"]:
        raise ValueError("Frozen input archive changed.")
    manifest = ROOT / "outputs/261024_clean220_random/frozen_split.csv"
    if sha256(manifest) != lock["split_manifest_sha256"]:
        raise ValueError("Frozen split changed.")
    configs = list((OUT / "01_screen").glob("*/seed_*/registered_config.json"))
    actual = {(p.parent.parent.name, int(p.parent.name.split("_")[-1])) for p in configs}
    if len(configs) > 6 or not actual <= set(TARGETS):
        raise ValueError("Unregistered task or seventh run.")
    return lock


def data_audit(base):
    from deepblink_torch.spotiflow_bridge import ARCHIVE_SHA256, SPLIT_SHA256
    manifest = ROOT / "outputs/261024_clean220_random/frozen_split.csv"
    if sha256(base.ARCHIVE) != ARCHIVE_SHA256 or sha256(manifest) != SPLIT_SHA256:
        raise ValueError("Frozen archive/split mismatch.")
    rows = read_rows(manifest)
    if len(rows) != 220 or any(r["sample_id"] in {"sample_230", "sample_286"} for r in rows):
        raise ValueError("Unexpected sample inventory.")
    for row in rows:
        for sub, suffix, key in (("tif", ".tif", "image_sha256"), ("csv", ".csv", "csv_sha256")):
            if sha256(ROOT / "data/raw" / sub / (row["sample_id"] + suffix)) != row[key]:
                raise ValueError("Raw data changed: " + row["sample_id"])
    return dict(raw_files_verified=440, test_arrays_loaded=False, archive_sha256=ARCHIVE_SHA256,
                split_manifest_sha256=SPLIT_SHA256)


def expected_order(base, ids, seed):
    buffer = io.StringIO(newline="")
    import csv
    writer = csv.DictWriter(buffer, fieldnames=["epoch", "position", "sample_id"])
    writer.writeheader()
    writer.writerows(dict(epoch=e+1, position=i, sample_id=str(ids[index]))
                    for e, order in enumerate(base.training_orders(176, 100, seed)) for i, index in enumerate(order))
    return buffer.getvalue().encode("utf-8")


def frozen_class(path, name, class_name):
    spec = importlib.util.spec_from_file_location("deepblink_torch.networks." + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def preflight(base):
    torch, np = base.torch, base.np
    disk_free_gib = shutil.disk_usage(ROOT).free / 2**30
    if disk_free_gib < 20:
        raise RuntimeError("At least 20 GiB free disk space is required for snapshots.")
    on_ac = None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ac", wintypes.BYTE), ("flag", wintypes.BYTE),
                        ("percent", wintypes.BYTE), ("reserved", wintypes.BYTE),
                        ("life", wintypes.DWORD), ("full_life", wintypes.DWORD)]

        power = PowerStatus()
        known = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power))
        on_ac = bool(known and power.ac == 1)
        if not on_ac:
            raise RuntimeError("Connect AC power before six sequential training runs.")
    previous = read_json(PARENTS[0] / "01_screen/R0/seed_42/environment.json")
    if not torch.xpu.is_available() or str(torch.__version__) != previous["torch"] or torch.xpu.get_device_name(0) != previous["device"]:
        raise ValueError("Same XPU runtime as the paired references is required.")
    original = read_json(PARENTS[0] / "01_screen/R0/seed_42/registered_config.json")["source_hashes"]
    changed = [name for name, digest in original.items() if sha256(ROOT / name) != digest]
    if changed != ["deepblink_pytorch/deepblink_torch/cli/predict.py"]:
        raise ValueError("Unexpected changes in original v3 sources: " + repr(changed))
    corpus = base.DevelopmentCorpus()
    if [len(corpus.splits[s]["x"]) for s in ("train", "valid")] != [176, 22] or [sum(map(len, corpus.splits[s]["y"])) for s in ("train", "valid")] != [4156, 372]:
        raise ValueError("Development split counts changed.")
    order_hashes = {}
    import hashlib
    for seed in (42, 43, 44):
        expected = expected_order(base, corpus.splits["train"]["sample_id"], seed)
        reference = PARENTS[0] / f"01_screen/R0/seed_{seed}/sample_order.csv"
        if expected != reference.read_bytes():
            raise ValueError("Sample order differs from R0.")
        order_hashes[str(seed)] = hashlib.sha256(expected).hexdigest()
    data = corpus.splits["train"]
    selected = [int(np.flatnonzero(data["original_shape"][:, 0] == size)[0]) for size in (256, 512)]
    regression = []
    for key, cls_name, filename in (("R0", "UNetDeepBlink", "unet.py"), ("R1", "UNetDeepLociDetailRoute", "detail_route.py")):
        folder = PARENTS[0] / f"01_screen/{key}/seed_42"
        old_type = frozen_class(folder / "source_snapshot/deepblink_pytorch/deepblink_torch/networks" / filename,
                                "_closing_frozen_" + key, cls_name)
        old = old_type().eval()
        current, _ = base.predict._load_model(folder / "best_model.pt", torch.device("cpu"))
        old.load_state_dict(current.state_dict())
        current.eval()
        with torch.no_grad():
            for i in selected:
                image = torch.from_numpy(data["x"][i:i+1, None])
                h, w = map(int, data["original_shape"][i])
                before = base.predict._forward_with_geometry(old, image, h, w)
                after = base.predict._forward_with_geometry(current, image, h, w)
                diff = float((after - before).abs().max())
                if diff != 0:
                    raise ValueError("Frozen/current R0/R1 prediction regression.")
                regression.append(dict(model=key, sample_id=str(data["sample_id"][i]), max_abs=diff))
        del old, current
    dataset = base.DevelopmentDataset(data)
    indices = [int(i) for size in (256, 512) for i in np.flatnonzero(data["original_shape"][:, 0] == size)[:4]]
    images, targets = zip(*(dataset[i] for i in indices))
    x = torch.stack(images).to("xpu").contiguous(memory_format=torch.channels_last)
    y = torch.stack(targets).to("xpu")
    pilots = []
    for key, branch in (("G0", "grid_projection"), ("Qglobal", "query")):
        base.reset_rng(42)
        model = base.make_model(key, "xpu").to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        torch.xpu.reset_peak_memory_stats()
        torch.xpu.synchronize()
        start = time.perf_counter()
        loss = base.native_epoch(model, [(x, y), (x, y)], base.combined_dice_rmse_loss, "xpu", optimizer)
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - start
        params = list(getattr(model, branch).parameters())
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad) for p in params):
            raise ValueError("Inactive/nonfinite control branch gradient.")
        model.eval()
        with torch.no_grad():
            prediction = base.forward(model, x, y)
        if prediction.shape != (8, 128, 128, 3) or not torch.isfinite(prediction).all():
            raise ValueError("Full batch inference failed.")
        pilots.append(dict(model=key, discarded_optimizer_steps=2, batch_size=8,
                           seconds=elapsed, mean_loss=loss,
                           peak_allocated_mib=torch.xpu.max_memory_allocated() / 2**20,
                           parameters=sum(p.numel() for p in model.parameters())))
        del prediction, optimizer, model
        torch.xpu.empty_cache()
    result = dict(source_changes_from_original=changed, inherited_core_sources_unchanged=True,
                  paired_order_sha256=order_hashes, frozen_class_regression=regression,
                  full_batch_pilots=pilots, pilot_weights_discarded=True, test_arrays_loaded=False,
                  device=previous["device"], torch=str(torch.__version__),
                  disk_free_gib=disk_free_gib, on_ac_power=on_ac,
                  targeted_tests_report=read_json(OUT / "00_protocol/tests.json"))
    return result


def prepare(base):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "00_protocol"
    protocol.mkdir(exist_ok=True)
    if (protocol / "lock.json").exists():
        return validate_lock(base)
    plan = read_json(PLAN)
    validate_plan(plan)
    if list((OUT / "01_screen").glob("*/seed_*/registered_config.json")):
        raise ValueError("Unregistered trials already exist.")
    shutil.copyfile(PARENTS[1] / "00_protocol/America_New_York.tzif", protocol / "America_New_York.tzif")
    if not read_json(protocol / "tests.json").get("passed"):
        raise ValueError("Targeted tests must pass before protocol registration.")
    write_json(protocol / "data_audit.json", data_audit(base))
    write_json(protocol / "preflight.json", preflight(base))
    historical = [read_json(p)["elapsed_seconds"] for p in (PARENTS[1] / "01_screen").glob("*/seed_*/status.json")]
    lock = dict(plan, protocol_id=STUDY, thresholds=base.THRESHOLDS_V3,
                source_hashes=base.source_hashes(), source_base_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                **data_audit(base), historical_seconds_per_task_median=statistics.median(historical),
                reference_artifact_hashes={p.relative_to(ROOT).as_posix(): sha256(p) for p in reference_files()},
                preflight_artifact_hashes={p.relative_to(ROOT).as_posix(): sha256(p) for p in protocol.iterdir()
                                          if p.is_file() and p.name != "lock.json"},
                registered_user_time=local_time(), resumed_tasks_require_audit=True)
    write_json(protocol / "lock.json", lock)
    for relative, digest in lock["source_hashes"].items():
        destination = protocol / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
        if sha256(destination) != digest:
            raise ValueError("Source snapshot byte mismatch.")
    return validate_lock(base)


def task_rows():
    result = []
    for key, seed in TARGETS:
        folder = OUT / f"01_screen/{key}/seed_{seed}"
        status_path, progress_path = folder / "status.json", folder / "progress.json"
        status = read_json(status_path) if status_path.exists() else {}
        progress = read_json(progress_path) if progress_path.exists() else {}
        result.append(dict(model=key, seed=seed, status=status.get("status", "pending"),
                           epoch=100 if status.get("status") == "complete" else progress.get("epoch", 0),
                           best_validation_f1=progress.get("best_validation_f1"),
                           first_epoch=status.get("first_epoch", 1), started=status.get("started"),
                           elapsed_seconds=status.get("elapsed_seconds")))
    return result


def estimate_remaining(tasks, now, default_seconds_per_task):
    observed = [r["elapsed_seconds"] / 100 for r in tasks if r["status"] == "complete" and r["elapsed_seconds"]]
    for row in tasks:
        since_start = row["epoch"] - row.get("first_epoch", 1) + 1
        if row["status"] == "running" and since_start >= 5 and row.get("started"):
            observed.append((now - row["started"]) / since_start)
    seconds_per_epoch = statistics.median(observed) if observed else default_seconds_per_task / 100
    remaining_epochs = sum(100-r["epoch"] for r in tasks)
    return max(0., remaining_epochs * seconds_per_epoch)


def update_doc(state):
    text = DOC.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("Progress document markers changed.")
    lines = [f"状态：**{state['stage']}**；更新时间：{state['updated_user_time']}。", "",
             f"已完成 **{state['completed_tasks']}/6次**，合计 **{state['completed_epochs']}/600轮**。"]
    if state.get("model"):
        lines += [f"当前：{state['model']}，seed {state['seed']}。"]
    if state.get("eta_earliest"):
        lines += [f"预计剩余约{state['remaining_hours']:.1f}小时；预计完成窗口：{state['eta_earliest']} 至 {state['eta_latest']}（America/New_York）。",
                  "预计时间会随实际吞吐更新；电脑需接电，其他负载或休眠会延长时间。"]
    if state.get("error"):
        lines += ["", "错误：`" + state["error"] + "`；查看controller.log和failure.txt。"]
    lines += ["", "| 模型 | seed | 状态 | 已完成轮次 | 当前最佳开发F1 |", "|---|---:|---|---:|---:|"]
    for task in state["tasks"]:
        score = "待评价" if task["best_validation_f1"] is None else f"{task['best_validation_f1']:.4f}"
        lines.append(f"| {task['model']} | {task['seed']} | {task['status']} | {task['epoch']}/100 | {score} |")
    if state["stage"] == "complete":
        lines += ["", "六次训练及最终验收全部完成。结论与配对比较见输出目录`完成汇报.md`，均为开发验证结果。"]
    before, tail = text.split(BEGIN)
    _, after = tail.split(END)
    temporary = DOC.with_name(DOC.name + ".tmp")
    temporary.write_text(before + BEGIN + "\n\n" + "\n".join(lines) + "\n\n" + END + after, encoding="utf-8")
    os.replace(temporary, DOC)


def update_status(stage=None, **fields):
    with MUTEX:
        if stage:
            STATE["stage"] = stage
        STATE.update(fields)
        tasks = task_rows()
        stamp = time.time()
        STATE.update(pid=os.getpid(), updated_unix=stamp, updated_user_time=local_time(stamp), tasks=tasks,
                     completed_tasks=sum(r["status"] == "complete" for r in tasks),
                     completed_epochs=sum(r["epoch"] for r in tasks), max_new_tasks=6)
        STATE.pop("eta_earliest", None)
        STATE.pop("eta_latest", None)
        if STATE["stage"] in {"training", "preflight"}:
            lock = read_json(OUT / "00_protocol/lock.json")
            seconds = estimate_remaining(tasks, stamp, lock["historical_seconds_per_task_median"])
            STATE.update(remaining_hours=seconds/3600, eta_earliest=local_time(stamp + seconds*.85),
                         eta_latest=local_time(stamp + seconds*1.25))
        write_json(OUT / "controller_status.json", STATE)
        update_doc(STATE)
        if stage:
            with (OUT / "execution_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(STATE, ensure_ascii=False, allow_nan=False) + "\n")


def heartbeat():
    while not STOP_HEARTBEAT.wait(15):
        try:
            update_status()
        except Exception:
            traceback.print_exc()
            STATE["heartbeat_error"] = traceback.format_exc()
            return


@contextmanager
def single_worker():
    import msvcrt
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".worker.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def verify_trial(base, key, seed):
    folder = OUT / f"01_screen/{key}/seed_{seed}"
    config = read_json(folder / "registered_config.json")
    reference = PARENTS[0] / f"01_screen/R0/seed_{seed}"
    check_recipe(config, read_json(reference / "registered_config.json"))
    if sha256(folder / "sample_order.csv") != sha256(reference / "sample_order.csv"):
        raise ValueError("Paired R0 sample order differs.")
    if config["source_hashes"] != base.source_hashes() or config["model"] != NAMES[key]:
        raise ValueError("Trial source/model differs from registration.")
    if read_json(folder / "status.json")["status"] != "complete":
        raise ValueError("Trial is incomplete.")
    history = read_rows(folder / "training_metrics.csv")
    if [int(r["epoch"]) for r in history] != list(range(1, 101)) or int(history[-1]["optimizer_updates"]) != 2200:
        raise ValueError("Incomplete epoch/update budget.")
    if not all(math.isfinite(float(r[k])) for r in history for k in ("train_loss", "val_loss", "seconds")):
        raise ValueError("Nonfinite training history.")
    selection = read_json(folder / "selection.json")
    if selection["epoch_threshold_pairs"] != 620 or selection["checkpoint_candidates"] != 20:
        raise ValueError("Selection budget differs.")
    pool = []
    checkpoint_rows = read_rows(folder / "checkpoint_metrics.csv")
    for epoch in range(5, 101, 5):
        snapshot = folder / f"epoch_{epoch:03d}"
        values = read_json(snapshot / "threshold_metrics.json")
        if [r["threshold"] for r in values] != base.THRESHOLDS_V3 or any(r["tp"]+r["fn"] != 372 for r in values):
            raise ValueError("Invalid threshold/count accounting.")
        pool.extend(dict(epoch=epoch, **r) for r in values)
        with base.np.load(snapshot / "candidates.npz", allow_pickle=False) as arrays:
            if len(arrays["sample_id"]) != 22 or len(arrays["offsets"]) != 23 or not base.np.isfinite(arrays["coordinates"]).all() or not base.np.isfinite(arrays["scores"]).all():
                raise ValueError("Invalid validation candidates.")
    primary = max(pool, key=base.selection_key)
    native_epoch = int(min(checkpoint_rows, key=lambda r: (float(r["val_loss"]), int(r["epoch"])))["epoch"])
    secondary = max((r for r in pool if r["epoch"] == native_epoch), key=base.selection_key)
    for role, expected, name in (("primary", primary, "best_model.pt"), ("native_secondary", secondary, "native_selected_model.pt")):
        if any(selection[role][k] != expected[k] for k in ("epoch", "threshold", "f1", "tp", "fp", "fn")):
            raise ValueError("Registered selection rule differs.")
        weights = base.torch.load(folder / name, map_location="cpu", weights_only=False)
        if weights["epoch"] != expected["epoch"] or weights["model_name"] != NAMES[key] or not all(base.torch.isfinite(t).all() for t in weights["model_state_dict"].values()):
            raise ValueError("Selected checkpoint validation failed.")
        if sha256(folder / name) != sha256(folder / f"epoch_{expected['epoch']:03d}/model.pt"):
            raise ValueError("Selected checkpoint bytes differ.")
    return dict(model=key, seed=seed, epochs=100, updates=2200, primary=selection["primary"],
                native_secondary=selection["native_secondary"], sample_order_sha256=sha256(folder / "sample_order.csv"),
                primary_weight_sha256=sha256(folder / "best_model.pt"))


def write_report(base, checks):
    from analyze_v3 import main as analyze
    analyze([*(p / "01_screen" for p in PARENTS), OUT / "01_screen"], output_root=OUT)
    scores, comparison_rows = {}, []
    for parent in (*PARENTS, OUT):
        for path in (parent / "01_screen").glob("*/seed_*/selection.json"):
            config = read_json(path.with_name("registered_config.json"))
            scores[config["key"], config["seed"]] = read_json(path)["primary"]
    summaries = []
    for left, right in (("R1", "G0"), ("R1", "Qglobal"), ("G0", "R0"), ("Qglobal", "R0"), ("R1", "S1")):
        deltas = [scores[left, seed]["f1"] - scores[right, seed]["f1"] for seed in (42, 43, 44)]
        summaries.append(dict(comparison=f"{left}-{right}", paired_f1_deltas=deltas,
                              mean_delta=statistics.mean(deltas), sample_sd=statistics.stdev(deltas),
                              positive_seeds=sum(d > 0 for d in deltas), significance_test=False))
    for (key, seed), score in sorted(scores.items()):
        comparison_rows.append(dict(model=key, seed=seed, **score))
    base.write_csv(OUT / "validation_comparison.csv", comparison_rows)
    write_json(OUT / "paired_summary.json", summaries)
    lines = ["# 六次消融训练完成报告", "", f"完成时间：{local_time()}（America/New_York）。", "",
             "G0/Q-global各三种子，600轮、13,200次更新。所有比较均为开发验证；每次20×31选模，未读取测试数组。", "",
             "| 模型 | seed42 F1 | seed43 F1 | seed44 F1 | 均值±样本SD | 弱点Recall均值 |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in ("R0", "R1", "S1", "G0", "Qglobal"):
        values = [scores[key, s]["f1"] for s in (42, 43, 44)]
        weak = statistics.mean(scores[key, s]["weak_recall"] for s in (42, 43, 44))
        lines.append(f"| {key} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {statistics.mean(values):.4f}±{statistics.stdev(values):.4f} | {weak:.2%} |")
    for row in summaries:
        lines += ["", f"{row['comparison']}：平均ΔF1 {row['mean_delta']:+.4f}，{row['positive_seeds']}/3种子为正，配对差样本SD {row['sample_sd']:.4f}。"]
    lines += ["", "R1-G0检验新增头容量解释，R1-Qglobal检验逐格查询的必要性。控制持平或更优时，应删除相应机制主张；小均值差和三个种子不构成独立显著性或泛化证明。", "",
              "固定R0误检预算、尺寸分层、定位误差及逐点得失见02_analysis。参数/显存预检见00_protocol/preflight.json；预检耗时含热身，不作公平速度结论。", "",
              "独立生物学验证和盲法标注复核尚缺；Spotiflow正式微调不属于这六次预算。预算用满后停止，不自动追加调参或组合。"]
    (OUT / "完成汇报.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "delivery_checks.json", dict(status="complete", study_id=STUDY, training_tasks=6,
               epochs=600, updates=13200, trials=checks, raw_data_audit=data_audit(base),
               parent_artifacts_unchanged=True, test_arrays_loaded=False,
               completion_user_time=local_time(), paired_comparisons=summaries))


def run(base, resume=False):
    with single_worker():
        validate_lock(base)
        update_status("preflight", model=None, seed=None, started_user_time=local_time())
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            data_audit(base)
            corpus = base.DevelopmentCorpus()
            from training_wake_guard import TrainingWakeGuard
            checks = []
            with TrainingWakeGuard():
                for key, seed in TARGETS:
                    if (OUT / "STOP_REQUESTED").exists():
                        raise InterruptedError("STOP_REQUESTED at trial boundary.")
                    if STATE.get("heartbeat_error"):
                        raise RuntimeError("Progress writer failed: " + STATE["heartbeat_error"])
                    validate_lock(base)
                    folder = OUT / f"01_screen/{key}/seed_{seed}"
                    update_status("training", model=key, seed=seed)
                    base.run_trial(key, seed, corpus, "xpu", resume=resume and (folder / "status.json").exists())
                    checks.append(verify_trial(base, key, seed))
                    write_json(OUT / "completed_trial_checks.json", checks)
                    update_status("training", model=key, seed=seed)
            del corpus
            update_status("final_validation", model=None, seed=None)
            validate_lock(base)
            write_report(base, checks)
            validate_lock(base)
            update_status("complete", model=None, seed=None, test_evaluated=False, finished_user_time=local_time())
        except BaseException as error:
            (OUT / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            update_status("stopped" if isinstance(error, InterruptedError) else "needs_investigation", error=repr(error))
            raise
        finally:
            STOP_HEARTBEAT.set()
            thread.join(timeout=20)


def launch(base):
    validate_lock(base)
    status = OUT / "controller_status.json"
    if status.exists():
        from continue_v3 import process_alive
        previous = read_json(status)
        if process_alive(previous["pid"]) or previous["stage"] == "complete":
            raise RuntimeError("Worker already active or study complete; do not launch again.")
        raise RuntimeError("Previous worker stopped; investigate before explicit --run --resume.")
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    with (OUT / "controller.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--run"], cwd=ROOT,
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=flags, close_fds=True)
    write_json(OUT / "launch.json", dict(launcher_child_pid=child.pid, launched_user_time=local_time(),
               interpreter=sys.executable, mode="detached_hidden", source_sha256=sha256(__file__)))
    print(json.dumps(dict(launched=True, pid=child.pid, progress_file=str(DOC)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Only after auditing an interrupted identical trial")
    args = parser.parse_args()
    base = configure()
    if args.prepare_only:
        lock = prepare(base)
        print(json.dumps(dict(registered=True, study_id=lock["study_id"], tasks=6, training_started=False)))
    elif args.check_only:
        validate_lock(base)
        print("Source, data, budget, preflight and reference locks verified.")
    elif args.launch:
        launch(base)
    else:
        run(base, args.resume)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    main()
