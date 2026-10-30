"""Four explicitly authorized seed completions using the unchanged v3 trainer.

The original twelve-run study is read-only. This extension changes only its
output/authorization lock; models, sampling, objectives and selection stay fixed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "outputs/next_stage_v3"
OUT = ROOT / "outputs/next_stage_v3_detail_completion"
DOC = ROOT / "docs/next_stage_plan_20261025/round2_20261026/detail_completion.md"
STUDY = "pooled220_methods_v3_detail_completion"
TARGETS = (("R1", 43), ("S1", 43), ("R1", 44), ("S1", 44))
BEGIN, END = "<!-- DETAIL_COMPLETION_STATUS_BEGIN -->", "<!-- DETAIL_COMPLETION_STATUS_END -->"
SOURCE_NAMES = ("optimize_v3.py", "continue_v3.py", "analyze_v3.py", "complete_detail_v3.py", "training_wake_guard.py")
STATE, STATE_LOCK, HEARTBEAT_STOP = {}, threading.Lock(), threading.Event()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def user_time():
    # The isolated training environment has no tzdata package. Use a separately
    # hashed TZif export rather than changing the shared environment or UTC date.
    with (OUT / "00_protocol/America_New_York.tzif").open("rb") as handle:
        zone = ZoneInfo.from_file(handle, key="America/New_York")
    return datetime.now(zone).isoformat(timespec="seconds")


def update_status(stage=None, **extra):
    with STATE_LOCK:
        if stage is not None:
            STATE["stage"] = stage
        STATE.update(extra, pid=os.getpid(), updated_unix=time.time())
        key, seed = STATE.get("model"), STATE.get("seed")
        progress = OUT / f"01_screen/{key}/seed_{seed}/progress.json"
        STATE.pop("progress", None)
        if key and progress.exists():
            STATE["progress"] = read_json(progress)
        STATE["completed_tasks"] = sum(
            path.exists() and read_json(path)["status"] == "complete"
            for path in (OUT / f"01_screen/{k}/seed_{s}/status.json" for k, s in TARGETS))
        write_json(OUT / "controller_status.json", STATE)
        if stage is not None:
            with (OUT / "execution_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(STATE, user_time=user_time()), ensure_ascii=False)+"\n")


def heartbeat():
    while not HEARTBEAT_STOP.wait(15):
        update_status()


def update_log(text):
    content = DOC.read_text(encoding="utf-8")
    if content.count(BEGIN) != 1 or content.count(END) != 1:
        raise ValueError("The extension log status markers changed.")
    before, tail = content.split(BEGIN)
    _, after = tail.split(END)
    temporary = DOC.with_name(DOC.name + ".tmp")
    temporary.write_text(before+BEGIN+"\n\n"+text+"\n\n"+END+after, encoding="utf-8")
    os.replace(temporary, DOC)


def parent_files():
    paths = [PARENT / name for name in ("00_protocol/lock.json", "delivery_checks.json", "decision.json", "validation_comparison.csv")]
    for folder in sorted((PARENT / "01_screen").glob("*/seed_*")):
        paths += [folder / name for name in ("registered_config.json", "status.json", "selection.json", "sample_order.csv", "best_model.pt", "native_selected_model.pt")]
    return paths


def audit_data(base, lock):
    manifest = ROOT / "outputs/261024_clean220_random/frozen_split.csv"
    if sha256(base.ARCHIVE) != lock["archive_sha256"] or sha256(manifest) != lock["split_manifest_sha256"]:
        raise ValueError("The input archive or split manifest changed.")
    entries = read_rows(manifest)
    if len(entries) != 220:
        raise ValueError("Unexpected paired-data inventory.")
    checked = 0
    for row in entries:
        for sub, suffix, key in (("tif", ".tif", "image_sha256"), ("csv", ".csv", "csv_sha256")):
            if sha256(ROOT / "data/raw" / sub / (row["sample_id"]+suffix)) != row[key]:
                raise ValueError("Raw data changed: " + row["sample_id"])
            checked += 1
    return dict(raw_file_hashes_verified=checked, archive_unchanged=True, manifest_unchanged=True,
                test_arrays_read=False, checked_user_time=user_time())


def prepare(base):
    destination = OUT / "00_protocol/lock.json"
    if destination.exists():
        return validate_lock(base)
    summary = read_json(PARENT / "delivery_checks.json")
    if summary["status"] != "complete" or summary["training_tasks"] != 12:
        raise ValueError("The parent twelve-run study must already be complete.")
    configs = list((PARENT / "01_screen").glob("*/seed_*/registered_config.json"))
    if len(configs) != 12:
        raise ValueError("Unexpected parent trial inventory.")
    expected = read_json(configs[0])["source_hashes"]
    if any(read_json(path)["source_hashes"] != expected for path in configs) or base.source_hashes() != expected:
        raise ValueError("The original training source changed; do not reuse R0.")
    if any((OUT / f"01_screen/{key}/seed_{seed}/registered_config.json").exists() for key, seed in TARGETS):
        raise ValueError("Unregistered extension trial exists.")
    parent_lock = read_json(PARENT / "00_protocol/lock.json")
    audit = audit_data(base, parent_lock)
    write_json(OUT / "00_protocol/data_audit.json", audit)
    previous_environment = read_json(PARENT / "01_screen/R0/seed_42/environment.json")
    current_device = base.torch.xpu.get_device_name(0) if base.torch.xpu.is_available() else None
    if str(base.torch.__version__) != previous_environment["torch"] or current_device != previous_environment["device"]:
        raise ValueError("PyTorch/device differs from the paired reference environment.")
    sources = {f"deepblink_pytorch/scripts/{name}": sha256(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
    lock = dict(parent_lock, study_id=STUDY, parent_protocol_id=base.PROTOCOL,
                parent_lock_sha256=sha256(PARENT / "00_protocol/lock.json"),
                max_new_training_tasks=4, max_new_epochs=400, original_completed_tasks=12,
                authorized_targets=[list(t) for t in TARGETS],
                user_execution_authorization="User requested completion of R1/S1 seeds43/44 and log update after all training finishes.",
                source_hashes=expected, auxiliary_source_hashes=sources,
                data_audit_sha256=sha256(OUT / "00_protocol/data_audit.json"),
                parent_artifact_hashes={p.relative_to(ROOT).as_posix(): sha256(p) for p in parent_files()},
                timezone_sha256=sha256(OUT / "00_protocol/America_New_York.tzif"),
                analysis="Descriptive R1-R0 and R1-S1 three-seed paired deltas including existing seed42; no new tuning or test inference.",
                selected_route=None, test_evaluated=False)
    write_json(destination, lock)
    for name, digest in sources.items():
        target = OUT / "00_protocol/source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        if sha256(target) != digest:
            raise ValueError("Auxiliary source snapshot failed.")
    write_json(OUT / "decision.json", dict(study_id=STUDY, selected_route=None, no_new_route_selection=True))
    return validate_lock(base)


def validate_lock(base):
    lock = read_json(OUT / "00_protocol/lock.json")
    if lock["study_id"] != STUDY or lock["authorized_targets"] != [list(t) for t in TARGETS] or lock["max_new_training_tasks"] != 4:
        raise ValueError("Unexpected extension authorization/budget.")
    if base.source_hashes() != lock["source_hashes"]:
        raise ValueError("Original training source changed.")
    for name, digest in {**lock["auxiliary_source_hashes"], **lock["parent_artifact_hashes"]}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Protected file changed: " + name)
    if sha256(OUT / "00_protocol/America_New_York.tzif") != lock["timezone_sha256"]:
        raise ValueError("Time-zone source changed.")
    if sha256(OUT / "00_protocol/data_audit.json") != lock["data_audit_sha256"]:
        raise ValueError("Initial data audit changed.")
    if sha256(base.ARCHIVE) != lock["archive_sha256"]:
        raise ValueError("Training input archive changed.")
    actual = {(p.parent.parent.name, int(p.parent.name.split("_")[-1]))
              for p in (OUT / "01_screen").glob("*/seed_*/registered_config.json")}
    if not actual <= set(TARGETS) or len(actual) > 4:
        raise ValueError("Unregistered model/seed or fifth task.")
    return lock


def verify_pairing(base, key, seed):
    trial = OUT / f"01_screen/{key}/seed_{seed}"
    reference = PARENT / f"01_screen/R0/seed_{seed}"
    current, previous = read_json(trial / "registered_config.json"), read_json(reference / "registered_config.json")
    fields = ("source_hashes", "epochs", "batch_size", "lr", "weight_decay", "device", "training_amp",
              "threads", "channels_last", "train_images", "validation_images", "updates", "loss", "sampler", "test_arrays_loaded")
    if any(current[k] != previous[k] for k in fields):
        raise ValueError("The paired R0 recipe/source does not match.")
    if sha256(trial / "sample_order.csv") != sha256(reference / "sample_order.csv"):
        raise ValueError("The paired R0 image order does not match.")
    original = read_json(PARENT / f"01_screen/{key}/seed_42/registered_config.json")
    if current["model"] != original["model"]:
        raise ValueError("Architecture differs from the registered seed42 trial.")


def write_report(base, checks):
    from analyze_v3 import main as analyze
    analyze([PARENT / "01_screen", OUT / "01_screen"], output_root=OUT)
    indexed, rows = {}, []
    for folder in (PARENT / "01_screen", OUT / "01_screen"):
        for path in sorted(folder.glob("*/seed_*/selection.json")):
            config = read_json(path.with_name("registered_config.json"))
            result = read_json(path)
            row = dict(model=config["key"], seed=config["seed"], **result["primary"],
                       native_secondary_f1=result["native_secondary"]["f1"], source_folder=str(path.parent.relative_to(ROOT)))
            indexed[row["model"], row["seed"]] = row
            rows.append(row)
    base.write_csv(OUT / "validation_comparison.csv", rows)
    comparisons = []
    for comparator in ("R0", "S1"):
        deltas = [indexed["R1", seed]["f1"]-indexed[comparator, seed]["f1"] for seed in (42, 43, 44)]
        comparisons.append(dict(comparison="R1-"+comparator, seeds=[42, 43, 44], paired_f1_deltas=deltas,
                                mean_delta=statistics.mean(deltas), sample_sd=statistics.stdev(deltas),
                                positive_seeds=sum(v > 0 for v in deltas), significance_test=False))
    write_json(OUT / "paired_summary.json", comparisons)
    lines = ["# R1/S1种子补齐完成报告", "", f"完成时间：{user_time()}（America/New_York）。", "",
             "新增4次训练、400轮；与原12次结果合计16次。原训练器、模型、样本顺序及620次选模规则均不变，原R0直接复用。全部为开发验证成绩，未进行测试推理。", "",
             "| 模型 | seed42 F1 | seed43 F1 | seed44 F1 | 均值±样本SD | 平均弱点Recall |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in ("R0", "R1", "S1", "R2", "S2"):
        scores = [indexed[key, seed]["f1"] for seed in (42, 43, 44)]
        weak = statistics.mean(indexed[key, seed]["weak_recall"] for seed in (42, 43, 44))
        lines.append(f"| {key} | {scores[0]:.4f} | {scores[1]:.4f} | {scores[2]:.4f} | {statistics.mean(scores):.4f}±{statistics.stdev(scores):.4f} | {weak:.2%} |")
    for item in comparisons:
        lines += ["", f"{item['comparison']}：三个配对ΔF1依次为{', '.join(f'{v:+.4f}' for v in item['paired_f1_deltas'])}，均值{item['mean_delta']:+.4f}，样本SD {item['sample_sd']:.4f}；{item['positive_seeds']}/3个种子为正。这是开发稳定性描述，不是显著性检验。"]
    if all(c["mean_delta"] > 0 and c["positive_seeds"] >= 2 for c in comparisons):
        conclusion = "R1在本次开发复验中相对R0和S1均呈正向平均差，但仍须核对弱点约束、固定FP预算和独立验证；不能据此认定架构创新或独立泛化收益。"
    else:
        conclusion = "当前结果尚未同时支持R1稳定优于原主干与简单投影；不把单种子或仅相对较弱对照的收益写成核心结构贡献。"
    lines += ["", conclusion, "", "逐点新增/丢失、相同FP预算、尺寸分层及历史旧权重参照见02_analysis/结果解释.md。原seed42为已观察筛查结果，追加种子不是新的盲法验证。人工标注复核和生物学独立验证仍待完成。", "",
              "全部4次验收通过后才更新本完成记录及文档日志；原12次关键结果哈希保持一致。本次预算用满即停止，不自动追加训练。"]
    report = "\n".join(lines)+"\n"
    (OUT / "完成汇报.md").write_text(report, encoding="utf-8")
    (OUT / "本轮完成记录.md").write_text(report, encoding="utf-8")
    checks.update(study_id=STUDY, max_new_training_tasks=4, original_trials_preserved=12,
                  combined_available_trials=16, original_source_files_unchanged=len(base.source_hashes()),
                  paired_r0_order_and_recipe_verified=True, completion_user_time=user_time(),
                  repeat_descriptive_comparisons=comparisons, targeted_tests_reused_with_unchanged_sources=True)
    write_json(OUT / "delivery_checks.json", checks)
    update_log(f"状态：**全部完成并通过验收**；完成时间{user_time()}（America/New_York）。新增4次/400轮，原12次结果保持不变。\n\n"
               +"；".join(f"{r['comparison']}平均ΔF1 {r['mean_delta']:+.4f}，{r['positive_seeds']}/3种子为正" for r in comparisons)
               +"。\n\n"+conclusion+"\n\n结果见[完成汇报](../../../outputs/next_stage_v3_detail_completion/完成汇报.md)及[最终验收](../../../outputs/next_stage_v3_detail_completion/delivery_checks.json)。")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Only after auditing an interrupted identical configuration.")
    args = parser.parse_args()
    import optimize_v3 as base
    base.torch.set_num_threads(4)
    if args.prepare_only:
        lock = prepare(base)
        print(json.dumps(dict(registered=True, study_id=lock["study_id"], targets=TARGETS, launches_training=False)))
        return
    lock = validate_lock(base)
    if args.check_only:
        print(json.dumps(dict(source_and_parent_valid=True, targets=TARGETS, launches_training=False)))
        return
    if not base.torch.xpu.is_available():
        raise RuntimeError("The same XPU environment is required.")
    with single_worker():
        thread = threading.Thread(target=heartbeat, daemon=True)
        update_status("preflight", study_id=STUDY, max_new_tasks=4)
        thread.start()
        try:
            # Redirect only authorization/output globals. run_trial, its models,
            # data access, RNG reset, loss, optimizer and selection are unchanged.
            base.OUT = OUT
            base.lock_protocol = lambda: validate_lock(base)
            audit_data(base, lock)
            corpus = base.DevelopmentCorpus()
            if sum(map(len, corpus.splits["train"]["y"])) != 4156 or sum(map(len, corpus.splits["valid"]["y"])) != 372:
                raise ValueError("Frozen point counts changed.")
            update_log(f"状态：**运行中**；启动时间{user_time()}（America/New_York）。仅补齐R1/S1的seed43/44，共4次训练。\n\n"
                       "最终三种子结论将在全部4次完成并验收后自动写入本日志。当前进度见[controller_status.json](../../../outputs/next_stage_v3_detail_completion/controller_status.json)。")
            from training_wake_guard import TrainingWakeGuard
            with TrainingWakeGuard():
                for key, seed in TARGETS:
                    if (OUT / "STOP_REQUESTED").exists():
                        raise InterruptedError("Requested stop at a trial boundary.")
                    validate_lock(base)
                    update_status("training", model=key, seed=seed)
                    base.run_trial(key, seed, corpus, device="xpu", resume=args.resume and (OUT / f"01_screen/{key}/seed_{seed}/status.json").exists())
                    verify_pairing(base, key, seed)
            del corpus
            update_status("final_validation", model=None, seed=None)
            import continue_v3 as verifier
            verifier.OUT = OUT
            checks = verifier.verify_completed()
            validate_lock(base)
            write_report(base, checks)
            update_status("complete", training_tasks=4, epochs=400, test_evaluated=False,
                          finished_user_time=user_time(), model=None, seed=None)
        except BaseException as error:
            update_status("stopped" if isinstance(error, InterruptedError) else "needs_investigation", error=repr(error))
            update_log(f"状态：**需检查，尚未完成**；{user_time()}。错误：`{error!r}`。保留原结果及完整epoch恢复文件，不自动更改参数或重试算法错误。")
            raise
        finally:
            HEARTBEAT_STOP.set()
            thread.join(timeout=20)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    main()
