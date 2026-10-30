"""Frozen data access and append-only overnight status/report helpers."""
from datetime import datetime
import csv
import io
import json
import os
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
from deepblink_torch.experiment_log import sha256, write_json
from deepblink_torch.spotiflow_bridge import ARCHIVE_SHA256, SPLIT_SHA256, checked_coordinates

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/"outputs/next_stage_spotiflow_20261030"
DOC = ROOT/"docs/next_stage_plan_20261025/overnight_progress_20261030.md"
REPORT = ROOT/"docs/next_stage_plan_20261025/work_report_20261030.md"
ARCHIVE = ROOT/"outputs/261024_clean220_random/dna_clean220_random.npz"
SPLIT = ROOT/"outputs/261024_clean220_random/frozen_split.csv"
WEIGHTS = ROOT/".venv/spotiflow_models/general"
STATE, MUTEX = {}, threading.Lock()


def now(stamp=None):
    data=(ROOT/"outputs/next_stage_v3_closing_controls/00_protocol/America_New_York.tzif").read_bytes()
    zone=ZoneInfo.from_file(io.BytesIO(data),key="America/New_York")
    return datetime.fromtimestamp(time.time() if stamp is None else stamp,zone).isoformat(timespec="seconds")


def dump(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,data)


def rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    return value


def status(stage=None,**fields):
    with MUTEX:
        if stage is not None:STATE["stage"]=stage
        STATE.update(fields,pid=os.getpid(),updated_user_time=now(),updated_unix=time.time())
        tasks=[]
        for seed in (42,43,44):
            folder=OUT/f"02_finetune/seed_{seed}"
            record=json.loads((folder/"status.json").read_text(encoding="utf-8")) if (folder/"status.json").exists() else {"status":"not_started"}
            progress=json.loads((folder/"progress.json").read_text(encoding="utf-8")) if (folder/"progress.json").exists() else {}
            tasks.append(dict(seed=seed,status=record["status"],epoch=progress.get("epoch",0),max_epochs=100))
        STATE["tasks"]=tasks
        STATE["completed_training_tasks"]=sum(r["status"]=="complete" for r in tasks)
        dump(OUT/"controller_status.json",clean(STATE))
        lines=["# Spotiflow过夜任务进度", "",f"更新时间：{STATE['updated_user_time']}（America/New_York）。",
               f"阶段：**{STATE.get('stage','initializing')}**；真实工作进程PID：{os.getpid()}。", "",
               "| seed | 状态 | 已完成轮次 |", "|---:|---|---:|"]
        lines += [f"| {r['seed']} | {r['status']} | {r['epoch']}/100 |" for r in tasks]
        for key in ("activity","device_decision","estimated_three_seed_hours","error"):
            if key in STATE:lines += ["",f"{key}: {STATE[key]}"]
        lines += ["", "全部数值为开发集工程/比较证据，不读取测试数组。",
                  "日志：`outputs/next_stage_spotiflow_20261030/controller.log`；工作汇报：同目录`morning_report.md`及本目录`work_report_20261030.md`。",
                  "失败记录和检查点均保留；暂停/跳过正式训练不代表该方法低准确率。"]
        temp=DOC.with_suffix(".md.tmp")
        temp.write_text("\n".join(lines)+"\n",encoding="utf-8")
        os.replace(temp,DOC)
        if stage is not None:
            with (OUT/"execution_events.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(clean(STATE),ensure_ascii=False,allow_nan=False)+"\n")
        print(json.dumps(dict(stage=STATE.get("stage"),**fields),ensure_ascii=False),flush=True)


def load_development():
    if sha256(ARCHIVE)!=ARCHIVE_SHA256 or sha256(SPLIT)!=SPLIT_SHA256:
        raise ValueError("Frozen data or split changed.")
    result={}
    with np.load(ARCHIVE,allow_pickle=True) as archive:
        for subset in ("train","valid"):
            images=archive[f"x_{subset}"]
            truth=archive[f"y_{subset}"]
            ids=archive[f"sample_id_{subset}"]
            shapes=archive[f"original_shape_{subset}"]
            raw=[]
            for image,coords,shape in zip(images,truth,shapes):
                h,w=map(int,shape)
                checked_coordinates(coords,(h,w))
                raw.append(np.asarray(image[:h,:w],dtype=np.float32).copy())
            result[subset]=dict(images=raw,y=truth,sample_id=np.asarray(ids,dtype=str),original_shape=shapes)
    if len(result["train"]["images"])!=176 or len(result["valid"]["images"])!=22:
        raise ValueError("Unexpected development split.")
    return result


def source_hashes():
    scripts=Path(__file__).parent
    paths=[p for p in scripts.rglob("*") if p.is_file() and p.suffix in (".py",".sh")]
    paths += [scripts.parent/"run_spotiflow_overnight.py",scripts.parent/"cloud_spotiflow_train.py",
              ROOT/"docs/next_stage_plan_20261025/autodl_runbook_20261030.md"]
    paths += list((ROOT/"deepblink_pytorch/deepblink_torch").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in paths}


def protected_references():
    paths=[]
    for study in ("next_stage_v3","next_stage_v3_detail_completion","next_stage_v3_closing_controls"):
        root=ROOT/"outputs"/study
        paths += [root/name for name in ("00_protocol/lock.json","delivery_checks.json","validation_comparison.csv")]
        for p in (root/"01_screen").glob("*/seed_*/best_model.pt"):
            paths += [p,p.with_name("selection.json")]
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in paths}


def verify_lock():
    lock=json.loads((OUT/"00_protocol/lock.json").read_text(encoding="utf-8"))
    if lock["source_hashes"]!=source_hashes():raise ValueError("Overnight sources changed after locking.")
    for name,digest in lock["protected_references"].items():
        if sha256(ROOT/name)!=digest:raise ValueError("Historical artifact changed: "+name)
    for name,digest in lock["pretrained_files"].items():
        if sha256(WEIGHTS/name)!=digest:raise ValueError("General pretrained weights changed.")
    if lock["max_training_tasks"]!=3 or lock["seeds"]!=[42,43,44]:raise ValueError("Invalid training budget.")
    return lock
