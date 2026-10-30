"""Bounded engineering/pretrained/audit queue with a measured CPU training gate."""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")

from spotiflow_overnight.common import (ROOT,OUT,WEIGHTS,ARCHIVE,SPLIT,source_hashes,
    protected_references,verify_lock,load_development,status,dump,now)
from deepblink_torch.experiment_log import sha256


@contextmanager
def single_worker():
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/".worker.lock").open("a+b") as stream:
        if stream.tell()==0:stream.write(b"0");stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def prepare(cpu_hours,formal_backend="cuda"):
    if (OUT/"00_protocol/lock.json").exists():return verify_lock()
    from spotiflow_overnight.test_engineering import run_checks
    from spotiflow_overnight.engine import default_training
    status("engineering_validation",activity="Final synthetic recovery and current-epoch snapshot checks")
    checks_path=OUT/"00_engineering/recovery_check_v3/checks.json"
    checks=json.loads(checks_path.read_text()) if checks_path.exists() else run_checks(checks_path.parent)
    if not checks["passed"]:raise ValueError("Engineering tests failed.")
    load_development()
    env={name:version(name) for name in ("torch","torchvision","numpy","scipy","tifffile","spotiflow","lightning","matplotlib")}
    native=ROOT/".venv/spotiflow_cpu/Lib/site-packages/spotiflow"
    config=vars(default_training())
    lock=dict(study_id="spotiflow_clean220_overnight_20261030",max_training_tasks=3,seeds=[42,43,44],
        epoch_budget_per_seed=100,updates_per_seed=4400,checkpoint_epochs=list(range(5,101,5)),
        thresholds="identical 31 THRESHOLDS_V3",primary_selection="micro_F1_then_precision_then_earlier_epoch_then_higher_threshold",
        secondary_selection="minimum_native_loss_among_same_20_snapshots_then_same_31_thresholds",
        data_archive_sha256=sha256(ARCHIVE),split_sha256=sha256(SPLIT),
        test_arrays_loaded=False,training_device="cpu",formal_training_backend=formal_backend,cpu_gate_hours=cpu_hours,margin_factor=1.25,
        training_config=config,native_optimizer_weight_decay=.01,augmentation=True,
        engineering_tests=checks,environment=env,source_hashes=source_hashes(),
        native_sources={p.relative_to(native).as_posix():sha256(p) for p in native.rglob("*.py")},
        pretrained_files={p.name:sha256(p) for p in WEIGHTS.iterdir() if p.is_file()},
        protected_references=protected_references(),registered_user_time=now(),
        pretraining_data_overlap="unresolved",human_annotation_review="pending",
        authorization="2026-10-30 user authorized engineering, repairs and planned conditional overnight tasks")
    dump(OUT/"00_protocol/lock.json",lock)
    for name in lock["source_hashes"]:
        target=OUT/"00_protocol/source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    dump(OUT/"00_protocol/preflight_checks.json",dict(passed=True,synthetic_checks=checks,
        data_hashes_verified=True,protected_reference_files=len(lock["protected_references"]),
        disk_free_gib=shutil.disk_usage(ROOT).free/2**30))
    return verify_lock()


def run(resume=False):
    from spotiflow_overnight.evaluation import pretrained,finalize_finetune
    from spotiflow_overnight.engine import train_native,evaluate_snapshot,default_training
    from spotiflow_overnight.pilot import measure
    from spotiflow_overnight.audit import review,efficiency
    from spotiflow_overnight.package import build
    from spotiflow_overnight.report import write_report
    from spotiflow.model import Spotiflow
    import torch
    torch.set_num_threads(4)
    with single_worker():
        lock=verify_lock()
        stop=threading.Event()
        heartbeat_errors=[]
        def beat():
            while not stop.wait(15):
                try:status()
                except Exception as error:heartbeat_errors.append(repr(error));traceback.print_exc();return
        worker=threading.Thread(target=beat,daemon=True)
        status("starting",activity="Frozen engineering/data/source checks passed",started_user_time=now())
        worker.start()
        try:
            from training_wake_guard import TrainingWakeGuard
            with TrainingWakeGuard():
                data=load_development()
                pretrained(data["valid"])
                pilot=measure(data)
                gate=pilot["estimated_three_seed_hours_with_25pct_margin"]<=lock["cpu_gate_hours"] and lock["formal_training_backend"]=="cpu_auto"
                status("device_decision",device_decision="cpu_three_seed_queue" if gate else "cpu_over_budget_prepare_cuda",
                    estimated_three_seed_hours=pilot["estimated_three_seed_hours"],
                    activity="Three complete seed runs required; runtime gate does not inspect validation score")
                dump(OUT/"device_decision.json",dict(cpu_training_gate_passed=gate,
                    estimated_hours=pilot["estimated_three_seed_hours"],with_margin_hours=pilot["estimated_three_seed_hours_with_25pct_margin"],
                    available_hours=lock["cpu_gate_hours"],cuda_rented=False,decision_time=now()))
                if gate:
                    deadline=time.time()+lock["cpu_gate_hours"]*3600
                    for seed in lock["seeds"]:
                        if (OUT/"STOP_REQUESTED").exists():raise InterruptedError("Stop requested at seed boundary.")
                        if heartbeat_errors:raise RuntimeError(heartbeat_errors[-1])
                        verify_lock()
                        folder=OUT/f"02_finetune/seed_{seed}"
                        path=folder/"status.json"
                        if path.exists() and json.loads(path.read_text())["status"]=="complete":continue
                        if time.time()+pilot["estimated_seconds_per_seed"]*1.25>deadline:
                            status("time_budget_reached",activity="Next planned seed retained for next window; no score-based pruning")
                            break
                        checkpoint=folder/"recovery/last.ckpt"
                        if path.exists() and not resume:raise RuntimeError("Existing partial seed requires audited --resume.")
                        folder.mkdir(parents=True,exist_ok=True)
                        dump(path,dict(status="running",seed=seed,started=now(),source_hashes=lock["source_hashes"]))
                        status("finetuning",activity=f"Spotiflow seed {seed}, 100 native epochs, 20 full-image evaluations")
                        model=Spotiflow.from_folder(WEIGHTS,which="best",inference_mode=False)
                        train_native(model,data["train"]["images"],list(data["train"]["y"]),
                            data["valid"]["images"],list(data["valid"]["y"]),folder,seed=seed,
                            config=default_training(),evaluate=evaluate_snapshot(data["valid"]),
                            event=lambda event,**fields:status("finetuning",activity=event,**fields),
                            resume=checkpoint if resume and checkpoint.exists() else None)
                        finalize_finetune(folder,data["valid"])
                        dump(path,dict(status="complete",seed=seed,completed=now(),epochs=100,updates=4400))
                        del model
                review(data)
                efficiency(data)
                build(data)
                verify_lock()
                write_report("engineering_complete")
                status("engineering_complete",activity="Z0/Z1, recovery, audit, CPU timings and local CUDA bundle complete; formal F status listed separately")
                dump(OUT/"engineering_delivery_checks.json",dict(status="complete",historical_artifacts_unchanged=True,
                    sources_unchanged=True,test_arrays_loaded=False,human_review_completed=False,
                    formal_training_gate_passed=gate,completed_user_time=now()))
        except BaseException as error:
            (OUT/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
            status("needs_investigation",error=repr(error))
            write_report("needs_investigation")
            raise
        finally:
            stop.set();worker.join(timeout=20)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only",action="store_true")
    modes.add_argument("--check-only",action="store_true")
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--launch",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--cpu-budget-hours",type=float,default=5.5)
    parser.add_argument("--formal-backend",choices=["cuda","cpu_auto"],default="cuda")
    args=parser.parse_args()
    if args.prepare_only:
        with single_worker():prepare(args.cpu_budget_hours,args.formal_backend)
        print("Overnight protocol locked; no formal training launched.")
    elif args.check_only:
        verify_lock();print("Frozen sources, pretrained files and historical artifacts verified.")
    elif args.launch:
        verify_lock()
        from continue_v3 import process_alive
        current=OUT/"controller_status.json"
        if current.exists():
            previous=json.loads(current.read_text())
            if process_alive(previous["pid"]):raise RuntimeError("Another overnight worker is active.")
            if previous["stage"]=="engineering_complete":raise RuntimeError("Engineering queue already complete.")
        flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP
        with (OUT/"controller.log").open("a",encoding="utf-8") as stream:
            child=subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--run"]+(["--resume"] if args.resume else []),
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags,close_fds=True)
        dump(OUT/"launch.json",dict(child_pid=child.pid,time=now(),source_sha256=sha256(__file__)))
        print("Detached overnight worker launched:",child.pid)
    else:run(args.resume)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8",line_buffering=True)
    main()
