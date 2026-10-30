"""Manifest-checked dev-only bundle entry point; CUDA must pass its own smoke."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only",action="store_true")
    mode.add_argument("--smoke",action="store_true")
    mode.add_argument("--train",action="store_true")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--seeds",type=int,nargs="+",choices=[42,43,44],default=[42,43,44])
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/spotiflow_cuda")
    args=parser.parse_args()
    import numpy as np
    import torch
    from deepblink_torch.experiment_log import sha256,write_json
    from deepblink_torch.spotiflow_bridge import checked_coordinates
    manifest=json.loads((ROOT/"bundle_manifest.json").read_text(encoding="utf-8"))
    for relative,digest in manifest["files"].items():
        path=ROOT/relative
        if not path.resolve().is_relative_to(ROOT.resolve()) or sha256(path)!=digest:
            raise ValueError("Bundle path/hash mismatch: "+relative)
    data={}
    with np.load(ROOT/"assets/development_only.npz",allow_pickle=True) as archive:
        expected={f"{k}_{s}" for k in ("x","y","sample_id","original_shape") for s in ("train","valid")}
        if set(archive.files)!=expected:raise ValueError("Bundle includes unexpected/test arrays.")
        for subset,size,count in (("train",176,4156),("valid",22,372)):
            x,y,ids,shapes=[archive[f"{key}_{subset}"] for key in ("x","y","sample_id","original_shape")]
            if len(x)!=size or sum(map(len,y))!=count:raise ValueError("Development counts differ.")
            for image,truth,shape in zip(x,y,shapes):
                if image.shape!=tuple(shape):raise ValueError("Bundled image geometry differs.")
                checked_coordinates(truth,shape)
            data[subset]=dict(images=list(x),y=y,sample_id=ids,original_shape=shapes)
    if len(set(args.seeds))!=len(args.seeds):raise ValueError("Duplicate training seed.")
    if args.verify_only:
        print(json.dumps(dict(bundle_verified=True,train_images=176,valid_images=22,test_arrays_present=False)))
        return
    torch.set_num_threads(4)
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; do not silently switch backend.")
    if args.smoke:
        from spotiflow_overnight.test_engineering import run_checks
        run_checks(args.output/"engineering_smoke",device=args.device)
        return
    marker=args.output/"engineering_smoke/checks.json"
    smoke=json.loads(marker.read_text()) if marker.exists() else {}
    if not smoke.get("passed") or smoke.get("device")!=args.device or smoke.get("torch_version")!=torch.__version__:
        raise ValueError("Run --smoke on the selected device before formal training.")
    from spotiflow.model import Spotiflow
    from spotiflow_overnight.engine import train_native,default_training,evaluate_snapshot
    from spotiflow_overnight.evaluation import finalize_finetune
    args.output.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    worker_lock=FileLock(str(args.output/".runner.lock"),timeout=0)
    worker_lock.acquire()
    config=dict(device=args.device,seeds=[42,43,44],maximum_tasks=3,epochs=100,updates_per_seed=4400,
                checkpoint_epochs=list(range(5,101,5)),threshold_count=31,torch=torch.__version__,
                device_name=torch.cuda.get_device_name(0) if args.device=="cuda" else "CPU",
                bundle_manifest_sha256=sha256(ROOT/"bundle_manifest.json"),test_arrays_loaded=False)
    path=args.output/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError("Frozen cloud config changed.")
    write_json(path,config)
    for seed in args.seeds:
        folder=args.output/f"seed_{seed}"
        status_path=folder/"status.json"
        if status_path.exists():
            if json.loads(status_path.read_text())["status"]=="complete":continue
            if not args.resume:raise ValueError("Partial cloud trial requires audited --resume.")
        folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/"recovery/last.ckpt"
        if args.resume and status_path.exists() and not checkpoint.exists():raise ValueError("No complete epoch to resume.")
        write_json(status_path,dict(status="running",seed=seed,started_unix=time.time()))
        model=Spotiflow.from_folder(ROOT/"assets/general",inference_mode=False)
        try:
            train_native(model,data["train"]["images"],list(data["train"]["y"]),data["valid"]["images"],list(data["valid"]["y"]),
                folder,seed=seed,config=default_training(),evaluate=evaluate_snapshot(data["valid"],args.device),
                device=args.device,resume=checkpoint if args.resume and checkpoint.exists() else None,
                event=lambda event,**fields:print(json.dumps(dict(event=event,**fields)),flush=True))
            finalize_finetune(folder,data["valid"])
        except BaseException as error:
            write_json(status_path,dict(status="failed_or_interrupted",seed=seed,error=repr(error),time_unix=time.time()))
            (folder/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
            raise
        write_json(status_path,dict(status="complete",seed=seed,finished_unix=time.time(),test_evaluated=False))
        del model
        if args.device=="cuda":torch.cuda.empty_cache()
    worker_lock.release()


if __name__=="__main__":
    main()
