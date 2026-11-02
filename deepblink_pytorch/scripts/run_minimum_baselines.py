"""Sequential R0/H0 handoff: six already registered tasks, no recipe expansion."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"deepblink_pytorch"),str(ROOT/"deepblink_pytorch/scripts")]


def main():
    import torch
    from filelock import FileLock
    from research_v4.common import read_json,dump,sha256
    output=ROOT/"outputs/research_v4_cuda";output.mkdir(parents=True,exist_ok=True)
    recipe=read_json(output/"recipe/protocol.json")
    if not torch.cuda.is_available():raise RuntimeError("Actual CUDA required")
    if str(torch.__version__)!=recipe["torch"] or torch.cuda.get_device_name(0)!=recipe["gpu"]:
        raise ValueError("Continue on the same recorded recipe CUDA environment/GPU; preserve a changed setup as a separate protocol")
    if sha256(ROOT/"bundle_manifest.json")!=recipe["manifest_sha256"]:raise ValueError("Training bundle changed")
    runner=ROOT/"deepblink_pytorch/scripts/cloud_recipe_train.py"
    control=ROOT/"deepblink_pytorch/scripts/cloud_job_control.py"
    def run(script,*arguments):subprocess.run([sys.executable,"-u",str(script),*arguments],cwd=ROOT,check=True)
    with FileLock(str(output/".minimum_baselines.lock"),timeout=0):
        run(runner,"--check-complete","--queue","recipe")
        for queue in ("baseline_r0","baseline_h0"):
            dump(output/"minimum_baseline_status.json",dict(status="running",queue=queue,pid=os.getpid(),maximum_new_tasks=6,extra_search=False))
            with FileLock(str(output/".gpu_runner.lock"),timeout=0):
                run(runner,"--smoke","--device","cuda","--queue",queue)
            run(runner,"--train","--device","cuda","--queue",queue,"--resume")
            run(runner,"--check-complete","--queue",queue)
            run(control,"pack","--queue",queue)
        dump(output/"minimum_baseline_status.json",dict(status="complete",queues=["baseline_r0","baseline_h0"],formal_tasks=6,spotiflow_scratch_started=False))
        print("Minimum baseline queues complete. Download baseline_r0_results.tar.gz and baseline_h0_results.tar.gz plus both.sha256 files.",flush=True)


if __name__=="__main__":
    try:main()
    except BaseException as error:
        destination=ROOT/"outputs/research_v4_cuda/minimum_baseline_failure.txt"
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(__import__("traceback").format_exc(),encoding="utf-8")
        status=destination.with_name("minimum_baseline_status.json")
        if status.exists():
            current=json.loads(status.read_text(encoding="utf-8"))
            if current.get("pid")==os.getpid():
                current.update(status="failed_or_interrupted",error=repr(error))
                status.write_text(json.dumps(current,indent=2)+"\n",encoding="utf-8")
        raise
