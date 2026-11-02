"""Start a single explicit queue or package completed selected results."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/"deepblink_pytorch/scripts/cloud_recipe_train.py"


def run(*args):
    subprocess.run([sys.executable,str(RUNNER),*args],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["start","resume","pack"])
    p.add_argument("--queue",choices=["recipe","baseline_r0","baseline_h0","spotiflow_scratch"],default="recipe")
    args=p.parse_args()
    output=ROOT/"outputs/research_v4_cuda";output.mkdir(parents=True,exist_ok=True)
    if args.action in ("start","resume"):
        run("--verify-only")
        run("--ready-check","--queue",args.queue)
        # The runner enforces actual CUDA smoke, a manifest binding and one-GPU lock.
        log=output/(args.queue+"_runner.log")
        command=[sys.executable,"-u",str(RUNNER),"--train","--queue",args.queue,"--device","cuda"]
        if args.action=="resume":command.append("--resume")
        with log.open("ab",buffering=0) as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!="nt")
        (output/(args.queue+"_launch.json")).write_text(json.dumps(dict(pid=process.pid,command=command,log=str(log)),indent=2)+"\n",encoding="utf-8")
        print(f"Started {args.queue}; PID {process.pid}. Check bash 04_status.sh and {log}")
        return
    run("--check-complete","--queue",args.queue)
    target=ROOT/(args.queue+"_results.tar.gz")
    folder=output/args.queue
    chosen=[]
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or "snapshots" in path.parts or "orphaned_snapshots" in path.parts or "recovery" in path.parts or path.name=="recovery.pt" or any(s.startswith("epoch_") for s in path.relative_to(folder).parts):continue
        if path.suffix in (".pt",".ckpt",".npz") and path.name not in ("best_model.pt","native_selected_model.pt","best.pt"):continue
        if path.suffix in (".tmp",):continue
        chosen.append(path)
    chosen+=[ROOT/"bundle_manifest.json"]
    chosen+=[p for p in (ROOT/"environment_actual.txt",ROOT/"nvidia_smi_setup.txt") if p.exists()]
    chosen+=list(output.glob(args.queue+"_*.log"))
    chosen+=list((output/"engineering_smoke").rglob("checks.json"))+list((output/"engineering_smoke").glob("*.json"))
    with tarfile.open(target,"w:gz",compresslevel=5) as archive:
        for path in dict.fromkeys(chosen):archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    with tarfile.open(target,"r:gz") as archive:
        names=archive.getnames()
        if any(n.startswith("/") or ".." in Path(n).parts for n in names):raise ValueError("Unsafe archive member")
        # Read all member bytes, detecting truncated gzip data before delivery.
        for member in archive:
            stream=archive.extractfile(member)
            if stream:
                while stream.read(8*1024*1024):pass
    h=hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    target.with_name(target.name+".sha256").write_text(h.hexdigest()+"  "+target.name+"\n",encoding="ascii")
    print(f"Download {target} and {target.name}.sha256; {len(names)} files. Recovery/epoch caches remain on server.")


if __name__=="__main__":main()
