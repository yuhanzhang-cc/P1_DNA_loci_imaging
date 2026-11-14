"""Portable sealed-kit controller. Verification/status never start training."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tarfile,time
ROOT=Path(__file__).resolve().parents[4]


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def verify(root):
    root=Path(root).resolve();manifest=read(root/"bundle_manifest.json")
    if manifest["kind"]!="signed_offset_v1" or manifest["test_arrays_present"]:raise ValueError("Unexpected bundle")
    for name,digest in manifest["files"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=digest:raise ValueError("Bundle changed: "+name)
    return dict(bundle_verified=True,files=len(manifest["files"]),test_arrays_present=False)


def command(root,method,mode,seed=42,resume=False):
    root=Path(root);entry=root/"deepblink_pytorch/scripts/diagnostics/mechanism_v1/cli.py"
    cmd=[sys.executable,"-u",str(entry),"--config",str(root/"configs"/(method+".json")),
        "--data-root",str(root),"--assets",str(root/"assets"),"--device","cuda","--seed",str(seed),"--"+mode]
    if mode=="smoke":cmd += ["--output",str(root/"outputs/probes"/(method+"_"+time.strftime("%Y%m%dT%H%M%S")))]
    elif mode in ("train","ready-check"):
        cmd += ["--approval",str(root/"decisions"/(method+".json")),"--probe",str(root/"decisions"/(method+"_probe.json"))]
        if mode=="train":cmd += ["--output",str(root/"outputs/trials"/method/f"seed_{seed}")]
    if resume:cmd.append("--resume")
    return cmd


def status(root):
    result=[]
    for method in ("S0","P0"):
        for seed in (42,43,44):
            p=Path(root)/"outputs/trials"/method/f"seed_{seed}/status.json"
            s=read(p) if p.exists() else dict(status="not_started",epoch=0,updates=0)
            row=dict(method=method,seed=seed,**s);result.append(row)
            print(method,seed,s["status"],f"epoch {s.get('epoch',0)}/100",f"updates {s.get('updates',0)}/2200")
    return result


def check_complete(folder):
    folder=Path(folder);state=read(folder/"status.json");selection=read(folder/"selection.json")
    if state.get("status")!="complete" or state.get("engineering_only") is not False or state.get("test_evaluated") is not False:
        raise ValueError("Formal task not complete")
    if (state["epoch"],state["updates"],selection["checkpoint_count"],selection["epoch_threshold_pairs"])!=(100,2200,20,620):
        raise ValueError("Formal budget incomplete")
    for role,name in (("primary","best_model.pt"),("native_secondary","native_selected_model.pt")):
        if sha(folder/name)!=selection[role]["checkpoint_sha256"]:raise ValueError("Selected checkpoint changed")
        metrics=read(folder/role/"metrics.json")
        if any(metrics[k]!=selection[role][k] for k in ("tp","fp","fn","f1")):raise ValueError("Selection/export differs")
    snapshots=read(folder/"snapshots.json")
    if [s["epoch"] for s in snapshots]!=list(range(5,101,5)):raise ValueError("Snapshot schedule differs")
    for snapshot in snapshots:
        source=folder/f"epoch_{snapshot['epoch']:03d}"
        if sha(source/"model.pt")!=snapshot["weight_sha256"] or sha(source/"candidates.npz")!=snapshot["candidates_sha256"]:
            raise ValueError("Snapshot content differs")
    return True


def pack(root,method,seeds):
    root=Path(root);paths=[]
    for seed in seeds:
        folder=root/"outputs/trials"/method/f"seed_{seed}";check_complete(folder)
        for p in folder.rglob("*"):
            if p.is_file() and (p.suffix in (".json",".csv") or p.name in ("best_model.pt","native_selected_model.pt","recovery.pt")):
                if "orphaned_snapshots" not in p.relative_to(folder).parts:paths.append(p)
        for role in ("primary","native_secondary"):
            epoch=read(folder/"selection.json")[role]["epoch"]
            paths.append(folder/f"epoch_{epoch:03d}/candidates.npz")
    paths += [root/"bundle_manifest.json",root/"configs"/(method+".json"),root/"decisions"/(method+".json"),root/"decisions"/(method+"_probe.json")]
    # Return effective source, but never the private development archive.
    paths += list((root/"deepblink_pytorch/deepblink_torch").rglob("*.py"))
    for directory in ("research_v4","diagnostics/mechanism_v1","diagnostics/scaffolds"):
        paths += list((root/"deepblink_pytorch/scripts"/directory).glob("*.py"))
    paths += list((root/"configs").glob("*.json"))
    paths += [p for p in (root/"outputs/controller").glob(method+"*.log") if p.is_file()]
    name=root/(method+"_results_"+time.strftime("%Y%m%dT%H%M%S")+".tar.gz")
    with name.open("xb") as f:
        with tarfile.open(fileobj=f,mode="w:gz") as archive:
            for p in sorted(set(paths)):archive.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
    with tarfile.open(name) as archive:
        if len(archive.getmembers())!=len(set(paths)):raise ValueError("Archive member count differs")
    name.with_name(name.name+".sha256").write_bytes((sha(name)+"  "+name.name+"\n").encode("ascii"))
    print(json.dumps(dict(path=str(name),bytes=name.stat().st_size,files=len(set(paths)))))


def worker(root,method,seeds,resume):
    out=Path(root)/"outputs/controller";out.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    with FileLock(str(out/"formal_queue.lock"),timeout=0):
        try:
            for seed in seeds:subprocess.run(command(root,method,"train",seed,resume),cwd=root,check=True)
            for seed in seeds:check_complete(Path(root)/"outputs/trials"/method/f"seed_{seed}")
            (out/(method+"_complete.json")).write_text(json.dumps(dict(status="complete",seeds=seeds)),encoding="utf-8")
        except BaseException:
            import traceback
            (out/(method+"_failure.txt")).write_text(traceback.format_exc(),encoding="utf-8");raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["verify","status","smoke","start","resume","worker","pack"])
    p.add_argument("--method",choices=["S0","P0"],default="S0");p.add_argument("--seeds",nargs="+",type=int,choices=[42,43,44],default=[42])
    p.add_argument("--resume-worker",action="store_true");p.add_argument("--dry-run",action="store_true")
    a=p.parse_args();print(json.dumps(verify(ROOT)))
    if len(a.seeds)!=len(set(a.seeds)):raise ValueError("Duplicate seeds")
    if a.action=="verify":return
    if a.action=="status":status(ROOT);return
    if a.dry_run:
        print(json.dumps(dict(dry_run=True,commands=[command(ROOT,a.method,"train" if a.action in ("start","resume","worker") else "smoke",s,a.action=="resume") for s in a.seeds],training_started=False)));return
    if a.action=="smoke":subprocess.run(command(ROOT,a.method,"smoke"),cwd=ROOT,check=True);return
    if a.action=="pack":pack(ROOT,a.method,a.seeds);return
    if a.action=="worker":worker(ROOT,a.method,a.seeds,a.resume_worker);return
    subprocess.run(command(ROOT,a.method,"ready-check"),cwd=ROOT,check=True)
    out=ROOT/"outputs/controller";out.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    with FileLock(str(out/"launch.lock"),timeout=0):
        for method in ("S0","P0"):
            record=out/(method+"_process.json")
            if record.exists():
                try:os.kill(read(record)["pid"],0)
                except ProcessLookupError:pass
                else:raise ValueError("An earlier controller PID is still alive; inspect before relaunch")
        logfile=out/(a.method+"_"+time.strftime("%Y%m%dT%H%M%S")+".log")
        cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"worker","--method",a.method,"--seeds",*map(str,a.seeds)]
        if a.action=="resume":cmd.append("--resume-worker")
        with logfile.open("xb") as f:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record=out/(a.method+"_process.json")
        record.write_text(json.dumps(dict(pid=process.pid,log=str(logfile),seeds=a.seeds)),encoding="utf-8")
        print(json.dumps(dict(started=True,pid=process.pid,log=str(logfile))))


if __name__=="__main__":main()
