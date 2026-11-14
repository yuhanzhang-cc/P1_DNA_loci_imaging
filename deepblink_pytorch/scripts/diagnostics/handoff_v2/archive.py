"""Lean return artifacts and safe local ingestion. Study data never enters Git."""
from pathlib import Path,PurePosixPath
import gzip,tarfile,uuid
from .common import read,write,sha,fresh,inside


def pack_return(root,tasks,probes,gate_path,engineering_fixture=False):
    root=Path(root).resolve()
    from diagnostics.mechanism_v1.controller import check_complete
    selected=set();task_records=[]
    for task in tasks:
        folder=root/"outputs/trials"/task["method"]/f"seed_{task['seed']}"
        if not engineering_fixture:check_complete(folder)
        selection=read(folder/"selection.json")
        for p in folder.rglob("*"):
            if p.is_file() and "orphaned_snapshots" not in p.parts and (p.suffix in (".json",".csv") or p.name in ("best_model.pt","native_selected_model.pt","recovery.pt")):selected.add(p)
        for role in ("primary","native_secondary"):selected.add(folder/f"epoch_{selection[role]['epoch']:03d}/candidates.npz")
        task_records.append(dict(**task,folder=folder.relative_to(root).as_posix(),budget=[100,2200,20,620],
            cloud_all_snapshot_hashes_verified=True))
    for directory in ("decisions","configs","outputs/pipeline"):
        selected|={p for p in (root/directory).rglob("*") if p.is_file() and p.suffix in (".json",".csv",".txt")}
    for directory in ("deepblink_pytorch/deepblink_torch","deepblink_pytorch/scripts/research_v4","deepblink_pytorch/scripts/diagnostics"):
        selected|={p for p in (root/directory).rglob("*") if p.is_file() and p.suffix in (".py",".html",".js") and "__pycache__" not in p.parts}
    selected|={root/"bundle_manifest.json",Path(gate_path)}|{Path(p) for p in probes.values()}
    selected|={p for p in (root/"outputs/pipeline").glob("*.log") if p.is_file()}
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(selected)}
    if any("assets/" in name or "source_key" in name or "extracted_" in name for name in files):raise ValueError("Unexpected private source asset in result packet")
    manifest=dict(schema="p1_return_v2",files=files,tasks=task_records,test_evaluated=False,
        engineering_only=engineering_fixture,selected_candidates_included=True,unselected_snapshot_models_omitted=True,
        all_prethreshold_candidates=True,bundle_sha256=sha(root/"bundle_manifest.json"))
    manifest_path=root/("return_manifest_"+uuid.uuid4().hex[:12]+".json");write(manifest_path,manifest)
    archive=root/(manifest_path.stem+".tar.gz")
    with archive.open("xb") as f:
        with tarfile.open(fileobj=f,mode="w:gz") as tar:
            for p in sorted(selected):tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
            tar.add(manifest_path,arcname="return_manifest.json",recursive=False)
    digest=sha(archive);archive.with_name(archive.name+".sha256").write_bytes((digest+"  "+archive.name+"\n").encode("ascii"))
    return dict(path=str(archive),sha256=digest,bytes=archive.stat().st_size,members=len(files)+1)


def verify_return(root,allow_engineering=False):
    root=Path(root).resolve();manifest=read(root/"return_manifest.json")
    if manifest.get("schema")!="p1_return_v2" or manifest.get("test_evaluated") is not False:raise ValueError("Wrong return protocol")
    if manifest.get("engineering_only") is not False and not allow_engineering:raise ValueError("Engineering fixtures cannot complete a formal experiment")
    if not 1<=len(manifest.get("tasks",[]))<=6:raise ValueError("No complete registered tasks in return")
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual!=set(manifest["files"])|{"return_manifest.json"}:raise ValueError("Undeclared or missing return files")
    for name,digest in manifest["files"].items():
        if sha(inside(root,name))!=digest:raise ValueError("Return hash differs: "+name)
    seen=set()
    for task in manifest["tasks"]:
        key=(task["method"],task["seed"])
        if key in seen or task["method"] not in ("S0","P0") or task["seed"] not in (42,43,44):raise ValueError("Unregistered/duplicate task")
        seen.add(key);folder=inside(root,task["folder"])
        state=read(folder/"status.json");choice=read(folder/"selection.json");snapshots=read(folder/"snapshots.json")
        if (state.get("status"),state.get("epoch"),state.get("updates"),choice["checkpoint_count"],choice["epoch_threshold_pairs"])!=("complete",100,2200,20,620):raise ValueError("Incomplete formal budget")
        if state.get("engineering_only") is not False and not allow_engineering:raise ValueError("Engineering trial in formal packet")
        if [r["epoch"] for r in snapshots]!=list(range(5,101,5)):raise ValueError("Wrong snapshot schedule")
        from .common import table
        history=table(folder/"training_metrics.csv")
        if len(history)!=100 or [int(r["epoch"]) for r in history]!=list(range(1,101)) or int(history[-1]["optimizer_updates"])!=2200:raise ValueError("Training history budget differs")
        locks=read(folder/"method_locks.json")
        if len(locks)!=1 or locks[0]["method"]!=task["method"] or locks[0]["seed"]!=task["seed"] or locks[0]["selection_sha256"]!=sha(folder/"selection.json"):raise ValueError("Model lock differs")
        for role,name in (("primary","best_model.pt"),("native_secondary","native_selected_model.pt")):
            if sha(folder/name)!=choice[role]["checkpoint_sha256"]:raise ValueError("Selected model differs")
            metrics=read(folder/role/"metrics.json")
            if metrics.get("test_evaluated") is not False or any(metrics[k]!=choice[role][k] for k in ("tp","fp","fn","f1")):raise ValueError("Selection/export mismatch")
            cache=folder/f"epoch_{choice[role]['epoch']:03d}/candidates.npz"
            snapshot=next(r for r in snapshots if r["epoch"]==choice[role]["epoch"])
            if sha(cache)!=snapshot["candidates_sha256"]:raise ValueError("Selected candidates changed")
            ap=read(folder/role/"ranked_ap.json")
            if ap.get("threshold_floor") is not None or ap.get("policy")!="ranked_point_ap_greedy_step_v1" or ap.get("ap")!=metrics.get("ranked_ap"):raise ValueError("Ranked AP provenance differs")
    return dict(verified=True,tasks=len(seen),engineering_only=manifest["engineering_only"],test_evaluated=False,
        local_selected_hashes_verified=True,all_snapshot_hashes_scope="cloud verification recorded; lean return omits unselected weights")


def ingest(archive,checksum,output,allow_engineering=False,max_bytes=8*1024**3,source_manifest=None):
    archive=Path(archive);expected=Path(checksum).read_text(encoding="utf-8-sig").split()[0]
    if len(expected)!=64 or sha(archive)!=expected:raise ValueError("Download SHA256 mismatch")
    # Drain gzip first to check CRC/trailer even when tar ends before gzip EOF.
    with gzip.open(archive,"rb") as f:
        total=0
        for block in iter(lambda:f.read(8*1024*1024),b""):
            total+=len(block)
            if total>max_bytes:raise ValueError("Archive exceeds extraction budget")
    with tarfile.open(archive,"r:gz") as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(members)>10000 or len(names)!=len(set(names)):raise ValueError("Too many or duplicate archive members")
        for m in members:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or ".." in p.parts or "\\" in m.name or ":" in m.name or any(s in (".git",".env") for s in p.parts):raise ValueError("Unsafe archive entry")
        output=fresh(output)
        for m in members:
            target=inside(output,m.name);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as src,target.open("xb") as dst:
                while block:=src.read(1024*1024):dst.write(block)
    result=verify_return(output,allow_engineering)
    if source_manifest and read(Path(output)/"return_manifest.json")["bundle_sha256"]!=sha(source_manifest):raise ValueError("Return belongs to another uploaded source bundle")
    return dict(**result,path=str(output),download_sha256=expected,uploaded_source_binding_verified=source_manifest is not None)
