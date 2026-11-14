from pathlib import Path
import argparse,json,os,subprocess,sys,uuid,zipfile
from .common import ROOT,read,write,sha


def main():
    p=argparse.ArgumentParser(description="Prefabricated no-training preparation / gated future CUDA pipeline")
    sub=p.add_subparsers(dest="action",required=True)
    for name in ("plan","verify","status","run","start","failure","pack"):
        s=sub.add_parser(name);s.add_argument("--root",type=Path,default=ROOT)
        if name in ("run","start"):s.add_argument("--resume",action="store_true");s.add_argument("--dry-run",action="store_true")
    s=sub.add_parser("reviewer");s.add_argument("--public",type=Path,required=True);s.add_argument("--output",type=Path,required=True);s.add_argument("--synthetic",action="store_true")
    s=sub.add_parser("check-review");s.add_argument("--responses",type=Path,required=True);s.add_argument("--expected-ids",type=Path,required=True)
    s=sub.add_parser("freeze");s.add_argument("--responses",type=Path,required=True);s.add_argument("--decision",type=Path,required=True);s.add_argument("--output",type=Path,required=True);s.add_argument("--config-dir",type=Path)
    s=sub.add_parser("doctor");s.add_argument("--output",type=Path,required=True);s.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    s=sub.add_parser("build");s.add_argument("--data-root",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("ingest");s.add_argument("--archive",type=Path,required=True);s.add_argument("--checksum",type=Path,required=True);s.add_argument("--output",type=Path,required=True);s.add_argument("--source-manifest",type=Path,required=True)
    s=sub.add_parser("report");s.add_argument("--data-root",type=Path,required=True);s.add_argument("--returned",type=Path);s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("corrections");s.add_argument("--responses",type=Path,required=True);s.add_argument("--source-key",type=Path,required=True);s.add_argument("--data-root",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("prepare-corrections");s.add_argument("--source-key",type=Path,required=True);s.add_argument("--data-root",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("check-revision");s.add_argument("--reviewed-csv",type=Path,required=True);s.add_argument("--data-root",type=Path,required=True);s.add_argument("--output",type=Path,required=True);s.add_argument("--coordinate-units",choices=["pixels"],required=True)
    s=sub.add_parser("fiji-qa");s.add_argument("--fiji-root",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("propose-labels");s.add_argument("--data-root",type=Path,required=True);s.add_argument("--reviewed-csv",type=Path,nargs="+",required=True);s.add_argument("--reviewer",required=True);s.add_argument("--reason",required=True);s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("recheck-labels");s.add_argument("--data-root",type=Path,required=True);s.add_argument("--revision",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=None
    if a.action in ("plan","status","run","start"):
        from .pipeline import plan,status,execute
        root=a.root.resolve()
        if a.action=="plan" or a.action in ("run","start") and a.dry_run:result=plan(root)
        elif a.action=="status":result=status(root)
        elif a.action=="run":result=execute(root,a.resume)
        else:
            from diagnostics.mechanism_v1.controller import verify
            from .freeze import verify_frozen
            verify(root);verify_frozen(root/"decisions")
            from filelock import FileLock
            folder=root/"outputs/pipeline";folder.mkdir(parents=True,exist_ok=True)
            with FileLock(str(folder/"launch.lock"),timeout=0):
                record=folder/"process.json"
                if record.exists():
                    pid=read(record)["pid"];proc=Path(f"/proc/{pid}/cmdline")
                    if proc.exists() and b"handoff_v2" in proc.read_bytes():raise ValueError("This pipeline process is still alive")
                logfile=folder/("run_"+uuid.uuid4().hex[:12]+".log")
                cmd=[sys.executable,"-u",str(Path(__file__).with_name("cli.py")),"run","--root",str(root)]
                if a.resume:cmd.append("--resume")
                kwargs=dict(start_new_session=True) if os.name!="nt" else dict(creationflags=subprocess.CREATE_NO_WINDOW)
                with logfile.open("xb") as f:child=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,**kwargs)
                result=dict(pid=child.pid,log=str(logfile),started=True);write(record,result,replace=True)
    elif a.action=="verify":
        from diagnostics.mechanism_v1.controller import verify
        result=verify(a.root)
    elif a.action=="reviewer":
        from .reviewer import build
        result=build(a.public,a.output,a.synthetic)
    elif a.action=="check-review":
        from .reviewer import validate_answers
        from .common import table
        result=validate_answers(table(a.responses),read(a.expected_ids))
    elif a.action=="freeze":
        from .freeze import freeze,verify_frozen
        if (a.output/"freeze_manifest.json").exists():
            marker,decision=verify_frozen(a.output)
            if marker["files"]["blind_audit_responses.csv"]!=sha(a.responses) or marker["files"]["research_decision.json"]!=sha(a.decision):raise ValueError("Existing frozen decision differs; choose a new revision")
            result=dict(already_frozen=True,verified=True,formal_training_started=False)
        else:result=freeze(a.responses,a.decision,a.output,a.config_dir)
    elif a.action=="doctor":
        from .doctor import diagnose
        result=diagnose(a.output,a.device)
    elif a.action=="build":
        from .bundle import build
        result=build(a.data_root,a.output)
    elif a.action=="ingest":
        from .archive import ingest
        result=ingest(a.archive,a.checksum,a.output,source_manifest=a.source_manifest)
    elif a.action=="report":
        from .report import render
        result=render(a.data_root,a.output,a.returned)
    elif a.action=="corrections":
        from .corrections import build
        result=build(a.responses,a.source_key,a.data_root,a.output)
    elif a.action=="prepare-corrections":
        from .corrections import prepare
        result=prepare(a.source_key,a.data_root,a.output)
    elif a.action=="check-revision":
        from .common import table
        from .corrections import check_revision
        from diagnostics.peak_offset_diag import load_validation
        rows=table(a.reviewed_csv);ids={r["sample_id"] for r in rows}
        if len(ids)!=1:raise ValueError("Provide one image's complete reviewed point list per CSV")
        sid=next(iter(ids));valid,_=load_validation(a.data_root/"outputs/next_stage_v4_recipe_20261101/04_bundles/ce2d1821ceec/p1_v4_training")
        if sid not in valid:raise ValueError("Unknown validation image")
        result=check_revision(valid[sid]["truth"],[[float(r["y"]),float(r["x"])] for r in rows],valid[sid]["shape"])
        result.update(sample_id=sid,reviewed_csv_sha256=sha(a.reviewed_csv),coordinate_units="pixels",original_labels_overwritten=False)
        write(a.output,result)
    elif a.action=="fiji-qa":
        from .fiji_qa import run
        result=run(a.fiji_root,a.output)
    elif a.action=="propose-labels":
        from diagnostics.mechanism_v1.runner import corpus
        from .label_revision import propose
        data=corpus(a.data_root)
        result=propose(data.paths["development"],sha(data.paths["development"]),a.reviewed_csv,a.output,a.reviewer,a.reason)
    elif a.action=="recheck-labels":
        from .label_revision import reevaluate
        result=reevaluate(a.data_root,a.revision,a.output)
    elif a.action=="failure":
        root=a.root.resolve();folder=root/"outputs/pipeline";folder.mkdir(parents=True,exist_ok=True)
        archive=root/("failure_evidence_"+uuid.uuid4().hex[:12]+".zip")
        files=[root/"bundle_manifest.json"]+[p for p in folder.rglob("*") if p.is_file() and p.suffix in (".json",".log",".txt")]
        with archive.open("xb") as f:
            with zipfile.ZipFile(f,"w",zipfile.ZIP_DEFLATED) as z:
                for path in sorted(set(files)):z.write(path,path.relative_to(root).as_posix())
        archive.with_name(archive.name+".sha256").write_bytes((sha(archive)+"  "+archive.name+"\n").encode())
        result=dict(path=str(archive),sha256=sha(archive),private_assets_included=False)
    elif a.action=="pack":
        from .archive import pack_return
        root=a.root.resolve();state=read(root/"outputs/pipeline/status.json")
        if not state["completed_tasks"]:raise ValueError("No complete tasks to pack")
        gate=root/"outputs/pipeline"/("manual_pack_gate_"+uuid.uuid4().hex[:12]+".json")
        write(gate,state.get("gate",dict(passed=None,reason="screen not completed",test_evaluated=False)))
        result=pack_return(root,state["completed_tasks"],{m:Path(v) for m,v in state.get("probes",{}).items()},gate)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
