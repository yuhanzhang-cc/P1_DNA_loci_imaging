"""Explicit roots, development-only inputs and atomic, traceable outputs."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
import yaml


SEEDS = (42, 43, 44)
WEAK_CUTOFF = 2.4566830772038957
FEATURES = "outputs/next_stage_v3_closing_controls/02_analysis/validation_truth_features.csv"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def replace_with_retry(source,destination):
    """Windows indexers/readers can briefly deny replacement of an existing file."""
    for attempt in range(9):
        try:
            os.replace(source,destination)
            return
        except PermissionError:
            if attempt==8:raise
            time.sleep(min(.05*2**attempt,1.))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, records, fields=None):
    records = list(records)
    if fields is None:
        if not records: raise ValueError("An empty table needs explicit columns")
        fields = list(records[0])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(clean(r) for r in records)
    replace_with_retry(temporary, path)


def user_time(data_root=None, stamp=None):
    try:
        zone = ZoneInfo("America/New_York")
    except Exception:
        if data_root is None: raise
        payload = (Path(data_root)/"outputs/next_stage_v3_closing_controls/00_protocol/America_New_York.tzif").read_bytes()
        zone = ZoneInfo.from_file(io.BytesIO(payload), key="America/New_York")
    return datetime.fromtimestamp(time.time() if stamp is None else stamp, zone).isoformat(timespec="seconds")


def load_spec(path):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if spec["spec_id"] != "P1-NIGHT-V4-20261101" or spec["data"]["test_arrays_allowed"]:
        raise ValueError("Unsupported or unsafe v4 specification")
    return spec


class Corpus:
    def __init__(self, data_root, spec, assets_root=None):
        self.data_root, self.spec = Path(data_root).resolve(), spec
        self.accesses = []
        path = Path(assets_root)/"development_only.npz" if assets_root else self.data_root/spec["data"]["development_npz"]
        if sha256(path) != spec["data"]["development_sha256"]: raise ValueError("Development archive changed")
        self.splits = {}
        with np.load(path, allow_pickle=True) as archive:
            if set(archive.files) != set(spec["data"]["allowed_npz_keys"]): raise ValueError("Unexpected/test archive keys")
            for subset in ("train", "valid"):
                data = {}
                for key in ("x", "y", "sample_id", "original_shape"):
                    name = f"{key}_{subset}"
                    self.accesses.append(name)
                    data[key] = archive[name]
                data["sample_id"] = np.asarray(data["sample_id"], dtype=str)
                data["x"] = [np.asarray(x, dtype=np.float32).copy() for x in data["x"]]
                data["y"] = [np.asarray(y, dtype=np.float32).copy() for y in data["y"]]
                if len(data["x"]) != spec["data"][subset+"_images"] or sum(map(len,data["y"])) != spec["data"][subset+"_points"]:
                    raise ValueError("Development counts differ")
                for x, y, shape in zip(data["x"],data["y"],data["original_shape"]):
                    if x.shape != tuple(shape) or not np.isfinite(x).all() or y.ndim != 2 or y.shape[1] != 2:
                        raise ValueError("Malformed image/coordinates")
                    if not np.isfinite(y).all() or np.any(y < 0) or np.any(y >= shape): raise ValueError("Coordinates outside original FOV")
                self.splits[subset] = data
        a,b = (set(self.splits[s]["sample_id"]) for s in ("train","valid"))
        if a & b or set(spec["data"]["excluded_samples"]) & (a|b): raise ValueError("Split overlap or excluded samples")
        features = Path(assets_root)/"validation_truth_features.csv" if assets_root else self.data_root/FEATURES
        table = rows(features)
        index = {(r["sample_id"],int(r["true_index"])):r for r in table}
        weak = []
        valid = self.splits["valid"]
        for sid, truth in zip(valid["sample_id"],valid["y"]):
            flags = []
            for i,(y,x) in enumerate(truth):
                r=index[(sid,i)]
                if not np.allclose([float(r["y"]),float(r["x"])],[y,x],atol=1e-6,rtol=0): raise ValueError("Weak feature coordinates changed")
                flags.append(r["weak"]=="True")
            weak.append(np.asarray(flags,dtype=bool))
        if sum(map(np.count_nonzero,weak)) != 96 or len(index)!=372: raise ValueError("Weak validation reference changed")
        valid["weak"] = weak
        self.paths = dict(development=path,features=features)


class Context:
    def __init__(self, code_root, data_root, output_root, spec_path):
        self.code_root, self.data_root = Path(code_root).resolve(), Path(data_root).resolve()
        self.out, self.spec_path = Path(output_root).resolve(), Path(spec_path).resolve()
        self.spec = load_spec(self.spec_path)
        self.out.mkdir(parents=True,exist_ok=True)
        self.start = time.monotonic()
        self.cpu_start=time.process_time()
        self.keep_awake=False
        if os.name=="nt":
            import ctypes
            self.keep_awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        self.state = dict(spec_id=self.spec["spec_id"],pid=os.getpid(),started_user_time=user_time(self.data_root),
                          stage="initializing",tasks={},formal_training_tasks_started=0,cuda_verified=False,human_review_completed=False,
                          temporary_keep_awake=self.keep_awake)
        prior=self.out/"controller_status.json"
        if prior.exists():self.state["tasks"]=read_json(prior).get("tasks",{})
        self.lock = threading.RLock()
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat,daemon=True)
        self.thread.start()
        self.corpus = None

    def _heartbeat(self):
        while not self.halt.wait(30): self.status()

    def status(self, stage=None, **fields):
        with self.lock:
            if stage is not None: self.state["stage"]=stage
            self.state.update(fields,updated_user_time=user_time(self.data_root),elapsed_seconds=time.monotonic()-self.start)
            self.state["cpu_seconds"]=time.process_time()-self.cpu_start
            dump(self.out/"controller_status.json",self.state)
            if stage:
                with (self.out/"execution_events.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(clean(self.state),ensure_ascii=False,allow_nan=False)+"\n")
                print(json.dumps(dict(stage=stage,**fields),ensure_ascii=False),flush=True)

    def task(self,name,status,**details):
        with self.lock:
            self.state["tasks"][name]=dict(status=status,**clean(details))
        self.status(stage=name+"_"+status)

    def close(self):
        self.halt.set(); self.thread.join(timeout=2)
        if self.keep_awake:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        self.status(process_active=False)


def source_hashes(code_root):
    root=Path(code_root)
    paths=list((root/"deepblink_pytorch/deepblink_torch").rglob("*.py"))
    paths+=list((root/"deepblink_pytorch/scripts/research_v4").rglob("*.py"))
    paths+=list((root/"deepblink_pytorch/scripts/research_v4").rglob("*.sh"))
    for name in ("run_recipe_overnight.py","cloud_recipe_train.py","cloud_job_control.py","evaluate_external.py","analyze_research_v4.py","validate_ablation_plan.py","error_context_diagnostics.py"):
        p=root/"deepblink_pytorch/scripts"/name
        if p.exists():paths.append(p)
    paths+=list((root/"deepblink_pytorch").glob("requirements-research-v4.txt"))
    for name in ("__init__.py","common.py","engine.py","evaluation.py"):
        paths.append(root/"deepblink_pytorch/scripts/spotiflow_overnight"/name)
    for name in ("pyproject.toml","README.md"):paths.append(root/"deepblink_pytorch"/name)
    return {p.relative_to(root).as_posix():sha256(p) for p in sorted(paths)}


def verify_protected(data_root, expected):
    for name,digest in expected.items():
        if sha256(Path(data_root)/name)!=digest: raise ValueError("Protected artifact changed: "+name)


def input_audit(ctx):
    ctx.task("input_audit","running")
    spec=ctx.spec; root=ctx.data_root
    corpus=Corpus(root,spec);ctx.corpus=corpus
    split=root/spec["data"]["split_csv"]
    if sha256(split)!=spec["data"]["split_sha256"]: raise ValueError("Split manifest changed")
    split_rows=rows(split)
    by_id={r["sample_id"]:r for r in split_rows}
    raw_hashes={}
    for subset,data in corpus.splits.items():
        for sid,image,truth in zip(data["sample_id"],data["x"],data["y"]):
            r=by_id[sid]
            if r["subset"]!=subset or [int(r["height"]),int(r["width"])]!=list(image.shape) or int(r["n_points"])!=len(truth): raise ValueError("Manifest geometry mismatch")
            for suffix,folder,key in ((".tif","tif","image_sha256"),(".csv","csv","csv_sha256")):
                name=f"data/raw/{folder}/{sid}{suffix}"
                digest=sha256(root/name)
                if digest!=r[key]:raise ValueError("Raw development data changed: "+sid)
                raw_hashes[name]=digest
    old=read_json(root/"outputs/next_stage_spotiflow_20261030/00_protocol/lock.json")
    protected=dict(old["protected_references"])
    for seed in SEEDS:
        for name in ("selection.json","primary/best.pt","primary/thresholds.yaml"):
            path=f"outputs/next_stage_spotiflow_20261030/06_cuda_results/extracted/outputs/spotiflow_cuda/seed_{seed}/{name}"
            protected[path]=sha256(root/path)
    verify_protected(root,protected)
    protocol=ctx.out/"00_protocol";protocol.mkdir(exist_ok=True)
    dump(protocol/"protected_hashes.json",protected)
    dump(protocol/"raw_development_hashes.json",raw_hashes)
    if not (protocol/"source_hashes_at_start.json").exists():dump(protocol/"source_hashes_at_start.json",source_hashes(ctx.code_root))
    dump(protocol/f"source_hashes_pid_{os.getpid()}.json",source_hashes(ctx.code_root))
    groups={s:{r["source_group"].split("|")[0] for r in split_rows if r["subset"]==s} for s in ("train","valid","test")}
    overlap=groups["train"]&groups["valid"]
    related=[dict(sample_id=r["sample_id"],subset=r["subset"],source_group=r["source_group"],biological_relationship="unknown") for r in split_rows if r["source_group"].split("|")[0] in overlap]
    annotation=rows(root/"data/raw/logs/annoation_record.csv")
    annotation_by_id={Path(r["raw_tif_name"]).stem:r for r in annotation}
    for record in related:
        r=annotation_by_id.get(record["sample_id"],{})
        for key in ("source_path","source_file_name","frame","z_axis","channel"):record[key]=r.get(key,"")
    write_csv(protocol/"source_overlap_private.csv",related)
    duplicate_pixels={}
    for data in corpus.splits.values():
        for sid,img in zip(data["sample_id"],data["x"]):
            digest=hashlib.sha256(np.asarray(img.shape,dtype=np.int64).tobytes()+img.tobytes()).hexdigest()
            duplicate_pixels.setdefault(digest,[]).append(str(sid))
    audit=dict(train_images=176,valid_images=22,train_points=4156,valid_points=372,weak_valid_points=96,
               test_arrays_loaded=False,accessed_npz_keys=corpus.accesses,raw_development_files_verified=len(raw_hashes),
               protected_artifacts=len(protected),source_containers=len(set.union(*groups.values())),
               train_valid_container_overlap=len(overlap),biological_independence="unresolved",
               exact_development_pixel_duplicate_groups=[v for v in duplicate_pixels.values() if len(v)>1],
               duplicate_screen_scope="exact float32 pixels of 198 development images; no guarantee of biological independence")
    if len(overlap)!=2 or len(raw_hashes)!=396:raise ValueError("Source/data audit counts changed")
    dump(protocol/"input_audit.json",audit)
    ctx.task("input_audit","completed",**audit)
    return audit
