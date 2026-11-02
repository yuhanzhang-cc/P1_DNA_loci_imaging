"""Frozen-threshold evaluation on explicit new-data manifests, no calibration."""
from pathlib import Path
import csv
from datetime import date
import hashlib
import time
import numpy as np
import tifffile
import torch

from .common import rows,read_json,dump,write_csv,sha256,canonical_hash
from .evaluation import export_predictions


def pixel_hash(image):
    return hashlib.sha256(np.asarray(image.shape,dtype=np.int64).tobytes()+np.asarray(image,dtype=np.float32).tobytes()).hexdigest()


def exclusion_manifest(corpus,split_rows):
    return dict(pixel_hashes=[pixel_hash(x) for d in corpus.splits.values() for x in d["x"]],
                sample_ids=[str(sid) for d in corpus.splits.values() for sid in d["sample_id"]],
                source_containers=sorted({r["source_group"].split("|")[0].replace("\\","/").casefold() for r in split_rows}),
                scope="198 development pixel arrays plus 220 source-container metadata records; no test arrays read")


def load_manifest(path,exclusions,*,engineering=False):
    path=Path(path).resolve();table=rows(path)
    if not table:raise ValueError("External manifest contains no observations; template is not data")
    ids=[];images=[];truth=[];shapes=[];checks=[];seen=set()
    forbidden_pixels=set(exclusions["pixel_hashes"]);forbidden_ids=set(exclusions["sample_ids"])
    forbidden_containers=set(exclusions["source_containers"])
    for row in table:
        sid=row["sample_id"]
        if not sid or sid in ids or sid in forbidden_ids:raise ValueError("Missing, duplicate or development sample ID")
        if row["coordinate_unit"]!="pixel":raise ValueError("External labels must explicitly be pixel y,x; do not guess calibration")
        expected_role="synthetic_engineering" if engineering else "external_new"
        if row["dataset_role"]!=expected_role:raise ValueError("External dataset role was not declared correctly")
        for key in ("biological_group_id","acquisition_id","source_container","collection_date","annotation_version"):
            if row[key].strip().casefold() in ("","unknown","na","none"):raise ValueError("Missing scientific metadata: "+key)
        date.fromisoformat(row["collection_date"])
        if row["source_container"].replace("\\","/").casefold() in forbidden_containers:raise ValueError("Known source container overlaps current study")
        image_path=(path.parent/row["image_file"]).resolve();label_path=(path.parent/row["label_file"]).resolve()
        if not image_path.is_relative_to(path.parent) or not label_path.is_relative_to(path.parent):raise ValueError("Manifest paths must stay inside dataset folder")
        if sha256(image_path)!=row["image_sha256"] or sha256(label_path)!=row["label_sha256"]:raise ValueError("External file hash mismatch")
        x=np.asarray(tifffile.imread(image_path),dtype=np.float32)
        if x.ndim!=2 or not np.isfinite(x).all() or list(x.shape)!=[int(row["height"]),int(row["width"])]:raise ValueError("Image shape/channel/finite check failed")
        if max(x.shape)>512:raise ValueError("Larger FOV requires a separately locked tiling policy; no automatic resize")
        with label_path.open(encoding="utf-8-sig",newline="") as stream:
            reader=csv.DictReader(stream)
            if reader.fieldnames!=["y","x"]:raise ValueError("Label CSV headers must be exactly y,x")
            labels=list(reader)
        y=np.asarray([[float(r["y"]),float(r["x"])] for r in labels],dtype=np.float32).reshape(-1,2)
        if not np.isfinite(y).all() or np.any(y<0) or np.any(y>=x.shape):raise ValueError("External point coordinates invalid")
        digest=pixel_hash(x)
        if digest in forbidden_pixels or digest in seen:raise ValueError("Exact duplicate of development or another external image")
        seen.add(digest)
        calibration=row.get("um_per_pixel","").strip()
        if calibration and (not np.isfinite(float(calibration)) or float(calibration)<=0):raise ValueError("Invalid supplied calibration")
        ids.append(sid);images.append(x);truth.append(y);shapes.append(x.shape)
        checks.append(dict(sample_id=sid,biological_group_id=row["biological_group_id"],acquisition_id=row["acquisition_id"],
                           source_container=row["source_container"],pixels_sha256=digest,um_per_pixel=calibration))
    return dict(x=images,y=truth,sample_id=np.asarray(ids),original_shape=np.asarray(shapes)),checks


def locked_candidates(lock,model_root,data,device):
    from deepblink_torch.cli import predict
    from deepblink_torch.spotiflow_bridge import spotiflow_candidates
    from .engine import infer_candidates
    checkpoint=(Path(model_root)/lock["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(Path(model_root).resolve()):raise ValueError("Model path escapes model root")
    weight=checkpoint if lock["kind"]=="own" else checkpoint/"best.pt"
    if sha256(weight)!=lock["weight_sha256"]:raise ValueError("Locked model weight changed")
    if lock["kind"]=="own":
        model,_=predict._load_model(checkpoint,torch.device(device))
        candidates=infer_candidates(model,data,device,canvas=512)
    elif lock["kind"]=="spotiflow":
        from spotiflow.model import Spotiflow
        for name,digest in lock["configuration_hashes"].items():
            if sha256(checkpoint/name)!=digest:raise ValueError("Native model configuration/threshold file changed")
        model=Spotiflow.from_folder(checkpoint,map_location=device,inference_mode=True)
        candidates=[spotiflow_candidates(model,image,device) for image in data["x"]]
    else:raise ValueError("Unknown locked model adapter")
    del model
    return candidates


def evaluate(manifest_path,locks,model_root,exclusions,output,*,device="cpu",engineering=False,contrasts=()):
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("Requested CUDA is unavailable")
    torch.set_num_threads(4)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data,metadata=load_manifest(manifest_path,exclusions,engineering=engineering)
    protocol=dict(manifest_sha256=sha256(manifest_path),method_locks=locks,device=device,engineering_only=engineering,
                  threshold_selection_allowed=False,metadata_checks_not_proof_of_biological_independence=True,planned_contrasts=contrasts)
    if (output/"protocol.json").exists():raise ValueError("External evaluation output already exists; preserve the first readout")
    dump(output/"protocol.json",protocol)
    records=[];all_images=[]
    for lock in locks:
        if lock["method"]!=Path(lock["method"]).name:raise ValueError("Unsafe method name")
        start=time.perf_counter()
        candidates=locked_candidates(lock,model_root,data,device)
        threshold=float(lock["threshold"])
        if not 0<=threshold<=1:raise ValueError("Locked probability threshold invalid")
        policy=dict(kind="all" if threshold==0 else "none" if threshold==1 else "finite",value=None if threshold in (0,1) else threshold)
        folder=output/lock["method"]/f"seed_{lock['seed']}"
        metrics=export_predictions(folder,candidates,data,policy)
        metrics.pop("development_validation_only")
        metrics.update(external_manifest=True,engineering_only=engineering,test_threshold_tuning=False)
        dump(folder/"metrics.json",metrics)
        records.append(dict(method=lock["method"],seed=lock["seed"],threshold=threshold,
                             **{k:metrics[k] for k in ("tp","fp","fn","precision","recall","f1","macro_f1","count_mae","mean_error_px")},
                             end_to_end_seconds=time.perf_counter()-start))
        by_id={r["sample_id"]:r for r in metadata}
        for row in rows(folder/"per_image_metrics.csv"):
            all_images.append(dict(method=lock["method"],seed=lock["seed"],biological_group_id=by_id[row["sample_id"]]["biological_group_id"],**row))
    write_csv(output/"per_seed_summary.csv",records)
    write_csv(output/"all_per_image_metrics.csv",all_images)
    write_csv(output/"manifest_audit.csv",metadata)
    dump(output/"checks.json",dict(passed=True,images=len(data["x"]),methods=len(locks),thresholds_frozen=True,
                                  engineering_only=engineering,biological_independence="requires acquisition review"))
    return records
