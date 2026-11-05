"""Frozen-checkpoint inference, full ranked AP and locked-threshold F1.

CLI defaults to metadata-only verification. Validation replay is explicitly
development-only. Independent-test execution needs reviewed provenance, disjoint
IDs/content versus the frozen development set, and locked model selection.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
import tifffile
from .contracts import ROOT,read,sha,canonical,write_new
from .runner import corpus,load_model,infer
from research_v4.common import dump,write_csv
from research_v4.engine import configure_device,infer_candidates
from research_v4.evaluation import export_predictions
from deepblink_torch.cli import predict
from diagnostics.scaffolds.formal_test_eval import verify_locks
from diagnostics.scaffolds.ranked_ap import ranked_point_ap
from diagnostics.scaffolds.weak_policy import local_zscore,classify


def fingerprint(image):
    image=np.asarray(image,dtype="<f4",order="C")
    return hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()


def safe_path(root,name):
    path=(Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()):raise ValueError("Manifest path escapes dataset root")
    return path


def load_manifest(path,data_root,development):
    spec=read(path)
    if spec.get("dataset_role")!="independent_test" or spec.get("independence_reviewed") is not True:
        raise ValueError("Independent data review is incomplete")
    if not spec.get("reviewer") or not spec.get("reviewed_at") or not spec.get("independence_notes"):
        raise ValueError("Missing reviewed independence provenance")
    evidence=safe_path(Path(path).parent,spec["independence_evidence_file"])
    if sha(evidence)!=spec["independence_evidence_sha256"]:raise ValueError("Independence evidence differs")
    rows=spec["images"]
    if not rows or len({r["sample_id"] for r in rows})!=len(rows):raise ValueError("Empty/duplicate image manifest")
    ids={str(s) for d in development.splits.values() for s in d["sample_id"]}
    # Include previously observed internal test IDs, without reading its images.
    split=Path(data_root)/"outputs/261024_clean220_random/frozen_split.csv"
    if not split.is_file():raise ValueError("Historical split inventory required")
    import csv
    with split.open(encoding="utf-8-sig",newline="") as f:ids|={r["sample_id"] for r in csv.DictReader(f)}
    seen={fingerprint(x) for d in development.splits.values() for x in d["x"]}
    data=dict(x=[],y=[],sample_id=[],original_shape=[]);new_hashes=set()
    for row in rows:
        if row["sample_id"] in ids:raise ValueError("Historical image ID cannot be independent")
        image_path=safe_path(Path(path).parent,row["image"]);truth_path=safe_path(Path(path).parent,row["truth"])
        if sha(image_path)!=row["image_sha256"] or sha(truth_path)!=row["truth_sha256"]:raise ValueError("Input file hash differs")
        image=np.asarray(tifffile.imread(image_path),np.float32)
        if image.ndim!=2 or image.shape not in ((256,256),(512,512)) or not np.isfinite(image).all():raise ValueError("Unsupported raw image")
        content=fingerprint(image)
        if content in seen or content in new_hashes:raise ValueError("Duplicate/development image content")
        new_hashes.add(content)
        with truth_path.open(encoding="utf-8-sig",newline="") as f:truth=np.asarray([[float(r["y"]),float(r["x"])] for r in csv.DictReader(f)],np.float32).reshape(-1,2)
        if not np.isfinite(truth).all() or np.any(truth<0) or np.any(truth>=image.shape):raise ValueError("Invalid truth coordinates")
        for key,value in dict(x=image,y=truth,sample_id=row["sample_id"],original_shape=list(image.shape)).items():data[key].append(value)
    data["sample_id"]=np.asarray(data["sample_id"],str);data["original_shape"]=np.asarray(data["original_shape"])
    return data,spec


def evaluate(lock,root,data,folder,*,device="cpu",role="validation",manifest_sha256=None,synthetic=False,independence_verified=False):
    if role not in ("validation","independent_test"):raise ValueError("Unknown dataset role")
    if role=="independent_test" and not independence_verified:raise ValueError("Independent manifest/provenance must be verified first")
    verify_locks([lock],Path(root))
    checkpoint=Path(root)/lock["checkpoint"]
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if saved.get("model_name")=="signed_heatmap_isolated_v1":
        model,_=load_model(checkpoint,device);candidates=infer(model,data,device)
    else:
        model=predict._build_model_for_checkpoint(saved["model_name"],cell_size=saved["cell_size"],device=torch.device(device))
        model.load_state_dict(saved["model_state_dict"]);candidates=infer_candidates(model,data,device)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    policy=dict(kind="all" if lock["threshold"]==0 else "none" if lock["threshold"]==1 else "finite",value=lock["threshold"])
    metrics=export_predictions(folder,candidates,data,policy)
    truth={str(s):t for s,t in zip(data["sample_id"],data["y"])}
    scored={str(s):c for s,c in zip(data["sample_id"],candidates)}
    ranked=ranked_point_ap(truth,scored,radius=3.)
    write_csv(folder/"ranked_predictions.csv",ranked.pop("ranked_records"),["sample_id","pred_index","score","is_tp","true_index"])
    write_csv(folder/"ranked_pr_curve.csv",ranked.pop("curve"),["score","tp","fp","recall","precision"])
    dump(folder/"ranked_ap.json",ranked)
    metrics.update(ranked_ap=ranked["ap"],ap_policy=ranked["policy"],all_prethreshold_candidates=True,score_floor=None,
        development_validation_only=role=="validation",test_evaluated=role=="independent_test" and not synthetic,
        engineering_only=synthetic,threshold_selection_on_test=False)
    if "weak_unclassifiable" in data:metrics["weak_unclassifiable_n"]=int(data["weak_unclassifiable"])
    dump(folder/"metrics.json",metrics)
    dump(folder/"provenance.json",dict(lock=lock,manifest_sha256=manifest_sha256,dataset_role=role,
        inference_device=device,torch=str(torch.__version__),candidate_count=sum(len(c[0]) for c in candidates),
        model_selection_source="frozen_development_validation",synthetic_fixture=synthetic))
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,default=ROOT)
    p.add_argument("--locks",type=Path,required=True);p.add_argument("--method",required=True);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--dataset-role",choices=["validation","independent_test"],default="validation")
    p.add_argument("--manifest",type=Path);p.add_argument("--output",type=Path);p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    m=p.add_mutually_exclusive_group(required=True);m.add_argument("--dry-run",action="store_true");m.add_argument("--evaluate",action="store_true")
    args=p.parse_args();locks=[r for r in read(args.locks) if r["method"]==args.method and r["seed"]==args.seed]
    if len(locks)!=1:raise ValueError("Select exactly one validation-locked model")
    verify_locks(locks,args.data_root.resolve())
    if args.dry_run:
        print(json.dumps(dict(passed=True,dry_run=True,locks_verified=1,dataset_role=args.dataset_role,images_read=0,test_evaluated=False,ranked_ap_implemented=True)));return
    if args.output is None:raise ValueError("Fresh output directory required")
    configure_device(args.device);development=corpus(args.data_root)
    if args.dataset_role=="validation":data=development.splits["valid"];manifest=sha(development.paths["development"])
    else:
        if args.manifest is None:raise ValueError("Reviewed independent manifest required")
        data,_=load_manifest(args.manifest,args.data_root,development);manifest=sha(args.manifest)
        policy=read(ROOT/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/configs/weak_policy.json")
        labels=[classify([local_zscore(x,y,z,policy) for y,z in points],policy) for x,points in zip(data["x"],data["y"])]
        data["weak"]=[l=="weak" for l in labels];data["weak_unclassifiable"]=sum(int((l=="unclassifiable").sum()) for l in labels)
    print(json.dumps(evaluate(locks[0],args.data_root,data,args.output,device=args.device,role=args.dataset_role,manifest_sha256=manifest,
        independence_verified=args.dataset_role=="independent_test"),indent=2))


if __name__=="__main__":main()
