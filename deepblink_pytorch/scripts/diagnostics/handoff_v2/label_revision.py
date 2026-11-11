"""Versioned annotation proposals and locked-model diagnostic reevaluation.

Never overwrites the parent, approves new gold labels, refits a threshold or
silently makes the existing formal training protocol accept a changed dataset.
"""
from pathlib import Path
import numpy as np
from .common import ROOT,OLD,read,table,sha,write,fresh,csv_write


def propose(parent,expected_sha,reviewed_csvs,output,reviewer,reason,synthetic=False):
    if not reviewer.strip() or not reason.strip():raise ValueError("Reviewer and correction reason required")
    if sha(parent)!=expected_sha:raise ValueError("Parent development archive differs")
    allowed={f"{k}_{s}" for s in ("train","valid") for k in ("x","y","sample_id","original_shape")}
    with np.load(parent,allow_pickle=True) as z:
        if set(z.files)!=allowed:raise ValueError("Unexpected or test arrays in parent")
        arrays={k:z[k].copy() for k in z.files}
    for split in ("train","valid"):
        values=[np.asarray(y,np.float32).copy() for y in arrays[f"y_{split}"]]
        packed=np.empty(len(values),dtype=object)
        for i,value in enumerate(values):packed[i]=value
        arrays[f"y_{split}"]=packed
    index={str(sid):(split,i) for split in ("train","valid") for i,sid in enumerate(arrays[f"sample_id_{split}"])}
    changes=[];seen=set();inputs={}
    for filename in reviewed_csvs:
        rows=table(filename)
        if not rows or len({r["sample_id"] for r in rows})!=1:raise ValueError("One complete image point list per revision CSV")
        sid=rows[0]["sample_id"]
        if sid not in index or sid in seen:raise ValueError("Unknown or repeated revision image")
        seen.add(sid);split,i=index[sid];shape=arrays[f"original_shape_{split}"][i]
        new=np.asarray([[float(r["y"]),float(r["x"])] for r in rows],np.float32).reshape(-1,2)
        if not np.isfinite(new).all() or np.any(new<0) or np.any(new>=shape):raise ValueError("Reviewed coordinates outside original FOV")
        old=np.asarray(arrays[f"y_{split}"][i]);arrays[f"y_{split}"][i]=new
        changes.append(dict(sample_id=sid,split=split,old_points=len(old),new_points=len(new),
            changed=not np.array_equal(old,new),same_pixel_collisions=len(new)-len(np.unique(np.floor(new),axis=0))))
        inputs[Path(filename).name]=sha(filename)
    if not changes:raise ValueError("No revision files supplied")
    output=fresh(output);np.savez_compressed(output/"development_revision.npz",**arrays)
    with np.load(output/"development_revision.npz",allow_pickle=True) as check:
        for split in ("train","valid"):
            assert np.array_equal(check[f"sample_id_{split}"],arrays[f"sample_id_{split}"])
            assert all(np.array_equal(a,b) for a,b in zip(check[f"x_{split}"],arrays[f"x_{split}"]))
    csv_write(output/"label_changes.csv",changes,list(changes[0]))
    marker=dict(schema="p1_label_revision_proposal_v1",parent_development_sha256=expected_sha,
        development_file="development_revision.npz",development_sha256=sha(output/"development_revision.npz"),
        reviewed_csv_sha256=inputs,annotation_reviewer=reviewer,reason=reason,synthetic_fixture=synthetic,
        raw_images_changed=False,splits_changed=False,test_arrays_present=False,approved_for_training=False,
        current_training_protocol_compatible=False,requires_dataset_protocol_review=True,
        train_points=sum(len(y) for y in arrays["y_train"]),valid_points=sum(len(y) for y in arrays["y_valid"]),
        same_pixel_collisions=sum(r["same_pixel_collisions"] for r in changes),
        limitation="Selected development-error curation is not an unbiased or independent new gold standard")
    write(output/"revision_manifest.json",marker);return marker


def reevaluate(data_root,revision,output,synthetic=False):
    from diagnostics.mechanism_v1.runner import corpus
    from diagnostics.mechanism_v1.evaluate import evaluate
    from diagnostics.scaffolds.weak_policy import local_zscore,classify
    root=Path(data_root);revision=Path(revision);manifest=read(revision/"revision_manifest.json")
    if manifest["schema"]!="p1_label_revision_proposal_v1" or manifest["synthetic_fixture"] is not synthetic:raise ValueError("Revision provenance differs")
    data_path=revision/manifest["development_file"]
    if sha(data_path)!=manifest["development_sha256"]:raise ValueError("Revision archive differs")
    original=corpus(root)
    if sha(original.paths["development"])!=manifest["parent_development_sha256"]:raise ValueError("Revision belongs to another parent")
    with np.load(data_path,allow_pickle=True) as z:
        if set(z.files)!={f"{k}_{s}" for s in ("train","valid") for k in ("x","y","sample_id","original_shape")}:raise ValueError("Unexpected arrays")
        data={k:z[k+"_valid"] for k in ("x","y","sample_id","original_shape")}
    old=original.splits["valid"]
    if not np.array_equal(data["sample_id"],old["sample_id"]) or not np.array_equal(data["original_shape"],old["original_shape"]) or not all(np.array_equal(a,b) for a,b in zip(data["x"],old["x"])):
        raise ValueError("Reevaluation allows label changes only, not images/splits")
    policy=read(root/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/configs/weak_policy.json")
    labels=[classify([local_zscore(x,y,z,policy) for y,z in truth],policy) for x,truth in zip(data["x"],data["y"])]
    data["weak"]=[l=="weak" for l in labels];data["weak_unclassifiable"]=sum(int((l=="unclassifiable").sum()) for l in labels)
    output=fresh(output);rows=[]
    for lock in read(root/OLD/"configs/validation_model_locks.json"):
        folder=output/lock["method"]/f"seed_{lock['seed']}"
        result=evaluate(lock,root,data,folder,role="validation",manifest_sha256=sha(revision/"revision_manifest.json"),synthetic=synthetic)
        result.update(dataset_role="revised_validation_proposal",approved_gold_standard=False,test_evaluated=False,original_threshold_retained=True)
        # All raw/recomputed results go to this fresh diagnostic directory only.
        from research_v4.common import dump
        dump(folder/"metrics.json",result)
        rows.append(dict(method=lock["method"],seed=lock["seed"],f1=result["f1"],ranked_ap=result["ranked_ap"],
            weak_recall=result["weak_recall"],test_evaluated=False,approved_gold_standard=False))
    csv_write(output/"diagnostic_recheck.csv",rows,list(rows[0]))
    write(output/"provenance.json",dict(revision_manifest_sha256=sha(revision/"revision_manifest.json"),
        new_training=False,threshold_refitted=False,original_results_modified=False,independent_test_evaluated=False,
        approved_for_training=False,requires_dataset_protocol_review=True,synthetic_fixture=synthetic))
    return dict(path=str(output),models=len(rows),diagnostic_only=True,approved_gold_standard=False)
