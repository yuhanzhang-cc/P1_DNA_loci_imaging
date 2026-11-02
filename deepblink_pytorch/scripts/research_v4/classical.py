"""Bounded LoG/DoG scale-space references, with train-derived response grids."""
from __future__ import annotations
import itertools
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_laplace,maximum_filter,label
from scipy.spatial import cKDTree
from .common import dump,read_json,canonical_hash,sha256,write_csv
from .evaluation import sweep,selection_key,row_policy,export_predictions,save_candidates


def configurations(spec,method):
    cfg=spec["classical"][method.lower()]
    keys=("min_sigma","max_sigma","num_scales","min_distance_px") if method=="LoG" else ("min_sigma","max_sigma","sigma_ratio","min_distance_px")
    return [dict(config_id=i,**dict(zip(keys,values))) for i,values in enumerate(itertools.product(*(cfg[k] for k in keys)))]


def normalize(image):
    lo,hi=np.percentile(image,[1,99.8])
    if hi<=lo:return np.zeros_like(image,dtype=np.float32)
    return np.asarray(np.clip((image-lo)/(hi-lo),0,1),dtype=np.float32)


def peaks(response,distance):
    maxima=(response>0)&(response==maximum_filter(response,size=3,mode="constant",cval=-np.inf))
    # One lexicographically first pixel per connected local-max plateau.
    labels,n=label(maxima,np.ones((3,3),dtype=np.uint8))
    if not n:return np.empty((0,2),np.float32),np.empty(0,np.float32)
    coords=np.argwhere(maxima)
    component=labels[coords[:,0],coords[:,1]]
    _,first=np.unique(component,return_index=True)
    coords=coords[np.sort(first)]
    scores=response[coords[:,0],coords[:,1]]
    order=np.lexsort((coords[:,1],coords[:,0],-scores))
    tree=cKDTree(coords);suppressed=np.zeros(len(coords),bool);selected=[]
    for i in order:
        if suppressed[i]:continue
        selected.append(i)
        suppressed[tree.query_ball_point(coords[i],r=float(distance))]=True
    selected=np.asarray(selected,dtype=int)
    return coords[selected].astype(np.float32),scores[selected].astype(np.float32)


def image_candidates(image,configs,method):
    image=normalize(image)
    cache={};result=[]
    def gaussian(sigma):
        key=("g",float(sigma))
        if key not in cache:cache[key]=gaussian_filter(image,sigma=float(sigma),mode="reflect",truncate=4.)
        return cache[key]
    for config in configs:
        if method=="LoG":
            scales=np.geomspace(config["min_sigma"],config["max_sigma"],config["num_scales"])
            maps=[]
            for sigma in scales:
                key=("l",float(sigma))
                if key not in cache:cache[key]=-float(sigma)**2*gaussian_laplace(image,sigma=float(sigma),mode="reflect",truncate=4.)
                maps.append(cache[key])
        else:
            ratio=config["sigma_ratio"];scales=[];sigma=config["min_sigma"]
            while sigma<=config["max_sigma"]*(1+1e-12):scales.append(sigma);sigma*=ratio
            maps=[(gaussian(sigma)-gaussian(sigma*ratio))/(ratio-1) for sigma in scales]
        response=np.maximum.reduce(maps)
        result.append(peaks(response,config["min_distance_px"]))
    return result


def response_policies(scores,spec):
    if len(scores):
        q=spec["classical"]["thresholds"]
        q=np.r_[np.linspace(q["quantiles_linear"]["start"],q["quantiles_linear"]["stop"],q["quantiles_linear"]["count"]),q["quantiles_tail"]]
        values=np.unique(np.quantile(scores,q)).tolist()
    else:values=[]
    return [dict(kind="all",value=None)]+[dict(kind="finite",value=float(x)) for x in values]+[dict(kind="none",value=None)]


def run(ctx):
    ctx.task("classical_baselines","running")
    corpus=ctx.corpus.splits
    out=ctx.out/"02_classical";out.mkdir(exist_ok=True)
    lock=dict(config=ctx.spec["classical"],development_sha256=ctx.spec["data"]["development_sha256"],
              algorithm_sha256=sha256(__file__),evaluation_sha256=sha256(Path(__file__).with_name("evaluation.py")),
              plateau="one lexicographically first pixel per 8-connected local maximum plateau",nms_suppression="distance <= registered radius")
    lock_path=out/"lock.json"
    if lock_path.exists() and read_json(lock_path)!=lock:raise ValueError("Classical calibration lock changed; preserve old run and use a new revision")
    dump(lock_path,lock)
    summary=[];all_rows=[]
    for method in ("LoG","DoG"):
        folder=out/method;folder.mkdir(exist_ok=True)
        bank=folder/"candidate_cache";bank.mkdir(exist_ok=True)
        configs=configurations(ctx.spec,method)
        if len(configs)!=24:raise ValueError("Classical configuration budget changed")
        dump(folder/"configurations.json",configs)
        completed=0;tick=time.monotonic()
        for subset in ("train","valid"):
            data=corpus[subset]
            for sid,image in zip(data["sample_id"],data["x"]):
                path=bank/(str(sid)+".npz")
                if not path.exists():
                    candidates=image_candidates(image,configs,method)
                    payload={f"{i}_{key}":value for i,(coords,scores) in enumerate(candidates) for key,value in (("coords",coords),("scores",scores))}
                    temporary=path.with_name(path.name+".tmp")
                    with temporary.open("wb") as stream:np.savez_compressed(stream,**payload)
                    temporary.replace(path)
                completed+=1
                if completed%10==0 or completed==198:
                    ctx.status(activity=f"{method} candidates {completed}/198",classical_candidate_images=completed,
                               classical_candidate_seconds=time.monotonic()-tick)
        write_csv(folder/"candidate_hashes.csv",[dict(sample_id=p.stem,sha256=sha256(p)) for p in sorted(bank.glob("*.npz"))])
        method_rows=[];best_candidates=None;best=None
        for config in configs:
            index=config["config_id"];sets={}
            for subset in ("train","valid"):
                candidate=[]
                for sid in corpus[subset]["sample_id"]:
                    with np.load(bank/(str(sid)+".npz"),allow_pickle=False) as z:
                        candidate.append((z[f"{index}_coords"],z[f"{index}_scores"]))
                sets[subset]=candidate
            # Only responses, never labels, define this threshold grid.
            pooled=np.concatenate([s for _,s in sets["train"]])
            policies=response_policies(pooled,ctx.spec)
            if len(policies)>31:raise ValueError("Threshold budget exceeded")
            dump(folder/f"config_{index:02d}_thresholds.json",dict(policies=policies,training_candidate_count=len(pooled),
                 training_labels_used=False,training_score_sha256=__import__("hashlib").sha256(pooled.tobytes()).hexdigest()))
            scores=[dict(method=method,config_id=index,**r) for r in sweep(sets["valid"],corpus["valid"],policies)]
            method_rows.extend(scores)
            choice=max(scores,key=selection_key)
            if best is None or selection_key(choice)>selection_key(best):best,best_candidates=choice,sets["valid"]
            write_csv(folder/"all_configuration_threshold_metrics.csv",method_rows)
            ctx.status(activity=f"{method} calibrated config {index+1}/24",classical_rows=len(method_rows))
        if len(method_rows)>744:raise ValueError("Classical evaluation budget exceeded")
        save_candidates(folder/"selected_candidates.npz",best_candidates,corpus["valid"])
        metrics=export_predictions(folder/"selected",best_candidates,corpus["valid"],row_policy(best))
        if (metrics["tp"],metrics["fp"],metrics["fn"])!=(best["tp"],best["fp"],best["fn"]):raise ValueError("Classical sweep/export mismatch")
        selected=dict(method=method,configuration=configs[best["config_id"]],primary=best,metrics=metrics,
                      configurations_completed=24,configuration_threshold_pairs=len(method_rows),training_replicates=0,
                      validation_selection_optimism=True,test_evaluated=False,elapsed_seconds=time.monotonic()-tick)
        dump(folder/"selection.json",selected)
        dump(folder/"status.json",dict(status="complete",configurations=24))
        summary.append(dict(method=method,config_id=best["config_id"],configuration_threshold_pairs=len(method_rows),
                            **{k:metrics[k] for k in ("tp","fp","fn","precision","recall","f1","weak_recall","macro_f1","count_mae","mean_error_px")}))
        all_rows.extend(method_rows)
    write_csv(out/"summary.csv",summary)
    write_csv(out/"all_threshold_metrics.csv",all_rows)
    dump(out/"checks.json",dict(passed=True,configuration_count=48,pairs=len(all_rows),pair_cap=1488,
         training_labels_used_for_thresholds=False,test_arrays_loaded=False,pseudo_training_repeats=False))
    ctx.task("classical_baselines","completed",summary=summary,pairs=len(all_rows))
    return summary
