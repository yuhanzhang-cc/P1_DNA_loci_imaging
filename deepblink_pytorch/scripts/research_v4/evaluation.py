"""Common point metrics for probabilities and unbounded classical responses."""
from pathlib import Path
import numpy as np
from deepblink_torch.evaluation import match_coordinates, metrics_from_match
from deepblink_torch.round2_protocol import THRESHOLDS_V3, sweep_candidate_counts
from .common import dump, write_csv


def policy_numeric(policy):
    if policy["kind"]=="all":return 0.
    if policy["kind"]=="none":return float("inf")
    return float(policy["value"])


def policies_probability():
    return [dict(kind="all",value=None) if t==0 else dict(kind="none",value=None) if t==1 else dict(kind="finite",value=float(t)) for t in THRESHOLDS_V3]


def keep_scores(scores,policy):
    if policy["kind"]=="all":return np.ones(len(scores),dtype=bool)
    if policy["kind"]=="none":return np.zeros(len(scores),dtype=bool)
    return scores>float(policy["value"])


def save_candidates(path,candidates,data):
    for (coords,scores),shape in zip(candidates,data["original_shape"]):
        if coords.shape!=(len(scores),2) or not np.isfinite(coords).all() or not np.isfinite(scores).all():raise ValueError("Malformed candidates")
        if np.any(coords<0) or np.any(coords>=shape):raise ValueError("Candidates out of bounds")
    np.savez_compressed(path,coordinates=np.concatenate([c[0] for c in candidates]),scores=np.concatenate([c[1] for c in candidates]),
        offsets=np.r_[0,np.cumsum([len(c[0]) for c in candidates])],sample_id=data["sample_id"],original_shape=data["original_shape"])


def load_candidates(path,data):
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z["sample_id"],data["sample_id"]) or not np.array_equal(z["original_shape"],data["original_shape"]):raise ValueError("Candidate data identity differs")
        offsets=z["offsets"];coords=z["coordinates"];scores=z["scores"]
        if len(offsets)!=len(data["sample_id"])+1 or offsets[0]!=0 or offsets[-1]!=len(coords) or len(coords)!=len(scores) or np.any(np.diff(offsets)<0):raise ValueError("Invalid candidate offsets")
        result=[(coords[a:b].copy(),scores[a:b].copy()) for a,b in zip(offsets[:-1],offsets[1:])]
    return result


def sweep(candidates,data,policies):
    thresholds=[policy_numeric(p) for p in policies]
    counts=[sweep_candidate_counts(coords,scores,truth,thresholds=thresholds) for (coords,scores),truth in zip(candidates,data["y"])]
    result=[]
    for i,policy in enumerate(policies):
        tp,fp,fn=[int(sum(row[i][j] for row in counts)) for j in range(3)]
        ntrue=tp+fn
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
        row=dict(threshold_index=i,threshold_kind=policy["kind"],threshold=policy["value"],tp=tp,fp=fp,fn=fn,
                 precision=tp/(tp+fp) if tp+fp else float(ntrue==0),recall=tp/ntrue if ntrue else 1.,f1=f1,
                 macro_f1=float(np.mean([2*r[i][0]/(2*r[i][0]+r[i][1]+r[i][2]) if sum(r[i][:3]) else 1. for r in counts])),
                 count_mae=float(np.mean([abs(r[i][1]-r[i][2]) for r in counts])))
        if "weak" in data:
            weak_tp=sum(int(w[c[i][3]].sum()) for w,c in zip(data["weak"],counts))
            weak_n=sum(int(w.sum()) for w in data["weak"])
            row.update(weak_tp=weak_tp,weak_n=weak_n,weak_recall=weak_tp/weak_n if weak_n else None)
        result.append(row)
    return result


def selection_key(row):
    return row["f1"],row["precision"],-row.get("epoch",0),-row.get("config_id",0),row["threshold_index"]


def row_policy(row):return dict(kind=row["threshold_kind"],value=row["threshold"])


def export_predictions(folder,candidates,data,policy):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    images,points,matches,distances=[],[],[],[]
    weak_tp=0
    for index,(sid,shape,truth,(coords,scores)) in enumerate(zip(data["sample_id"],data["original_shape"],data["y"],candidates)):
        keep=keep_scores(scores,policy);coords,scores=coords[keep],scores[keep]
        match=match_coordinates(coords,truth,3.)
        metrics=metrics_from_match(match)
        metrics={k:float(v) if np.isfinite(v) else None for k,v in metrics.items()}
        images.append(dict(sample_id=str(sid),height=int(shape[0]),width=int(shape[1]),n_true=len(truth),n_pred=len(coords),**metrics))
        for (y,x),p in zip(coords,scores):points.append(dict(sample_id=str(sid),y=float(y),x=float(x),p=float(p)))
        for pi,ti,d in zip(match.matched_pred_indices,match.matched_true_indices,match.matched_distances):
            matches.append(dict(sample_id=str(sid),pred_index=int(pi),true_index=int(ti),distance_px=float(d)))
        distances.extend(map(float,match.matched_distances))
        if "weak" in data:weak_tp+=int(data["weak"][index][match.matched_true_indices].sum())
    tp,fp,fn=[int(sum(r[k] for r in images)) for k in ("tp","fp","fn")]
    result=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else float(tp+fn==0),recall=tp/(tp+fn) if tp+fn else 1.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,macro_f1=float(np.mean([r["f1"] for r in images])),
                count_mae=float(np.mean([abs(r["n_pred"]-r["n_true"]) for r in images])),
                mean_error_px=float(np.mean(distances)) if distances else None,
                median_error_px=float(np.median(distances)) if distances else None,
                rmse_error_px=float(np.sqrt(np.mean(np.square(distances)))) if distances else None,
                development_validation_only=True,test_evaluated=False)
    if "weak" in data:
        weak_n=sum(int(w.sum()) for w in data["weak"])
        result.update(weak_tp=weak_tp,weak_n=weak_n,weak_recall=weak_tp/weak_n if weak_n else None)
    write_csv(folder/"per_image_metrics.csv",images)
    write_csv(folder/"prediction_points.csv",points,["sample_id","y","x","p"])
    write_csv(folder/"matches.csv",matches,["sample_id","pred_index","true_index","distance_px"])
    dump(folder/"metrics.json",result)
    return result
