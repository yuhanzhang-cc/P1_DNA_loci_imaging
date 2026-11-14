"""Full scored-point ranked AP, not AP31 or trapezoidal PR-AUC.

Predictions ranked globally by descending score, ties by image ID/index. Each
prediction greedily claims the nearest unmatched same-image truth within radius;
GT distance ties prefer the original GT index. Equal-score groups contribute one
precision/recall step. This AP matching policy is distinct from the frozen
maximum-cardinality matching used for existing thresholded F1.
"""
import numpy as np

POLICY="ranked_point_ap_greedy_step_v1"
REFERENCE="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html"


def step_ap(labels,scores,n_true):
    labels=np.asarray(labels);scores=np.asarray(scores,dtype=float)
    if labels.shape!=scores.shape or labels.ndim!=1 or not np.isfinite(scores).all():raise ValueError("Invalid ranked labels/scores")
    if np.any((labels!=0)&(labels!=1)) or not isinstance(n_true,(int,np.integer)) or n_true<0 or labels.sum()>n_true:raise ValueError("Invalid TP labels/truth denominator")
    labels=labels.astype(int)
    if n_true==0:return dict(ap=None,reason="undefined_no_ground_truth",curve=[])
    order=np.argsort(-scores,kind="stable");labels,scores=labels[order],scores[order]
    tp=fp=0;previous_recall=0.;ap=0.;curve=[]
    for i in range(len(scores)):
        tp+=int(labels[i]);fp+=int(not labels[i])
        if i+1<len(scores) and scores[i+1]==scores[i]:continue
        recall=tp/n_true;precision=tp/(tp+fp)
        ap+=(recall-previous_recall)*precision
        curve.append(dict(score=float(scores[i]),tp=tp,fp=fp,recall=recall,precision=precision))
        previous_recall=recall
    return dict(ap=float(ap),curve=curve,reason=None)


def ranked_point_ap(truth,candidates,radius=3.):
    if not np.isfinite(radius) or radius<=0:raise ValueError("Invalid radius")
    if set(truth)!=set(candidates):raise ValueError("Candidate/truth image identities differ")
    if any(not isinstance(sid,str) for sid in truth):raise ValueError("Image IDs must be strings")
    gt={};used={};ranked=[]
    for sid in sorted(truth):
        coordinates=np.asarray(truth[sid],float).reshape(-1,2)
        points,scores=candidates[sid];points=np.asarray(points,float).reshape(-1,2);scores=np.asarray(scores,float)
        if scores.shape!=(len(points),) or not np.isfinite(coordinates).all() or not np.isfinite(points).all() or not np.isfinite(scores).all():raise ValueError("Nonfinite or malformed point data")
        gt[sid]=coordinates;used[sid]=np.zeros(len(coordinates),bool)
        for index,(point,score) in enumerate(zip(points,scores)):ranked.append((float(score),str(sid),index,point))
    ranked.sort(key=lambda r:(-r[0],r[1],r[2]))
    records=[]
    for score,sid,index,point in ranked:
        distance=np.linalg.norm(gt[sid]-point,axis=1)
        eligible=np.flatnonzero((distance<=radius)&~used[sid])
        selected=int(eligible[np.argmin(distance[eligible])]) if len(eligible) else None
        if selected is not None:used[sid][selected]=True
        records.append(dict(sample_id=sid,pred_index=index,score=score,is_tp=int(selected is not None),true_index=selected))
    result=step_ap([r["is_tp"] for r in records],[r["score"] for r in records],sum(map(len,gt.values())))
    result.update(policy=POLICY,ranked_records=records,n_true=sum(map(len,gt.values())),n_candidates=len(records),
                  matching="confidence_ordered_greedy_nearest_unmatched",threshold_floor=None,interpolation="none")
    return result
