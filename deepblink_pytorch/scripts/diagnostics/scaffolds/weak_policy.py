"""Frozen weak-point definition; no quantile refitting on validation/test."""
from pathlib import Path
import argparse,json
import numpy as np
from scaffold_common import read,table


def validate(policy):
    if policy["comparison"]!="strict_less_than" or policy["std_ddof"]!=0 or policy["radius_px"]!=10:
        raise ValueError("Unsupported weak-point feature contract")
    if not np.isfinite(policy["threshold"]) or policy["std_epsilon"]<=0:raise ValueError("Invalid fixed threshold/epsilon")
    if policy["policy_id"]=="legacy_train_q25_local_zscore_v1" and policy["threshold"]!=2.4566830772038957:
        raise ValueError("Do not drift the frozen policy in place; a changed definition needs a new policy ID")
    return policy


def local_zscore(image,y,x,policy):
    validate(policy);image=np.asarray(image,np.float32)
    if image.ndim!=2 or not np.isfinite(image).all() or not np.isfinite([y,x]).all():raise ValueError("Invalid original image/coordinate")
    h,w=image.shape
    if not (0<=y<h and 0<=x<w):raise ValueError("Truth outside original FOV")
    cy=int(np.clip(round(float(y)),0,h-1));cx=int(np.clip(round(float(x)),0,w-1));r=policy["radius_px"]
    patch=image[max(0,cy-r):min(h,cy+r+1),max(0,cx-r):min(w,cx+r+1)]
    std=float(patch.std(ddof=0))
    return (float(image[cy,cx])-float(patch.mean()))/std if std>policy["std_epsilon"] else float("nan")


def classify(values,policy):
    validate(policy);values=np.asarray(values,float)
    return np.where(~np.isfinite(values),"unclassifiable",np.where(values<policy["threshold"],"weak","non_weak"))


def weak_recall(values,matched_true_indices,policy):
    labels=classify(values,policy);indices=np.asarray(matched_true_indices,dtype=int)
    if len(indices)!=len(set(map(int,indices))) or np.any(indices<0) or np.any(indices>=len(labels)):
        raise ValueError("Invalid/duplicate matched truth indices")
    n=int((labels=="weak").sum());tp=int((labels[indices]=="weak").sum())
    return dict(weak_n=n,weak_tp=tp,weak_recall=tp/n if n else None,unclassifiable_n=int((labels=="unclassifiable").sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--features",type=Path);p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args();policy=validate(read(args.config));result=dict(passed=True,dry_run=True,threshold=policy["threshold"],test_arrays_read=False,refitted=False)
    if args.features:
        data=table(args.features);labels=classify([float(r["local_zscore"]) for r in data],policy)
        recorded=np.asarray([r["weak"]=="True" for r in data])
        if not np.array_equal(labels=="weak",recorded):raise ValueError("Frozen weak labels drifted")
        result.update(truth_points=len(data),weak_points=int((labels=="weak").sum()),recorded_labels_exact=True)
    print(json.dumps(result))


if __name__=="__main__":main()
