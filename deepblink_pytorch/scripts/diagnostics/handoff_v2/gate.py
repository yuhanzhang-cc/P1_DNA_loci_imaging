"""Predeclared development-screen gate; near-truth FP is a geometry proxy."""
from pathlib import Path
import math
from .common import read,table


def trial_points(folder):
    folder=Path(folder);matches=table(folder/"primary/matches.csv");predictions=table(folder/"primary/prediction_points.csv")
    points={};matched={};errors={}
    for r in predictions:points.setdefault(r["sample_id"],[]).append((float(r["y"]),float(r["x"])))
    for r in matches:
        key=(r["sample_id"],int(r["true_index"]))
        if key in errors:raise ValueError("Repeated matched truth")
        errors[key]=float(r["distance_px"]);matched.setdefault(r["sample_id"],set()).add(int(r["pred_index"]))
    return points,matched,errors


def near_truth_fp(points,matched,truth):
    n=0
    for sid,coords in points.items():
        if sid not in truth:raise ValueError("Unknown sample ID")
        for i,(y,x) in enumerate(coords):
            if i in matched.get(sid,set()):continue
            n+=int(any(math.hypot(y-ty,x-tx)<=3. for ty,tx in truth[sid]))
    return n


def evaluate_gate(s0,p0,truth,policy):
    a=read(Path(s0)/"primary/metrics.json");b=read(Path(p0)/"primary/metrics.json")
    ap,am,ae=trial_points(s0);bp,bm,be=trial_points(p0)
    common=sorted(set(ae)&set(be));improvement=sum(ae[k]-be[k] for k in common)/len(common) if common else None
    f1_delta=b["f1"]-a["f1"];near_a=near_truth_fp(ap,am,truth);near_b=near_truth_fp(bp,bm,truth)
    passed=len(common)>=policy["minimum_common_tp"] and improvement is not None and improvement>=policy["min_common_error_improvement_px"] and f1_delta>=policy["min_f1_delta"] and near_b-near_a<=policy["max_near_truth_fp_delta"]
    return dict(passed=passed,common_tp=len(common),common_tp_error_improvement_px=improvement,
        f1_delta=f1_delta,s0_near_truth_fp=near_a,p0_near_truth_fp=near_b,near_truth_fp_delta=near_b-near_a,
        policy=policy,test_evaluated=False,development_screen_only=True,
        proxy_warning="Near-truth unmatched predictions are geometric proxies, not human-confirmed biological duplicate/background classes")
