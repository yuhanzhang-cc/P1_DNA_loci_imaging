"""Audit the six returned G0 trials and explain the registered recipe contrast.

All new strata, radius sensitivity and error decompositions are exploratory.
The original 3px micro-F1 endpoint and selected weights/thresholds remain fixed.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"deepblink_pytorch"),str(ROOT/"deepblink_pytorch/scripts")]
from research_v4.common import read_json,rows,dump,write_csv,sha256,canonical_hash
from deepblink_torch.evaluation import match_coordinates

METHODS=("G0_ref_cuda","G0_aug_cuda")
SEEDS=(42,43,44)
METRICS=("f1","precision","recall","weak_recall","bright_recall","macro_f1","count_mae","mean_error_px","rmse_error_px","tp","fp","fn")


def exact_paired(differences):
    differences=np.asarray(differences,float)
    null=np.asarray([np.mean(differences*s) for s in itertools.product((-1,1),repeat=len(differences))])
    return float(np.mean(np.abs(null)>=abs(differences.mean())-1e-12))


def holm(values):
    order=np.argsort(values);result=np.zeros(len(values));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,(len(order)-rank)*values[index]);result[index]=min(previous,1.)
    return result.tolist()


def aggregate(images):
    tp,fp,fn=[sum(float(r[k]) for r in images) for k in ("tp","fp","fn")]
    nweak=sum(int(r["weak_n"]) for r in images);nbright=sum(int(r["bright_n"]) for r in images)
    weak=sum(int(r["weak_tp"]) for r in images);bright=sum(int(r["bright_tp"]) for r in images)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else float(tp+fn==0),recall=tp/(tp+fn) if tp+fn else 1.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,macro_f1=float(np.mean([float(r["f1"]) for r in images])),
                weak_recall=weak/nweak if nweak else None,bright_recall=bright/nbright if nbright else None,
                weak_tp=weak,weak_n=nweak,bright_tp=bright,bright_n=nbright,
                count_mae=float(np.mean([abs(int(r["n_pred"])-int(r["n_true"])) for r in images])),
                mean_error_px=sum(float(r["distance_sum"]) for r in images)/tp if tp else None,
                rmse_error_px=float(np.sqrt(sum(float(r["distance_sq_sum"]) for r in images)/tp)) if tp else None)


def verify(ctx):
    from cloud_recipe_train import check_complete
    root,queue,out=ctx["data"],ctx["queue"],ctx["out"]
    check_complete(queue.parent,"recipe")
    returned=queue.parents[2]
    original=root/"outputs/next_stage_v4_recipe_20261101/04_bundles/ce2d1821ceec/p1_v4_training"
    if (returned/"bundle_manifest.json").read_bytes()!=(original/"bundle_manifest.json").read_bytes():raise ValueError("Returned bundle identity differs")
    manifest=read_json(returned/"bundle_manifest.json")
    for name,digest in manifest["files"].items():
        if sha256(original/name)!=digest:raise ValueError("Original sealed bundle changed: "+name)
    if sha256(returned/"bundle_manifest.json")!=read_json(queue/"protocol.json")["manifest_sha256"]:raise ValueError("Source lock differs")
    audit=read_json(root/"outputs/next_stage_v3_closing_controls/00_protocol/data_audit.json")
    if sha256(root/"outputs/261024_clean220_random/frozen_split.csv")!=audit["split_manifest_sha256"]:raise ValueError("Old/current data split differs")
    trial=[]
    for method in METHODS:
        for seed in SEEDS:
            p=queue/method/f"seed_{seed}";training=rows(p/"training_metrics.csv");sweep=rows(p/"all_epoch_threshold_metrics.csv")
            if [int(r["epoch"]) for r in training]!=list(range(1,101)):raise ValueError("Epoch sequence differs")
            if [int(r["optimizer_updates"]) for r in training]!=list(range(22,2201,22)):raise ValueError("Updates differ")
            if len(sweep)!=620 or any(sum(int(r["epoch"])==epoch for r in sweep)!=31 for epoch in range(5,101,5)):raise ValueError("620-pair budget differs")
            if not all(np.isfinite(float(r[k])) for r in training for k in ("train_loss","val_loss","seconds")):raise ValueError("Nonfinite training history")
            trial.append(dict(method=method,seed=seed,epochs=100,updates=2200,pairs=620,epoch_loop_seconds=sum(float(r["seconds"]) for r in training)))
    for seed in SEEDS:
        a=read_json(queue/METHODS[0]/f"seed_{seed}/initialization.json");b=read_json(queue/METHODS[1]/f"seed_{seed}/initialization.json")
        if a!=b:raise ValueError("Paired initial weights/sampler differ")
    result=dict(passed=True,formal_tasks=6,epochs=600,updates=13200,selection_pairs=3720,paired_initialization_and_order=True,
                original_bundle_hashes_verified=len(manifest["files"]),same_split_as_legacy_G0=True,source_manifest_sha256=sha256(returned/"bundle_manifest.json"),
                gpu=read_json(queue/"protocol.json")["gpu"],test_arrays_read=False,trial_budgets=trial)
    dump(out/"audit.json",result)
    return result


def truth_reference(root):
    features=rows(root/"outputs/next_stage_v3_closing_controls/02_analysis/validation_truth_features.csv")
    split=rows(root/"outputs/261024_clean220_random/frozen_split.csv")
    valid={r["sample_id"]:r for r in split if r["subset"]=="valid"}
    truth={sid:[] for sid in valid};weak={sid:[] for sid in valid};index={}
    for row in features:
        sid=row["sample_id"];i=int(row["true_index"])
        if i!=len(truth[sid]):raise ValueError("Truth ordering changed")
        truth[sid].append([float(row["y"]),float(row["x"])]);weak[sid].append(row["weak"]=="True")
        index[(sid,i)]=row
    if len(valid)!=22 or len(features)!=372 or sum(sum(v) for v in weak.values())!=96:raise ValueError("Truth reference differs")
    return valid,{k:np.asarray(v,np.float32) for k,v in truth.items()},{k:np.asarray(v,bool) for k,v in weak.items()},index


def read_trials(ctx,valid,truth,weak,methods=METHODS):
    records=[];images=[];point_sets={};radii=[];thresholds=[];curves=[];timings=[]
    for method in methods:
        for seed in SEEDS:
            folder=ctx["queue"]/method/f"seed_{seed}";selection=read_json(folder/"selection.json")
            history=rows(folder/"training_metrics.csv");sweep=rows(folder/"all_epoch_threshold_metrics.csv")
            for role in ("primary","native_secondary"):
                choice=selection[role];metrics=read_json(folder/role/"metrics.json")
                per_image=rows(folder/role/"per_image_metrics.csv");matches=rows(folder/role/"matches.csv");predictions=rows(folder/role/"prediction_points.csv")
                if {r["sample_id"] for r in per_image}!=set(valid):raise ValueError("Image identities differ")
                point_sets[(method,seed,role)]={(r["sample_id"],int(r["true_index"])):float(r["distance_px"]) for r in matches}
                current=[]
                for row in per_image:
                    sid=row["sample_id"];match=[r for r in matches if r["sample_id"]==sid]
                    pred=[r for r in predictions if r["sample_id"]==sid]
                    coords=np.asarray([[float(r["y"]),float(r["x"])] for r in pred],np.float32).reshape(-1,2)
                    shape=[int(valid[sid][k]) for k in ("height","width")]
                    if not np.isfinite(coords).all() or np.any(coords<0) or np.any(coords>=shape):raise ValueError("Prediction outside native FOV")
                    if any(float(r["p"])<float(choice["threshold"])-1e-12 for r in pred):raise ValueError("Prediction below locked threshold")
                    rematch=match_coordinates(coords,truth[sid],3.)
                    if (rematch.tp,rematch.fp,rematch.fn)!=tuple(int(float(row[k])) for k in ("tp","fp","fn")):raise ValueError("3px rematching failed")
                    if set(rematch.matched_true_indices)!={int(r["true_index"]) for r in match}:raise ValueError("Matched truth identities differ")
                    if int(row["n_true"])!=len(truth[sid]) or len(coords)!=int(row["n_pred"]):raise ValueError("Geometry/count mismatch")
                    weak_tp=int(weak[sid][rematch.matched_true_indices].sum())
                    enriched=dict(row,method=method,seed=seed,role=role,weak_tp=weak_tp,weak_n=int(weak[sid].sum()),
                        bright_tp=rematch.tp-weak_tp,bright_n=len(truth[sid])-int(weak[sid].sum()),
                        distance_sum=float(np.sum(rematch.matched_distances)),distance_sq_sum=float(np.sum(rematch.matched_distances**2)))
                    current.append(enriched);images.append(enriched)
                    if role=="primary":
                        for radius in (.5,1.,2.,3.):
                            mm=match_coordinates(coords,truth[sid],radius)
                            radii.append(dict(method=method,seed=seed,radius_px=radius,sample_id=sid,tp=mm.tp,fp=mm.fp,fn=mm.fn))
                aggregate_metrics=aggregate(current)
                for key in ("tp","fp","fn","f1","precision","recall","weak_recall","count_mae","macro_f1","mean_error_px","rmse_error_px"):
                    if not np.isclose(aggregate_metrics[key],metrics[key],atol=2e-7,rtol=1e-6):raise ValueError("Aggregate export mismatch: "+key)
                records.append(dict(method=method,seed=seed,role=role,epoch=choice["epoch"],threshold=choice["threshold"],**aggregate_metrics))
            selected_epoch=int(selection["primary"]["epoch"]);selected=[r for r in sweep if int(r["epoch"])==selected_epoch]
            fixed=next(r for r in selected if float(r["threshold"] or -1)==.5)
            best=float(selection["primary"]["f1"])
            near=[float(r["threshold"]) for r in selected if r["threshold"] and float(r["f1"])>=best-.005]
            thresholds.append(dict(method=method,seed=seed,selected_epoch=selected_epoch,selected_threshold=selection["primary"]["threshold"],best_f1=best,
                fixed_0p5_f1=float(fixed["f1"]),within_0p005_f1_threshold_min=min(near),within_0p005_f1_threshold_max=max(near),
                native_selected_epoch=selection["native_secondary"]["epoch"],native_selected_f1=selection["native_secondary"]["f1"]))
            for row in history:
                epoch=int(row["epoch"]);best_at_epoch=max((float(r["f1"]) for r in sweep if int(r["epoch"])==epoch),default=None)
                curves.append(dict(method=method,seed=seed,epoch=epoch,train_loss=float(row["train_loss"]),val_loss=float(row["val_loss"]),best_point_f1=best_at_epoch))
            timings.append(dict(method=method,seed=seed,epoch_loop_seconds=sum(float(r["seconds"]) for r in history),
                first5_seconds=sum(float(r["seconds"]) for r in history[:5]),last20_train_loss=np.mean([float(r["train_loss"]) for r in history[-20:]]),
                last20_val_loss=np.mean([float(r["val_loss"]) for r in history[-20:]]),
                last20_train_slope=np.polyfit(range(20),[float(r["train_loss"]) for r in history[-20:]],1)[0],
                last20_val_slope=np.polyfit(range(20),[float(r["val_loss"]) for r in history[-20:]],1)[0]))
    return records,images,point_sets,radii,thresholds,curves,timings


def summarize(ctx,records,images,point_sets,radii,valid,weak,index):
    out=ctx["out"];summary=[];stats=[]
    for role in ("primary","native_secondary"):
        for method in METHODS:
            data=[r for r in records if r["role"]==role and r["method"]==method]
            summary.append(dict(method=method,role=role,n_seeds=3,**{k+suffix:float(fn([r[k] for r in data])) for k in METRICS for suffix,fn in (("_mean",np.mean),("_sd",lambda x:np.std(x,ddof=1)))}))
        for metric in METRICS:
            a=[next(r[metric] for r in records if r["method"]==METHODS[0] and r["role"]==role and r["seed"]==s) for s in SEEDS]
            b=[next(r[metric] for r in records if r["method"]==METHODS[1] and r["role"]==role and r["seed"]==s) for s in SEEDS]
            d=np.asarray(b)-a
            stats.append(dict(role=role,metric=metric,ref_mean=float(np.mean(a)),aug_mean=float(np.mean(b)),mean_delta=float(d.mean()),sd_delta=float(d.std(ddof=1)),
                positive_seeds=int((d>0).sum()),p_exact=exact_paired(d),p_holm_secondary=None,
                status="registered_primary" if role=="primary" and metric=="f1" else "exploratory_supporting",
                delta_seed42=float(d[0]),delta_seed43=float(d[1]),delta_seed44=float(d[2])))
    secondary=[r for r in stats if r["role"]=="primary" and r["metric"]!="f1"]
    for r,p in zip(secondary,holm([r["p_exact"] for r in secondary])):r["p_holm_secondary"]=p
    strata=[]
    for label,selector in (("256_native",lambda r:int(r["height"])==256),("512_native",lambda r:int(r["height"])==512),
                           ("points_le20",lambda r:int(r["n_true"])<=20),("points_gt20",lambda r:int(r["n_true"])>20)):
        for method in METHODS:
            for seed in SEEDS:
                selected=[r for r in images if r["role"]=="primary" and r["method"]==method and r["seed"]==seed and selector(r)]
                strata.append(dict(stratum=label,method=method,seed=seed,n_images=len(selected),**aggregate(selected)))
    exchange=[];common=[];persistence=[]
    for seed in SEEDS:
        a=point_sets[(METHODS[0],seed,"primary")];b=point_sets[(METHODS[1],seed,"primary")]
        for label,points in (("ref_only",a.keys()-b.keys()),("aug_only",b.keys()-a.keys()),("both",a.keys()&b.keys())):
            nw=sum(weak[sid][i] for sid,i in points)
            exchange.append(dict(seed=seed,category=label,n_points=len(points),weak_points=int(nw),bright_points=len(points)-int(nw)))
        overlap=sorted(a.keys()&b.keys())
        common.append(dict(seed=seed,common_tp=len(overlap),ref_mean_error_px=np.mean([a[k] for k in overlap]),aug_mean_error_px=np.mean([b[k] for k in overlap]),
            paired_mean_delta_px=np.mean([b[k]-a[k] for k in overlap])))
    for key,row in index.items():
        na=sum(key in point_sets[(METHODS[0],s,"primary")] for s in SEEDS);nb=sum(key in point_sets[(METHODS[1],s,"primary")] for s in SEEDS)
        persistence.append(dict(sample_id=key[0],true_index=key[1],weak=row["weak"],local_zscore=float(row["local_zscore"]),ref_detected_seeds=na,aug_detected_seeds=nb))
    radius_metrics=[]
    for method in METHODS:
        for seed in SEEDS:
            for radius in (.5,1.,2.,3.):
                selected=[r for r in radii if r["method"]==method and r["seed"]==seed and r["radius_px"]==radius]
                tp,fp,fn=[sum(r[k] for r in selected) for k in ("tp","fp","fn")]
                radius_metrics.append(dict(method=method,seed=seed,radius_px=radius,tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn),recall=tp/(tp+fn)))
    jackknife=[]
    for sid in valid:
        f=[]
        for method in METHODS:
            f.append(np.mean([aggregate([r for r in images if r["role"]=="primary" and r["method"]==method and r["seed"]==seed and r["sample_id"]!=sid])["f1"] for seed in SEEDS]))
        jackknife.append(dict(omitted_sample=sid,mean_delta_f1=float(f[1]-f[0])))
    for name,data in (("per_seed_metrics.csv",records),("summary.csv",summary),("paired_metric_effects.csv",stats),("size_density_strata.csv",strata),
                      ("truth_exchange.csv",exchange),("common_tp_localization.csv",common),("truth_detection_persistence_private.csv",persistence),
                      ("radius_sensitivity.csv",radius_metrics),("leave_one_image_out_private.csv",jackknife),("per_image_enriched_private.csv",images)):
        write_csv(out/name,data)
    return summary,stats,strata,exchange,common,persistence,radius_metrics,jackknife


def geometry_and_history(ctx,valid):
    root,out=ctx["data"],ctx["out"]
    geometry=rows(root/"outputs/next_stage_v4_recipe_20261101/01_preflight/geometry_collision_ledger.csv")
    collision=[]
    for subset in ("train","valid"):
        for k,flip in itertools.product(range(4),("False","True")):
            chosen=[r for r in geometry if r["subset"]==subset and int(r["k"])==k and r["flip"]==flip]
            collision.append(dict(subset=subset,k=k,flip=flip,images=len(chosen),points=sum(int(r["points"]) for r in chosen),
                collision_points=sum(int(r["collisions"]) for r in chosen)))
    write_csv(out/"geometry_collisions.csv",collision)
    oldsplit=rows(root/"outputs/260628test/dna_clean_113.split.csv")
    oldtest={r["sample_id"] for r in oldsplit if r["subset"]=="test"}
    oldall={r["sample_id"] for r in oldsplit}
    legacy=rows(root/"docs/next_stage_plan_20261025/cuda_analysis_20261030/per_seed_metrics.csv")
    legacy=[r for r in legacy if r["model"] in ("G0","R0","Spotiflow_F") and r["role"]=="primary"]
    write_csv(out/"legacy_same_validation.csv",legacy)
    history=dict(historical_113=dict(images=12,truth_points=213,tp=174,fp=23,fn=39,f1=348/410,threshold=.4,
        split_counts={s:sum(r["subset"]==s for r in oldsplit) for s in ("train","valid","test")},
        role="single historical model; threshold inspected on historical test; not current baseline"),
        current=dict(train_images=176,valid_images=22,truth_points=372,test_images=22,valid_old113_images=len(set(valid)&oldall),
            valid_new107_images=len(set(valid)-oldall),shared_images_with_historical_test=len(set(valid)&oldtest),
            matching_radius_px=3.,metric="mean of three seed micro-F1 scores on development validation"),
        metric_formula_changed=False,causal_explanation_of_cross_cohort_difference="not_identifiable",
        legacy_G0=dict(f1_mean=float(np.mean([float(r["f1"]) for r in legacy if r["model"]=="G0"])),device="Intel Arc140T XPU",amp="bfloat16",split_same=True),
        new_G0=dict(device=read_json(ctx["queue"]/"protocol.json")["gpu"],precision="CUDA float32",same_split_as_legacy=True),
        independence="biological groups unresolved; no new independent validation performed")
    dump(out/"history_comparability.json",history)
    return history,collision,legacy


def replay(ctx,truth,weak,valid):
    import torch
    from deepblink_torch.cli import predict
    from research_v4.engine import infer_candidates,model_hash
    from research_v4.evaluation import save_candidates,sweep,policies_probability,selection_key
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    root,out=ctx["data"],ctx["out"];package=root/"outputs/next_stage_v4_recipe_20261101/04_bundles/ce2d1821ceec/p1_v4_training"
    (out/"cpu_replay").mkdir(exist_ok=True)
    archive=package/"assets/development_only.npz"
    keys=[k+"_valid" for k in ("x","y","sample_id","original_shape")]
    with np.load(archive,allow_pickle=True) as z:
        if len(z.files)!=8 or any("test" in k for k in z.files):raise ValueError("Unexpected archive keys")
        data={k:z[k+"_valid"] for k in ("x","y","sample_id","original_shape")}
    data["x"]=[np.asarray(x,np.float32) for x in data["x"]];data["sample_id"]=np.asarray(data["sample_id"],str)
    data["weak"]=[weak[sid] for sid in data["sample_id"]]
    for sid,y in zip(data["sample_id"],data["y"]):
        if not np.allclose(y,truth[sid],atol=1e-6,rtol=0):raise ValueError("CPU replay truth changed")
    checks=[]
    for method in METHODS:
        for seed in SEEDS:
            folder=ctx["queue"]/method/f"seed_{seed}";selection=read_json(folder/"selection.json")
            model,_cell_size=predict._load_model(folder/"best_model.pt",torch.device("cpu"))
            payload=torch.load(folder/"best_model.pt",map_location="cpu",weights_only=True)
            if model_hash(model)!=payload["model_state_sha256"]:raise ValueError("Reloaded model state differs")
            start=time.perf_counter();candidates=infer_candidates(model,data,"cpu");elapsed=time.perf_counter()-start
            save_candidates(out/"cpu_replay"/(method+f"_{seed}.npz"),candidates,data)
            results=sweep(candidates,data,policies_probability());choice=results[int(selection["primary"]["threshold_index"])]
            expected=selection["primary"]
            exact_counts=all(choice[k]==expected[k] for k in ("tp","fp","fn","weak_tp"))
            for row in results:row.update(method=method,seed=seed,role="CPU diagnostic replay; not a new threshold selection")
            write_csv(out/"cpu_replay"/(method+f"_{seed}_frozen_grid.csv"),results)
            checks.append(dict(method=method,seed=seed,model_reloaded=True,model_state_hash_verified=True,images=22,counts_identical=exact_counts,
                gpu_tp=expected["tp"],cpu_tp=choice["tp"],gpu_fp=expected["fp"],cpu_fp=choice["fp"],gpu_fn=expected["fn"],cpu_fn=choice["fn"],
                seconds=elapsed,source_threshold=expected["threshold"],new_optimizer_updates=0))
            print(json.dumps(dict(stage="cpu_replay",method=method,seed=seed,counts_identical=exact_counts,seconds=elapsed)),flush=True)
            del model
    write_csv(out/"cpu_replay_checks.csv",checks)
    return dict(passed=all(r["counts_identical"] for r in checks),models=6,images_per_model=22,accessed_array_keys=keys,test_arrays_read=False,new_optimizer_updates=0)


def figures(ctx,records,curves,radius_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=ctx["out"]/"figures";out.mkdir(exist_ok=True)
    labels=("G0 reference","G0 joint augmentation");colors=("#0072B2","#D55E00")
    with plt.rc_context({"svg.fonttype":"none","font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42}):
        fig,axes=plt.subplots(2,3,figsize=(12,7.4),layout="constrained")
        for ax,metric,label in zip(axes.ravel(),("f1","precision","recall","weak_recall","count_mae","mean_error_px"),
            ("3px micro-F1","Precision","Recall","Weak-locus recall","Count MAE / image","Matched localization error / px")):
            values=[[next(r[metric] for r in records if r["method"]==m and r["seed"]==s and r["role"]=="primary") for s in SEEDS] for m in METHODS]
            ax.bar([0,1],[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) for v in values],color=colors,edgecolor="black",capsize=4)
            for j,s in enumerate(SEEDS):ax.plot([0,1],[values[0][j],values[1][j]],"o-",color="black",alpha=.55,ms=4,lw=.8)
            ax.set(xticks=[0,1],xticklabels=["Ref","Aug"],ylabel=label,ylim=(0,1) if metric in ("f1","precision","recall","weak_recall") else (0,max(max(v) for v in values)*1.25))
            ax.set_title(f"Aug - Ref = {np.mean(values[1])-np.mean(values[0]):+.4f}")
        fig.suptitle("Fixed selected models on 22 development images; mean ± sample SD, n=3 training seeds\nPrimary micro-F1 exact paired p=1.00; secondary metrics exploratory",fontsize=11)
        for ext in ("png","svg","pdf"):fig.savefig(out/("recipe_metric_bars."+ext),dpi=220)
        plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
        for method,label,color in zip(METHODS,labels,colors):
            for metric,ax in (("train_loss",axes[0]),("val_loss",axes[1])):
                a=np.asarray([[next(r[metric] for r in curves if r["method"]==method and r["seed"]==s and r["epoch"]==e) for e in range(1,101)] for s in SEEDS])
                ax.plot(range(1,101),a.mean(0),color=color,label=label);ax.fill_between(range(1,101),a.mean(0)-a.std(0,ddof=1),a.mean(0)+a.std(0,ddof=1),color=color,alpha=.13)
            epochs=list(range(5,101,5));a=np.asarray([[next(r["best_point_f1"] for r in curves if r["method"]==method and r["seed"]==s and r["epoch"]==e) for e in epochs] for s in SEEDS])
            axes[2].plot(epochs,a.mean(0),"o-",color=color,label=label,ms=3)
        for ax,title in zip(axes,("Training native loss","Validation native loss","Best registered threshold per checkpoint")):
            ax.set(xlabel="Epoch",title=title);ax.legend(frameon=False,fontsize=8)
        axes[2].set(ylabel="Development micro-F1",ylim=(0,1))
        for ext in ("png","svg","pdf"):fig.savefig(out/("learning_curves."+ext),dpi=220)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4.4),layout="constrained")
        for method,label,color in zip(METHODS,labels,colors):
            radii=(.5,1.,2.,3.)
            y=[[next(r["f1"] for r in radius_metrics if r["method"]==method and r["seed"]==s and r["radius_px"]==x) for s in SEEDS] for x in radii]
            ax.errorbar(radii,[np.mean(v) for v in y],yerr=[np.std(v,ddof=1) for v in y],fmt="o-" if method==METHODS[0] else "s--",color=color,label=label,capsize=4)
        ax.set(xlabel="Matching radius / px",ylabel="Micro-F1",ylim=(0,1),title="Exploratory radius sensitivity; registered endpoint remains 3px")
        ax.legend(frameon=False)
        for ext in ("png","svg","pdf"):fig.savefig(out/("radius_sensitivity."+ext),dpi=220)
        plt.close(fig)
    for p in out.glob("*.svg"):p.write_text("\n".join(line.rstrip() for line in p.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True);p.add_argument("--results-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replay-selected-models",action="store_true")
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    ctx=dict(data=args.data_root.resolve(),queue=args.results_root.resolve(),out=out)
    audit=verify(ctx);valid,truth,weak,index=truth_reference(ctx["data"])
    records,images,point_sets,radii,thresholds,curves,timings=read_trials(ctx,valid,truth,weak)
    result=summarize(ctx,records,images,point_sets,radii,valid,weak,index)
    history,collisions,legacy=geometry_and_history(ctx,valid)
    write_csv(out/"threshold_diagnostics.csv",thresholds);write_csv(out/"training_timing_and_slopes.csv",timings);write_csv(out/"learning_curve_source.csv",curves)
    replay_check=replay(ctx,truth,weak,valid) if args.replay_selected_models else dict(performed=False)
    figures(ctx,records,curves,result[6])
    summary=dict(passed=True,audit=audit,metrics=result[0],effects=result[1],common_tp_localization=result[4],history=history,
        cpu_replay=replay_check,valid_grid_collisions=sum(r["collision_points"] for r in collisions if r["subset"]=="valid" and r["k"]==0 and r["flip"]=="False"),
        jackknife_delta_range=[min(r["mean_delta_f1"] for r in result[7]),max(r["mean_delta_f1"] for r in result[7])],
        code_sha256=sha256(Path(__file__)),python=sys.version,platform=platform.platform(),
        registered_endpoint="mean of3 paired training-seed micro-F1 values at3px; existing620-pair selection",new_strata="exploratory",selection_adjusted=False,
        new_training_tasks=0,test_arrays_read=False,new_thresholds_selected_for_deployment=False,
        input_file_hashes={p.relative_to(ctx["queue"]).as_posix():sha256(p) for p in sorted(ctx["queue"].rglob("*")) if p.is_file()})
    dump(out/"deep_analysis.json",summary)
    print(json.dumps(dict(passed=True,output=str(out),gpu=audit["gpu"],primary_delta=result[1][0]["mean_delta"],primary_p=result[1][0]["p_exact"],cpu_replay=replay_check),ensure_ascii=False),flush=True)


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    main()
