"""Separate native-default, calibrated and fine-tuned full-image evaluation."""
import json
from pathlib import Path
import shutil
import time
import yaml

import numpy as np
from spotiflow.model import Spotiflow
from deepblink_torch.evaluation import match_coordinates, metrics_from_match
from deepblink_torch.round2_protocol import threshold_mask
from deepblink_torch.spotiflow_bridge import (save_candidates, read_candidates,
    evaluate_checkpoints, spotiflow_candidates, validate_candidates, write_csv)
from .common import ROOT, OUT, WEIGHTS, dump, clean, status, now
from deepblink_torch.experiment_log import sha256


def bind_threshold(folder,threshold):
    path=Path(folder)/"thresholds.yaml"
    config=yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    config["prob_thresh_best"]=[float(threshold)]
    path.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")


def summarize(candidates, data, folder, threshold=None):
    from error_context_diagnostics import _extract_patch_features
    images,points,matches=[],[],[]
    totals=np.zeros(3,dtype=int)
    weak_tp=weak_n=0
    for sid,shape,truth,image,(coords,scores) in zip(data["sample_id"],data["original_shape"],data["y"],data["images"],candidates):
        if threshold is not None:
            keep=threshold_mask(scores,threshold)
            coords,scores=coords[keep],scores[keep]
        matched=match_coordinates(coords,truth,3.)
        metrics=metrics_from_match(matched)
        totals += [matched.tp,matched.fp,matched.fn]
        weak=np.array([_extract_patch_features(image,y,x,10)["local_zscore"]<2.4566830772038957 for y,x in truth])
        weak_n+=int(weak.sum())
        weak_tp+=int(weak[matched.matched_true_indices].sum())
        images.append(dict(sample_id=str(sid),height=int(shape[0]),width=int(shape[1]),
            **{k:(None if not np.isfinite(v) else float(v)) for k,v in metrics.items()}))
        points += [dict(sample_id=str(sid),y=float(y),x=float(x),p=float(p)) for (y,x),p in zip(coords,scores)]
        matches += [dict(sample_id=str(sid),pred_index=int(pi),true_index=int(ti),distance_px=float(d))
                    for pi,ti,d in zip(matched.matched_pred_indices,matched.matched_true_indices,matched.matched_distances)]
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    write_csv(folder/"per_image_metrics.csv",images,list(images[0]))
    write_csv(folder/"prediction_points.csv",points,["sample_id","y","x","p"])
    write_csv(folder/"matches.csv",matches,["sample_id","pred_index","true_index","distance_px"])
    tp,fp,fn=map(int,totals)
    result=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn),weak_tp=weak_tp,weak_n=weak_n,
                weak_recall=weak_tp/weak_n,development_validation_only=True,test_evaluated=False)
    if tp+fn!=372 or weak_n!=96:raise ValueError("Frozen truth/weak counts differ.")
    dump(folder/"metrics.json",result)
    return result


def pretrained(data):
    model=Spotiflow.from_folder(WEIGHTS,which="best",inference_mode=True)
    model.eval()
    folder=OUT/"01_pretrained/Z0_default"
    if not (folder/"status.json").exists() or json.loads((folder/"status.json").read_text())["status"]!="complete":
        folder.mkdir(parents=True,exist_ok=True)
        status("pretrained_default",activity="Native general defaults on 22 full validation images")
        default_threshold=clean(model._prob_thresh)
        candidates=[]
        start=time.perf_counter()
        for image in data["images"]:
            # Omit threshold, subpixel and NMS arguments: retain official defaults.
            points,details=model.predict(image,device="cpu",verbose=False)
            coords=np.asarray(points,dtype=np.float32).reshape(-1,2)
            scores=np.asarray(details.prob,dtype=np.float32).reshape(-1)
            inside=(coords>=0).all(1)&(coords<image.shape).all(1)
            candidates.append(validate_candidates(coords[inside],scores[inside],image.shape))
        save_candidates(folder/"candidates.npz",candidates,data["sample_id"],data["original_shape"])
        metrics=summarize(candidates,data,folder)
        dump(folder/"selection.json",dict(track="Z0",native_threshold=default_threshold,
             calibration_opportunities=0,checkpoint_count=1,metrics=metrics,inference_seconds=time.perf_counter()-start,
             native_defaults_retained=True,pretrained_weight_sha256=sha256(WEIGHTS/"best.pt")))
        dump(folder/"status.json",dict(status="complete",completed=now()))
    folder=OUT/"01_pretrained/Z1_validation_calibrated"
    if not (folder/"status.json").exists() or json.loads((folder/"status.json").read_text())["status"]!="complete":
        folder.mkdir(parents=True,exist_ok=True)
        status("pretrained_calibrated",activity="One frozen general checkpoint, 31 global thresholds")
        tick=time.perf_counter()
        candidates=[spotiflow_candidates(model,image,"cpu") for image in data["images"]]
        inference_seconds=time.perf_counter()-tick
        save_candidates(folder/"candidates.npz",candidates,data["sample_id"],data["original_shape"])
        rows,selected,_=evaluate_checkpoints([(0,candidates)],data)
        write_csv(folder/"threshold_metrics.csv",rows,list(rows[0]))
        metrics=summarize(candidates,data,folder,selected["threshold"])
        dump(folder/"selection.json",dict(track="Z1",**selected,metrics=metrics,checkpoint_count=1,
             threshold_count=31,inference_seconds=inference_seconds,total_seconds=time.perf_counter()-tick,
             weights_finetuned=False,pretrained_weight_sha256=sha256(WEIGHTS/"best.pt")))
        dump(folder/"status.json",dict(status="complete",completed=now()))
    return model


def finalize_finetune(folder,data):
    folder=Path(folder)
    snapshots=json.loads((folder/"snapshots.json").read_text(encoding="utf-8"))
    if [r["epoch"] for r in snapshots]!=list(range(5,101,5)):
        raise ValueError("Require exactly 20 current-epoch checkpoints.")
    candidate_sets=[]
    for record in snapshots:
        path=folder/f"epoch_{record['epoch']:03d}"
        if sha256(path/"best.pt")!=record["weight_sha256"] or sha256(path/"candidates.npz")!=record["candidate_sha256"]:
            raise ValueError("Snapshot weight/candidates changed.")
        candidate_sets.append((record["epoch"],read_candidates(path/"candidates.npz",data)))
    all_rows,selected,_=evaluate_checkpoints(candidate_sets,data)
    native=min(snapshots,key=lambda r:(r["native_val_loss"],r["epoch"]))
    secondary=max([r for r in all_rows if r["epoch"]==native["epoch"]],
                  key=lambda r:(r["f1"],r["precision"],r["threshold"]))
    for role,choice in (("primary",selected),("native_secondary",secondary)):
        source=folder/f"epoch_{choice['epoch']:03d}"
        target=folder/role
        target.mkdir(exist_ok=True)
        for name in ("best.pt","config.yaml","thresholds.yaml"):
            shutil.copyfile(source/name,target/name)
        bind_threshold(target,choice["threshold"])
        candidates=dict(candidate_sets)[choice["epoch"]]
        summarize(candidates,data,target,choice["threshold"])
    optimizer=json.loads((folder/"optimizer_summary.json").read_text())
    if optimizer["completed_epochs"]!=100 or optimizer["optimizer_updates"]!=4400:
        raise ValueError("Incomplete formal epoch/update budget.")
    write_csv(folder/"all_epoch_threshold_metrics.csv",all_rows,list(all_rows[0]))
    result=dict(primary=selected,native_secondary=secondary,epoch_threshold_pairs=len(all_rows),
                checkpoint_count=20,test_evaluated=False,validation_selection_optimism=True,
                primary_weight_sha256=sha256(folder/"primary/best.pt"),selected_model_threshold_bound=True)
    dump(folder/"selection.json",result)
    return result
