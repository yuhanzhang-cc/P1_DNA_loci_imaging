"""Fixed-weight review material and same-CPU inference timings, no relabeling."""
import csv
import json
from pathlib import Path
import random
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from spotiflow.model import Spotiflow

from deepblink_torch.cli import predict
from deepblink_torch.preprocessing import prepare_image
from deepblink_torch.round2_protocol import decode_candidates, threshold_mask
from deepblink_torch.evaluation import match_coordinates
from deepblink_torch.spotiflow_bridge import spotiflow_candidates, write_csv
from .common import ROOT, OUT, WEIGHTS, rows, dump, status


def local_method(key):
    root=ROOT/("outputs/next_stage_v3/01_screen/R0/seed_42" if key=="R0" else "outputs/next_stage_v3_closing_controls/01_screen/G0/seed_42")
    model,_=predict._load_model(root/"best_model.pt",torch.device("cpu"))
    model.eval().requires_grad_(False)
    selection=json.loads((root/"selection.json").read_text(encoding="utf-8"))["primary"]
    return model,selection["threshold"],root


def own_candidates(model,image):
    tensor=torch.from_numpy(prepare_image(image,512)[None,None])
    with torch.inference_mode():
        output=predict._forward_with_geometry(model,tensor,*image.shape)[0].numpy()
    return decode_candidates(output,image.shape)


def spot_reference():
    folder=OUT/"02_finetune/seed_42"
    if (folder/"selection.json").exists() and (folder/"status.json").exists() and json.loads((folder/"status.json").read_text())["status"]=="complete":
        choice=json.loads((folder/"selection.json").read_text())["primary"]
        return Spotiflow.from_folder(folder/"primary"),choice["threshold"],"Spotiflow_F_seed42"
    choice=json.loads((OUT/"01_pretrained/Z1_validation_calibrated/selection.json").read_text())
    return Spotiflow.from_folder(WEIGHTS),choice["threshold"],"Spotiflow_Z1"


def review(data):
    destination=OUT/"03_error_review"
    if (destination/"checks.json").exists():return json.loads((destination/"checks.json").read_text())
    destination.mkdir(parents=True,exist_ok=True)
    status("error_review",activity="Existing 40-image queue; anonymized methods, labels stay unchanged")
    queue=rows(ROOT/"outputs/next_stage_v2/08_innovation/blinded_annotation_review_queue.csv")
    if len(queue)!=40:raise ValueError("Original blinded review queue differs.")
    references={}
    for subset,d in data.items():
        for index,sample in enumerate(d["sample_id"]):references[str(sample)]=(subset,index)
    r0,p0,root0=local_method("R0")
    g0,pg,rootg=local_method("G0")
    spot,ps,spot_name=spot_reference()
    methods=[("R0",r0,p0,root0),("G0",g0,pg,rootg),(spot_name,spot,ps,None)]
    random.Random(20261030).shuffle(methods)
    blind={f"Method {chr(65+i)}":m[0] for i,m in enumerate(methods)}
    dump(OUT/"review_key_private.json",dict(mapping=blind,representative_seed=42,
         selection="fixed seed 42, not best seed; Spotiflow availability rule specified before audit"))
    cached={}
    for key,_,_,source in methods:
        if source is not None:cached[key]=rows(source/"prediction_points.csv")
    review_rows,error_rows=[],[]
    for qindex,q in enumerate(queue):
        sample=q["sample_id"]
        subset,index=references[sample]
        image=data[subset]["images"][index]
        truth=data[subset]["y"][index]
        low,high=np.percentile(image,[1,99.8])
        if high<=low:high=low+1
        panels=[]
        all_error_locations=[]
        for mi,(key,model,threshold,source) in enumerate(methods):
            blind_name=f"Method {chr(65+mi)}"
            if subset=="valid" and source is not None:
                points=np.array([[float(r["y"]),float(r["x"])] for r in cached[key] if r["sample_id"]==sample],dtype=np.float32).reshape(-1,2)
                origin="frozen_validation_cache"
            else:
                coords,scores=(own_candidates(model,image) if source is not None else spotiflow_candidates(model,image,"cpu"))
                points=coords[threshold_mask(scores,threshold)]
                origin="CPU_frozen_weights_and_threshold"
            match=match_coordinates(points,truth,3.)
            panels.append((blind_name,points,match))
            review_rows.append(dict(sample_id=sample,subset=subset,method=blind_name,tp=match.tp,fp=match.fp,fn=match.fn,
                                    prediction_source=origin,human_review_status="pending"))
            for kind,coords in (("reference_relative_FP",points[match.unmatched_pred_indices]),("reference_relative_FN",truth[match.unmatched_true_indices])):
                for y,x in coords:
                    edge=min(float(y),float(x),image.shape[0]-1-float(y),image.shape[1]-1-float(x))
                    error_rows.append(dict(sample_id=sample,subset=subset,method=blind_name,type=kind,
                        y=float(y),x=float(x),near_border_10px=edge<10,automatic_candidate_only=True,
                        human_confirmation="pending",reference_is_independent_gold_standard=False))
                    all_error_locations.append((float(y),float(x)))
        fig,axes=plt.subplots(1,3,figsize=(12,4.7))
        for ax,(name,points,match) in zip(axes,panels):
            ax.imshow(image,cmap="gray",vmin=low,vmax=high,interpolation="nearest")
            ax.scatter(truth[:,1],truth[:,0],s=22,facecolors="none",edgecolors="#00D9E8",linewidths=.7)
            if len(points):ax.scatter(points[:,1],points[:,0],s=18,c="#FFB000",marker="+",linewidths=.7)
            ax.set_title(f"{name}: TP {match.tp}, FP {match.fp}, FN {match.fn}",fontsize=10)
            ax.set_axis_off()
        fig.suptitle(f"{sample} | {subset} | cyan: reference, orange: prediction",fontsize=12)
        fig.text(.04,.025,"Same 1st–99.8th percentile display limits across panels. Automated candidates; human review pending.",fontsize=9)
        fig.subplots_adjust(left=.02,right=.98,top=.84,bottom=.12,wspace=.06)
        fig.savefig(destination/f"{qindex+1:02d}_{sample}.png",dpi=160,facecolor="white")
        plt.close(fig)
        locations=sorted(set(all_error_locations))[:4]
        if locations:
            fig,axes=plt.subplots(len(locations),3,figsize=(8,2.3*len(locations)),squeeze=False)
            for ri,(y,x) in enumerate(locations):
                y0,y1=max(0,int(y)-12),min(image.shape[0],int(y)+13)
                x0,x1=max(0,int(x)-12),min(image.shape[1],int(x)+13)
                for ci,(name,points,_) in enumerate(panels):
                    ax=axes[ri,ci]
                    ax.imshow(image[y0:y1,x0:x1],extent=(x0-.5,x1-.5,y1-.5,y0-.5),cmap="gray",vmin=low,vmax=high,interpolation="nearest")
                    for coords,color,marker in ((truth,"#00D9E8","o"),(points,"#FFB000","+")):
                        keep=(coords[:,0]>=y0)&(coords[:,0]<y1)&(coords[:,1]>=x0)&(coords[:,1]<x1)
                        c=coords[keep]
                        if len(c):ax.scatter(c[:,1],c[:,0],s=60,marker=marker,facecolors="none" if marker=="o" else color,edgecolors=color,linewidths=.8)
                    ax.set_title(f"{name} | ({y:.1f}, {x:.1f})",fontsize=9)
                    ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
            fig.tight_layout()
            fig.savefig(destination/f"{qindex+1:02d}_{sample}_patches.png",dpi=160,facecolor="white")
            plt.close(fig)
        if qindex%5==0:status("error_review",activity=f"Prepared {qindex+1}/40 blinded review images")
    write_csv(destination/"review_index.csv",review_rows,list(review_rows[0]))
    write_csv(destination/"automatic_error_candidates.csv",error_rows,list(error_rows[0]) if error_rows else ["sample_id"])
    write_csv(destination/"human_review_queue.csv",queue,list(queue[0]))
    checks=dict(queue_images=40,method_panels=120,error_candidate_rows=len(error_rows),
                human_review_completed=False,labels_modified=False,test_arrays_loaded=False,
                patch_sampling="first four lexicographically ordered unique error coordinates per image; not selected by winning method",
                display="per-image original intensity percentiles 1 and 99.8, fixed across method panels; no raw image modification")
    dump(destination/"checks.json",checks)
    (destination/"README.md").write_text("# Blinded development review material\n\nOrange plus: prediction; cyan circle: reference.\nAll FP/FN labels are relative to existing references, not independent human truth.\nDo not open ../review_key_private.json before reviewing.\nFill the copied queue; the original labels and queue were not modified.\nPredetermined representative seed 42; pretrained Spotiflow is one checkpoint, not three trained replicates.\n",encoding="utf-8")
    return checks


def efficiency(data):
    path=OUT/"04_efficiency/cpu_timings.csv"
    if path.exists():return rows(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    status("efficiency",activity="Same CPU, four threads, batch 1; separate 256/512 original images")
    r0,p0,_=local_method("R0")
    g0,pg,_=local_method("G0")
    spot,ps,spot_name=spot_reference()
    result=[]
    for size in (256,512):
        index=next(i for i,x in enumerate(data["train"]["images"]) if x.shape==(size,size))
        image=data["train"]["images"][index]
        for name,model,threshold in (("R0",r0,p0),("G0",g0,pg),(spot_name,spot,ps)):
            values=[]
            for repeat in range(12):
                tick=time.perf_counter()
                coords,scores=own_candidates(model,image) if name in ("R0","G0") else spotiflow_candidates(model,image,"cpu")
                coords[threshold_mask(scores,threshold)]
                if repeat>=2:values.append(time.perf_counter()-tick)
            result.append(dict(model=name,original_size=size,sample_id=str(data["train"]["sample_id"][index]),
                device="cpu",threads=torch.get_num_threads(),batch_size=1,precision="float32",warmups=2,repeats=10,
                end_to_end_median_seconds=statistics.median(values),min_seconds=min(values),max_seconds=max(values),
                parameters=sum(p.numel() for p in model.parameters()),
                includes="normalization; model-specific canvas; forward; candidates; threshold; excludes evaluation matcher"))
    write_csv(path,result,list(result[0]))
    return result
