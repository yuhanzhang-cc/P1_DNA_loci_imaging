"""Versioned, blinded review of frozen adapted models, never relabeling data."""
from collections import defaultdict
from pathlib import Path
import random
import html
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from deepblink_torch.cli import predict
from deepblink_torch.preprocessing import prepare_image
from deepblink_torch.round2_protocol import decode_candidates,threshold_mask
from deepblink_torch.evaluation import match_coordinates
from deepblink_torch.spotiflow_bridge import spotiflow_candidates
from .common import rows,read_json,dump,write_csv,sha256,canonical_hash


def method_folder(root,key,seed):
    if key=="R0":return root/f"outputs/next_stage_v3/01_screen/R0/seed_{seed}"
    if key=="G0":return root/f"outputs/next_stage_v3_closing_controls/01_screen/G0/seed_{seed}"
    return root/f"outputs/next_stage_spotiflow_20261030/06_cuda_results/extracted/outputs/spotiflow_cuda/seed_{seed}"


def cached_predictions(folder,spot=False):
    points=rows(folder/("primary/prediction_points.csv" if spot else "prediction_points.csv"))
    cache=defaultdict(list)
    for r in points:cache[r["sample_id"]].append((float(r["y"]),float(r["x"]),float(r["p"])))
    return {sid:np.asarray(v,np.float32).reshape(-1,3) for sid,v in cache.items()}


def font(size):
    for p in (Path("C:/Windows/Fonts/arial.ttf"),Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if p.exists():return ImageFont.truetype(str(p),size),str(p)
    return ImageFont.load_default(),"Pillow_default_draft_fallback"


def draw_marks(draw,points,truth,match,origin=(0,0),box=None):
    oy,ox=origin
    def pixel(y,x):return float(x)+ox,float(y)+oy
    for y,x in truth:
        if box is not None and not(box[0]<=y<box[2] and box[1]<=x<box[3]):continue
        xx,yy=pixel(y,x);draw.line((xx-2,yy,xx+2,yy),fill="#FFD700",width=1);draw.line((xx,yy-2,xx,yy+2),fill="#FFD700",width=1)
    for y,x in points:
        if box is not None and not(box[0]<=y<box[2] and box[1]<=x<box[3]):continue
        xx,yy=pixel(y,x);draw.ellipse((xx-3,yy-3,xx+3,yy+3),outline="#00D5FF",width=1)
    for i in match.unmatched_pred_indices:
        y,x=points[i]
        if box is not None and not(box[0]<=y<box[2] and box[1]<=x<box[3]):continue
        xx,yy=pixel(y,x);draw.rectangle((xx-4,yy-4,xx+4,yy+4),outline="#FF44DD",width=1)
    for i in match.unmatched_true_indices:
        y,x=truth[i]
        if box is not None and not(box[0]<=y<box[2] and box[1]<=x<box[3]):continue
        xx,yy=pixel(y,x);draw.line((xx-3,yy-3,xx+3,yy+3),fill="#FF7F00",width=1);draw.line((xx-3,yy+3,xx+3,yy-3),fill="#FF7F00",width=1)


def cross_seed_stability(root,data,keys):
    result=[];checks=[]
    for key in keys:
        caches={seed:cached_predictions(method_folder(root,key,seed),key=="Spotiflow_F_seed42") for seed in (42,43,44)}
        totals={seed:np.zeros(3,int) for seed in (42,43,44)}
        for sid,truth in zip(data["sample_id"],data["y"]):
            all_matches={};fps={}
            for seed,cache in caches.items():
                p=cache.get(str(sid),np.empty((0,3),np.float32))[:,:2]
                match=match_coordinates(p,truth,3.);all_matches[seed]=match;fps[seed]=p[match.unmatched_pred_indices]
                totals[seed]+=[match.tp,match.fp,match.fn]
            for i,(y,x) in enumerate(truth):
                missing=[int(i in all_matches[s].unmatched_true_indices) for s in (42,43,44)]
                result.append(dict(sample_id=str(sid),method=key,kind="reference_point_FN",anchor_index=i,y=float(y),x=float(x),
                                   seed42=missing[0],seed43=missing[1],seed44=missing[2],support=sum(missing)))
            anchor=fps[42];support={42:np.ones(len(anchor),int)}
            for seed in (43,44):
                m=match_coordinates(fps[seed],anchor,3.)
                hits=np.zeros(len(anchor),int);hits[m.matched_true_indices]=1;support[seed]=hits
            for i,(y,x) in enumerate(anchor):
                result.append(dict(sample_id=str(sid),method=key,kind="seed42_FP_spatial_anchor",anchor_index=i,y=float(y),x=float(x),
                                   seed42=1,seed43=int(support[43][i]),seed44=int(support[44][i]),support=int(1+support[43][i]+support[44][i])))
        for seed,count in totals.items():
            expected=read_json(method_folder(root,key,seed)/"selection.json")["primary"]
            if count.tolist()!=[expected[k] for k in ("tp","fp","fn")]:raise ValueError("Frozen review prediction counts differ")
            checks.append(dict(method=key,seed=seed,tp=int(count[0]),fp=int(count[1]),fn=int(count[2]),passed=True))
    return result,checks


def run(ctx):
    from spotiflow.model import Spotiflow
    ctx.task("finetuned_error_review","running")
    root=ctx.data_root;out=root/ctx.spec["review"]["output"]
    out.mkdir(parents=True,exist_ok=True)
    keys=["R0","G0","Spotiflow_F_seed42"]
    fingerprints={}
    caches={};models={};thresholds={}
    for key in keys:
        folder=method_folder(root,key,42);spot=key.startswith("Spotiflow")
        choice=read_json(folder/"selection.json")["primary"]
        if read_json(folder/"status.json")["status"] not in ("complete",):raise ValueError("Review requires completed models")
        model_path=folder/("primary/best.pt" if spot else "best_model.pt")
        fingerprints[key]=dict(weight_sha256=sha256(model_path),selection_sha256=sha256(folder/"selection.json"),
                               prediction_sha256=sha256(folder/("primary/prediction_points.csv" if spot else "prediction_points.csv")),threshold=choice["threshold"])
        thresholds[key]=choice["threshold"]
        caches[key]=cached_predictions(folder,spot)
        models[key]=Spotiflow.from_folder(folder/"primary",inference_mode=True) if spot else predict._load_model(model_path,torch.device("cpu"))[0]
    lock=dict(methods=fingerprints,development_sha256=ctx.spec["data"]["development_sha256"],representative_seed=42,
              source_sha256=sha256(__file__),display="per-image p1/p99.8, same in all three panels; native pixels, no physical scale bar")
    if (out/"lock.json").exists() and read_json(out/"lock.json")!=lock:raise ValueError("Review identity changed; preserve old output and version explicitly")
    dump(out/"lock.json",lock)
    original=rows(root/ctx.spec["review"]["original_queue"])
    ids=sorted({r["sample_id"] for r in original}|set(ctx.corpus.splits["valid"]["sample_id"]))
    references={str(sid):(subset,i) for subset,data in ctx.corpus.splits.items() for i,sid in enumerate(data["sample_id"])}
    if len(ids)!=60 or sum(references[s][0]=="train" for s in ids)!=38:raise ValueError("Review queue differs")
    ordered=keys.copy();random.Random(20261102).shuffle(ordered)
    names={key:f"Method {chr(65+i)}" for i,key in enumerate(ordered)}
    dump(out/"review_key_private.json",dict(mapping={names[k]:k for k in keys},models=fingerprints,seed=42,selection="fixed representative, not best seed"))
    for name in ("full_images","patches","display_native"):(out/name).mkdir(exist_ok=True)
    f,font_path=font(16);small,_=font(12)
    manifest=[];panels=[];errors=[];human=[];patch_ledger=[]
    for number,sid in enumerate(ids,1):
        token=f"I{number:03d}";subset,i=references[sid];data=ctx.corpus.splits[subset]
        image,truth=data["x"][i],data["y"][i];h,w=image.shape
        low,high=map(float,np.percentile(image,[1,99.8]))
        if high<=low:high=low+1.
        raster=Image.fromarray(np.uint8(np.clip((image-low)/(high-low),0,1)*255)).convert("RGB")
        raster.save(out/"display_native"/(token+".png"))
        group=Image.new("RGB",(1656,618),"white");draw=ImageDraw.Draw(group)
        draw.text((16,2),"Reference + (yellow); prediction circle (cyan); relative FP square (magenta); relative FN x (orange)",font=small,fill="black")
        methods=[];locations=[]
        for column,key in enumerate(ordered):
            if subset=="valid":
                values=caches[key].get(sid,np.empty((0,3),np.float32));coords,scores=values[:,:2],values[:,2]
                origin="frozen_validation_predictions"
            else:
                if key.startswith("Spotiflow"):coords,scores=spotiflow_candidates(models[key],image,"cpu")
                else:
                    x=torch.from_numpy(prepare_image(image,512)[None,None])
                    with torch.inference_mode():prediction=predict._forward_with_geometry(models[key],x,h,w)[0].numpy()
                    coords,scores=decode_candidates(prediction,image.shape)
                keep=threshold_mask(scores,thresholds[key]);coords,scores=coords[keep],scores[keep]
                origin="CPU_frozen_model_and_threshold"
            match=match_coordinates(coords,truth,3.);methods.append((key,coords,match))
            group.paste(raster,(column*552+20,65))
            draw.text((column*552+20,28),f"{token} | {names[key]} | {h} x {w} px",font=f,fill="black")
            draw.rectangle((column*552+19,64,column*552+20+w,65+h),outline="#888888")
            draw_marks(draw,coords,truth,match,origin=(65,column*552+20))
            panels.append(dict(image_token=token,sample_id=sid,subset=subset,method=names[key],tp=match.tp,fp=match.fp,fn=match.fn,prediction_source=origin))
            human.append(dict(image_token=token,method=names[key],image_file=f"full_images/{token}.png",reference_status="pending",candidate_status="pending",reviewer="",notes=""))
            for kind,points,indices in (("reference_relative_FP",coords[match.unmatched_pred_indices],match.unmatched_pred_indices),("reference_relative_FN",truth[match.unmatched_true_indices],match.unmatched_true_indices)):
                for point,index in zip(points,indices):
                    locations.append(tuple(np.round(point,3)))
                    errors.append(dict(image_token=token,method=names[key],type=kind,y=float(point[0]),x=float(point[1]),index=int(index),human_status="pending"))
        group.save(out/"full_images"/(token+".png"))
        picks=sorted(set(locations))[:4]
        patch=Image.new("RGB",(456,max(1,len(picks))*116+40),"white");pd=ImageDraw.Draw(patch)
        for column,key in enumerate(ordered):pd.text((column*152+8,7),names[key],font=small,fill="black")
        for row,(y,x) in enumerate(picks):
            yc,xc=int(round(y)),int(round(x));y0=max(0,yc-32);x0=max(0,xc-32);y1=min(h,yc+32);x1=min(w,xc+32)
            box=(y0,x0,y1,x1)
            for column,(key,coords,match) in enumerate(methods):
                px,py=column*152+10,row*116+42
                patch.paste(raster.crop((x0,y0,x1,y1)),(px,py))
                draw_marks(pd,coords,truth,match,origin=(py-y0,px-x0),box=box)
                pd.text((column*152+8,row*116+112),f"y{y0}:{y1},x{x0}:{x1}",font=small,fill="black")
            patch_ledger.append(dict(image_token=token,patch_index=row,y0=y0,x0=x0,y1=y1,x1=x1,policy="lexicographic_error_union_first_four"))
        if not picks:pd.text((10,45),"No reference-relative errors",font=small,fill="black")
        patch.save(out/"patches"/(token+".png"))
        manifest.append(dict(file=f"data/raw/tif/{sid}.tif",output_name=token,sample_id=sid,subset=subset,batch=token,
             height=h,width=w,um_per_pixel="",scale_bar="none_pixel_coordinates_only",display_min=low,display_max=high,gamma=1,lut="gray",
             raw_sha256=sha256(root/f"data/raw/tif/{sid}.tif"),font_file=font_path,display_sha256=sha256(out/"display_native"/(token+".png"))))
        if number%5==0:ctx.status(activity=f"frozen adapted-model review {number}/60")
    write_csv(out/"review_manifest.csv",manifest)
    write_csv(out/"panel_metrics_private.csv",panels)
    write_csv(out/"error_candidates.csv",errors,["image_token","method","type","y","x","index","human_status"])
    if not (out/"human_review.csv").exists():write_csv(out/"human_review.csv",human)
    write_csv(out/"patch_manifest.csv",patch_ledger,["image_token","patch_index","y0","x0","y1","x1","policy"])
    stability,frozen_checks=cross_seed_stability(root,ctx.corpus.splits["valid"],keys)
    write_csv(out/"cross_seed_stability_private.csv",stability)
    dump(out/"frozen_prediction_checks.json",frozen_checks)
    gallery=['<!doctype html><meta charset="utf-8"><title>Blinded frozen-model review</title>',
        '<style>body{font-family:Arial;background:#eee;margin:24px}img{max-width:100%;image-rendering:pixelated}section{background:white;padding:12px;margin:16px 0}summary{cursor:pointer}</style>',
        '<h1>Frozen-model review</h1><p>Reference-relative errors are review candidates, not confirmed label errors. Native pixels; no inferred physical scale. Use human_review.csv.</p>']
    for number in range(1,61):
        token=f"I{number:03d}"
        gallery.append(f'<section><h2>{token}</h2><img loading="lazy" src="full_images/{token}.png"><details><summary>Native-pixel patches</summary><img loading="lazy" src="patches/{token}.png"></details></section>')
    (out/"index.html").write_text("\n".join(gallery),encoding="utf-8")
    result=dict(passed=True,queue_images=60,train_images=38,valid_images=22,method_panels=len(panels),full_images=60,patch_sheets=60,
       representative_seed=42,spotiflow_reference="finetuned_CUDA_primary",error_candidates=len(errors),human_review_completed=False,
       labels_modified=False,test_arrays_loaded=False,physical_scale_bar=False,resampling="none; native pixel rasters",
       font=font_path,cross_seed_alignment="spatial matching of FP coordinates and stable reference indices, not pred_index")
    if len(panels)!=180:raise AssertionError("Review panel count failed")
    dump(out/"checks.json",result)
    ctx.task("finetuned_error_review","completed",**result)
    return result
