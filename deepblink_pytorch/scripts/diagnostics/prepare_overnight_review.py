"""No-training FP reconciliation, blinded review packet and overnight readiness.

The reviewer ZIP deliberately excludes model/stratum identities and numeric crops.
Geometry is a reproducible proxy; biological classifications remain human inputs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import random
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from peak_offset_diag import CODE_ROOT, REL_OUTPUT, load_validation
from research_v4.common import rows, read_json, sha256, write_csv, dump
from deepblink_torch.evaluation import match_coordinates

SEED = 20261102
METHODS = ("R0", "H0", "G0")
NEAR_RADIUS = 3.
PATCH_SIZE = 64
DISPLAY_LOW, DISPLAY_HIGH = -1., 8.  # whole-original-image z-score; all patches identical window


def load_exports(root, validation):
    base = root / "outputs/next_stage_v4_recipe_20261101"
    queue = dict(R0=base / "extracted_r0/outputs/research_v4_cuda/baseline_r0/R0_cuda_reference",
                 H0=base / "extracted_h0/outputs/research_v4_cuda/baseline_h0/H0_cuda_reference",
                 G0=base / "08_returned_gpu/extracted/outputs/research_v4_cuda/recipe/G0_ref_cuda")
    cache, inputs = {}, {}
    for method in METHODS:
        for seed in (42,43,44):
            folder = queue[method] / f"seed_{seed}"
            for name in ("prediction_points.csv", "matches.csv", "per_image_metrics.csv", "metrics.json"):
                p = folder / "primary" / name
                inputs[str(p)] = sha256(p)
            prediction_rows = rows(folder / "primary/prediction_points.csv")
            match_rows = rows(folder / "primary/matches.csv")
            image_metrics = rows(folder / "primary/per_image_metrics.csv")
            for sid, data in validation.items():
                pr = [r for r in prediction_rows if r["sample_id"] == sid]
                coords = np.asarray([[float(r["y"]), float(r["x"])] for r in pr], np.float32).reshape(-1,2)
                match = match_coordinates(coords, data["truth"], NEAR_RADIUS)
                expected = next(r for r in image_metrics if r["sample_id"] == sid)
                if (match.tp,match.fp,match.fn) != tuple(int(float(expected[k])) for k in ("tp","fp","fn")):
                    raise ValueError("Accepted export differs")
                recorded = {(int(r["pred_index"]),int(r["true_index"])) for r in match_rows if r["sample_id"] == sid}
                if recorded != set(zip(map(int,match.matched_pred_indices),map(int,match.matched_true_indices))):
                    raise ValueError("Accepted match identity differs")
                cache[(method,seed,sid)] = dict(coords=coords, rows=pr, match=match,
                     tp_map={int(ti):int(pi) for pi,ti in zip(match.matched_pred_indices,match.matched_true_indices)},
                     fp_indices=match.unmatched_pred_indices.astype(int))
    return cache, inputs, queue


def spatial_fp_sets(h0, g0, radius):
    hi, gi = h0["fp_indices"], g0["fp_indices"]
    linked = match_coordinates(h0["coords"][hi], g0["coords"][gi], radius)
    shared = {int(hi[a]):int(gi[b]) for a,b in zip(linked.matched_pred_indices, linked.matched_true_indices)}
    return shared, [int(hi[i]) for i in linked.unmatched_pred_indices], [int(gi[i]) for i in linked.unmatched_true_indices]


def geometric_tag(point, truth, match):
    distance = np.linalg.norm(truth-point, axis=1)
    nearby = np.flatnonzero(distance <= 3.)
    nearest = int(np.argmin(distance))
    if len(nearby) == 1:
        label = "nearby_duplicate_proxy"
    elif len(nearby) > 1:
        label = "assignment_conflict_proxy"
    else:
        label = "other_requires_visual_review"
    assigned = {int(t) for t in match.matched_true_indices}
    return dict(geometry_class=label, nearby_truth_indices="|".join(map(str, nearby)),
                nearest_truth_index=nearest, nearest_truth_distance_px=float(distance[nearest]),
                nearby_truth_already_assigned=all(int(i) in assigned for i in nearby) if len(nearby) else False,
                old_proxy="duplicate_or_assignment_conflict" if len(nearby) else "near_reference_3to6px" if distance[nearest] <= 6 else "far_from_reference_gt6px")


def reconcile(cache, validation, out):
    events, pairs, sensitivity, g0_only = [], [], [], []
    old_counts = {m:dict(fp=0,duplicate_or_assignment_conflict=0,near_reference_3to6px=0,far_from_reference_gt6px=0) for m in ("H0","G0")}
    for seed in (42,43,44):
        for sid, data in validation.items():
            h0,g0=cache[("H0",seed,sid)],cache[("G0",seed,sid)]
            for radius in (1.,2.,3.):
                shared, honly, gonly=spatial_fp_sets(h0,g0,radius)
                sensitivity.append(dict(seed=seed,sample_id=sid,radius_px=radius,h0_fp=len(h0["fp_indices"]),g0_fp=len(g0["fp_indices"]),
                                        shared=len(shared),h0_only=len(honly),g0_only=len(gonly),net=len(honly)-len(gonly)))
            shared,honly,gonly=spatial_fp_sets(h0,g0,3.)
            for i,j in shared.items():
                pairs.append(dict(seed=seed,sample_id=sid,h0_pred_index=i,g0_pred_index=j,distance_px=float(np.linalg.norm(h0["coords"][i]-g0["coords"][j]))))
            for i in gonly:
                g0_only.append(dict(seed=seed,sample_id=sid,g0_pred_index=i,y=float(g0["coords"][i,0]),x=float(g0["coords"][i,1])))
            for method,item in (("H0",h0),("G0",g0)):
                for i in item["fp_indices"]:
                    tag=geometric_tag(item["coords"][i],data["truth"],item["match"])
                    old_counts[method]["fp"]+=1;old_counts[method][tag["old_proxy"]]+=1
                    if method=="H0":
                        point=item["coords"][i]
                        events.append(dict(event_id=f"H0_s{seed}_{sid}_p{int(i)}",seed=seed,sample_id=sid,pred_index=int(i),
                             y=float(point[0]),x=float(point[1]),score=float(item["rows"][i]["p"]),
                             h0_only_vs_g0=int(i) in honly,shared_g0_pred_index=shared.get(int(i)),
                             geometry_rule_confidence="deterministic_from_frozen_annotations",
                             biological_class="pending_human_blind_review",biological_confidence="unassessed",
                             human_class="",human_confidence="",**tag))
    if old_counts["H0"]["fp"]-old_counts["G0"]["fp"] != 65:
        raise ValueError("The reported net FP difference does not reconcile")
    if old_counts["H0"]["duplicate_or_assignment_conflict"]-old_counts["G0"]["duplicate_or_assignment_conflict"] != 59:
        raise ValueError("The reported proxy difference does not reconcile")
    write_csv(out / "h0_extra_fp_breakdown.csv",events)
    write_csv(out / "fp_shared_pairs.csv",pairs,["seed","sample_id","h0_pred_index","g0_pred_index","distance_px"])
    write_csv(out / "g0_only_fp.csv",g0_only,["seed","sample_id","g0_pred_index","y","x"])
    write_csv(out / "fp_spatial_sensitivity.csv",sensitivity)
    totals = {k:sum(r[k] for r in sensitivity if r["radius_px"]==3) for k in ("h0_fp","g0_fp","shared","h0_only","g0_only","net")}
    totals.update(old_proxy_counts=old_counts,geometry_class_counts={c:sum(r["geometry_class"]==c for r in events) for c in
                  ("nearby_duplicate_proxy","assignment_conflict_proxy","other_requires_visual_review")},
                  net_65_is_not_a_set_of_65_distinct_candidates=True,human_review_completed=False)
    dump(out / "fp_reconciliation.json",totals)
    return events,totals


def candidate_pools(cache, validation, fp_events):
    pools={"common_tp":[],"h0_only_tp":[],"h0_only_fp":[]}
    for seed in (42,43,44):
        for sid,data in validation.items():
            maps={m:cache[(m,seed,sid)]["tp_map"] for m in METHODS}
            shared=set(maps["H0"]) & set(maps["R0"]) & set(maps["G0"])
            only=set(maps["H0"]) - (set(maps["R0"]) | set(maps["G0"]))
            for stratum,indices in (("common_tp",shared),("h0_only_tp",only)):
                for ti in sorted(indices):
                    y,x=map(float,data["truth"][ti])
                    pools[stratum].append(dict(stratum=stratum,seed=seed,sample_id=sid,true_index=ti,y=y,x=x,
                                              method="",pred_index=None,event_id=""))
    for event in fp_events:
        if event["h0_only_vs_g0"]:
            pools["h0_only_fp"].append(dict(stratum="h0_only_fp",seed=event["seed"],sample_id=event["sample_id"],
                 true_index=None,y=event["y"],x=event["x"],method="H0",pred_index=event["pred_index"],event_id=event["event_id"]))
    return pools


def sample_cases(pools, count=20):
    rng=random.Random(SEED)
    chosen=[];counts={}
    # Rarer strata are filled first. Sampling identities remain private.
    for stratum in ("h0_only_tp","h0_only_fp","common_tp"):
        pool=sorted(pools[stratum],key=lambda r:(r["sample_id"],r["seed"],r["y"],r["x"]))
        rng.shuffle(pool)
        picks=[]
        # Avoid reviewing one physical location repeatedly across seeds/strata.
        for row in pool:
            if any(row["sample_id"]==r["sample_id"] and np.hypot(row["y"]-r["y"],row["x"]-r["x"])<16 for r in chosen+picks):
                continue
            picks.append(row)
            if len(picks)==count:break
        chosen+=picks
        counts[stratum]=dict(requested=count,event_pool=len(pool),selected=len(picks))
    rng.shuffle(chosen)
    common_methods=list(METHODS)*((count+2)//3)
    rng.shuffle(common_methods)
    for row in chosen:
        if row["stratum"]=="common_tp":row["method"]=common_methods.pop()
        elif row["stratum"]=="h0_only_tp":row["method"]="H0"
    return chosen,counts


def crop_patch(image, y, x):
    cy,cx=int(round(y)),int(round(x))
    y0,x0=cy-PATCH_SIZE//2,cx-PATCH_SIZE//2
    h,w=image.shape
    y1,x1=y0+PATCH_SIZE,x0+PATCH_SIZE
    patch=np.full((PATCH_SIZE,PATCH_SIZE),np.nan,np.float32)
    a,b,c,d=max(0,y0),min(h,y1),max(0,x0),min(w,x1)
    patch[a-y0:b-y0,c-x0:d-x0]=image[a:b,c:d]
    return patch,(y0,x0,y1,x1)


def prepare_blind_packet(out, cases, cache, validation, counts):
    import tifffile
    public=out / "blind_audit_public"
    private=out / "private"
    if public.exists() or private.exists():raise FileExistsError("Preserve the existing blind packet; use a new output subdirectory")
    (public / "patches").mkdir(parents=True);(private / "quantitative_patches").mkdir(parents=True)
    source_manifest=[]; normalized={}
    for sid,data in validation.items():
        image=data["image"]
        mean,std=float(image.mean()),float(image.std())
        normalized[sid]=(image-mean)/std if std else np.zeros_like(image)
        source_manifest.append(dict(sample_id=sid,height=image.shape[0],width=image.shape[1],mean=mean,std=std,
             display_min_z=DISPLAY_LOW,display_max_z=DISPLAY_HIGH,gamma=1.,lut="gray",resampled=False,
             fraction_display_clipped_low=float((normalized[sid]<DISPLAY_LOW).mean()),fraction_display_clipped_high=float((normalized[sid]>DISPLAY_HIGH).mean())))
    manifest=[];key=[];html=["<!doctype html><meta charset='utf-8'><title>Anonymous candidate review</title>",
        "<style>body{font:16px Arial}article{display:inline-block;margin:16px}img{width:256px;height:256px;image-rendering:pixelated}</style>",
        "<h1>Anonymous candidate review</h1><p>Raw view (left); candidate ring (right). Native pixels, identical display rule. Fill the CSV before opening the source key.</p>"]
    for number,case in enumerate(cases,1):
        aid=f"A{number:04d}"
        sid,seed,method=case["sample_id"],case["seed"],case["method"]
        item=cache[(method,seed,sid)]
        pi=case["pred_index"] if case["true_index"] is None else item["tp_map"][case["true_index"]]
        candidate=item["coords"][pi]
        raw,bounds=crop_patch(validation[sid]["image"],case["y"],case["x"])
        normalized_patch,_=crop_patch(normalized[sid],case["y"],case["x"])
        scaled=(normalized_patch-DISPLAY_LOW)/(DISPLAY_HIGH-DISPLAY_LOW)
        raster=np.full(raw.shape,127,np.uint8)
        valid=np.isfinite(raw)
        raster[valid]=np.uint8(np.clip(scaled[valid],0,1)*255)
        image=Image.fromarray(raster)
        raw_name=f"patches/{aid}_raw.png";marked_name=f"patches/{aid}_marked.png"
        image.save(public/raw_name)
        marked=image.convert("RGB");draw=ImageDraw.Draw(marked)
        px,py=float(candidate[1]-bounds[1]),float(candidate[0]-bounds[0])
        draw.ellipse((px-4,py-4,px+4,py+4),outline="#00CFFF",width=1)
        marked.save(public/marked_name)
        tifffile.imwrite(private/"quantitative_patches"/(aid+".tif"),raw,metadata=None)
        manifest.append(dict(anonymous_id=aid,raw_file=raw_name,marked_file=marked_name,
             candidate_is_real_spot="",number_of_spots="",background_description="",suggested_center_y_patch_px="",
             suggested_center_x_patch_px="",review_confidence="",reviewer="",notes=""))
        key.append(dict(case,anonymous_id=aid,method=method,pred_index=int(pi),candidate_y=float(candidate[0]),candidate_x=float(candidate[1]),
             crop_y0=bounds[0],crop_x0=bounds[1],crop_y1=bounds[2],crop_x1=bounds[3],
             raw_png_sha256=sha256(public/raw_name),marked_png_sha256=sha256(public/marked_name)))
        html.append(f"<article><h2>{aid}</h2><img src='{raw_name}' alt='{aid} raw'><img src='{marked_name}' alt='{aid} candidate ring'></article>")
    write_csv(public/"blind_audit_manifest.csv",manifest)
    write_csv(private/"blind_audit_source_key.csv",key)
    write_csv(private/"source_image_manifest.csv",source_manifest)
    (public/"index.html").write_text("\n".join(html),encoding="utf-8")
    text="""# Anonymous candidate review

Review each anonymous ID using the raw image and the same candidate-ring view.
The ring center is the frozen predicted coordinate, not a new annotation.
Fill blind_audit_manifest.csv: candidate_is_real_spot = yes/no/uncertain;
number_of_spots = integer or uncertain; review_confidence = low/medium/high.
Suggested center coordinates, if supplied, are in native 64x64 patch pixels.
Do not turn these edits into training labels or tune detection thresholds.

Display: normalize each full original image by its own population mean/SD,
then apply fixed z-score window [-1,8], gamma1 and grayscale to every crop.
No local patch contrast fitting, resampling or physical scale bars. Quantitative
crops and acquisition/source/model/stratum identities remain outside this ZIP.
Native PNGs are 64x64. The HTML uses a declared nearest-pixel CSS enlargement.
Gray padding marks any portion outside the original FOV. Raw pixels are preserved.
The packet covers only selected development examples, not an unbiased prevalence
sample, a complete annotation audit or independent biological validation.

Keep the private source key closed until all review answers are saved and locked.
If two reviewers disagree, preserve both original answers for later adjudication.
"""
    (public/"README.md").write_text(text,encoding="utf-8")
    target=out/"blind_audit_public.zip"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in sorted(public.rglob("*")):
            if p.is_file():archive.write(p,p.relative_to(public).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any("private" in n or "source_key" in n for n in archive.namelist())
    target.with_name(target.name+".sha256").write_bytes((sha256(target)+"  "+target.name+"\n").encode("ascii"))
    dump(out/"blind_audit_sampling.json",dict(seed=SEED,counts=counts,patch_size=PATCH_SIZE,
         minimum_between_case_center_distance_same_image_px=16,blinding="Model/seed/stratum/source keys excluded from public ZIP",
         review_complete=False,no_human_labels_fabricated=True))
    return key


def source_range_check(peak_csv,out):
    data=[r for r in rows(peak_csv) if r["method"]=="H0" and r["cpu_source_linked"]=="True"]
    result=[]
    for r in data:
        target=np.asarray([float(r["true_y"]),float(r["true_x"])])
        anchor=np.asarray([float(r["peak_anchor_y"]),float(r["peak_anchor_x"])])
        required=target-anchor
        lower_bound=float(np.linalg.norm(np.maximum(0,np.maximum(-required,required-1))))
        observed=float(np.hypot(float(r["dy_px"]),float(r["dx_px"])))
        if observed+1e-4<lower_bound:raise ValueError("Observed residual violates source-offset representability bound")
        result.append(dict(sample_id=r["sample_id"],seed=r["seed"],true_index=r["true_index"],
             required_offset_y_px=float(required[0]),required_offset_x_px=float(required[1]),
             target_reachable_in_closed_0_1_offset_box=bool(((required>=0)&(required<=1)).all()),
             unavoidable_error_lower_bound_px=lower_bound,observed_error_px=observed))
    write_csv(out/"peak_offset_source_range.csv",result)
    summary=dict(events=len(result),unreachable_target_events=sum(not r["target_reachable_in_closed_0_1_offset_box"] for r in result),
         minimum_bound_mean_px=float(np.mean([r["unavoidable_error_lower_bound_px"] for r in result])),
         bound_definition="Distance of truth from source_anchor+[0,1]^2, generous closed-box bound for float32 sigmoid outputs",
         no_prediction_coordinates_changed=True)
    dump(out/"peak_offset_source_range_summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--patches-per-stratum",type=int,default=20)
    args=parser.parse_args()
    root=args.data_root.resolve();allowed=(root/REL_OUTPUT).resolve()
    out=args.output.resolve() if args.output else allowed/"overnight_ready"
    if not out.is_relative_to(allowed):raise ValueError("Output must remain under diagnostics")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"readiness_provenance.json").exists():raise FileExistsError("Use a new diagnostics subdirectory for a new packet")
    sealed=root/"outputs/next_stage_v4_recipe_20261101/04_bundles/ce2d1821ceec/p1_v4_training"
    validation,keys=load_validation(sealed)
    cache,inputs,queues=load_exports(root,validation)
    peak_csv=allowed/"peak_offset_diag.csv";inputs[str(peak_csv)]=sha256(peak_csv)
    inputs[str(sealed/"assets/development_only.npz")]=sha256(sealed/"assets/development_only.npz")
    events,counts=reconcile(cache,validation,out)
    cases,sampling=sample_cases(candidate_pools(cache,validation,events),args.patches_per_stratum)
    key=prepare_blind_packet(out,cases,cache,validation,sampling)
    support=source_range_check(peak_csv,out)
    for p,h in inputs.items():
        if sha256(p)!=h:raise ValueError("Immutable input changed")
    dump(out/"readiness_provenance.json",dict(source_sha256=sha256(__file__),seed=SEED,inputs=inputs,
         archive_keys_read=keys,test_arrays_read=False,new_optimizer_updates=0,architecture_modified=False,
         fp=counts,source_range=support,patches=len(key),human_review_pending=True))
    print(json.dumps(dict(fp=counts,source_range=support,sampling=sampling,patches=len(key)),indent=2))


if __name__=="__main__":main()
