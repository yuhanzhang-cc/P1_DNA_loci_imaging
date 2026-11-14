"""Verify anonymous crops, net/set counts and frozen-source integrity offline."""
from pathlib import Path
import argparse
import hashlib
import sys
import zipfile
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_overnight_review import REL_OUTPUT, load_validation, crop_patch, DISPLAY_LOW, DISPLAY_HIGH
from research_v4.common import read_json, rows, sha256, dump


def main():
    import tifffile
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    root=args.data_root.resolve();allowed=(root/REL_OUTPUT).resolve()
    out=args.output.resolve() if args.output else allowed/"overnight_ready"
    if not out.is_relative_to(allowed):raise ValueError("Output outside requested diagnostics directory")
    provenance=read_json(out/"readiness_provenance.json")
    for p,digest in provenance["inputs"].items():
        if sha256(p)!=digest:raise ValueError("Frozen analysis input changed: "+p)
    stage1=read_json(allowed/"provenance.json")
    for p,digest in stage1["immutable_inputs"].items():
        if sha256(p)!=digest:raise ValueError("Stage-1 input or training source changed: "+p)
    sealed=root/"outputs/next_stage_v4_recipe_20261101/04_bundles/ce2d1821ceec/p1_v4_training"
    data,_=load_validation(sealed)
    public=out/"blind_audit_public"
    key=rows(out/"private/blind_audit_source_key.csv")
    form=rows(public/"blind_audit_manifest.csv")
    assert len(key)==len(form)==60
    expected_strata={x:sum(r["stratum"]==x for r in key) for x in ("common_tp","h0_only_tp","h0_only_fp")}
    assert set(expected_strata.values())=={20}
    forbidden={"method","sample_id","seed","stratum","true_index","event_id"}
    assert not forbidden & set(form[0])
    assert {r["anonymous_id"] for r in form}=={r["anonymous_id"] for r in key}
    for r in form:
        assert all(not r[k] for k in r if k not in ("anonymous_id","raw_file","marked_file")),"Reviewer answers were fabricated"
    exact_hashes={};duplicates=[];sheet=Image.new("RGB",(6*144,10*88),"white");draw=ImageDraw.Draw(sheet)
    for n,r in enumerate(key):
        aid=r["anonymous_id"];image=data[r["sample_id"]]["image"]
        raw,bounds=crop_patch(image,float(r["y"]),float(r["x"]))
        numerical=tifffile.imread(out/"private/quantitative_patches"/(aid+".tif"))
        np.testing.assert_array_equal(numerical,raw)
        normalized=(image-float(image.mean()))/float(image.std()) if image.std() else np.zeros_like(image)
        patch,_=crop_patch(normalized,float(r["y"]),float(r["x"]))
        expected=np.full(raw.shape,127,np.uint8);valid=np.isfinite(raw)
        expected[valid]=np.uint8(np.clip((patch[valid]-DISPLAY_LOW)/(DISPLAY_HIGH-DISPLAY_LOW),0,1)*255)
        raster=np.asarray(Image.open(public/f"patches/{aid}_raw.png"))
        np.testing.assert_array_equal(raster,expected)
        digest=hashlib.sha256(raster.tobytes()).hexdigest()
        if digest in exact_hashes:duplicates.append([exact_hashes[digest],aid])
        exact_hashes[digest]=aid
        assert sha256(public/f"patches/{aid}_raw.png")==r["raw_png_sha256"]
        assert sha256(public/f"patches/{aid}_marked.png")==r["marked_png_sha256"]
        marked=Image.fromarray(expected).convert("RGB");md=ImageDraw.Draw(marked)
        px=float(r["candidate_x"])-bounds[1];py=float(r["candidate_y"])-bounds[0]
        md.ellipse((px-4,py-4,px+4,py+4),outline="#00CFFF",width=1)
        np.testing.assert_array_equal(np.asarray(Image.open(public/f"patches/{aid}_marked.png")),np.asarray(marked))
        x,y=(n%6)*144,(n//6)*88
        draw.text((x+2,y+2),aid,fill="black")
        sheet.paste(Image.fromarray(raster).convert("RGB"),(x+2,y+20));sheet.paste(marked,(x+72,y+20))
    for i,a in enumerate(key):
        for b in key[i+1:]:
            if a["sample_id"]==b["sample_id"]:
                assert np.hypot(float(a["y"])-float(b["y"]),float(a["x"])-float(b["x"]))>=16
    sheet.save(out/"private/qa_contact_sheet.png")
    with zipfile.ZipFile(out/"blind_audit_public.zip") as archive:
        assert archive.testzip() is None
        names=archive.namelist()
        assert len(names)==123
        assert all(not ("private" in n or "source_key" in n or "quantitative" in n) for n in names)
        for name in names:
            assert archive.read(name)==(public/name).read_bytes()
    fp=read_json(out/"fp_reconciliation.json")
    assert fp["shared"]+fp["h0_only"]==fp["h0_fp"]==230
    assert fp["shared"]+fp["g0_only"]==fp["g0_fp"]==165
    assert fp["h0_only"]-fp["g0_only"]==65
    detail=rows(out/"h0_extra_fp_breakdown.csv")
    assert len(detail)==230 and sum(r["h0_only_vs_g0"]=="True" for r in detail)==147
    assert all(r["biological_class"]=="pending_human_blind_review" and not r["human_class"] for r in detail)
    support=rows(out/"peak_offset_source_range.csv")
    assert len(support)==914
    assert all(float(r["observed_error_px"])+1e-4>=float(r["unavoidable_error_lower_bound_px"]) for r in support)
    result=dict(passed=True,native_quantitative_crops_verified=60,display_mappings_verified=60,candidate_markings_verified=60,
         stratum_counts=expected_strata,public_zip_files=123,source_key_excluded=True,
         human_responses_blank=True,input_and_training_source_hashes_unchanged=True,exact_raster_duplicate_pairs=duplicates,
         image_qa_limit="Exact pixels/duplicates and known source-crop provenance verified; SIFT/rotated-crop screening not run: OpenCV unavailable",
         representability_bounds_verified=914,optimizer_updates=0)
    dump(out/"qa_checks.json",result)
    print(result)


if __name__=="__main__":main()
