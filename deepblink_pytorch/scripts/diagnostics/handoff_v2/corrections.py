"""Owner-only image correction materials after completed blind answers are locked.

No relabeling is automatic. Export native raw arrays and editable Fiji point
selections; preserve the original TrackMate project and all old training labels.
"""
from pathlib import Path
import json,re
from .common import ROOT,REVIEW,read,table,sha,write,fresh,csv_write
from .reviewer import validate_answers

REFERENCE="https://imagej.net/ij/developer/macro/functions2.html"


def make_macro(sid,truth,candidates):
    def arr(values):return "newArray("+",".join(format(float(v),".9g") for v in values)+")"
    lines=['// Owner-only post-blind visual review; original labels are never saved over.','requires("1.52i");',
        'base=getDirectory("Choose this correction package folder");',f'open(base+"images/{sid}.tif");',
        'Overlay.clear;']
    for item in candidates:
        y,x=item[:2]
        lines += [f'makeOval({float(x)-4:.9g},{float(y)-4:.9g},8,8);','Overlay.addSelection("cyan",1);']
        if len(item)>2:
            label=str(item[2])
            if not re.fullmatch(r"A\d{4}",label):raise ValueError("Unsafe case label")
            lines += ['setColor("cyan");',f'Overlay.drawString("{label}",{float(x)+5:.9g},{float(y):.9g});']
    if len(truth):lines += [f'makeSelection("point small green cross",{arr([p[1] for p in truth])},{arr([p[0] for p in truth])});']
    lines += ['Overlay.show;','setTool("multipoint");',
        'waitForUser("Post-blind review", "Green editable points: original annotations. Cyan circles: candidates for review. Edit points only if justified, or use your original TrackMate project. Click OK only when ready to export a NEW revision; Cancel leaves originals unchanged.");',
        'if (selectionType()!=10) exit("A multipoint selection is required; nothing exported.");',
        'getSelectionCoordinates(xs,ys);',
        'destination=getDirectory("Choose a NEW revision output folder")+"'+sid+'_reviewed.csv";',
        'if (File.exists(destination)) exit("Existing file protected. Choose a new revision filename.");',
        'text="sample_id,point_index,y,x\\n";',
        'for(i=0;i<xs.length;i++) text=text+"'+sid+',"+i+","+d2s(ys[i],6)+","+d2s(xs[i],6)+"\\n";',
        'File.saveString(text,destination);']
    return "\n".join(lines)+"\n"


def build(responses,source_key,data_root,output):
    source_key=Path(source_key);key=table(source_key);answers=table(responses)
    validate_answers(answers,[r["anonymous_id"] for r in key],True,False)
    # This function is explicitly post-blind: opening the key is recorded.
    mapping={r["anonymous_id"]:r for r in key}
    selected={r["anonymous_id"] for r in answers if r["candidate_is_real_spot"] in ("no","uncertain") or r["review_confidence"]=="low" or r.get("suggested_center_y_patch_px") or r.get("notes") or
        r["candidate_is_real_spot"]=="yes" and mapping[r["anonymous_id"]]["stratum"]=="h0_only_fp"}
    cases=[r for r in key if r["anonymous_id"] in selected]
    return export_materials(cases,key,source_key,data_root,output,sha(responses))


def prepare(source_key,data_root,output):
    """Prebuild all owner-only overlays; never manufacture review answers."""
    key=table(source_key)
    if len(key)!=60 or len({r["anonymous_id"] for r in key})!=60:raise ValueError("Expected the fixed 60-case owner key")
    return export_materials(key,key,source_key,data_root,output,None)


def export_materials(cases,key,source_key,data_root,output,responses_hash):
    from diagnostics.peak_offset_diag import load_validation
    sealed=Path(data_root)/"outputs/next_stage_v4_recipe_20261101/04_bundles/ce2d1821ceec/p1_v4_training"
    validation,_=load_validation(sealed)
    output=fresh(output);(output/"images").mkdir();(output/"macros").mkdir()
    import numpy as np,tifffile
    for sid in sorted({r["sample_id"] for r in cases}):
        if not re.fullmatch(r"sample_\d+",sid):raise ValueError("Unsafe sample name")
        item=validation[sid];image=np.asarray(item["image"],np.float32)
        tifffile.imwrite(output/"images"/(sid+".tif"),image,metadata=None)
        if not np.array_equal(tifffile.imread(output/"images"/(sid+".tif")),image):raise AssertionError("Raw image export changed pixels")
        subset=[r for r in cases if r["sample_id"]==sid]
        macro=make_macro(sid,item["truth"],[(float(r["candidate_y"]),float(r["candidate_x"]),r["anonymous_id"]) for r in subset])
        (output/"macros"/(sid+".ijm")).write_bytes(macro.encode())
        csv_write(output/(sid+"_original_points.csv"),[dict(sample_id=sid,point_index=i,y=float(y),x=float(x)) for i,(y,x) in enumerate(item["truth"])],["sample_id","point_index","y","x"])
    csv_write(output/"post_blind_queue.csv",cases,list(key[0]))
    write(output/"provenance.json",dict(responses_sha256=responses_hash,source_key_sha256=sha(source_key),human_review_completed=responses_hash is not None,
        selected_cases=len(cases),selected_images=len({r["sample_id"] for r in cases}),post_blind_only=True,
        raw_pixels_verified=True,labels_modified=False,training_inputs_modified=False,
        fiji_macro_reference=REFERENCE,fiji_runtime_tested=False,trackmate_project_modified=False,
        note="Requires a separate dataset/protocol revision before corrected labels may enter training"))
    (output/"README.md").write_bytes(('Owner-only post-blind materials. Open a macros/*.ijm in Fiji via Plugins > Macros > Run; choose this package folder when prompted. Native float32 TIFFs preserve all pixels. Editable green points are the original annotations; cyan candidate rings are an overlay. Save a NEW reviewed CSV only. You may instead edit your preserved original TrackMate project and export its reviewed coordinates. This is not a TrackMate XML converter. Macro source follows official ImageJ APIs but Fiji execution is not yet verified here. Do not infer true subpixel accuracy from manual dragging.\n').encode())
    return dict(path=str(output),cases=len(cases),images=len({r["sample_id"] for r in cases}),labels_modified=False)


def check_revision(original,reviewed,shape):
    import numpy as np
    old=np.asarray(original,float).reshape(-1,2);new=np.asarray(reviewed,float).reshape(-1,2)
    if not np.isfinite(new).all() or np.any(new<0) or np.any(new>=shape):raise ValueError("Reviewed points outside original FOV")
    return dict(original_points=len(old),reviewed_points=len(new),point_count_delta=len(new)-len(old),
        exact_original=np.array_equal(old,new),approved_for_training=False,
        remaining="Reviewer/reason provenance and an explicit dataset/protocol revision are required; nothing is applied automatically")
