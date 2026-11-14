"""Response validation and owner-only aggregation; never modify original review files."""
from pathlib import Path
import argparse,json,itertools,zipfile
from scaffold_common import table,csv_new,save_new

FORBIDDEN={"method","seed","sample_id","stratum","event_id","true_index","pred_index"}
ANSWER_FIELDS=["anonymous_id","raw_file","marked_file","candidate_is_real_spot","number_of_spots","background_description",
 "suggested_center_y_patch_px","suggested_center_x_patch_px","review_confidence","reviewer","notes",
 "adjudicated_detection_label","adjudication_phase","synthetic_fixture"]


def validate_responses(data,key,*,complete=False,synthetic=False):
    ids=[r["anonymous_id"] for r in data];expected={r["anonymous_id"] for r in key}
    if len(key)!=60 or len(expected)!=60 or len(data)!=60 or len(set(ids))!=60 or set(ids)!=expected:raise ValueError("Exactly the 60 unique expected anonymous IDs are required")
    if any(FORBIDDEN & set(r) for r in data):raise ValueError("Private source columns leaked into response form")
    missing=[]
    for row in data:
        if (row.get("synthetic_fixture","False")=="True")!=synthetic:raise ValueError("Synthetic/real response provenance differs")
        for field in ("candidate_is_real_spot","number_of_spots","review_confidence","reviewer"):
            if not row.get(field,"").strip():missing.append(row["anonymous_id"]+":"+field)
        presence=row.get("candidate_is_real_spot","")
        if presence and presence not in ("yes","no","uncertain"):raise ValueError("Invalid spot presence answer")
        confidence=row.get("review_confidence","")
        if confidence and confidence not in ("low","medium","high"):raise ValueError("Invalid ordinal confidence")
        count=row.get("number_of_spots","")
        if count and count!="uncertain" and (not count.isdigit() or int(count)<0):raise ValueError("Invalid spot count")
        y,x=row.get("suggested_center_y_patch_px",""),row.get("suggested_center_x_patch_px","")
        if bool(y)!=bool(x):raise ValueError("Both suggested center coordinates are required")
        if y and not (0<=float(y)<64 and 0<=float(x)<64):raise ValueError("Suggested center outside native patch")
        detection=row.get("adjudicated_detection_label","")
        if detection and (detection not in ("TP","FP","uncertain") or row.get("adjudication_phase")!="post_blind_with_annotations"):
            raise ValueError("Detection TP/FP requires post-blind annotation context; presence alone is insufficient")
    if complete and missing:raise ValueError("Incomplete real reviewer answers: "+str(len(missing)))
    if len({r["reviewer"] for r in data if r.get("reviewer")})>1:raise ValueError("Each response file must identify one reviewer")
    return dict(schema_valid=True,cases=60,responses_complete=not missing,missing_required_fields=len(missing),synthetic_fixture=synthetic)


def check_public_zip(path):
    import re,struct
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts or any(k in name.lower() for k in ("source_key","private/","quantitative/")):
                raise ValueError("Private or unsafe member in public ZIP")
            allowed=name in ("README.md","index.html","blind_audit_manifest.csv") or re.fullmatch(r"patches/A\d{4}_(raw|marked)\.png",name)
            if not allowed:raise ValueError("Unexpected public ZIP member; possible source-map leak")
            if name.endswith(".csv"):
                import csv,io
                headers=next(csv.reader(io.StringIO(archive.read(name).decode("utf-8-sig"))))
                if FORBIDDEN & set(headers):raise ValueError("Mapping columns leaked into public ZIP")
            if name.endswith(".png"):
                payload=archive.read(name)
                if not payload.startswith(b"\x89PNG\r\n\x1a\n"):raise ValueError("Malformed public PNG")
                offset=8
                while offset+12<=len(payload):
                    size=struct.unpack(">I",payload[offset:offset+4])[0];kind=payload[offset+4:offset+8]
                    if kind in (b"tEXt",b"zTXt",b"iTXt",b"eXIf"):raise ValueError("Public image carries text/EXIF metadata")
                    offset+=12+size
                    if kind==b"IEND":break
    return dict(public_zip_source_key_absent=True)


def agreement(responses,key):
    index={r["anonymous_id"]:r for r in key};rows=[];cross=[]
    reviewers=[next((r["reviewer"] for r in data if r.get("reviewer")),None) for data in responses]
    if len(set(reviewers))!=len(reviewers):raise ValueError("Inter-rater comparison requires distinct reviewer IDs")
    for a,b in itertools.combinations(range(len(responses)),2):
        aa={r["anonymous_id"]:r for r in responses[a]};bb={r["anonymous_id"]:r for r in responses[b]}
        for stratum in sorted({r["stratum"] for r in key}):
            ids=[i for i in aa if index[i]["stratum"]==stratum]
            for outcome,allowed in (("candidate_is_real_spot",{"yes","no"}),("adjudicated_detection_label",{"TP","FP"})):
                comparable=[i for i in ids if aa[i].get(outcome) in allowed and bb[i].get(outcome) in allowed]
                n=len(comparable);same=sum(aa[i][outcome]==bb[i][outcome] for i in comparable)
                rows.append(dict(reviewer_pair=f"{a+1}-{b+1}",stratum=stratum,outcome=outcome,total_cases=len(ids),
                     comparable_pairs=n,missing_or_uncertain_pairs=len(ids)-n,agreements=same,agreement=same/n if n else None))
    for number,data in enumerate(responses,1):
        combinations={}
        for r in data:
            source=index[r["anonymous_id"]]
            group=(source["stratum"],source.get("auto_geometry_class","not_applicable"),r["candidate_is_real_spot"],r.get("adjudicated_detection_label","") or "not_adjudicated")
            combinations[group]=combinations.get(group,0)+1
        for group,n in sorted(combinations.items()):
            cross.append(dict(reviewer=number,stratum=group[0],auto_geometry_proxy=group[1],human_spot_presence=group[2],post_blind_detection_label=group[3],n=n))
    return rows,cross


def synthetic_example(output):
    key=[];responses=[[],[]]
    for i in range(60):
        aid=f"SIM{i+1:04d}";stratum=("common_tp","h0_only_tp","h0_only_fp")[i//20]
        key.append(dict(anonymous_id=aid,stratum=stratum,auto_geometry_class="nearby_duplicate_proxy" if i>=40 and i%2 else "other_or_not_applicable"))
        for rater in range(2):
            detection="FP" if stratum=="h0_only_fp" else "TP"
            if i%(7+rater*2)==0:detection="TP" if detection=="FP" else "FP"
            responses[rater].append(dict.fromkeys(ANSWER_FIELDS,"")|dict(anonymous_id=aid,raw_file="synthetic",marked_file="synthetic",
              candidate_is_real_spot="yes" if i%3 else "no",number_of_spots="1" if i%3 else "0",review_confidence="medium",
              reviewer=f"SYNTHETIC_REVIEWER_{rater+1}",adjudicated_detection_label=detection,
              adjudication_phase="post_blind_with_annotations",synthetic_fixture="True",notes="ENGINEERING FIXTURE ONLY, not real reviewer data"))
    for r in responses:validate_responses(r,key,complete=True,synthetic=True)
    if output:
        csv_new(output/"synthetic_key.csv",key,list(key[0]))
        for i,data in enumerate(responses,1):csv_new(output/f"synthetic_reviewer_{i}.csv",data,ANSWER_FIELDS)
    return key,responses


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["validate","aggregate"])
    p.add_argument("--responses",type=Path,nargs="+");p.add_argument("--key",type=Path);p.add_argument("--public-zip",type=Path)
    p.add_argument("--output",type=Path);p.add_argument("--require-complete",action="store_true")
    p.add_argument("--synthetic-demo",action="store_true");p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args()
    if args.synthetic_demo:key,data=synthetic_example(args.output)
    else:
        if not args.key or not args.responses:p.error("Owner key and response file(s) required")
        key=table(args.key);data=[table(path) for path in args.responses]
    checks=[validate_responses(r,key,complete=args.require_complete or args.action=="aggregate",synthetic=args.synthetic_demo) for r in data]
    leak=check_public_zip(args.public_zip) if args.public_zip else {}
    if args.action=="aggregate":
        if len(data)<2:raise ValueError("Inter-rater agreement needs at least two real reviewer files")
        summaries,cross=agreement(data,key)
        if args.output:
            csv_new(args.output/"agreement_by_stratum.csv",summaries,list(summaries[0]))
            csv_new(args.output/"automatic_proxy_crosstab.csv",cross,list(cross[0]))
            save_new(args.output/"provenance.json",dict(synthetic_fixture=args.synthetic_demo,real_human_review_complete=False if args.synthetic_demo else True,
               interpretation="Spot presence is not detection TP; geometry proxy is not biological ground truth",no_label_or_model_changes=True))
    print(json.dumps(dict(passed=True,dry_run=True,checks=checks,**leak,synthetic_demo=args.synthetic_demo,
                         real_human_review_completed=False if args.synthetic_demo else all(r["responses_complete"] for r in checks))))


if __name__=="__main__":main()
