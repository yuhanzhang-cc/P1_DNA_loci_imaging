"""Generate an offline image-based form from the existing anonymous PNG packet."""
from pathlib import Path
import base64,json
from .common import fresh,table,sha,canonical,write,inside

FIELDS=["anonymous_id","raw_file","marked_file","candidate_is_real_spot","number_of_spots","background_description",
 "suggested_center_y_patch_px","suggested_center_x_patch_px","review_confidence","reviewer","notes",
 "adjudicated_detection_label","adjudication_phase","synthetic_fixture","review_packet_id"]
FORBIDDEN={"method","seed","sample_id","stratum","event_id","true_index","pred_index"}


def validate_answers(rows,expected,complete=True,synthetic=False,expected_packet_id=None):
    ids=[r["anonymous_id"] for r in rows]
    if len(ids)!=60 or len(set(ids))!=60 or set(ids)!=set(expected) or len(expected)!=60:raise ValueError("Expected exactly 60 unique anonymous cases")
    missing=[];reviewers=set();counts=dict(yes=0,no=0,uncertain=0,low=0)
    for r in rows:
        if FORBIDDEN & r.keys():raise ValueError("Unblinding columns in answer form")
        if expected_packet_id is not None and r.get("review_packet_id")!=expected_packet_id:raise ValueError("Answers belong to a different image-review packet")
        if r.get("synthetic_fixture","False").lower() not in ("true","false") or (r.get("synthetic_fixture","False").lower()=="true")!=synthetic:
            raise ValueError("Synthetic versus real provenance differs")
        for field in ("candidate_is_real_spot","number_of_spots","review_confidence","reviewer"):
            if not r.get(field,"").strip():missing.append(r["anonymous_id"]+":"+field)
        presence=r.get("candidate_is_real_spot","");confidence=r.get("review_confidence","")
        if presence and presence not in ("yes","no","uncertain"):raise ValueError("Invalid spot answer")
        if confidence and confidence not in ("low","medium","high"):raise ValueError("Invalid confidence")
        if presence:counts[presence]+=1
        counts["low"]+=int(confidence=="low")
        count=r.get("number_of_spots","")
        if count and count!="uncertain" and (not count.isdigit() or int(count)<0):raise ValueError("Invalid spot count")
        y,x=r.get("suggested_center_y_patch_px",""),r.get("suggested_center_x_patch_px","")
        if bool(y)!=bool(x):raise ValueError("Incomplete suggested center")
        if y and not (0<=float(y)<64 and 0<=float(x)<64):raise ValueError("Center outside native patch")
        if r.get("adjudicated_detection_label") or r.get("adjudication_phase"):raise ValueError("Initial blind answers must not assert annotation-based TP/FP")
        if r.get("reviewer","").strip():reviewers.add(r["reviewer"].strip())
    if len(reviewers)>1:raise ValueError("One reviewer per answer file")
    if complete and missing:raise ValueError(f"Incomplete answers: {len(missing)} required fields")
    return dict(cases=60,complete=not missing,missing_fields=len(missing),reviewers=sorted(reviewers),answer_counts=counts,
        human_review_completed=not synthetic and not missing,synthetic_fixture=synthetic,
        interpretation="Spot presence is not one-to-one detection TP; review answers do not modify labels")


def build(public,output,synthetic=False):
    public=Path(public);rows=table(public/"blind_audit_manifest.csv")
    if len(rows)!=60 or any(FORBIDDEN & r.keys() for r in rows):raise ValueError("Wrong or unblinded input packet")
    cases=[];hashes={}
    for row in rows:
        item={k:row[k] for k in ("anonymous_id","raw_file","marked_file")}
        if not row["anonymous_id"].isalnum():raise ValueError("Unsafe anonymous ID")
        for view,field in (("raw","raw_file"),("marked","marked_file")):
            p=inside(public,row[field]);payload=p.read_bytes()
            if not payload.startswith(b"\x89PNG\r\n\x1a\n"):raise ValueError("Review images must be PNG")
            item[view]="data:image/png;base64,"+base64.b64encode(payload).decode();hashes[row[field]]=sha(p)
        cases.append(item)
    if len({c["anonymous_id"] for c in cases})!=60:raise ValueError("Duplicate IDs")
    identity=[{k:c[k] for k in ("anonymous_id","raw_file","marked_file")} for c in cases]
    packet=dict(schema="anonymous_review_ui_v2",packet_id=canonical(dict(cases=identity,image_hashes=hashes)),synthetic_fixture=synthetic,cases=cases,fields=FIELDS)
    output=fresh(output);code=Path(__file__).parent
    html=(code/"reviewer.html").read_text(encoding="utf-8").replace("__PACKET__",json.dumps(packet,ensure_ascii=False).replace("<","\\u003c"))
    html=html.replace("__SCRIPT__",(code/"reviewer.js").read_text(encoding="utf-8"))
    (output/"index.html").write_bytes(html.encode())
    (output/"decision.html").write_bytes((code/"decision.html").read_bytes())
    write(output/"packet_checks.json",dict(cases=60,packet_id=packet["packet_id"],images_embedded=120,
        source_mapping_included=False,human_answers_prefilled=False,network_required=False,synthetic_fixture=synthetic,image_sha256=hashes))
    return dict(path=str(output/"index.html"),packet_id=packet["packet_id"],cases=60)
