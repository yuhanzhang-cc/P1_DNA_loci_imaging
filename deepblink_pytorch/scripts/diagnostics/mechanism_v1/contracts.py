"""Prospective contracts, separate from the historical scaffold registry."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[4]
REL="docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/implementation_20261104"
RECIPE=dict(epochs=100,updates=2200,batch_size=8,canvas=512,optimizer="Adam",lr=.001,period=5,threshold_count=31,precision="float32",augment=False)


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")


def code_hashes():
    selected=list((ROOT/"deepblink_pytorch/deepblink_torch").rglob("*.py"))
    for directory in ("research_v4","diagnostics/mechanism_v1","diagnostics/scaffolds"):
        selected+=list((ROOT/"deepblink_pytorch/scripts"/directory).glob("*.py"))
    # New handoff runtime/static UI are part of the source-bound CUDA probe.
    handoff=ROOT/"deepblink_pytorch/scripts/diagnostics/handoff_v2"
    for suffix in ("*.py","*.html","*.js"):selected+=list(handoff.glob(suffix))
    selected+=list((ROOT/"deepblink_pytorch/scripts/diagnostics/completion_v1").glob("*.py"))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(selected)}


def validate(config,require_parameters=True):
    from .adapter import supervision
    if config.get("schema")!="signed_offset_experiment_v1" or config.get("method") not in ("S0","P0"):
        raise ValueError("Unknown experiment schema/method")
    if config["recipe"]!=RECIPE or config["seeds"]!=[42,43,44] or config["expected_parameters"]!=671719:
        raise ValueError("Frozen comparison budget/capacity changed")
    if config["status"]!="not_started" or config["formal_training_counted"]:
        raise ValueError("Configuration is not a prospective experiment")
    if require_parameters:supervision(config)
    return canonical(config)


def require_review(config,approval):
    validate(config)
    if (approval.get("config_sha256")!=canonical(config) or approval.get("mechanism_approved") is not True
        or not isinstance(approval.get("reviewer"),str) or not approval["reviewer"].strip()
        or not approval.get("reviewed_at") or approval.get("synthetic_fixture") is not False):
        raise ValueError("Explicit human decision bound to this exact config required")
    if config["method"]=="P0" and (approval.get("blind_audit_completed") is not True or
        not isinstance(approval.get("blind_audit_sha256"),str) or len(approval["blind_audit_sha256"])!=64):
        raise ValueError("P0 needs completed blind-audit evidence before formal launch")


def verify_audit_file(approval,approval_root):
    """Validate completed anonymous answers, without loading an unblinding key."""
    import csv
    root=Path(approval_root).resolve();audit=(root/approval["blind_audit_file"]).resolve()
    if not audit.is_relative_to(root) or sha(audit)!=approval["blind_audit_sha256"]:raise ValueError("Blind-audit file/hash differs")
    expected_path=ROOT/"configs/blind_audit_expected_ids.json"
    if not expected_path.exists():expected_path=ROOT/REL/"configs/blind_audit_expected_ids.json"
    expected=set(read(expected_path));rows=list(csv.DictReader(audit.open(encoding="utf-8-sig",newline="")))
    if len(expected)!=60 or len(rows)!=60 or {r["anonymous_id"] for r in rows}!=expected:raise ValueError("Expected 60 anonymous review cases")
    forbidden={"method","seed","sample_id","stratum","event_id","true_index","pred_index"}
    for row in rows:
        if forbidden & row.keys() or row.get("synthetic_fixture","False") not in ("False","false"):raise ValueError("Synthetic/unblinded review evidence")
        if row.get("candidate_is_real_spot") not in ("yes","no","uncertain") or row.get("review_confidence") not in ("low","medium","high") or not row.get("reviewer","").strip():raise ValueError("Incomplete real review evidence")
        count=row.get("number_of_spots","")
        if count!="uncertain" and (not count.isdigit() or int(count)<0):raise ValueError("Incomplete/invalid spot count")
    if len({r["reviewer"] for r in rows})!=1:raise ValueError("One reviewer per completed file")
    return dict(cases=60,completed=True,synthetic_fixture=False)
