"""Validate real answers and freeze one explicit research decision, without training."""
from pathlib import Path
import shutil,uuid
from .common import ROOT,OLD,read,table,sha,canonical,write,fresh
from .reviewer import validate_answers

METHODS=["S0","P0"]
DEFAULT_GATE=dict(minimum_common_tp=20,min_common_error_improvement_px=.10,min_f1_delta=-.005,max_near_truth_fp_delta=0)


def validate_decision(decision,synthetic=False):
    if decision.get("schema") not in ("p1_research_decision_v2","p1_research_decision_v3") or decision.get("approved") is not True:
        raise ValueError("An explicit approved research decision is required")
    if decision.get("synthetic_fixture") is not synthetic or not decision.get("approved_by","").strip() or not decision.get("approved_at"):
        raise ValueError("Reviewer/date and real-vs-synthetic provenance required")
    if decision.get("labels_need_revision") is not False:raise ValueError("Label corrections need a separate dataset revision; do not silently train on changed labels")
    if decision.get("methods")!=METHODS:raise ValueError("This paired mechanism plan requires S0 and P0")
    if decision.get("seeds")!=[42,43,44]:raise ValueError("Seed plan changed")
    gate=decision["continuation_gate"]
    if gate!=DEFAULT_GATE:raise ValueError("This revision implements only the declared continuation gate; change protocol/version for another rule")
    if not isinstance(decision.get("auto_expand_if_gate_passes"),bool):raise ValueError("Explicit continuation choice is required")
    if decision.get("review_limitations_acknowledged") is not True:raise ValueError("Acknowledge uncertain answers and development-only scope")
    if decision.get("additional_baseline_queues",[]) not in ([],["spotiflow_scratch"]):raise ValueError("Unknown additional baseline authorization")
    if decision.get('schema')!='p1_research_decision_v3' and (decision.get('additional_baseline_queues') or decision.get('completion_strategy')=='full_registered_six'):raise ValueError('New completion options require a v3 decision and a new completion kit')
    if decision.get("completion_strategy","gate_conditioned") not in ("gate_conditioned","full_registered_six"):raise ValueError("Unknown completion strategy")
    if decision.get("completion_strategy")=="full_registered_six" and decision.get("auto_expand_if_gate_passes") is not True:raise ValueError("Full registered ablation requires explicit six-run authorization")


def freeze(responses,decision_path,output,config_dir=None,synthetic=False):
    """No automatic approval: all human-owned fields must already be supplied."""
    from diagnostics.mechanism_v1.contracts import validate as validate_config,canonical as config_hash
    config_dir=Path(config_dir) if config_dir else ROOT/OLD/"configs"
    expected=read(config_dir/"blind_audit_expected_ids.json")
    packet_path=config_dir/"review_packet_binding.json"
    packet_id=read(packet_path)["packet_id"] if packet_path.exists() else None
    if not synthetic and packet_id is None:raise ValueError("Frozen review-packet identity is required")
    answers=table(responses);review=validate_answers(answers,expected,complete=True,synthetic=synthetic,expected_packet_id=packet_id)
    decision=read(decision_path);validate_decision(decision,synthetic)
    configs={m:read(config_dir/(m+".json")) for m in METHODS}
    configs["P0"]["supervision"]=decision["p0_supervision"]
    for config in configs.values():
        config['decision_status']='approved_exact_decision_and_review_hashes_in_freeze_manifest'
        if 'implemented_options' in config:
            config['implemented_options']['radius_px']='integer 1..8 supported; approved value is frozen in supervision.radius_px'
        validate_config(config)
    requested=Path(output).resolve()
    if requested.exists():raise FileExistsError("Existing decision directory is protected")
    output=fresh(requested.parent/("."+requested.name+".pending_"+uuid.uuid4().hex[:12]))
    frozen=output/"frozen_configs";frozen.mkdir()
    shutil.copyfile(responses,output/"blind_audit_responses.csv")
    shutil.copyfile(decision_path,output/"research_decision.json")
    paths={"blind_audit_responses.csv":sha(responses),"research_decision.json":sha(decision_path)}
    for method,config in configs.items():
        write(frozen/(method+".json"),config)
        approval=dict(config_sha256=config_hash(config),mechanism_approved=True,reviewer=decision["approved_by"],
            reviewed_at=decision["approved_at"],synthetic_fixture=synthetic,blind_audit_completed=True,
            blind_audit_file="blind_audit_responses.csv",blind_audit_sha256=sha(responses))
        write(output/(method+".json"),approval)
        paths["frozen_configs/"+method+".json"]=sha(frozen/(method+".json"));paths[method+".json"]=sha(output/(method+".json"))
    write(output/"review_summary.json",review);paths["review_summary.json"]=sha(output/"review_summary.json")
    marker=dict(schema="p1_frozen_decision_v2",files=paths,decision_sha256=sha(decision_path),
        configurations={m:config_hash(c) for m,c in configs.items()},synthetic_fixture=synthetic,
        human_review_completed=not synthetic,review_packet_id=packet_id,formal_training_started=False)
    write(output/"freeze_manifest.json",marker)
    # Publish only a complete freeze. Interrupted pending folders are preserved;
    # a retry can start fresh without overwriting a partly written decision.
    if output.resolve().parent!=requested.parent or not output.resolve().is_relative_to(requested.parent) or requested.exists():
        raise ValueError("Unsafe or already published decision path")
    output.rename(requested)
    return marker


def verify_frozen(folder,allow_synthetic=False):
    from .common import inside
    folder=Path(folder);marker=read(folder/"freeze_manifest.json")
    if marker["schema"]!="p1_frozen_decision_v2" or marker["synthetic_fixture"] and not allow_synthetic:raise ValueError("Synthetic decisions cannot launch formal work")
    for name,digest in marker["files"].items():
        if sha(inside(folder,name))!=digest:raise ValueError("Frozen review/decision/config was modified: "+name)
    decision=read(folder/"research_decision.json");validate_decision(decision,marker["synthetic_fixture"])
    return marker,decision
