"""Synthetic handoff tests; no real review, training or GPU execution."""
from pathlib import Path
import copy,csv,gzip,io,json,sys,tarfile
import pytest
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/"deepblink_pytorch"),str(ROOT/"deepblink_pytorch/scripts")]
from diagnostics.handoff_v2 import reviewer,freeze,archive,gate,report
from diagnostics.handoff_v2.common import OLD,read,write,sha,csv_write


def answers():
    return [dict.fromkeys(reviewer.FIELDS,"")|dict(anonymous_id=f"SIM{i:04d}",raw_file="synthetic.png",marked_file="synthetic.png",
        candidate_is_real_spot="yes",number_of_spots="1",review_confidence="medium",reviewer="SYNTHETIC_QA",synthetic_fixture="True") for i in range(60)]


def decision():
    return dict(schema="p1_research_decision_v2",approved=True,approved_by="SYNTHETIC_QA",approved_at="SYNTHETIC_DATE",synthetic_fixture=True,
        labels_need_revision=False,methods=["S0","P0"],seeds=[42,43,44],p0_supervision=dict(radius_px=2,ownership="nearest_truth_then_index",normalization="per_truth_mean"),
        continuation_gate=copy.deepcopy(freeze.DEFAULT_GATE),auto_expand_if_gate_passes=True,review_limitations_acknowledged=True)


@pytest.fixture
def frozen(tmp_path):
    configdir=tmp_path/"configs";configdir.mkdir()
    for method in ("S0","P0"):(configdir/(method+".json")).write_bytes((ROOT/OLD/"configs"/(method+".json")).read_bytes())
    write(configdir/"blind_audit_expected_ids.json",[r["anonymous_id"] for r in answers()])
    response=tmp_path/"responses.csv";csv_write(response,answers(),reviewer.FIELDS)
    dec=tmp_path/"decision.json";write(dec,decision())
    return configdir,response,dec


def test_synthetic_answers_never_pass_real_gate():
    rows=answers();ids=[r["anonymous_id"] for r in rows]
    with pytest.raises(ValueError,match="provenance"):reviewer.validate_answers(rows,ids)
    result=reviewer.validate_answers(rows,ids,synthetic=True);assert result["complete"] and not result["human_review_completed"]


@pytest.mark.parametrize("mutation",["duplicate","missing","private","bad_count","bad_center","post_blind","mixed_reviewer"])
def test_bad_answers_reject(mutation):
    rows=answers();expected=[r["anonymous_id"] for r in rows]
    if mutation=="duplicate":rows[-1]["anonymous_id"]=rows[0]["anonymous_id"]
    if mutation=="missing":rows[0]["review_confidence"]=""
    if mutation=="private":rows[0]["sample_id"]="forbidden"
    if mutation=="bad_count":rows[0]["number_of_spots"]="-1"
    if mutation=="bad_center":rows[0]["suggested_center_y_patch_px"]="nan";rows[0]["suggested_center_x_patch_px"]="1"
    if mutation=="post_blind":rows[0]["adjudicated_detection_label"]="TP"
    if mutation=="mixed_reviewer":rows[0]["reviewer"]="OTHER"
    with pytest.raises(ValueError):reviewer.validate_answers(rows,expected,synthetic=True)


def test_uncertain_is_valid_without_implying_training_label():
    rows=answers();rows[0].update(candidate_is_real_spot="uncertain",number_of_spots="uncertain",review_confidence="low")
    result=reviewer.validate_answers(rows,[r["anonymous_id"] for r in rows],synthetic=True)
    assert result["answer_counts"]["uncertain"]==1 and "not one-to-one" in result["interpretation"]


def test_freeze_and_tamper_protection(frozen,tmp_path):
    cfg,res,dec=frozen;out=tmp_path/"frozen"
    marker=freeze.freeze(res,dec,out,cfg,synthetic=True)
    assert marker["formal_training_started"] is False and marker["human_review_completed"] is False
    assert read(out/"frozen_configs/P0.json")["supervision"]["radius_px"]==2
    assert read(out/"frozen_configs/P0.json")["decision_status"].startswith('approved_exact_')
    with pytest.raises(ValueError,match="Synthetic"):freeze.verify_frozen(out)
    assert freeze.verify_frozen(out,True)[0]==marker
    (out/"blind_audit_responses.csv").write_text("changed")
    with pytest.raises(ValueError,match="modified"):freeze.verify_frozen(out,True)


def test_interrupted_freeze_never_publishes_partial_approval(frozen,tmp_path,monkeypatch):
    cfg,res,dec=frozen;out=tmp_path/"atomic_decision";original_copy=freeze.shutil.copyfile
    with monkeypatch.context() as patch:
        patch.setattr(freeze.shutil,"copyfile",lambda *args,**kwargs:(_ for _ in ()).throw(OSError("SYNTHETIC disk interruption")))
        with pytest.raises(OSError):freeze.freeze(res,dec,out,cfg,synthetic=True)
    assert not out.exists() and list(tmp_path.glob(".atomic_decision.pending_*"))
    freeze.freeze(res,dec,out,cfg,synthetic=True)
    assert freeze.verify_frozen(out,True)[0]["synthetic_fixture"]


@pytest.mark.parametrize("field,value",[("approved",False),("labels_need_revision",True),("review_limitations_acknowledged",False),("approved_by",""),("synthetic_fixture",False)])
def test_research_decisions_are_not_autoapproved(field,value):
    dec=decision();dec[field]=value
    with pytest.raises(ValueError):freeze.validate_decision(dec,True)


def test_gate_is_fixed_and_not_a_biological_fp_label(tmp_path):
    truth={"SIM":[[4.,4.]]}
    folders=[]
    for method,error in (("S0",.5),("P0",.3)):
        f=tmp_path/method;folders.append(f)
        write(f/"primary/metrics.json",dict(f1=.8))
        csv_write(f/"primary/matches.csv",[dict(sample_id="SIM",pred_index=0,true_index=0,distance_px=error)],["sample_id","pred_index","true_index","distance_px"])
        csv_write(f/"primary/prediction_points.csv",[dict(sample_id="SIM",y=4.,x=4.,p=.9)],["sample_id","y","x","p"])
    policy=freeze.DEFAULT_GATE|dict(minimum_common_tp=1)
    result=gate.evaluate_gate(*folders,truth,policy);assert result["passed"] and result["common_tp_error_improvement_px"]==pytest.approx(.2)
    assert "proxies" in result["proxy_warning"]
    policy["minimum_common_tp"]=2;assert not gate.evaluate_gate(*folders,truth,policy)["passed"]


def make_trial(root,method="S0",seed=42):
    """Metadata/byte fixture only; no optimization, no valid real checkpoint."""
    folder=root/"outputs/trials"/method/f"seed_{seed}";folder.mkdir(parents=True)
    metrics=dict(tp=1,fp=0,fn=0,f1=1.,precision=1.,recall=1.,weak_recall=1.,mean_error_px=.2,ranked_ap=1.,test_evaluated=False,engineering_only=True)
    choice={};snapshots=[]
    for epoch in range(5,101,5):
        path=folder/f"epoch_{epoch:03d}";path.mkdir()
        (path/"candidates.npz").write_bytes(b"SYNTHETIC_CACHE_NOT_NUMPY")
        snapshots.append(dict(epoch=epoch,candidates_sha256=sha(path/"candidates.npz"),weight_sha256="SYNTHETIC"))
    for role,name in (("primary","best_model.pt"),("native_secondary","native_selected_model.pt")):
        (folder/name).write_bytes(b"SYNTHETIC_NOT_A_TORCH_MODEL")
        choice[role]=dict(epoch=100,checkpoint_sha256=sha(folder/name),calibrated_probability=.5,**{k:metrics[k] for k in ("tp","fp","fn","f1")})
        write(folder/role/"metrics.json",metrics);write(folder/role/"ranked_ap.json",dict(ap=1.,threshold_floor=None,policy="ranked_point_ap_greedy_step_v1"))
    choice.update(checkpoint_count=20,epoch_threshold_pairs=620,test_evaluated=False)
    write(folder/"selection.json",choice);write(folder/"snapshots.json",snapshots)
    write(folder/"status.json",dict(status="complete",epoch=100,updates=2200,test_evaluated=False,engineering_only=True))
    csv_write(folder/"training_metrics.csv",[dict(epoch=i,optimizer_updates=i*22) for i in range(1,101)],["epoch","optimizer_updates"])
    write(folder/"method_locks.json",[dict(method=method,seed=seed,selection_sha256=sha(folder/"selection.json"))])
    return folder


def test_lean_return_roundtrip_and_formal_rejection(tmp_path):
    root=tmp_path/"kit";root.mkdir();make_trial(root)
    write(root/"bundle_manifest.json",dict(SYNTHETIC=True));gatefile=root/"outputs/pipeline/gate.json";write(gatefile,dict(passed=False,synthetic_fixture=True))
    packed=archive.pack_return(root,[dict(method="S0",seed=42)],{},gatefile,engineering_fixture=True)
    a=Path(packed["path"])
    with pytest.raises(ValueError,match="Engineering"):archive.ingest(a,a.with_name(a.name+".sha256"),tmp_path/"rejected")
    result=archive.ingest(a,a.with_name(a.name+".sha256"),tmp_path/"accepted",True)
    assert result["verified"] and result["engineering_only"]
    assert (tmp_path/"accepted/outputs/trials/S0/seed_42/epoch_100/candidates.npz").exists()
    assert not (tmp_path/"accepted/outputs/trials/S0/seed_42/epoch_005/candidates.npz").exists()
    (tmp_path/"accepted/outputs/trials/S0/seed_42/best_model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="hash"):archive.verify_return(tmp_path/"accepted",True)


@pytest.mark.parametrize("name,kind",[("../escape.txt","file"),("C:/escape.txt","file"),("safe\\bad.txt","file"),("link","symlink")])
def test_unsafe_tar_never_extracts(tmp_path,name,kind):
    a=tmp_path/"unsafe.tar.gz"
    with tarfile.open(a,"w:gz") as tar:
        info=tarfile.TarInfo(name)
        if kind=="symlink":info.type=tarfile.SYMTYPE;info.linkname="../escape"
        else:info.size=1
        tar.addfile(info,io.BytesIO(b"x") if kind=="file" else None)
    checksum=tmp_path/"hash";checksum.write_text(sha(a))
    with pytest.raises(ValueError,match="Unsafe"):archive.ingest(a,checksum,tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_bad_download_hash_and_crc_reject(tmp_path):
    a=tmp_path/"bad.tar.gz";a.write_bytes(b"invalid");h=tmp_path/"hash";h.write_text("0"*64)
    with pytest.raises(ValueError,match="SHA256"):archive.ingest(a,h,tmp_path/"out")
    h.write_text(sha(a))
    with pytest.raises((OSError,EOFError)):archive.ingest(a,h,tmp_path/"out")


def test_report_missing_tasks_are_not_zero():
    rows=[dict(method="G0",f1=.8,precision=.9,recall=.7,weak_recall=.4,mean_error_px=.3,ranked_ap=None)]
    summary=report.summarize(rows);ap=next(r for r in summary if r["metric"]=="ranked_ap")
    assert ap["mean"] is None and ap["n_seeds"]==0
    assert next(r for r in summary if r["metric"]=="f1")["sd"] is None


def test_correction_macro_preserves_xy_and_never_saves_original():
    from diagnostics.handoff_v2.corrections import make_macro,check_revision
    macro=make_macro("sample_1",[[2.25,4.5]],[[2.,4.]])
    assert "newArray(4.5),newArray(2.25)" in macro
    assert "File.exists(destination)" in macro and "saveAs(" not in macro
    assert not check_revision([[2,4]],[[2.2,4]],(8,8))["approved_for_training"]
    with pytest.raises(ValueError):check_revision([[2,4]],[[8,4]],(8,8))


def test_synthetic_freeze_stops_before_any_cuda_or_training(frozen,tmp_path,monkeypatch):
    cfg,res,dec=frozen;root=tmp_path/"kit";root.mkdir()
    freeze.freeze(res,dec,root/"decisions",cfg,synthetic=True)
    write(root/"bundle_manifest.json",dict(kind="signed_offset_v1",files={},test_arrays_present=False))
    from diagnostics.handoff_v2.pipeline import execute
    with pytest.raises(ValueError,match="Synthetic"):execute(root,run_command=lambda cmd:pytest.fail("must not execute"))
    assert not (root/"outputs").exists()


def test_frontend_csv_roundtrip_contract(tmp_path):
    import shutil,subprocess
    node=shutil.which("node")
    if not node:pytest.skip("Node is needed for the pure-JS form integration check")
    csv_path=tmp_path/"SYNTHETIC_QA.csv"
    result=subprocess.run([node,str(Path(__file__).with_name("test_reviewer.js")),str(csv_path)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)["passed"]
    from diagnostics.handoff_v2.common import table
    rows=table(csv_path);ids=[r["anonymous_id"] for r in rows]
    checked=reviewer.validate_answers(rows,ids,synthetic=True)
    assert checked["complete"] and not checked["human_review_completed"]
    assert rows[0]["notes"]=='comma, quote " and\nnewline'
    with pytest.raises(ValueError,match="provenance"):reviewer.validate_answers(rows,ids)


@pytest.mark.parametrize("passed,approved,n",[(True,True,4),(True,False,0),(False,True,0),(False,False,0)])
def test_queue_expands_only_under_prespecified_two_gates(passed,approved,n):
    from diagnostics.handoff_v2.pipeline import expansion_tasks
    tasks=expansion_tasks(dict(passed=passed),dict(auto_expand_if_gate_passes=approved))
    assert len(tasks)==n
    if n:assert tasks==[("S0",43),("P0",43),("S0",44),("P0",44)]


def test_wrong_packet_cannot_be_frozen():
    rows=answers();ids=[r["anonymous_id"] for r in rows]
    for r in rows:r["review_packet_id"]="SYNTHETIC_PACKET_A"
    assert reviewer.validate_answers(rows,ids,synthetic=True,expected_packet_id="SYNTHETIC_PACKET_A")["complete"]
    with pytest.raises(ValueError,match="different image"):reviewer.validate_answers(rows,ids,synthetic=True,expected_packet_id="SYNTHETIC_PACKET_B")


def test_versioned_label_proposal_preserves_parent_and_blocks_training(tmp_path):
    import numpy as np
    from diagnostics.handoff_v2.label_revision import propose
    arrays={}
    for split in ("train","valid"):
        arrays.update({f"x_{split}":np.zeros((1,16,16),np.float32),f"y_{split}":np.array([[[2.,3.]]],np.float32),
            f"sample_id_{split}":np.array(["SYN_"+split]),f"original_shape_{split}":np.array([[16,16]])})
    parent=tmp_path/"parent.npz";np.savez_compressed(parent,**arrays);digest=sha(parent)
    reviewed=tmp_path/"reviewed.csv";csv_write(reviewed,[dict(sample_id="SYN_valid",point_index=i,y=y,x=x) for i,(y,x) in enumerate([[2.25,3.5],[2.5,3.75]])],["sample_id","point_index","y","x"])
    result=propose(parent,digest,[reviewed],tmp_path/"revision","SYNTHETIC_QA","synthetic test only",synthetic=True)
    assert not result["approved_for_training"] and not result["current_training_protocol_compatible"] and result["synthetic_fixture"]
    assert result["valid_points"]==2 and result["same_pixel_collisions"]==1 and sha(parent)==digest
    with np.load(tmp_path/"revision/development_revision.npz",allow_pickle=True) as z:
        assert np.array_equal(z["x_valid"],arrays["x_valid"])
        assert len(z["y_valid"][0])==2
    with pytest.raises(ValueError,match="Parent"):propose(parent,"0"*64,[reviewed],tmp_path/"bad","QA","reason",True)
    csv_write(tmp_path/"invalid.csv",[dict(sample_id="SYN_valid",y=16,x=3)],["sample_id","y","x"])
    with pytest.raises(ValueError,match="outside"):propose(parent,digest,[tmp_path/"invalid.csv"],tmp_path/"bad2","QA","reason",True)


def test_revised_label_recheck_is_diagnostic_and_keeps_locked_thresholds(tmp_path,monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from diagnostics.handoff_v2.label_revision import propose,reevaluate
    sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts/diagnostics/scaffolds"))
    from diagnostics.mechanism_v1 import runner,evaluate as ev
    from diagnostics.handoff_v2.common import OLD
    arrays={}
    for split in ("train","valid"):
        arrays.update({f"x_{split}":np.zeros((1,16,16),np.float32),f"y_{split}":np.array([[[2.,3.]]],np.float32),
            f"sample_id_{split}":np.array(["SYN_"+split]),f"original_shape_{split}":np.array([[16,16]])})
    parent=tmp_path/"parent.npz";np.savez_compressed(parent,**arrays)
    reviewed=tmp_path/"reviewed.csv";csv_write(reviewed,[dict(sample_id="SYN_valid",y=2.2,x=3.3)],["sample_id","y","x"])
    propose(parent,sha(parent),[reviewed],tmp_path/"revision","SYNTHETIC_QA","test only",True)
    old={k:arrays[k+"_valid"] for k in ("x","y","sample_id","original_shape")}
    monkeypatch.setattr(runner,"corpus",lambda root:SimpleNamespace(paths=dict(development=parent),splits=dict(valid=old)))
    called=[]
    def fake_evaluate(lock,root,data,folder,**kwargs):
        called.append((lock["threshold"],kwargs["synthetic"]))
        assert float(data["y"][0][0][0])==pytest.approx(2.2)
        return dict(f1=.5,ranked_ap=.4,weak_recall=None,test_evaluated=False,engineering_only=True)
    monkeypatch.setattr(ev,"evaluate",fake_evaluate)
    write(tmp_path/OLD/"configs/validation_model_locks.json",[dict(method="SYNTHETIC",seed=42,threshold=.123)])
    p=Path("docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/configs/weak_policy.json")
    target=tmp_path/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/p).read_bytes())
    result=reevaluate(tmp_path,tmp_path/"revision",tmp_path/"recheck",synthetic=True)
    assert result["diagnostic_only"] and not result["approved_gold_standard"] and called==[(.123,True)]
