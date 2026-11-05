"""Synthetic correctness tests. No dataset training or GPU usage."""
from pathlib import Path
import copy,json,sys
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"));sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts"))
sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts/diagnostics/scaffolds"))
from diagnostics.mechanism_v1.adapter import make_model,encode,decode,loss,supervision,PARAMETERS
from diagnostics.mechanism_v1.contracts import RECIPE,canonical,require_review,validate
from diagnostics.mechanism_v1.runner import smoke,load_model
from deepblink_torch.round2_protocol import encode_dense_targets,dense_heatmap_loss
from research_v4.engine import reset_rng,model_hash


def config(method="S0"):
    return dict(schema="signed_offset_experiment_v1",method=method,status="not_started",formal_training_counted=False,
        recipe=copy.deepcopy(RECIPE),seeds=[42,43,44],expected_parameters=PARAMETERS,
        supervision=dict(radius_px=0,ownership="single_floor_anchor",normalization="anchor_mean") if method=="S0" else
        dict(radius_px=2,ownership="nearest_truth_then_index",normalization="per_truth_mean"))


def test_same_capacity_state_and_heatmap():
    torch.set_num_threads(2);reset_rng(42);h0=make_model("H0");reset_rng(42);s0=make_model("S0");reset_rng(42);p0=make_model("P0")
    assert model_hash(h0)==model_hash(s0)==model_hash(p0)
    assert sum(p.numel() for p in s0.parameters())==PARAMETERS
    x=torch.randn(1,1,128,128);h0.eval();s0.eval()
    with torch.no_grad():a=h0(x);b=s0(x)
    assert torch.equal(a[...,0],b[...,0]);assert torch.equal(a[...,1:],b[...,1:].sigmoid())
    with torch.no_grad():s0.dense_head.weight.zero_();s0.dense_head.bias.copy_(torch.tensor([0.,-2.,3.]))
    with torch.no_grad():out=s0(x)
    assert torch.equal(out[0,0,0],torch.tensor([0.,-2.,3.]))


def test_s0_exact_target_and_loss():
    truth=np.array([[1.,2.],[5.5,7.75],[15.999,15.999]],np.float32)
    target,_=encode(truth,(16,16),32,config());old=encode_dense_targets(truth,(16,16),32)
    assert np.array_equal(target,old)
    prediction=torch.randn(1,32,32,3,requires_grad=True);t=torch.from_numpy(target[None])
    assert torch.equal(loss(t,prediction,"S0"),dense_heatmap_loss(t,prediction))


@pytest.mark.parametrize("method",["S0","P0"])
@pytest.mark.parametrize("point",[[0.,0.],[.5,.5],[7.25,8.75],[15.999,15.999]])
def test_target_decoding_roundtrip(method,point):
    truth=np.asarray([point],np.float32);target,owner=encode(truth,(16,16),32,config(method))
    prediction=np.zeros((32,32,3),np.float32);prediction[...,0]=-20
    anchor=np.floor(truth[0]).astype(int)
    if method=="P0":
        choices=np.argwhere(owner==0);anchor=choices[-1] # Negative offset target away from the floor.
    y,x=anchor;prediction[y,x,0]=20;prediction[y,x,1:]=target[y,x,1:3]
    points,scores=decode(prediction,(16,16));assert any(np.allclose(p,truth[0],atol=1e-6) for p in points)
    assert len(scores)==len(points);assert (owner[16:]==-1).all();assert (target[16:,:,0]==-1).all()


def test_overlap_tie_and_equal_truth_weight():
    truth=np.array([[2.,2.],[2.,4.]],np.float32);target,owner=encode(truth,(8,8),16,config("P0"))
    assert owner[2,3]==0
    for i in range(2):
        yy,xx=np.nonzero(owner==i)
        np.testing.assert_allclose(target[yy,xx,3].sum(),1.,rtol=1e-6)
        np.testing.assert_allclose(np.column_stack([yy,xx])+target[yy,xx,1:3],np.broadcast_to(truth[i],(len(yy),2)),atol=1e-6)
    assert (target[8:,:,3]==0).all()


def test_weighted_rmse_and_padding_gradient():
    target,owner=encode([[0.,0.],[7.,7.]],(8,8),16,config("P0"))
    pred=torch.from_numpy(target[None,:,:,:3].copy()).requires_grad_()
    with torch.no_grad():
        pred[...,0]=0
        pred[0,:,:,1][torch.from_numpy(owner==0)]+=1
        pred[0,:,:,1][torch.from_numpy(owner==1)]+=3
    value=loss(torch.from_numpy(target[None]),pred,"P0");value.backward()
    assert torch.isfinite(value);assert torch.count_nonzero(pred.grad[0,8:])==0
    # Weighted squared error: each GT has mass 1 regardless of boundary clipping.
    weights=target[...,3];errors=(pred.detach().numpy()[0,...,1:]-target[...,1:3])**2
    assert float((errors.sum(-1)*weights).sum()/weights.sum())==pytest.approx(5.)


def test_two_close_peaks_and_signed_boundary_filter():
    target,_=encode([[3.5,3.5],[3.5,5.5]],(16,16),32,config("P0"))
    prediction=target[...,:3].copy();prediction[...,0]=-10
    for y,x in [(3,3),(3,5)]:prediction[y,x,0]=10
    points,scores=decode(prediction,(16,16));assert len(points[scores>.5])==2
    prediction[...,0]=-10;prediction[0,0]=[10,-.25,.1]
    points,_=decode(prediction,(16,16));assert not np.any(points<0)
    prediction[...,0]=-10;prediction[15,15]=[10,1.-1e-8,1.-1e-8]
    points,_=decode(prediction,(16,16));assert np.all(points<16)


def test_collisions_nonfinite_and_todos_reject():
    with pytest.raises(ValueError):encode([[2.1,2.1],[2.2,2.2]],(8,8),16,config())
    c=config("P0");c["supervision"]["radius_px"]=None
    validate(c,require_parameters=False)
    with pytest.raises(ValueError):validate(c)
    with pytest.raises(FloatingPointError):decode(np.full((8,8,3),np.nan),(8,8))


def test_human_gate_cannot_be_replaced_by_dry_run():
    c=config();approval=dict(config_sha256=canonical(c),mechanism_approved=True,reviewer="human",reviewed_at="date",synthetic_fixture=False)
    require_review(c,approval)
    for key in ["mechanism_approved","reviewer","reviewed_at","config_sha256","synthetic_fixture"]:
        bad=approval.copy();bad.pop(key)
        with pytest.raises(ValueError):require_review(c,bad)
    p=config("P0");approval["config_sha256"]=canonical(p)
    with pytest.raises(ValueError):require_review(p,approval)


def test_cpu_one_unique_step_recovery_and_export(tmp_path):
    result=smoke(config(),tmp_path/"smoke")
    assert result["passed"] and result["exact_recovery"] and result["real_optimizer_updates"]==0
    assert result["distinct_synthetic_steps"]==1
    model,saved=load_model(tmp_path/"smoke/after.pt")
    assert saved["method"]=="S0" and saved["epoch"]==0
    with pytest.raises(FileExistsError):smoke(config(),tmp_path/"smoke")


def test_formal_cpu_and_missing_review_stop_before_data():
    from diagnostics.mechanism_v1.runner import train
    c=config();approval=dict(config_sha256=canonical(c),mechanism_approved=True,reviewer="human",reviewed_at="date",synthetic_fixture=False)
    with pytest.raises(ValueError,match="CUDA"):train(c,None,None,42,"cpu",approval,{})
    with pytest.raises(ValueError,match="human"):train(c,None,None,42,"cpu",{}, {})


def test_portable_verify_status_and_tamper_gate(tmp_path,capsys):
    from diagnostics.mechanism_v1.controller import verify,status,command
    from diagnostics.mechanism_v1.contracts import write_new,sha
    (tmp_path/"source.py").write_text("pass\n")
    write_new(tmp_path/"bundle_manifest.json",dict(kind="signed_offset_v1",files={"source.py":sha(tmp_path/"source.py")},test_arrays_present=False))
    assert verify(tmp_path)["files"]==1
    rows=status(tmp_path);assert len(rows)==6 and all(r["status"]=="not_started" for r in rows)
    cmd=command(tmp_path,"S0","train",43,True);assert cmd[-1]=="--resume" and "43" in cmd
    (tmp_path/"source.py").write_text("changed")
    with pytest.raises(ValueError,match="changed"):verify(tmp_path)


def test_independent_test_requires_verified_manifest(tmp_path):
    from diagnostics.mechanism_v1.evaluate import evaluate
    with pytest.raises(ValueError,match="provenance"):evaluate({},tmp_path,{},tmp_path/"out",role="independent_test")


def test_review_gate_rejects_blank_and_synthetic_answers(tmp_path,monkeypatch):
    import csv
    from diagnostics.mechanism_v1 import contracts as ct
    (tmp_path/"configs").mkdir();ids=[f"SIM_{i:04d}" for i in range(60)]
    ct.write_new(tmp_path/"configs/blind_audit_expected_ids.json",ids);monkeypatch.setattr(ct,"ROOT",tmp_path)
    fields=["anonymous_id","candidate_is_real_spot","number_of_spots","review_confidence","reviewer","synthetic_fixture"]
    audit=tmp_path/"responses.csv"
    def write(rows):
        with audit.open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    rows=[dict.fromkeys(fields,"")|dict(anonymous_id=i) for i in ids];write(rows)
    approval=dict(blind_audit_file="responses.csv",blind_audit_sha256=ct.sha(audit))
    with pytest.raises(ValueError,match="review evidence"):ct.verify_audit_file(approval,tmp_path)
    rows=[dict(anonymous_id=i,candidate_is_real_spot="uncertain",number_of_spots="uncertain",review_confidence="low",reviewer="SYNTHETIC",synthetic_fixture="True") for i in ids]
    write(rows);approval["blind_audit_sha256"]=ct.sha(audit)
    with pytest.raises(ValueError,match="Synthetic"):ct.verify_audit_file(approval,tmp_path)


def test_nonempty_foreign_trials_and_outside_outputs_are_protected(tmp_path):
    from diagnostics.mechanism_v1.runner import prepare_trial_folder
    foreign=tmp_path/"outputs/trials/H0/seed_42";foreign.mkdir(parents=True)
    marker=foreign/"best_model.pt";marker.write_bytes(b"PROTECTED")
    with pytest.raises(ValueError,match="protected"):prepare_trial_folder(foreign,tmp_path)
    assert marker.read_bytes()==b"PROTECTED" and not (foreign/"registration.json").exists()
    with pytest.raises(ValueError,match="inside"):prepare_trial_folder(tmp_path/"extracted_h0",tmp_path)
    assert prepare_trial_folder(tmp_path/"outputs/trials/S0/seed_42",tmp_path).is_dir()


def test_ranked_ap_and_locks_export_after_selection(tmp_path):
    from diagnostics.mechanism_v1.runner import export_ranked_selection
    from research_v4.evaluation import save_candidates
    from research_v4.common import dump
    folder=tmp_path/"outputs/trials/S0/seed_42";(folder/"epoch_005").mkdir(parents=True)
    data=dict(sample_id=np.array(["SYNTHETIC"]),original_shape=np.array([[16,16]]),y=[np.array([[3.,4.]],np.float32)])
    candidates=[(np.array([[3.,4.],[10.,10.]],np.float32),np.array([.9,.1],np.float32))]
    save_candidates(folder/"epoch_005/candidates.npz",candidates,data)
    selection={r:dict(epoch=5,calibrated_probability=.5,checkpoint_sha256="SYNTHETIC_FIXTURE") for r in ("primary","native_secondary")}
    for role in selection:dump(folder/role/"metrics.json",dict(tp=1,fp=0,fn=0,f1=1.,engineering_only=True,test_evaluated=False))
    exported=export_ranked_selection(folder,data,selection,"S0",42,tmp_path)
    assert exported["primary"]["metrics"]["ranked_ap"]==1.
    locks=json.loads((folder/"method_locks.json").read_text());assert locks[0]["threshold"]==.5
    assert json.loads((folder/"primary/ranked_ap.json").read_text())["n_candidates"]==2


def test_empty_truth_has_finite_loss():
    for method in ["S0","P0"]:
        target,_=encode([],(16,16),32,config(method));pred=torch.randn(1,32,32,3,requires_grad=True)
        value=loss(torch.from_numpy(target[None]),pred,method);value.backward();assert torch.isfinite(value)


def test_scored_frozen_evaluation_integration(tmp_path,monkeypatch):
    from diagnostics.mechanism_v1 import evaluate as ev
    from research_v4.common import dump
    from diagnostics.mechanism_v1.runner import payload
    model=make_model("S0");checkpoint=tmp_path/"model.pt";torch.save(payload(model,config(),42,0),checkpoint)
    from diagnostics.mechanism_v1.contracts import sha
    threshold=.5
    dump(tmp_path/"selection.json",dict(primary=dict(checkpoint_sha256=sha(checkpoint),calibrated_probability=threshold)))
    lock=dict(checkpoint="model.pt",weight_sha256=sha(checkpoint),selection="selection.json",selection_sha256=sha(tmp_path/"selection.json"),threshold=threshold)
    data=dict(x=[np.zeros((16,16),np.float32)],y=[np.array([[3.,4.]],np.float32)],sample_id=np.array(["SYNTHETIC"]),original_shape=np.array([[16,16]]),weak=[np.array([True])])
    monkeypatch.setattr(ev,"infer",lambda *a,**kw:[(np.array([[3.,4.],[10.,10.]],np.float32),np.array([.9,.1],np.float32))])
    metrics=ev.evaluate(lock,tmp_path,data,tmp_path/"evaluation",synthetic=True)
    assert metrics["f1"]==1 and metrics["ranked_ap"]==1 and metrics["weak_recall"]==1
    assert not metrics["test_evaluated"] and metrics["engineering_only"]
    assert json.loads((tmp_path/"evaluation/ranked_ap.json").read_text())["n_candidates"]==2
