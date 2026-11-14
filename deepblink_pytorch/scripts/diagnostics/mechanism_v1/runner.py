"""CPU one-step readiness probes and gated future CUDA experiments.

Formal execution requires --train, an exact-config human decision, and a CUDA
probe bound to the same source, data and backend. Default is configuration check.
"""
from pathlib import Path
import argparse,json,time,uuid
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .contracts import ROOT,REL,read,sha,canonical,write_new,validate,require_review,verify_audit_file,code_hashes
from .adapter import make_model,encode,loss,decode,PARAMETERS,REPRESENTATION
from research_v4.augmentation import prepare_sample
from research_v4.common import Corpus,load_spec,dump,write_csv
from research_v4.engine import (configure_device,reset_rng,model_hash,rng_state,set_rng,
    preserve_rng,save_torch,equal_nested,finalize)
from research_v4.evaluation import save_candidates,load_candidates,sweep,policies_probability,selection_key
from deepblink_torch.round2_protocol import training_orders
from deepblink_torch.preprocessing import prepare_image


class MechanismDataset(Dataset):
    def __init__(self,data,config,canvas=512):
        self.samples=[];self.ownership_counts=[]
        for image,truth,shape in zip(data["x"],data["y"],data["original_shape"]):
            x,old,_,_=prepare_sample(image,truth,canvas=canvas,dense=True,augment=False)
            target,owners=encode(truth,shape,canvas,config)
            if not np.array_equal(old[...,0],target[...,0]):raise AssertionError("Heatmap target changed")
            self.samples.append((torch.from_numpy(x),torch.from_numpy(target)))
            self.ownership_counts.append(int(np.count_nonzero(owners>=0)))
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):return self.samples[i]


def corpus(root,assets=None):
    spec_path=(Path(assets).parent/"protocol_proposed.yaml") if assets else Path(root)/"docs/next_stage_plan_20261025/overnight_protocol_v4_proposed_20261101.yaml"
    return Corpus(root,load_spec(spec_path),assets_root=assets)


def binding(config,data,device):
    import scipy,platform
    return dict(config_sha256=canonical(config),source_sha256=canonical(code_hashes()),
        development_sha256=sha(data.paths["development"]),features_sha256=sha(data.paths["features"]),
        device=device,torch=str(torch.__version__),numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0) if device=="cuda" else None)


def reference_initialization(seed,initial,device):
    path=ROOT/"configs/reference_initializations.json"
    if not path.exists():path=ROOT/REL/"configs/reference_initializations.json"
    refs=read(path);expected=refs[str(seed)]["initial_model_state_sha256"]
    result=dict(seed=seed,reference_backend=refs[str(seed)]["backend"],reference_sha256=expected,
        actual_sha256=initial,matches_historical_cuda=initial==expected)
    if device=="cuda" and initial!=expected:raise ValueError("Historical CUDA initialization differs: investigate backend/RNG before any formal update")
    return result


def optimizer_step(model,opt,x,target,method):
    model.train();opt.zero_grad(set_to_none=True)
    value=loss(target,model(x).float(),method)
    if not torch.isfinite(value):raise FloatingPointError("Nonfinite loss")
    value.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError("Nonfinite gradient")
    opt.step();return float(value.detach())


@torch.no_grad()
def infer(model,data,device,canvas=512):
    model.eval();result=[]
    for image,shape in zip(data["x"],data["original_shape"]):
        tensor=torch.from_numpy(prepare_image(image,canvas)[None,None]).to(device).contiguous(memory_format=torch.channels_last)
        result.append(decode(model(tensor)[0].float().cpu().numpy(),shape))
    return result


def payload(model,config,seed,epoch):
    return dict(model_state_dict=model.state_dict(),model_name="signed_heatmap_isolated_v1",method=config["method"],
        output_representation=REPRESENTATION,cell_size=1,input_size=512,seed=seed,epoch=epoch,
        preprocessing="normalize_original_then_zero_pad_right_bottom",parameter_count=PARAMETERS,
        experiment_config=config,config_sha256=canonical(config),model_state_sha256=model_hash(model),
        protocol_id="signed_offset_experiment_v1",training_augmentation=False)


def load_model(path,device="cpu"):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if saved.get("model_name")!="signed_heatmap_isolated_v1" or saved.get("output_representation")!=REPRESENTATION:
        raise ValueError("Not an isolated signed-offset checkpoint")
    config=saved["experiment_config"];validate(config)
    if saved["config_sha256"]!=canonical(config):raise ValueError("Checkpoint config differs")
    model=make_model(config["method"],device);model.load_state_dict(saved["model_state_dict"])
    if model_hash(model)!=saved["model_state_sha256"]:raise ValueError("Checkpoint state differs")
    model.eval();return model,saved


def prepare_trial_folder(folder,root=ROOT):
    folder=Path(folder).resolve();root=Path(root).resolve()
    if not folder.is_relative_to(root/"outputs/trials"):raise ValueError("Formal output must be inside this kit's outputs/trials")
    if folder.exists() and any(folder.iterdir()) and not (folder/"registration.json").exists():
        raise ValueError("Nonempty unregistered output is protected; choose a fresh trial directory")
    folder.mkdir(parents=True,exist_ok=True);return folder


def export_ranked_selection(folder,data,selection,method,seed,root=ROOT):
    """Post-selection AP uses complete cached candidates, with no new selection."""
    from diagnostics.scaffolds.ranked_ap import ranked_point_ap
    folder=Path(folder);locks=[]
    for role,name in (("primary","best_model.pt"),("native_secondary","native_selected_model.pt")):
        choice=selection[role]
        candidates=load_candidates(folder/f"epoch_{choice['epoch']:03d}/candidates.npz",data)
        truth={str(s):t for s,t in zip(data["sample_id"],data["y"])}
        scored={str(s):c for s,c in zip(data["sample_id"],candidates)}
        ranked=ranked_point_ap(truth,scored,radius=3.)
        write_csv(folder/role/"ranked_predictions.csv",ranked.pop("ranked_records"),["sample_id","pred_index","score","is_tp","true_index"])
        write_csv(folder/role/"ranked_pr_curve.csv",ranked.pop("curve"),["score","tp","fp","recall","precision"])
        dump(folder/role/"ranked_ap.json",ranked)
        metrics=read(folder/role/"metrics.json");metrics.update(ranked_ap=ranked["ap"],ap_policy=ranked["policy"],score_floor=None)
        dump(folder/role/"metrics.json",metrics);selection[role]["metrics"]=metrics
    dump(folder/"selection.json",selection)
    locks.append(dict(method=method,seed=seed,checkpoint=(folder/"best_model.pt").relative_to(root).as_posix(),
        weight_sha256=selection["primary"]["checkpoint_sha256"],selection=(folder/"selection.json").relative_to(root).as_posix(),
        selection_sha256=sha(folder/"selection.json"),threshold=selection["primary"]["calibrated_probability"],validation_selected=True))
    dump(folder/"method_locks.json",locks)
    return selection


def smoke(config,folder,device="cpu",data=None):
    """One unique synthetic update, replayed once to verify exact restoration."""
    validate(config);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    configure_device(device);reset_rng(42)
    canvas,batch=(512,8) if device=="cuda" else (128,1)
    rng=np.random.default_rng(20261104)
    image=rng.normal(size=(canvas,canvas)).astype(np.float32)
    truth=np.asarray([[20.25,30.5],[70.1,91.8]],np.float32)
    target,_=encode(truth,image.shape,canvas,config)
    x=torch.from_numpy(prepare_image(image,canvas)[None,None]).repeat(batch,1,1,1).to(device).contiguous(memory_format=torch.channels_last)
    y=torch.from_numpy(target[None]).repeat(batch,1,1,1).to(device)
    model=make_model(config["method"],device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    initial=model_hash(model);reference=reference_initialization(42,initial,device);state=rng_state()
    save_torch(folder/"before.pt",dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=state))
    value=optimizer_step(model,opt,x,y,config["method"])
    expected=model_hash(model);expected_opt=opt.state_dict();expected_rng=rng_state()
    probe_payload=payload(model,config,42,0);probe_payload.update(engineering_only=True,synthetic_fixture=True)
    save_torch(folder/"after.pt",probe_payload)
    loaded,_=load_model(folder/"after.pt",device)
    if model_hash(loaded)!=expected:raise AssertionError("Export/reload differs")
    before=torch.load(folder/"before.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(before["model"]);opt.load_state_dict(before["optimizer"]);set_rng(before["rng"])
    replay=optimizer_step(model,opt,x,y,config["method"])
    if model_hash(model)!=expected or not equal_nested(opt.state_dict(),expected_opt) or not equal_nested(rng_state(),expected_rng):
        raise AssertionError("One-step recovery replay differs")
    real=[]
    if data is not None:
        for subset in ("train","valid"):
            dataset=MechanismDataset(data.splits[subset],config)
            real.append(dict(split=subset,images=len(dataset),target_encoding_passed=True))
        # Geometry probes are inference only, no real-image optimizer updates.
        model.eval()
        with torch.no_grad():
            for size in (256,512):
                i=next(i for i,s in enumerate(data.splits["train"]["original_shape"]) if tuple(s)==(size,size))
                sample={k:[v[i]] for k,v in data.splits["train"].items() if k in ("x","y","sample_id","original_shape")}
                points,scores=infer(model,sample,device)[0]
                if np.any(points<0) or np.any(points>=size) or not np.isfinite(scores).all():raise AssertionError("Geometry export invalid")
                real.append(dict(original_shape=[size,size],canvas=512,candidate_count=len(points),valid_fov=True))
    result=dict(passed=True,engineering_only=True,formal_training_started=False,test_evaluated=False,
        device=device,torch=str(torch.__version__),batch_size=batch,canvas=canvas,parameters=PARAMETERS,
        finite_loss=value,replay_loss=replay,exact_recovery=True,export_reload_exact=True,
        distinct_synthetic_steps=1,discarded_optimizer_step_executions=2,real_optimizer_updates=0,
        initial_state_sha256=initial,initialization_reference=reference,real_data_checks=real,binding=binding(config,data,device) if data else None)
    write_new(folder/"checks.json",result);print(json.dumps(result),flush=True);return result


def train(config,data,folder,seed,device,approval,probe,resume=False,approval_root=None):
    """Future formal runner; never called by tonight's QA commands."""
    require_review(config,approval)
    if config["method"]=="P0":
        verify_audit_file(approval,approval_root)
    if device!="cuda":raise ValueError("Formal runner requires CUDA; no fallback")
    configure_device(device)
    lock=binding(config,data,device)
    if probe.get("binding")!=lock or not probe.get("passed") or not probe.get("exact_recovery") or probe.get("batch_size")!=8:
        raise ValueError("Full-geometry CUDA probe is missing, stale or from a different source/backend")
    folder=prepare_trial_folder(folder)
    registration=dict(binding=lock,seed=seed,approval_sha256=canonical(approval),protocol_id="signed_offset_experiment_v1")
    if (folder/"registration.json").exists():
        if read(folder/"registration.json")!=registration:raise ValueError("Trial identity changed")
        if not resume:raise ValueError("Existing trial requires --resume")
        if (folder/"status.json").exists() and read(folder/"status.json").get("status")=="complete":
            selection=read(folder/"selection.json")
            for role,name in (("primary","best_model.pt"),("native_secondary","native_selected_model.pt")):
                if sha(folder/name)!=selection[role]["checkpoint_sha256"]:raise ValueError("Completed checkpoint changed")
            from .controller import check_complete
            check_complete(folder)
            return selection
    else:write_new(folder/"registration.json",registration)
    reset_rng(seed);model=make_model(config["method"],device);initial=model_hash(model)
    reference_initialization(seed,initial,device)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    train_data=MechanismDataset(data.splits["train"],config);valid_data=MechanismDataset(data.splits["valid"],config)
    orders=training_orders(len(train_data),100,seed)
    refs=ROOT/"configs/reference_initializations.json"
    if not refs.exists():refs=ROOT/REL/"configs/reference_initializations.json"
    if canonical(orders)!=read(refs)[str(seed)]["sampler_orders_sha256"]:raise ValueError("Historical sampler order differs")
    init_path=folder/"initialization.json"
    init=dict(initial_model_state_sha256=initial,sampler_orders_sha256=canonical(orders),parameters=PARAMETERS,training_images=len(train_data),epochs=100)
    if init_path.exists():
        if read(init_path)!=init:raise ValueError("Initial state/sampler registration changed")
    else:write_new(init_path,init)
    history=[];snapshots=[];first=updates=0
    if resume and (folder/"recovery.pt").exists():
        saved=torch.load(folder/"recovery.pt",map_location="cpu",weights_only=True)
        if saved["registration"]!=registration or saved["initial_hash"]!=initial:raise ValueError("Recovery identity changed")
        model.load_state_dict(saved["model_state_dict"]);opt.load_state_dict(saved["optimizer_state_dict"])
        history=saved["history"];snapshots=saved["snapshots"];first=saved["epoch"];updates=saved["updates"];set_rng(saved["rng"])
    elif resume and (folder/"snapshots.json").exists():raise ValueError("Recovery missing for an existing epoch ledger")
    dump(folder/"status.json",dict(status="running",epoch=first,updates=updates,test_evaluated=False))
    try:
        for epoch in range(first,100):
            tick=time.perf_counter();values=[];validation=[]
            loader=DataLoader(train_data,batch_size=8,sampler=orders[epoch],num_workers=0,generator=torch.Generator().manual_seed(seed+100000+epoch))
            for x,y in loader:
                values.append(optimizer_step(model,opt,x.to(device).contiguous(memory_format=torch.channels_last),y.to(device),config["method"]))
                updates+=1
            model.eval()
            with torch.no_grad():
                loader=DataLoader(valid_data,batch_size=8,num_workers=0,generator=torch.Generator().manual_seed(seed+200000+epoch))
                for x,y in loader:validation.append(float(loss(y.to(device),model(x.to(device).contiguous(memory_format=torch.channels_last)).float(),config["method"])))
            val=float(np.mean(validation))
            if (epoch+1)%5==0:
                snapshot=folder/f"epoch_{epoch+1:03d}"
                if snapshot.exists():
                    orphan=folder/"orphaned_snapshots"/(snapshot.name+"_"+uuid.uuid4().hex[:8]);orphan.parent.mkdir(exist_ok=True);snapshot.rename(orphan)
                snapshot.mkdir()
                with preserve_rng():
                    save_torch(snapshot/"model.pt",payload(model,config,seed,epoch+1))
                    candidates=infer(model,data.splits["valid"],device)
                    save_candidates(snapshot/"candidates.npz",candidates,data.splits["valid"])
                    scores=[dict(epoch=epoch+1,native_val_loss=val,**r) for r in sweep(candidates,data.splits["valid"],policies_probability())]
                    dump(snapshot/"threshold_metrics.json",scores)
                    snapshots.append(dict(epoch=epoch+1,native_val_loss=val,best_at_epoch=max(scores,key=selection_key),weight_sha256=sha(snapshot/"model.pt"),candidates_sha256=sha(snapshot/"candidates.npz")))
            row=dict(epoch=epoch+1,train_loss=float(np.mean(values)),val_loss=val,optimizer_updates=updates,seconds=time.perf_counter()-tick)
            history.append(row)
            save_torch(folder/"recovery.pt",dict(registration=registration,initial_hash=initial,epoch=epoch+1,updates=updates,
                model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),history=history,snapshots=snapshots,rng=rng_state()))
            dump(folder/"snapshots.json",snapshots);write_csv(folder/"training_metrics.csv",history)
            dump(folder/"status.json",dict(status="running",**row,test_evaluated=False));print(json.dumps(row),flush=True)
        if updates!=2200 or len(snapshots)!=20:raise AssertionError("Formal budget changed")
        selection=finalize(folder,data.splits["valid"])
        if selection["epoch_threshold_pairs"]!=620:raise AssertionError("Selection budget changed")
        selection=export_ranked_selection(folder,data.splits["valid"],selection,config["method"],seed)
        dump(folder/"status.json",dict(status="complete",epoch=100,updates=updates,engineering_only=False,test_evaluated=False))
        return selection
    except BaseException as error:
        dump(folder/"status.json",dict(status="failed_or_interrupted",epoch=len(history),updates=updates,error=repr(error),test_evaluated=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--data-root",type=Path,default=ROOT)
    p.add_argument("--assets",type=Path);p.add_argument("--output",type=Path)
    mode=p.add_mutually_exclusive_group();mode.add_argument("--dry-run",action="store_true");mode.add_argument("--smoke",action="store_true");mode.add_argument("--train",action="store_true");mode.add_argument("--ready-check",action="store_true")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu");p.add_argument("--seed",type=int,choices=[42,43,44],default=42)
    p.add_argument("--approval",type=Path);p.add_argument("--probe",type=Path);p.add_argument("--resume",action="store_true")
    args=p.parse_args();config=read(args.config)
    validate(config,require_parameters=args.smoke or args.train or args.ready_check)
    if not args.smoke and not args.train and not args.ready_check:
        try:validate(config);ready=True
        except ValueError:ready=False
        print(json.dumps(dict(passed=True,dry_run=True,parameters_configured=ready,adapter_implemented=True,
            formal_launch_ready=False,human_decision_required=True,optimizer_updates=0,config_sha256=canonical(config))));return
    if args.output is None and not args.ready_check:raise ValueError("An explicit fresh output directory is required")
    # Fail before loading data or accessing CUDA if the human decision is absent.
    if args.train or args.ready_check:
        if args.approval is None or args.probe is None:raise ValueError("--approval and --probe required")
        require_review(config,read(args.approval))
        if config["method"]=="P0":verify_audit_file(read(args.approval),args.approval.parent)
    data=corpus(args.data_root,args.assets)
    if args.smoke:smoke(config,args.output,args.device,data);return
    if args.ready_check:
        if args.device!="cuda":raise ValueError("Formal readiness requires CUDA")
        configure_device(args.device);probe=read(args.probe)
        if probe.get("binding")!=binding(config,data,args.device) or not probe.get("passed") or not probe.get("exact_recovery") or probe.get("batch_size")!=8:
            raise ValueError("Stale or missing full-geometry CUDA probe")
        print(json.dumps(dict(ready=True,training_started=False,optimizer_updates=0)));return
    from filelock import FileLock
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(args.output.parent/".mechanism_runner.lock"),timeout=0):
        train(config,data,args.output,args.seed,args.device,read(args.approval),read(args.probe),args.resume,args.approval.parent)
