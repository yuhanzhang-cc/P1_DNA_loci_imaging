"""Portable epoch-boundary recovery and full-image selection for existing models."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from deepblink_torch.cli import predict
from deepblink_torch.losses import combined_dice_rmse_loss
from deepblink_torch.round2_protocol import training_orders,decode_candidates,dense_heatmap_loss
from .augmentation import RecipeDataset,prepare_sample
from .common import dump,read_json,write_csv,sha256,canonical_hash,replace_with_retry
from .evaluation import policies_probability,sweep,selection_key,save_candidates,load_candidates,export_predictions,row_policy


MODELS=dict(R0="deepblink",G0="deeploci_grid_control",H0="deeploci_heatmap",R1="deeploci_detail",S1="deeploci_detail_simple",
            R2="deeploci_adaptive_context",S2="deeploci_fixed_context",Qglobal="deeploci_global_query",
            A="deepblink_masked_se",B="deeploci_local",D="deeploci_evidence",D2="deeploci_feature_evidence")


def reset_rng(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng(state):
    random.setstate(state["python"])
    n=state["numpy"];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def preserve_rng():
    state=rng_state()
    try:yield
    finally:set_rng(state)


def configure_device(device):
    if device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no fallback")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False


def model_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(key,device="cpu"):
    if key not in MODELS:raise ValueError("Unregistered model")
    return predict._build_model_for_checkpoint(MODELS[key],cell_size=1 if key=="H0" else 4,device=torch.device(device)).to(memory_format=torch.channels_last)


def forward(model,images,targets):
    if getattr(model,"requires_valid_mask",False):
        mask=torch.nn.functional.interpolate((targets[...,0]>=0).float().unsqueeze(1),size=images.shape[-2:],mode="nearest")
        return model(images,valid_mask=mask)
    return model(images)


def native_epoch(model,batches,device,loss_fn,optimizer=None):
    model.train(optimizer is not None);losses=[]
    with torch.set_grad_enabled(optimizer is not None):
        for images,targets in batches:
            images=images.to(device).contiguous(memory_format=torch.channels_last);targets=targets.to(device)
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            prediction=forward(model,images,targets)
            loss=loss_fn(targets,prediction.float())
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            if optimizer is not None:
                loss.backward()
                norms=[p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
                if not torch.isfinite(torch.stack(norms).norm()):raise FloatingPointError("Nonfinite gradients")
                optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def infer_candidates(model,data,device,canvas=512):
    model.eval();result=[]
    dense=getattr(model,"output_representation","")=="dense_heatmap_offset_logits"
    for image,shape in zip(data["x"],data["original_shape"]):
        from deepblink_torch.preprocessing import prepare_image
        tensor=torch.from_numpy(prepare_image(image,canvas)[None,None]).to(device).contiguous(memory_format=torch.channels_last)
        output=predict._forward_with_geometry(model,tensor,int(shape[0]),int(shape[1]))
        result.append(decode_candidates(output[0].float().cpu().numpy(),shape,cell_size=1 if dense else 4,dense=dense))
    return result


def save_torch(path,payload):
    path=Path(path);temporary=path.with_name(path.name+".tmp")
    torch.save(payload,temporary);replace_with_retry(temporary,path)


def model_payload(model,key,seed,epoch,config,threshold=None):
    payload=dict(model_state_dict=model.state_dict(),model_name=MODELS[key],seed=seed,epoch=epoch,
        cell_size=1 if key=="H0" else 4,input_size=config["canvas"],preprocessing="normalize_original_then_zero_pad_right_bottom",
        output_representation=getattr(model,"output_representation","grid"),parameter_count=sum(p.numel() for p in model.parameters()),
        protocol_id=config["protocol_id"],architecture_config=dict(model_name=MODELS[key],round2_key=key),
        training_augmentation=config["augment"],model_state_sha256=model_hash(model))
    if threshold is not None:payload.update(calibrated_probability=float(threshold),calibration_source="frozen_development_validation")
    return payload


def finalize(folder,data):
    records=read_json(folder/"snapshots.json")
    all_rows=[]
    for r in records:
        path=folder/f"epoch_{r['epoch']:03d}"
        if sha256(path/"model.pt")!=r["weight_sha256"] or sha256(path/"candidates.npz")!=r["candidates_sha256"]:raise ValueError("Snapshot hash mismatch")
        all_rows.extend(read_json(path/"threshold_metrics.json"))
    primary=max(all_rows,key=selection_key)
    native_epoch_number=min(records,key=lambda r:(r["native_val_loss"],r["epoch"]))["epoch"]
    secondary=max([r for r in all_rows if r["epoch"]==native_epoch_number],key=selection_key)
    exports={}
    for role,choice in (("primary",primary),("native_secondary",secondary)):
        source=folder/f"epoch_{choice['epoch']:03d}"
        payload=torch.load(source/"model.pt",map_location="cpu",weights_only=True)
        threshold=0. if choice["threshold_kind"]=="all" else 1. if choice["threshold_kind"]=="none" else choice["threshold"]
        payload.update(calibrated_probability=float(threshold),calibration_source="frozen_development_validation")
        target=folder/("best_model.pt" if role=="primary" else "native_selected_model.pt")
        save_torch(target,payload)
        candidates=load_candidates(source/"candidates.npz",data)
        metrics=export_predictions(folder/role,candidates,data,row_policy(choice))
        if any(metrics[k]!=choice[k] for k in ("tp","fp","fn")):raise AssertionError("Sweep/export counts differ")
        exports[role]=dict(**choice,metrics=metrics,checkpoint_sha256=sha256(target),model_state_sha256=payload["model_state_sha256"],calibrated_probability=threshold)
    result=dict(**exports,checkpoint_count=len(records),epoch_threshold_pairs=len(all_rows),test_evaluated=False,validation_selection_optimism=True)
    write_csv(folder/"all_epoch_threshold_metrics.csv",all_rows)
    dump(folder/"selection.json",result)
    return result


def train(data,folder,*,key="G0",augment=False,seed=42,device="cuda",epochs=100,batch_size=8,canvas=512,
          period=5,resume=False,stop_after=None,engineering=False,source_lock=None,event=None):
    if not engineering and (device!="cuda" or epochs!=100 or batch_size!=8 or canvas!=512 or period!=5):raise ValueError("Formal fixed protocol/device changed")
    configure_device(device);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    config=dict(protocol_id="pooled220_recipe_v4",key=key,augment=bool(augment),seed=seed,device=device,epochs=epochs,
        batch_size=batch_size,canvas=canvas,period=period,optimizer="Adam",lr=.001,weight_decay=0.,precision="float32",
        tf32=False,engineering_only=engineering,source_lock=source_lock)
    path=folder/"registered_config.json"
    if path.exists() and read_json(path)!=config:raise ValueError("Trial configuration changed")
    if (folder/"status.json").exists() and read_json(folder/"status.json")["status"]=="complete":
        if not resume:raise ValueError("Completed task exists; use audited resume to skip")
        return read_json(folder/"selection.json")
    if path.exists() and not resume:raise ValueError("Partial task requires audited resume")
    if resume and not (folder/"recovery.pt").exists():raise ValueError("No epoch recovery file")
    reset_rng(seed);model=make_model(key,device)
    initial=model_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    dense=key=="H0";loss_fn=dense_heatmap_loss if dense else combined_dice_rmse_loss
    train_data=RecipeDataset(data["train"],seed=seed,augment=augment,canvas=canvas,dense=dense)
    valid_data=RecipeDataset(data["valid"],seed=seed,augment=False,canvas=canvas,dense=dense)
    orders=training_orders(len(train_data),epochs,seed)
    dump(folder/"initialization.json",dict(initial_model_state_sha256=initial,sampler_orders_sha256=canonical_hash(orders),
         parameters=sum(p.numel() for p in model.parameters()),training_images=len(train_data),epochs=epochs))
    dump(path,config)
    history=[];snapshots=[];first_epoch=0;updates=0
    if resume:
        saved=torch.load(folder/"recovery.pt",map_location="cpu",weights_only=True)
        if saved["config"]!=config or saved["initial_hash"]!=initial:raise ValueError("Recovery identity differs")
        model.load_state_dict(saved["model_state_dict"]);optimizer.load_state_dict(saved["optimizer_state_dict"])
        history=saved["history"];snapshots=saved["snapshots"];first_epoch=saved["epoch"];updates=saved["updates"]
        train_data.adjustment_records=saved.get("roundoff_records",[])
        set_rng(saved["rng"])
    dump(folder/"status.json",dict(status="running",epoch=first_epoch,engineering_only=engineering))
    try:
        for epoch in range(first_epoch,epochs):
            train_data.epoch=epoch
            tick=time.perf_counter()
            train_dl=DataLoader(train_data,batch_size=batch_size,sampler=orders[epoch],num_workers=0,
                generator=torch.Generator().manual_seed(seed+100000+epoch))
            valid_dl=DataLoader(valid_data,batch_size=batch_size,shuffle=False,num_workers=0,
                generator=torch.Generator().manual_seed(seed+200000+epoch))
            train_loss=native_epoch(model,train_dl,device,loss_fn,optimizer)
            updates+=len(train_dl)
            val_loss=native_epoch(model,valid_dl,device,loss_fn)
            if (epoch+1)%period==0:
                snapshot=folder/f"epoch_{epoch+1:03d}"
                if snapshot.exists():
                    if not resume or epoch+1<=first_epoch:raise FileExistsError("Unexpected snapshot identity; audit required")
                    # An interrupted epoch may have exported a snapshot before committing
                    # recovery. Preserve it, then replay from the previous complete epoch.
                    import uuid
                    quarantine=folder/"orphaned_snapshots"/f"epoch_{epoch+1:03d}_{uuid.uuid4().hex[:8]}"
                    quarantine.parent.mkdir(exist_ok=True)
                    if not snapshot.resolve().is_relative_to(folder.resolve()) or not quarantine.resolve().is_relative_to(folder.resolve()):raise ValueError("Unsafe recovery path")
                    snapshot.rename(quarantine)
                    dump(quarantine/"recovery_audit.json",dict(reason="snapshot beyond last committed recovery boundary",last_committed_epoch=first_epoch,replayed_epoch=epoch+1))
                snapshot.mkdir()
                with preserve_rng():
                    save_torch(snapshot/"model.pt",model_payload(model,key,seed,epoch+1,config))
                    candidates=infer_candidates(model,data["valid"],device,canvas)
                    save_candidates(snapshot/"candidates.npz",candidates,data["valid"])
                    scores=[dict(epoch=epoch+1,native_val_loss=val_loss,**r) for r in sweep(candidates,data["valid"],policies_probability())]
                    dump(snapshot/"threshold_metrics.json",scores)
                    snapshots.append(dict(epoch=epoch+1,native_val_loss=val_loss,best_at_epoch=max(scores,key=selection_key),
                        weight_sha256=sha256(snapshot/"model.pt"),candidates_sha256=sha256(snapshot/"candidates.npz")))
            row=dict(epoch=epoch+1,train_loss=train_loss,val_loss=val_loss,optimizer_updates=updates,seconds=time.perf_counter()-tick,learning_rate=.001)
            history.append(row)
            dump(folder/"snapshots.json",snapshots);write_csv(folder/"training_metrics.csv",history)
            dump(folder/"progress.json",dict(**row,max_epochs=epochs,checkpoints=len(snapshots)))
            saved=dict(config=config,initial_hash=initial,epoch=epoch+1,updates=updates,model_state_dict=model.state_dict(),
                       optimizer_state_dict=optimizer.state_dict(),history=history,snapshots=snapshots,rng=rng_state(),
                       roundoff_records=train_data.adjustment_records,loader_rng="independent_epoch_reconstructed_generators",
                       scheduler_state=None)
            save_torch(folder/"recovery.pt",saved)
            if event:event("epoch_completed",key=key,augment=augment,seed=seed,**row)
            if stop_after is not None and epoch+1>=stop_after:
                dump(folder/"status.json",dict(status="engineering_interrupted" if engineering else "interrupted",epoch=epoch+1,updates=updates))
                return saved
        selection=finalize(folder,data["valid"])
        write_csv(folder/"roundoff_snaps.csv",train_data.adjustment_records,["epoch","image_index","point_index","axis","before","after","delta","k","flip"])
        if not engineering and (len(history)!=100 or updates!=2200 or len(snapshots)!=20):raise ValueError("Formal training budget incomplete")
        dump(folder/"status.json",dict(status="complete",epoch=epochs,updates=updates,engineering_only=engineering,test_evaluated=False))
        return selection
    except BaseException as error:
        dump(folder/"status.json",dict(status="failed_or_interrupted",error=repr(error),epoch=len(history),engineering_only=engineering))
        raise


def equal_nested(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b


def smoke(folder,*,device="cpu",corpus=None):
    folder=Path(folder)
    contract=canonical_hash({p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name("augmentation.py"),Path(__file__).with_name("evaluation.py"))})
    if (folder/"checks.json").exists():
        result=read_json(folder/"checks.json")
        if result["device"]!=device or result["torch_version"]!=str(torch.__version__) or result.get("source_contract")!=contract:raise ValueError("Smoke backend/source changed; use a new revision directory")
        return result
    folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(81)
    images=[rng.normal(size=(128,128)).astype(np.float32) for _ in range(2)]
    truth=[np.asarray([[20.2,30.3],[70.1,90.8]],dtype=np.float32) for _ in images]
    d=dict(x=images,y=truth,sample_id=np.asarray([f"synthetic_{i}" for i in range(2)]),original_shape=np.asarray([[128,128]]*2))
    v={k:value[:1] for k,value in d.items()}
    data=dict(train=d,valid=v)
    checks=[];initial=[]
    for augment in (False,True):
        base=folder/("aug" if augment else "ref")
        train(data,base/"continuous",augment=augment,device=device,epochs=2,batch_size=2,canvas=128,period=1,engineering=True)
        train(data,base/"resumed",augment=augment,device=device,epochs=2,batch_size=2,canvas=128,period=1,stop_after=1,engineering=True)
        orphan=base/"resumed/epoch_002";orphan.mkdir()
        (orphan/"interrupted.txt").write_text("Synthetic uncommitted snapshot for recovery probe\n",encoding="utf-8")
        train(data,base/"resumed",augment=augment,device=device,epochs=2,batch_size=2,canvas=128,period=1,resume=True,engineering=True)
        a=torch.load(base/"continuous/recovery.pt",map_location="cpu",weights_only=True)
        b=torch.load(base/"resumed/recovery.pt",map_location="cpu",weights_only=True)
        delta=max(float((a["model_state_dict"][k]-b["model_state_dict"][k]).abs().max()) for k in a["model_state_dict"])
        if delta!=0 or not equal_nested(a["optimizer_state_dict"],b["optimizer_state_dict"]) or not equal_nested(a["rng"],b["rng"]):raise AssertionError("Exact recovery failed")
        payload=torch.load(base/"continuous/best_model.pt",map_location="cpu",weights_only=True)
        loaded,_=predict._load_model(base/"continuous/best_model.pt",torch.device("cpu"))
        if model_hash(loaded)!=payload["model_state_sha256"] or "calibrated_probability" not in payload:raise AssertionError("Export/reload/threshold failed")
        hashes=read_json(base/"continuous/initialization.json");initial.append(hashes)
        checks.append(dict(augmentation=augment,model_max_abs=delta,optimizer_exact=True,rng_exact=True,scheduler_state=None,
                           distinct_snapshots=len({r["weight_sha256"] for r in a["snapshots"]})==2,export_reload=True,threshold_bound=True,
                           orphan_preserved=len(list((base/"resumed/orphaned_snapshots").glob("*/interrupted.txt")))==1))
    if initial[0]!=initial[1]:raise AssertionError("Paired initialization or sampler differs")
    real=[]
    if corpus is not None:
        configure_device(device);reset_rng(42);model=make_model("G0",device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        data=corpus.splits["train"]
        for size in (256,512):
            indices=[i for i,s in enumerate(data["original_shape"]) if int(s[0])==size][:8 if device=="cuda" else 1]
            samples=[prepare_sample(data["x"][i],data["y"][i],augment=True,seed=42,epoch=0,index=i) for i in indices]
            x=torch.from_numpy(np.stack([a[0] for a in samples]));target=torch.from_numpy(np.stack([a[1] for a in samples]))
            loss=native_epoch(model,[(x,target)],device,combined_dice_rmse_loss,optimizer)
            real.append(dict(original_size=size,batch_size=len(indices),loss=loss,finite=True))
        del model,optimizer
    result=dict(passed=True,device=device,torch_version=str(torch.__version__),cuda_tested=device=="cuda",engineering_only=True,
                source_contract=contract,synthetic_optimizer_updates=8,discarded_real_optimizer_updates=len(real),paired_initialization_and_order=True,
                checks=checks,real_batches=real,formal_training_started=False)
    dump(folder/"checks.json",result)
    return result
