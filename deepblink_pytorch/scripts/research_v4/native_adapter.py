"""Explicit random-initialization Spotiflow control using the audited native runner."""
from pathlib import Path
import numpy as np
import torch
import yaml
from .common import dump,read_json,sha256
from .engine import reset_rng,model_hash


def scratch_model(config_path,seed):
    from spotiflow.model import Spotiflow,SpotiflowModelConfig
    reset_rng(seed)
    config=SpotiflowModelConfig.from_config_file(Path(config_path))
    model=Spotiflow(config)
    if sum(p.numel() for p in model.parameters())!=11826115:raise ValueError("Scratch/transfer architectures differ")
    return model


def train_scratch(data,folder,config_path,*,seed,device="cuda",resume=False,engineering=False):
    if not engineering and device!="cuda":raise ValueError("Formal scratch control requires CUDA")
    from spotiflow_overnight.engine import train_native,default_training,evaluate_snapshot
    from spotiflow_overnight.evaluation import finalize_finetune
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/"status.json").exists() and read_json(folder/"status.json")["status"]=="complete":
        if not resume:raise ValueError("Completed native scratch task exists")
        return read_json(folder/"selection.json")
    recovery=folder/"recovery/last.ckpt"
    if (folder/"initialization.json").exists() and not resume:raise ValueError("Partial scratch task needs audited resume")
    if resume and not recovery.exists():raise ValueError("No complete native epoch recovery")
    model=scratch_model(config_path,seed)
    init=dict(seed=seed,pretrained_weights_loaded=False,model_config_sha256=sha256(config_path),
              initial_model_state_sha256=model_hash(model),parameter_count=sum(p.numel() for p in model.parameters()))
    if (folder/"initialization.json").exists() and read_json(folder/"initialization.json")!=init:raise ValueError("Scratch initialization changed")
    dump(folder/"initialization.json",init)
    config=default_training(epochs=1 if engineering else 100,batch_size=2 if engineering else 4,crop_size=128 if engineering else 256)
    config.finetuned_from=None
    dump(folder/"status.json",dict(status="running",seed=seed,engineering_only=engineering))
    try:
        train_native(model,data["train"]["x"],data["train"]["y"],data["valid"]["x"],data["valid"]["y"],
            folder,seed=seed,config=config,device=device,resume=recovery if resume else None,
            evaluate=None if engineering else evaluate_snapshot(dict(images=data["valid"]["x"],**{k:data["valid"][k] for k in ("y","sample_id","original_shape")}),device),
            event=lambda event,**fields:print(__import__("json").dumps(dict(event=event,seed=seed,**{k:v for k,v in fields.items() if k!="seed"})),flush=True))
        if engineering:
            summary=read_json(folder/"optimizer_summary.json")
            if summary["optimizer_updates"]!=1:raise ValueError("Native engineering step cap exceeded")
            result=dict(passed=True,discarded_optimizer_updates=1,pretrained_weights_loaded=False,parameters=11826115,
                        initial_hash=init["initial_model_state_sha256"],weights_changed=model_hash(model)!=init["initial_model_state_sha256"],
                        device=device,engineering_only=True)
            if not result["weights_changed"]:raise AssertionError("Native scratch optimizer did not update weights")
            dump(folder/"checks.json",result)
            return result
        result=finalize_finetune(folder,dict(images=data["valid"]["x"],**{k:data["valid"][k] for k in ("y","sample_id","original_shape")}))
        dump(folder/"status.json",dict(status="complete",seed=seed,pretrained_weights_loaded=False,test_evaluated=False))
        return result
    except BaseException as error:
        dump(folder/"status.json",dict(status="failed_or_interrupted",error=repr(error),seed=seed,engineering_only=engineering))
        raise


def smoke(folder,config_path):
    folder=Path(folder)
    if (folder/"checks.json").exists():return read_json(folder/"checks.json")
    rng=np.random.default_rng(729)
    images=[rng.normal(size=(128,128)).astype(np.float32) for _ in range(2)]
    truth=[np.asarray([[30.,40.],[70.,80.]],dtype=np.float32) for _ in images]
    data=dict(train=dict(x=images,y=truth),valid=dict(x=images[:1],y=truth[:1]))
    return train_scratch(data,folder,config_path,seed=42,device="cpu",engineering=True)
