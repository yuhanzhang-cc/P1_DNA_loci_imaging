"""Synthetic integration checks, kept outside frozen v3's test inventory."""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from spotiflow.model import Spotiflow
from spotiflow.model.config import SpotiflowModelConfig

from deepblink_torch.experiment_log import write_json, sha256
from .engine import EpochSampler, DeterministicSamples, default_training, train_native


def compare(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys()==b.keys() and all(compare(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)):
        return len(a)==len(b) and all(compare(x,y) for x,y in zip(a,b))
    return a==b


def small_model():
    torch.manual_seed(123)
    return Spotiflow(SpotiflowModelConfig(initial_fmaps=4, levels=2, n_convs_per_level=3, batch_norm=False))


def run_checks(output, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    epoch = [0]
    sampler = EpochSampler(range(17), 42, lambda:epoch[0])
    first = list(sampler)
    assert sorted(first)==list(range(17)) and first==list(sampler)
    epoch[0]=1
    assert first!=list(sampler)
    class Fake:
        _sigma=1
        def __len__(self):return 2
        def __getitem__(self,index):return torch.rand(3),np.random.rand(),random.random()
    wrapped=DeterministicSamples(Fake(),42,lambda:epoch[0])
    a,b=wrapped[0],wrapped[0]
    assert torch.equal(a[0],b[0]) and a[1:]==b[1:]
    yy,xx=np.mgrid[:64,:64]
    centers=[(20,30),(35,25),(24,41),(38,39)]
    images=[np.exp(-((yy-y)**2+(xx-x)**2)/4).astype(np.float32) for y,x in centers]
    spots=[np.array([[y,x]],dtype=np.float32) for y,x in centers]
    config=default_training(epochs=2,batch_size=2,crop_size=32)
    continuous=small_model()
    ca=train_native(continuous,images,spots,images[:2],spots[:2],output/"continuous",
                    config=config,period=1,device=device)
    interrupted=small_model()
    cb=train_native(interrupted,images,spots,images[:2],spots[:2],output/"recovered",
                    config=config,period=1,stop_after=1,device=device)
    assert len(cb.history)==1
    resume=output/"recovered/recovery/last.ckpt"
    before=torch.load(resume,map_location="cpu",weights_only=False)
    assert before["optimizer_states"] and before["lr_schedulers"] and before["callbacks"]
    restored=small_model()
    cc=train_native(restored,images,spots,images[:2],spots[:2],output/"recovered",
                    config=config,period=1,resume=resume,device=device)
    delta=max(float((continuous.state_dict()[k]-restored.state_dict()[k]).abs().max()) for k in continuous.state_dict())
    if delta!=0:
        raise AssertionError(f"Continuous/resumed model differs: {delta}")
    ac=torch.load(output/"continuous/recovery/last.ckpt",map_location="cpu",weights_only=False)
    bc=torch.load(output/"recovered/recovery/last.ckpt",map_location="cpu",weights_only=False)
    assert compare(ac["optimizer_states"],bc["optimizer_states"])
    assert compare(ac["lr_schedulers"],bc["lr_schedulers"])
    assert len(ca.history)==len(cc.history)==2 and ac["global_step"]==bc["global_step"]==4
    assert sha256(output/"continuous/epoch_001/best.pt")!=sha256(output/"continuous/epoch_002/best.pt")
    reloaded=Spotiflow.from_folder(output/"recovered/epoch_002")
    assert all(torch.equal(v.cpu(),reloaded.state_dict()[k].cpu()) for k,v in restored.state_dict().items())
    from .evaluation import bind_threshold
    bind_threshold(output/"recovered/epoch_002",.7)
    bound=Spotiflow.from_folder(output/"recovered/epoch_002")
    assert np.allclose(bound._prob_thresh,[.7])
    result=dict(passed=True,checks=["no-replacement sampler", "augmentation keyed by seed/epoch/image",
        "actual distinct epoch snapshots", "continuous versus interrupted/resumed model exact",
        "optimizer state exact", "scheduler state exact", "native weight reload exact", "deployment threshold binding"],
        model_max_abs=delta,synthetic_only=True,full_training_started=False,
        device=device,torch_version=torch.__version__,cuda_tested=device=="cuda")
    write_json(output/"checks.json",result)
    print(result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    run_checks(parser.parse_args().output)
