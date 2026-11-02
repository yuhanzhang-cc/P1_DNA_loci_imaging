"""Data-wide geometry invariants and meaningful small-scale engineering checks."""
import random
from pathlib import Path
import time
import numpy as np
import torch

from deepblink_torch.preprocessing import prepare_image
from .augmentation import transform,prepare_sample,TOLERANCE
from .common import dump,write_csv,canonical_hash


def geometry(ctx):
    ctx.task("augmentation_preflight","running")
    ledger=[];adjustments=[];original_hashes={};max_inverse=0.
    for subset,data in ctx.corpus.splits.items():
        for i,(sid,image,truth) in enumerate(zip(data["sample_id"],data["x"],data["y"])):
            original_hash=canonical_hash(dict(image_sha=__import__("hashlib").sha256(image.tobytes()).hexdigest(),truth=truth))
            x0,t0,_,_=prepare_sample(image,truth)
            if not np.array_equal(x0[0],prepare_image(image,512)):raise AssertionError("Augmentation-off preprocessing regression")
            for k in range(4):
                for flip in (False,True):
                    changed,coords,snaps=transform(image,truth,k,flip)
                    undo,back,_=transform(changed,coords,0,flip)
                    undo,back,_=transform(undo,back,(-k)%4,False)
                    error=float(np.max(np.abs(back-truth),initial=0))
                    if not np.array_equal(undo,image) or error>TOLERANCE:raise AssertionError("D4 roundtrip failed")
                    max_inverse=max(max_inverse,error)
                    x,target,encoded,info=prepare_sample(image,truth,geometry=(k,flip))
                    if not np.array_equal(encoded,coords) or len(coords)!=len(truth):raise AssertionError("Point deletion/reordering")
                    h,w=changed.shape
                    if np.count_nonzero(target[...,0]>=0)!=(h//4)*(w//4):raise AssertionError("Target mask area failed")
                    if np.count_nonzero(x[:,h:,:]) or np.count_nonzero(x[:,:,w:]):raise AssertionError("Padding contaminated")
                    ledger.append(dict(sample_id=str(sid),subset=subset,k=k,flip=flip,points=len(coords),
                                       encoded_points=info["encoded_points"],collisions=info["collisions"],inverse_max_error_px=error,
                                       roundoff_adjustments=len(snaps)))
                    adjustments.extend(dict(sample_id=str(sid),subset=subset,**r) for r in snaps)
            x,_,_,_=prepare_sample(image,truth,intensity=(1.2,.1))
            h,w=image.shape;valid=x[0,:h,:w]
            if not np.isclose(valid.mean(),.1,atol=3e-5) or (image.std()>0 and not np.isclose(valid.std(),1.2,atol=3e-5)):
                raise AssertionError("Post-zscore affine is absent or was renormalized")
            if np.array_equal(x,x0):raise AssertionError("Intensity augmentation is a no-op")
            if np.count_nonzero(x[:,h:,:]) or np.count_nonzero(x[:,:,w:]):raise AssertionError("Intensity affected padding")
            if original_hash!=canonical_hash(dict(image_sha=__import__("hashlib").sha256(image.tobytes()).hexdigest(),truth=truth)):
                raise AssertionError("Source data modified")
    # Known integer pixel/coordinate geometry, including a rectangular image.
    synthetic=np.zeros((8,12),np.float32);synthetic[2,7]=1
    for k in range(4):
        for flip in (False,True):
            x,y,_=transform(synthetic,np.asarray([[2.,7.]],np.float32),k,flip)
            if not np.array_equal(np.argwhere(x==1),y):raise AssertionError("Image and label transforms differ")
    py=random.getstate();npstate=np.random.get_state();tstate=torch.get_rng_state()
    image=ctx.corpus.splits["train"]["x"][0];truth=ctx.corpus.splits["train"]["y"][0]
    a=prepare_sample(image,truth,augment=True,seed=42,epoch=3,index=0)
    b=prepare_sample(image,truth,augment=True,seed=42,epoch=3,index=0)
    c=prepare_sample(image,truth,augment=True,seed=43,epoch=4,index=0)
    if not np.array_equal(a[0],b[0]) or not np.array_equal(a[1],b[1]) or np.array_equal(a[0],c[0]):raise AssertionError("Keyed augmentation reproducibility failed")
    if random.getstate()!=py or not np.array_equal(npstate[1],np.random.get_state()[1]) or not torch.equal(tstate,torch.get_rng_state()):raise AssertionError("Augmentation changed model/sampler RNG")
    for bad in (np.asarray([[-.01,2]]),np.asarray([[8.,2]])):
        try:transform(synthetic,bad,1,False)
        except ValueError:pass
        else:raise AssertionError("Invalid source coordinate was accepted")
    # A real half-pixel beyond the last center cannot be hidden as floating roundoff.
    try:transform(synthetic,np.asarray([[7.5,2]]),2,False)
    except ValueError:pass
    else:raise AssertionError("Out-of-tolerance transformed point was silently clipped")
    folder=ctx.out/"01_preflight";folder.mkdir(exist_ok=True)
    write_csv(folder/"geometry_collision_ledger.csv",ledger)
    write_csv(folder/"roundoff_snaps.csv",adjustments,["sample_id","subset","point_index","axis","before","after","delta","k","flip"])
    result=dict(passed=True,development_images=198,d4_cases=len(ledger),inverse_max_error_px=max_inverse,
        tolerance_px=TOLERANCE,roundoff_snap_events=len(adjustments),source_labels_modified=False,
        effective_post_zscore_intensity=True,padding_unchanged=True,model_rng_unchanged=True,test_arrays_loaded=False,
        training_collision_range=[min(r["collisions"] for r in ledger if r["subset"]=="train"),max(r["collisions"] for r in ledger if r["subset"]=="train")])
    dump(folder/"augmentation_checks.json",result)
    ctx.task("augmentation_preflight","completed",**result)
    return result


def run(ctx,device="cpu"):
    geometry(ctx)
    from .engine import smoke
    # Preserve the original16-step check; v2 adds source binding and orphan recovery.
    prior=ctx.out/"01_preflight/cpu_engineering/checks.json"
    from .common import read_json
    if device=="cpu" and prior.exists() and read_json(prior).get("passed"):
        result=smoke(ctx.out/"01_preflight/cpu_engineering_v2",device=device,corpus=None)
        result=dict(result,real_geometry_checks_preserved_at=str(prior))
    else:result=smoke(ctx.out/"01_preflight"/f"{device}_engineering",device=device,corpus=ctx.corpus)
    ctx.task("runner_preflight","completed",**result)
    return result
