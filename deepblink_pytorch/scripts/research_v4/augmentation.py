"""D4 geometry, logged numerical boundary handling and effective post-zscore jitter."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from deepblink_torch.data_protocol import get_prediction_matrix
from deepblink_torch.preprocessing import normalize_image
from deepblink_torch.round2_protocol import encode_dense_targets
from .common import canonical_hash


TOLERANCE=1e-4


def transform(image,truth,k=0,flip=False):
    h,w=image.shape
    coords=np.asarray(truth,dtype=np.float64).copy().reshape(-1,2)
    if np.any(coords<0) or np.any(coords>=[h,w]) or not np.isfinite(coords).all():raise ValueError("Invalid source coordinates")
    transformed=np.rot90(image,int(k)%4)
    for _ in range(int(k)%4):
        coords=np.column_stack([w-1-coords[:,1],coords[:,0]])
        h,w=w,h
    if flip:
        transformed=np.flip(transformed,axis=1)
        coords[:,1]=w-1-coords[:,1]
    adjustments=[]
    for i,point in enumerate(coords):
        for axis,limit in enumerate((h,w)):
            value=float(point[axis])
            if value<0:
                if value < -TOLERANCE:raise ValueError("Geometry produced a negative coordinate beyond numerical tolerance")
                adjustments.append(dict(point_index=i,axis=axis,before=value,after=0.,delta=-value,k=int(k),flip=bool(flip)))
                point[axis]=0.
            elif value>=limit:
                if value-limit>TOLERANCE:raise ValueError("Geometry produced a coordinate beyond numerical tolerance")
                after=float(np.nextafter(np.float32(limit),np.float32(0)))
                adjustments.append(dict(point_index=i,axis=axis,before=value,after=after,delta=after-value,k=int(k),flip=bool(flip)))
                point[axis]=after
    coords=coords.astype(np.float32)
    if np.any(coords<0) or np.any(coords>=[h,w]):raise ValueError("Transformed coordinates remain invalid")
    return np.ascontiguousarray(transformed),coords,adjustments


def keyed_rng(seed,epoch,index):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(epoch),int(index),20261101]))


def prepare_sample(image,truth,*,augment=False,seed=42,epoch=0,index=0,canvas=512,geometry=None,intensity=None,dense=False):
    rng=keyed_rng(seed,epoch,index)
    k=int(rng.integers(4)) if augment else 0
    flip=bool(rng.random()<.5) if augment else False
    if geometry is not None:k,flip=geometry
    original_shape=image.shape
    image,coords,adjustments=transform(image,truth,k,flip)
    h,w=image.shape
    x=normalize_image(image)
    gain,bias=1.,0.
    applied=bool(augment and rng.random()<.8)
    if applied:gain,bias=float(rng.uniform(.8,1.2)),float(rng.uniform(-.2,.2))
    if intensity is not None:gain,bias=intensity;applied=True
    x=np.asarray(x*gain+bias,dtype=np.float32)
    if canvas<max(h,w) or h%4 or w%4:raise ValueError("Unsupported canvas/geometry")
    padded=np.pad(x,((0,canvas-h),(0,canvas-w)),mode="constant")
    if dense:
        target=encode_dense_targets(coords,(h,w),canvas)
        occupied=int(np.count_nonzero(target[...,3]>0))
    else:
        target=get_prediction_matrix(coords,canvas,4).astype(np.float32)
        occupied=int(np.count_nonzero(target[...,0]>0))
        target[h//4:,:,0]=-1;target[:,w//4:,0]=-1
    info=dict(k=k,flip=flip,gain=gain,bias=bias,intensity_applied=applied,roundoff_adjustments=adjustments,
              points=len(coords),encoded_points=occupied,collisions=len(coords)-occupied,
              original_shape=list(original_shape),shape=[h,w],augmentation_key=[int(seed),int(epoch),int(index)])
    return np.ascontiguousarray(padded[None]),target,coords,info


class RecipeDataset(Dataset):
    def __init__(self,data,*,seed=42,augment=False,canvas=512,dense=False):
        self.data,self.seed,self.augment,self.canvas=data,seed,augment,canvas
        self.epoch=0
        self.dense=dense
        self.base=None
        self.geometry_expectations={}
        self.adjustment_count=0
        self.adjustment_records=[]
        if not augment:
            self.base=[prepare_sample(image,truth,canvas=canvas,dense=dense)[:2] for image,truth in zip(data["x"],data["y"])]
        else:
            for i,(image,truth) in enumerate(zip(data["x"],data["y"])):
                for k in range(4):
                    for flip in (False,True):
                        _,coords,logs=transform(image,truth,k,flip)
                        cells=np.floor(coords/(1 if dense else 4)).astype(int)
                        self.geometry_expectations[(i,k,flip)]=len(coords)-len(np.unique(cells,axis=0))

    def __len__(self):return len(self.data["x"])

    def __getitem__(self,index):
        if self.base is not None:x,target=self.base[index]
        else:
            x,target,_,info=prepare_sample(self.data["x"][index],self.data["y"][index],augment=True,
                seed=self.seed,epoch=self.epoch,index=index,canvas=self.canvas,dense=self.dense)
            if info["collisions"]!=self.geometry_expectations[(index,info["k"],info["flip"])]:raise ValueError("Geometry collision ledger drifted")
            self.adjustment_count+=len(info["roundoff_adjustments"])
            self.adjustment_records.extend(dict(epoch=self.epoch+1,image_index=int(index),**r) for r in info["roundoff_adjustments"])
        return torch.from_numpy(x),torch.from_numpy(target)
