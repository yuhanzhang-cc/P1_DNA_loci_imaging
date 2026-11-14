"""Parameter-identical H0/S0 adapter and configurable P0 supervision.

Coordinates are (y,x), in original-image pixels. Heatmaps stay centered at
floor(truth). S0/P0 offset activation is linear. P0 rules are prospective options.
"""
import numpy as np
import torch
from torch.nn import functional as F
from deepblink_torch.networks.dense_heatmap import UNetDeepLociHeatmap
from deepblink_torch.round2_protocol import encode_dense_targets, dense_heatmap_loss, decode_candidates
from deepblink_torch.losses import dice_score

PARAMETERS=671719
REPRESENTATION="dense_heatmap_signed_offset_v1"


class SignedHeatmap(UNetDeepLociHeatmap):
    output_representation=REPRESENTATION

    def forward(self,inputs):
        full,_=self._forward_first_decoder(inputs)
        return self.dense_head(full).permute(0,2,3,1).contiguous()


def make_model(method,device="cpu"):
    if method not in ("H0","S0","P0"):raise ValueError("Unknown isolated adapter")
    model=(UNetDeepLociHeatmap() if method=="H0" else SignedHeatmap()).to(device)
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:raise ValueError("Capacity contract changed")
    return model.to(memory_format=torch.channels_last)


def supervision(config):
    if config["method"]=="S0":
        if config["supervision"]!={"radius_px":0,"ownership":"single_floor_anchor","normalization":"anchor_mean"}:
            raise ValueError("S0 must preserve the H0 single-anchor target")
        return config["supervision"]
    if config["method"]!="P0":raise ValueError("Unknown mechanism")
    rule=config["supervision"];radius=rule.get("radius_px")
    if isinstance(radius,bool) or not isinstance(radius,int) or not 1<=radius<=8:
        raise ValueError("P0 radius is TODO or outside implemented 1..8 square support")
    if rule.get("ownership")!="nearest_truth_then_index" or rule.get("normalization")!="per_truth_mean":
        raise ValueError("P0 ownership/normalization is TODO or unsupported")
    return rule


def encode(truth,shape,canvas,config):
    rule=supervision(config);truth=np.asarray(truth,np.float32).reshape(-1,2)
    target=encode_dense_targets(truth,shape,canvas)
    owner=np.full((canvas,canvas),-1,dtype=np.int32);anchors=np.floor(truth).astype(int)
    if config["method"]=="S0":
        for i,(y,x) in enumerate(anchors):owner[y,x]=i
        return target,owner
    h,w=map(int,shape);best=np.full((h,w),np.inf,dtype=np.float64);r=rule["radius_px"]
    # Strict less-than retains the lower GT input index on an exact distance tie.
    for i,((y,x),point) in enumerate(zip(anchors,truth)):
        ys=slice(max(0,y-r),min(h,y+r+1));xs=slice(max(0,x-r),min(w,x+r+1))
        yy,xx=np.mgrid[ys,xs];d=(yy-point[0])**2+(xx-point[1])**2
        better=d<best[ys,xs];best[ys,xs][better]=d[better];owner[ys,xs][better]=i
    target[...,1:]=0
    for i,point in enumerate(truth):
        yy,xx=np.nonzero(owner==i)
        if not len(yy):raise ValueError("Truth has no owned support pixels; review required")
        target[yy,xx,1:3]=point-np.column_stack([yy,xx]);target[yy,xx,3]=1./len(yy)
    return target,owner


def loss(target,prediction,method):
    if method=="S0":return dense_heatmap_loss(target,prediction)
    if method!="P0":raise ValueError("Unknown loss")
    valid=target[...,0]>=0
    if not bool(valid.any()):raise ValueError("No valid pixels")
    labels,logits=target[...,0][valid],prediction[...,0][valid]
    heat=1-dice_score(labels,logits.sigmoid())+F.binary_cross_entropy_with_logits(logits,labels)
    weights=target[...,3][valid]
    if bool((weights>0).any()):
        errors=(prediction[...,1:3][valid]-target[...,1:3][valid]).square().sum(-1)
        rmse=torch.sqrt((errors*weights).sum()/weights.sum()+torch.finfo(prediction.dtype).eps)
    else:rmse=prediction[...,1:3].sum()*0
    return heat+2*rmse


def decode(prediction,shape):
    # Preserve original peaks/scores and filter after float32 serialization.
    # Never clamp or fit a post-hoc bias correction.
    return decode_candidates(prediction,shape,dense=True,cell_size=1)
