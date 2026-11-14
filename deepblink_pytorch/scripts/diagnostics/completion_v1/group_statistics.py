"""Fixed-model biological-group bootstrap; no seed pseudoreplication."""
import numpy as np
from diagnostics.handoff_v2.common import table,write,csv_write,fresh,read,sha
from .analysis import CONTRASTS


def micro(counts):
    counts=np.asarray(counts,float);den=2*counts[...,0]+counts[...,1]+counts[...,2]
    return np.divide(2*counts[...,0],den,out=np.ones_like(den),where=den!=0)


def cluster_interval(a,b,draws=10000,seed=20261111):
    """Arrays: training seed x independent biological group x (tp,fp,fn)."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[-1]!=3 or a.shape[1]<2 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<0) or np.any(b<0):raise ValueError('Invalid paired biological-group counts')
    if draws<100:raise ValueError('Too few bootstrap draws')
    rng=np.random.default_rng(seed);indices=rng.integers(0,a.shape[1],(draws,a.shape[1]))
    d=(micro(a[:,indices].sum(2))-micro(b[:,indices].sum(2))).mean(0)
    return dict(mean_f1_delta=float((micro(a.sum(1))-micro(b.sum(1))).mean()),ci_low=float(np.quantile(d,.025)),ci_high=float(np.quantile(d,.975)),
        independent_groups=a.shape[1],fixed_model_seeds=a.shape[0],resampling_unit='biological_group_id',bootstrap_draws=draws,bootstrap_seed=seed,
        uncertainty='paired fixed-model biological-group percentile bootstrap, conditional on selected models',p_value=None,stars='')


def independent_summary(evaluation,output,allow_engineering=False):
    from pathlib import Path
    evaluation=Path(evaluation)
    completion=read(evaluation/'complete.json')
    if completion.get('test_evaluated') is not True:raise ValueError('Independent evaluation not complete')
    if completion.get('engineering_only') is not False and not allow_engineering:raise ValueError('Engineering readout is not independent scientific evidence')
    rows=table(evaluation/'per_image_metrics.csv');out=[]
    for left,right in CONTRASTS:
        a=[r for r in rows if r['method']==left];b=[r for r in rows if r['method']==right]
        if not a or not b:out.append(dict(comparison=left+' - '+right,status='pending_models'));continue
        def indexed(rr):
            d={}
            for r in rr:
                key=(r['seed'],r['sample_id'])
                if key in d:raise ValueError('Duplicate image/model row')
                d[key]=r
            return d
        aa,bb=indexed(a),indexed(b)
        if set(aa)!=set(bb) or any(aa[k]['biological_group_id']!=bb[k]['biological_group_id'] for k in aa):raise ValueError('Incomplete paired image/group identities')
        groups=sorted({r['biological_group_id'] for r in a});seeds=sorted({r['seed'] for r in a})
        if seeds!=['42','43','44'] or len(groups)<2:out.append(dict(comparison=left+' - '+right,status='insufficient_complete_seeds_or_biological_groups'));continue
        def array(rr):return [[[sum(float(r[k]) for r in rr if r['seed']==s and r['biological_group_id']==g) for k in ('tp','fp','fn')] for g in groups] for s in seeds]
        out.append(dict(comparison=left+' - '+right,status='complete_conditional_estimation',**cluster_interval(array(a),array(b))))
    output=fresh(output);fields=list(dict.fromkeys(k for r in out for k in r))
    csv_write(output/'independent_group_intervals.csv',[{k:r.get(k) for k in fields} for r in out],fields)
    write(output/'provenance.json',dict(source_sha256=sha(evaluation/'complete.json'),inference='estimation_only; no significance stars from bootstrap CI',
        experimental_unit='reviewed independent biological group',selection_adjusted=False))
    return dict(output=str(output),comparisons=len(out))
