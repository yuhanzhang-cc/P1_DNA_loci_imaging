"""Common fixed-threshold metrics and full ranked AP for all completed methods."""
from pathlib import Path
import statistics,itertools,math
import numpy as np
from diagnostics.handoff_v2.common import read,write,fresh,csv_write,sha,table
from diagnostics.scaffolds.ranked_ap import ranked_point_ap
from research_v4.evaluation import load_candidates,export_predictions
from .catalog import inventory,verify_record

METRICS=("precision","recall","f1","ranked_ap","weak_recall","mean_error_px","count_mae")
CONTRASTS=(("S0","H0"),("P0","S0"),("P0","G0"),("P0","R0"),("P0","Spotiflow_F"))


def sign_flip(differences):
    d=np.asarray(differences,float)
    if d.ndim!=1 or not np.isfinite(d).all() or len(d)>16:raise ValueError("Invalid exact paired-seed differences")
    if not len(d):return None
    observed=abs(float(d.mean()));permuted=[abs(float(np.mean(d*np.asarray(s)))) for s in itertools.product((-1,1),repeat=len(d))]
    return sum(v>=observed-1e-14 for v in permuted)/len(permuted)


def holm(values,family_size=None):
    """Missing planned tests retain their place in the multiplicity family."""
    size=len(values) if family_size is None else family_size
    if size<len(values):raise ValueError("Family cannot shrink")
    present=sorted((float(p),i) for i,p in enumerate(values) if p is not None)
    if any(not 0<=p<=1 or not math.isfinite(p) for p,_ in present):raise ValueError("Invalid p")
    out=[None]*len(values);previous=0.
    for rank,(p,i) in enumerate(present):previous=max(previous,min(1.,p*(size-rank)));out[i]=previous
    return out


def classify_outcome(rows):
    groups={m:{int(r['seed']):r for r in rows if r['method']==m and r.get('seed') is not None} for m in ('G0','S0','P0','Spotiflow_F')}
    if set(groups['P0'])!={42,43,44} or set(groups['S0'])!={42,43,44}:return "partial_mechanism_results_keep_G0"
    delta=statistics.mean(groups['P0'][s]['f1']-groups['G0'][s]['f1'] for s in (42,43,44))
    return "complete_development_candidate_requires_review" if delta>0 else "complete_negative_development_keep_G0"


def candidates(record,data,device="cpu"):
    verify_record(record)
    if record.get("cache"):
        return load_candidates(record["cache"],data)
    # Legacy lean baseline returns omit caches. Reconstruct once from locked weights.
    from .inference import predict_record
    return predict_record(record,data,device)


def analyze(root,output,returned=None,scratch=None,device="cpu",records=None):
    from diagnostics.mechanism_v1.runner import corpus
    from research_v4.engine import configure_device
    root=Path(root);assets=root/'mechanism/assets'
    configure_device(device);output=fresh(output);data=corpus(root,assets if assets.exists() else None).splits['valid']
    records=inventory(root,returned,scratch) if records is None else records
    from .hardware_policy import context
    hardware=context(root)
    if hardware is None and returned:hardware=context(returned)
    truth={str(s):t for s,t in zip(data['sample_id'],data['y'])};rows=[];all_images=[];source={};curves=[]
    for r in records:
        print(f"Evaluating frozen {r['method']} seed {r['seed']}",flush=True)
        name=f"seed_{r['seed']}" if r['seed'] is not None else "deterministic"
        folder=output/'evaluated'/r['method']/name
        c=candidates(r,data,device)
        policy=r.get('policy') or dict(kind='all' if r['threshold']==0 else 'none' if r['threshold']==1 else 'finite',value=r['threshold'])
        metrics=export_predictions(folder,c,data,policy)
        # Require exact count agreement with the actual frozen CUDA/export source.
        original=read(Path(r['folder'])/('selected' if r['kind']=='classical' else 'primary')/'metrics.json')
        if any(metrics[k]!=original[k] for k in ('tp','fp','fn')):raise ValueError(f"Frozen count replay differs for {r['method']}/{r['seed']}; investigate, do not recalibrate")
        ap=ranked_point_ap(truth,{str(s):v for s,v in zip(data['sample_id'],c)},radius=3.)
        csv_write(folder/'ranked_pr_curve.csv',ap['curve'],['score','tp','fp','recall','precision'])
        csv_write(folder/'ranked_predictions.csv',ap['ranked_records'],['sample_id','pred_index','score','is_tp','true_index'])
        write(folder/'ranked_ap.json',{k:v for k,v in ap.items() if k not in ('curve','ranked_records')})
        row=dict(method=r['method'],seed=r['seed'],track=r['track'],**metrics,ranked_ap=ap['ap'])
        rows.append(row);write(folder/'metrics_with_ranked_ap.json',row)
        for item in table(folder/'per_image_metrics.csv'):all_images.append(dict(method=r['method'],seed=r['seed'],**item))
        curves.extend(dict(method=r['method'],seed=r['seed'],**v) for v in ap['curve'])
        source[r['selection']]=r['selection_sha256']
    fields=['method','seed','track',*METRICS,'tp','fp','fn','weak_tp','weak_n','test_evaluated']
    csv_write(output/'per_seed_metrics.csv',[{k:r.get(k) for k in fields} for r in rows],fields)
    csv_write(output/'per_image_metrics.csv',all_images,list(all_images[0]))
    csv_write(output/'ranked_pr_curves.csv',curves,['method','seed','score','tp','fp','recall','precision'])
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        for metric in METRICS:
            v=[r[metric] for r in group if r.get(metric) is not None]
            summary.append(dict(method=method,metric=metric,n=len(v),mean=statistics.mean(v) if v else None,sd=statistics.stdev(v) if len(v)>1 else None,
                replicate_unit='deterministic_configuration' if group[0]['seed'] is None else 'training_seed',dataset_role='development_validation'))
    csv_write(output/'summary.csv',summary,list(summary[0]))
    comparisons=[]
    for left,right in CONTRASTS:
        a={r['seed']:r for r in rows if r['method']==left};b={r['seed']:r for r in rows if r['method']==right}
        paired=sorted(s for s in set(a)&set(b) if s is not None)
        d=[a[s]['f1']-b[s]['f1'] for s in paired]
        comparisons.append(dict(comparison=left+' - '+right,n_paired_seeds=len(d),mean_f1_delta=statistics.mean(d) if d else None,
            p_exact_seed_signflip=sign_flip(d) if paired==[42,43,44] else None,p_holm=None,stars='',
            scope=('same_new_hardware_track; ' if hardware and (left,right)==('P0','S0') else 'cross-hardware_development_reference; ' if hardware else '')+'conditional_training_variability; not biological inference',status='complete' if paired==[42,43,44] else 'pending'))
    for r,p in zip(comparisons,holm([r['p_exact_seed_signflip'] for r in comparisons])):r['p_holm']=p;r['stars']='***' if p is not None and p<.001 else '**' if p is not None and p<.01 else '*' if p is not None and p<.05 else ''
    csv_write(output/'planned_f1_comparisons.csv',comparisons,list(comparisons[0]))
    # Common-truth pairs separate localization from differing TP composition.
    localization=[]
    for left,right in CONTRASTS:
        for seed in (42,43,44):
            lf=output/'evaluated'/left/f'seed_{seed}'/'matches.csv';rf=output/'evaluated'/right/f'seed_{seed}'/'matches.csv'
            if not lf.exists() or not rf.exists():continue
            a={(r['sample_id'],int(r['true_index'])):float(r['distance_px']) for r in table(lf)}
            b={(r['sample_id'],int(r['true_index'])):float(r['distance_px']) for r in table(rf)}
            for sid,ti in sorted(set(a)&set(b)):localization.append(dict(candidate=left,reference=right,seed=seed,sample_id=sid,true_index=ti,
                candidate_error_px=a[(sid,ti)],reference_error_px=b[(sid,ti)],error_delta_px=a[(sid,ti)]-b[(sid,ti)]))
    csv_write(output/'common_tp_errors.csv',localization,['candidate','reference','seed','sample_id','true_index','candidate_error_px','reference_error_px','error_delta_px'])
    status=classify_outcome(rows)
    decision_root=root/'mechanism/decisions' if (root/'mechanism/decisions').exists() else Path(returned)/'decisions' if returned else None
    annotations={}
    if decision_root and (decision_root/'freeze_manifest.json').exists():
        from diagnostics.handoff_v2.freeze import verify_frozen
        _,decision=verify_frozen(decision_root)
        annotations=dict(approved_supervision=decision['p0_supervision'],completion_strategy=decision.get('completion_strategy','gate_conditioned'),review_summary=read(decision_root/'review_summary.json'),decision_sha256=sha(decision_root/'research_decision.json'))
    screen=None
    state=root/'mechanism/outputs/pipeline/status.json' if (root/'mechanism/outputs/pipeline/status.json').exists() else Path(returned)/'outputs/pipeline/status.json' if returned else None
    if state and state.exists():screen=read(state).get('gate')
    write(output/'analysis_manifest.json',dict(schema='completion_analysis_v1',inputs=source,model_catalog=records,analysis_source_sha256=sha(__file__),
        status=status,dataset_role='development_validation',test_evaluated=False,threshold_refitted=False,all_frozen_counts_exact=True,
        biological_replicates_known=False,known_train_valid_source_overlap=2,formal_training_performed=False,mechanism_context=annotations,screening_gate=screen,hardware_context=hardware))
    return dict(output=str(output),models=len(rows),methods=len({r['method'] for r in rows}),status=status,test_evaluated=False)
