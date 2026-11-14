"""Fixed-adapter prediction and once-only independent evaluation suite."""
from pathlib import Path
import numpy as np
from diagnostics.handoff_v2.common import read,write,fresh,sha,csv_write
from .catalog import verify_record


def predict_record(record,data,device='cpu'):
    verify_record(record)
    if record['kind']=='classical':
        from research_v4.classical import image_candidates
        return [image_candidates(x,[record['configuration']],record['method'])[0] for x in data['x']]
    if record['kind']=='signed':
        from diagnostics.mechanism_v1.runner import load_model,infer
        model,_=load_model(record['weight'],device);return infer(model,data,device)
    if record['kind']=='own':
        import torch
        from deepblink_torch.cli import predict
        from research_v4.engine import infer_candidates
        model,_=predict._load_model(record['weight'],torch.device(device));return infer_candidates(model,data,device)
    if record['kind']=='spotiflow':
        from spotiflow.model import Spotiflow
        from deepblink_torch.spotiflow_bridge import spotiflow_candidates
        model=Spotiflow.from_folder(Path(record['weight']).parent,map_location=device,inference_mode=True)
        return [spotiflow_candidates(model,x,device) for x in data['x']]
    raise ValueError('Unknown frozen adapter')


def review_manifest(manifest,root,read_pixels=False):
    """JSON provenance + existing strict CSV loader; dry-run reads no test pixels."""
    from research_v4.external import load_manifest,exclusion_manifest
    from research_v4.common import rows
    from diagnostics.mechanism_v1.runner import corpus
    p=Path(manifest).resolve();s=read(p)
    if s.get('dataset_role')!='independent_test' or s.get('independence_reviewed') is not True or s.get('synthetic_fixture') is not False:raise ValueError('Reviewed real independent provenance required')
    if any(not isinstance(s.get(k),str) or not s[k].strip() or s[k].startswith('TODO') for k in ('reviewer','reviewed_at','independence_notes')):raise ValueError('Missing provenance decision')
    def local(name):
        q=(p.parent/name).resolve()
        if not q.is_relative_to(p.parent):raise ValueError('Manifest path escapes dataset')
        return q
    csv_path=local(s['images_csv']);evidence=local(s['independence_evidence_file'])
    if sha(csv_path)!=s['images_csv_sha256'] or sha(evidence)!=s['independence_evidence_sha256']:raise ValueError('Independent manifest binding differs')
    rows_csv=rows(csv_path)
    if not rows_csv:raise ValueError('Empty independent dataset')
    if len({r['sample_id'] for r in rows_csv})!=len(rows_csv):raise ValueError('Duplicate image ID')
    # Provenance grouping is required; metadata does not establish independence by itself.
    for r in rows_csv:
        if any(not r.get(k,'').strip() or r[k].startswith('TODO') for k in ('biological_group_id','acquisition_id','source_container')):raise ValueError('Missing biological/acquisition identity')
    if not read_pixels:return dict(verified_metadata_only=True,images=len(rows_csv),images_read=0,biological_groups=len({r['biological_group_id'] for r in rows_csv}),data=None,metadata=rows_csv)
    portable=Path(root)/'assets/reference_exclusions.json'
    if portable.exists():exclusions=read(portable)
    else:
        c=corpus(root);split=rows(Path(root)/'outputs/261024_clean220_random/frozen_split.csv')
        exclusions=exclusion_manifest(c,split)
        exclusions['sample_ids']+= [r['sample_id'] for r in split]
    data,metadata=load_manifest(csv_path,exclusions)
    from diagnostics.scaffolds.weak_policy import local_zscore,classify
    from diagnostics.handoff_v2.common import ROOT
    policy=read(ROOT/'docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/configs/weak_policy.json')
    labels=[classify([local_zscore(x,y,z,policy) for y,z in points],policy) for x,points in zip(data['x'],data['y'])]
    data['weak']=[l=='weak' for l in labels]
    return dict(data=data,metadata=metadata,weak_unclassifiable_n=sum(int((l=='unclassifiable').sum()) for l in labels))


def locked_catalog(analysis):
    spec=read(Path(analysis)/'analysis_manifest.json');records=spec['model_catalog']
    if spec.get('portable_paths'):
        from diagnostics.handoff_v2.common import inside
        for r in records:
            for key in ('folder','selection','weight'):
                if key in r:r[key]=str(inside(analysis,r[key]))
    for r in records:verify_record(r)
    return records


def evaluate_suite(root,analysis,manifest,output,device='cpu',dry_run=True):
    records=locked_catalog(analysis)
    review=review_manifest(manifest,root,read_pixels=not dry_run)
    if dry_run:return dict(dry_run=True,images_read=0,locked_models=len(records),metadata_verified=True,training_started=False)
    from research_v4.engine import configure_device
    from research_v4.evaluation import export_predictions
    from diagnostics.scaffolds.ranked_ap import ranked_point_ap
    configure_device(device);output=fresh(output);data=review['data'];summary=[];images=[]
    write(output/'evaluation_lock.json',dict(manifest_sha256=sha(manifest),analysis_manifest_sha256=sha(Path(analysis)/'analysis_manifest.json'),
        model_catalog=records,threshold_selection_on_test=False,declared_before_readout=True,device=device))
    for r in records:
        folder=output/r['method']/(f"seed_{r['seed']}" if r['seed'] is not None else 'deterministic')
        c=predict_record(r,data,device)
        policy=r.get('policy') or dict(kind='all' if r['threshold']==0 else 'none' if r['threshold']==1 else 'finite',value=r['threshold'])
        metrics=export_predictions(folder,c,data,policy)
        ap=ranked_point_ap({str(s):t for s,t in zip(data['sample_id'],data['y'])},{str(s):v for s,v in zip(data['sample_id'],c)})
        csv_write(folder/'ranked_pr_curve.csv',ap.pop('curve'),['score','tp','fp','recall','precision'])
        csv_write(folder/'ranked_predictions.csv',ap.pop('ranked_records'),['sample_id','pred_index','score','is_tp','true_index'])
        write(folder/'ranked_ap.json',ap)
        metrics.update(ranked_ap=ap['ap'],development_validation_only=False,test_evaluated=True,threshold_selection_on_test=False,
            weak_unclassifiable_n=review['weak_unclassifiable_n'])
        write(folder/'metrics.json',metrics,replace=True)
        summary.append(dict(method=r['method'],seed=r['seed'],**metrics))
        byid={x['sample_id']:x for x in review['metadata']}
        from diagnostics.handoff_v2.common import table
        for item in table(folder/'per_image_metrics.csv'):images.append(dict(method=r['method'],seed=r['seed'],biological_group_id=byid[item['sample_id']]['biological_group_id'],**item))
    csv_write(output/'per_seed_metrics.csv',summary,list(summary[0]));csv_write(output/'per_image_metrics.csv',images,list(images[0]))
    write(output/'complete.json',dict(test_evaluated=True,engineering_only=False,models=len(records),manifest_sha256=sha(manifest),threshold_selection_on_test=False))
    return dict(output=str(output),models=len(records),test_evaluated=True)


def predict_image(analysis,image_path,method,seed,output,device='cpu'):
    """Label-free single-image deployment using the validation-frozen threshold."""
    import tifffile
    from research_v4.evaluation import keep_scores
    from research_v4.engine import configure_device
    records=[r for r in locked_catalog(analysis) if r['method']==method and r['seed'] in (None,seed)]
    if len(records)!=1:raise ValueError('Select one available frozen method/seed; no automatic best-seed selection')
    r=records[0];image_path=Path(image_path);image=np.asarray(tifffile.imread(image_path),np.float32)
    if image.ndim!=2 or image.shape not in ((256,256),(512,512)) or not np.isfinite(image).all():raise ValueError('Use finite native 2D 256/512 TIFF; no resize or unapproved tiling')
    configure_device(device);data=dict(x=[image],y=[np.empty((0,2),np.float32)],sample_id=[image_path.stem],original_shape=[image.shape])
    points,scores=predict_record(r,data,device)[0]
    if not np.isfinite(points).all() or not np.isfinite(scores).all() or points.shape!=(len(scores),2) or np.any(points<0) or np.any(points>=image.shape):raise ValueError('Invalid native-field prediction')
    policy=r.get('policy') or dict(kind='all' if r['threshold']==0 else 'none' if r['threshold']==1 else 'finite',value=r['threshold'])
    mask=keep_scores(scores,policy);points,scores=points[mask],scores[mask];output=fresh(output)
    csv_write(output/'prediction_points.csv',[dict(y=float(y),x=float(x),score=float(s)) for (y,x),s in zip(points,scores)],['y','x','score'])
    write(output/'prediction_manifest.json',dict(image_sha256=sha(image_path),shape=list(image.shape),method=method,seed=r['seed'],
        selected_weight_sha256=r.get('weight_sha256'),selection_sha256=r['selection_sha256'],threshold_policy=policy,labels_used=False,
        coordinate_units='native pixel-center indices, y then x',device=device,model_recommended_by_this_command=False,training_performed=False))
    return dict(output=str(output),predictions=len(points),labels_used=False,training_performed=False)
