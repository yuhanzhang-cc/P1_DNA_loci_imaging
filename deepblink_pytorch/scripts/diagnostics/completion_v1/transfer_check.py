"""Read-only CUDA transfer preflight, enforced before any formal update."""
from pathlib import Path
import uuid,os
from diagnostics.handoff_v2.common import read,write,table,sha
from .hardware_policy import profile,lock_backend


def compare_exports(expected,actual,sample_ids,tolerance=.005):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    expected,actual=Path(expected),Path(actual)
    a=read(expected/'metrics.json');b=read(actual/'metrics.json')
    counts=all(int(a[k])==int(b[k]) for k in ('tp','fp','fn'))
    def points(folder):
        result={s:[] for s in sample_ids}
        for r in table(folder/'prediction_points.csv'):
            if r['sample_id'] not in result:raise ValueError('Unexpected reference image ID')
            result[r['sample_id']].append([float(r['y']),float(r['x']),float(r['p'])])
        return {s:np.asarray(p,float).reshape(-1,3) for s,p in result.items()}
    aa,bb=points(expected),points(actual);same=True;max_distance=max_score_delta=0.
    for sid in sample_ids:
        x,y=aa[sid],bb[sid]
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite parity points')
        if len(x)!=len(y):same=False;continue
        if not len(x):continue
        distances=cdist(x[:,:2],y[:,:2]);left,right=linear_sum_assignment(distances)
        max_distance=max(max_distance,float(distances[left,right].max()))
        max_score_delta=max(max_score_delta,float(np.abs(x[left,2]-y[right,2]).max()))
    return dict(passed=counts and same and max_distance<=tolerance,thresholded_counts_exact=counts,
        per_image_prediction_counts_exact=same,max_coordinate_difference_px=max_distance,
        max_absolute_score_difference=max_score_delta,coordinate_tolerance_px=tolerance,
        score_difference_policy='record only; not a claim of training equivalence',optimizer_updates=0)


def preflight(root,dry_run=True):
    root=Path(root).resolve();spec=profile(root)
    if spec is None:return dict(required=False,reason='Original 5090 track retains its original strict doctor',training_started=False)
    if dry_run:return dict(dry_run=True,profile_id=spec['profile_id'],training_started=False,optimizer_updates=0,
        checks=['CUDA profile and three historical initializer hashes','fixed-checkpoint full validation count/coordinate parity','same-hardware resume identity'],
        independent_test_images_read=0,inference_does_not_establish_training_equivalence=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from filelock import FileLock
    from diagnostics.handoff_v2.doctor import diagnose
    from diagnostics.mechanism_v1.runner import corpus
    from research_v4.evaluation import export_predictions
    from .catalog import load_portable
    from .inference import predict_record
    work=root/'outputs/hardware_transfer';work.mkdir(parents=True,exist_ok=True)
    with FileLock(str(work/'preflight.lock'),timeout=0):
        attempt=work/('attempt_'+uuid.uuid4().hex[:12]);attempt.mkdir()
        doctor=diagnose(attempt/'doctor',device='cuda',context_root=root/'mechanism')
        if not doctor['passed']:raise RuntimeError('vGPU hardware or historical initialization check failed; preserve doctor.json, no training allowed')
        identity=dict(profile_sha256=sha(root/'configs/hardware_profile.json'),bundle_sha256=sha(root/'completion_bundle_manifest.json'),hardware=doctor['hardware_identity'])
        lock_backend(work/'backend_lock.json',identity)
        complete=work/'complete.json'
        if complete.exists():
            old=read(complete)
            if old['identity']!=identity or not old['passed']:raise ValueError('Stale transfer preflight')
            for name,digest in old['evidence_hashes'].items():
                if sha(work/name)!=digest:raise ValueError('Transfer evidence changed')
            return dict(passed=True,reused_verified_parity=True,identity=identity,optimizer_updates=0)
        data=corpus(root,root/'mechanism/assets').splits['valid'];ids=[str(s) for s in data['sample_id']]
        checks=[];records=load_portable(root/'baseline_payload')
        expected={(m,s) for m in ('R0','H0','G0','Spotiflow_F') for s in (42,43,44)}|{('LoG',None),('DoG',None)}
        if len(records)!=14 or {(r['method'],r['seed']) for r in records}!=expected:raise ValueError('Incomplete frozen reference transfer panel')
        for record in records:
            # Force inference from each locked weight, never from a scored cache.
            candidates=predict_record(record,data,'cuda')
            folder=attempt/'references'/record['method']/(f"seed_{record['seed']}" if record['seed'] is not None else 'deterministic')
            policy=record.get('policy') or dict(kind='all' if record['threshold']==0 else 'none' if record['threshold']==1 else 'finite',value=record['threshold'])
            export_predictions(folder,candidates,data,policy)
            source=Path(record['folder'])/('selected' if record['kind']=='classical' else 'primary')
            result=compare_exports(source,folder,ids)
            checks.append(dict(method=record['method'],seed=record['seed'],**result))
            write(folder/'parity.json',result)
            print(str(record['method'])+' '+str(record['seed'])+' parity='+str(result['passed']),flush=True)
        if not all(r['passed'] for r in checks):
            write(attempt/'failed.json',dict(passed=False,checks=checks,optimizer_updates=0))
            raise RuntimeError('Fixed-checkpoint validation parity failed; no formal training, do not loosen tolerance after observing this failure')
        hashes={p.relative_to(work).as_posix():sha(p) for p in attempt.rglob('*') if p.is_file()}
        result=dict(passed=True,identity=identity,checks=checks,evidence_hashes=hashes,optimizer_updates=0,
            inference_only=True,independent_test_evaluated=False,training_equivalence_proven=False)
        write(complete,result);return result
