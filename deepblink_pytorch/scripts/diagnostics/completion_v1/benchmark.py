"""Same-device frozen-model inference timing; explicitly no optimization."""
from pathlib import Path
import time,statistics,platform
from diagnostics.handoff_v2.common import write,csv_write,fresh


def measure(records,data,output,device='cuda',dry_run=True,predict=None,synchronize=None,repeats=3):
    sizes=sorted({tuple(s) for s in data['original_shape']})
    if dry_run:return dict(dry_run=True,models=len(records),native_shapes=sizes,warmups=1,repeats=repeats,optimizer_updates=0)
    if repeats<3:raise ValueError('At least three timing repeats required')
    from .inference import predict_record
    if predict is None:
        import torch
        if device!='cuda' or not torch.cuda.is_available():raise ValueError('Formal efficiency comparison requires actual shared CUDA')
        synchronize=torch.cuda.synchronize;gpu=torch.cuda.get_device_name(0);torch_version=str(torch.__version__)
    else:gpu='SYNTHETIC_ENGINEERING';torch_version='synthetic'
    predict=predict or predict_record;synchronize=synchronize or (lambda:None)
    output=fresh(output);rows=[]
    for r in records:
        for shape in sizes:
            i=next(i for i,s in enumerate(data['original_shape']) if tuple(s)==shape)
            sample={k:[v[i]] for k,v in data.items() if k in ('x','y','sample_id','original_shape')}
            # Adapter reload is intentionally included: reproducible end-to-end
            # single-image cold-load workflow, not a network throughput claim.
            predict(r,sample,device);synchronize();elapsed=[]
            for _ in range(repeats):
                synchronize();start=time.perf_counter();result=predict(r,sample,device);synchronize()
                elapsed.append(time.perf_counter()-start)
            rows.append(dict(method=r['method'],seed=r['seed'],height=shape[0],width=shape[1],median_seconds=statistics.median(elapsed),
                min_seconds=min(elapsed),max_seconds=max(elapsed),repeats=repeats,n_candidates=len(result[0][0]),device=device,gpu=gpu,
                scope='single_image_adapter_load_plus_preprocessing_plus_inference_plus_decode',engineering_only=gpu=='SYNTHETIC_ENGINEERING'))
    csv_write(output/'latency.csv',rows,list(rows[0]));write(output/'provenance.json',dict(gpu=gpu,torch=torch_version,platform=platform.platform(),
        optimizer_updates=0,includes_model_load=True,batch_size=1,cached_inference_used=False,speed_superiority_claim=False))
    return dict(output=str(output),measurements=len(rows),optimizer_updates=0)
