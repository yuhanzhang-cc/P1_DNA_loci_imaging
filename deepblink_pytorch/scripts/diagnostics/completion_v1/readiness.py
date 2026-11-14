"""Evidence inventory and measured timing context; no claims of CUDA readiness."""
from pathlib import Path
import statistics
from diagnostics.handoff_v2.common import ROOT,read,table,write,fresh,sha,csv_write
from .catalog import inventory,BASE


def audit(root,output):
    root=Path(root);output=fresh(output);records=inventory(root)
    # Historical immutable manifest uses absolute paths to the shared source tree.
    protected=read(root/'docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/evidence/immutable_before.json')
    changed=[name for name,digest in protected.items() if sha(name)!=digest]
    if changed:raise ValueError('Protected historical artifacts changed: '+str(changed[:3]))
    timing=[]
    for r in records:
        p=Path(r['folder'])/'training_metrics.csv'
        if not p.exists():continue
        rr=table(p);keys=[k for k in ('seconds','epoch_seconds') if k in rr[0]]
        if not keys:continue
        v=[float(x[keys[0]]) for x in rr]
        timing.append(dict(method=r['method'],seed=r['seed'],epochs=len(v),sum_recorded_epoch_seconds=sum(v),median_epoch_seconds=statistics.median(v),
            interpretation='historical epoch callbacks; excludes setup/download and may exclude some finalization; not a new runtime guarantee'))
    csv_write(output/'historical_timing.csv',timing,['method','seed','epochs','sum_recorded_epoch_seconds','median_epoch_seconds','interpretation'])
    checks=dict(protected_files=len(protected),protected_unchanged=True,available_models=len(records),available_methods=len({r['method'] for r in records}),
        formal_training_started=False,gpu_used=False,test_arrays_read=False,human_review_checked_by_this_command=False,source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        unresolved_scientific_inputs=['real 60-case review','explicit S0/P0 radius and continuation approval','optional scratch authorization',
        'actual CUDA device/dependencies/initializer gate','prospective outcomes','independent dataset/provenance','acquisition and author metadata','target journal and release permissions'])
    write(output/'readiness.json',checks);return checks
