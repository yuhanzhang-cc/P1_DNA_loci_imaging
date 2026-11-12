"""Lean return for the existing Spotiflow scratch queue, retaining selected scores."""
from pathlib import Path
import tarfile,uuid
from diagnostics.handoff_v2.common import read,write,sha


def pack(root):
    root=Path(root).resolve()
    import importlib.util
    spec=importlib.util.spec_from_file_location('completion_registered_cloud',root/'deepblink_pytorch/scripts/cloud_recipe_train.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.check_complete(root/'outputs/research_v4_cuda','spotiflow_scratch')
    queue=root/'outputs/research_v4_cuda/spotiflow_scratch';selected=set()
    for p in queue.rglob('*'):
        if p.is_file() and not any(s.startswith('epoch_') or s in ('recovery','orphaned_snapshots') for s in p.relative_to(queue).parts) and p.suffix in ('.json','.csv','.yaml','.yml','.pt'):selected.add(p)
    for seed in (42,43,44):
        folder=queue/'Spotiflow_scratch_control'/f'seed_{seed}';selection=read(folder/'selection.json')
        for role in ('primary','native_secondary'):selected.add(folder/f"epoch_{selection[role]['epoch']:03d}/candidates.npz")
    selected.add(root/'bundle_manifest.json')
    selected|={p for p in (root/'outputs/research_v4_cuda/engineering_smoke').rglob('*.json') if p.is_file()}
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(selected)}
    marker=root/('scratch_return_'+uuid.uuid4().hex[:12]+'.json')
    write(marker,dict(schema='completion_scratch_v1',files=files,source_bundle_sha256=sha(root/'bundle_manifest.json'),test_evaluated=False,engineering_only=False))
    archive=marker.with_suffix('.tar.gz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for p in sorted(selected):tar.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
            tar.add(marker,arcname='scratch_return_manifest.json',recursive=False)
    digest=sha(archive);archive.with_name(archive.name+'.sha256').write_bytes((digest+'  '+archive.name+'\n').encode())
    return dict(path=str(archive),sha256=digest,bytes=archive.stat().st_size)


def check_return(root):
    from diagnostics.handoff_v2.common import inside
    root=Path(root);m=read(root/'scratch_return_manifest.json')
    if m.get('schema')!='completion_scratch_v1' or m.get('engineering_only') is not False or m.get('test_evaluated') is not False:raise ValueError('Wrong scratch result type')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(m['files'])|{'scratch_return_manifest.json'}:raise ValueError('Undeclared/missing scratch result files')
    for name,digest in m['files'].items():
        if sha(inside(root,name))!=digest:raise ValueError('Scratch return hash differs')
    for seed in (42,43,44):
        folder=root/'outputs/research_v4_cuda/spotiflow_scratch/Spotiflow_scratch_control'/f'seed_{seed}'
        s=read(folder/'selection.json');budget=read(folder/'optimizer_summary.json');state=read(folder/'status.json')
        if state.get('status')!='complete' or state.get('test_evaluated') is not False or (budget['completed_epochs'],budget['optimizer_updates'],s['checkpoint_count'],s['epoch_threshold_pairs'])!=(100,4400,20,620):raise ValueError('Incomplete scratch selection/budget')
        if sha(folder/'primary/best.pt')!=s['primary_weight_sha256']:raise ValueError('Scratch selected weight differs')
        snapshots=read(folder/'snapshots.json')
        if [r['epoch'] for r in snapshots]!=list(range(5,101,5)):raise ValueError('Scratch snapshot schedule differs')
        for role in ('primary','native_secondary'):
            snap=next(r for r in snapshots if r['epoch']==s[role]['epoch'])
            if sha(folder/f"epoch_{snap['epoch']:03d}/candidates.npz")!=snap['candidate_sha256']:raise ValueError('Scratch selected candidate hash differs')
    return dict(verified=True,tasks=3,test_evaluated=False)
