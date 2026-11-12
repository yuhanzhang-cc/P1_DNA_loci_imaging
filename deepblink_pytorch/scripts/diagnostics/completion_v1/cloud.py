"""Composite sequential queue; immutable children preserve their own protocols."""
from pathlib import Path
import subprocess,sys,time,uuid,os
from diagnostics.handoff_v2.common import read,write,sha


def plan(root):
    root=Path(root)
    return dict(dry_run=True,formal_training_started=False,maximum_new_runs=9,
        stages=['verify_composite_and_children','freeze_real_image_review_and_research_decision','actual_matched_CUDA_doctor',
        'S0/P0_seed42','preapproved_conditional_or_full_registered_S0/P0_seeds43_44','optional_preapproved_Spotiflow_scratch_3seeds',
        'verify_budgets_and_selected_caches','automatic_return_archives','local_finish:analysis_statistics_figures_manuscript'],
        mechanism=root.joinpath('mechanism').as_posix(),scratch=root.joinpath('p1_v4_training').as_posix(),
        historical_results_retrained=False,independent_data_included=False,scientific_success_guaranteed=False)


def verify(root):
    from diagnostics.handoff_v2.common import inside
    root=Path(root).resolve();m=read(root/'completion_bundle_manifest.json')
    if m.get('schema')!='completion_kit_v1' or m.get('test_arrays_present') is not False:raise ValueError('Wrong completion kit')
    for name,digest in m['files'].items():
        if sha(inside(root,name))!=digest:raise ValueError('Composite source differs: '+name)
    # Verify child manifests against their own sealed sources too.
    for child in ('mechanism','p1_v4_training'):
        c=read(root/child/'bundle_manifest.json')
        for name,digest in c['files'].items():
            if sha(inside(root/child,name))!=digest:raise ValueError('Child source differs: '+child+'/'+name)
    from .hardware_policy import profile
    policy=profile(root)
    if policy is not None:
        if sha(root/'configs/hardware_profile.json')!=sha(root/'mechanism/configs/hardware_profile.json'):raise ValueError('Composite and child hardware profiles differ')
    return dict(verified=True,sealed_files=len(m['files']),test_arrays_present=False)


def execute(root,resume=False,launch=None):
    from filelock import FileLock
    root=Path(root).resolve();verify(root)
    # Read the actual frozen decision through the child CLI validation first.
    from diagnostics.handoff_v2.freeze import verify_frozen
    marker,decision=verify_frozen(root/'mechanism/decisions')
    queues=decision.get('additional_baseline_queues',[])
    if queues not in ([],['spotiflow_scratch']):raise ValueError('Unknown additional baseline queue')
    folder=root/'outputs/completion';folder.mkdir(parents=True,exist_ok=True)
    state_path=folder/'status.json';identity=dict(bundle_sha256=sha(root/'completion_bundle_manifest.json'),freeze_sha256=sha(root/'mechanism/decisions/freeze_manifest.json'))
    previous=read(state_path) if state_path.exists() else None
    if previous and (not resume or previous['identity']!=identity):raise ValueError('Existing composite run needs same-identity resume')
    from .hardware_policy import context
    state=dict(identity=identity,status='preflight',completed=[],formal_training_started=False,independent_test_evaluated=False,hardware_context=context(root))
    def update(stage,**kw):state.update(status=stage,**kw);write(state_path,state,replace=True);print(stage,flush=True)
    run=launch or (lambda cmd,cwd:subprocess.run(cmd,cwd=cwd,check=True))
    entry='deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py'
    with FileLock(str(folder/'single_gpu.lock'),timeout=0):
        try:
            import shutil
            free=shutil.disk_usage(root).free
            if free<12*1024**3:raise RuntimeError('At least 12 GiB free disk after environment setup is required; preserve results and expand storage before training')
            update('disk_preflight_passed',free_disk_bytes=free)
            from .hardware_policy import profile
            if profile(root) is not None:
                from .transfer_check import preflight
                update('hardware_transfer_preflight')
                preflight(root,dry_run=False)
            update('mechanism_queue',formal_training_requested=True)
            cmd=[sys.executable,'-u',str(root/'mechanism'/entry),'run','--root',str(root/'mechanism')]
            if resume:cmd.append('--resume')
            run(cmd,root/'mechanism');state['completed'].append('mechanism');state['formal_training_started']=True
            if 'spotiflow_scratch' in queues:
                update('spotiflow_scratch_queue')
                child=root/'p1_v4_training';runner=child/'deepblink_pytorch/scripts/cloud_recipe_train.py'
                run([sys.executable,'-u',str(runner),'--smoke','--device','cuda','--queue','spotiflow_scratch'],child)
                cmd=[sys.executable,'-u',str(runner),'--train','--device','cuda','--queue','spotiflow_scratch']
                if resume:cmd.append('--resume')
                run(cmd,child)
                # Pack in a child process so the original runner's ROOT remains child-bound.
                run([sys.executable,str(root/'deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py'),'pack-scratch','--root',str(child)],root)
                state['completed'].append('spotiflow_scratch')
            else:state['scratch_status']='not_authorized_not_run'
            archives=[]
            for child,pattern in (('mechanism','return_manifest_*.tar.gz'),('p1_v4_training','scratch_return_*.tar.gz')):
                for p in sorted((root/child).glob(pattern)):archives.append(dict(path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
            update('local_publication_pipeline')
            from .catalog import load_portable,live_record
            records=load_portable(root/'baseline_payload')
            for task in read(root/'mechanism/outputs/pipeline/status.json')['completed_tasks']:
                records.append(live_record(task['method'],task['seed'],root/'mechanism/outputs/trials'/task['method']/f"seed_{task['seed']}",'signed'))
            if 'spotiflow_scratch' in queues:
                for seed in (42,43,44):records.append(live_record('Spotiflow_scratch',seed,root/'p1_v4_training/outputs/research_v4_cuda/spotiflow_scratch/Spotiflow_scratch_control'/f'seed_{seed}','spotiflow'))
            # Cache-first analysis makes no extra optimizer updates. Inference and
            # timing may use CUDA. A failure
            # never destroys complete raw results; retry writes a new report revision.
            from .main import finish
            delivery=folder/('delivery_'+uuid.uuid4().hex[:12])
            summary=finish(root,delivery,records=records,device='cuda',benchmark_device='cuda')
            if profile(root) is not None:
                import shutil
                shutil.copytree(root/'outputs/hardware_transfer',delivery/'hardware_transfer',ignore=shutil.ignore_patterns('*.lock'))
                shutil.copyfile(root/'configs/hardware_profile.json',delivery/'hardware_profile.json')
            from .bundle import seal,evaluation_build
            artifact=seal(delivery,name='delivery_manifest.json',schema='completion_delivery_v1',extra=dict(completion_bundle_sha256=identity['bundle_sha256'],contains_formal_training_results=True))
            evalkit=evaluation_build(root,delivery/'analysis',folder/('frozen_evaluation_'+uuid.uuid4().hex[:12]))
            # Exactly one mechanism archive belongs to this finished child state;
            # older manual/resume packages are retained but not mixed into it.
            current=read(root/'mechanism/outputs/pipeline/status.json')['return_archive']
            chosen=[dict(path=Path(current['path']).relative_to(root).as_posix(),sha256=current['sha256'],bytes=current['bytes'])]
            if 'spotiflow_scratch' in queues:
                latest=max((root/'p1_v4_training').glob('scratch_return_*.tar.gz'),key=lambda p:p.stat().st_mtime_ns)
                chosen.append(dict(path=latest.relative_to(root).as_posix(),sha256=sha(latest),bytes=latest.stat().st_size))
            from .bundle import all_results
            combined=all_results(root,chosen,artifact,evalkit,folder/('all_results_'+uuid.uuid4().hex[:12]))
            update('complete_available_scope',archives=chosen,delivery=artifact,frozen_evaluation_kit=evalkit,all_results=combined,summary=summary)
            return state
        except BaseException as error:
            import traceback
            observed=status(root)['tasks']
            started=any(int(t.get('updates',0))>0 or t.get('status')=='complete' for t in observed)
            update('failed_or_interrupted',formal_training_started=started,error=repr(error),traceback=traceback.format_exc());raise


def start(root,resume=False):
    from filelock import FileLock
    root=Path(root).resolve();verify(root)
    from diagnostics.handoff_v2.freeze import verify_frozen
    verify_frozen(root/'mechanism/decisions')
    folder=root/'outputs/completion';folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'launch.lock'),timeout=0):
        record=folder/'process.json'
        if record.exists():
            pid=read(record)['pid'];proc=Path(f'/proc/{pid}/cmdline')
            if proc.exists() and b'completion_v1' in proc.read_bytes():raise ValueError('Composite worker still alive')
        log=folder/('run_'+uuid.uuid4().hex[:12]+'.log')
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('cli.py')),'cloud-run','--root',str(root),'--execute']
        if resume:cmd.append('--resume')
        kw=dict(start_new_session=True) if os.name!='nt' else dict(creationflags=subprocess.CREATE_NO_WINDOW)
        with log.open('xb') as f:p=subprocess.Popen(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,**kw)
        result=dict(pid=p.pid,log=str(log));write(record,result,replace=True);return result


def status(root):
    import statistics
    from diagnostics.handoff_v2.common import table
    root=Path(root);state=root/'outputs/completion/status.json';child=root/'mechanism/outputs/pipeline/status.json';tasks=[]
    decision=root/'mechanism/decisions/research_decision.json';d=read(decision) if decision.exists() else {}
    for method,seeds,base in [('S0',(42,43,44),root/'mechanism/outputs/trials/S0'),('P0',(42,43,44),root/'mechanism/outputs/trials/P0'),
        ('Spotiflow_scratch',(42,43,44),root/'p1_v4_training/outputs/research_v4_cuda/spotiflow_scratch/Spotiflow_scratch_control')]:
        for seed in seeds:
            folder=base/f'seed_{seed}';p=folder/'status.json';t=read(p) if p.exists() else dict(status='not_started',epoch=0,updates=0)
            progress=folder/'progress.json'
            if progress.exists():t.update({k:v for k,v in read(progress).items() if k in ('epoch','updates')})
            callback=None;history=folder/'training_metrics.csv'
            if history.exists():
                rows=table(history);seconds=[float(r['seconds']) for r in rows[-10:] if r.get('seconds')]
                if seconds:callback=statistics.median(seconds)
                if rows:t['updates']=int(rows[-1].get('optimizer_updates',rows[-1].get('updates',t.get('updates',0))))
            t['updates']=int(t.get('updates',t.get('optimizer_updates',0)))
            tasks.append(dict(method=method,seed=seed,**t,median_recent_callback_seconds=callback))
    full=d.get('completion_strategy')=='full_registered_six';scratch='spotiflow_scratch' in d.get('additional_baseline_queues',[])
    remaining_min=remaining_max=0.
    for t in tasks:
        if t['status']=='complete':continue
        if t['method']=='Spotiflow_scratch' and not scratch:continue
        rate=t['median_recent_callback_seconds'] or (3.4 if t['method']=='Spotiflow_scratch' else 2.6)
        seconds=max(0,100-int(t.get('epoch',0)))*rate
        remaining_max+=seconds
        if t['seed']==42 or full or t['method']=='Spotiflow_scratch':remaining_min+=seconds
    completion=read(state) if state.exists() else {'status':'not_started'}
    completion['formal_training_started']=any(int(t.get('updates',0))>0 or t.get('status')=='complete' for t in tasks)
    from .hardware_policy import context
    return dict(completion=completion,hardware_context=context(root),mechanism=read(child) if child.exists() else {'status':'not_started'},tasks=tasks,
        conditional_remaining_epoch_seconds=[remaining_min,remaining_max] if d else None,
        eta_warning='Callback-only extrapolation; excludes setup, probes, selection, postprocessing, I/O and failures. Unstarted tasks use historical proxies. Not a wall-clock promise.')
