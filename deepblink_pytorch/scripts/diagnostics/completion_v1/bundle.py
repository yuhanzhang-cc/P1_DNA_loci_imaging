"""Build fresh immutable composite/evaluation kits; never patch an uploaded kit."""
from pathlib import Path,PurePosixPath
import shutil,zipfile,copy
from diagnostics.handoff_v2.common import ROOT,REL,read,write,fresh,sha


def seal(folder,name='completion_bundle_manifest.json',schema='completion_kit_v1',extra=None):
    folder=Path(folder);files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    write(folder/name,dict(schema=schema,files=files,test_arrays_present=False,formal_training_performed=False)|(extra or {}))
    archive=folder.with_name(folder.name+'.zip')
    with archive.open('xb') as f:
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for p in sorted(folder.rglob('*')):
                if p.is_file():z.write(p,(Path(folder.name)/p.relative_to(folder)).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
    digest=sha(archive);archive.with_name(archive.name+'.sha256').write_bytes((digest+'  '+archive.name+'\n').encode())
    return dict(path=str(archive),sha256=digest,bytes=archive.stat().st_size,members=len(files)+1)


def copy_runtime(target):
    from diagnostics.mechanism_v1.contracts import code_hashes
    for name in code_hashes():
        q=target/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,q)
    weak='docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/configs/weak_policy.json'
    q=target/weak;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/weak,q)
    from .manuscript import DOC
    for p in (ROOT/DOC).rglob('*'):
        if p.is_file() and (p.parent.name=='manuscript_templates' or p.name in ('REFERENCES.bib','SOFTWARE_ASSISTANCE.md')):
            q=target/DOC/p.relative_to(ROOT/DOC);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)


def build(root,output,review_dir=None,hardware_profile=None):
    from diagnostics.handoff_v2.bundle import build as mechanism_build
    root=Path(root);output=fresh(output)
    child=mechanism_build(root,output/'mechanism',hardware_profile)
    original=root/'outputs/next_stage_v4_recipe_20261101/04_bundles/ce2d1821ceec/p1_v4_training.zip'
    expected='11fa9c7341fee2803dd550d7d2479f3ef3f56404c963dcf1771149a39a1734e3'
    if sha(original)!=expected:raise ValueError('Historical scratch kit differs')
    with zipfile.ZipFile(original) as z:
        if z.testzip():raise ValueError('Historical kit CRC failure')
        if sum(i.file_size for i in z.infolist())>1024**3:raise ValueError('Unexpected old kit size')
        for i in z.infolist():
            p=PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in i.filename or ':' in i.filename or not p.parts or p.parts[0]!='p1_v4_training':raise ValueError('Unsafe historical kit member')
            if i.is_dir():continue
            q=output/i.filename;q.parent.mkdir(parents=True,exist_ok=True)
            with z.open(i) as src,q.open('xb') as dst:shutil.copyfileobj(src,dst)
    copy_runtime(output)
    if hardware_profile:
        (output/'configs').mkdir();shutil.copyfile(hardware_profile,output/'configs/hardware_profile.json')
    from .catalog import inventory,portable_payload
    portable_payload(inventory(root),output/'baseline_payload')
    from research_v4.external import exclusion_manifest
    from research_v4.common import rows
    from diagnostics.mechanism_v1.runner import corpus
    split=rows(root/'outputs/261024_clean220_random/frozen_split.csv');excluded=exclusion_manifest(corpus(root),split)
    excluded['sample_ids'] += [r['sample_id'] for r in split]
    write(output/'assets/reference_exclusions.json',excluded)
    # The intermediate child ZIP is not an input to runtime, retain it outside.
    child_zip=output/'mechanism.zip'
    child_zip.rename(output.parent/(output.name+'_mechanism_child.zip'))
    old_checksum=output/'mechanism.zip.sha256';renamed=output.parent/(output.name+'_mechanism_child.zip')
    old_checksum.rename(output.parent/(output.name+'_mechanism_child_original.sha256'))
    renamed.with_name(renamed.name+'.sha256').write_bytes((sha(renamed)+'  '+renamed.name+'\n').encode())
    (output/'mechanism_build.json').rename(output.parent/(output.name+'_mechanism_build.json'))
    entry='deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py';py='"${P1_PYTHON:-p1_v4_training/.venv_cuda/bin/python}"'
    scripts={
        '00_verify.sh':f'python3 {entry} cloud-verify --root .\n',
        '01_setup.sh':'test ! -e p1_v4_training/.venv_cuda || { echo "Existing environment protected; reuse via P1_PYTHON"; exit 1; }\nbash p1_v4_training/01_setup.sh\n',
        '02_import_review.sh':'responses="${1:-incoming/blind_audit_responses.csv}"\ndecision="${2:-incoming/research_decision.json}"\n'+f'{py} mechanism/deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py freeze --responses "$responses" --decision "$decision" --output mechanism/decisions --config-dir mechanism/configs\n',
        '03_plan.sh':f'python3 {entry} cloud-plan --root .\n',
        '04_start.sh':f'{py} {entry} cloud-start --root . --execute\n',
        '05_status.sh':f'python3 {entry} cloud-status --root .\n',
        '06_resume.sh':f'{py} {entry} cloud-start --root . --execute --resume\n',
        '07_failure.sh':f'python3 {entry} cloud-status --root .\n'+f'{py} mechanism/deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py failure --root mechanism\n',
    }
    for name,body in scripts.items():(output/name).write_bytes(('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'+body).encode())
    (output/'incoming').mkdir()
    if review_dir:
        from diagnostics.handoff_v2.freeze import freeze,verify_frozen
        review_dir=Path(review_dir)
        for name in ('blind_audit_responses.csv','research_decision.json'):shutil.copyfile(review_dir/name,output/'incoming'/name)
        freeze(output/'incoming/blind_audit_responses.csv',output/'incoming/research_decision.json',output/'mechanism/decisions',output/'mechanism/configs')
        verify_frozen(output/'mechanism/decisions')
    (output/'README.md').write_text('''# Completion runtime

No formal training has run. Run 00_verify.sh and 03_plan.sh first.
If decisions were pre-frozen at build time, 02_import_review.sh is an idempotent
confirmation with the already included original inputs; no new approval is needed.
Use the historical validated CUDA environment via absolute P1_PYTHON.
If creating a fresh environment: bash 01_setup.sh, then set P1_PYTHON to
$PWD/p1_v4_training/.venv_cuda/bin/python (also the default after fresh setup).
Import two REAL review/decision files with 02_import_review.sh; start 04_start.sh.
The actual matched-backend gate must pass. S0/P0 expand only if the declared gate
passes and expansion was approved, or if full_registered_six was explicitly
approved before new results. The additional three-seed scratch queue runs
only if explicitly approved in additional_baseline_queues. Sequential, one GPU.
Read outputs/completion/status.json for exact return archive paths and hashes.
Download every listed tar.gz AND its sha256. Preserve server until local ingest.
Do not overwrite or reuse an older cloud directory. Private development images
are included; do not upload this ZIP to GitHub. No independent/test arrays.
The independent evaluation and local manuscript/figure finish are separate
explicit commands, because they require returned models and actual reviewed data.
''',encoding='utf-8')
    if hardware_profile:
        (output/'08_hardware_preflight.sh').write_bytes(('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'+f'{py} {entry} hardware-preflight --root . --execute\n').encode())
        with (output/'README.md').open('a',encoding='utf-8') as f:f.write('\nAlternate hardware track: run 08_hardware_preflight.sh before 04_start.sh. It is also enforced automatically before training. Historical initializer hashes remain mandatory. This is not the original 5090 replication track.\n')
    return seal(output,extra=dict(real_approval_included=review_dir is not None,hardware_profile_id=read(hardware_profile)['profile_id'] if hardware_profile else None))


def evaluation_build(root,analysis,output):
    """Works now for frozen baselines; rerun fresh after future models return."""
    from research_v4.external import exclusion_manifest
    from research_v4.common import rows
    from diagnostics.mechanism_v1.runner import corpus
    root=Path(root);analysis=Path(analysis);output=fresh(output);copy_runtime(output)
    records=copy.deepcopy(read(analysis/'analysis_manifest.json')['model_catalog'])
    from .catalog import verify_record
    for r in records:
        verify_record(r);folder=output/'models'/r['method']/(f"seed_{r['seed']}" if r['seed'] is not None else 'deterministic');folder.mkdir(parents=True)
        selection=folder/'selection.json';shutil.copyfile(r['selection'],selection)
        if r['kind']!='classical':
            weight=folder/Path(r['weight']).name;shutil.copyfile(r['weight'],weight)
            for name in r['native_config_hashes']:shutil.copyfile(Path(r['weight']).parent/name,folder/name)
            r['weight']=weight.relative_to(output).as_posix()
        r.update(folder=folder.relative_to(output).as_posix(),selection=selection.relative_to(output).as_posix(),cache=None,cache_sha256=None)
    if (root/'assets/reference_exclusions.json').exists():excluded=read(root/'assets/reference_exclusions.json')
    else:
        c=corpus(root);split=rows(root/'outputs/261024_clean220_random/frozen_split.csv');excluded=exclusion_manifest(c,split)
        excluded['sample_ids']+= [r['sample_id'] for r in split]
    write(output/'assets/reference_exclusions.json',excluded)
    write(output/'analysis_manifest.json',dict(model_catalog=records,portable_paths=True,dataset_role='frozen_models_before_independent_readout',hardware_context=read(analysis/'analysis_manifest.json').get('hardware_context')))
    return seal(output,schema='completion_evaluation_kit_v1')


def all_results(root,archives,delivery,evaluation,output):
    """One download containing only completed result archives and checksums."""
    root=Path(root);output=fresh(output);components={}
    for r in archives:
        name='scratch_results.tar.gz' if 'scratch_return_' in r['path'] else 'mechanism_results.tar.gz'
        if name in components:raise ValueError('Ambiguous result revision; select exact completed archives')
        components[name]=root/r['path']
    components['delivery.zip']=Path(delivery['path']);components['frozen_evaluation.zip']=Path(evaluation['path'])
    for name,source in components.items():
        shutil.copyfile(source,output/name)
        (output/(name+'.sha256')).write_bytes((sha(source)+'  '+name+'\n').encode())
    return seal(output,name='all_results_manifest.json',schema='completion_results_v1',extra=dict(completion_bundle_sha256=sha(root/'completion_bundle_manifest.json'),
        components=list(components),contains_formal_training_results=True,formal_training_performed=True,engineering_only=False))
