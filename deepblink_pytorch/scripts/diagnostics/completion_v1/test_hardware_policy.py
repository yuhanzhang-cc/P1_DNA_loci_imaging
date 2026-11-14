"""Synthetic hardware identities and export fixtures; no CUDA validation."""
from pathlib import Path
import sys,copy
import pytest
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'deepblink_pytorch'),str(ROOT/'deepblink_pytorch/scripts')]
from diagnostics.completion_v1 import hardware_policy as hp
from diagnostics.completion_v1.transfer_check import compare_exports,preflight
from diagnostics.handoff_v2.common import write,read,csv_write


def spec():
    return dict(profile_id=hp.PROFILE_ID,experiment_track='vgpu32_development_v1',torch='2.11.0+cu128',
        cuda_runtime='12.8',compute_capability=[8,9],require_historical_initializations=True,same_hardware_as_historical_baselines=False)


def actual():
    return dict(gpu='NVIDIA vGPU-32GB',torch='2.11.0+cu128',cuda_runtime='12.8',compute_capability=[8,9],memory_bytes=32*1024**3)


def initializations():return [dict(seed=s,paired_exact=True,historical_exact=True) for s in (42,43,44)]


@pytest.mark.parametrize('name,cc,memory,passed',[
    ('NVIDIA GeForce RTX 5090 D',[12,0],32,True),
    ('NVIDIA GeForce RTX 5090D',[12,0],32,True),
    ('NVIDIA GeForce RTX 5090',[12,0],32,False),
    ('NVIDIA GeForce RTX 5090 D v2',[12,0],24,False),
    ('NVIDIA GeForce RTX 5090 D',[8,9],32,False),
    ('NVIDIA GeForce RTX 5090 D',[12,0],24,False),
    ('NVIDIA GeForce RTX 4090',[8,9],24,False)])
def test_5090d_profile_is_separate_and_specific(name,cc,memory,passed):
    s=spec();s.update(profile_id='autodl_5090d_blackwell_transfer_v1',compute_capability=[12,0],experiment_track='rtx5090d_development_v1')
    a=actual();a.update(gpu=name,compute_capability=cc,memory_bytes=memory*1024**3)
    assert hp.judge(s,a,initializations())['passed']==passed
    i=initializations();i[0]['historical_exact']=False
    assert not hp.judge(s,a,i)['passed']


@pytest.mark.parametrize('name',['NVIDIA vGPU-32GB','vGPU-32G','NVIDIA GeForce RTX 4080 SUPER','NVIDIA GeForce RTX 4080 SUPER 32GB'])
def test_only_registered_aliases_pass(name):
    a=actual();a['gpu']=name
    assert hp.judge(spec(),a,initializations())['passed']


@pytest.mark.parametrize('key,value',[('gpu','NVIDIA GeForce RTX 5090'),('gpu','unknown 32GB'),('torch','2.8.0+cu128'),('cuda_runtime','13.0'),('compute_capability',[12,0]),('memory_bytes',24*1024**3)])
def test_wrong_device_or_software_rejects(key,value):
    a=actual();a[key]=value
    assert not hp.judge(spec(),a,initializations())['passed']


@pytest.mark.parametrize('mutation',['missing','paired','historical'])
def test_initialization_requirements_cannot_be_waived(mutation):
    i=initializations()
    if mutation=='missing':i.pop()
    else:i[1][mutation+'_exact']=False
    assert not hp.judge(spec(),actual(),i)['passed']


@pytest.mark.parametrize('key,value',[('require_historical_initializations',False),('same_hardware_as_historical_baselines',True),('compute_capability',[12,0]),('experiment_track','original_5090')])
def test_weakened_profile_rejected(key,value):
    s=spec();s[key]=value
    with pytest.raises(ValueError):hp.validate(s)


def test_legacy_profile_is_absent_and_dryrun_never_touches_cuda(tmp_path):
    assert hp.profile(tmp_path) is None
    assert hp.binding(tmp_path,None,'cuda')=={}
    assert not preflight(tmp_path)['required']
    write(tmp_path/'configs/hardware_profile.json',spec())
    result=preflight(tmp_path)
    assert result['dry_run'] and result['optimizer_updates']==0 and not result['training_started']
    assert not (tmp_path/'outputs').exists()
    assert hp.context(tmp_path)['preflight_receipt_sha256'] is None


def test_resume_hardware_identity_is_locked(tmp_path):
    path=tmp_path/'lock.json';hp.lock_backend(path,actual());hp.lock_backend(path,actual())
    other=actual();other['gpu_uuid']='NEW_DEVICE'
    with pytest.raises(ValueError,match='identity changed'):hp.lock_backend(path,other)
    assert read(path)==actual()


def exports(path,points,counts=(2,0,0)):
    write(path/'metrics.json',dict(zip(('tp','fp','fn'),counts)))
    csv_write(path/'prediction_points.csv',[dict(sample_id=s,y=y,x=x,p=p) for s,y,x,p in points],['sample_id','y','x','p'])


@pytest.mark.parametrize('mutation,passed',[('reorder',True),('tiny',True),('coordinate',False),('counts',False),('image',False)])
def test_frozen_prediction_parity(mutation,passed,tmp_path):
    a=[('SYN',10.,10.,.9),('SYN',20.,20.,.8)];b=copy.deepcopy(a);counts=(2,0,0)
    if mutation=='reorder':b.reverse()
    if mutation=='tiny':b[0]=('SYN',10.001,10.,.89999)
    if mutation=='coordinate':b[0]=('SYN',10.01,10.,.9)
    if mutation=='counts':counts=(1,1,1)
    if mutation=='image':b[0]=('SYN2',10.,10.,.9)
    exports(tmp_path/'old',a);exports(tmp_path/'new',b,counts)
    r=compare_exports(tmp_path/'old',tmp_path/'new',['SYN','SYN2'])
    assert r['passed']==passed and r['optimizer_updates']==0


def test_unknown_image_or_nonfinite_parity_rejects(tmp_path):
    exports(tmp_path/'old',[('SYN',10.,10.,.9)])
    exports(tmp_path/'new',[('OTHER',10.,10.,.9)])
    with pytest.raises(ValueError,match='image ID'):compare_exports(tmp_path/'old',tmp_path/'new',['SYN'])


def test_doctor_requires_actual_cuda_for_alternate_profile(tmp_path):
    # CPU doctor result is engineering-only even if CPU initializers are paired.
    from diagnostics.handoff_v2.doctor import diagnose
    import shutil
    old=ROOT/'docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics'
    cfg=tmp_path/'configs';cfg.mkdir()
    write(cfg/'hardware_profile.json',spec())
    shutil.copyfile(old/'implementation_20261104/configs/reference_initializations.json',cfg/'reference_initializations.json')
    shutil.copyfile(old/'handoff_20261111/reference_environment.json',cfg/'reference_environment.json')
    r=diagnose(tmp_path/'doctor',device='cpu',context_root=tmp_path)
    assert r['passed'] and r['optimizer_updates']==0 and r['alternate_hardware_cuda_verified'] is False
    assert any(not row['historical_exact'] for row in r['initializations'])


def test_transfer_failure_stops_before_loading_data(tmp_path,monkeypatch):
    from diagnostics.handoff_v2 import doctor
    write(tmp_path/'configs/hardware_profile.json',spec())
    monkeypatch.setattr(doctor,'diagnose',lambda *a,**k:dict(passed=False,optimizer_updates=0))
    with pytest.raises(RuntimeError,match='no training allowed'):preflight(tmp_path,dry_run=False)
    assert not (tmp_path/'outputs/hardware_transfer/complete.json').exists()
    assert not (tmp_path/'outputs/hardware_transfer/backend_lock.json').exists()
