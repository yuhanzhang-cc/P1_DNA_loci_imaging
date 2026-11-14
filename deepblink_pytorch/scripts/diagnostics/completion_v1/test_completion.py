"""Synthetic contract/failure tests only; no formal training or human review."""
from pathlib import Path
import sys,copy,json
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'deepblink_pytorch'),str(ROOT/'deepblink_pytorch/scripts'),str(ROOT/'deepblink_pytorch/scripts/diagnostics/scaffolds')]
from diagnostics.completion_v1 import analysis,catalog,cloud,inference,group_statistics,benchmark
from diagnostics.handoff_v2.common import write,sha,csv_write,read


@pytest.mark.parametrize('values,expected',[([1,1,1],.25),([0,0,0],1.),([-1,-1,-1],.25),([],None)])
def test_exact_seed_probability(values,expected):assert analysis.sign_flip(values)==expected


def test_missing_tests_do_not_shrink_holm_family():
    assert analysis.holm([.01,None,.03,None,.2])==[.05,None,.12,None,.6000000000000001]
    with pytest.raises(ValueError):analysis.holm([float('nan')])


def test_partial_and_negative_outcomes_keep_simple_candidate():
    rows=[dict(method=m,seed=s,f1=.8 if m!='P0' else .7) for m in ('S0','P0','G0') for s in (42,43,44)]
    assert analysis.classify_outcome(rows)=='complete_negative_development_keep_G0'
    assert analysis.classify_outcome(rows[:-4])=='partial_mechanism_results_keep_G0'
    for r in rows:
        if r['method']=='P0':r['f1']=.9
    assert analysis.classify_outcome(rows)=='complete_development_candidate_requires_review'


def test_group_bootstrap_preserves_seed_and_group_axes():
    a=np.tile([[[10,0,0],[10,0,0]]],(3,1,1));b=np.tile([[[5,5,5],[5,5,5]]],(3,1,1))
    r=group_statistics.cluster_interval(a,b,draws=100)
    assert r['mean_f1_delta']==r['ci_low']==r['ci_high']==.5
    assert r['independent_groups']==2 and r['fixed_model_seeds']==3 and r['p_value'] is None
    with pytest.raises(ValueError):group_statistics.cluster_interval(a[:,:1],b[:,:1])
    with pytest.raises(ValueError):group_statistics.cluster_interval(a,b[:2])


def synthetic_manifest(tmp_path):
    images=tmp_path/'images.csv';csv_write(images,[dict(sample_id='SYN1',biological_group_id='SYN_GROUP',acquisition_id='SYN_ACQ',source_container='SYN_CONTAINER')],['sample_id','biological_group_id','acquisition_id','source_container'])
    evidence=tmp_path/'evidence.txt';evidence.write_text('SYNTHETIC METADATA TEST ONLY')
    spec=dict(dataset_role='independent_test',independence_reviewed=True,synthetic_fixture=False,reviewer='SYNTHETIC_METADATA_TEST',reviewed_at='SYNTHETIC_DATE',independence_notes='metadata-only fixture; not scientific data',
        images_csv=images.name,images_csv_sha256=sha(images),independence_evidence_file=evidence.name,independence_evidence_sha256=sha(evidence))
    path=tmp_path/'manifest.json';write(path,spec);return path,spec


def test_independent_metadata_dryrun_reads_no_images(tmp_path,monkeypatch):
    p,s=synthetic_manifest(tmp_path)
    # No image paths even exist: metadata validation must not imply pixel validity.
    r=inference.review_manifest(p,tmp_path,read_pixels=False)
    assert r['images_read']==0 and r['verified_metadata_only']
    s['independence_reviewed']=False;write(p,s,replace=True)
    with pytest.raises(ValueError,match='provenance'):inference.review_manifest(p,tmp_path,False)


@pytest.mark.parametrize('mutation',['hash','escape','group','synthetic'])
def test_independent_manifest_rejects_bad_provenance(tmp_path,mutation):
    p,s=synthetic_manifest(tmp_path)
    if mutation=='hash':s['images_csv_sha256']='0'*64
    if mutation=='escape':s['images_csv']='../escape.csv'
    if mutation=='group':
        q=tmp_path/'images.csv';csv_write(tmp_path/'bad.csv',[dict(sample_id='SYN1',biological_group_id='TODO',acquisition_id='A',source_container='S')],['sample_id','biological_group_id','acquisition_id','source_container']);s['images_csv']='bad.csv';s['images_csv_sha256']=sha(tmp_path/'bad.csv')
    if mutation=='synthetic':s['synthetic_fixture']=True
    write(p,s,replace=True)
    with pytest.raises(ValueError):inference.review_manifest(p,tmp_path,False)


def test_cloud_plan_never_launches_and_exposes_all_gates():
    p=cloud.plan(Path('/SYNTHETIC'))
    assert p['dry_run'] and not p['formal_training_started'] and p['maximum_new_runs']==9
    assert not p['scientific_success_guaranteed'] and not p['independent_data_included']


def test_status_is_readonly_and_training_eta_is_qualified(tmp_path):
    s=cloud.status(tmp_path)
    assert len(s['tasks'])==9 and all(r['status']=='not_started' for r in s['tasks']) and s['conditional_remaining_epoch_seconds'] is None
    assert not s['completion']['formal_training_started']
    assert not list(tmp_path.iterdir())
    write(tmp_path/'mechanism/decisions/research_decision.json',dict(completion_strategy='full_registered_six',additional_baseline_queues=['spotiflow_scratch']))
    folder=tmp_path/'mechanism/outputs/trials/S0/seed_42';write(folder/'status.json',dict(status='running',epoch=10,updates=220))
    csv_write(folder/'training_metrics.csv',[dict(epoch=i,seconds=2) for i in range(1,11)],['epoch','seconds'])
    s=cloud.status(tmp_path);assert s['conditional_remaining_epoch_seconds'][0]==s['conditional_remaining_epoch_seconds'][1]
    assert s['tasks'][0]['median_recent_callback_seconds']==2
    assert s['completion']['formal_training_started']
    write(tmp_path/'mechanism/outputs/trials/P0/seed_42/status.json',dict(status='running',epoch=1,optimizer_updates=22))
    assert next(t for t in cloud.status(tmp_path)['tasks'] if t['method']=='P0' and t['seed']==42)['updates']==22


def test_composite_rejects_source_change_before_launch(tmp_path):
    f=tmp_path/'source.py';f.write_text('SYNTHETIC')
    write(tmp_path/'completion_bundle_manifest.json',dict(schema='completion_kit_v1',test_arrays_present=False,files={'source.py':sha(f)}))
    f.write_text('changed')
    with pytest.raises(ValueError,match='source differs'):cloud.execute(tmp_path,launch=lambda *args:pytest.fail('must not launch'))


@pytest.mark.parametrize('state',['running','complete'])
def test_status_accepts_native_scratch_identity_fields(tmp_path,state):
    folder=tmp_path/'p1_v4_training/outputs/research_v4_cuda/spotiflow_scratch/Spotiflow_scratch_control/seed_43'
    path=folder/'status.json'
    write(path,dict(method='native_adapter',seed=43,status=state,epoch=100,optimizer_updates=4400))
    before=path.read_bytes()
    result=cloud.status(tmp_path)
    task=next(t for t in result['tasks'] if t['method']=='Spotiflow_scratch' and t['seed']==43)
    assert task['status']==state and task['updates']==4400 and task['epoch']==100
    assert path.read_bytes()==before


def test_synthetic_timing_is_not_gpu_validation(tmp_path):
    calls=[]
    def predict(r,data,device):calls.append(1);return [(np.zeros((1,2)),np.ones(1))]
    data=dict(x=[np.zeros((8,8))],original_shape=[(8,8)],sample_id=['SYN'],y=[np.empty((0,2))])
    records=[dict(method='SYN',seed=42)]
    assert benchmark.measure(records,data,tmp_path/'dry')['dry_run'] and not calls
    result=benchmark.measure(records,data,tmp_path/'measured',predict=predict,dry_run=False)
    assert result['optimizer_updates']==0 and len(calls)==4
    assert read(tmp_path/'measured/provenance.json')['gpu']=='SYNTHETIC_ENGINEERING'


def test_baseline_payload_roundtrip_and_tampering(tmp_path):
    folder=tmp_path/'source';folder.mkdir();write(folder/'selection.json',dict(primary=dict(threshold=.3)))
    write(folder/'selected/metrics.json',dict(f1=.2));(folder/'selected_candidates.npz').write_bytes(b'SYNTHETIC NOT NUMPY')
    record=dict(method='LoG',seed=None,kind='classical',folder=str(folder),selection=str(folder/'selection.json'),selection_sha256=sha(folder/'selection.json'),cache=str(folder/'selected_candidates.npz'),cache_sha256=sha(folder/'selected_candidates.npz'))
    catalog.portable_payload([record],tmp_path/'payload');r=catalog.load_portable(tmp_path/'payload')
    assert r[0]['method']=='LoG'
    Path(r[0]['cache']).write_bytes(b'TAMPERED')
    with pytest.raises(ValueError,match='cache changed'):catalog.load_portable(tmp_path/'payload')


def test_optional_scratch_is_explicit_not_automatic():
    from diagnostics.handoff_v2.freeze import validate_decision
    from diagnostics.handoff_v2.test_handoff import decision
    d=decision();validate_decision(d,synthetic=True)
    d['schema']='p1_research_decision_v3';d['additional_baseline_queues']=['spotiflow_scratch'];validate_decision(d,synthetic=True)
    d['additional_baseline_queues']=['baseline_r0']
    with pytest.raises(ValueError,match='baseline'):validate_decision(d,synthetic=True)


def test_complete_ablation_needs_explicit_preoutcome_authorization():
    from diagnostics.handoff_v2.pipeline import expansion_tasks
    from diagnostics.handoff_v2.freeze import validate_decision
    from diagnostics.handoff_v2.test_handoff import decision
    d=decision();d['schema']='p1_research_decision_v3';d['completion_strategy']='full_registered_six';validate_decision(d,synthetic=True)
    assert len(expansion_tasks(dict(passed=False),d))==4
    d['auto_expand_if_gate_passes']=False
    with pytest.raises(ValueError,match='six-run'):validate_decision(d,synthetic=True)
    d.pop('completion_strategy')
    assert not expansion_tasks(dict(passed=False),d)


def test_new_decision_cannot_silently_run_in_legacy_kit():
    from diagnostics.handoff_v2.freeze import validate_decision
    from diagnostics.handoff_v2.test_handoff import decision
    d=decision();d['additional_baseline_queues']=['spotiflow_scratch']
    with pytest.raises(ValueError,match='v3'):validate_decision(d,synthetic=True)


def fake_analysis(folder):
    fields=['method','seed','f1','ranked_ap','weak_recall','mean_error_px','precision','recall','track']
    rows=[dict(method='G0',seed=s,f1=.8,ranked_ap=.7,weak_recall=.4,mean_error_px=.3,precision=.9,recall=.7,track='SYNTHETIC') for s in (42,43,44)]
    csv_write(folder/'per_seed_metrics.csv',rows,fields)
    csv_write(folder/'summary.csv',[dict(method='G0',metric='f1',n=3,mean=.8,sd=0)],['method','metric','n','mean','sd'])
    csv_write(folder/'ranked_pr_curves.csv',[dict(method='G0',seed=42,score=.8,tp=1,fp=0,recall=.5,precision=1)],['method','seed','score','tp','fp','recall','precision'])
    csv_write(folder/'common_tp_errors.csv',[],['candidate','reference','seed','sample_id','true_index','candidate_error_px','reference_error_px','error_delta_px'])
    csv_write(folder/'planned_f1_comparisons.csv',[],['comparison','p_holm'])
    write(folder/'analysis_manifest.json',dict(dataset_role='development_validation',test_evaluated=False,status='SYNTHETIC_ENGINEERING_ONLY',model_catalog=[]))


def test_figures_and_paper_keep_missing_results_pending(tmp_path):
    from diagnostics.completion_v1 import figures,manuscript
    a=tmp_path/'analysis';a.mkdir();fake_analysis(a)
    result=figures.render(a,tmp_path/'figures')
    assert result['figures']==5 and len(list((tmp_path/'figures').glob('*.pdf')))==5
    paper=manuscript.render(a,tmp_path/'paper')
    assert not paper['submission_ready'] and paper['pending_slots']>0
    text=(tmp_path/'paper/RESULTS.md').read_text(encoding='utf-8')
    assert '[PENDING: P0_f1]' in text and '0.8000' in text
    with pytest.raises(FileExistsError):figures.render(a,tmp_path/'figures')


def test_bad_group_pairing_rejects_before_inference(tmp_path):
    from diagnostics.completion_v1.group_statistics import independent_summary
    f=tmp_path/'evaluation';f.mkdir();write(f/'complete.json',dict(test_evaluated=True,engineering_only=False))
    rows=[dict(method=m,seed=s,sample_id='SYN',biological_group_id='A' if m=='S0' else 'B',tp=1,fp=0,fn=0) for m in ('S0','H0') for s in (42,43,44)]
    csv_write(f/'per_image_metrics.csv',rows,list(rows[0]))
    with pytest.raises(ValueError,match='group identities'):independent_summary(f,tmp_path/'stats')


def test_delivery_zip_roundtrip_preserves_source_binding(tmp_path):
    from diagnostics.completion_v1.bundle import seal
    from diagnostics.completion_v1.result_io import delivery
    source=tmp_path/'source_manifest.json';write(source,dict(SYNTHETIC=True))
    folder=tmp_path/'delivery';folder.mkdir();write(folder/'SYNTHETIC_ONLY.json',dict(engineering_only=True))
    artifact=seal(folder,name='delivery_manifest.json',schema='completion_delivery_v1',extra=dict(completion_bundle_sha256=sha(source)))
    path=Path(artifact['path']);out=tmp_path/'ingested'
    assert delivery(path,path.with_name(path.name+'.sha256'),source,out)['verified']
    assert read(out/'SYNTHETIC_ONLY.json')['engineering_only']
    with pytest.raises(FileExistsError):delivery(path,path.with_name(path.name+'.sha256'),source,out)
    other=tmp_path/'other.json';write(other,dict(SYNTHETIC='different'))
    with pytest.raises(ValueError,match='different uploaded'):delivery(path,path.with_name(path.name+'.sha256'),other,tmp_path/'wrong')


@pytest.mark.parametrize('name',['../escape','/absolute','C:/escape','x\\escape','.git/config'])
def test_new_ingestion_rejects_unsafe_paths(name):
    from diagnostics.completion_v1.result_io import safe_member
    with pytest.raises(ValueError):safe_member(name)


def test_source_binding_never_accepts_tampered_selected_weight(tmp_path):
    folder=tmp_path/'model';folder.mkdir();(folder/'weight.pt').write_bytes(b'SYNTHETIC')
    write(folder/'selection.json',dict(primary=dict(calibrated_probability=.5)))
    record=dict(kind='own',selection=str(folder/'selection.json'),selection_sha256=sha(folder/'selection.json'),weight=str(folder/'weight.pt'),weight_sha256=sha(folder/'weight.pt'),threshold=.5,native_config_hashes={})
    assert catalog.verify_record(record)
    record['threshold']=.2
    with pytest.raises(ValueError,match='threshold'):catalog.verify_record(record)
    record['threshold']=.5;(folder/'weight.pt').write_bytes(b'TAMPERED')
    with pytest.raises(ValueError,match='Weight'):catalog.verify_record(record)


def test_independent_complete_summary_and_figure_synthetic(tmp_path):
    from diagnostics.completion_v1.group_statistics import independent_summary
    from diagnostics.completion_v1.figures import independent
    f=tmp_path/'SYNTHETIC_evaluation';f.mkdir();write(f/'complete.json',dict(test_evaluated=True,engineering_only=True))
    # Explicit synthetic stand-in exercises the post-readout functions only;
    # this directory is never passed to formal evaluation or real evidence.
    rows=[dict(method=m,seed=s,sample_id=f'SYN_{g}',biological_group_id=f'SYN_GROUP_{g}',tp=5 if m=='S0' else 4,fp=1,fn=1) for m in ('S0','H0') for s in (42,43,44) for g in (1,2)]
    csv_write(f/'per_image_metrics.csv',rows,list(rows[0]))
    csv_write(f/'per_seed_metrics.csv',[dict(method=m,seed=s,f1=.8) for m in ('S0','H0') for s in (42,43,44)],['method','seed','f1'])
    with pytest.raises(ValueError,match='Engineering'):independent_summary(f,tmp_path/'rejected')
    result=independent_summary(f,tmp_path/'stats',allow_engineering=True);assert result['comparisons']==5
    assert independent(f,tmp_path/'stats',tmp_path/'figures',allow_engineering=True)['exports']==3


def test_qualitative_export_preserves_pixels_and_coordinate_units(tmp_path):
    from diagnostics.completion_v1.qualitative import montage,build
    image=np.arange(80*80,dtype=np.float32).reshape(80,80);old=image.copy()
    case=dict(anonymous_id='SYN0001',sample_id='SYN',crop_y0=4,crop_x0=8,crop_y1=68,crop_x1=72,candidate_is_real_spot='uncertain')
    result=montage([case],{'SYN':image},{'SYN':[[20.25,30.5]]},{'G0':{'SYN':[[20.5,30.75]]}},tmp_path/'montage',engineering_only=True)
    assert result['exports']==3 and np.array_equal(image,old)
    with np.load(tmp_path/'montage/SYN0001_raw.private.npz') as z:assert np.array_equal(z['image'],old[4:68,8:72])
    p=read(tmp_path/'montage/provenance.json');assert p['raw_pixels_modified'] is False and 'native pixels' in p['scale_bar']
    assert build(tmp_path,tmp_path,None,None,None,tmp_path/'dry',True)['images_read']==0


def test_deployment_needs_no_labels_and_uses_frozen_threshold(tmp_path,monkeypatch):
    import tifffile
    model=tmp_path/'model';model.mkdir();write(model/'selection.json',dict(primary=dict(calibrated_probability=.5)));(model/'weight.pt').write_bytes(b'SYNTHETIC_MODEL')
    record=dict(method='S0',seed=42,kind='signed',folder=str(model),selection=str(model/'selection.json'),selection_sha256=sha(model/'selection.json'),weight=str(model/'weight.pt'),weight_sha256=sha(model/'weight.pt'),threshold=.5,native_config_hashes={})
    a=tmp_path/'analysis';write(a/'analysis_manifest.json',dict(model_catalog=[record]))
    image=tmp_path/'SYNTHETIC.tif';tifffile.imwrite(image,np.zeros((256,256),np.float32));old=sha(image)
    monkeypatch.setattr(inference,'predict_record',lambda *args:[(np.asarray([[2.25,3.5],[4,5]],np.float32),np.asarray([.7,.5]))])
    result=inference.predict_image(a,image,'S0',42,tmp_path/'predicted')
    assert result['predictions']==1 and not result['labels_used'] and sha(image)==old
    from diagnostics.handoff_v2.common import table
    assert float(table(tmp_path/'predicted/prediction_points.csv')[0]['x'])==3.5
