"""Render English manuscript components from verified result tables."""
from pathlib import Path
import re,shutil
from diagnostics.handoff_v2.common import ROOT,read,table,write,fresh,sha

DOC=Path('docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/completion_20261111')


def render(analysis,output,independent=None,statistics_folder=None,author_metadata=None):
    analysis=Path(analysis);output=fresh(output);manifest=read(analysis/'analysis_manifest.json')
    summary=table(analysis/'summary.csv');values={}
    for r in summary:
        if not r['mean']:continue
        mean=float(r['mean']);sd=float(r['sd']) if r['sd'] else None
        values[r['method']+'_'+r['metric']]=f'{mean:.4f}'+(f' ± {sd:.4f}' if sd is not None else ' (deterministic)' if r['method'] in ('LoG','DoG') else ' (one seed)')
    actual=table(analysis/'per_seed_metrics.csv')
    prospective=[r for r in actual if r['method'] in ('S0','P0')]
    values['prospective_completed_training_count']=str(len(prospective))
    context=manifest.get('mechanism_context',{})
    if context:
        values['approved_continuation_strategy']=context.get('completion_strategy','gate_conditioned')
        values['approved_p0_radius_px']=str(context['approved_supervision']['radius_px'])
        values['blind_audit_status']='completed and hash-bound to the approved experiment decision'
        counts=context['review_summary'].get('answer_counts',{})
        values['blind_audit_summary']='60 completed candidate-presence reviews; '+', '.join(str(k)+'='+str(v) for k,v in counts.items())+'; assignment TP/FP remains a separate question'
    if manifest.get('screening_gate'):
        gate=manifest['screening_gate'];values['screen_gate_result']='passed' if gate['passed'] else 'not passed'
        values['p0_s0_near_truth_fp_difference']=str(gate['near_truth_fp_delta'])+' at seed42 (geometry proxy)'
    errors=table(analysis/'common_tp_errors.csv');v=[float(r['error_delta_px']) for r in errors if r['candidate']=='P0' and r['reference']=='S0']
    if v:
        import statistics
        seeds=sorted({r['seed'] for r in errors if r['candidate']=='P0' and r['reference']=='S0'})
        means=[statistics.mean(float(r['error_delta_px']) for r in errors if r['candidate']=='P0' and r['reference']=='S0' and r['seed']==s) for s in seeds]
        values['p0_s0_common_tp_error']=f'{statistics.mean(means):+.4f} px (mean of {len(seeds)} paired seed estimates)'
        values['prospective_mechanism_abstract_result']=values.get('P0_f1','pending')+' development micro-F1 for P0, with '+values['p0_s0_common_tp_error']+' common-TP error relative to S0'
    independent_complete=False;additional_sources={}
    if independent:
        independent=Path(independent);completion=read(independent/'complete.json')
        if completion.get('test_evaluated') is not True or completion.get('engineering_only') is not False:raise ValueError('Only a completed real independent readout can fill manuscript results')
        observations=table(independent/'per_image_metrics.csv');groups={r['biological_group_id'] for r in observations};images={r['sample_id'] for r in observations}
        values.update(independent_dataset_description=f'{len(images)} images from {len(groups)} reviewed biological groups',independent_image_count=str(len(images)),
            independent_biological_groups=str(len(groups)),independent_evaluation_status='completed with frozen validation-selected models and thresholds')
        additional_sources['independent_complete_sha256']=sha(independent/'complete.json');independent_complete=True
        if statistics_folder:
            comparisons=table(Path(statistics_folder)/'independent_group_intervals.csv')
            text=[]
            for c in comparisons:
                if c['status']=='complete_conditional_estimation':text.append(c['comparison']+f": delta F1 {float(c['mean_f1_delta']):+.4f}, conditional 95% group interval [{float(c['ci_low']):+.4f}, {float(c['ci_high']):+.4f}]")
            if text:values['independent_results_summary']='; '.join(text);values['independent_primary_result']=next((t for t in text if t.startswith('P0 - G0')),text[0])
            additional_sources['independent_intervals_sha256']=sha(Path(statistics_folder)/'independent_group_intervals.csv')
    if author_metadata:
        meta=read(author_metadata)
        if meta.get('reviewed') is not True or not meta.get('reviewer') or not meta.get('reviewed_at'):raise ValueError('Author metadata must be actually reviewed')
        allowed={'microscope_and_optics','sample_preparation_and_labeling','exposure_pixel_size_channels','annotator_number_and_rules','biological_and_acquisition_groups',
            'blind_audit_reviewer_details','target_journal_and_article_type','figure1_image_and_calibration','reviewed_error_examples','final_mechanism_interpretation',
            'independent_generalization_interpretation','biological_validation_scope'}
        for key,value in meta.get('fields',{}).items():
            if key not in allowed or not isinstance(value,str):raise ValueError('Unknown or nontext author field')
            if value.strip() and not value.startswith('TODO'):values[key]=value
        additional_sources['author_metadata_sha256']=sha(author_metadata)
    tokens={};remaining=[]
    for name in ('METHODS.md','INTRODUCTION.md','RESULTS.md','DISCUSSION.md','ABSTRACT.md','FIGURE_PLAN.md'):
        path=ROOT/DOC/'manuscript_templates'/name;text=path.read_text(encoding='utf-8')
        def sub(match):
            key=match.group(1)
            if key in values:tokens[key]=values[key];return values[key]
            remaining.append(dict(file=name,slot=key,reason='not yet executed, supplied or approved'));return '[PENDING: '+key+']'
        text=re.sub(r'\{\{([A-Za-z0-9_]+)\}\}',sub,text)
        if name in ('METHODS.md','RESULTS.md','DISCUSSION.md') and manifest.get('hardware_context'):
            text+='\n\nHardware-track qualification: S0, P0 and the new scratch control were run on the separately registered vGPU32 track. P0 versus S0 is a within-track contrast. Comparisons against historical H0, R0, G0 and transferred Spotiflow are cross-hardware development references and do not isolate architecture or supervision effects from hardware. Passing fixed-checkpoint inference checks does not establish training equivalence.\n'
        if name=='RESULTS.md':
            text+='\n\n### Automatic outcome status\n\n'+manifest['status']+'. '
            text+='The current practical candidate remains G0 pending review of the prospective mechanism results and independent evidence. No superiority claim is generated automatically.\n'
        (output/name).write_text(text,encoding='utf-8')
    (output/'MANUSCRIPT_DRAFT.md').write_text('\n\n'.join((output/n).read_text(encoding='utf-8') for n in ('ABSTRACT.md','INTRODUCTION.md','METHODS.md','RESULTS.md','DISCUSSION.md')),encoding='utf-8')
    for name in ('REFERENCES.bib','SOFTWARE_ASSISTANCE.md'):shutil.copyfile(ROOT/DOC/name,output/name)
    write(output/'pending_slots.json',remaining)
    write(output/'claim_checks.json',dict(submission_ready=False,independent_validation_complete=independent_complete,architecture_superiority_established=False,
        human_review_fabricated=False,pending_slots=len(remaining),existing_conclusion_changed=False,analysis_sha256=sha(analysis/'analysis_manifest.json'),
        source_template_hashes={p.name:sha(p) for p in (ROOT/DOC/'manuscript_templates').glob('*.md')},bound_values=tokens,additional_sources=additional_sources))
    return dict(output=str(output),filled_numeric_slots=len(tokens),pending_slots=len(remaining),submission_ready=False)
