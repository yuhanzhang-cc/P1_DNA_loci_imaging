"""Unified completion entry; execution and independent readout are opt-in."""
from pathlib import Path
import argparse,json
from diagnostics.handoff_v2.common import ROOT,read,write,fresh


def finish(root,output,returned=None,scratch=None,records=None,device='cpu',benchmark_device=None):
    from .analysis import analyze
    from .figures import render as figures
    from .manuscript import render as manuscript
    output=fresh(output)
    result=analyze(root,output/'analysis',returned,scratch,device,records)
    results=dict(analysis=result,figures=figures(output/'analysis',output/'figures'),manuscript=manuscript(output/'analysis',output/'manuscript'))
    if benchmark_device:
        from .benchmark import measure
        from diagnostics.mechanism_v1.runner import corpus
        root=Path(root);assets=root/'mechanism/assets'
        data=corpus(root,assets if assets.exists() else None).splits['valid']
        results['benchmark']=measure(read(output/'analysis/analysis_manifest.json')['model_catalog'],data,output/'benchmark',device=benchmark_device,dry_run=False)
    write(output/'completion.json',dict(**results,formal_training_performed=False,independent_test_evaluated=False,submission_ready=False))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    for name in ('audit','build','catalog','finish'):
        a=s.add_parser(name);a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--output',type=Path,required=True)
        if name=='build':a.add_argument('--review-dir',type=Path)
        if name in ('finish','catalog'):a.add_argument('--returned',type=Path);a.add_argument('--scratch',type=Path)
    for name in ('figures','manuscript'):
        a=s.add_parser(name);a.add_argument('--analysis',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if name=='manuscript':a.add_argument('--independent',type=Path);a.add_argument('--statistics',type=Path);a.add_argument('--author-metadata',type=Path)
    for name in ('ingest-delivery','ingest-scratch','ingest-all'):
        a=s.add_parser(name);a.add_argument('--archive',type=Path,required=True);a.add_argument('--checksum',type=Path,required=True);a.add_argument('--source-manifest',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    for name in ('eval-bundle','independent'):
        a=s.add_parser(name);a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--analysis',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if name=='independent':
            a.add_argument('--manifest',type=Path,required=True);a.add_argument('--device',choices=['cpu','cuda'],default='cpu');a.add_argument('--execute',action='store_true')
    a=s.add_parser('independent-statistics');a.add_argument('--evaluation',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('predict');a.add_argument('--analysis',type=Path,required=True);a.add_argument('--image',type=Path,required=True);a.add_argument('--method',required=True);a.add_argument('--seed',type=int,choices=[42,43,44],default=42);a.add_argument('--output',type=Path,required=True);a.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=s.add_parser('independent-finish');a.add_argument('--evaluation',type=Path,required=True);a.add_argument('--analysis',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--author-metadata',type=Path)
    a=s.add_parser('prepare-independent');a.add_argument('--metadata-csv',type=Path,required=True);a.add_argument('--evidence',type=Path,required=True);a.add_argument('--reviewer',required=True);a.add_argument('--reviewed-at',required=True);a.add_argument('--notes',required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--confirm-independence',action='store_true')
    a=s.add_parser('review-summary');a.add_argument('--responses',type=Path,required=True);a.add_argument('--source-key',type=Path,required=True);a.add_argument('--packet',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('browser-qa');a.add_argument('--public',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--chrome',type=Path,required=True)
    a=s.add_parser('qualitative');a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--analysis',type=Path,required=True);a.add_argument('--responses',type=Path,required=True);a.add_argument('--source-key',type=Path,required=True);a.add_argument('--packet',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--execute',action='store_true')
    for name in ('cloud-plan','cloud-verify','cloud-status','cloud-run','cloud-start','pack-scratch'):
        a=s.add_parser(name);a.add_argument('--root',type=Path,default=ROOT)
        if name in ('cloud-run','cloud-start'):a.add_argument('--execute',action='store_true');a.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.action=='audit':
        from .readiness import audit
        result=audit(a.root,a.output)
    elif a.action=='build':
        from .bundle import build
        result=build(a.root,a.output,a.review_dir)
    elif a.action=='catalog':
        from .catalog import export
        result=export(a.root,a.output,a.returned,a.scratch)
    elif a.action=='finish':result=finish(a.root,a.output,a.returned,a.scratch)
    elif a.action in ('figures','manuscript'):
        from . import figures,manuscript
        result=figures.render(a.analysis,a.output) if a.action=='figures' else manuscript.render(a.analysis,a.output,a.independent,a.statistics,a.author_metadata)
    elif a.action in ('ingest-delivery','ingest-scratch','ingest-all'):
        from .result_io import delivery,scratch,all_results
        result={'ingest-delivery':delivery,'ingest-scratch':scratch,'ingest-all':all_results}[a.action](a.archive,a.checksum,a.source_manifest,a.output)
    elif a.action=='eval-bundle':
        from .bundle import evaluation_build
        result=evaluation_build(a.root,a.analysis,a.output)
    elif a.action=='independent':
        from .inference import evaluate_suite
        result=evaluate_suite(a.root,a.analysis,a.manifest,a.output,a.device,not a.execute)
    elif a.action=='independent-statistics':
        from .group_statistics import independent_summary
        result=independent_summary(a.evaluation,a.output)
    elif a.action=='predict':
        from .inference import predict_image
        result=predict_image(a.analysis,a.image,a.method,a.seed,a.output,a.device)
    elif a.action=='independent-finish':
        from .group_statistics import independent_summary
        from .figures import independent
        output=fresh(a.output);result=dict(statistics=independent_summary(a.evaluation,output/'statistics'))
        result['figures']=independent(a.evaluation,output/'statistics',output/'figures')
        from .manuscript import render
        result['manuscript']=render(a.analysis,output/'manuscript',a.evaluation,output/'statistics',a.author_metadata)
    elif a.action=='prepare-independent':
        from .independent_prepare import prepare
        result=prepare(a.metadata_csv,a.evidence,a.reviewer,a.reviewed_at,a.notes,a.output,a.confirm_independence)
    elif a.action=='review-summary':
        from .review import summarize
        result=summarize(a.responses,a.source_key,a.packet,a.output)
    elif a.action=='browser-qa':
        from .browser_qa import run
        result=run(a.public,a.output,a.chrome)
    elif a.action=='qualitative':
        from .qualitative import build
        result=build(a.root,a.analysis,a.responses,a.source_key,a.packet,a.output,not a.execute)
    elif a.action=='pack-scratch':
        from .scratch import pack
        result=pack(a.root)
    elif a.action.startswith('cloud-'):
        from .cloud import plan,verify,execute,start,status
        if a.action=='cloud-plan' or a.action in ('cloud-run','cloud-start') and not a.execute:result=plan(a.root)
        elif a.action=='cloud-verify':result=verify(a.root)
        elif a.action=='cloud-status':
            result=status(a.root)
        else:result=(start if a.action=='cloud-start' else execute)(a.root,a.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
