"""Provisional manuscript figures with source tables; no fabricated missing bars."""
from pathlib import Path
import shutil,statistics
import numpy as np
from diagnostics.handoff_v2.common import table,read,write,fresh,sha

METHODS=['LoG','DoG','R0','H0','G0','S0','P0','Spotiflow_scratch','Spotiflow_F']
COLORS=['#555555','#888888','#0072B2','#009E73','#D55E00','#CC79A7','#332288','#AA4499','#117733']
DISPLAY={'Spotiflow_scratch':'Spotiflow\nscratch','Spotiflow_F':'Spotiflow\ntransfer'}


def render(analysis,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    analysis=Path(analysis);output=fresh(output);rows=table(analysis/'per_seed_metrics.csv')
    manifest=read(analysis/'analysis_manifest.json')
    if manifest['test_evaluated'] or manifest['dataset_role']!='development_validation':raise ValueError('This figure revision expects development results')
    sources={}
    for name in ('per_seed_metrics.csv','summary.csv','ranked_pr_curves.csv','common_tp_errors.csv','planned_f1_comparisons.csv'):
        shutil.copyfile(analysis/name,output/name);sources[name]=sha(analysis/name)
    exports=[];captions=[]
    def save(fig,name,caption,alt):
        fig.suptitle('Development validation; selection-conditioned',fontsize=10)
        for ext in ('svg','pdf','png'):
            path=output/(name+'.'+ext);fig.savefig(path,dpi=300,facecolor='white')
            exports.append(dict(file=path.name,sha256=sha(path),width_mm=round(fig.get_figwidth()*25.4,3),height_mm=round(fig.get_figheight()*25.4,3),format=ext))
        captions.append(dict(figure=name,caption=caption,alt_text=alt));plt.close(fig)
    def values(method,metric):return [float(r[metric]) for r in rows if r['method']==method and r.get(metric) not in ('',None)]
    with plt.rc_context({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'font.family':'DejaVu Sans'}):
        fig,axes=plt.subplots(1,3,figsize=(7.1,2.8),layout='constrained')
        for ax,method in zip(axes,('H0','S0','P0')):
            ax.set(xlim=(-.5,5.5),ylim=(5.5,-.5),aspect='equal',xticks=range(6),yticks=range(6),xlabel='x pixel index',ylabel='y pixel index',title=method)
            ax.grid(alpha=.2);ax.plot(2.6,2.25,'k*',ms=10,label='reference')
            sources=[(y,x) for y in range(0,5) for x in range(0,5)] if method=='P0' else [(2,2)]
            for y,x in sources:
                ax.plot(x,y,'s',color='#0072B2',ms=3)
                ax.annotate('',xy=(2.6,2.25),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#0072B2',lw=.5,alpha=.6))
            ax.plot(3,3,'o',mfc='none',mec='#D55E00',ms=9)
            ax.text(.03,.98,'bounded offsets' if method=='H0' else 'signed; single anchor' if method=='S0' else 'signed; support radius 2',va='top',transform=ax.transAxes,fontsize=7)
        save(fig,'fig01_supervision_schematic','Conceptual pixel-coordinate example, not measured data. Star: reference at (y,x)=(2.25,2.60); blue squares: supervised source pixels; arrows: truth-minus-source targets. Orange ring: illustrative off-anchor peak. H0 bounds offsets, S0 changes their signed range at the floor anchor, and P0 extends supervision. The displayed radius 2 is the proposed example, not an approved or optimal value. Multiple-truth ownership is described in Methods.','Three pixel grids illustrate the source-pixel/target relationship and the distinction between changing output range and extending supervision.')
        fig,axes=plt.subplots(2,2,figsize=(7.1,5.6),layout='constrained')
        for ax,metric,label in zip(axes.ravel(),('f1','ranked_ap','weak_recall','mean_error_px'),('Micro-F1','Full ranked AP','Weak-reference recall','Matched-TP mean error (px)')):
            for i,(m,color) in enumerate(zip(METHODS,COLORS)):
                v=values(m,metric)
                if not v:
                    ax.text(i,.02,'pending',rotation=90,ha='center',va='bottom',color='#666666',transform=ax.get_xaxis_transform(),fontsize=7);continue
                ax.bar(i,statistics.mean(v),yerr=statistics.stdev(v) if len(v)>1 else None,color=color,edgecolor='black',linewidth=.5,capsize=2,hatch='//' if m=='Spotiflow_F' else '')
                ax.plot(i+np.linspace(-.12,.12,len(v)),v,'ko',ms=2)
            ax.set(xticks=range(len(METHODS)),xticklabels=[DISPLAY.get(m,m) for m in METHODS],ylabel=label,ylim=(0,None if metric=='mean_error_px' else 1))
            ax.tick_params(axis='x',labelrotation=45)
        save(fig,'fig02_method_comparison','Fixed-threshold metrics and complete ranked AP on the same 22 development images. Bars show means and sample SD across available training seeds, with individual seeds shown. LoG/DoG each have one deterministic calibrated configuration; no seed SD is assigned. Missing trials are pending. Localization uses each method\'s own matched TPs. Spotiflow transfer uses external pretraining and a different native recipe (hatched); these bars compare complete methods and do not isolate architecture.','Four bar panels compare detection, ranking, weak-point recall and localization; S0/P0 and scratch remain pending until returned.')
        fig,axes=plt.subplots(1,3,figsize=(7.1,2.8),layout='constrained')
        for ax,metric,label in zip(axes,('f1','weak_recall','mean_error_px'),('Micro-F1','Weak-reference recall','Matched-TP error (px)')):
            for seed,style in zip((42,43,44),('o-','s--','^:')):
                series=[]
                for m in ('H0','S0','P0'):
                    rr=next((r for r in rows if r['method']==m and r['seed']==str(seed)),None)
                    series.append(float(rr[metric]) if rr and rr.get(metric) not in ('',None) else np.nan)
                ax.plot(range(3),series,style,label=f'seed {seed}',ms=3)
            ax.set(xticks=range(3),xticklabels=['H0','S0','P0'],ylabel=label,ylim=(0,None if metric=='mean_error_px' else 1))
            ax.legend(frameon=False,fontsize=6)
        save(fig,'fig03_sequential_ablation','Sequential parameter-identical comparisons: H0 has bounded single-anchor offsets; S0 uses signed single-anchor offsets; P0 adds neighborhood supervision with nearest-truth ownership and per-truth weighting. P0−S0 identifies the joint support/ownership/weighting change, not its individual factors. Missing points interrupt lines. This is a constrained sequential ablation, not a complete factorial design.','Paired seed trajectories across H0, S0 and P0; only available measurements appear.')
        curves=table(analysis/'ranked_pr_curves.csv');fig,ax=plt.subplots(figsize=(4.6,3.8),layout='constrained')
        for m,color in zip(METHODS,COLORS):
            for j,seed in enumerate(sorted({r['seed'] for r in curves if r['method']==m})):
                c=[r for r in curves if r['method']==m and r['seed']==seed]
                # Add origin only; never extrapolate unreached recall to one.
                ax.step([0]+[float(r['recall']) for r in c],[1]+[float(r['precision']) for r in c],where='post',color=color,linewidth=.8,alpha=.7,label=m if j==0 else None)
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision');ax.legend(fontsize=6,frameon=False,ncol=2)
        save(fig,'fig04_ranked_pr','Complete score-ranked precision–recall curves, grouped at tied scores, using confidence-ordered greedy nearest-unmatched assignment within 3 px. AP is a non-interpolated recall-weighted precision sum; thresholded F1 uses the separate frozen maximum-cardinality policy. Each curve is one selected seed/configuration.','All available seed precision–recall curves are displayed; curves end at attained recall.')
        errors=table(analysis/'common_tp_errors.csv');fig,ax=plt.subplots(figsize=(5.4,3.4),layout='constrained')
        pairs=[('S0','H0'),('P0','S0'),('P0','G0')]
        for i,(a,b) in enumerate(pairs):
            for seed,marker in zip((42,43,44),('o','s','^')):
                v=[float(r['error_delta_px']) for r in errors if r['candidate']==a and r['reference']==b and r['seed']==str(seed)]
                if v:ax.plot(i+(seed-43)*.12,statistics.mean(v),marker,color=COLORS[i+2],ms=5)
            if not any(r['candidate']==a and r['reference']==b for r in errors):ax.text(i,.5,'pending',ha='center',transform=ax.get_xaxis_transform())
        ax.axhline(0,color='black',lw=.7);ax.set(xticks=range(3),xticklabels=[a+' − '+b for a,b in pairs],ylabel='Common-TP error difference (px)')
        save(fig,'fig05_common_tp_localization','Mean candidate-minus-reference localization error on the intersection of matched truth identities within each seed. Negative values indicate lower error. Each marker is a seed; matched spots and seeds are not independent biological replicates. Missing contrasts remain pending.','Seed-level localization differences for common true positives; negative is better.')
    for p in output.glob('*.png'):
        with Image.open(p) as im:
            im.load()
            if min(im.size)<500 or im.info.get('dpi',(0,))[0]<299:raise ValueError('Raster export failed')
    write(output/'figure_manifest.json',dict(exports=exports,input_sha256=sources,matplotlib=matplotlib.__version__,
        figure_source_sha256=sha(__file__),publisher='TODO: target journal not selected',journal_compliance_claim=False,
        editable_svg_text=True,png_dpi=300,biological_inference=False))
    write(output/'captions_and_alt_text.json',captions)
    (output/'CAPTIONS.md').write_text('\n\n'.join('**'+c['figure']+'** '+c['caption']+'\n\nAlt text: '+c['alt_text'] for c in captions)+'\n',encoding='utf-8')
    return dict(figures=len(captions),exports=len(exports),output=str(output),provisional_publisher_style=True)


def independent(evaluation,statistics_folder,output,allow_engineering=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evaluation=Path(evaluation);statistics_folder=Path(statistics_folder)
    if read(evaluation/'complete.json').get('test_evaluated') is not True:raise ValueError('No completed independent readout')
    if read(evaluation/'complete.json').get('engineering_only') is not False and not allow_engineering:raise ValueError('Engineering figure is not independent scientific evidence')
    rows=table(evaluation/'per_seed_metrics.csv');contrasts=table(statistics_folder/'independent_group_intervals.csv');output=fresh(output)
    for name,p in [('per_seed_metrics.csv',evaluation/'per_seed_metrics.csv'),('independent_group_intervals.csv',statistics_folder/'independent_group_intervals.csv')]:shutil.copyfile(p,output/name)
    methods=[m for m in METHODS if any(r['method']==m for r in rows)]
    with plt.rc_context({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.7),layout='constrained')
        for i,m in enumerate(methods):
            v=[float(r['f1']) for r in rows if r['method']==m]
            axes[0].bar(i,statistics.mean(v),yerr=statistics.stdev(v) if len(v)>1 else None,color=COLORS[METHODS.index(m)],capsize=2)
            axes[0].plot(i+np.linspace(-.1,.1,len(v)),v,'ko',ms=2)
        axes[0].set(xticks=range(len(methods)),xticklabels=[DISPLAY.get(m,m) for m in methods],ylim=(0,1),ylabel='Independent micro-F1; mean ± seed SD');axes[0].tick_params(axis='x',rotation=45)
        complete=[r for r in contrasts if r['status']=='complete_conditional_estimation']
        for i,r in enumerate(complete):
            x=float(r['mean_f1_delta']);low=float(r['ci_low']);high=float(r['ci_high'])
            axes[1].plot([low,high],[i,i],'-',color='#0072B2');axes[1].plot(x,i,'ko',ms=4)
        axes[1].axvline(0,color='black',lw=.7);axes[1].set(yticks=range(len(complete)),yticklabels=[r['comparison'] for r in complete],xlabel='F1 difference; 95% biological-group interval')
        for ext in ('svg','pdf','png'):fig.savefig(output/('fig06_independent_readout.'+ext),dpi=300,facecolor='white')
        plt.close(fig)
    write(output/'figure_manifest.json',dict(test_evaluated=True,submission_style_provisional=True,source_sha256=sha(__file__),
        inputs={p.name:sha(p) for p in output.glob('*.csv')},stars_generated=False,
        caption='Frozen-model independent readout. Left: available seed means and sample SD. Right: paired biological-group percentile bootstrap intervals conditional on fixed selected models; intervals do not constitute significance tests. No test recalibration.'))
    return dict(output=str(output),figures=1,exports=3)
