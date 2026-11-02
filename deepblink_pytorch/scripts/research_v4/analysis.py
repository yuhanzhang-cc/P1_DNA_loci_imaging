"""Predeclared pair summaries for returned queues or fixed-model external results."""
import itertools
from pathlib import Path
import numpy as np
from .common import dump,write_csv


def micro(counts):
    tp,fp,fn=np.moveaxis(np.asarray(counts,dtype=float),-1,0)
    return np.divide(2*tp,2*tp+fp+fn,out=np.ones_like(tp),where=2*tp+fp+fn>0)


def summarize(seed_rows,image_rows,output,contrasts,*,group_column="sample_id",validation_only=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=[]
    for method in dict.fromkeys(r["method"] for r in seed_rows):
        selected=[r for r in seed_rows if r["method"]==method]
        row=dict(method=method,n_seeds=len(selected))
        for key in ("f1","precision","recall","tp","fp","fn","weak_recall","macro_f1","count_mae","mean_error_px"):
            values=np.asarray([float(r[key]) for r in selected if r.get(key) not in (None,"")],float)
            row[key+"_mean"]=float(values.mean()) if len(values) else None
            row[key+"_sd"]=float(values.std(ddof=1)) if len(values)>1 else None
        summary.append(row)
    comparisons=[]
    for a,b in contrasts:
        ar={int(r["seed"]):r for r in seed_rows if r["method"]==a};br={int(r["seed"]):r for r in seed_rows if r["method"]==b}
        if set(ar)!=set(br) or len(ar)!=3:raise ValueError("Primary comparison needs the same three registered seed identifiers")
        for seed in ar:
            aa=[r["sample_id"] for r in image_rows if r["method"]==a and int(r["seed"])==seed]
            bb=[r["sample_id"] for r in image_rows if r["method"]==b and int(r["seed"])==seed]
            if not aa or len(aa)!=len(set(aa)) or sorted(aa)!=sorted(bb):raise ValueError("Paired image identities differ or are duplicated")
        if len(ar)!=len([r for r in seed_rows if r["method"]==a]) or len(br)!=len([r for r in seed_rows if r["method"]==b]):raise ValueError("Duplicate method/seed rows")
        seeds=sorted(ar);d=np.asarray([float(ar[s]["f1"])-float(br[s]["f1"]) for s in seeds])
        null=np.asarray([np.mean(d*sign) for sign in itertools.product((-1,1),repeat=3)])
        row=dict(comparison=a+"-"+b,n_pairs=3,mean_delta_f1=float(d.mean()),sample_sd_delta=float(d.std(ddof=1)),
                 positive_seeds=int((d>0).sum()),p_exact=float(np.mean(np.abs(null)>=abs(d.mean())-1e-12)),p_holm=0.,
                 bootstrap_low=None,bootstrap_high=None,resampling_unit=group_column,validation_only=validation_only)
        groups=sorted({r[group_column] for r in image_rows if r["method"]==a})
        def grouped(method):
            values=[]
            for seed in seeds:
                data=[]
                for group in groups:
                    matched=[r for r in image_rows if r["method"]==method and int(r["seed"])==seed and r[group_column]==group]
                    if not matched:raise ValueError("Incomplete matched resampling groups")
                    data.append([sum(float(r[k]) for r in matched) for k in ("tp","fp","fn")])
                values.append(data)
            return np.asarray(values)
        if len(groups)>=2:
            rng=np.random.default_rng(20261025);draws=rng.integers(0,len(groups),size=(10000,len(groups)))
            aa,bb=grouped(a),grouped(b)
            delta=(micro(aa[:,draws].sum(2))-micro(bb[:,draws].sum(2))).mean(0)
            row.update(bootstrap_low=float(np.quantile(delta,.025)),bootstrap_high=float(np.quantile(delta,.975)))
        comparisons.append(row)
    order=np.argsort([r["p_exact"] for r in comparisons]);previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,(len(order)-rank)*comparisons[index]["p_exact"])
        comparisons[index]["p_holm"]=min(previous,1.)
    write_csv(output/"model_summary.csv",summary)
    write_csv(output/"paired_comparisons.csv",comparisons,["comparison","n_pairs","mean_delta_f1","sample_sd_delta","positive_seeds","p_exact","p_holm","bootstrap_low","bootstrap_high","resampling_unit","validation_only"])
    dump(output/"analysis_provenance.json",dict(contrasts=contrasts,group_column=group_column,bootstrap_draws=10000,bootstrap_seed=20261025,
         estimator="mean of per-training-seed micro-F1",uncertainty="sample SD across seeds plus fixed-model group bootstrap",validation_only=validation_only,
         selection_adjusted=False,interpretation="Development intervals are conditional on selected models; group metadata do not establish biological independence"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"svg.fonttype":"none","font.size":10}):
        fig,ax=plt.subplots(figsize=(max(6,len(summary)*2),4.8),layout="constrained")
        for i,r in enumerate(summary):
            ax.bar(i,r["f1_mean"],yerr=r["f1_sd"],capsize=4,color="\x230072B2",edgecolor="black")
            values=[float(t["f1"]) for t in seed_rows if t["method"]==r["method"]]
            ax.plot(i+np.linspace(-.1,.1,len(values)),values,"ko",ms=4)
        ax.set(xticks=range(len(summary)),xticklabels=[r["method"] for r in summary],ylim=(0,1),ylabel="Micro-F1",title="Frozen model comparison: mean ± sample SD")
        ax.tick_params(axis="x",labelrotation=15)
        fig.savefig(output/"model_f1_bars.png",dpi=240);fig.savefig(output/"model_f1_bars.svg");plt.close(fig)
        if contrasts:
            fig,axes=plt.subplots(1,len(contrasts),figsize=(5*len(contrasts),4.8),squeeze=False,layout="constrained")
            for ax,(a,b),c in zip(axes.ravel(),contrasts,comparisons):
                for seed in sorted({int(r["seed"]) for r in seed_rows if r["method"]==a}):
                    y=[next(float(r["f1"]) for r in seed_rows if r["method"]==m and int(r["seed"])==seed) for m in (b,a)]
                    ax.plot([0,1],y,"o-",label=f"seed {seed}")
                ax.set(xticks=[0,1],xticklabels=[b,a],ylabel="Micro-F1",title=f"Δ={100*c['mean_delta_f1']:+.2f} pp; Holm p={c['p_holm']:.2f}")
                ax.legend(frameon=False)
            fig.savefig(output/"paired_seed_lines.png",dpi=240);fig.savefig(output/"paired_seed_lines.svg");plt.close(fig)
    for p in output.glob("*.svg"):p.write_text("\n".join(l.rstrip() for l in p.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
    return dict(summary=summary,comparisons=comparisons)
