"""Automatic descriptive development report; never invent missing experiments."""
from pathlib import Path
import statistics
from .common import read,table,write,csv_write,fresh,sha,ROOT,OLD
from .archive import verify_return

METRICS=["precision","recall","f1","weak_recall","mean_error_px","ranked_ap"]


def summarize(rows):
    output=[]
    for method in sorted({r["method"] for r in rows}):
        group=[r for r in rows if r["method"]==method]
        for metric in METRICS:
            values=[float(r[metric]) for r in group if r.get(metric) not in (None,"")]
            output.append(dict(method=method,metric=metric,n_seeds=len(values),mean=statistics.mean(values) if values else None,
                sd=statistics.stdev(values) if len(values)>1 else None,status="three_seed_development" if len(values)==3 else "partial_or_missing",
                test_evaluated=False))
    return output


def baseline_rows(data_root):
    root=Path(data_root);base=root/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102"
    source=base/"per_seed_metrics.csv";ap_path=root/OLD/"evidence/validation_replay_summary.csv"
    ap={(r["method"],int(r["seed"])):r for r in table(ap_path)};rows=[]
    for r in table(source):
        if r["role"]!="primary":continue
        method=r["method"].split("_")[0]
        row={k:float(r[k]) if r.get(k) not in (None,"") else None for k in METRICS if k!="ranked_ap"}
        row.update(method=method,seed=int(r["seed"]),ranked_ap=float(ap[(method,int(r["seed"]))]["ranked_ap"]),
            source="existing_matched_CUDA_reference",test_evaluated=False)
        rows.append(row)
    if len(rows)!=9:raise ValueError("Nine matched baseline exports required")
    return rows,{str(source):sha(source),str(ap_path):sha(ap_path)}


def render(data_root,output,returned=None,allow_engineering=False):
    rows,inputs=baseline_rows(data_root);verification=None;error_rows=[]
    if returned:
        returned=Path(returned);verification=verify_return(returned,allow_engineering)
        manifest=read(returned/"return_manifest.json");inputs[str(returned/"return_manifest.json")]=sha(returned/"return_manifest.json")
        for task in manifest["tasks"]:
            metrics=read(returned/task["folder"]/"primary/metrics.json")
            rows.append(dict(method=task["method"],seed=task["seed"],source="SYNTHETIC_ENGINEERING_FIXTURE" if manifest["engineering_only"] else "new_fixed_protocol_development",
                test_evaluated=False,**{k:metrics.get(k) for k in METRICS}))
        if not manifest["engineering_only"]:
            # Owner-side validation truth only; no test arrays are accessed.
            from diagnostics.peak_offset_diag import load_validation
            from .gate import trial_points,near_truth_fp
            validation,_=load_validation(Path(data_root)/"outputs/next_stage_v4_recipe_20261101/04_bundles/ce2d1821ceec/p1_v4_training")
            truth={sid:item["truth"] for sid,item in validation.items()}
            for task in manifest["tasks"]:
                new=returned/task["folder"];seed=task["seed"]
                g0=Path(data_root)/f"outputs/next_stage_v4_recipe_20261101/08_returned_gpu/extracted/outputs/research_v4_cuda/recipe/G0_ref_cuda/seed_{seed}"
                pairs=[("G0",g0,task["method"],new)]
                if task["method"]=="P0" and any(t["method"]=="S0" and t["seed"]==seed for t in manifest["tasks"]):pairs.append(("S0",returned/f"outputs/trials/S0/seed_{seed}","P0",new))
                for left,lf,right,rf in pairs:
                    for folder in (lf,rf):
                        ids={r["sample_id"] for r in table(folder/"primary/per_image_metrics.csv")}
                        if ids!=set(truth):raise ValueError("Error review uses a different validation image set")
                    lp,lm,le=trial_points(lf);rp,rm,re=trial_points(rf);common=sorted(set(le)&set(re))
                    error_rows.append(dict(reference=left,candidate=right,seed=seed,common_tp=len(common),
                        reference_common_error_px=statistics.mean([le[k] for k in common]) if common else None,
                        candidate_common_error_px=statistics.mean([re[k] for k in common]) if common else None,
                        error_improvement_px=statistics.mean([le[k]-re[k] for k in common]) if common else None,
                        reference_near_truth_fp=near_truth_fp(lp,lm,truth),candidate_near_truth_fp=near_truth_fp(rp,rm,truth),
                        biological_fp_classification="pending_human_review; geometry proxy only"))
    output=fresh(output);summary=summarize(rows)
    csv_write(output/"per_seed_metrics.csv",rows,["method","seed",*METRICS,"source","test_evaluated"])
    csv_write(output/"model_summary.csv",summary,["method","metric","n_seeds","mean","sd","status","test_evaluated"])
    comparisons=[]
    for left,right in (("S0","H0"),("P0","S0"),("P0","G0")):
        a={r["seed"]:r for r in rows if r["method"]==left};b={r["seed"]:r for r in rows if r["method"]==right}
        seeds=sorted(set(a)&set(b))
        for metric in METRICS:
            differences=[a[s][metric]-b[s][metric] for s in seeds if a[s].get(metric) is not None and b[s].get(metric) is not None]
            comparisons.append(dict(comparison=left+" - "+right,metric=metric,n_paired_seeds=len(differences),
                mean_delta=statistics.mean(differences) if differences else None,sd_delta=statistics.stdev(differences) if len(differences)>1 else None,
                interpretation="development_only; seeds are training variability, not biological replicates",p_value=None,significance_claim=False))
    csv_write(output/"planned_comparisons.csv",comparisons,list(comparisons[0]))
    csv_write(output/"paired_error_review.csv",error_rows,["reference","candidate","seed","common_tp","reference_common_error_px","candidate_common_error_px","error_improvement_px","reference_near_truth_fp","candidate_near_truth_fp","biological_fp_classification"])
    pending=[dict(method=m,seed=s,status="not_started_or_not_returned") for m in ("S0","P0") for s in (42,43,44) if not any(r["method"]==m and r["seed"]==s for r in rows)]
    write(output/"pending_tasks.json",pending)
    lines=["# Automatic development report","","Engineering fixture outputs are not scientific results." if verification and verification["engineering_only"] else "No independent test was evaluated. Existing conclusions remain unchanged until prospective results are returned.","",
        "| Method | Seeds available | F1 mean ± sample SD |","|---|---:|---:|"]
    for m in ("R0","H0","G0","S0","P0"):
        r=next((r for r in summary if r["method"]==m and r["metric"]=="f1"),None)
        value="pending" if r is None else f"{r['mean']:.6f}"+(f" ± {r['sd']:.6f}" if r['sd'] is not None else " (one seed; no SD)")
        lines.append(f"| {m} | {0 if r is None else r['n_seeds']} | {value} |")
    lines += ["","P/R, weak Recall, localization error and full ranked AP are in model_summary.csv. AP31 is never substituted for full ranked AP.",
        "Planned differences are reported even when negative or missing. Missing values are empty, never zero.",
        "No significance stars are generated from three seeds. This report is descriptive and does not establish independence, statistical superiority or architecture novelty.",
        "Spotiflow's prior result has a different training recipe and is not silently inserted into the matched-protocol causal contrast.",
        "Current practical candidate remains G0. A new release recommendation requires the preregistered mechanism gate, error review and subsequent independent validation."]
    (output/"REPORT.md").write_bytes(("\n".join(lines)+"\n").encode())
    write(output/"provenance.json",dict(inputs=inputs,return_verification=verification,formal_training_performed=False,
        no_test_arrays_read=True,descriptive_only=True,missing_tasks=len(pending),source_artifacts_modified=False))
    return dict(path=str(output),rows=len(rows),pending=len(pending),independent_test_evaluated=False)
