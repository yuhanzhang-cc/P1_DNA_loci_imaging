"""Aggregate existing exports; absent/not-started methods never become zero scores.

Future directory contract: METHOD/seed_N/status.json + primary/metrics.json +
selection.json (frozen validation selection). Complete status required for scores.
"""
from pathlib import Path
import argparse,json,statistics
from scaffold_common import read,csv_new,save_new

FIELDS=["method","seed","role","status","epoch","threshold","tp","fp","fn","precision","recall","f1","weak_recall","mean_error_px","ranked_ap","source_kind"]
METRICS=["precision","recall","f1","weak_recall","mean_error_px"]


def trial(folder,method,seed):
    state=read(folder/"status.json") if (folder/"status.json").exists() else dict(status="not_started")
    if state["status"]!="complete":return [dict.fromkeys(FIELDS)|dict(method=method,seed=seed,role="primary",status=state["status"],source_kind="missing_not_zero")]
    if state.get("engineering_only"):raise ValueError("Engineering output cannot be aggregated as a formal result")
    selection=read(folder/"selection.json");result=[]
    for role in ("primary","native_secondary"):
        choice=selection[role];metrics=read(folder/role/"metrics.json")
        if any(choice[k]!=metrics[k] for k in ("tp","fp","fn","f1")):raise ValueError("Selection/export mismatch")
        row={k:metrics.get(k) for k in FIELDS}
        row.update(method=method,seed=seed,role=role,status="complete",epoch=choice["epoch"],threshold=choice["calibrated_probability"],
                   ranked_ap=metrics.get("ranked_ap"),source_kind="reproduced_existing_only")
        result.append(row)
    return result


def existing(root):
    b=Path(root)/"outputs/next_stage_v4_recipe_20261101"
    paths=dict(R0=b/"extracted_r0/outputs/research_v4_cuda/baseline_r0/R0_cuda_reference",
               H0=b/"extracted_h0/outputs/research_v4_cuda/baseline_h0/H0_cuda_reference",
               G0=b/"08_returned_gpu/extracted/outputs/research_v4_cuda/recipe/G0_ref_cuda")
    if any(not (p/f"seed_{seed}/status.json").is_file() or read(p/f"seed_{seed}/status.json")["status"]!="complete" for p in paths.values() for seed in (42,43,44)):
        raise ValueError("Existing-baseline example requires all nine completed exported trials")
    return [r for method,p in paths.items() for seed in (42,43,44) for r in trial(p/f"seed_{seed}",method,seed)]


def summarize(data):
    output=[]
    for method in sorted({r["method"] for r in data}):
        for role in ("primary","native_secondary"):
            group=[r for r in data if r["method"]==method and r["role"]==role and r["status"]=="complete"]
            if not group:continue
            seeds=[r["seed"] for r in group]
            if len(set(seeds))!=len(seeds):raise ValueError("Duplicate method/role/seed")
            row=dict(method=method,role=role,n_seeds=len(group),source_kind="existing_exports_no_new_training")
            for metric in METRICS:
                values=[r[metric] for r in group if r[metric] is not None]
                row[metric+"_mean"]=statistics.mean(values) if values else None
                row[metric+"_sd"]=statistics.stdev(values) if len(values)>1 else None
            output.append(row)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--output",type=Path);p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args()
    if bool(args.data_root)==bool(args.input):p.error("Choose --data-root existing example or --input future result directory")
    data=existing(args.data_root) if args.data_root else [r for method in sorted(args.input.iterdir()) if method.is_dir() for folder in sorted(method.glob("seed_*")) for r in trial(folder,method.name,int(folder.name.split("_")[-1]))]
    summary=summarize(data)
    if args.output:
        csv_new(args.output/"per_seed_metrics.csv",data,FIELDS)
        csv_new(args.output/"model_summary.csv",summary,list(summary[0]) if summary else ["method","role","n_seeds"])
        save_new(args.output/"schema.json",dict(input="METHOD/seed_N/{status.json,selection.json,primary/metrics.json}",columns=FIELDS,not_started_metrics=None,ranked_ap_not_substituted_by_ap31=True))
    print(json.dumps(dict(passed=True,dry_run=True,rows=len(data),summaries=len(summary),new_training=0,ranked_ap_derived_from_thresholded_csv=False)))


if __name__=="__main__":main()
