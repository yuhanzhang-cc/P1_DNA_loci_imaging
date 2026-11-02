"""Analyze returned GPU queue results without opening a training/test archive."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"deepblink_pytorch"),str(ROOT/"deepblink_pytorch/scripts")]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root",type=Path,required=True,help="One completed queue directory, containing protocol.json")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--external",action="store_true",help="Read frozen-model external results, resampling biological groups")
    args=p.parse_args()
    from research_v4.common import read_json,rows
    from research_v4.analysis import summarize
    protocol=read_json(args.results_root/"protocol.json")
    if args.external:
        if protocol.get("engineering_only"):raise ValueError("Synthetic engineering output cannot become a scientific report")
        if not read_json(args.results_root/"checks.json").get("passed"):raise ValueError("External evaluation incomplete")
        print(summarize(rows(args.results_root/"per_seed_summary.csv"),rows(args.results_root/"all_per_image_metrics.csv"),
                        args.output,protocol["planned_contrasts"],group_column="biological_group_id",validation_only=False))
        return
    from cloud_recipe_train import check_complete
    check_complete(args.results_root.parent,protocol["queue"])
    seed_rows=[];image_rows=[]
    for item in protocol["tasks"]:
        folder=args.results_root/item["task_id"]/f"seed_{item['seed']}"
        if read_json(folder/"status.json")["status"]!="complete":raise ValueError("Refuse partial or best-seed-only analysis")
        metrics=read_json(folder/"primary/metrics.json")
        seed_rows.append(dict(method=item["task_id"],seed=item["seed"],**metrics))
        image_rows.extend(dict(method=item["task_id"],seed=item["seed"],**r) for r in rows(folder/"primary/per_image_metrics.csv"))
    contrasts=[("G0_aug_cuda","G0_ref_cuda")] if protocol["queue"]=="recipe" else []
    print(summarize(seed_rows,image_rows,args.output,contrasts,validation_only=True))


if __name__=="__main__":main()
