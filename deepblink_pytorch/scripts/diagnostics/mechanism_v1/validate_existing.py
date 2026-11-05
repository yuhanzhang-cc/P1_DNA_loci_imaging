"""No-training validation replay. Full candidate ranking is a new metric only.

Compare thresholded counts with the stored CUDA exports; report CPU/backend
differences, never overwrite the historical files or fit thresholds on replay.
"""
from pathlib import Path
import argparse,json
from .contracts import ROOT,REL,read,write_new,sha
from .runner import corpus
from .evaluate import evaluate
from research_v4.engine import configure_device
from research_v4.common import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--locks",type=Path,default=ROOT/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/configs/validation_model_locks.json")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);configure_device("cpu");data=corpus(a.data_root)
    records=[]
    for lock in read(a.locks):
        folder=a.output/lock["method"]/f"seed_{lock['seed']}"
        result=evaluate(lock,a.data_root,data.splits["valid"],folder,manifest_sha256=sha(data.paths["development"]))
        original=read((a.data_root/lock["checkpoint"]).parent/"primary/metrics.json")
        counts_equal=all(result[k]==original[k] for k in ("tp","fp","fn"))
        row=dict(method=lock["method"],seed=lock["seed"],counts_equal=counts_equal,
            original_f1=original["f1"],replayed_f1=result["f1"],f1_delta=result["f1"]-original["f1"],
            ranked_ap=result["ranked_ap"],weak_recall=result["weak_recall"],test_evaluated=False)
        records.append(row);print(json.dumps(row),flush=True)
    write_csv(a.output/"replay_summary.csv",records)
    write_new(a.output/"checks.json",dict(passed=all(r["counts_equal"] for r in records),models=len(records),
        all_thresholded_counts_equal=all(r["counts_equal"] for r in records),full_ranked_ap=True,
        threshold_refitted=False,formal_training_started=False,test_evaluated=False,real_optimizer_updates=0))
    if not all(r["counts_equal"] for r in records):raise AssertionError("CPU/GPU count differences require review; do not overwrite baseline")

if __name__=="__main__":main()
