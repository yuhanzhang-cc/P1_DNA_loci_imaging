"""Clean-checkout handoff: mount existing returns, reproduce tables without training."""
from pathlib import Path
import argparse,json,subprocess,sys
from scaffold_common import base


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output",type=Path)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--dry-run",action="store_true");mode.add_argument("--run-readonly",action="store_true")
    args=p.parse_args();root=args.data_root.resolve()
    required=[root/"outputs/next_stage_v4_recipe_20261101"/r for r in ("extracted_r0","extracted_h0","08_returned_gpu/extracted")]
    steps=["Checkout a revision containing these scaffold files (current working tree is not yet published)",
           "Use Python >=3.10; table reproduction is standard-library-only",
           "Download/verify/extract R0,H0 and six-run G0 archives into a separate results root; no raw training images required",
           "Supply --data-root pointing at that results root; never overwrite a protected extracted directory",
           "Read frozen primary and native-secondary exports; reproduce per-seed and mean/sample-SD tables",
           "Keep test_evaluated=false and formal training count=41"]
    missing=[str(x) for x in required if not x.is_dir()]
    if args.run_readonly:
        if not args.output:p.error("--run-readonly requires a fresh --output directory")
        if missing:raise ValueError("Missing private result artifacts: "+str(missing))
        subprocess.run([sys.executable,str(Path(__file__).with_name("aggregate_results.py")),"--data-root",str(root),"--output",str(args.output),"--dry-run"],check=True)
    print(json.dumps(dict(passed=not missing,dry_run=args.dry_run,steps=steps,missing_artifact_mounts=missing,
                         artifact_data_not_in_git=True,new_training=0,clean_checkout_executed=False),indent=2))


if __name__=="__main__":main()
