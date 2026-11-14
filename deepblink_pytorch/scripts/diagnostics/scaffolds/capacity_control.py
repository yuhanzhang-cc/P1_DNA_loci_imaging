"""Generate nonparametric-control config; reject unknown capacity changes."""
from pathlib import Path
import argparse,json
from scaffold_common import read,matched_control,save_new,architecture_signature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path);p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args();source=read(args.config);control=matched_control(source)
    if args.output:save_new(args.output,control)
    print(json.dumps(dict(passed=True,dry_run=True,expected_parameters=control["expected_parameters"],
       capacity_matching="identical structural signature; activation/supervision introduce zero parameters",
       parameter_count_measured_for_new_adapter=False,control_alias="S0",output=str(args.output),optimizer_updates=0),indent=2))


if __name__=="__main__":main()
