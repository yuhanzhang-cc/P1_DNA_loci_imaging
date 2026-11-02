"""Night-L engineering; formal GPU/data-collection work is packaged, not launched."""
from pathlib import Path
import argparse
import json
import os
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"))
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["audit","geometry","preflight","review","classical","bundle","verify","scaffold","report","local"],default="local")
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--spec",type=Path,default=ROOT/"docs/next_stage_plan_20261025/overnight_protocol_v4_proposed_20261101.yaml")
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    from research_v4.common import Context,input_audit,dump,verify_protected,read_json,user_time
    from filelock import FileLock
    output=args.output_root or args.data_root/"outputs/next_stage_v4_recipe_20261101"
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/".local_runner.lock"),timeout=0):
        ctx=Context(ROOT,args.data_root,output,args.spec)
        try:
            input_audit(ctx)
            if args.stage in ("geometry",):
                from research_v4.preflight import geometry
                geometry(ctx)
            if args.stage in ("local","preflight"):
                from research_v4.preflight import run
                run(ctx)
            if args.stage in ("local","review"):
                from research_v4.review import run
                run(ctx)
            if args.stage in ("local","classical"):
                from research_v4.classical import run
                run(ctx)
            if args.stage in ("local","scaffold"):
                from research_v4.scaffold import run
                run(ctx)
            if args.stage in ("local","bundle"):
                from research_v4.packaging import build
                build(ctx)
            if args.stage in ("local","verify"):
                from research_v4.verification import run
                run(ctx)
            if args.stage in ("local","report"):
                from research_v4.reporting import report
                report(ctx)
            verify_protected(ctx.data_root,read_json(output/"00_protocol/protected_hashes.json"))
            final=read_json(output/"07_report/completion_checks.json")["status"] if args.stage in ("local","report") else "stage_finished"
            ctx.status(stage=final,requested_stage=args.stage,finished_user_time=user_time(ctx.data_root))
        except BaseException as error:
            (output/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
            ctx.status(stage="failed",error=repr(error))
            raise
        finally:
            ctx.close()


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    main()
