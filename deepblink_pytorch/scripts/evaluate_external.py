"""Evaluate a declared external manifest using frozen model/threshold locks."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"deepblink_pytorch"),str(ROOT/"deepblink_pytorch/scripts")]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--manifest",type=Path)
    p.add_argument("--locks",type=Path,default=ROOT/"method_locks.json")
    p.add_argument("--exclusions",type=Path,default=ROOT/"assets/reference_exclusions.json")
    p.add_argument("--model-root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,default=ROOT/"outputs/external_evaluation")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=p.parse_args()
    from research_v4.common import read_json,sha256
    from research_v4.external import evaluate
    manifest=ROOT/"bundle_manifest.json"
    if manifest.exists():
        for name,digest in read_json(manifest)["files"].items():
            path=ROOT/name
            if not path.resolve().is_relative_to(ROOT.resolve()) or sha256(path)!=digest:raise ValueError("Evaluation bundle changed: "+name)
    locks=read_json(args.locks);exclusions=read_json(args.exclusions)
    if args.verify_only:
        for lock in locks:
            path=args.model_root/lock["checkpoint"]
            if lock["kind"]=="spotiflow":path=path/"best.pt"
            if sha256(path)!=lock["weight_sha256"]:raise ValueError("Model lock mismatch")
        print(dict(verified_models=len(locks),new_data_required=True,training_performed=False));return
    if args.manifest is None:p.error("--manifest is required for evaluation")
    contrasts=read_json(ROOT/"contrasts.json") if (ROOT/"contrasts.json").exists() else []
    print(evaluate(args.manifest,locks,args.model_root,exclusions,args.output,device=args.device,contrasts=contrasts))


if __name__=="__main__":main()
