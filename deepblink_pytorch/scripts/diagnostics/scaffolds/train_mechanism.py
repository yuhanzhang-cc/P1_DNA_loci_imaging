"""Configuration-only S0/P0 entry. This phase has no executable train adapter."""
from pathlib import Path
import argparse
import json
from scaffold_common import read,validate_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--seed",type=int,choices=[42,43,44],default=42)
    p.add_argument("--dry-run",action="store_true",required=True)
    args=p.parse_args();result=validate_config(read(args.config));result.update(seed=args.seed,config=str(args.config))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
