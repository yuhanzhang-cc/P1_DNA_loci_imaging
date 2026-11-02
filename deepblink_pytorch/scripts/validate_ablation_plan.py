"""Validate the locked core-ablation contract without scheduling training."""
from pathlib import Path
import argparse
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parent))
from research_v4.ablation import validate

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan",type=Path)
    args=parser.parse_args()
    print(validate(yaml.safe_load(args.plan.read_text(encoding="utf-8"))))
