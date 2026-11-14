"""Read-only overlay status. Does not expand the sealed training queues."""
from pathlib import Path
import argparse
from scaffold_common import read,base


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True)
    args=p.parse_args();registry=read(base(args.data_root)/"configs/registry.json")
    for row in registry["tasks"]:
        alias=f" alias_of={row['alias_of']} (no additional run)" if row["alias_of"] else ""
        print(f"{row['method']} seed {row['seed']}: {row['status']}; epoch 0/100; updates 0; scaffold only{alias}")
    print(f"Verified completed training count: {registry['formal_completed_training_count']}; scaffold new training: 0")


if __name__=="__main__":main()
