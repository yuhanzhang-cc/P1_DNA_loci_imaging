"""Export only frozen clean220 train/validation, in Spotiflow native format."""
import argparse
from pathlib import Path

from deepblink_torch.spotiflow_bridge import export_development

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/closing_preparation/spotiflow_data")
    args = parser.parse_args()
    rows = export_development(ROOT / "outputs/261024_clean220_random/dna_clean220_random.npz",
                              ROOT / "outputs/261024_clean220_random/frozen_split.csv", args.output)
    print(f"Verified {len(rows)} development pairs; test arrays not loaded.")


if __name__ == "__main__":
    main()
