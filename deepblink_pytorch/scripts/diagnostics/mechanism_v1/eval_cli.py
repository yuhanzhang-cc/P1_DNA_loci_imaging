"""Portable entry for frozen validation replay or reviewed independent evaluation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"));sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts"))
# Existing standalone scaffolds use sibling imports.
sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts/diagnostics/scaffolds"))
from diagnostics.mechanism_v1.evaluate import main
if __name__=="__main__":main()
