"""Unified preparation entry. Defaults never run formal training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"));sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts"))
sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts/diagnostics/scaffolds"))
from diagnostics.handoff_v2.main import main
if __name__=="__main__":main()
