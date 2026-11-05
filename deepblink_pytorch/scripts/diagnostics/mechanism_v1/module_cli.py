"""Portable module dispatcher for packaging and future auxiliary commands."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"));sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts"))
from diagnostics.mechanism_v1.build_bundle import main
if __name__=="__main__":main()
