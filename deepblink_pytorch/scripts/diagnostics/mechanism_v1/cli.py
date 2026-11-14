"""Isolated experiment CLI. No training occurs on import."""
import os,sys
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"))
sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts"))
from diagnostics.mechanism_v1.runner import main

if __name__=="__main__":main()
