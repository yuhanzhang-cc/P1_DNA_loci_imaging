from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'deepblink_pytorch'),str(ROOT/'deepblink_pytorch/scripts'),str(ROOT/'deepblink_pytorch/scripts/diagnostics/scaffolds')]
from diagnostics.completion_v1.main import main
if __name__=='__main__':main()
