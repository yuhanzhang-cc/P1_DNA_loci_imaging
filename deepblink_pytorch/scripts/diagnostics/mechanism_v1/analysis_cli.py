"""One-line read-only readiness/replay commands, with explicit roots."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"deepblink_pytorch"));sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts"))
sys.path.insert(0,str(ROOT/"deepblink_pytorch/scripts/diagnostics/scaffolds"))
if len(sys.argv)<2 or sys.argv[1] not in ("check","replay"):
    raise SystemExit("Usage: analysis_cli.py {check|replay} --data-root ROOT --output FRESH_OUTPUT")
task=sys.argv.pop(1)
if task=="check":
    from diagnostics.mechanism_v1.check_readiness import main
else:
    from diagnostics.mechanism_v1.validate_existing import main
if __name__=="__main__":main()
