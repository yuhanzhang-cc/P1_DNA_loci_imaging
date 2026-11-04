"""Explicit no-training overnight runner: reproducibility, packet QA and stop.

--validate-ready checks the prepared packet. --execute creates an independent
timestamped replay, compares its point table, rebuilds the packet and validates it.
It never starts a train runner, changes architecture, installs software or opens
the reviewer key on the user's behalf. Human judgments remain pending.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from peak_offset_diag import REL_OUTPUT
from research_v4.common import dump,sha256


def run(script,arguments,log):
    with log.open("ab",buffering=0) as stream:
        subprocess.run([sys.executable,"-u",str(HERE/script),*map(str,arguments)],stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    from filelock import FileLock
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-ready",action="store_true")
    mode.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    root=args.data_root.resolve();base=(root/REL_OUTPUT).resolve()
    if not (base/"overnight_ready/readiness_provenance.json").exists():raise ValueError("Prepare the fixed packet first")
    with FileLock(str(base/".overnight_no_train.lock"),timeout=0):
        if args.validate_ready:
            run("validate_overnight_review.py",["--data-root",root],base/"overnight_ready/readiness_validation.log")
            dump(base/"overnight_ready/readiness_gate.json",dict(ready=True,training_enabled=False,architecture_changes_enabled=False,
                 reviewer_zip_sha256=sha256(base/"overnight_ready/blind_audit_public.zip"),human_review_pending=True))
            print("Ready: no-training replay and anonymous packet QA; human review pending.")
            return
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out=base/"overnight_runs"/stamp
        out.mkdir(parents=True,exist_ok=False)
        log=out/"runner.log"
        status=dict(status="running",started_utc=stamp,optimizer_updates=0,training_enabled=False,stage="preflight")
        dump(out/"status.json",status)
        try:
            run("validate_overnight_review.py",["--data-root",root],log)
            status["stage"]="independent_peak_offset_replay";dump(out/"status.json",status)
            print("Replaying six frozen models on CPU; no training.",flush=True)
            run("peak_offset_diag.py",["--data-root",root,"--output",out/"peak_replay"],log)
            if sha256(out/"peak_replay/peak_offset_diag.csv")!=sha256(base/"peak_offset_diag.csv"):
                raise ValueError("Point-table replay differs; preserve outputs and investigate, never change baseline coordinates")
            status["stage"]="anonymous_packet_replay";dump(out/"status.json",status)
            run("prepare_overnight_review.py",["--data-root",root,"--output",out/"review_pack"],log)
            run("validate_overnight_review.py",["--data-root",root,"--output",out/"review_pack"],log)
            status.update(status="complete",stage="all_automated_checks_complete",point_table_exact=True,
                 human_review_pending=True,completed_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            dump(out/"status.json",status)
            print(json.dumps(status),flush=True)
        except BaseException as error:
            status.update(status="failed_or_interrupted",error=repr(error))
            dump(out/"status.json",status)
            raise


if __name__=="__main__":main()
