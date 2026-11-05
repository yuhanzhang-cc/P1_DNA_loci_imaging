"""Build an allowlisted local upload kit; never publish study data to Git."""
from pathlib import Path
import argparse,json,shutil,zipfile
from .contracts import ROOT,REL,read,sha,write_new,code_hashes
from .controller import verify


def build(data_root,output,config_dir=None):
    data_root=Path(data_root).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    selected=[ROOT/name for name in code_hashes()]
    selected += [ROOT/"deepblink_pytorch/scripts/diagnostics/mechanism_v1/requirements-mechanism.txt"]
    for path in selected:
        target=output/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    (output/"configs").mkdir(exist_ok=True)
    config_dir=Path(config_dir) if config_dir else ROOT/REL/"configs"
    for name in ("S0.json","P0.json","approval.template.json","reference_initializations.json","blind_audit_expected_ids.json"):
        shutil.copyfile(config_dir/name,output/"configs"/name)
    from .contracts import validate
    for method in ("S0","P0"):validate(read(output/"configs"/(method+".json")),require_parameters=method=="S0")
    import yaml
    source=data_root/"docs/next_stage_plan_20261025/overnight_protocol_v4_proposed_20261101.yaml"
    spec=yaml.safe_load(source.read_text(encoding="utf-8"))
    development=data_root/spec["data"]["development_npz"]
    if sha(development)!=spec["data"]["development_sha256"]:raise ValueError("Development archive differs")
    import numpy as np
    with np.load(development,allow_pickle=True) as data:
        if set(data.files)!=set(spec["data"]["allowed_npz_keys"]) or any("test" in n for n in data.files):raise ValueError("Test arrays forbidden")
    assets=output/"assets";assets.mkdir()
    shutil.copyfile(development,assets/"development_only.npz")
    shutil.copyfile(data_root/"outputs/next_stage_v3_closing_controls/02_analysis/validation_truth_features.csv",assets/"validation_truth_features.csv")
    shutil.copyfile(source,output/"protocol_proposed.yaml")
    (output/"decisions").mkdir()
    entry="deepblink_pytorch/scripts/diagnostics/mechanism_v1/controller.py"
    body={
        "00_verify.sh":f'python3 {entry} verify\n',
        "02_smoke.sh":f'"${{P1_PYTHON:-.venv_cuda/bin/python}}" {entry} smoke --method "${{1:-S0}}"\n',
        "03_start.sh":f'method="${{1:-S0}}"; shift || true\n"${{P1_PYTHON:-.venv_cuda/bin/python}}" {entry} start --method "$method" --seeds "${{@:-42}}"\n',
        "04_status.sh":f'python3 {entry} status\n',
        "05_pack_results.sh":f'method="${{1:-S0}}"; shift || true\npython3 {entry} pack --method "$method" --seeds "${{@:-42}}"\n',
        "06_resume.sh":f'method="${{1:-S0}}"; shift || true\n"${{P1_PYTHON:-.venv_cuda/bin/python}}" {entry} resume --method "$method" --seeds "${{@:-42}}"\n',
        "07_plan.sh":f'python3 {entry} start --method "${{1:-S0}}" --seeds 42 --dry-run\n'}
    body["01_setup.sh"]='''python3 -c 'import sys; assert sys.version_info[:2]==(3,12), "Use Python 3.12"'
test ! -e .venv_cuda || { echo "Existing venv: inspect/reuse with P1_PYTHON; do not overwrite"; exit 1; }
python3 -m venv .venv_cuda
.venv_cuda/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv_cuda/bin/python -m pip install -r deepblink_pytorch/scripts/diagnostics/mechanism_v1/requirements-mechanism.txt
.venv_cuda/bin/python -m pip check
.venv_cuda/bin/python -m pip freeze > environment_actual.txt
nvidia-smi > nvidia_smi_setup.txt
'''
    for name,commands in body.items():
        (output/name).write_bytes(('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'+commands).encode())
    (output/"README.md").write_text("""# Signed-offset prospective kit

No formal training has run. P0 parameters and human decisions remain TODO.
Local CPU weights reproduce the local H0 initializer but not the historical
CUDA hash. The CUDA probe must match the historical seed42 initializer before
proceeding. A mismatch is an investigation gate, not evidence of model failure.
1. Verify ZIP SHA256, unzip into a new directory; `bash 00_verify.sh`.
2. Reuse the original validated CUDA venv via P1_PYTHON, or `bash 01_setup.sh`.
3. After reviewing the mechanism/config, `bash 02_smoke.sh S0` (one synthetic
   step, replayed once; full CUDA batch8/canvas512; no real-data updates).
4. Copy the successful probe checks.json to decisions/S0_probe.json.
5. Copy configs/approval.template.json to decisions/S0.json and fill reviewer,
   date, mechanism_approved=true, synthetic_fixture=false and exact config SHA
   printed by the CLI dry-run. Do not use this to substitute for human review.
6. `bash 07_plan.sh S0`; then `bash 03_start.sh S0 42` after review.
7. `bash 04_status.sh`; inspect seed42 technical readout before seeds43/44.
8. `bash 05_pack_results.sh S0 42`; download archive AND SHA256.
9. Resume only with the same source/config/GPU/PyTorch identity:
   `bash 06_resume.sh S0 42`. Completed selected checkpoints are verified.

For P0: confirm TODO parameters from blind review, write a new config revision,
re-seal a new bundle, run a fresh P0 CUDA probe, and bind the actual completed
blind-audit file/hash in its approval. Editing a sealed file invalidates verify.
No test images are included. This upload ZIP contains private development data;
keep it off GitHub. Full 6-run CUDA budget is provisional, not measured here.
""",encoding="utf-8",newline="\n")
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_new(output/"bundle_manifest.json",dict(kind="signed_offset_v1",files=files,test_arrays_present=False,formal_training_performed=False))
    verify(output)
    archive=output.with_name(output.name+".zip")
    with archive.open("xb") as stream:
        with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for p in sorted(output.rglob("*")):
                if p.is_file():z.write(p,(Path(output.name)/p.relative_to(output)).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError("ZIP CRC failed")
    archive.with_name(archive.name+".sha256").write_bytes((sha(archive)+"  "+archive.name+"\n").encode("ascii"))
    result=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files)+1,private_data=True,formal_training_started=False)
    write_new(output.parent/(output.name+"_build.json"),result);print(json.dumps(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config-dir",type=Path,help="Reviewed config revision directory; a fresh bundle is always sealed")
    a=p.parse_args();build(a.data_root,a.output,a.config_dir)
if __name__=="__main__":main()
