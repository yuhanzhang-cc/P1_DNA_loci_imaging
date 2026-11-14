"""Prebuild a sealed runtime that accepts review files later, without rebuilding."""
from pathlib import Path
import shutil,zipfile
from .common import ROOT,OLD,REL,read,sha,write,fresh


def build(data_root,output,hardware_profile=None):
    from diagnostics.mechanism_v1.contracts import code_hashes
    from diagnostics.mechanism_v1.controller import verify
    from diagnostics.mechanism_v1.runner import corpus
    data_root=Path(data_root);data=corpus(data_root);output=fresh(output)
    for name in code_hashes():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    (output/"configs").mkdir();(output/"assets").mkdir();(output/"incoming").mkdir()
    for p in (ROOT/OLD/"configs").glob("*.json"):shutil.copyfile(p,output/"configs"/p.name)
    shutil.copyfile(ROOT/REL/"research_decision.template.json",output/"configs/research_decision.template.json")
    shutil.copyfile(ROOT/REL/"review_packet_binding.json",output/"configs/review_packet_binding.json")
    shutil.copyfile(ROOT/REL/"reference_environment.json",output/"configs/reference_environment.json")
    if hardware_profile:
        from diagnostics.completion_v1.hardware_policy import validate
        validate(read(hardware_profile));shutil.copyfile(hardware_profile,output/'configs/hardware_profile.json')
    shutil.copyfile(data.paths["development"],output/"assets/development_only.npz")
    shutil.copyfile(data.paths["features"],output/"assets/validation_truth_features.csv")
    shutil.copyfile(data_root/"docs/next_stage_plan_20261025/overnight_protocol_v4_proposed_20261101.yaml",output/"protocol_proposed.yaml")
    req="deepblink_pytorch/scripts/diagnostics/mechanism_v1/requirements-mechanism.txt"
    shutil.copyfile(ROOT/req,output/req)
    entry="deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py"
    py='"${P1_PYTHON:-.venv_cuda/bin/python}"'
    scripts={
        "00_verify.sh":f'python3 {entry} verify --root .\n',
        "02_import_review.sh":f': "${{1:?Pass completed review CSV}}" "${{2:?Pass approved research decision JSON}}"\n{py} {entry} freeze --responses "$1" --decision "$2" --output decisions --config-dir configs\n',
        "03_plan.sh":f'python3 {entry} plan --root .\n',
        "04_start.sh":f'{py} {entry} start --root .\n',
        "05_status.sh":f'python3 {entry} status --root .\n',
        "06_resume.sh":f'{py} {entry} start --root . --resume\n',
        "07_pack_completed.sh":f'{py} {entry} pack --root .\n',
        "08_collect_failure.sh":f'python3 {entry} failure --root .\n'}
    scripts["01_setup.sh"]='''python3 -c 'import sys; assert sys.version_info[:2]==(3,12), "Use Python 3.12"'
test ! -e .venv_cuda || { echo "Existing environment protected; reuse via P1_PYTHON"; exit 1; }
python3 -m venv .venv_cuda
.venv_cuda/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv_cuda/bin/python -m pip install -r deepblink_pytorch/scripts/diagnostics/mechanism_v1/requirements-mechanism.txt
.venv_cuda/bin/python -m pip check
.venv_cuda/bin/python -m pip freeze > environment_actual.txt
nvidia-smi > nvidia_smi_setup.txt
'''
    for name,body in scripts.items():(output/name).write_bytes(('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'+body).encode())
    (output/"README.md").write_bytes(('''# Prefabricated review-to-report CUDA kit v2

No formal training has run. Real review and actual CUDA remain unverified.
Upload the completed blind_audit_responses.csv and approved research_decision.json
later to incoming/. Config hashes, approvals, probe paths and queue expansion are
handled by the runtime; there is no need to rebuild this ZIP after ordinary review.

1. bash 00_verify.sh; set P1_PYTHON to the preserved validated CUDA Python.
2. bash 03_plan.sh (no execution); bash 05_status.sh.
3. bash 02_import_review.sh incoming/blind_audit_responses.csv incoming/research_decision.json
4. bash 04_start.sh; bash 05_status.sh.
5. Pipeline: CUDA doctor + initialization -> two one-step probes -> S0/P0 seed42
   -> fixed development gate -> four remaining seeds only if preapproved and gate
   passes -> verify all complete artifacts -> automatic lean return tar.gz + SHA256.
6. Download BOTH return archive and SHA256. Local CLI ingest -> report.
7. Interrupted: bash 08_collect_failure.sh; fix environment/problem, preserve old
   artifacts, then bash 06_resume.sh with unchanged source/config/backend.

The initializers must match historical CUDA references for all seeds. Do not
waive mismatches. A negative scientific gate produces a valid two-run packet;
it does not trigger new architectures or ordinary baselines. If labels actually
need correction, preserve this protocol and create a separately reviewed dataset
revision; this runner will not apply annotation edits automatically.

Secrets/private unblinding keys/test images are absent. Private development data
are included: do not commit the upload kit. The original frozen baseline folders
and the old training recipe/network files remain untouched.
''').encode())
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write(output/"bundle_manifest.json",dict(kind="signed_offset_v1",handoff_version=2,files=files,
        test_arrays_present=False,formal_training_performed=False))
    verify(output)
    archive=output.with_name(output.name+".zip")
    with archive.open("xb") as f:
        with zipfile.ZipFile(f,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for p in sorted(output.rglob("*")):
                if p.is_file():z.write(p,(Path(output.name)/p.relative_to(output)).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError("ZIP CRC failed")
    digest=sha(archive);archive.with_name(archive.name+".sha256").write_bytes((digest+"  "+archive.name+"\n").encode())
    result=dict(path=str(archive),sha256=digest,bytes=archive.stat().st_size,members=len(files)+1,private_development_data=True,
        parameters_still_todo=read(output/"configs/P0.json")["supervision"],formal_training_started=False)
    write(output.parent/(output.name+"_build.json"),result);return result
