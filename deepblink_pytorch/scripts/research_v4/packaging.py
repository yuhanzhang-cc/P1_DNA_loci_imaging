"""Allowlisted portable training and frozen-model evaluation kits."""
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from .common import dump,read_json,rows,sha256,write_csv
from .external import exclusion_manifest


def copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)


def sources(ctx,target):
    root=ctx.code_root
    selected=list((root/"deepblink_pytorch/deepblink_torch").rglob("*.py"))
    selected+=list((root/"deepblink_pytorch/scripts/research_v4").glob("*.py"))
    for name in ("cloud_recipe_train.py","cloud_job_control.py","evaluate_external.py","analyze_research_v4.py","validate_ablation_plan.py","error_context_diagnostics.py"):
        selected.append(root/"deepblink_pytorch/scripts"/name)
    for name in ("__init__.py","common.py","engine.py","evaluation.py"):
        selected.append(root/"deepblink_pytorch/scripts/spotiflow_overnight"/name)
    for name in ("pyproject.toml","README.md","requirements-research-v4.txt"):
        selected.append(root/"deepblink_pytorch"/name)
    for path in selected:copy(path,target/path.relative_to(root))
    copy(ctx.data_root/"outputs/next_stage_v3_closing_controls/00_protocol/America_New_York.tzif",
         target/"outputs/next_stage_v3_closing_controls/00_protocol/America_New_York.tzif")


SETUP='''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -c 'import sys; assert sys.version_info[:2]==(3,12), "Use Python 3.12"'
python3 -m venv .venv_cuda
.venv_cuda/bin/python -m pip install --upgrade pip
.venv_cuda/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv_cuda/bin/python -m pip install -r deepblink_pytorch/requirements-research-v4.txt
.venv_cuda/bin/python -m pip install --no-deps -e deepblink_pytorch
.venv_cuda/bin/python -m pip check
.venv_cuda/bin/python -m pip freeze > environment_actual.txt
nvidia-smi > nvidia_smi_setup.txt
'''


def shells(target,training):
    files={"01_setup.sh":SETUP}
    if training:
        files.update({
            "02_smoke.sh":'queue="${1:-recipe}"\n.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_recipe_train.py --smoke --device cuda --queue "$queue"\n',
            "03_start.sh":'.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_job_control.py start --queue "${1:-recipe}"\n',
            "04_status.sh":'.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_recipe_train.py --status\n',
            "05_pack_results.sh":'.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_job_control.py pack --queue "${1:-recipe}"\n',
            "06_resume.sh":'.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_job_control.py resume --queue "${1:-recipe}"\n'})
    else:
        files["02_verify.sh"]='.venv_cuda/bin/python deepblink_pytorch/scripts/evaluate_external.py --verify-only\n'
        files["03_evaluate.sh"]=': "${1:?Pass the new-data manifest CSV path}"\n.venv_cuda/bin/python deepblink_pytorch/scripts/evaluate_external.py --manifest "$1" --device "${2:-cuda}"\n'
    for name,body in files.items():
        if name!="01_setup.sh":body='#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'+body
        (target/name).write_text(body,encoding="utf-8",newline="\n")


def seal(target,kind):
    # A previous partial build is never silently reused.
    files={p.relative_to(target).as_posix():sha256(p) for p in sorted(target.rglob("*")) if p.is_file() and p.name!="bundle_manifest.json"}
    if any("__pycache__" in Path(n).parts or ".git" in Path(n).parts or ".venv" in Path(n).parts for n in files):raise ValueError("Unexpected local files in bundle")
    dump(target/"bundle_manifest.json",dict(kind=kind,files=files,test_arrays_present=False,formal_training_performed=False))
    archive=target.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(target.rglob("*")):
            if p.is_file():z.write(p,(Path(target.name)/p.relative_to(target)).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError("ZIP CRC check failed")
    digest=sha256(archive)
    archive.with_name(archive.name+".sha256").write_text(digest+"  "+archive.name+"\n",encoding="ascii")
    return dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(files)+1)


def frozen_models(ctx,target):
    locks=[]
    for method,study in (("R0","next_stage_v3"),("G0","next_stage_v3_closing_controls")):
        for seed in (42,43,44):
            source=ctx.data_root/f"outputs/{study}/01_screen/{method}/seed_{seed}"
            selection=read_json(source/"selection.json");relative=f"models/{method}/seed_{seed}/best_model.pt"
            copy(source/"best_model.pt",target/relative)
            if sha256(target/relative)!=selection["checkpoint_sha256"]:raise ValueError("Archived selected model hash differs")
            locks.append(dict(method=method,seed=seed,kind="own",checkpoint=relative,
                weight_sha256=sha256(target/relative),threshold=selection["primary"]["threshold"],pretrained=False,configuration_hashes={}))
    for seed in (42,43,44):
        source=ctx.data_root/f"outputs/next_stage_spotiflow_20261030/06_cuda_results/extracted/outputs/spotiflow_cuda/seed_{seed}"
        selection=read_json(source/"selection.json");relative=f"models/Spotiflow_F/seed_{seed}"
        for name in ("best.pt","config.yaml","thresholds.yaml"):copy(source/"primary"/name,target/relative/name)
        if sha256(target/relative/"best.pt")!=selection["primary_weight_sha256"]:raise ValueError("CUDA selected weight differs")
        locks.append(dict(method="Spotiflow_F",seed=seed,kind="spotiflow",checkpoint=relative,
            weight_sha256=sha256(target/relative/"best.pt"),threshold=selection["primary"]["threshold"],pretrained=True,
            configuration_hashes={n:sha256(target/relative/n) for n in ("config.yaml","thresholds.yaml")}))
    dump(target/"method_locks.json",locks)
    return locks


def build(ctx):
    ctx.task("portable_bundles","running")
    out=ctx.out/"04_bundles";out.mkdir(exist_ok=True)
    # Use a new revision directory after source changes, never mutate a sealed kit.
    from .common import source_hashes,canonical_hash
    package_inputs=dict(sources=source_hashes(ctx.code_root),spec=sha256(ctx.spec_path),handoff={n:sha256(ctx.code_root/"docs/next_stage_plan_20261025"/n) for n in ("gpu_handoff_20261102.md","external_validation_spec_20261102.md")})
    stamp=canonical_hash(package_inputs)[:12]
    folder=out/stamp;folder.mkdir(exist_ok=True)
    if (folder/"checks.json").exists():
        result=read_json(folder/"checks.json")
        for item in result["archives"]:
            if sha256(item["path"])!=item["sha256"]:raise ValueError("Sealed package changed")
        dump(out/"latest.json",result)
        ctx.task("portable_bundles","completed",**result);return result
    train=folder/"p1_v4_training";external=folder/"p1_v4_external"
    if train.exists() or external.exists():raise ValueError("Unsealed partial kit exists; inspect before rebuilding in a new revision")
    train.mkdir();external.mkdir()
    for target in (train,external):sources(ctx,target)
    for target in (train,external):
        for name in ("gpu_handoff_20261102.md","external_validation_spec_20261102.md"):
            copy(ctx.code_root/"docs/next_stage_plan_20261025"/name,target/"docs"/name)
    copy(ctx.spec_path,train/"protocol_proposed.yaml")
    copy(ctx.corpus.paths["development"],train/"assets/development_only.npz")
    copy(ctx.corpus.paths["features"],train/"assets/validation_truth_features.csv")
    cloud=ctx.data_root/"outputs/next_stage_spotiflow_20261030/06_cuda_results/extracted/outputs/spotiflow_cuda/seed_42/primary/config.yaml"
    copy(cloud,train/"assets/spotiflow_config.yaml")
    from .queues import queues
    dump(train/"gpu_queues.json",queues())
    dump(train/"execution_contract.json",dict(status="local_prepared_GPU_deferred",source_revision=stamp,
         original_spec_metadata="proposal preserved verbatim; scope authorized 2026-11-02",default_queue="recipe",default_tasks=6,optional_baseline_tasks=9,
         cuda_float32=True,no_AMP=True,no_TF32=True,core_ablation_enabled=False))
    shells(train,True);shells(external,False)
    locks=frozen_models(ctx,external)
    exclusions=exclusion_manifest(ctx.corpus,rows(ctx.data_root/ctx.spec["data"]["split_csv"]))
    dump(external/"assets/reference_exclusions.json",exclusions)
    dump(external/"contrasts.json",[["Spotiflow_F","G0"],["G0","R0"]])
    from .scaffold import EXTERNAL_FIELDS
    write_csv(external/"external_data_collection_template.csv",[],EXTERNAL_FIELDS)
    notes="""# P1 v4 portable handoff

Python3.12, Linux NVIDIA CUDA. Recommended single RTX4090 24GB, >=8 CPU cores, >=32GB RAM, >=30GB free disk. Setup uses an isolated .venv_cuda; it does not reuse or replace an old training environment. Driver must support CUDA12.8. Driver/GPU execution and dependency installation must be checked on the rented host.

Training kit (default: six100-epoch G0 reference/augmentation tasks, paired seeds42/43/44):

```bash
bash 01_setup.sh
bash 02_smoke.sh recipe
bash 03_start.sh recipe
bash 04_status.sh
# After all six complete:
bash 05_pack_results.sh recipe
```

Do not change batch size or recipe to fix OOM; retain failure evidence and use a protocol revision. Each optional queue baseline_r0, baseline_h0 or spotiflow_scratch requires its own `bash 02_smoke.sh QUEUE` then `bash 03_start.sh QUEUE`. Nothing automatically starts the other nine tasks. `bash 06_resume.sh QUEUE` resumes from an audited epoch boundary. Formal training requires CUDA and a manifest-bound CUDA smoke including the actual adapter geometry; CPU tests cannot authorize it. See docs/gpu_handoff_20261102.md for details.

External kit: follow docs/external_validation_spec_20261102.md and fill the template with newly collected pixel y,x data and real biological/acquisition identifiers. Run `bash 01_setup.sh`, `bash 02_verify.sh`, then `bash 03_evaluate.sh /absolute/path/manifest.csv cuda`. All nine weights and thresholds are frozen. No development image arrays are included. Metadata overlap checks are not proof of biological independence. First external evaluation output is preserved. Larger FOVs than512 need a separate tiling protocol.

Download the completed queue_results.tar.gz and its.sha256. Only selected weights, metrics, selections, protocols and logs are returned; intermediate snapshots/recovery remain on server. Keep server until download/hash and local evaluation checks pass. The kit itself includes no test image arrays and no review images/private identity mapping. Classical methods are deterministic local references, not additional trained seeds. Core architecture ablation remains disabled until a justified core method exists.
"""
    for target in (train,external):(target/"README.md").write_text(notes,encoding="utf-8")
    result=dict(passed=True,source_revision=stamp,archives=[seal(train,"training"),seal(external,"external_evaluation")],
                default_gpu_tasks=6,optional_gpu_tasks=9,frozen_external_models=len(locks),cuda_tested=False,network_setup_tested=False)
    dump(folder/"checks.json",result);dump(out/"latest.json",result)
    ctx.task("portable_bundles","completed",**result)
    return result
