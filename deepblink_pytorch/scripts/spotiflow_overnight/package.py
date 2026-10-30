"""Build a private, local dev-only CUDA handoff; never upload or rent resources."""
import json
from pathlib import Path
import shutil
import zipfile
from importlib.metadata import version

import numpy as np
from deepblink_torch.experiment_log import sha256
from .common import ROOT,OUT,WEIGHTS,ARCHIVE,source_hashes,dump,status


def object_array(values):
    result=np.empty(len(values),dtype=object)
    result[:]=values
    return result


def build(data,force=False):
    destination=OUT/"05_cuda_package"
    root=destination/"bundle"
    marker=destination/"package_checks.json"
    if marker.exists() and not force:return json.loads(marker.read_text())
    root.mkdir(parents=True,exist_ok=True)
    status("cuda_handoff",activity="Local package: 176 train + 22 validation only; no test arrays or upload")
    assets=root/"assets"
    (assets/"general").mkdir(parents=True,exist_ok=True)
    values={}
    for subset,d in data.items():
        values[f"x_{subset}"]=object_array(d["images"])
        values[f"y_{subset}"]=object_array(list(d["y"]))
        values[f"sample_id_{subset}"]=d["sample_id"]
        values[f"original_shape_{subset}"]=d["original_shape"]
    np.savez_compressed(assets/"development_only.npz",**values)
    with np.load(assets/"development_only.npz",allow_pickle=True) as archive:
        assert len(archive.files)==8 and not any("test" in key for key in archive.files)
        for subset,d in data.items():
            assert all(np.array_equal(a,b) for a,b in zip(archive[f"x_{subset}"],d["images"]))
            assert all(np.array_equal(a,b) for a,b in zip(archive[f"y_{subset}"],d["y"]))
    for name in ("best.pt","config.yaml","thresholds.yaml","train_config.yaml"):
        shutil.copyfile(WEIGHTS/name,assets/"general"/name)
    source_files=list(source_hashes())+["deepblink_pytorch/scripts/error_context_diagnostics.py",
        "deepblink_pytorch/pyproject.toml","deepblink_pytorch/README.md"]
    for relative in source_files:
        path=root/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,path)
    requirements="spotiflow==0.6.5\nlightning==2.6.6\ntorchmetrics==1.9.0\ntorch==2.11.0\ntorchvision==0.26.0\n"
    requirements += "".join(f"{name}=={version(name)}\n" for name in ("numpy","scipy","tifffile","matplotlib"))
    (root/"requirements-cuda.txt").write_text(requirements,encoding="utf-8")
    setup="""#!/usr/bin/env bash
set -euo pipefail
python -c 'import sys; assert sys.version_info[:2] == (3,12), "Choose a Python 3.12 image"'
python -m venv .venv_cuda
.venv_cuda/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv_cuda/bin/python -m pip install -r requirements-cuda.txt
.venv_cuda/bin/python -m pip install --no-deps -e deepblink_pytorch
.venv_cuda/bin/python -m pip check
.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_spotiflow_train.py --verify-only
"""
    (root/"setup_cuda.sh").write_text(setup,encoding="utf-8",newline="\n")
    readme="""# Private CUDA handoff (not uploaded)

Data: 176 training + 22 validation images and pixel y,x labels. No test images or test arrays.
Suggested starting instance: one NVIDIA GPU with 24GB VRAM, Python 3.12, adequate CPU/RAM and disk.
This is a conservative proposed configuration, not a measured minimum or a rental action.

1. Check the NVIDIA driver supports the CUDA 12.8 runtime; select a compatible image.
2. From this bundle root: `bash setup_cuda.sh` (isolated environment).
3. `.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_spotiflow_train.py --smoke --device cuda`
4. After the CUDA recovery smoke passes, run the same entry with `--train --device cuda`.
5. Audit before using `--resume`; return the entire output including epoch candidates and recovery state.

Formal configuration: general fine-tuning; 100 epochs, batch4, crop256, lr0.0003, AdamW wd0.01,
native smart crop/augmentations; no-replacement sampler; native RNG keyed to seed/epoch/image.
Twenty actual current-epoch snapshots, 31 global thresholds each, full-image 3px validation matching.
This is not equal-compute training with the local batch8 model and not an independent test.

CUDA execution and Linux dependency installation have NOT been verified locally; only CPU engineering
and bundle data/source integrity have been verified. Do not treat a successful manifest check as a GPU test.
Run all 42/43/44 on the same CUDA stack; do not merge CPU pilot weights into a CUDA seed.
Do not use the Windows CPU full lock to install Linux CUDA dependencies. Export pip freeze after setup.

No credentials, environment directories, test arrays or private Git metadata are included.
The general checkpoint is included for private reproducibility; pretrained data overlap remains unresolved.
Keep this dataset bundle off public GitHub. No cloud account has been accessed, no upload has occurred.

Official installation references checked 2026-10-30:
https://pytorch.org/get-started/previous-versions/#v2110
https://weigertlab.org/spotiflow/installation.html
"""
    (root/"RUNME.md").write_text(readme,encoding="utf-8")
    guide="docs/next_stage_plan_20261025/autodl_runbook_20261030.md"
    shutil.copyfile(ROOT/guide,root/"AUTODL_README.md")
    commands=[]
    for path in (Path(__file__).parent/"commands").glob("*.sh"):
        (root/path.name).write_bytes(path.read_bytes().replace(b"\r\n",b"\n"))
        commands.append(path.name)
    payload=set(source_files)|set(commands)|{"requirements-cuda.txt","setup_cuda.sh","RUNME.md","AUTODL_README.md","assets/development_only.npz"}
    payload.update("assets/general/"+name for name in ("best.pt","config.yaml","thresholds.yaml","train_config.yaml"))
    files={name:sha256(root/name) for name in sorted(payload)}
    dump(root/"bundle_manifest.json",dict(files=files,training_images=176,validation_images=22,
        test_arrays_present=False,original_archive_sha256=sha256(ARCHIVE),source_provenance=source_hashes(),
        cuda_verified=False,cloud_uploaded=False))
    archive_path=destination/"spotiflow_dev_cuda_bundle.zip"
    with zipfile.ZipFile(archive_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name in sorted(files)+["bundle_manifest.json"]:archive.write(root/name,name)
    result=dict(archive=str(archive_path.relative_to(ROOT)),sha256=sha256(archive_path),bytes=archive_path.stat().st_size,
        source_files=len(files),dev_images=198,test_images=0,cuda_locally_verified=False,uploaded=False,
        local_data_roundtrip_passed=True)
    dump(marker,result)
    (destination/"SHA256SUMS.txt").write_text(result["sha256"]+"  "+archive_path.name+"\n",encoding="ascii")
    return result
