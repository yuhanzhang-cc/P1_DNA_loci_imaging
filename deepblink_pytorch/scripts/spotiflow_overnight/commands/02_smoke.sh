#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/spotiflow_cuda
.venv_cuda/bin/python -u deepblink_pytorch/scripts/cloud_spotiflow_train.py --smoke --device cuda 2>&1 | tee outputs/spotiflow_cuda/smoke.log
