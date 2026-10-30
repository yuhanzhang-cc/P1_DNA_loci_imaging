#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/spotiflow_cuda
if [[ -f outputs/spotiflow_cuda/run.pid ]] && kill -0 "$(cat outputs/spotiflow_cuda/run.pid)" 2>/dev/null; then
  echo "A recorded worker is still alive; inspect status before launching again."
  exit 1
fi
.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_spotiflow_train.py --verify-only
nohup .venv_cuda/bin/python -u deepblink_pytorch/scripts/cloud_spotiflow_train.py --train --device cuda > outputs/spotiflow_cuda/training.log 2>&1 < /dev/null &
echo $! > outputs/spotiflow_cuda/run.pid
echo "Started three sequential seeds; PID $(cat outputs/spotiflow_cuda/run.pid)."
echo "Run: bash 04_status.sh"
