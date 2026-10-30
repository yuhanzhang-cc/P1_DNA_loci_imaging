#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f outputs/spotiflow_cuda/run.pid ]] && kill -0 "$(cat outputs/spotiflow_cuda/run.pid)" 2>/dev/null; then
  echo "Worker is alive. Do not start a duplicate resume."
  exit 1
fi
.venv_cuda/bin/python deepblink_pytorch/scripts/cloud_spotiflow_train.py --verify-only
nohup .venv_cuda/bin/python -u deepblink_pytorch/scripts/cloud_spotiflow_train.py --train --device cuda --resume >> outputs/spotiflow_cuda/training.log 2>&1 < /dev/null &
echo $! > outputs/spotiflow_cuda/run.pid
echo "Audited same-config recovery started. PID $(cat outputs/spotiflow_cuda/run.pid)."
