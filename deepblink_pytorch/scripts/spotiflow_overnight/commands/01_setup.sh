#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash setup_cuda.sh
.venv_cuda/bin/python -m pip freeze > installed_cuda_environment.txt
