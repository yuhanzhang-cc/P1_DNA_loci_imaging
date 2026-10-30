#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv_cuda/bin/python - <<'PY'
import json
from pathlib import Path
for seed in (42,43,44):
    root=Path(f'outputs/spotiflow_cuda/seed_{seed}')
    state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'status':'not_started'}
    progress=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else {}
    print(f"seed {seed}: {state['status']}; epoch {progress.get('epoch',0)}/100; updates {progress.get('optimizer_updates',0)}")
PY
if [[ -f outputs/spotiflow_cuda/training.log ]]; then tail -n 20 outputs/spotiflow_cuda/training.log; fi
nvidia-smi
