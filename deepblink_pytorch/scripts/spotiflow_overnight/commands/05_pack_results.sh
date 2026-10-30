#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv_cuda/bin/python - <<'PY'
import json
from pathlib import Path
for seed in (42,43,44):
    p=Path(f'outputs/spotiflow_cuda/seed_{seed}/status.json')
    if not p.exists() or json.loads(p.read_text())['status']!='complete':
        raise SystemExit('Some seeds are not complete; inspect status before final packaging.')
print('All three seeds complete.')
PY
tar -czf spotiflow_cuda_results.tar.gz outputs/spotiflow_cuda installed_cuda_environment.txt bundle_manifest.json
echo "Download spotiflow_cuda_results.tar.gz, then shut down the instance manually."
