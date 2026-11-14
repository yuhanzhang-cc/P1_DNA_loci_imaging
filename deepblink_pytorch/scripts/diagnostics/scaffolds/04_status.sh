#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${P1_SCAFFOLD_PYTHON:-python}" "$here/status.py" "$@"
