#!/usr/bin/env bash
# Upload this additive patch into the existing p1_v4_training directory.
set -euo pipefail
cd "$(dirname "$0")"
test -f bundle_manifest.json
test -x .venv_cuda/bin/python
sha256sum -c h0_boundary_patch_files.sha256

# This gates on the actual failed CUDA model and leaves original source intact.
.venv_cuda/bin/python deepblink_pytorch/scripts/recover_h0_boundary.py --audit-only
bash 05_pack_results.sh baseline_r0
stamp=$(date -u +%Y%m%dT%H%M%SZ)
backup="h0_recovery_backup_${stamp}.tar.gz"
test ! -e "$backup"
tar -czf "$backup" \
  outputs/research_v4_cuda/baseline_h0 \
  outputs/research_v4_cuda/minimum_baselines.log \
  bundle_manifest.json
sha256sum "$backup" > "${backup}.sha256"
printf 'Backup preserved: %s (download this and its .sha256 if desired)\n' "$backup"
.venv_cuda/bin/python deepblink_pytorch/scripts/recover_h0_boundary.py --start
printf 'Progress: bash 04_status.sh\nLog: tail -n 40 outputs/research_v4_cuda/baseline_h0_boundary_recovery.log\n'
