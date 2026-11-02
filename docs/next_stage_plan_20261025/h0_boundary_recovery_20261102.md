# H0 CUDA boundary-export incident and recovery — 2026-11-02

Completed update: all three H0 seeds and the paired R0 queue have passed
[local returned-result acceptance](baseline_cuda_analysis_20261102.md). The sections
below preserve the original incident and recovery procedure. The result pack has
all four requested recovery artifacts; epoch-cache verification limits are stated
in the acceptance report. No further recovery training is required.

## Confirmed state

Server logs report R0 seeds 42/43/44 and H0 seed 42 completed. H0 seed 43 failed
while exporting validation candidates at epoch 60. The previous epoch-59 recovery
checkpoint should contain 1,298 optimizer updates. `epoch_060/model.pt` was written
before candidate export failed. These server files have not yet been downloaded
or independently inspected locally.

The underlying numerical defect is reproduced locally: adding float32 offsets to
integer pixel indices creates float64 coordinates. The decoder checked those
coordinates before converting them to float32. A valid `255.99999994039536` becomes
`256.0` during that conversion, which correctly fails the strict serializer bounds
check. This affects both dense and grid decoding near an upper image edge.

No current evidence indicates damaged training data, an OOM, nonfinite training
loss, or invalid learned weights. The actual failed CUDA snapshot must reproduce
this exact numerical issue before the recovery launcher permits further training.

## Fix and evidence

`deepblink_torch/round2_protocol.py` now converts to the saved float32 representation
before applying the strict half-open original-FOV bounds check. Scores remain
paired with their originating cells. This excludes rounded-out predictions; it
does not clip predictions, edit annotations, relax the serializer, change padding,
or change the optimizer/loss/training recipe.

49 targeted tests passed on PyTorch 2.11.0+cpu, covering dense/grid edge cases,
rectangular 256/512 fields, source-score identity, serializer round trips,
nonfinite/error rejection, matching and existing v4 invariants. Pytest was loaded
from the existing base environment while torch and project dependencies came from
the existing `spotiflow_cpu` venv; no dependency or environment files were changed.

A separate synthetic H0 failure/recovery probe executed five optimizer updates:
continuous two epochs, interrupted second epoch, then replay of that epoch. The
recovered model tensors, optimizer state, RNG state and final exported metrics
were identical to uninterrupted training, and the interrupted snapshot remained
under `orphaned_snapshots/`. These are engineering checks, not additional formal
experiments or actual CUDA recovery results.

Local evidence is under
`outputs/next_stage_v4_recipe_20261101/14_h0_incident/`:
`boundary_bug_reproduction.json`, `dense_target_audit.json`,
`cpu_engineering_2_updates/probe_checks.json`, and
`h0_exact_recovery_probe/checks.json`. All 176 training and 22 validation dense
targets also passed construction/bounds checks. No test arrays were read.

## Recover the sealed server bundle

Do not replace a manifest-listed source file in the existing bundle, change its
manifest to bypass verification, delete trial folders, or restart the whole
minimum-baseline queue. The original source and registered training identity must
remain inspectable. Future bundles use the source fix normally.

The additive recovery overlay filters the *old decoder's returned float32*
coordinates using the same bounds as the source fix. Its equivalence is tested
for the registered integer image shapes. It rejects other malformed or nonfinite
coordinates. The original training/threshold search budget remains fixed.
The effective evaluation change is explicitly recorded alongside the original
manifest identity in `numerical_decode_addendum.json`, including the overlay's
SHA256, its source, actual CUDA reproduction and pre-recovery hashes. Do not
describe the effective evaluation implementation as unchanged.

Upload `p1_h0_boundary_recovery_v1.zip` into the existing server directory
`/root/autodl-tmp/p1_v4_training`, then run:

```bash
cd /root/autodl-tmp/p1_v4_training
unzip -o p1_h0_boundary_recovery_v1.zip
bash 08_h0_recover.sh
```

This adds files outside the original manifest; it does not overwrite original
training source. `08_h0_recover.sh` is copied from the repository's
`deepblink_pytorch/scripts/08_h0_recover.sh` to the ZIP root for execution.

The script will:

1. Verify patch files, original bundle, GPU/torch identity and completed queues.
2. Verify saved H0 candidates and hash completed R0/H0 artifacts for preservation.
3. Load the actual failed epoch-60 model, confirm tensor identity, reproduce the
   rounded boundary, and record affected images/points. If the cause differs, stop.
4. Pack the completed R0 result and create a timestamped H0 backup plus SHA256.
5. Detach the H0 recovery process. Seed 42 is skipped; seed 43 restores epoch 59;
   the old epoch-60 snapshot is retained under `orphaned_snapshots/`.
6. Require replayed epoch-60 model tensors to exactly match the original failed
   checkpoint before exporting its corrected candidates, then finish seeds 43/44.
7. Recheck that protected results are byte-identical and automatically pack H0.

Monitoring:

```bash
bash 04_status.sh
tail -n 40 outputs/research_v4_cuda/baseline_h0_boundary_recovery.log
```

Download `baseline_r0_results.tar.gz` / `.sha256` and the timestamped
`h0_recovery_backup_*.tar.gz` / `.sha256` for a local backup. Download
`baseline_h0_results.tar.gz` / `.sha256` after recovery succeeds. The latter
contains the numerical addendum, recovery overlay source, boundary-event log and
exact-replay check. Full epoch caches and recovery checkpoints remain on the
server, with the pre-recovery H0 state also preserved in the backup archive.

If interrupted after successful replay, repeat
`.venv_cuda/bin/python deepblink_pytorch/scripts/recover_h0_boundary.py --start`.
The existing addendum is validated and the latest committed epoch is resumed.
Inspect the recovery log if any gate fails; never clear the folders to bypass it.

## Acceptance and remaining work

Actual CUDA recovery is pending the user's server run. Accept H0 only after all
three seeds pass the original 100-epoch/2,200-update, 20-snapshot/620-pair checks,
the exact epoch-60 replay check, and completed-result preservation checks.
Then return the R0/H0 archives for local integrity verification and fair analysis.
Keep the existing G0 reference recipe as the simple candidate. Baseline completion
does not by itself establish an architectural improvement or independent test
performance; the previously planned core mechanism remains a separate experiment.
