# Signed-offset implementation and no-training handoff

最新总览：[OVERNIGHT_REPORT_20261104.md](OVERNIGHT_REPORT_20261104.md)。服务器操作：[HANDOFF.md](HANDOFF.md)。机器可读证据：[CHECKS.json](CHECKS.json)。源代码：`deepblink_pytorch/scripts/diagnostics/mechanism_v1/`。

S0/P0 adapter implemented; CPU engineering checks passed; **formal launch remains pending human decisions and actual CUDA verification**. No new formal experiments or test scores. G0 remains the practical candidate. Historical scaffold outputs and prior conclusions are preserved.

## Coordinate and loss contract

- Coordinates and offsets: `(y,x)` in original-image pixels; origin is the top-left original pixel. Normalize the original image, then zero-pad right/bottom to 512; do not rescale or shift coordinates.
- S0: H0 architecture/state keys and initialization, with a linear two-channel offset output. Keep H0 floor-anchor target and exact `dense_heatmap_loss`; no post-hoc offset correction, extra channels or attention.
- P0: same S0 model. Configurable square support about each floor anchor, clipped to original FOV. Shared pixels belong to the closest true coordinate; exact ties prefer the earlier original GT index. Same-pixel collisions and truth with zero owned pixels are explicit errors.
- P0 per-pixel offset target is `truth - source_pixel`. Weight each owned pixel by inverse owned-pixel count for its GT. Compute weighted mean squared Euclidean error over all GT in the batch, then square root and multiply by the unchanged coefficient 2. Heatmap Dice+BCE remains unchanged. Thus wider support does not multiply a point's total weight.
- Peak selection: frozen 3x3 maxima and tie rule, on original FOV; confidence is the source peak's score. No score floor for ranked AP. Convert coordinates to float32, then filter outside FOV; do not clamp signed values.
- Full ranked AP: confidence-ordered greedy nearest unmatched GT within 3px, equal-score groups merged, sum of recall increments times precision; no interpolation, AP31 approximation or COCO naming. Locked-threshold F1 retains frozen maximum-cardinality matching. Both matching policies are explicitly distinct.
- Weak Recall uses the frozen local-zscore policy from `scaffold_20261103/configs/weak_policy.json`; no threshold refitting. Constant/nonfinite local SNR is unclassifiable and counted separately for new independent data.

## One-line reproduction commands

Run from repository root. `python` must be an isolated environment with Torch/Numpy/Scipy/PyYAML/Tifffile/Filelock; pytest is additionally needed for unit tests. The current local CPU interpreter is `.venv/spotiflow_cpu/Scripts/python.exe`. No shared environment was modified. The optional cloud dependency file reuses the already validated v4 versions.

|Output|One-line command|Evidence|
|---|---|---|
|S0 configuration check|`python deepblink_pytorch/scripts/diagnostics/mechanism_v1/cli.py --config docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/implementation_20261104/configs/S0.json --dry-run`|`evidence/config_S0.txt`|
|P0 TODO check|Same command with `P0.json`|`evidence/config_P0.txt`: parameters_configured=false; formal launch false|
|Coordinate/loss/review/restore tests|`python -m pytest -q deepblink_pytorch/scripts/diagnostics/mechanism_v1/test_mechanism.py`|`evidence/unit_mechanism_and_diagnostics.txt`: 37 passed including 13 prior diagnostic tests|
|Existing scaffold tests|`python -m pytest -q deepblink_pytorch/scripts/diagnostics/scaffolds/test_scaffolds.py`|`evidence/unit_scaffolds.txt`: 22 passed|
|Initialization/sampler/P0 fixture/protected-file audit|`python deepblink_pytorch/scripts/diagnostics/mechanism_v1/analysis_cli.py check --data-root . --output outputs/mechanism_readiness_fresh.json`|`evidence/readiness.txt`; use a fresh output path|
|CPU one-step probe + real encoding/FOV checks|`python deepblink_pytorch/scripts/diagnostics/mechanism_v1/cli.py --config docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/implementation_20261104/configs/S0.json --data-root . --output outputs/mechanism_probe_fresh --smoke --device cpu`|`evidence/cpu_smoke.json`; 1 distinct synthetic step, replayed once; 0 real updates|
|Nine-model validation/AP replay|`python deepblink_pytorch/scripts/diagnostics/mechanism_v1/analysis_cli.py replay --data-root . --output outputs/mechanism_validation_replay_fresh`|`evidence/validation_replay_checks.json`: all thresholded counts equal|
|Independent-evaluation dry-run|`python deepblink_pytorch/scripts/diagnostics/mechanism_v1/eval_cli.py --data-root . --locks docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/configs/validation_model_locks.json --method G0 --seed 42 --dataset-role independent_test --dry-run`|`evidence/independent_eval_dry_run.txt`: images_read=0|
|Fresh local upload ZIP|`python deepblink_pytorch/scripts/diagnostics/mechanism_v1/module_cli.py --data-root . --output outputs/p1_signed_offset_fresh`|`CHECKS.json`: 95 ZIP members; SHA256+CRC verified|

The exact local test command in the evidence loads the already installed pure-Python pytest tooling, then removes that optional path before Torch initialization. Leaving the Python 3.13 base package directory on a Python 3.12 Torch process incorrectly exposed an incompatible optional `optree` extension; no dependencies were installed or upgraded to fix that tooling issue.

## Input and output boundaries

Inputs: frozen `development_only.npz` (train176/valid22 only), existing 372-point validation features, nine validation-selected weight/selection locks, historical H0 initialization and sampling hashes. Input hashes are checked before using the trusted local NPZ object arrays. Existing R0/H0/G0 exports are read only.

Public outputs here are aggregate checks, nine-model metric summaries, configs and logs. Private per-prediction inference outputs, synthetic checkpoints and upload ZIPs stay under `outputs/next_stage_v4_recipe_20261101/18_mechanism_ready_20261104/`. The upload archive includes confidential development data and must not be committed. The original `extracted_r0/`, `extracted_h0/`, `overnight_ready/` are not overwritten.

The formal runner exists but this phase did not execute its 100-epoch path. It requires reviewed exact-config decisions, a source/data/backend-bound CUDA probe and historical initialization checks, protects nonempty foreign directories, recovers at epoch boundaries, preserves orphan snapshots and validates 100 epochs/2200 updates/20 checkpoints/620 epoch-threshold pairs before completion. Completed results are hash-checked before skipping. Its post-selection hook exports complete ranked AP and validation-bound model locks from the selected cached candidates; this hook passed a synthetic integration test. No AP metric is invented for not-started experiments.
