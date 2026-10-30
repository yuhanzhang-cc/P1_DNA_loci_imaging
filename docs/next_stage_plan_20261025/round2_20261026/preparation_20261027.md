# Closing experiments: local preparation and verification

Status (2026-10-27): G0 and Q-global implemented; native Spotiflow CPU environment
and bridge verified. No full control runs, biological fine-tuning, or test-set
inference started in this preparation task. Smoke results establish engineering
readiness only, not improved detection performance.

## Implemented controls

`deepblink_pytorch/deepblink_torch/networks/closing_controls.py` adds:

| Control | Input to added branch | Total parameters | Added to R0 |
|---|---|---:|---:|
| G0 (`deeploci_grid_control`) | Grid F4 only, 128->16->10 projection and common residual head | 1,265,504 | 6,781 |
| Q-global (`deeploci_global_query`) | Original R1 tokens, geometry-masked global F4 query | 1,265,350 | 6,627 |
| R1 reference | Original R1 tokens, per-cell F4 query | 1,265,350 | 6,627 |

G0 differs by 154 parameters (2.3% of R1's added branch), with no inactive dummy
weights. Q-global has the same state-dict layout and parameter count as R1;
only query input changes. Both preserve zero-initialized residual equivalence
to R0 and require image geometry masks. The existing v3 architecture classes
and training scripts remain unchanged; prediction CLI has two new model names.

## Data and evaluation

`scripts/export_spotiflow_dev.py` exports 176 train + 22 validation pairs from
the frozen input archive, without requesting test arrays. It crops archive
pixels to the original 256/512 shape, retains original intensity values, and
writes same-name TIFF/CSV with pixel `y,x` coordinates. Spotiflow applies its
native percentile normalization; our local models keep valid-pixel z-score
then right/bottom zero padding. No raw files are modified.

All 198 exports passed pixel-exact TIFF and float32-exact coordinate round trips.
The actual Spotiflow reader also verified every image and label. The local
export lives under ignored `outputs/closing_preparation/spotiflow_data/` and
must not be committed or uploaded as part of code delivery.

`scripts/evaluate_spotiflow.py` provides separate `infer` and `evaluate` commands.
Inference uses native peak detection/subpixel flow at probability threshold 0,
without a 0.5 prefilter. Evaluation reuses v3's maximum-cardinality matching at
3 pixels and identical 31 thresholds. It selects by micro F1, precision,
earlier epoch, then higher threshold. Fine-tuned snapshots are restricted to
epochs 5,10,...,100; zero-shot uses epoch 0. Runs with different IDs, seeds,
devices, or pretrained status cannot be combined. Candidate hashes, sample
order, geometry and completion status are checked before evaluation.

A partial checkpoint list is explicitly labeled as partial budget; it must not
be presented as a full 20-checkpoint comparison. Zero-shot calibration is a
validation-calibrated reference, not an untouched test result. The formal
Spotiflow training callback that saves all 20 epoch snapshots is still a
separate launch-time requirement; native `best` alone does not meet that budget.

## Environments and commands

The shared XPU environment was not changed. New `.venv/spotiflow_cpu` is a
Python 3.12 venv with `include-system-site-packages=false`, torch 2.11.0+cpu,
torchvision 0.26.0+cpu and Spotiflow 0.6.5. Pip dependency check passes.
`requirements-spotiflow.txt` constrains the matched torch/torchvision pair to
prevent an unplanned runtime upgrade. The separate CPU Windows lock records
the resolved packages; do not apply that lock to a Linux CUDA instance.

From repository root (use a fresh output name when rerunning):

```powershell
.venv\clean220_xpu\Scripts\python.exe -m pytest deepblink_pytorch/tests/test_closing_preparation.py deepblink_pytorch/tests/test_round2_architectures.py deepblink_pytorch/tests/test_round2_protocol.py -q
.venv\clean220_xpu\Scripts\python.exe deepblink_pytorch/scripts/preflight_closing.py --output outputs/closing_preparation/control_smoke_new
.venv\clean220_xpu\Scripts\python.exe deepblink_pytorch/scripts/export_spotiflow_dev.py --output outputs/closing_preparation/spotiflow_data_new
.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/smoke_spotiflow.py --output outputs/closing_preparation/spotiflow_smoke_new
```

The isolated baseline environment was installed using official CPU torch and
torchvision wheels, then `pip install -r deepblink_pytorch/requirements-spotiflow.txt`
and `pip install --no-deps --no-build-isolation -e deepblink_pytorch`.
Model downloads are cached inside ignored `.venv/spotiflow_models/`.

Example baseline evaluation after formal snapshot preparation:

```powershell
.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/evaluate_spotiflow.py infer --pretrained general --epoch 0 --run-id general_zeroshot --device cpu --output outputs/closing_preparation/general_zeroshot
.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/evaluate_spotiflow.py evaluate --runs outputs/closing_preparation/general_zeroshot --output outputs/closing_preparation/general_zeroshot_eval
```

These two example commands were not run for a new scientific comparison in
this preparation task. For fine-tuned inference replace `--pretrained general`
with `--model-folder PATH`, add `--seed`, and retain the same `--run-id` across
the 20 snapshots of one seed. Different seeds must be evaluated separately.

## Validation evidence and repaired issue

- 45 targeted tests passed, including old v3 tests, parameter counts, query
  geometry, gradients, model reload, candidate matching and failure logging.
- `outputs/closing_preparation/control_smoke_v2/`: actual XPU bf16 forward/backward
  for G0 and Q-global, two steps each, using training images `sample_204` (256)
  and `sample_100` (512). Both weights reload exactly on CPU. Two-step losses
  are engineering diagnostics and do not establish convergence.
- `outputs/closing_preparation/spotiflow_smoke_v2/`: native general pretrained
  inference on two training images, followed by one synthetic CPU training
  epoch and native checkpoint reload. No biological-image fine-tuning.
- Spotiflow 0.6.5's checkpoint callback dereferences the logger when logging is
  disabled; first smoke failed. Using a local TensorBoard logger resolved this
  without patching third-party packages. The failed run and traceback remain
  under `spotiflow_smoke/`; successful retry is `spotiflow_smoke_v2/`.
- `ExperimentLog` creates new output directories exclusively, writes atomic
  status plus flushed event logs, and preserves failure tracebacks.

## Before full training

Create a new protocol lock and bounded runner for G0/Q-global seeds 42/43/44,
reusing the exact v3 sampler/optimizer/100-epoch and 20x31 selection protocol.
New sources intentionally differ from old frozen hashes: do not bypass old
locks or write into old result directories. Regress the inherited backbone
and sample orders against archived R0/R1 before starting the six runs.

For Spotiflow fine-tuning: add the registered snapshot callback, freeze native
training/augmentation settings and pretrained-weight hashes, and measure a
representative full-size runtime pilot before choosing CPU versus rental CUDA.
Keep the six local controls on XPU for matched comparisons. A CUDA migration
of the core models would require same-backend reference runs.

Official API/format references checked for this implementation:
[data format](https://weigertlab.org/spotiflow/train.html),
[fine-tuning](https://weigertlab.org/spotiflow/finetune.html),
[prediction API](https://weigertlab.org/spotiflow/api.html).
