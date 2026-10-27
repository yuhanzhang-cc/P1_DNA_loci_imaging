# deepblink_pytorch

PyTorch migration of deepBlink core functionality.

## Quick start

```bash
pip install -e .
```

## CLI

```bash
deepblink-torch create --images ./images --labels ./labels --output ./dataset.npz
deepblink-torch train --data ./dataset.npz --epochs 50 --batch-size 16 --lr 1e-3 --device cuda
deepblink-torch predict --model ./artifacts/checkpoints/best_model.pt --images ./images --output ./predictions --probability 0.5 --device cuda --output-unit px
deepblink-torch overlay --images ./images --csv ./predictions --output ./overlays --mode side-by-side
deepblink-torch test --data ./dataset.npz --images ./images --model ./artifacts/checkpoints/best_model.pt --output ./evaluation --subset test --probability-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --match-radius-px 3.0 --device cuda --overlay-mode side-by-side
```

`predict` defaults to pixel coordinates and writes CSV columns as
`y [px], x [px], p`. Use `--output-unit micron` when you want physical units;
that writes `y [um], x [um], p`. Each prediction CSV also gets a matching
`.meta.json` sidecar with the source image file name and model checkpoint file
name.

`overlay` renders PNG previews from image files plus pixel-coordinate CSVs
exported by `predict`. When the prediction sidecar is present, each PNG footer
includes the source image file name and model checkpoint file name.
Use `--mode single` for a red-point overlay only, or keep the default
`side-by-side` for original image plus overlay comparison.

`test` evaluates a split directly from the `.npz` archive and writes a tidy
report tree:

```text
evaluation/
  summary/metrics_summary.csv
  summary/metrics_summary.json
  summary/per_image_metrics.csv
  summary/threshold_metrics.csv  # only when --probability-sweep is provided
  predictions/*.csv
  matches/*.csv
  overlays/*.png
  match_overlays/*.png
  figures/pr_curve.png
  figures/per_image_f1_boxplot.png
  figures/f1_curve.png      # only with --verbose
  figures/error_cdf.png      # only with --verbose
  figures/localization_error_vs_confidence.png  # only with --verbose
```

The console output highlights the numbers that are usually the first readout:
`precision`, `recall`, `F1`, `median_error_px`, and `mean_f1_integral`, plus
the total `tp`, `fp`, and `fn` counts. The command also uses the split metadata
stored in the `.npz` file, so the per-image outputs keep the original
`sample_id` / `image_name` mapping visible.
When `--probability-sweep` is provided without an explicit `--probability`,
`test` uses the sweep threshold with the highest F1 that does not reduce
precision below the p=0.5 reference point for `predictions/`, `matches/`, and
overlays. If `--probability` is provided, that value remains the output
threshold and the sweep is reported only as calibration metadata. The
`threshold_metrics.csv` table marks the threshold used for outputs with
`selected_for_outputs`.
In addition to prediction-only overlays, `test` writes TP/FP/FN match overlays:
TP is green, FP is red, and FN is blue.

You can also run training directly:

```bash
python train.py --data ./dataset.npz --epochs 50 --batch-size 16 --lr 1e-3 --device cuda
```

## Mixed-size clean220 experiment

The current version-training protocol pools all 220 samples before a new random
176/22/22 split, without using old/new origin to assign subsets. From the repo
root run `python deepblink_pytorch/scripts/run_clean220_random.py` in the isolated
XPU environment. The output is `outputs/261024_clean220_random/`. All three seeds
train from scratch on the same frozen split. Checkpoints and thresholds are
selected on validation before testing. Historical113 scores retain their own
original test split and are not a matched-test estimate of data-size effects.
Use `--prepare-only` to inspect the new manifest before training.

For a same-holdout data-expansion comparison, run
`python deepblink_pytorch/scripts/run_clean220_control.py` in that same isolated
XPU environment. It trains only the 89 original-batch images within the frozen
176-image training split and reuses the completed 176-image arm after checking
input hashes, core-code hashes and training arguments. Both arms share the exact
22-image validation and 22-image test arrays, three seeds and 100-epoch recipe.
The output is `outputs/261025_clean220_control/`. The primary endpoint is the
mean paired-seed micro-F1 difference, with a paired image bootstrap interval.
Equal epochs mean different optimizer-update counts (1200 versus 2200). The
test was already observed; this is an internal controlled comparison, not a new
blind test or a pure sample-count experiment. `--prepare-only` audits without
starting training; partial runs require investigation before restarting.

The earlier protocol below is retained for historical reproducibility:

From the repository root, run `python deepblink_pytorch/scripts/run_clean220.py`.
This audits 220 paired samples, excludes `sample_230` and `sample_286`, preserves
the old113 split, and freezes a size-stratified new107 split before training.
The output is `outputs/261019_clean220/`; raw images and CSVs are never changed.
Use `--prepare-only` to audit and create the archives without training.

The new archive stores original image shapes and a fixed `input_size=512`.
Images are normalized using original pixels, then zero-padded on the right and
bottom. Pixel coordinates are unchanged. Artificial cells use occupancy `-1`
and are excluded from all training losses and validation classification metrics.
Original, unencoded coordinates remain the evaluation truth, including grid
collisions. TrackMate descriptor rows distinguish pixel from micron CSVs.

New checkpoints store the input size so prediction and evaluation use the same
preprocessing and discard coordinates outside the original image. Checkpoints
without that metadata retain the legacy native-size/reflect-padding behavior.
`create --input-size 512 --exclude-samples sample_230,sample_286
--split-manifest frozen_split.csv` exposes the preparation options independently.

The experiment trains three seeds for 100 epochs, selects checkpoints by minimum
validation loss, and locks thresholds on validation before testing. Old12 and
new20 test results are reported separately. The frozen historical checkpoint is
evaluated on the same held-out images for comparison. Historical test results
are regression checks because they were used in earlier development.

`--device xpu` supports Intel GPUs when an XPU PyTorch build is installed.
The optional `requirements-xpu.txt` is for an isolated environment; do not
replace the team's shared environment. The experiment runner auto-selects an
available CUDA/XPU device, otherwise CPU. Training progress CSVs are written
after every epoch, and completed seeds are not trained again on rerun.

## Frozen clean220 delivery and first architecture screen

See [the clean220 release index](../docs/clean220_release_20261026/README.md)
for checkpoint locations, artifact hashes, inference and archive reconstruction.
The pooled220 three-seed model and the old89 matched-holdout control are complete.
The subsequent v2 screen uses train/validation only and four 100-epoch seed42
prototypes: geometric masked-SE, a local occupancy head, radial image evidence
and learned feature evidence. None exceeds the same-seed DeepBlink reference
under the registered v2 validation calibration. These are retained negative
results, not a completed architecture contribution or a new test-set benchmark.

The v2 runner is `scripts/optimize_v2.py`; `--development-only` in the train CLI
loads no test arrays. Prediction/evaluation load prototype types from checkpoint
metadata and apply geometry-only masks when required. The point matcher maximizes
the number of in-radius matches before minimizing distance, and decoding retains
the source cell's probability at offset boundaries. The release keeps legacy
reports unchanged and labels their evaluation rules separately.

Checkpoint metadata does not override the prediction CLI's default probability.
Pass the intended calibrated `--probability` explicitly. At the frozen release
commit `9edd529`, v3 selection, detail-routing and adaptive-context designs were
scope documents only.

## V3 optimization on `codex/clean220-round2`

The current branch implements the bounded v3 screen with
`scripts/optimize_v3.py`. The general train CLI continues to use its existing
training workflow; use this dedicated runner for the registered v3 experiments.
The [execution record](../docs/next_stage_plan_20261025/round2_20261026/execution.md)
links the actual protocol lock, preflight checks and live controller state.

The bounded XPU job completed all twelve 100-epoch model/seed runs, including
the conditional R2/S2 screen and R0/R2/S2 seed extensions. The controller passed
final verification and exited. R2 improved on R0 in one of three paired seeds;
its mean validation F1 was below simple control S2. Stable core-method gain
has not been established. See the
[2026-10-26 report](../docs/next_stage_plan_20261025/round2_20261026/daily_report_20261026.md)
for results and next-step limits. Do not restart the completed controller.

Completed trials, checkpoints, candidate predictions, curves and reports live
under `outputs/next_stage_v3/`. Training and selection read train/validation
arrays only. All scores from this screen are development results, and model
novelty and blinded annotation review remain unconfirmed.
