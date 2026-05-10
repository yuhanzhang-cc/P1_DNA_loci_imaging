# thread0510_2 summary

## Current state

The deepblink_torch chain is now logically complete:

`create -> train -> predict -> overlay -> test`

Key conventions are aligned:

- `create` writes train/valid/test splits into `.npz`
- `create` also writes reversible split metadata: `sample_id_*`, `image_name_*`, `label_name_*`, `source_index_*`
- `predict` defaults to pixel coordinates and writes `y [px], x [px], p`
- `predict` can switch to microns only when explicitly asked with `--output-unit micron`
- `predict` writes `.meta.json` sidecars with image/model file names
- `overlay` consumes pixel-coordinate CSVs and renders PNGs with matplotlib only
- `test` reads the `.npz` split metadata, re-runs inference, matches predictions to truth, and writes metrics + QC outputs

## Important files

- `deepblink_pytorch/deepblink_torch/cli/create.py`
- `deepblink_pytorch/deepblink_torch/cli/predict.py`
- `deepblink_pytorch/deepblink_torch/cli/overlay.py`
- `deepblink_pytorch/deepblink_torch/cli/evaluate.py`
- `deepblink_pytorch/deepblink_torch/cli/main.py`
- `deepblink_pytorch/deepblink_torch/evaluation.py`
- `deepblink_pytorch/deepblink_torch/datasets/spots.py`
- `deepblink_pytorch/README.md`

## What the chain does

### 1) `create`

Purpose:
Convert raw microscopy images plus TrackMate-style CSV labels into a deepBlink-style `.npz`.

Meaning:
This prepares supervised training data. Biologically, the labels are spot coordinates for loci/local puncta in fluorescence images.

Input:

```powershell
deepblink-torch create --images ./images --labels ./labels --output ./dataset.npz
```

Normal output:

- `dataset.npz` exists
- `x_train/y_train`, `x_valid/y_valid`, `x_test/y_test` exist
- split metadata arrays exist for each subset
- a sidecar `dataset.split.csv` exists
- console prints counts for train/valid/test and the seed

What is normal:

- image/label names are preserved through `sample_id_*`, `image_name_*`, `label_name_*`
- train code still consumes the standard arrays without caring about extra metadata

### 2) `train`

Purpose:
Fit `UNetDeepBlink` on the train split and validate on the valid split.

Meaning:
The model learns to detect spot-like loci positions from annotated examples.

Input:

```powershell
deepblink-torch train --data ./dataset.npz --epochs 50 --batch-size 8 --device cuda
```

Normal output:

- checkpoint saved, usually `artifacts/checkpoints/best_model.pt`
- epoch logs show `train_loss`, `val_loss`, `val_dice_loss`, `val_rmse_loss`
- best checkpoint is updated when validation loss improves

What is normal:

- training only uses train/valid
- test split is not consumed during training

### 3) `predict`

Purpose:
Run inference on new images and export coordinate CSVs.

Meaning:
This produces the model’s spot predictions in image pixel space by default.

Input:

```powershell
deepblink-torch predict --model ./artifacts/checkpoints/best_model.pt --images ./images --output ./predictions --probability 0.5 --device cuda --output-unit px
```

Normal output:

- CSV files written per image
- default columns are `y [px], x [px], p`
- if `--output-unit micron`, columns become `y [um], x [um], p`
- `.meta.json` sidecar written next to each CSV

What is normal:

- overlay should consume the pixel CSVs
- micron CSVs are only for explicit physical-unit export, not for overlay

### 4) `overlay`

Purpose:
Visual QC for predictions.

Meaning:
Lets a human quickly inspect whether predicted loci land on the right image structures.

Input:

```powershell
deepblink-torch overlay --images ./images --csv ./predictions --output ./overlays --mode side-by-side
```

Normal output:

- PNG files written per image
- default mode is `side-by-side`
- left panel is the original image
- right panel is the image with red predicted points
- footer includes image file name and model file name when metadata is present

What is normal:

- CSV input must be pixel coordinates from `predict`
- no ImageJ dependency
- `single` mode is a red-point-only view

### 5) `test`

Purpose:
Evaluate the held-out split with precision/recall/F1 and visual summaries.

Meaning:
This is the reporting layer for model performance on the test set, not a training step.

Input:

```powershell
deepblink-torch test --data ./dataset.npz --images ./images --model ./artifacts/checkpoints/best_model.pt --output ./evaluation --subset test --probability 0.5 --match-radius-px 3.0 --device cuda --overlay-mode side-by-side
```

Normal output tree:

- `summary/metrics_summary.csv`
- `summary/metrics_summary.json`
- `summary/per_image_metrics.csv`
- `predictions/*.csv`
- `matches/*.csv`
- `overlays/*.png`
- `figures/f1_curve.png`
- `figures/error_cdf.png`

Console readout prioritized:

- `precision`
- `recall`
- `F1`
- `median_error_px`
- `mean_f1_integral`
- `tp`, `fp`, `fn`

What is normal:

- it uses the `.npz` split metadata to recover `sample_id` / `image_name`
- it uses raw images from the supplied `--images` directory
- it evaluates pixel-space coordinates

## Known constraints

- Full runtime verification in the Codex shell was limited because the environment did not have `pytest` and `torch` available in the active `python`.
- The codebase now expects `scipy` as a runtime dependency because evaluation uses Hungarian matching.
- `test` assumes the raw image folder still contains the original image names recorded in the `.npz` metadata.

## Next-thread recommendation

Start by reading:

1. this file
2. `deepblink_pytorch/README.md`
3. `deepblink_pytorch/deepblink_torch/cli/evaluate.py`
4. `deepblink_pytorch/deepblink_torch/cli/predict.py`
5. `deepblink_pytorch/deepblink_torch/cli/create.py`

That is enough to continue debugging, extending, or validating the full chain without re-discovering the conventions.
