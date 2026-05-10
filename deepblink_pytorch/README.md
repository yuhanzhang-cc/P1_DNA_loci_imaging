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
deepblink-torch test --data ./dataset.npz --images ./images --model ./artifacts/checkpoints/best_model.pt --output ./evaluation --subset test --probability 0.5 --match-radius-px 3.0 --device cuda --overlay-mode side-by-side
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
  predictions/*.csv
  matches/*.csv
  overlays/*.png
  figures/f1_curve.png
  figures/error_cdf.png
```

The console output highlights the numbers that are usually the first readout:
`precision`, `recall`, `F1`, `median_error_px`, and `mean_f1_integral`, plus
the total `tp`, `fp`, and `fn` counts. The command also uses the split metadata
stored in the `.npz` file, so the per-image outputs keep the original
`sample_id` / `image_name` mapping visible.

You can also run training directly:

```bash
python train.py --data ./dataset.npz --epochs 50 --batch-size 16 --lr 1e-3 --device cuda
```
