# Error-context diagnostics

## Inputs

- Dataset: `D:\Projects\P1_DNA_loci_imaging\outputs\260628test\dna_clean_113.npz`
- Evaluation summary: `D:\Projects\P1_DNA_loci_imaging\outputs\260628test\evaluation_final`
- Model checkpoint: `D:\Projects\P1_DNA_loci_imaging\outputs\260628test\best_model_clean_113.pt`
- Optional image directory: `D:\Projects\P1_DNA_loci_imaging\data\raw\tif`
- Subset: `test`
- Baseline probability: `0.4`
- Match radius px: `3.0`

Command:

```bash
python deepblink_pytorch/scripts/error_context_diagnostics.py --data outputs\260628test\dna_clean_113.npz --evaluation outputs\260628test\evaluation_final --model outputs\260628test\best_model_clean_113.pt --output outputs\260629_error_context --subset test --patch-radius 10 --top-k-patches 12 --device cpu
```

## Outputs

- `error_context.csv`: one row per TP/FP/FN event at the baseline threshold.
- `postprocess_ablation.csv`: offline comparison of threshold-only, fixed Top-N, threshold+max-N, and validation-prior Top-N strategies.
- `hard_examples.csv`: prioritized examples for manual patch review.
- `ppt_takeaways.md`: English speaker-note text for slides.
- `figures/`: diagnostic plots and patch montages.
- `figures_ppt/`: black-background, English, presentation-ready figures.
- `patches/TP`, `patches/FP`, `patches/FN`: representative event patches.
- `patches/hard_examples`: patch images referenced by `hard_examples.csv`.

## Field Notes

Patch features are measured around the event anchor. TP/FP use the predicted coordinate;
FN uses the ground-truth coordinate. No nucleus/nucleolus/membrane class is assigned,
because no segmentation mask is provided.

Important fields:

- `confidence`: model probability for TP/FP; NaN for FN.
- `local_contrast`: center intensity minus local patch mean.
- `local_zscore`: local contrast divided by local patch std.
- `local_max_to_mean`: local max divided by local mean.
- `peak_sharpness`: local max minus local mean.
- `peak_component_size_px`: connected component size around the local max after thresholding at local mean + local std.
- `distance_to_border_px`: distance from event anchor to nearest image border.

## Diagnostic Summary

- Event counts: TP=174, FP=23, FN=39.
- Median confidence: TP=0.9999, FP=0.9796.
- Median local z-score: TP=3.7091, FP=2.6157, FN=1.6473.
- Median local contrast: TP=1383.7734, FP=171.1995, FN=521.1860.
- Median peak sharpness: TP=1882.7664, FP=344.5125, FN=974.1611.

Interpretation:

- FP mainly low confidence: `False`.
- FN mainly low local z-score relative to TP: `True`.
- FP high-z but lower-sharpness suspicious background proxy: `False`.
- Confidence filtering risk is high because FP confidence overlaps TP confidence.
- FN errors point toward dim-loci / low-SNR examples rather than a post-processing threshold problem.

## Post-processing Summary

- Baseline: baseline_threshold_from_evaluation, threshold=0.4, F1=0.8488, P=0.8832, R=0.8169, TP=174, FP=23, FN=39
- Best threshold-only: threshold_only, threshold=0.4, F1=0.8488, P=0.8832, R=0.8169, TP=174, FP=23, FN=39
- Precision-constrained threshold-only: threshold_only, threshold=0.4, F1=0.8488, P=0.8832, R=0.8169, TP=174, FP=23, FN=39
- Best fixed Top-N: fixed_top_n, top_n=16, F1=0.7506, P=0.7917, R=0.7136, TP=152, FP=40, FN=61
- Best threshold+max-N hybrid: threshold_max_n, threshold=0.4, max_n=40, F1=0.8488, P=0.8832, R=0.8169, TP=174, FP=23, FN=39
- Best validation-prior Top-N: validation_prior_top_n, top_n=17, F1=0.7578, P=0.7745, R=0.7418, TP=158, FP=46, FN=55

Conclusion:

Do not change the default inference strategy yet; use dice_rmse + current threshold sweep as default, and move to hard-negative mining / augmentation / context-aware model modification.

Explicit default-inference decision:

- Do not change default inference.
- Fixed Top-N is rejected because spot count varies strongly by image and high-count images become recall-capped.
- Confidence filtering alone is risky because high-confidence FP events overlap the TP confidence range.
- Threshold+Top-N hybrid does not improve over the baseline.
- This branch is offline diagnostics only and does not modify default training/evaluation/prediction behavior.

Rejected-strategy rules applied:

- Reject if F1 is below baseline.
- Reject if recall drops by more than about 0.02.
- Reject if FP decreases only slightly while TP loss is substantial.
- Reject fixed Top-N if high-count images are visibly recall-capped.

## Recommended Next Step

Use this output for manual FP/FN patch review. If FP patches are high-confidence structured
background, prioritize hard-negative mining or a lightweight local-context classifier.
If FN patches are dim/low-z, prioritize augmentation/sampling for dim loci. Do not infer
specific biological structures without masks.

Recommended merge-safe next work:

- Curate hard-positive and hard-negative patch sets from `hard_examples.csv`.
- Test dim-loci augmentation and hard-negative sampling before architecture changes.
- Keep these diagnostics out of the default inference pipeline until a strategy beats baseline without hurting recall.
