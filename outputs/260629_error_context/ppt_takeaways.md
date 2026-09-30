# PPT Takeaways

## One-sentence conclusion

Error-context diagnostics suggest that the remaining errors are not solved by confidence filtering; false negatives are enriched for low-SNR loci, while false positives often remain high-confidence.

## Key findings

- The current default remains the best choice: dice_rmse with threshold sweep at the baseline operating point.
- FP confidence overlaps TP confidence: median TP confidence = 0.9999, median FP confidence = 0.9796.
- FN events are enriched for low local z-score and lower local contrast: median local z-score TP = 3.71, FP = 2.62, FN = 1.65; median local contrast TP = 1384, FP = 171, FN = 521.
- Fixed Top-N and validation-prior Top-N are rejected because they recall-cap high-count images.
- Threshold+Top-N hybrid does not improve over the baseline.

## Recommended next technical step

- Do not change default inference yet.
- Build curated hard-positive and hard-negative patch sets from `hard_examples.csv`.
- Test dim-loci augmentation and hard-negative sampling before architecture modification.
- Use patch review to decide whether FP examples reflect structured background, annotation ambiguity, crowding, or artifacts; do not assign biological structure labels without masks.
