# Validation and protocol notes

- 15 targeted tests passed before training: pooled splitting, preprocessing,
  geometry/unit conversion, padding gradients, checkpoint loading and training CLI.
- Split tests confirm deterministic 176/22/22 assignments, unchanged when input
  inventory order, prior subset assignments or origin labels are altered.
- The random split seed (20261024) was fixed before model fitting. Training seeds
  42/43/44 all use the same new split, with fresh model and optimizer initialization.
- 440 raw file hashes matched the previously audited inventory. No raw files,
  old checkpoints or earlier experiment outputs were changed.
- All 220 source file/series groups are unique in the available logs. This check
  does not establish independent biological samples or acquisition batches.
- Six single-grid label collisions remain, all in the new training subset.
  No labels were removed from the coordinate truth used by evaluation.
- Checkpoints are selected by validation loss, thresholds on validation only;
  all threshold/model choices are locked before any new test inference.
- Historical113 metrics remain on its own historical test. Its weights are not
  scored on the new random test, which may contain its former training images.
- Test scores on different splits provide version-specific results; they do not
  isolate the causal effect of increasing the training set size.
- Existing legacy tests requiring missing TF CSVs or excluded historical raw
  fixtures were documented in the earlier experiment and were not restored.
