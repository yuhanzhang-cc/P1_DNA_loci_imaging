# clean220 pooled-random version

All 220 samples were pooled before random 176/22/22 assignment. Origin was not used.
Split seed: 20261024. Training seeds: [42, 43, 44].
Selected seed_42 by validation loss before test inference; probability 0.55.

| Seed | Epoch | P | R | Micro F1 | Macro F1 | TP | FP | FN | Median error px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed_42 | 94 | 0.8330 | 0.7828 | 0.8071 | 0.7808 | 429 | 86 | 119 | 0.3054 |
| seed_43 | 85 | 0.7769 | 0.7117 | 0.7429 | 0.7386 | 390 | 112 | 158 | 0.2978 |
| seed_44 | 98 | 0.8218 | 0.7153 | 0.7649 | 0.7539 | 392 | 85 | 156 | 0.2925 |

Three-seed F1: 0.7716 +/- 0.0327 (sample SD).

Historical 113 F1: 0.8488 on its own original 12-image test, with test-sweep threshold.
The new random test is different; numerical version differences do not isolate data-size effects.
The new test was not used for this round's fitting/checkpoint/threshold selection; some images were analyzed in earlier repository experiments.
These are image-held-out results, not independent biological/external validation.
SD across seeds describes training variation, not uncertainty due to the 22 test images.
Images are normalized on original pixels and padded right/bottom to 512. Padded cells are ignored by the loss.
Original truth coordinates are retained at evaluation, including any grid collisions.
Raw files and prior experiment folders are unchanged. See frozen_split.csv, data_audit.json, source_snapshot/ and run.log.
