# clean220 training results

Raw files were not modified. Excluded sample_230 and sample_286.
220 images / 5076 points. Train 166, validation 22, test 32 (historical 12 + new 20).
Fixed 512 canvas; normalize original pixels, zero-pad right/bottom, ignore padded loss cells.
Architecture and loss retained: UNetDeepBlink (1,258,723 parameters), Dice + 2*RMSE.
Selected seed_42, epoch 71, by minimum validation loss only.
Validation-locked probability: 0.55.

| Model | Test cohort | Precision | Recall | F1 | TP | FP | FN | Median error px |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seed_42 | old113 | 0.8643 | 0.8075 | 0.8350 | 172 | 27 | 41 | 0.2353 |
| seed_42 | new107 | 0.9222 | 0.7124 | 0.8039 | 332 | 28 | 134 | 0.3677 |
| seed_43 | old113 | 0.8967 | 0.7746 | 0.8312 | 165 | 19 | 48 | 0.2502 |
| seed_43 | new107 | 0.8934 | 0.7017 | 0.7861 | 327 | 39 | 139 | 0.3656 |
| seed_44 | old113 | 0.8836 | 0.7840 | 0.8308 | 167 | 22 | 46 | 0.2283 |
| seed_44 | new107 | 0.8830 | 0.6481 | 0.7475 | 302 | 40 | 164 | 0.3524 |
| historical_baseline | old113 | 0.8827 | 0.8122 | 0.8460 | 173 | 23 | 40 | 0.2117 |
| historical_baseline | new107 | 0.9259 | 0.3219 | 0.4777 | 150 | 12 | 316 | 0.4723 |
| historical_fixed_canvas | old113 | 0.8832 | 0.8169 | 0.8488 | 174 | 23 | 39 | 0.2103 |
| historical_fixed_canvas | new107 | 0.9441 | 0.3262 | 0.4848 | 152 | 9 | 314 | 0.4751 |

Replicate summaries:
```json
{
  "old113": {
    "mean_f1": 0.8323438281273035,
    "sd_f1": 0.0022666106542342584,
    "min_f1": 0.8308457711442785,
    "max_f1": 0.8349514563106796
  },
  "new107": {
    "mean_f1": 0.779152178930875,
    "sd_f1": 0.028802371477708037,
    "min_f1": 0.7475247524752475,
    "max_f1": 0.8038740920096853
  }
}
```

Interpretation boundaries:
- Old12 is a reused historical regression set, not blind validation.
- New20 is image-held-out, not an external biological cohort. Acquisition/batch independence is not fully established.
- Native historical inference and fixed-canvas historical inference use identical weights; their difference is preprocessing plus validation calibration.
- New training also changes data coverage, masking and numerical backend. Performance differences alone do not isolate a biological mechanism or a causal contribution from data size.
- Localization errors are conditional on successful matches and must be interpreted with recall.
- Three-seed SD describes training variability, not uncertainty from limited test images.
- Six grid collisions remain in training; evaluation retains all original truth coordinates.

Artifacts: recommended_model.pt/json, frozen_split.csv, data_audit.json, experiment_config.json,
environment.json, source_snapshot/, run.log, per-seed training_metrics.csv/training_curves.png,
validation threshold tables, test predictions/matches/overlays, stratified_metrics.csv and review_overlays/.
