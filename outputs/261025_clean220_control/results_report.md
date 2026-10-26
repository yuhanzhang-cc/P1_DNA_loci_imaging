# Same-holdout data expansion comparison

Old-only: 89 training images. Expanded: 176 (same 89 plus 87 added). Shared validation/test: 22/22.
Same code, XPU environment, architecture, loss, batch size, learning rate, 100 epochs and paired seeds.
100 epochs entail 1200 vs 2200 optimizer steps: this is a same-recipe expansion comparison.
The expanded arm is reused after verifying raw hashes, core-code hashes and training arguments.
The test was previously observed; this is internal controlled evidence, not independent blind validation.

| Arm | Seed | Best epoch | Threshold | P | R | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_only_89 | 42 | 83 | 0.55 | 0.8172 | 0.5712 | 0.6724 | 313 | 70 | 235 |
| old_only_89 | 43 | 37 | 0.90 | 0.8411 | 0.4635 | 0.5976 | 254 | 48 | 294 |
| old_only_89 | 44 | 71 | 0.50 | 0.7732 | 0.5785 | 0.6618 | 317 | 93 | 231 |
| expanded_176 | 42 | 94 | 0.55 | 0.8330 | 0.7828 | 0.8071 | 429 | 86 | 119 |
| expanded_176 | 43 | 85 | 0.70 | 0.7769 | 0.7117 | 0.7429 | 390 | 112 | 158 |
| expanded_176 | 44 | 98 | 0.65 | 0.8218 | 0.7153 | 0.7649 | 392 | 85 | 156 |

Across-seed summaries: {"old_only_89": {"precision": {"mean": 0.8104875701118118, "sd": 0.03444334484446914}, "recall": {"mean": 0.537712895377129, "sd": 0.06437063733650092}, "f1": {"mean": 0.6439459132735612, "sd": 0.04044474300311725}, "macro_f1": {"mean": 0.6391400694847107, "sd": 0.04356149371146594}, "median_error_px": {"mean": 0.3632434904575348, "sd": 0.038422054276361625}}, "expanded_176": {"precision": {"mean": 0.8105683580090769, "sd": 0.029697626600640042}, "recall": {"mean": 0.73661800486618, "sd": 0.04007680391734584}, "f1": {"mean": 0.7716282561024778, "sd": 0.03267343317675979}, "macro_f1": {"mean": 0.7577706376711527, "sd": 0.021339232595735103}, "median_error_px": {"mean": 0.29856664935747784, "sd": 0.00650763832939755}}}

Paired uncertainty: {"delta_mean_micro_f1": 0.12768234282891644, "percentile_95_ci": [0.0704422250095132, 0.18577667210886492], "n_resamples": 10000, "seed": 20261025, "n_images": 22, "n_fixed_training_seeds": 3, "method": "paired image-cluster percentile bootstrap; shared image resamples across seeds and arms", "limitation": "Conditional on trained models and one already observed split; image exchangeability assumed, biological/batch independence not established; no p-value."}

No test-based threshold or model selection. No outlier deletion. No biological independence claim.
Image bootstrap conditions on these three fitted seed pairs and assumes image exchangeability.
The training set changes in size, origin and size composition; this does not isolate pure sample count.
Historical 113 published F1 is deliberately not used as the control endpoint.
