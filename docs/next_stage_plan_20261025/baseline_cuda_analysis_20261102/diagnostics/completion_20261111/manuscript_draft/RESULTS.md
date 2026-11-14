# Results

## Frozen development benchmark

The common evaluation uses the same 22 validation images, without retraining references or refitting thresholds. LoG and DoG achieve micro-F1 values of 0.5724 (deterministic) and 0.6676 (deterministic), respectively. The matched CUDA references achieve 0.7887 ± 0.0108 (R0), 0.8088 ± 0.0022 (H0) and 0.8197 ± 0.0071 (G0). Adapted Spotiflow achieves 0.8606 ± 0.0034 under its native transfer recipe. These values are development estimates after validation-based selection. Differences between method families do not by themselves isolate architecture effects. Full ranked AP values are 0.8606 ± 0.0151 for G0 and 0.9126 ± 0.0029 for adapted Spotiflow; the source tables retain all methods and seeds.

## Localization motivates a controlled mechanism test

The completed H0 diagnostic identified an approximately −0.23-pixel signed shift on both axes and additional unmatched predictions concentrated near reference spots. Such geometry-derived categories are not validated biological false-positive labels. H0's average localization error on its own matched TPs is 0.6758 ± 0.0440, compared with 0.4463 ± 0.0169 for G0; common-TP comparisons are reported separately in Figure5. This motivated testing signed offsets and supervision support while preserving capacity. The blind review adjudicated 60 completed candidate-presence reviews; yes=41, no=12, uncertain=7, low=11; assignment TP/FP remains a separate question. Its enriched sampling prevents extrapolating the observed patch fraction to every image or every predicted spot.

## Prospective sequential ablation

S0 yields [PENDING: S0_f1] micro-F1 and [PENDING: S0_mean_error_px] matched-TP error; P0 yields [PENDING: P0_f1] and [PENDING: P0_mean_error_px], respectively. The P0−S0 common-TP error difference is [PENDING: p0_s0_common_tp_error], and the near-reference unmatched-prediction difference is [PENDING: p0_s0_near_truth_fp_difference]. The seed42 gate result is [PENDING: screen_gate_result]; the completed prospective training count is 0. Negative, partial and stopped runs remain visible in the experiment ledger. P0's joint support/ownership/normalization effect is not described as a separate effect for each ingredient.

## External reference and pretraining control

The Spotiflow scratch control yields [PENDING: Spotiflow_scratch_f1] micro-F1 and [PENDING: Spotiflow_scratch_ranked_ap] full ranked AP. The scratch-versus-transfer comparison is [PENDING: scratch_transfer_interpretation]. Neither this native-recipe comparison nor a higher development score establishes that our architecture is generally superior. Device time, updates and sample exposure are recorded separately; no speed ranking is inferred from incomparable devices or cached execution times.

## Independent evaluation

Independent evaluation remains [PENDING: independent_evaluation_status]. The dataset includes [PENDING: independent_biological_groups] reviewed biological groups and [PENDING: independent_image_count] images. Frozen-threshold F1 differences, paired biological-group intervals, weak-reference recall and common-TP localization are [PENDING: independent_results_summary]. The historically observed 22-image internal test is not used as an independent confirmatory substitute.


### Automatic outcome status

partial_mechanism_results_keep_G0. The current practical candidate remains G0 pending review of the prospective mechanism results and independent evidence. No superiority claim is generated automatically.
