# Results

## Frozen development benchmark

The common evaluation uses the same 22 validation images, without retraining references or refitting thresholds. LoG and DoG achieve micro-F1 values of {{LoG_f1}} and {{DoG_f1}}, respectively. The matched CUDA references achieve {{R0_f1}} (R0), {{H0_f1}} (H0) and {{G0_f1}} (G0). Adapted Spotiflow achieves {{Spotiflow_F_f1}} under its native transfer recipe. These values are development estimates after validation-based selection. Differences between method families do not by themselves isolate architecture effects. Full ranked AP values are {{G0_ranked_ap}} for G0 and {{Spotiflow_F_ranked_ap}} for adapted Spotiflow; the source tables retain all methods and seeds.

## Localization motivates a controlled mechanism test

The completed H0 diagnostic identified an approximately −0.23-pixel signed shift on both axes and additional unmatched predictions concentrated near reference spots. Such geometry-derived categories are not validated biological false-positive labels. H0's average localization error on its own matched TPs is {{H0_mean_error_px}}, compared with {{G0_mean_error_px}} for G0; common-TP comparisons are reported separately in Figure5. This motivated testing signed offsets and supervision support while preserving capacity. The blind review adjudicated {{blind_audit_summary}}. Its enriched sampling prevents extrapolating the observed patch fraction to every image or every predicted spot.

## Prospective sequential ablation

S0 yields {{S0_f1}} micro-F1 and {{S0_mean_error_px}} matched-TP error; P0 yields {{P0_f1}} and {{P0_mean_error_px}}, respectively. The P0−S0 common-TP error difference is {{p0_s0_common_tp_error}}, and the near-reference unmatched-prediction difference is {{p0_s0_near_truth_fp_difference}}. The seed42 gate result is {{screen_gate_result}}; the completed prospective training count is {{prospective_completed_training_count}}. Negative, partial and stopped runs remain visible in the experiment ledger. P0's joint support/ownership/normalization effect is not described as a separate effect for each ingredient.

## External reference and pretraining control

The Spotiflow scratch control yields {{Spotiflow_scratch_f1}} micro-F1 and {{Spotiflow_scratch_ranked_ap}} full ranked AP. The scratch-versus-transfer comparison is {{scratch_transfer_interpretation}}. Neither this native-recipe comparison nor a higher development score establishes that our architecture is generally superior. Device time, updates and sample exposure are recorded separately; no speed ranking is inferred from incomparable devices or cached execution times.

## Independent evaluation

Independent evaluation remains {{independent_evaluation_status}}. The dataset includes {{independent_biological_groups}} reviewed biological groups and {{independent_image_count}} images. Frozen-threshold F1 differences, paired biological-group intervals, weak-reference recall and common-TP localization are {{independent_results_summary}}. The historically observed 22-image internal test is not used as an independent confirmatory substitute.
