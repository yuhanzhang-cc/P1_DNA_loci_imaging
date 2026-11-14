# Offset-supervision alignment for DNA-locus detection in fluorescence microscopy

**Working title; prospective research manuscript, not a completed study.**

Reliable localization of fluorescent DNA loci requires both spot detection and subpixel coordinate estimation. We investigate whether the relationship between a predicted heatmap peak and the pixels receiving offset supervision limits localization in a small, curated microscopy benchmark. We assembled 220 image–annotation pairs and fixed a development partition of 176 training and 22 validation images; the remaining historical partition is excluded from new claims of independent validation. We compare direct grid-based and dense heatmap references with deterministic scale-space methods and a locally adapted external detector. A parameter-identical signed-offset control and a neighborhood-supervision model isolate two sequential changes while preserving the backbone and development selection budget. On development data, the retained simple candidate G0 achieved a micro-F1 of 0.8197 ± 0.0071, compared with 0.8606 ± 0.0034 for adapted Spotiflow under its native transfer recipe. The prospective mechanism comparison yielded [PENDING: prospective_mechanism_abstract_result]. Independent evaluation on [PENDING: independent_dataset_description] yielded [PENDING: independent_primary_result]. These experiments are designed to determine whether supervision alignment improves localization without increasing near-reference unmatched detections. Conclusions about mechanism benefit, novelty and generalization remain conditional on the pending experiments and annotation review.


# Introduction

Fluorescent locus imaging converts nuclear organization into a collection of point measurements. A detector must distinguish weak loci from background and nearby signals while assigning coordinates consistently. Detection errors affect counts and correspondence; coordinate errors affect spatial measurements even when an object is detected. The present study focuses on single-frame detection and localization. Tracking reliability, telomere splitting and downstream biological validity require additional experiments and are not inferred from single-frame F1.

Existing neural detectors provide strong starting points. deepBlink introduced neural detection and subpixel localization of diffraction-limited spots [1]. Spotiflow formulates detection using multiscale heatmaps and stereographic flow regression [2]. Their success motivates careful comparison with specialized spot detectors rather than relying exclusively on generic segmentation references. In a small local benchmark, retraining an existing backbone can improve domain fit, but this alone does not demonstrate a new mechanism or general advantage.

We examine a specific failure hypothesis: a heatmap maximum can occur away from the anchor at which a subpixel offset was supervised. A bounded offset also cannot represent every signed displacement from a neighboring source pixel. This suggests two separable engineering questions: whether signed output support changes behavior at the original anchor, and whether extending offset supervision around the anchor improves coordinates at the peaks actually decoded. The proposed study preserves the architecture's parameter count and evaluates a constrained sequence of controls. It tests supervision–decoding alignment, rather than assuming that a larger model or another attention module will improve the task.

The study has three intended contributions, subject to completion: an auditable DNA-locus benchmark workflow with immutable label versions; a parameter-controlled investigation of signed offsets and neighborhood supervision; and a common evaluation suite covering thresholded detection, full score-ranked AP, weak-reference recall and localization on common true positives. We will retain negative results and narrow the contribution if the mechanism fails to improve the prespecified endpoints. Claims of a new architecture, state-of-the-art accuracy or biological generalization require evidence beyond the current development benchmark.


# Methods

## Images, annotation and data separation

The curated collection contains 220 image–point-list pairs after excluding `sample_230` and `sample_286`. Development uses a fixed 176-image training set (4,156 reference points) and a 22-image validation set (372 points). Images have native dimensions of 256×256 or 512×512 pixels. Coordinates are expressed as `(y,x)` pixel-center indices in the native field of view. Training inputs are normalized on the original image and padded on the right and bottom to 512×512; padding is not new image information and is excluded from valid-field decoding.

Initial annotation used ImageJ/TrackMate-assisted point placement. **Author-supplied acquisition and annotation details remain required:** [PENDING: microscope_and_optics], [PENDING: sample_preparation_and_labeling], [PENDING: exposure_pixel_size_channels], [PENDING: annotator_number_and_rules], [PENDING: biological_and_acquisition_groups]. Two train/validation source containers are known to overlap; the biological relationship must be clarified rather than assumed independent. The historically observed internal test partition is not renamed as a new independent test. No additional test image is accessed by the preparation workflow.

## Blind annotation audit

A fixed 60-case image packet samples common true positives, H0-specific true positives and additional H0 unmatched predictions. Reviewers see raw and candidate-marked patches without model identity, seed or sampling stratum. They record whether the marked candidate is a real spot (yes/no/uncertain), visible spot count or uncertainty, and confidence. Candidate existence is not equivalent to a one-to-one evaluation true positive. The sampling is enriched for diagnostic cases; its fractions are not population error rates. Real review was completed and hash-bound to the approved experiment decision with [PENDING: blind_audit_reviewer_details]. Ambiguous cases are inspected in the original image before any separate annotation revision. Revised complete point lists receive a new version with reviewer, rationale and parent hashes; historical labels and metrics are preserved.

## Reference methods and comparison tracks

LoG and DoG are fixed scale-space implementations, each calibrated across 24 registered scale/suppression configurations and at most 31 response policies. Their threshold grids derive from training-image response quantiles and are selected on validation images. They are deterministic references, not three independently trained models and not claims of an official implementation of another paper.

R0 is the project's direct PyTorch grid-based reference. G0 is the retained simple candidate. H0 is a full-resolution heatmap/offset representation reference. The selected CUDA references use identical development images and the registered 100-epoch budget. External Spotiflow is evaluated both as the existing general-pretrained model adapted locally and, if approved and completed, as a randomly initialized control using its native training recipe. Native losses, crop sizes, initialization and optimization are documented rather than described as identical to the in-house models. Method-level accuracy comparisons are distinct from architecture-isolating ablations. A scratch run is a pretraining control; it is not automatically a fully optimized external baseline.

## Signed-offset and neighborhood-supervision controls

H0, S0 and P0 share a 671,719-parameter backbone and head structure. S0 preserves H0's heatmap target, floor anchor and original loss but uses linear signed offsets instead of bounded offsets. P0 uses the same signed model and extends offset supervision to a square neighborhood of radius 2 around each floor anchor. At source pixel `u`, the target is `truth − u`. Pixels in overlapping supports belong to the nearest reference point; exact ties use the original point index. Each truth's owned support has total weight one. Heatmap targets remain unchanged. The P0 loss combines the original heatmap Dice/BCE terms with twice the weighted coordinate RMSE. P0−S0 therefore tests the joint neighborhood/ownership/weighting change; it does not separately estimate those three ingredients.

S0 is already the exact capacity control for P0. `CAP_P0` is an alias of S0 and does not add three trials. A nominal bounded-offset neighborhood cell cannot represent negative neighbor-to-truth targets and is not treated as a valid independent control. This is a constrained sequential ablation, not a complete 2×2 factorial experiment. No new module is introduced merely to manufacture an architecture novelty claim.

## Training, selection and continuation

The in-house prospective recipe uses Adam (learning rate 0.001, zero weight decay), batch size 8, float32 without AMP/TF32, 100 epochs and 2,200 optimizer updates, with no training augmentation. Seeds 42, 43 and 44 use registered sample orders and historical CUDA initializer checks. Validation candidates are exported every five epochs, yielding 20 checkpoints and 31 threshold policies per checkpoint (620 selection pairs). Primary selection maximizes validation micro-F1, then precision, with frozen deterministic tie rules; native-loss-selected models are secondary. Weight, configuration, data and source hashes are bound to each selection. A one-step synthetic probe tests exact recovery before formal execution; successful CPU checks do not authorize CUDA training.

The prospective screen first trains S0 and P0 at seed42. The screen requires at least 20 common TPs, a mean common-TP error improvement of at least 0.10 px, an F1 difference no worse than −0.005 and no increase in near-reference unmatched predictions within 3 px. The explicitly approved continuation strategy is full_registered_six. Under gate-conditioned continuation, remaining seeds run only if the rule passes and expansion was authorized; a failed screen is reported as a two-run negative screen. Under an explicitly approved full-registered-six strategy, all six prespecified S0/P0 trials run regardless of the screen, retaining negative results and adding no new method. The strategy must be chosen before prospective results are observed. The screen is a development decision, not a significance test or proof of biological duplicates.

## Detection, ranked AP and localization

Thresholded predictions use the frozen maximum-cardinality point-matching policy within 3 px. Precision, recall and micro-F1 are computed from pooled TP/FP/FN counts; per-image metrics are retained. Localization error is Euclidean distance in pixels. We report both a model's matched-TP errors and paired errors on common matched truth identities; the latter limits changes caused solely by different TP composition.

Full ranked AP uses every candidate before thresholding. Predictions are ordered by decreasing score, with deterministic image/index ties, and greedily claim the nearest unmatched reference within 3 px. Equal-score groups contribute one precision–recall step. AP is the non-interpolated sum of precision weighted by recall increments [3]. This ranked matching policy is explicitly different from the thresholded F1 matching policy. Neither 31-policy AP nor trapezoidal PR-AUC is substituted for full ranked AP.

## Weak-reference definition and statistics

Weak-reference recall uses the frozen training-derived local-z-score cutoff 2.4566830772038957, with strict less-than classification. The feature samples the nearest rounded pixel and its clipped radius-10 neighborhood on the unpadded float32 image, using population SD; SD≤1e−6 is unclassifiable. The cutoff is not refitted on validation or test. The development reference contains 96 weak points. Unclassifiable points are reported separately.

Seed summaries use arithmetic means and sample SD. Five planned F1 contrasts are S0−H0, P0−S0, P0−G0, P0−R0 and P0−Spotiflow transfer. Exact paired-seed sign flips are conditional training-variability summaries, with Holm correction over the full five-contrast family; three seeds cannot yield a two-sided exact p below 0.25. Seeds, spots and source-overlapping images are not treated as biological replicates. On a genuinely independent dataset with reviewed group identities, a paired biological-group bootstrap resamples complete groups jointly across methods, retaining fixed selected models and averaging per-seed micro-F1 differences (10,000 draws; seed20261111). Its percentile interval is conditional on the selected models and does not create a p-value or significance star. No universal superiority conclusion is prewritten.

## Independent evaluation and reproducibility

The pending independent dataset is [PENDING: independent_dataset_description]. Before readout, model weights, validation thresholds, postprocessing, input metadata, file hashes and biological/acquisition provenance are frozen. No threshold or checkpoint is selected on test. Known image IDs, exact development-image content and known source containers are excluded; automated checks do not establish biological independence by themselves. The evaluation supports the in-house, signed-offset, Spotiflow and classical adapters through a common native-coordinate interface. The target journal and final export specifications remain [PENDING: target_journal_and_article_type]. Source tables accompany provisional SVG/PDF/300-dpi PNG figures. Code-assisted analysis and figure preparation are documented separately in SOFTWARE_ASSISTANCE.md.


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


# Discussion

The proposed experiments separate an operationally useful detector from a mechanistic explanation for improvement. G0 remains the practical candidate while H0's offset behavior is investigated. A gain from H0 to S0 would support changing the offset range at the original anchor; a gain from S0 to P0 would support the joint neighborhood-supervision design under unchanged capacity. A failure of either comparison is informative and must narrow the claim. Neither outcome alone establishes a novel architecture, and the relationship to prior localization and flow-regression methods requires a complete related-work review before submission.

Annotation uncertainty is especially important near adjacent spots. A visually real signal can be unmatched under one-to-one assignment; a near-reference false-positive proxy can reflect duplicate detections, assignment conflicts, an incomplete reference or localization error. The enriched blind patch audit helps distinguish these possibilities but is not an exhaustive relabeling of the full dataset. Corrections should be versioned, reviewed in images and evaluated with the original model locks before considering a new benchmark protocol.

The small development set, validation-based checkpoint and threshold selection, two known source-container overlaps and unresolved biological grouping limit inference. Training seeds measure algorithmic variability on one partition. They are not additional specimens. Repeated use of a development benchmark also creates adaptive selection that post-hoc confidence intervals do not remove. The externally pretrained Spotiflow reference remains important even if it outperforms an in-house method; a defensible contribution can concern reliable localization and error analysis without claiming the highest score in every metric.

The remaining evidence needed for a stronger manuscript includes [PENDING: final_mechanism_interpretation], [PENDING: independent_generalization_interpretation] and [PENDING: biological_validation_scope]. In particular, tracking robustness, locus splitting and biological conclusions require appropriate temporal or orthogonal validation. Target-journal selection, author review, acquisition details and data-release permissions remain human responsibilities. Automated reports and figures support reproducibility but cannot certify scientific validity or publication readiness.
