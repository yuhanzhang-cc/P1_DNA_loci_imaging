**fig01_supervision_schematic** Conceptual pixel-coordinate example, not measured data. Star: reference at (y,x)=(2.25,2.60); blue squares: supervised source pixels; arrows: truth-minus-source targets. Orange ring: illustrative off-anchor peak. H0 bounds offsets, S0 changes their signed range at the floor anchor, and P0 extends supervision. The displayed radius 2 is the proposed example, not an approved or optimal value. Multiple-truth ownership is described in Methods.

Alt text: Three pixel grids illustrate the source-pixel/target relationship and the distinction between changing output range and extending supervision.

**fig02_method_comparison** Fixed-threshold metrics and complete ranked AP on the same 22 development images. Bars show means and sample SD across available training seeds, with individual seeds shown. LoG/DoG each have one deterministic calibrated configuration; no seed SD is assigned. Missing trials are pending. Localization uses each method's own matched TPs. Spotiflow transfer uses external pretraining and a different native recipe (hatched); these bars compare complete methods and do not isolate architecture.

Alt text: Four bar panels compare detection, ranking, weak-point recall and localization; S0/P0 and scratch remain pending until returned.

**fig03_sequential_ablation** Sequential parameter-identical comparisons: H0 has bounded single-anchor offsets; S0 uses signed single-anchor offsets; P0 adds neighborhood supervision with nearest-truth ownership and per-truth weighting. P0−S0 identifies the joint support/ownership/weighting change, not its individual factors. Missing points interrupt lines. This is a constrained sequential ablation, not a complete factorial design.

Alt text: Paired seed trajectories across H0, S0 and P0; only available measurements appear.

**fig04_ranked_pr** Complete score-ranked precision–recall curves, grouped at tied scores, using confidence-ordered greedy nearest-unmatched assignment within 3 px. AP is a non-interpolated recall-weighted precision sum; thresholded F1 uses the separate frozen maximum-cardinality policy. Each curve is one selected seed/configuration.

Alt text: All available seed precision–recall curves are displayed; curves end at attained recall.

**fig05_common_tp_localization** Mean candidate-minus-reference localization error on the intersection of matched truth identities within each seed. Negative values indicate lower error. Each marker is a seed; matched spots and seeds are not independent biological replicates. Missing contrasts remain pending.

Alt text: Seed-level localization differences for common true positives; negative is better.
