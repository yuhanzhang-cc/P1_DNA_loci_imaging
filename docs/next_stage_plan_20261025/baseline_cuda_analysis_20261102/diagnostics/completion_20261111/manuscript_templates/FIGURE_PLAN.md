# Figure and source-data plan

1. **Study flow and offset geometry.** Native image and reference-coordinate convention; existing grid reference; H0 floor-anchor targets; S0 signed range; P0 neighborhood ownership and per-truth weights. Conceptual arrows are schematic, not measured biological trajectories. Native example selection and calibration: {{figure1_image_and_calibration}}.
2. **Method comparison.** F1, full ranked AP, weak-reference recall and own-TP localization. Available seeds over bars; sample SD only for repeat training. Classical references deterministic. Transfer and scratch separated and recipes disclosed. Missing trials visibly pending. Source: `per_seed_metrics.csv`.
3. **Sequential ablation.** Paired H0→S0→P0 seed lines; no line across missing cells. CAP_P0 is S0, not new training. Source: same table and fixed ablation matrix.
4. **Ranked precision–recall.** All seeds/configurations with no averaging that discards score ties. Curves end at attained recall. Source: `ranked_pr_curves.csv`.
5. **Common-TP localization.** Seed means of paired truth-identity error differences; geometric FP counts and blinded image examples supplied as a companion panel after review. Source: `common_tp_errors.csv`; image choices: {{reviewed_error_examples}}.
6. **Independent readout (pending data).** Frozen-model F1 differences with group-level paired intervals and native-image examples. Source: future independent evaluation tables; {{independent_figure_sources}}.

Supplementary tables retain every planned contrast, selection budget, parameter count, initializer/source hashes, hardware, optimizer updates, annotation versions and incomplete tasks. Compute-performance comparison requires actual same-device timed inference; cached calibration elapsed time is not a speed benchmark. Journal-specific sizing, font and TIFF requirements: {{target_journal_and_article_type}}.
