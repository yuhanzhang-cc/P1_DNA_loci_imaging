# CUDA recipe evidence, 2026-11-02

Primary: G0 augmentation-reference mean micro-F1 delta=0.00013691 at3px; paired exact p=1.00; three seeds. Other metrics and strata are supporting/exploratory. Sources include all six100-epoch trials,620 selections each. CPU replay reproduced all TP/FP/FN/weak-TP counts on22 validation images. Images, predictions, match identities and weights remain private.

Figures: recipe_metric_bars (mean+/-sample SD bars and paired seeds), learning_curves (native loss mean+/-sample SD and checkpoint-best F1), radius_sensitivity (frozen-model radius sensitivity; primary endpoint remains3px). Underlying source tables and provenance are included.

See ../recipe_cuda_analysis_20261102.md for the Chinese interpretation and ../short_term_closure_20261102.md for next steps.
