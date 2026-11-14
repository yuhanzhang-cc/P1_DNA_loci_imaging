# R0/H0 CUDA acceptance and comparison — 2026-11-02

## Decision

The returned R0/H0 runs meet the planned acceptance gates. All three archives and
their extracted files were independently checked locally. Six new baseline runs
are now verified, bringing the research training ledger from 35 to **41** runs.
Recovery replays and synthetic engineering updates are not additional experiments.
The server can be shut down; the following analysis uses local exports only.

Retain **G0 without joint augmentation** as the current simple model candidate.
H0 improves development 3px F1 over R0, but loses to G0 in every matched seed and
has worse localization, including on common true positives. H0 is a useful
representation/implementation reference and a smaller model, not an established
mechanism-innovation base. Do not launch further baseline or scratch training by
default. The application release remains the existing fixed Spotiflow/G0 candidate.

## Acceptance and recovery evidence

- Archive SHA256, gzip member reads and extracted-file bytes all passed:
  R0 **72**, H0 **79**, pre-recovery backup **130** files. Checksums accept LF/CRLF
  sidecars; no source/result archive was rewritten.
- Each R0/H0 seed has complete epochs 1–100, update counts 22–2200, finite losses,
  20 registered snapshots and exactly 620 distinct epoch/threshold pairs.
  Primary and native-secondary selections were recomputed using the locked
  tie-break rules. Exported selected-weight hashes passed the original completion
  checker. Predicted points were independently rematched against frozen truth;
  counts, weak recall and localization metrics reconcile with the exports.
- The backup recovery file is epoch **59 / 1298 updates**. The original failed
  epoch-60 model tensors were independently hashed; their hash agrees with both
  the addendum and `epoch60_replay_check.json`:
  `a09f6890ab76e9ec4ec9770168b57490342d0f99f9ba446cb99be50a5395935d`.
- `boundary_recovery_complete.json` reports exact epoch-60 tensor replay and
  byte-identical preservation of completed results. The overlay source hash agrees
  with the registered addendum. The only recorded correction removes one
  float32 upper-boundary candidate, with score **8.9627e-12**. It does not change
  training gradients or the optimizer trajectory. The final seed43 selection is
  epoch85 at threshold0.7834, rather than the interrupted epoch60.
- All four requested artifacts are present in
  `extracted_h0/outputs/research_v4_cuda/baseline_h0/`:
  `numerical_decode_addendum.json`, `boundary_roundout_events.jsonl`,
  `epoch60_replay_check.json`, and `boundary_overlay_source.py`.
  The completion record is a fifth useful artifact. Retained `failure.txt` is
  historical evidence of the original failure; the final controller/trial status
  and recovery completion record establish the current successful state.

There is one important scope limit to independent local verification. Of 364
protected file entries, **87** were checked against final returned bytes, **94**
were available only in the pre-recovery backup, and **183** omitted R0 epoch-cache
files were not returned. Full post-recovery protection of these caches is supported
by the recorded server gate, rather than a second local byte comparison. The new
epoch60 tensor file and orphan directory are also intentionally omitted by the
selected-result packer. Its replay hash can be verified against the original
failed tensors, but the two tensor files cannot be compared locally without an
additional cache download. These omissions follow the planned package design and
do not require retraining. To obtain a fully self-contained epoch audit, optionally
return the R0 epoch caches and H0 seed43 `epoch_060/` plus `orphaned_snapshots/`.

Machine-readable evidence: [acceptance](baseline_cuda_analysis_20261102/acceptance.json).

## Comparison contract

Use the same frozen **22 development images / 372 reference points**, including
**96 weak reference points**, and the original 3px maximum-cardinality matching.
All three methods use RTX5090, PyTorch2.11.0+cu128, float32, 176 training images,
100 epochs, batch8, 2200 updates and 20×31 development-selection budget. The same
seed identifiers and training sampler orders are paired. Different architectures
have different initial weights; they are not weight-identical pairs.

Primary = the registered point-F1-selected checkpoint and threshold. Always show
native-loss-selected secondary results alongside it. Keep each selected model and
threshold fixed for radius, image-size and FP analyses. No new threshold fitting,
model inference or test-array reading was performed in this analysis.

Summary statistics are **mean ± sample SD across the three training seeds**.
Localization is the Euclidean distance on matched TPs, in pixels; it is conditional
on the detection set. A second comparison uses only true points detected by both
methods. Seeds are training replicates, not independent biological samples.

## Primary results

| Metric | R0 | H0 | G0 reference |
|---|---:|---:|---:|
| 3px micro-F1 | 0.7887 ± 0.0108 | 0.8088 ± 0.0022 | **0.8197 ± 0.0071** |
| Precision | 0.7955 ± 0.0221 | 0.7999 ± 0.0227 | **0.8446 ± 0.0218** |
| Recall | 0.7823 ± 0.0097 | **0.8190 ± 0.0285** | 0.7975 ± 0.0324 |
| Weak recall | 0.4167 ± 0.0551 | 0.4618 ± 0.0636 | **0.4722 ± 0.0782** |
| Mean TP error, px | 0.4465 ± 0.0347 | 0.6758 ± 0.0440 | **0.4463 ± 0.0169** |
| TP RMSE, px | 0.5753 ± 0.0347 | 0.8429 ± 0.0442 | **0.5724 ± 0.0156** |
| Count MAE per image | 3.9394 ± 0.3531 | 3.4545 ± 0.2083 | **2.7727 ± 0.7925** |
| AP31 grid, 3px, exploratory | 0.7816 ± 0.0299 | 0.7855 ± 0.0217 | 0.7819 ± 0.0180 |
| Parameters | 1,258,723 | **671,719** | 1,265,504 |

Bold identifies a descriptive optimum, not statistical significance. H0 differs
from R0/G0 in representation, head, loss and parameter count. It is not a pure
single-factor representation ablation, nor does parameter count establish latency.

**AP qualification:** the delivered prediction CSVs have already been thresholded.
They cannot support full ranked detection AP. AP31 above integrates the precision
envelope at the existing 31 threshold policies of the selected epoch, with zero
area beyond attained recall. It is not COCO mAP or a full ranked AP, and must not
be relabeled as such. Full AP later requires the selected models' complete
pre-threshold scored candidates and one frozen matching/tie convention for every
method; no checkpoint or threshold may be reselected for AP.

Native-secondary results and per-seed values are retained in
[model summary](baseline_cuda_analysis_20261102/model_summary.csv) and
[per-seed table](baseline_cuda_analysis_20261102/per_seed_metrics.csv).

## Differences and uncertainty

| Contrast | Mean F1 difference | Paired seed differences, pp | Raw exact p | Holm p |
|---|---:|---|---:|---:|
| H0 − R0 | +2.01 pp | +2.40, +0.69, +2.94 | 0.25 | 0.75 |
| H0 − G0 | −1.09 pp | −0.55, −1.18, −1.55 | 0.25 | 0.75 |
| G0 − R0 | +3.10 pp | +2.95, +1.87, +4.49 | 0.25 | 0.75 |

These are exploratory paired, two-sided sign-flip contrasts, with Holm over this
three-contrast family. With three pairs the smallest attainable two-sided p is
0.25; **all are ns**. Normality cannot be meaningfully established at n=3, so no
normality-screen-driven t-test was introduced. There are no missing seed results,
outlier exclusions or spot-level independent tests.

The 10,000-draw paired image bootstrap gives conditional F1-difference intervals:
H0−R0 [−1.24, +5.42] pp; H0−G0 [−3.14, +1.00] pp; G0−R0 [+1.28, +5.24] pp.
The same image draws are shared across methods and seeds. These intervals condition
on selected models and 22 development images, without accounting for checkpoint
selection or unrecorded acquisition/biological clustering. The positive G0−R0
image interval does not override the seed-level ns result or establish independent
generalization. All secondary metrics remain descriptive.

## What explains H0's trade-off?

1. **More detections, insufficient precision.** H0 averages 304.7TP/76.7FP versus
   G0's 296.7TP/55.0FP. Across three seed-image evaluations, H0 adds 24TP and 65FP.
   Its weak TPs decrease from G0's 136 to133, while bright TPs increase by27.
   Its higher total recall therefore does not establish better weak-locus detection.
2. **Repeated nearby predictions dominate the additional FP events.** At 3px,
   unmatched predictions within3px of a reference point are labeled
   duplicate/assignment-conflict proxies. H0 has **80** such FP events across seeds,
   versus G0 **21**. The difference59 accounts for90.8% of H0's65 additional FP
   events. Far-from-reference events are138 versus136, and 3–6px events12 versus8.
   These are geometric tags relative to current annotations, not visually verified
   biological causes. Seed events can refer to the same physical position.
3. **Localization is independently weaker.** On common detected true points,
   H0's mean TP distance is worse in all three seeds against both R0 and G0.
   At the fixed 3px-selected predictions, 1px F1 is H0 **0.6242 ±0.0214**, R0
   **0.7264 ±0.0146**, G0 **0.7571 ±0.0093**. Pooled signed H0 residuals are about
   **dy=−0.229px, dx=−0.223px**, with negative means on both axes in every seed.
   This is a reason to inspect heatmap peak/offset alignment before adding modules.
4. **The scope of supervision suggests a testable hypothesis.** H0 heatmap targets
   use floor anchors, while offset loss is supervised only at those exact anchors.
   Decoding reads offsets at predicted heatmap maxima. When a maximum lands on a
   neighboring pixel, its offset need not have received the same target supervision.
   Together with 3×3 local-max extraction, this could contribute to localization
   bias and repeated nearby candidates. This is a hypothesis from code and errors,
   not a demonstrated cause or a justification for post-hoc coordinate correction.
5. **The numerical recovery event does not explain the broad deficit.** It removed
   one ultra-low-score edge candidate; all exported selected FP events are outside
   the 3px edge band. Exact tensor replay verifies the training trajectory through
   the interruption. No recovery-time change was made to training loss or labels.

Local, non-public detail tables include `fp_geometry_proxies.csv`,
`matched_point_residuals.csv`, `radius_sensitivity.csv`, `size_strata.csv` and
`shared_tp_*.csv` under `outputs/next_stage_v4_recipe_20261101/16_baseline_analysis_20261102/`.
The seven 256px images give H0 F1=0.8234 versus G0=0.8194; the fifteen 512px images
give H0=0.8044 versus G0=0.8197. These exploratory strata do not overturn the primary
endpoint or establish a size-specific advantage.

## Next short phase

1. Close R0/H0 baseline collection and preserve the existing G0 recipe and release.
   Read `acceptance.json`, `model_summary.csv` and `paired_comparisons.csv` first;
   then inspect per-seed selections, secondary metrics, shared-TP localization and
   FP geometry. Full raw training curves are supporting evidence, not the endpoint.
2. Perform a small no-training H0 diagnostic: retain the source heatmap maximum,
   measure its displacement from the labeled floor anchor, and inspect the offset
   prediction at the source pixel versus the anchor. Stratify common-TP residuals
   by that displacement. Confirm preprocessing and pixel conventions without
   recentering coordinates from validation residuals. This can distinguish an
   alignment/supervision defect from a merely underperforming head.
3. Finish the focused blinded FP/FN review. Geometric duplicate tags motivate a
   local candidate-consistency investigation, but annotation ambiguity and close
   real pairs must be checked before suppressing predictions. G0's own FP burden
   is predominantly far from the reference, so an H0-specific duplicate fix should
   not automatically become G0's proposed core mechanism.
4. Only after a reproducible failure mechanism is confirmed, lock one architecture
   change and its matched-capacity control. Run one engineering check and a
   prespecified exploratory screen, then a registered base/capacity/full/minus
   comparison if warranted. Fix the loss/training budget and attribution boundary.
   Do not label a decoder bug fix, generic heatmap head, NMS tuning or validation
   bias correction as the scientific innovation.
5. Full AP and independent data remain separate unfinished items. Export scored
   candidates for all fixed selected models to complete ranked AP. Collect new
   acquisition-level held-out images for the final independent comparison. Further
   use of the current development set cannot establish that generalization claim.

The core mechanism and its ablation matrix remain disabled pending this gate.
Spotiflow scratch remains deferred unless a specific pretrained-versus-scratch
claim requires it. No new formal training was started by this analysis.

## Reproduction and code review

Run the read-only analysis with the existing scientific Python environment:

```powershell
python deepblink_pytorch/scripts/analyze_baseline_return.py --data-root D:\Projects\P1_DNA_loci_imaging
```

The analysis and related geometry/recovery tests passed **56 targeted tests**.
The archived data, old package and checkpoints were not modified. No dependency
files were changed. Before committing, inspect the new archive/budget acceptance
checks, the explicit AP31 label, FP proxy definitions, the optional `methods`
argument added to the existing recipe reader, and the completed-versus-pending
scope updates in VS Code. Raw prediction/error tables stay local; only aggregate
evidence is copied into the documentation directory.
