# Contribution boundaries before writing a paper

Primary sources checked on 2026-11-11:

- [deepBlink, NAR2021](https://academic.oup.com/nar/article/49/13/7292/6312733): neural detection and subpixel localization are established prior work.
- [Spotiflow, Nature Methods2025](https://www.nature.com/articles/s41592-025-02662-x): multiscale heatmap and stereographic-flow regression provide a strong specialized reference.
- [Objects as Points, 2019](https://arxiv.org/abs/1904.07850): center-keypoint estimation and regression of object properties are also established outside microscopy.

Consequently, merely making an offset signed or adding dense coordinate supervision must not be presented as a first-of-its-kind architecture. The present S0/P0 program tests a narrow implementation/supervision hypothesis under exact capacity control. Whether its joint design has a useful, sufficiently distinct contribution is pending actual results, related-work comparison and author judgment. This is an inference from the overlap in problem formulation, not a claim that the implementations are identical.

| Proposed claim | Required evidence | Current disposition |
|---|---|---|
| P0 improves support alignment over S0 | Actual P0−S0 common-TP error, detection and FP-proxy results; real audit | Pending prospective trials |
| Signed range is useful at fixed anchor | Actual S0−H0 comparison | Pending |
| Benefit is independent of parameter count | Same 671719 parameters and matched training/source; S0 control | Engineering contract ready, effect pending |
| Ownership and weighting each improve performance | Separate legal controls beyond the joint P0−S0 experiment | Not answered by this matrix; no separate claim |
| Our method beats specialized external methods | Complete method-level comparison with transfer/scratch context and independent evidence | Not established; retain strong Spotiflow results |
| New architecture | An actual distinct architectural mechanism plus evidence and prior-art review | Not established by this unchanged-backbone experiment |
| Independent biological generalization | New reviewed data, group identity, frozen-model readout | Data not supplied |
| Better tracking or biological locus splitting | Temporal/orthogonal biological validation | Outside the current single-frame study |

If the approved matrix succeeds, finish and interpret that evidence before expanding the contribution claim. If it fails, deliver the negative mechanism study and retained G0 candidate; do not silently start another architecture or select a favorable seed. A one-day engineering completion target cannot guarantee novelty, statistical significance, superiority or journal acceptance.
