# Tractography incubation: measured results

## 2026-10-01 M0: capture and reference (HCP 101006, 100 k of 440,621 streamlines, 5 seeded runs)

`capture.py` then `reference.py`, on the M2 Air's CPU (torch 2.14.1, 4 threads). The full-brain
context takes 2.2 s and inference 234-249 s per run. Summaries are in `m0/`.

- **The capture is TractCloud's own:** 0 of the first 10,240 run-0 labels differ from upstream
  `run_inference` on the same context.
- **fp16 storage** moves the argmax for 3-8 of the 100 k streamlines per run.
- **Instability is large:** across the five runs the tract label changes for **28.0 %** of
  streamlines and the cluster for **69.0 %**. 25.6 % of streamlines take two distinct tracts,
  2.3 % take three and 47 take four.
- **"Other" is half the brain:** 48.7 % of run-0 labels are "Other", and 22.2 % are outlier
  classes (≥ 800).
- **The two margin definitions rarely disagree on the winner:** for 1.7 % of streamlines
  another tract holds more mass than the labeled tract, so their mass margin is negative.

## 2026-10-01 M1: run-0 margin vs instability

`analyze.py` → `m1.json`, `m1_instability.png`, `m1_pairs.png`. The target is a tract label
that differs from run 0 in any of runs 1-4. Every predictor comes from run 0 alone.

| predictor (run 0) | AUROC | 95 % CI |
|---|---|---|
| tract mass margin | **0.875** | 0.873-0.877 |
| tract best-class margin | 0.862 | 0.860-0.864 |
| p_max, winning cluster (baseline) | 0.655 | 0.652-0.659 |
| cluster margin (baseline) | 0.638 | 0.634-0.641 |

- **Exit test passes.** The mass margin beats the best one-byte baseline by **0.220 AUROC**
  (bootstrap 95 % CI 0.217-0.223, 200 resamples).
- **The relationship is monotone.** By decile of the mass margin, least confident first, the
  change rate is 79.9, 64.5, 49.7, 35.8, 23.9, 14.8, 7.6, 3.0, 0.5 and 0.03 %.
- **In logits:** streamlines whose best-class margin is under 0.25 change 79 % of the time;
  over 8 logits, 0.3 %.
- **The mass margin leads**, slightly but well outside the confidence intervals. The viewer
  should default to it.
- **"Other" dominates the changes:** 78.7 % of change events have "Other" on one side, and the
  top pairs are Other with Sup-F, Intra-CBLM-PaT, Sup-P and SF. The plan's expected pairs (AF vs
  SLF-III, CC subdivisions) are minor.
- **The result is not only "Other" against a tract.** The 37,390 streamlines that are a named
  tract in every run change 13.1 % of the time. Among them the mass margin scores 0.810 and the
  best-class margin 0.792, against 0.629 for p_max and 0.616 for the cluster margin.
- **Secondary:**
  - Averaging the margin over all five runs reaches 0.929, the ceiling the ensemble buys.
  - Outlier status (cluster ≥ 800 or not) changes for 23.5 % of streamlines. The run-0
    |plausibility margin| predicts that at 0.862 (mass) and 0.855 (best-class), against 0.708
    for p_max.

**What this does and does not justify.** The baselines lose because they are *cluster*-level:
probability spread over sibling clusters of one tract reads as low confidence, while the tract
is stable. But a single byte holding the run-0 *tract* mass margin, computed at inference for
the fixed 42-tract grouping, would score the same 0.875. The field earns its 19 bytes only
where the grouping is chosen after inference: categories, custom merges, the outlier threshold,
cluster-level views. M1 shows the margin is the right quantity. Whether deferring the grouping
is worth storing the field is the next test: repeat M1 at category level and for merged groups
(SLF I+II+III, CC1-7), each margin decoded from the same stored field.
