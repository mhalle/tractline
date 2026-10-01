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

## 2026-10-01 M1b: regrouping after inference, field margin vs scalars stored at inference

`regroup.py` → `m1b.json`, `m1b_regroup.png`. For each grouping, a streamline's label is the
group of its argmax cluster, and the target is that label changing in runs 1-4 relative to
run 0. Margins come from run 0's full fp16 field; M2 checks that rankfield reproduces them. The
competitors are every scalar that could be stored at inference: p_max, the cluster margin, and
the tract mass margin for TractCloud's fixed 42 tracts.

| grouping | groups | change | field mass margin | best scalar at inference | gain (95 % CI) |
|---|---|---|---|---|---|
| tract (TractCloud's) | 43 | 28.0 % | 0.875 | 0.875, tract margin | 0 (the same number) |
| category | 6 | 24.2 % | 0.878 | 0.861, tract margin | +0.017 (0.016-0.018) |
| merged tracts (SLF, CC) | 35 | 27.2 % | 0.877 | 0.873, tract margin | +0.004 (0.003-0.004) |
| tract, outliers folded in | 43 | 26.1 % | **0.886** | 0.725, tract margin | **+0.161** (0.158-0.165) |
| cluster, outliers folded in | 800 | 63.9 % | 0.786 | 0.743, p_max | +0.043 (0.041-0.044) |

**Reading:**
- **Coarsening the tracts buys almost nothing.** For a category or a merge of a few tracts,
  the stored 42-tract margin is nearly as good. Merging 2-7 tracts barely moves which
  streamlines are near a boundary, because most boundaries are against "Other".
- **The field's case is the outlier decision.** Fold each outlier twin into its cluster's tract
  and the stored tract margin collapses to 0.725: it measures a boundary, tract against Other,
  that this grouping removes. The field margin recovers 0.886. Moving the outlier threshold is
  the decision the plan's claim 4 wanted to defer, and it is where storing the field pays.
- **Finer groupings also gain.** At cluster level, with outliers folded, the field gains
  +0.043 over the best scalar.
- **This is not an argument against precomputing.** Any one grouping known at inference can be
  stored as its own byte and would match the field on that grouping. What 19 bytes buy over k
  bytes is every grouping, including the ones nobody named at inference.
- **The viewer's emphasis should follow.** The outlier threshold and outlier-folded tracts
  are the controls where the field shows something a stored label cannot. Category view and
  tract merges are conveniences.

## 2026-10-01 M2: rankfield encoding of run 0, fidelity against the full field

`encode.py` → `m2.json`. rankfield v0.3.10 with `keep="clip"`, clip 8, the default log byte
curve and a T = 1 tail. Run 0's (100 k, 1600) fp16 log-softmax goes in as a (1600, 100 k, 1, 1)
view. Encoding takes 1.5 s per depth on the CPU.

**Labels are exact** at every depth: 0 of 100,000 differ from the reference argmax.

**The field is wide.** Unlike the torso segmentation, where the shell held about 1.2 classes,
most streamlines have more than six classes within 8 logits of the winner:

| depth | kept planes (mean) | all planes used | tail median / 99th pct | tail > 1 % | raw B | blosc B |
|---|---|---|---|---|---|---|
| 4 | 3.87 | 91 % | 0.8 % / 24 % | 46 % | 13 | 9.5 |
| **6** | **5.44** | **74 %** | **0.2 % / 12 %** | **24 %** | **19** | **13.2** |
| 8 | 6.66 | 56 % | 0.09 % / 6 % | 12 % | 25 | 16.5 |

For comparison, per streamline: dense fp16 is 3200 B (2252 B compressed with blosc zstd), a
naive top-6 (fp16 values plus uint16 indices) is 24 B, argmax alone 2 B, argmax plus a p_max
byte 3 B. Depth 6 is 170 times smaller than the compressed dense field.

**AUROC from decoded margins against the full-field reference, depth 6** (the targets are the
M1/M1b ones):

| grouping | best-class decoded / ref | mass decoded / ref | best scalar at inference |
|---|---|---|---|
| tract (TractCloud's) | 0.860 / 0.862 | 0.863 / 0.875 | 0.875 |
| category | 0.864 / 0.867 | 0.867 / 0.878 | 0.861 |
| merged tracts (SLF, CC) | 0.862 / 0.865 | 0.865 / 0.877 | 0.873 |
| **tract, outliers folded in** | **0.872 / 0.877** | 0.868 / 0.886 | **0.725** |
| cluster, outliers folded in | 0.773 / 0.773 | 0.786 / 0.786 | 0.743 |
| plausibility, for outlier change | 0.853 / 0.855 | 0.859 / 0.862 | 0.708 (p_max) |

**Reading:**
- **The best-class margin survives encoding.** Its AUROC is within 0.005 of the full field for
  every grouping at depth 6, and within 0.001 at depth 8.
- **The headline survives too.** For outlier-folded tracts, the margin decoded from 19 bytes
  scores 0.872 where the best stored scalar scores 0.725.
- **The best-class margin overestimates where the depth cut hits.**
  - For 8.7 % of streamlines (tract grouping, depth 6) the decoded best-class margin is more
    than 0.05 logits too high: by up to 3.6 logits at the 99th percentile and 5.4 at the 99.9th.
  - The cause: `keep="clip"` fills the planes with the classes closest overall. When the winner's
    own tract has several sibling clusters close by, they take the planes and the nearest
    *non-member* is cut, so the margin is read against a farther class or the clip.
  - This is the "margin is an upper bound" caveat, in group form. It falls to 5.7 % at depth 8.
  - Everywhere else the error is the byte step, within 0.07 logits.
- **The mass margin is a loose lower bound.**
  - The tail is one number with no class attached, so counting it all against the group makes
    the decoded mass margin low by up to 10.8 logits at the 1st percentile at depth 6. Its sign
    flips for 274 streamlines (0.27 %).
  - The AUROC cost is 0.012 for the tract grouping and 0.018 for outlier-folded tracts at
    depth 6. Depth 8 halves it.
  - At depth 6 the decoded best-class margin beats the decoded mass margin, the reverse of the
    full-field ranking. The viewer should lead with the best-class margin unless it ships depth 8.
- **The plan's exit test passes in substance, not to the letter.** Labels are exact and every
  AUROC conclusion of M1 and M1b holds from the encoded field. But "margins within the byte
  step" holds for about 90 % of streamlines at depth 6, not all of them; the rest are the
  depth-cut overestimates above.

**Where rankfield could do better here (not changed, for discussion):** the depth cut is
group-blind by design, since the grouping is unknown when the field is written. Two cheap
options are raising the depth (8 costs 3.3 compressed bytes more per streamline) or storing a
second, outside-the-winner's-tract runner-up. That second option builds TractCloud's tract
table into the encoding, which gives up some of the point of deferring the grouping.

## 2026-10-01 M3: the full subject on Modal, and the web export

`modal_capture.py` → `m3.json`. All 440,621 streamlines, five seeded runs, on A10/A10G GPUs
(torch 2.14.1+cu130, cudnn off as upstream). Inputs are on the Modal Volume
`tractography-bench`, uploaded from DATA. The five 1.4 GB fields stay on the Volume, and only
`derived.npz` (77 MB) came down. Run 0's timing was not kept, and the timings of runs 1-4 were
lost to an edit that commented them out (fixed).

- **The full brain reproduces the 100 k subsample.**

  | | 100 k subsample | full subject |
  |---|---|---|
  | tract changes | 28.0 % | 27.9 % |
  | cluster changes | 69.0 % | 69.1 % |
  | run-0 AUROC, tract (best-class / mass / p_max) | 0.862 / 0.875 / 0.655 | 0.862 / 0.875 / 0.653 |
  | outlier-folded tracts (mass margin / fixed tract margin) | 0.886 / 0.725 | 0.886 / 0.722 |

- **GPU and CPU agree.** The same seed gives the same context on both, and the cluster labels
  of the 100 k targets differ between GPU and CPU at 4-12 streamlines per run: arithmetic only.
- **Encoding the whole brain** with rankfield (keep="clip", depth 6) takes 10.7 s on 8 CPU cores.

**The web export** (`export.py` → `DATA/hcp/web/`, 62 MB; `export_check.mjs` checks it):

| part | bytes | per streamline |
|---|---|---|
| geometry, 20 points, int16 at 0.01 mm, original RAS | 52.9 MB | 120 |
| field, rankfield depth 6 | 8.4 MB | 19.0 |
| extras (index, three change counts, margin spread) | 3.5 MB | 8 |
| dense fp16 field, for scale | 1410 MB | 3200 |

- **Chunks.** 50 chunks of about 8,813 streamlines, in one seeded shuffle, so any prefix of
  chunks is a uniform sample. The 100 k target is about 11 chunks, 14 MB.
- **Exit test passes.**
  - `node export_check.mjs` loads the manifest and all 150 chunk files in 146 ms.
  - Every size and sha256 matches the manifest and its own header.
  - The extra/ indices are a permutation of all streamlines.
  - The viewer's decode, ported to JS (the own-tract best-class margin from ranks and support
    through `levels.bin`), is bit-identical to `rankfield.decode_groups` on all 8,813
    streamlines of chunk 0.
  - A one-byte corruption fails the check twice, once by hash and once by decode.
- **Not done:** the TRX export, which was the plan's optional second target.
- **Geometry dominates the download.** The field is 13 % of it. Fewer points per line, or a
  delta encoding of the int16 coordinates, would shrink the export far more than any change to
  the field.

## 2026-10-01 M4: the PicoGL viewer, modes 1-2

`viewer/` (`index.html`, `main.js`, `decode.js`, `serve.py`) → `m4.json`, `m4_viewer.jpg`.
`serve.py` serves the viewer with the export mounted at `/data/`, on 127.0.0.1 only: HCP data,
local until the terms are checked.

**How it draws.** One draw call per 2 % chunk, chunks loading progressively. Streamlines are
`LINES` from a shared index buffer. Each streamline's group and two margins sit in an RGBA32F
texture read by `gl_VertexID / pointsPerLine`, so the threshold, softness and opacities are
uniforms: a threshold change re-uploads nothing.

**What it shows:**
- Mode 1 colors by tract. Mode 2 sets opacity from the margin, with "keep confident" or "keep
  ambiguous".
- The threshold is set on a histogram of the loaded streamlines' margins, which shows how many
  fall below it. Either margin can drive it: best-class (the default, per M2) or mass.
- "Other" is drawn in gray and can be hidden. It is half the brain, and Slicer's pink for it
  painted the whole view.
- View presets L/R/A/P/S/I; drag to rotate, shift-drag to pan, scroll to zoom. The brain is
  centered beside the controls, and on a phone it sits above a bottom sheet.

**Exit test passes.** A live threshold change at 96,943 streamlines (11 chunks) on the M2's
integrated GPU, at 2048×1536 with 4× MSAA, takes a median 23.8 ms per frame (p95 25.2 ms), about
40 fps. The full subject (440,621) runs at 102 ms, about 10 fps: usable, not smooth. In the
browser, decoding the field takes 12.8 ms for 97 k streamlines and 39 ms for all 440 k, so
regrouping in M5 can be live.

**The JS decode matches rankfield.** `export_check.mjs` imports `viewer/decode.js`. On chunk 0
the best-class margins and tracts are bit-identical to `rankfield.decode_groups`, and the mass
margins agree with `rankfield.probabilities` to 1.6e-7.

**Seen in the viewer:** with "keep ambiguous" the cerebellum stays bright, which is M1's second
most common change (Intra-CBLM-PaT and Other).

**Fixed along the way:**
- `gl.finish()` returned at once in this browser, so the first frame times (0.1 ms) were false.
  A one-pixel `readPixels` is the sync now.
- Histogram bins of 0.1 logits left empty bins past 5 logits, where the byte levels are
  farther apart. They are 0.2 logits now.

## 2026-10-01 Geometry encodings, and the prior art

`geometry_bench.py` → `geometry.json`. It covers all 21.6 M original vertices of the HCP
tractogram. UKF steps are 1.50-1.80 mm for 98 % of steps. Every predictive encoding is
round-tripped through its decoder before its size is reported.

| encoding | B/vertex | max error |
|---|---|---|
| float32 raw / zstd | 12.0 / 9.36 | 0 |
| float16 raw / zstd (TRX's suggested default) | 6.0 / 4.01 | 0.031 mm |
| 0.05 mm grid, 1st-order int8 deltas (the `.tt.gz` idea), bitshuffle zstd-9 | 2.08 | 0.025 mm |
| **0.05 mm grid, 2nd-order prediction, int8 residuals, bitshuffle zstd-9** | **1.55** | 0.025 mm |
| 0.01 mm grid, 2nd-order (archival) | 2.36 | 0.005 mm |
| 0.1 mm grid, 2nd-order | 1.24 | 0.05 mm |

At 0.05 mm the second-order encoding is 8 times smaller than float32, 2.6 times smaller than
float16 with zstd, and more accurate than float16. It is also smaller than the 20-point web
export, which discards more than half the vertices.

`prior-art.md` records three literature searches. What they mean for this work:
- **Geometry.** Nobody reports second-order prediction, lossless at a stated grid, with entropy
  coding. DSI Studio's `.tt.gz` ships first-order int8 deltas, without a paper and not lossless.
  QFib and Fiblets are lossy direction coders. The Allen Institute's Zarr Vectors draft already
  puts streamlines in Zarr with spatial chunks and multiscale, but without prediction. BIDS
  BEP046 mandates TRX.
- **Instability.** TractCloud's run-to-run instability has not been reported. The closest work,
  RapidParc (Imaging Neuroscience 2026), counts per-streamline flips for its own model and finds
  them at the inlier/"Other" boundary, but offers no single-run predictor. Tract-level
  aggregation for abstention is established in general ML (Hierarchical Selective
  Classification, NeurIPS 2024), not in this field.
- **Soft output.** The closest designs are LLM-distillation top-K with one residual mass, which
  is known to be biased. Shen et al. 2026 independently found rankfield's group-blind depth cut
  ("mass is not decision support"). No standard accepts top-k soft labels.
