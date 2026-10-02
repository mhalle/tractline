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

## 2026-10-01 TRAKO, head to head: size and decode speed

`trako_compare.py` → `trako.json`; `decode/decode_prep.py` and `decode/decode_bench.mjs` →
`decode_python.json`, `decode_js.json`. TRAKO (Haehn et al., MICCAI 2020; glTF + Draco) was run on
the same HCP tractogram, coordinates only, against the predictive encoding on Draco's grid. Draco
quantizes over the bounding box's largest extent (168 mm). Its float32 arithmetic puts 0.005-0.02 %
of coordinates one step from exact rounding; its error to the original stays within half a step,
as ours does, so the sizes compare like for like.

*Corrected the same day.* The first version of this section compared the predictive encoding
with TRAKO's whole `.tko` file. That file embeds its Draco buffers base64 in glTF JSON, a third
larger than the bytes themselves, which put TRAKO at 1.85-2.16× and credited the difference to
the entropy coder. The figures below use the raw Draco buffers.

| bits | grid | max error | raw Draco | `.tko` file | 1st order, our coder | 2nd order, our coder | Draco / ours |
|---|---|---|---|---|---|---|---|
| 14 (TRAKO's default) | 0.0102 mm | 0.0051 mm | 3.27 B/v | 4.37 | 2.89 | **2.37** | 1.38× |
| 12 | 0.0409 mm | 0.0205 mm | 2.54 | 3.39 | 2.14 | **1.65** | 1.54× |
| 11 | 0.0819 mm | 0.0410 mm | 2.17 | 2.89 | 1.79 | **1.34** | 1.62× |

- **Prediction order matters more than the coder.** At 12 bits, first-order deltas under
  bitshuffle and zstd save 16 % on Draco's first-order coding, and second-order prediction saves
  a further 23 %. Draco's compression level (1 or 10) changes its size by under 1 %.

**Decode speed**, all 21.6 M vertices to float32 positions, best of 5 (3 in Python), on the M2:

| | Draco (TRAKO's buffer) | predictive |
|---|---|---|
| Node 25: draco3d WASM vs numcodecs Blosc WASM + a JS loop | 894-1157 ms, **19-24 Mvertex/s** | 214-258 ms, **84-101 Mvertex/s** (decompress 107-169, reconstruct 87-108) |
| Python: TrakoDracoPy vs blosc + numpy | 1.9-2.2 s, plus 1.4-1.5 s to turn its point list into an array | 0.37-0.42 s decompress + 1.8-1.9 s reconstruct |

- **In JS the predictive decode is 4-4.7× faster than Draco.** The reconstruction is two running
  sums per streamline; Blosc's bitshuffle and zstd do the rest.
- **The Python reconstruction is not tuned:** whole-array numpy passes over int64. A compiled
  loop would cost about what JS's does.
- **Random access differs.** Prediction restarts at every streamline, so any chunk or streamline
  decodes on its own given the lengths. TRAKO stores one Draco buffer for the whole tractogram,
  so it decodes all or nothing; progressive or region-of-interest loading would need it split
  into many buffers, each a separate Draco stream.
- **What TRAKO has that this does not:** glTF packaging that browser glTF loaders with a Draco
  decoder can open; per-vertex and per-streamline attributes in the same container; an existing
  file format.
- **Running TRAKO in 2026 took repairs** (scratch environment only): its Draco binding's
  `setup.py` (a cmake < 3.15 pin and `packaging.LegacyVersion`), its pre-3.11 Cython output, and
  the NumPy aliases removed in 1.24.

## 2026-10-01 Time to display, TRAKO against the predictive encoding

`decode/render/` (a page plus its server) → `render.json`; `decode/pipeline_node.mjs` →
`pipeline_node.json`; inputs from `decode/render_prep.py`. The whole HCP tractogram (440,621
streamlines, 21.6 M vertices) goes from a localhost server to a drawn frame, each format the way
a viewer would take it.
- **TRAKO's path:** fetch the `.tko`, parse the glTF JSON, decode its base64 data-URI buffers
  (as three.js's GLTFLoader does), Draco-decode positions and lengths with Google's WASM decoder,
  upload float32.
- **Ours:** fetch the blosc blobs, decompress with Blosc WASM, reconstruct, upload either
  float32 or int16 grid coordinates that the shader dequantizes.
- **Both:** the same draw, one `multiDrawArrays` of line strips.

| | 14 bits: total | 12 bits: total | GPU |
|---|---|---|---|
| TRAKO | 3.64 s (base64 1.46, Draco 1.86) | 4.92 s (base64 1.32, Draco 3.15) | 260 MB |
| predictive, float32 | 1.47 s | 1.08 s | 260 MB |
| predictive, int16 grid | **1.32 s** | **1.02 s** | 130 MB |
| predictive, 50 chunks: first frame / all | **49 ms** / 1.41 s | **40 ms** / 1.53 s | 130 MB |

- **The whole tractogram is on screen 2.8-4.8× sooner.** First pixels arrive 75-120× sooner:
  progressive loading draws the first 8,813 streamlines while the rest load. TRAKO's single
  Draco stream shows nothing until all of it has decoded.
- **TRAKO loses time in two places:** base64 data URIs (1.3-1.5 s, eliminated by any binary
  container, glTF's `.glb` included) and Draco decoding (1.9-3.2 s here). Even with
  free base64, it would take 2.5-3.5 s here.
- **Absolute numbers are inflated.** The pane was hidden and driven over CDP, and every stage
  ran 2-4× slower than in Node. Node's CPU-only cross-check: TRAKO 1.05-1.24 s (base64 is cheap
  there), predictive 0.22-0.25 s, 4.7-5×.
- **Upload and draw cost the same for both** (about 50-90 ms and 105-127 ms). The int16 grid
  halves GPU memory and saves the float conversion; its dequantization in the vertex shader costs
  nothing measurable.
- **On the wire:** 51 MB against TRAKO's 95 MB (72 MB gzipped) at 14 bits; 36 MB against 73 MB
  (55 MB) at 12. On a 100 Mbit/s link that is 4.1 s against 7.6 s (5.7 s gzipped) at 14 bits,
  before any decoding. The chunked form overlaps decoding with downloading; a `.tko` can only
  be parsed after the last byte arrives.

## 2026-10-01 The whole payload: geometry plus field, against TRAKO with labels

`payload.json`, from `payload_field.json` and `payload_trako_labels.json`. Everything a browser
needs to show the full HCP tractogram (all 21.6 M original vertices) with TractCloud's labels:

| | 12 bits (max 0.02 mm) | 14 bits (max 0.005 mm) |
|---|---|---|
| predictive geometry | 35.8 MB | 51.2 MB |
| rankfield field, depth 6, blosc (12.4 B/streamline) | 5.5 MB | 5.5 MB |
| level table, class tables, manifest | 0.04 MB | 0.04 MB |
| **ours, total** | **41.3 MB** | **56.7 MB** |
| optional: instability counts and margin spread from 5 runs | +0.5 MB | +0.5 MB |
| TRAKO with each streamline's cluster label (`.tko`; raw Draco) | 74.0 MB (55.5) | 95.2 MB (71.4) |
| TRAKO with the dense 1600-class field, for scale | about 1,480 MB | about 1,500 MB |

- **The field is 13 % of the payload.** It costs about what TRAKO's base64 overhead alone adds,
  and it replaces a hard label with every decision downstream of the M1-M2 measurements:
  margins, regrouping, the outlier threshold.
- **For the same geometry accuracy, ours with the full field is 1.8× smaller than TRAKO with
  only a label** (and 1.3× smaller than TRAKO's raw Draco bytes, which no TRAKO file holds). In
  the browser it reaches the screen 2.8-4.8× sooner, with a first frame in 40-50 ms (render.json).
- **Hard labels alone would cost 0.13 MB** (tract, uint8) or 0.49 MB (cluster, uint16). Those
  are the bytes the field replaces.

**The fair baseline is TRX, not TRAKO** (`payload_trx.json`). TRAKO is a 2020 research
prototype: a paper with code, unmaintained since February 2025, and it needed four repairs to run.
The format the field adopted is TRX, which BIDS BEP046 requires. Written per its spec, with
TractCloud's tract labels as groups:

| TRX, all 21.6 M vertices, labels as groups | max position error | size |
|---|---|---|
| float16 positions (its suggested default), stored | 0.031 mm | 135.1 MB |
| float16, zip deflate (must be decompressed before loading) | 0.031 mm | 117.4 MB |
| float32, stored | 0 | 265.0 MB |

Against that, our 41.3 MB carries the full field as well, at a smaller position error (0.02 mm):
**3.3× smaller than TRX with labels only** (2.8× against deflated TRX). TRX's strength is
different: its stored arrays memory-map with no decode at all, which suits local analysis.

## 2026-10-01 Converting between the compact form and TRX

`convert_bench.py` → `convert.json`. The full HCP tractogram on the 12-bit grid (0.041 mm)
with run 0's depth-6 field: 41.2 MB compact. Best of 3 on the M2, Blosc with 8 threads. The
per-vertex loops run two ways: numpy (whole-array passes) and numba (compiled, one core, about
what a C or Rust converter gets).

**Compact → TRX** (TRX directory: 273 MB in float32, 143 MB in float16, field and groups included)

| stage | ms |
|---|---|
| decompress geometry / field | 17.5 / 5.0 |
| offsets from lengths | 1.0 |
| reconstruct positions, float32: numba / numpy | **56** / 2,107 |
| to float16 (TRX's suggested default) | 13 |
| labels to TRX groups, field planes to dps arrays | 21 + 3 |
| **in memory, numba** | **104 ms** (about 210 Mvertex/s) |
| write the TRX directory to the SSD, float32 / float16 | 145 / 53 |

**TRX → compact**

| stage | ms |
|---|---|
| open TRX (memory-mapped) | 0.5 |
| quantize + predict: numba / numpy | **80** / 1,752 |
| compress geometry, zstd 1 / 5 / 9 | 18 (42.6 MB) / 51 (39.2 MB) / 1,257 (35.8 MB) |
| compress the field from dps (zstd 9) | 327 |
| **total, numba, zstd 5** | **458 ms** |

- **Both directions take well under a second** for a 440 k-streamline, 21.6 M-vertex tractogram
  with a compiled loop. With numpy the loop alone is about 2 s each way.
- **The zstd level is the encode-side choice.** Level 9 saves 9 % over level 5 at 25× the time;
  decoding speed hardly depends on it.
- **Round trips:**
  - float32 TRX returns to the grid exactly: re-encoding reproduces every residual.
  - **float16 TRX does not.** 729,241 vertices (3.4 %) land off the 0.041 mm grid, because
    float16's spacing is 0.0625 mm beyond 64 mm from the origin. Convert to float32 TRX when the
    data must come back, or re-encode from the original.
- **The expensive encode is upstream.** Producing the field from TractCloud's logits
  (`rankfield.encode`) took 10.7 s for this subject on 8 cores (M3). It runs once, at inference;
  converting the stored field to and from TRX is only repacking (3 ms out, 327 ms back at zstd 9).

## 2026-10-01 Stage timings, tractogram to browser

`modal_timing.py` → `timing_gpu.json`; the other rows are from earlier entries (`m0/`,
`convert.json`, `render.json`). The full HCP subject: 440,621 streamlines, 21.6 M vertices. The
diffusion images, preprocessing and UKF tractography come before this table, ship as TractCloud's
test data, and were not run here.

| stage | where | time |
|---|---|---|
| read the tractogram (VTP, 370 MB) | M2 CPU | 3.6 s |
| TractCloud's 15-point features | M2 CPU | 2.4 s |
| load the model | Modal | 1.9 s |
| context: kNN within 10 k chunks, 80 global streamlines | CPU | 2.2-2.8 s |
| **TractCloud inference, 1600 classes** | **A10G** | **28.5 s** (M2 CPU: about 18 min, from 244 s per 100 k) |
| field encode, rankfield depth 6, from the logits on the GPU | A10G | **0.45 s** (identical ranks and gaps to the CPU's) |
| field encode on 8 CPU cores: this branch / v0.3.10 | Modal CPU | 2.5 s / 5.0 s |
| geometry: quantize + predict (numba) | M2 CPU | 0.08 s |
| geometry compress: zstd 5 / zstd 9 | M2 CPU | 0.05 s (39 MB) / 1.26 s (36 MB) |
| field compress, zstd 9 | M2 CPU | 0.33 s |
| **tractogram → 41 MB payload, total** | GPU + CPU | **about 40 s, three quarters of it inference** |
| download 41 MB at 100 Mbit/s / 1 Gbit/s | network | 3.3 s / 0.3 s |
| decode and first full frame in the browser | M2 | about 1 s (first chunk drawn at 40-50 ms) |

- **Inference is the pipeline.** Everything this experiment added - the field, the geometry
  encoding, the packaging - costs under 1 s beside TractCloud's 28.5 s on a GPU, and nothing beside
  its 18 minutes on a laptop CPU.
- **Encode the field where the logits are.** On the GPU the encode takes 0.45 s and only the
  19-byte planes leave it, not the 1.4 GB of fp16 logits.

## 2026-10-01 Where the test tractogram came from

`TestData/HCP/101006_ukf_pp_with_region.vtp` carries no provenance: no tracking parameters, no
field data. This is reconstructed from the file, the code and the papers; the confidence of
each step is marked.

1. **Diffusion data:** HCP subject 101006 (TractCloud's paper: "HCP dataset ... subjects not part of
   the training atlas"). Which HCP shell or preprocessing was used is not stated in what was read.
2. **Tractography: two-tensor UKF**, [pnlbwh/ukftractography](https://github.com/pnlbwh/ukftractography)
   (`UKFTractography`, also the SlicerDMRI extension). *Stated* by TractCloud's paper (§2.1, citing
   the ORG atlas, Zhang et al. 2018, for "the same parameter settings"). SupWMA
   ([arXiv 2207.08975](https://arxiv.org/abs/2207.08975), same group) gives them: seeds in brain-mask
   voxels with FA > 0.1, stop at FA < 0.08 or normalized mean signal < 0.06. These differ from UKF's
   defaults (0.18 / 0.15 / 0.1), so they were set explicitly.
   - *Measured, not stated:* points are recorded every 1.8 mm (median step 1.799 mm, 98 % within
     1.5-1.8). UKF's `recordLength` default is 0.9, so it was set to 1.8.
3. **"pp": [whitematteranalysis](https://github.com/SlicerDMRI/whitematteranalysis)
   `wm_preprocess_all.py`**, which writes `{subject}_pp.vtp` and removes fibers shorter than `-l` mm.
   *Consistent* with `-l 40`: the shortest streamline is 38.8 mm of arc (23 points at 1.8 mm), and the
   papers keep "streamlines longer than 40 mm". By default it does not keep per-point data, which
   matches the absence of UKF's FA and tensor arrays (`recordFA` and `recordTensors` default on).
4. **"with_region": per-point FreeSurfer labels.** `region_label` holds 183 FreeSurfer
   aparc/wmparc values (subcortical 1-99, cortex 1000s/2000s, white matter 3000s/4000s,
   unsegmented 5001/5002), so a wmparc-style label volume was sampled at every vertex.
   `region_mask` is 0/1. **The code that did this was not found**: not in TractCloud, not in WMA's
   scripts, not in a GitHub code search. Slicer's Probe Volume With Model, or a lab script, is a guess.
   TractCloud does not read either array.

## 2026-10-01 UKF in float32: what an Apple GPU's arithmetic does

`ukf32_compare.py` → `ukf32.json`. Stanford HARDI, ORG settings. `track(dtype=..., device=...)`
now runs the steps in float32 on the CPU or on MPS (which has no float64), from the same float64
seeds; the float64 path is bit-identical to before the change. `advance()` is one step, and
`DATA/ukf32/steps.npz` holds 3,335 captured step inputs (steps 1-835): the fixture a Metal kernel
will be tested against.

**One step, identical inputs, float32 against float64.** State relative error 1.3e-5 median
(2e-3 worst), direction 0.016° (90th percentile), position 1.4e-5 mm, FA 1.7e-5. Decisions: no
stop and no in-mask flips in 3,335 steps; 5 tensor-swap flips (0.15 %), where the two tensors are
nearly equidistant from the previous direction. CPU and MPS float32 are alike.

**Whole fibers, 2,011 seeds:**

| | ends within 0.06 mm | fibers within 0.1 mm everywhere | same point count | density r | worst |
|---|---|---|---|---|---|
| float64 CPU vs float64 CUDA (the floor) | 100 % | 1,729 / 1,729 | 1,729 | 1.000 | 6.6e-5 mm |
| float32 CPU vs float64 | 88.4 % | 1,256 / 1,728 | 1,508 | 0.972 | 35 mm |
| float32 MPS vs float64 | 88.7 % | 1,249 / 1,729 | 1,518 | 0.973 | 35 mm |
| float32 MPS vs float32 CPU | 85.9 % | 1,177 / 1,728 | 1,485 | 0.970 | 94 mm |

- **float64 is robust to rounding order; float32 is not.** Two float64 implementations agree to
  6.6e-5 mm on every fiber. In float32 a 1e-5 per-step error flips a swap or stop decision in about
  one fiber in eight, and that fiber then diverges (13 % change length by at least one 1.8 mm
  record point; the 99th percentile is 80 mm).
- **The device does not matter; the precision does.** MPS float32 differs from CPU float32 as much
  as either differs from float64.
- **albula-diffusion's own GPU-vs-CPU figures** (README: 90 % of fiber ends within 0.06 mm,
  density maps r = 0.97) are what float32 alone produces here (88 %, 0.97). His GPU port behaves
  like a correct float32 version of his CPU port.
- **Speed (eager torch, M2):** float64 CPU 4,600 steps/s, float32 CPU 8,100, float32 MPS 6,400.
  Eager MPS is launch-bound; the fused Metal kernel is what would change this.
- **Not yet decided:** whether float32's 12 % of diverging fibers matter. That is the bootstrap
  test: the spread the scan's own measurement noise gives the float64 pipeline.

## 2026-10-01 The noise floor: what the scan's own noise does to the float64 tracker

`ukf_noise_floor.py` → `ukf_noise_floor.json`. Wild-bootstrap replicates of HARDI (`_bootstrap.py`:
spherical harmonics order 6 fitted per voxel, residuals scaled by leverage, random signs per voxel
and volume, rounded to int16). The residuals are noise, not model error: their RMS hardly moves
from order 4 to 8 (22.5, 22.4, 22.5), below the b0 repeats' spread (33.5), as magnitude noise is at
low signal. Each replicate tracked in float64 from the same 2,011 seed points as the original
(`track(seed_points=...)`, starting from each point's state on the replicate).

| 2,011 seeds | ends within 0.06 mm | fibers within 0.1 mm everywhere | same point count | density r | median worst point |
|---|---|---|---|---|---|
| float32 vs float64 | 88 % | 1,256 / 1,728 | 1,508 | 0.972 | 0.012 mm |
| bootstrap replicate vs original (4) | 9-10 % | 1-2 / ~1,710 | ~320 | 0.740-0.747 | 1.2 mm |
| replicate vs replicate (3 pairs) | 5-8 % | 0 / ~1,705 | ~240 | 0.712-0.718 | 1.5-1.8 mm |

- **Data noise moves every fiber; float32 moves one in eight.** Replicates also change which seeds
  the binary would accept (29-40 of 2,011) and which fibers survive the 10-point rule (~35 per pair).
- At the fiber level float32 sits far below the noise floor: its density maps correlate with
  float64's at 0.97, where two equally valid acquisitions correlate at 0.72-0.75 (sparse maps from
  2,011 fibers; whole-brain maps correlate higher, in both cases).
- Whether the same holds for TractCloud's labels is `modal_ukf_labels.py`.

## 2026-10-01 float32 against the noise floor, through TractCloud

`modal_ukf_labels.py` → `ukf_labels.json`. Whole HARDI, the 98,491 seed points the binary accepts,
tracked four ways on A10Gs: float64 (the reference), float32, and float64 on two bootstrap
replicates. Each cut at 40 mm, resampled, labeled by upstream TractCloud (10 seeded context draws
for float64, 5 for the rest); 5-draw majority votes compared seed by seed.

| against float64's labels | single draw | 5-draw vote | tract mix r | per-tract count change (median) | Other |
|---|---|---|---|---|---|
| TractCloud's own draws (the floor) | 0.861 | 0.926 | 0.9993 | 5.1 % | 64.9 % vs 64.1 % |
| float32 | 0.852 | 0.905 | 0.9986 | 5.7 % | 63.6 % |
| bootstrap replicates (2) | 0.743-0.744 | 0.777-0.781 | 0.9988-0.9991 | 5.6-6.1 % | 63.2-64.2 % |

- **float32 costs about 2 points of label agreement; the scan's noise costs about 15.** float32's
  loss is the same on fibers that did not move (0.905) as on those that did: it comes from
  TractCloud's context, which shifts when any third of the brain's fibers shift, not from the
  fibers themselves.
- **Per-tract counts and the tract mix are at TractCloud's own floor for all three** (5-6 %, r ≥
  0.998): aggregates do not see float32, and barely see data noise.
- **Tracking speed, eager torch on an A10G:** float64 49,000 steps/s (560 s for the brain), float32
  114,000 (240 s).
- **Conclusion:** float32 tracking meets the bar set for every optimization: its effect is small
  against what the original pipeline already has (TractCloud's randomness, the scan's noise). A
  float32 Metal kernel is principled for a Mac. albula-diffusion's float32 GPU UKF is defensible on
  the same grounds; its other departures (free water, seeding, precision of the signal) are
  separate questions.
- One scan (b = 2000, 2 mm, SNR 24); the wild bootstrap's symmetric signs approximate magnitude
  noise. The same test on HCP-like data (b = 3000, 1.25 mm) would confirm.

## 2026-10-01 UKF as a Metal kernel

`_ukf_metal.py` (the kernel, compiled with `torch.mps.compile_shader`, as labelfield's restore
backend is), `ukf_metal_check.py` → `ukf_metal.json`. A drop-in for `advance()`:
`track(backend="metal")`. Eight lanes of a SIMD group per half-fiber, the filter's 10x10 algebra
redundant in each lane, the gradients split across the lanes; precise:: math, no FMA contraction.

- **Against float64:** one step on 3,335 fixtures, the same distance as torch's float32 step
  (state 1.3e-5 relative, direction 0.016°, no stop flips). 2,011 seeds: 87 % of fiber ends within
  0.06 mm, density r 0.971; the whole brain (98,491 seeds) against the float64 CUDA run: 86.7 %,
  r 0.9989. The float32 profile, which ukf_labels.json put below the noise floor.
- **Deterministic:** two launches bit-identical, per step and per fiber.
- **Speed, M2 (10 GPU cores), whole HARDI brain, 27.5 M fiber steps:**

  | | seconds | steps/s |
  |---|---|---|
  | Slicer's C++ binary, 8 threads, Rosetta | 1,649 | 16,700 |
  | torch float64, CPU | ~6,000 (est.) | 4,600 |
  | torch float32, MPS (eager) | ~4,300 (est.) | 6,400 |
  | torch float64, A10G | 560 | 49,000 |
  | torch float32, A10G | 240 | 114,000 |
  | **Metal kernel, M2** | **176** | **156,000** |

- **What made it fast**, one step at 16k fibers: 32 lanes per fiber 89 k steps/s; keeping the
  predicted signal between the two sigma-point passes (bit-identical) 99 k; 16 lanes 143 k; **8
  lanes 175-179 k**; 4 lanes 72 k (the per-lane arrays spill). Fast math adds 15-20 % with errors at
  float32's level but different bits; it stays off.
- The host loop (gathers, recording, compaction) is under 1 % of a step.

## 2026-10-01 The faithful pipeline on a Mac, end to end

`mac_pipeline.py` → `mac_pipeline.json`, `mac_labels.py` → `mac_labels.json`. Apple M2 (16 GB),
whole HARDI brain, ORG settings: DWI to the rankfield field in **232 s**.

| stage | seconds |
|---|---|
| load and normalize the DWI | 1.2 |
| seeds (float64, CPU) | 6.5 |
| UKF, Metal kernel (27.5 M steps, 142 k/s) | 194 |
| 40 mm cut, 15-point resampling | 0.9 |
| TractCloud context (upstream, CPU) | 0.4 |
| TractCloud network, MPS float32 | 10.6 (float16 autocast: 14.0, slower) |
| field encode, depth 6 (19.0 B/streamline) | 4.8 |

- **TractCloud on MPS gives the CPU's labels exactly** (74,281 of 74,281 on one context draw), and
  the same floor as CUDA (two 5-draw votes agree 0.9265). float16 on MPS is slower than float32 and
  changes 1,064 tract labels: float32 stays the Mac default.
- **Labels, all four tractographies labeled on MPS**, each against two independent float64 votes:
  Metal 0.889 / 0.896, CUDA float32 0.879 / 0.889, bootstrap 0.769 / 0.765, floor 0.927. The Metal
  kernel is as good as torch's float32 on CUDA; vote comparisons move 1-2 points from one set of
  draws to another (CUDA float32 scored 0.905 in ukf_labels.json).
- Open: TractCloud's single-draw floor is 0.861 on CUDA (cudnn off) and 0.846 on MPS / CPU with the
  same seeded draws, though the vote floors agree. CUDA's network arithmetic likely differs from
  the CPU's; which one upstream's published results used is not stated.
- UKF is 84 % of the time. On an M1 Max (32 GPU cores to the M2's 10, 4x the memory bandwidth)
  the whole pipeline should take roughly a minute; not measured.

## 2026-10-01 Writing the compact form on a laptop

`encode_timing.py` → `encode_timing.json`. Apple M2 (4 performance cores), the whole HCP 101006
subject (440,621 streamlines, 21.6 M vertices), inputs in memory, best of 3:

| | size | seconds |
|---|---|---|
| geometry, 0.010 mm grid, zstd 9 (payload.json's) | 51.2 MB | 3.5-3.7 |
| geometry, 0.041 mm grid, zstd 9 | 35.8 MB | 2.5-3.0 |
| geometry, 0.010 mm, zstd 5 | 64.6 MB | 1.2 |
| geometry, 0.041 mm, zstd 5 | 39.2 MB | 1.1 |
| field, depth 6, zstd 9 (rankfield 3efef7f) | 5.5 MB | 3.1 |
| field, rankfield 0.3.10 as released | 5.5 MB | 7.3 |
| **all of it, 0.010 mm, zstd 9** | **56.7 MB** | **6.5** |
| **all of it, 0.041 mm, zstd 9** | **41.3 MB** | **5.5** |
| all of it, 0.041 mm, zstd 5 | 44.9 MB | 3.8 |

- The field's top rank equals the argmax for every streamline; the geometry round-trips to the grid.
- int32 residual arithmetic gives identical residuals and halves that step (0.65 s against
  1.5 s); zstd 9's compression is then most of the geometry time.
- On an A10G the field encodes in 0.45 s (timing_gpu.json). Geometry was not timed on a GPU.

## 2026-10-01 The field at connectome scale: DeepMultiConnectome

`dmc_field.py` → `dmc_field.json`. DeepMultiConnectome's pretrained two-head model
(SlicerDMRI/DeepMultiConnectome 6c606ec; PointNet, no context) on HCP 101006's UKF tractogram
(440,621 streamlines), both log-softmax heads encoded as rank fields (depth 6, clip 8) on an M2.
The input is off the model's domain (UKF in ACPC space; trained on iFOD2 in MNI), so this is a
format-and-speed test, not an accuracy test.

- **The heads are 3,655 and 13,695 classes,** not the paper's 3,571 and 13,631: the weights count
  "unknown" as a node (85 and 165 nodes, n(n+1)/2 pairs).
- **Size:** dense float16 is 3.2 GB and 12.1 GB for this subject (21 GB and 80 GB at the paper's
  2.93 M streamlines). The fields are 5.6 MB and 6.2 MB (12.7 and 14.1 B/streamline compressed,
  19.0 raw), the same bytes per streamline as TractCloud's 1,600 classes. Top rank = argmax for
  every streamline in both heads.
- **The outputs are diffuse here:** 79 % (DK) and 85 % (Destrieux) of streamlines keep all 6
  planes; median dropped mass 0.7 % and 2.3 %, 99th percentile 33 % and 43 %. Off-domain input
  likely widens them; on its own data a deeper field (8-12) may still be worth it for Destrieux.
- **Time, M2:** features 2.9 s, inference (MPS, both heads) 16.4 s (26,900 streamlines/s), encode
  on the CPU 5.7 s (3,655 classes) and 26.2 s (13,695): encoding cost grows with the class count.
  At the paper's 2.93 M streamlines: about 110 s inference, 40 s and 175 s encode on this laptop's
  CPU. The GPU encode path (0.45 s for TractCloud on an A10G) is the obvious next step.

## 2026-10-01 UKF as a Triton kernel (CUDA)

`_ukf_triton_block.py` (compact: [F,16] states, [F,16,16] matrices, batched IEEE tl.dot, Cholesky
loops), `modal_ukf_triton.py` → `ukf_triton_block_{smoke,tune,full}.json`. The first, fully unrolled
generator (`_ukf_triton.py`) is correct but took 1,901 s to compile, and its whole-brain run hit
Modal's 2-hour limit without finishing; it is kept only as the first version. The block kernel
compiles in about 40-140 s.

- **One step, 3,335 fixtures, against float64:** state error 3.3e-7 median, 40x smaller than torch's
  float32 step (1.3e-5); 2 swap flips against torch float32's 7; no stop flips. The Cholesky-based
  inverses are the likely reason. Two launches bit-identical.
- **Launch sweep (one step, 50k half-fibers, A10G):** 456 k steps/s at F 2 / 8 warps, 454 k at F 1 /
  4 warps; F 4 exceeds shared memory.
- **Whole HARDI brain (98,491 seeds), F 2 / 4 warps:** 78 s steady state (352 k steps/s; the first
  run, 213 s, includes compiling). Against float64: 85 % of fibers within 0.1 mm everywhere (Metal
  69 %), 94.4 % of ends within 0.06 mm (Metal 86.7 %, torch float32 88 %), density r 0.9994 (Metal
  0.9989). The more accurate float32 arithmetic shows as fewer diverging fibers.

| whole HARDI brain, UKF | seconds |
|---|---|
| Slicer C++ binary, M2, 8 threads (Rosetta) | 1,649 |
| torch float64, A10G | 560 |
| torch float32, A10G | 240 |
| Metal kernel, M2 | 176 |
| **Triton block kernel, A10G** | **78** |

## 2026-10-01 Is albula's sparse seeding reasonable? PAT16

`pat16_prep.py` (ds001226 PAT16, CC0: b0 + the b = 2800 shell, 2.5 mm, median_otsu mask; the
gradient table checked by axis flips: mean fiber 77 mm as converted, 47-54 mm with any axis flipped),
`pat16_seeding.py` → `pat16_seeding.json`, `pat16_seeding_floor.json`. Same scan, mask, tracker
(Metal kernel, ORG settings) and TractCloud; only the seeds differ. Faithful: UKF's own 47,856 seeds.
Sparse: albula's rule, 25,000 of the 28,037 voxels with FA > 0.2 (89 % of the candidates: on this
2.5 mm scan the draw changes little but the position inside each voxel), 5 draws.

| | faithful | sparse (5 draws) |
|---|---|---|
| streamlines >= 40 mm | 32,137 | 20,225-20,277 |
| Other | 56.1 % | 58.3-60.3 % |
| tract mix r against faithful's 5-draw vote | 0.995-0.999 (single context draws) | 0.988-0.995 |
| tracking on the M2 | 41 s (11.5 M steps) | 25 s (7.1 M steps) |

- **Tract shares:** median 9-12 % relative error per tract against faithful, 7 % spread across draws.
  The largest shortfalls are where FA is low: cerebellar Intra-CBLM-PaT (7 % of faithful's share) and
  the superficial tracts (Sup-FP 68 %, Sup-PO 70 %, Sup-O 76 %, Sup-P 80 %).
- **The near rule** (a tract is near when >= 5 of its streamlines pass within 8 mm), 20 test spheres
  (10 mm, at FA > 0.3; PAT16's tumor outline is not public), against faithful's 5-draw vote:

  | | recall of faithful's near tracts | draw-to-draw Jaccard | spheres where draws disagree |
  |---|---|---|---|
  | faithful, single context draws (the floor) | 0.972 | 0.914 | 20 / 20 |
  | sparse, rule as albula applies it | 0.899 | 0.888 | 19 / 20 |
  | sparse, against faithful with the threshold scaled to its density | 0.945 (precision 0.949) | | |

- **Reading it:** the near-tract answer moves from run to run mainly because of TractCloud's own
  random context (Jaccard 0.914 with the tractogram fixed); sparse seeding adds a little (0.888) and
  misses about 7 % more of the tracts faithful finds near a region, mostly through lower density
  (scaling the threshold recovers half). Defensible on this scan, with a known bias against low-FA
  (superficial, cerebellar, and plausibly edematous peritumoral) white matter.
- **Caveats:** our tracker and settings, not albula's free-water defaults, which give about 11,000
  streamlines from the same 25,000 seeds (sparser still, so the effects would be larger); test
  spheres, not a tumor; one subject. At HCP's 1.25 mm the same 25,000 seeds would be about 5 % of
  faithful seeding, not 52 %.

## 2026-10-01 The faithful pipeline on PAT16, end to end on an M2

`mac_pipeline.py --data pat16` → `mac_pipeline_pat16.json`. ds001226 PAT16 (b0 + b = 2800, 2.5 mm),
UKF's own 47,856 seeds, ORG settings, Metal kernel; TractCloud on MPS; field and geometry encoded.

| stage | seconds |
|---|---|
| load and normalize | 0.4 |
| seeds (float64, CPU) | 1.0 |
| UKF, Metal (11.5 M steps, 261 k/s: 50 gradients here against HARDI's 150) | 43.9 |
| 40 mm cut, resampling | 0.3 |
| TractCloud context + network (MPS float32) | 4.0 |
| field encode | 0.3 |
| geometry encode (two grids, with round-trip check) | 0.8 |
| **total** | **55** |

Sizes, the 32,137 streamlines >= 40 mm (1.73 M points, a point every 1.8 mm): float32 points 20.7 MB;
compact 3.0 MB at a 0.05 mm grid, 4.5 MB at 0.01 mm, field included (0.44 MB, 19 B/streamline raw).
UKF is 80 % of the time. (Distortion correction is not in this pipeline yet.)

## 2026-10-01 Metal kernel: Cholesky inverses (more accurate and faster)

The Triton block kernel's accuracy came from its inverses: both 10x10 inverses in the UKF update
(Pm and Yk + I) are symmetric positive definite, and inverting them through Cholesky (L^-T L^-1)
instead of Gauss-Jordan cuts the Metal kernel's per-step state error against float64 from 1.3e-5 to
3.5e-7 (covariance 1.8e-6 to 4.0e-7). With unpacked 10x10 scratch matrices it ran at half speed
(82 k against 172 k steps/s, register spills); with L and L^-1 as packed 55-float triangles it is
faster than Gauss-Jordan (207 k against 183 k). Now the default (`SPDINV`).

| Metal kernel against float64, 2,011 HARDI seeds | Gauss-Jordan | Cholesky |
|---|---|---|
| fibers within 0.1 mm everywhere | 1,213 / 1,729 | 1,495 / 1,729 |
| fiber ends within 0.06 mm | 87.3 % | 95.1 % |
| density r | 0.971 | 0.986 |
| deterministic (two launches) | yes | yes |

PAT16 end to end (`mac_pipeline_pat16.json`): UKF 34.3 s (334 k steps/s) against 43.9 s; the
pipeline 41 s without the float16 comparison pass.

## 2026-10-01 Metal kernel: where the time goes

One step of 16,384 half-fibers (HARDI, 150 gradients), M2, defaults (8 lanes, cached H, Cholesky
inverses, precise math): 195-207 k steps/s. Two options measured and left off: float16 signal
(+1.4 %, 5x the per-step error) and the Triton kernel's Ht-free update (same accuracy, 6 % slower
here). Ablations on a throwaway copy (wrong results, timings only):

| removed or cheapened | steps/s | share of a step |
|---|---|---|
| nothing | 195 k | |
| exp() in the predicted signal (fast or none) | 220 k | ~11 % |
| the 3x3x3 signal gathers | 216 k | ~10 % |
| both 10x10 inverses | 226 k | ~14 % |
| divides made fast | 216 k | ~10 % |

No single hotspot: the rest (~55 %) is the per-fiber algebra every one of a fiber's 8 lanes repeats
(sigma points, their normalization and tensors, the 21-point covariance sum, the Cholesky of P).
The remaining structural lever is to split that algebra across the lanes instead of repeating it.

## 2026-10-01 Metal kernel: sigma points split across the lanes

Each of a half-fiber's 8 lanes used to build all 21 sigma points four times (mean, covariance,
predicted signal, cross-covariance passes) and all 21 tensors. Now (`SPLIT`) each lane builds the
2-3 it owns once; the mean and covariance are sums across the lanes; each sigma point's tensors, then
its deviation, are broadcast from its owner (simd_shuffle) for the two passes over the lane's own
gradients. It only pays with the per-lane arrays sized to the data's gradient count (`NGRAD`,
compiled per count) instead of the worst case of 256: otherwise they spill.

| one step, 16,384 half-fibers, HARDI (150 gradients), M2 | steps/s |
|---|---|
| Cholesky inverses (previous default) | 207 k |
| + arrays sized to the data | 220 k |
| + sigma points split, worst-case arrays | 116 k |
| **+ split, arrays sized to the data (new default)** | **230 k** |
| the same with 16 or 32 lanes | 196 k, 161 k |
| + the Ht-free update (SFORM) | 208 k |

Accuracy unchanged (state error 3.4e-7; 1 swap flip in 3,335 fixtures instead of 6; no stop flips),
deterministic. 2,011 HARDI seeds: 1,491 / 1,729 fibers within 0.1 mm of float64, 95 % of ends within
0.06 mm, density r 0.988. **PAT16 (50 gradients): UKF 27.6 s (416 k steps/s)**, against 34.3 s with
the Cholesky inverses alone and 43.9 s before; the pipeline 38 s with the float16 comparison pass
(34 s without). The fewer the gradients, the more the right-sized arrays help.
Whole HARDI brain with the new default (`mac_pipeline.json`): UKF 135 s (203 k steps/s), against
176-194 s with the first Metal kernel; the pipeline 154 s without the float16 comparison pass.

## 2026-10-01 Distortion correction: FSL topup as the reference, and what it changes on PAT16

`topup_ref.py` → `topup_ref.json`. FSL topup + applytopup (fsl-topup from FSL's conda channel,
FSL 2412.6, in DATA/fsl-env; a test reference only, FSL's license is non-commercial), PAT16's 6 AP +
2 PA b0s, total readout 0.0266 s, b02b0.cnf; applytopup on the 102 AP volumes, Jacobian modulation.
On the M2: topup 617 s, applytopup 48 s. Off-resonance field -105 to +82 Hz (1st-99th percentile);
displacement along phase encoding 3.2 voxels (8 mm) at the 99th percentile, 7.8 voxels (19.5 mm) at
most.

`pat16_topup_compare.py` → `pat16_topup_compare.json`: the faithful pipeline (UKF's own seeds, ORG
settings, Metal kernel; TractCloud 5-draw vote) on the scan as acquired and as corrected.

| | as acquired | topup-corrected |
|---|---|---|
| seeds | 47,856 | 47,470 |
| streamlines >= 40 mm | 32,129 | 32,155 |
| median length | 85.8 mm | 87.5 mm |
| Other | 56.9 % | 58.0 % |

- **Tract mix r 0.9926**, below TractCloud's own redraw floor (0.995-0.999): per tract a median
  8.8 % change in share, 26 % at the 90th percentile. The largest where susceptibility is: the
  posterior fossa (ICP x0.62, intracerebellar x0.64) and the frontal and occipital poles (SP x1.57,
  Sup-O x1.35, TO x0.73).
- **Tracts move:** each named tract's center shifts by a median 2.3 mm, 4.5 mm at the 90th
  percentile, 8.6 mm at most - the scale of a planning margin (albula's 8 mm); local shifts near the
  most distorted regions are larger than a whole tract's center shows.
- **Conclusion:** correction matters for planning; the faithful clinical pipeline gets a correction
  stage, topup's output being the reference our own implementation has to match.

## 2026-10-01 Our susceptibility correction (_susc.py), first version, against topup

topup's model in torch (B-spline field, displacement along phase encoding with the Jacobian, rigid
motion, bending-energy regularization weighted by the current mean squared difference, b02b0.cnf's 9
levels; L-BFGS; trilinear), `susc_check.py` → `susc_check_*.json`, PAT16's 8 b0s, on MPS: **33 s**
against topup's 617 s on the CPU.

- Getting there: the 3-axis einsum built a huge intermediate (out of memory, slow) - contracted one
  axis at a time; lambda over five orders of magnitude made no difference (r 0.893 throughout); the
  motion was the problem - holding it at zero gave r 0.958, better than estimating it - because L-BFGS
  stepped field coefficients (tens of Hz) and rotations (hundredths of a radian) alike; with the
  parameters scaled (10 Hz, 0.01 rad per unit) the estimate with motion reaches **r 0.967**.
- **Field against topup, in the brain:** median 0.44 mm of displacement difference, 2.4 mm at the
  99th percentile (1.7 mm deep inside, 4.5 mm at the edge).
- **The corrected b0s:** AP-PA disagreement (relative RMS, what both minimize) 0.123 ours, 0.122
  topup's, 0.543 uncorrected; correlation with topup's corrected images 0.964 (uncorrected 0.899).
- **Downstream** (`pat16_topup_compare.py`, all 102 volumes corrected, faithful pipeline), against
  topup's correction (topup + applytopup):

  | | tract mix r | tract centers moved, median / 90th / max |
  |---|---|---|
  | uncorrected | 0.993 | 2.3 / 4.5 / 8.6 mm |
  | ours (field and trilinear application) | 0.995 | 1.7 / 3.7 / 8.3 mm |
  | control: topup's field, our trilinear application | 0.998 | 1.4 / 3.1 / 7.6 mm |

  The control is the floor (tracking is chaotic; any interpolation difference perturbs it) and
  confirms the sign convention. Ours closes about two-thirds of the gap on tract centers and a third
  on the tract mix: the field itself has to come closer. Next: cubic B-spline sampling along the
  phase-encoding axis at the fine levels (topup's interp=spline) and in the application.

## 2026-10-01 Our susceptibility correction against topup: within the scan's noise

- **Cubic B-splines along phase encoding** (fine levels and the application, topup's interp=spline):
  no change to the field's agreement (r 0.965 against 0.967); kept as the default, it matches
  topup's interpolation. **Lambda** up to x1e8: no trend (r 0.955-0.974, optimizer path noise).
- **The offset:** our field ran a constant ~6 Hz above topup's (a uniform 0.4 mm). topup_movpar.txt
  shows its convention: the first volume of each acquisition (the first PA b0) has no translation
  along the phase-encoding axis, which fixes the otherwise free trade between a field offset and
  those translations. Adopted: field r 0.972, displacement difference median 0.24 mm, 99th
  percentile 2.1 mm (1.6 deep, 3.9 at the edge); corrected b0s correlate 0.975 with topup's; AP-PA
  disagreement after correction 0.126 (topup 0.122). 50 s on MPS. Our motion parameters still follow
  another convention than topup's (a common rotation of about -0.7 deg about y against volume 0,
  some translations with the opposite sign); left as is, the downstream test below being the bar.
- **Downstream, against topup + applytopup** (faithful pipeline, `pat16_topup_compare_*.json`):

  | | tract mix r | per-tract change, median / 90th | tract centers moved, median / 90th / max |
  |---|---|---|---|
  | uncorrected | 0.993 | 9 % / 26 % | 2.3 / 4.5 / 8.6 mm |
  | **ours** | **0.998** | 5 % / 21 % | **1.5 / 3.7 / 8.1 mm** |
  | topup's field, our cubic application (interpolation floor) | 0.999 | 5 % / 13 % | 1.2 / 2.5 / 4.5 mm |
  | scan noise: 2 wild-bootstrap replicates of the corrected scan | 0.996 | 7-9 % / 21-24 % | 2.3-2.5 / 4.5-5.9 / 7.5-8.6 mm |

  Ours sits between the interpolation floor and the scan's noise on every measure. The largest
  center moves (TO 8.1 mm, ILF 6.1) are inside what noise does to those tracts (TO 6.3-8.6, ILF
  2.0-7.0).
- **A correction to the previous entry:** within diffusion space, topup's correction moves tract
  centers about as much as the scan's noise does (2.3 / 4.5 / 8.6 against 2.3-2.5 / 4.5-5.9 /
  7.5-8.6 mm); its effect on the tract mix (r 0.993) is outside the noise. The main reason to
  correct for planning is alignment with the T1 the tumor is outlined on (displacements up to
  19.5 mm), which tract centers in diffusion space do not measure; a T1-alignment test would.

## 2026-10-01 Against the T1: correction is what puts PAT16's tracts where the anatomy is

`t1_alignment.py`: each arm's mean AP b0 aligned rigidly to the T1 (MPRAGE 1 mm, which none of the
corrections saw), then jointly with a smooth residual displacement along phase encoding (B-spline,
15 mm knots, mean held at zero), normalized-gradient-field cost (contrast-free: T1 against a
T2-weighted b0). The residual is what an overlay on the T1 would be off by. ~2 min per arm on CPU.

- **The measure is valid:** on the uncorrected scan the T1 alone recovers topup's displacement map,
  r 0.91, slope 0.78 (the smooth model underestimates the largest displacements by about a fifth).
  Precision (fit on b0s 1-3 against 4-6): 0.07 / 0.16-0.19 / 0.33-0.37 mm.
- **|residual| mm, median / 90th / 99th** (`t1_alignment.json`, maps in `t1_alignment.png`):

  | | brain | where topup displaces > 3 mm (orbitofrontal, temporal poles) | elsewhere |
  |---|---|---|---|
  | uncorrected | 0.95 / 3.25 / 6.79 | 3.68 / 6.38 / 8.62 | 0.78 / 2.22 / 3.47 |
  | topup | 0.27 / 0.79 / 1.65 | 0.42 / 1.21 / 2.12 | 0.26 / 0.72 / 1.44 |
  | ours | 0.34 / 0.82 / 1.53 | 0.45 / 1.09 / 1.99 | 0.33 / 0.77 / 1.37 |

  Both corrections leave sub-voxel residuals (voxel 2.5 mm); the T1 cannot tell them apart (ours
  slightly lower at the 99th percentile, slightly higher at the median). Their residuals do not
  correlate with topup's displacement (r -0.13, 0.00): nothing systematic is left of the distortion.
- **For the demo:** uncorrected, tracts overlaid on the T1 are off by 4-9 mm (more, given the slope)
  across the orbitofrontal cortex and temporal poles, and by up to 3.5 mm elsewhere; corrected,
  by under 2 mm almost everywhere. Within diffusion space the tract centers barely moved (previous
  entry); against the anatomy they do. Correction belongs in any pipeline whose tracts are shown
  on a T1.

### PAT16's tumor against the distortion

ds001226 `participants.tsv`: anaplastic astrocytoma II-III, 50 cm3, fronto-temporal. The dataset's
mask (`derivatives/tumor_masks/sub-PAT16/anat/sub-PAT16_space_T1_label-tumor.nii`, manual +
disconnectome; its array stored left-right flipped against the T1's, its header matching: read by
the header it covers the hypointense lesion, T1 mean 159.5 against 209.5 in the mirror region):
45 cm3, centroid RAS (31, 36, -13) mm: **right** anterior temporal lobe, insula and frontal
operculum. It borders both distortion lobes: the orbitofrontal one medially (displaced ~ -8 mm),
the temporal-base one inferiorly (+6-8 mm). Carried onto each arm's grid by its own rigid fit
(`t1_alignment.json`, the figure now cuts through and outlines it):

| |mm, median / 90th / 99th | tumor | 10 mm margin around it |
|---|---|---|---|
| topup's displacement of the scan (median over brain removed) | | 1.0 / 5.3 / 12.3 | 1.2 / 6.0 / 16.3 |
| residual against the T1 | uncorrected | 0.7 / 4.6 / 7.5 | 1.0 / 4.9 / 8.6 |
| | topup | 0.2 / 0.8 / 1.4 | 0.3 / 0.9 / 1.9 |
| | ours | 0.2 / 0.7 / 1.2 | 0.4 / 0.8 / 2.0 |

Uncorrected, a tenth of the tumor and its margin would be drawn 5-16 mm from where the T1 shows
it (the T1 measure recovers about 0.78 of that); corrected, under 2 mm throughout, ours and topup
alike. The margin is where the uncinate, IFOF and arcuate run past it.

## 2026-10-01 Correction, faster on the M2: 62 s -> 30 s

- **Application** (`susc_apply.py`): every AP volume shares one field and one readout, so the
  sampling positions, weights and indices are now computed once (`_susc.sample_pe` takes positions
  (1, X, Y, Z) and expands them). 102 volumes in **0.8-0.9 s on the CPU** (float64, output
  bit-identical to before), against 6 s; the GPU (float32, `--device mps`) takes 1.2 s, the copies
  dominating, output within 0.004 of the CPU's. The 11.7 s timed before included 5.5 s of gzip
  writing the NIfTI, now reported apart (`write_gzip_s`): a pipeline holding the volumes in memory
  skips it.
- **Estimate** (`susc_check_trilinear_conv.json`): trilinear sampling at every level, **28.9 s**
  against 50.6 s with cubic B-splines along phase encoding. Against topup: field r 0.969 (cubic
  0.972), displacement difference median 0.27 mm, 99th 2.17 (0.24, 2.11); AP-PA disagreement after
  correction 0.125 (0.126; topup 0.122). Against the T1 (`t1_alignment.json`, arm `ours_fast`):
  brain 0.36 / 0.85 / 1.52 mm, tumor 0.25 / 0.67 / 1.16, its margin 0.37 / 0.74 / 1.90 - within 0.07
  mm of the cubic estimate everywhere, the measure's precision (half-split) 0.06-0.07 / 0.18 / 0.39.
  **Trilinear is now the default** of `_susc.estimate` and `susc_check.py`; the application keeps
  cubic B-splines along phase encoding (applytopup's splines, under a second anyway).
- **PAT16 end to end on the M2:** correction 30 s (estimate 29, application 1) + the pipeline 34 s
  (UKF 27.6, TractCloud 3.7, the rest 2.8) = **about 64 s** before any file writing; topup +
  applytopup alone take 665 s.

## 2026-10-02 Twelve patients: correction, pipeline and T1 check across ds001226

`cohort.py --sub <PAT>` per patient, on the M2: our correction (`_susc`, estimate on MPS, application
on CPU), the faithful pipeline on the scan as acquired and as corrected (`_prep`, UKF Metal,
`_tractcloud` 5-draw vote), both against the patient's T1 and tumor mask (`_t1check`, the measure of
`t1_alignment.py`, now shared). `cohort_summary.py` → `cohort_summary.md` / `.png`. Patients chosen
for spread: meningiomas PAT13, PAT19 (skull base), PAT23, PAT08 (frontal), PAT14 (parietal); gliomas
and an ependymoma PAT07, PAT25, PAT26 (temporal), PAT29, PAT05 (frontal), PAT20 (parietal); PAT16.

- **Data, two surprises:** PAT03's "PA" series was phase-encoded left-right (`i-`, its image agrees),
  the only one of 25 patients: no reversed pair, replaced by PAT14. PAT19, PAT20, PAT23 and PAT29 have
  the PA slab rotated 0.8 deg about its center against the AP: the PA b0s are put onto the AP grid by
  the headers first (`_subject.py`), the estimate's motion taking what is left (max 0.6-2.0 mm).
- **The table** (`cohort_summary.md`; tumor margin = 10 mm around the mask; mm, percentiles):

| subject | tumor | cm3 | side | our displacement at the margin, 99th | margin vs T1, uncorrected 90th / 99th | margin vs T1, ours 90th / 99th | validation r / slope | tract mix r | centers moved median / max | scan to labels |
|---|---|---|---|---|---|---|---|---|---|---|
| PAT05 | Oligo-astrocytoma II, Frontal | 11.4 | L | 5.63 | 3.04 / 6.08 | 0.74 / 1.08 | 0.863 / 0.874 | 0.9948 | 2.6 / 9.3 | 75.7 s |
| PAT07 | Ependymoma II, Temporal | 29.7 | L | 5.19 | 2.52 / 3.86 | 1.2 / 1.64 | 0.798 / 0.715 | 0.9984 | 1.9 / 8.0 | 83.5 s |
| PAT08 | Meningioma I, Frontal | 17.7 | mid | 3.11 | 1.99 / 3.27 | 0.67 / 1.37 | 0.818 / 0.76 | 0.9973 | 2.8 / 10.7 | 85.3 s |
| PAT13 | Meningioma I, Skullbase | 1.7 | mid | 18.53 | 7.8 / 8.97 | 1.96 / 3.41 | 0.846 / 0.752 | 0.9955 | 2.0 / 10.0 | 73.9 s |
| PAT14 | Meningioma I, Parietal | 3.5 | L | 1.83 | 1.09 / 1.48 | 0.45 / 0.62 | 0.835 / 0.724 | 0.9955 | 1.9 / 8.2 | 66.3 s |
| PAT16 | Anaplastic astrocytoma II-III, Fronto-temporal | 45.4 | R | 16.81 | 4.79 / 8.17 | 0.96 / 2.45 | 0.861 / 0.732 | 0.9945 | 1.9 / 10.2 | 75.9 s |
| PAT19 | Meningioma I, Frontal skullbase | 2.8 | R | 16.05 | 6.47 / 7.89 | 1.83 / 2.64 | 0.863 / 0.708 | 0.9918 | 1.9 / 11.3 | 76.3 s |
| PAT20 | Anaplastic astrocytoma III, Parietal | 12.5 | R | 3.48 | 1.81 / 3.01 | 0.6 / 0.85 | 0.911 / 0.856 | 0.998 | 2.1 / 11.9 | 69.0 s |
| PAT23 | Meningioma I, Frontal | 103.5 | mid | 11.34 | 3.32 / 4.9 | 1.36 / 5.2 | 0.833 / 0.799 | 0.9961 | 2.8 / 13.8 | 84.9 s |
| PAT25 | Glioma II, Temporal | 16.5 | R | 7.75 | 3.97 / 5.9 | 1.24 / 2.12 | 0.696 / 0.608 | 0.9938 | 2.7 / 12.3 | 75.6 s |
| PAT26 | Anaplastic astrocytoma III, Temporal | 55.3 | R | 8.47 | 1.58 / 4.2 | 0.73 / 1.27 | 0.768 / 0.665 | 0.9976 | 1.6 / 7.2 | 82.3 s |
| PAT29 | Oligo-astrocytoma III, Frontal | 33.8 | L | 5.35 | 3.15 / 4.57 | 1.1 / 1.85 | 0.807 / 0.711 | 0.9926 | 2.9 / 11.2 | 80.7 s |

Medians over 12: margin vs T1 99th, uncorrected 4.74 mm, ours 1.75 mm; brain 99th 6.02 / 1.71 mm; precision (half-split, brain 99th) 0.40 mm; estimate sensitivity at the margin 99th 0.33 mm; validation r 0.83, slope 0.73; tract mix r 0.9955; scan to labels 76 s (field estimate 29 s, UKF 28 s).

- **Correction where it matters:** the margin's misplacement against the T1 falls from a median 99th
  percentile of 4.7 mm to 1.75 mm; how much it matters is where the tumor is - frontal-base and
  midline frontal tumors (PAT13, PAT19, PAT16) are displaced 16-19 mm at their margin, parietal ones
  (PAT14, PAT20) 2-3.5 mm. Over the brain, 99th percentile 6.0 → 1.7 mm (1.25-2.5 for ours).
- **The measure holds in every patient:** the uncorrected residual recovers our displacement map,
  r 0.70-0.91 (median 0.83), slope 0.61-0.87 (0.73); the corrected residual does not follow it
  (r -0.16 to 0.18). Precision (b0 halves, brain 99th) 0.21-0.74 mm.
- **PAT23, the exception:** margin 99th 5.2 mm corrected against 4.9 uncorrected (90th: 1.36 against
  3.32) - 398 margin voxels (4 %) at the frontal base under the 104 cm3 meningioma, where the T1 check
  puts the corrected scan 4 mm off and the uncorrected 1.8. `cohort_topup.py`: FSL topup from the same
  b0 stack agrees with ours (field r 0.971; displacement difference 0.21 / 0.62 / 1.88 mm over the
  brain, 0.32 / 1.19 / 3.03 at the spot; median displacement there -2.52 ours, -2.30 topup) and its
  corrected scan is as far off at the spot (4.22 / 6.50 / 7.78 against ours 4.03 / 6.29 / 7.51; margin
  99th 5.42). Not our implementation: either the T1 check is misled at the meningioma's base (tumor
  and edema differ between b0 and T1) or both reversed-pair corrections fail there alike; these data
  cannot tell which. topup 614 s + 45 s, ours 29.7 s.
- **The estimate's own sensitivity:** the same b0s plus noise of SD 0.01 (signals in the hundreds)
  move the field by 0.14-1.08 mm at the margin's 99th percentile (median 0.33), 0.04-0.08 mm at the
  median: L-BFGS's path, deterministic for a given input (two runs on one input: identical) but not
  insensitive to it. Found when PAT16's cohort run gave margin 99th 2.45 mm against 1.96 before: the
  input differed by topup_ref.py's int16 quantization of the b0s (≤ 0.03), which moved the field by
  0.08 / 0.68 mm (median / 99th). The T1 check's own fit moves by up to 0.05 mm when the voxel sizes
  change by 1e-8 (the affine's column norms against the header's): `_t1check` takes the header's.
- **What correction changes in the tracts:** tract-mix r 0.992-0.998, tract centers moved median
  1.6-2.9 mm (max 7-14) - the size of the scan's own noise on PAT16 (bootstrap 2.3-2.5 / max 7.5-8.6),
  as there: the change that matters is against the anatomy, not within the scan. "Other" 56-71 % of
  streamlines (PAT16 57 % before; HCP 49 %), highest under PAT23's meningioma.
- **Time, scan to labels with correction: median 76 s (66-85)** - field estimate 29 s, application
  1.1 s, preparation 12 s (10 s of it DIPY median_otsu for the mask; writing and reading the NRRD
  the tracker loads costs ~0.15 s, so an in-memory hand-off would save almost nothing), UKF 28 s
  (20-36, 408 k steps/s median, ~32 k streamlines ≥ 40 mm), TractCloud one draw 4.5 s. The T1 check
  (3 min) and the 5-draw vote are measurement, not pipeline.

## 2026-10-02 The brain mask: 10 s → 0.12 s, the same mask

The cohort's 12 s preparation was DIPY `median_otsu` (10 s: four passes of SciPy's 9^3 median
filter); writing and reading the NRRD the tracker loads costs ~0.15 s. `_median.py` computes the same
thing exactly: the volume's values replaced by their ranks (a median is one of its window's values, so
all passes stay in rank space and map back exactly), each pass by Huang's sliding-window histogram
along x (81 values leave, 81 enter per step; rows in parallel with numba), then DIPY's own `otsu`.
`median_check.py` → `median_check.json`: the filter identical to `scipy.ndimage.median_filter` on
random volumes with ties and odd shapes; the mask identical to `dipy.segment.mask.median_otsu`,
voxel for voxel, on all 14 volumes (the 12 patients as acquired, PAT16 corrected by topup and by
ours); **0.12 s against 10.07 s** (median). A torch version (unfold + median, MPS or CPU) was also
exact but 3.0 s: sorting 729 values per voxel. `_prep.py` uses it: preparation 0.22 s (int16) /
0.36 s (float32) instead of 10-12 s, so scan to labels with correction is about **64 s** (76 s less
11.9 s; not re-measured end to end). The first call in a fresh environment compiles (numba, cached).

### No files between the scan and the labels

`_prep.prepare` builds the tracker's input in memory (the DWI as stored, its NRRD header, the mask)
and `_ukf_torch.from_arrays` takes it - `load` now reads the files and calls it, `_prep.prep` writes
them for the scripts that want them. Tracker inputs identical, tensor for tensor, both ways (int16
and float32 scans). `cohort.py` keeps everything in memory (only the results are written): PAT16 end
to end reproduces the committed `cohort/PAT16.json` exactly apart from timings, **scan to labels
65.2 s** (75.9 before; measured).
- The corrected scan's mask took 2.3 s there, not 0.12: its b0 has 541,775 distinct values (7,430 as
  acquired), and `_median` allocated and zeroed a histogram that size per row. Now one histogram per
  thread, cleared by removing each row's last window, with a coarse layer of 256-bin block counts the
  median's walk skips through: **0.29 s corrected, 0.18 s as acquired**; `median_check.py` still
  identical on all 14 volumes and the random filters.

## 2026-10-02 TRX output

`_trx.py`, an option of the pipeline (`_pipeline.run(..., trx=path)`, `run_pipeline.py --trx`): the
tractogram as TRX - a stored zip (memory-mappable) or a directory - with the DWI's affine and grid in
the header, the rank field's meta block under "RANKFIELD", tracts as groups, and per streamline:
tract, TractCloud's top cluster, the tract's probability (its clusters' summed) and its margin over
the next tract, length, seed index, and the rank field's arrays. Every streamline the tracker kept
is written, those under 40 mm unlabeled (tract 255): the 40 mm cut a reader's filter, as
pipeline.md recommended, not a deletion; `labeled_only` writes the payload's set.
`trx_check.py` → `trx_check.json`, PAT16, read back by trx-python 0.6: header, every point (float32:
exact; float16: within 0.031 mm), groups partitioning the labeled streamlines, and every
per-streamline array row for row - all pass. 0.5 s to write. Sizes: **24.4 MB** (zip, all 41,895
streamlines, float32), **11.8 MB** (labeled 31,993, float16), against the compact payload's 3.0 MB.
Median tract margin 0.93 (probability units).

## 2026-10-02 The CPU tracker, 4x: one process per core, the fast algebra

The whole pipeline on the CPU (`cpu_timing.py`, PAT16, M2, 8 threads): 714 s scan to payload against
64 s on the GPU - UKF 580 s (19.7 k steps/s, float32), TractCloud 53 s, field estimate 78 s (float64).
One CPU step profiled (`_ukf_torch`, float32): H about 60 %, the rest of the filter 30 %, interp 5 %;
4 threads no faster than 8 - memory and per-op overhead, not arithmetic. Then:
- **Bit-identical** (against the committed tracker, ~2,400 seeds, float32 and float64 on the CPU,
  Metal): H computed for the N gradients and repeated (the other N are their negatives; u'Du is
  unchanged bit for bit); the step loop carries the compacted live arrays instead of gathering and
  scattering every half-fiber's state each step, one `nonzero` per step (boolean masks synchronized
  MPS four times a step: Metal fell to 117 k, back to 144-148 k). CPU float32 19.1 → 25.7 k steps/s,
  float64 12.2 → 17.8 k.
- **Batch composition changes bits**, even in float64 (vectorized math rounds by position): batches
  of 512 or 4,096 differ from 50,000. So worker processes take whole fixed blocks (`batch`
  half-fibers, the serial loop's own blocks), and `workers=k` equals the serial run at that batch,
  bit for bit (1, 4, 8 workers checked). Spawned, D's tensors shared (macOS and Windows cannot fork).
- **Workers** (a quarter of the seeds): 1 x 8 threads 29 k steps/s; 4 x 1 47 k; 8 x 1, batch 1,024
  55 k (2,048: 51 k; 4,096: 50 k; 4 x 2 threads 44 k). The M2's efficiency cores add less than its
  performance cores; a uniform many-core machine should scale further.
- **Fast algebra** (`track(fast=True)`: u'Du as one matrix product of the tensor's six components and
  the gradients' six products; both inverses from Cholesky factors): H 3x, a step 1.6x. Against
  float64 (`ukf_cpu_check.py` → `ukf_cpu_check.json`, 2,101 fibers) it is **closer than the binary's
  operation order in float32**: all points within 0.1 mm 90.2 % (75.9 %), ends within 0.06 mm 93.4 %
  (83.9 %), same point count 95.5 % (89.4 %), density r 0.994 (0.987) - fewer roundings, as the GPU
  kernels showed. Batch 1,024 against 50,000: the same to the fourth decimal.
- **8 workers, fast: 79 k steps/s, 4x the CPU tracker's 19.7 k.** The pipeline's CPU setting
  (`_pipeline.track(device="cpu")`): float32, fast, batch 1,024, one process per core. float64 stays
  the binary's arithmetic and order, the reference.
- **The pipeline on the CPU, again** (`cpu_timing.json`): **259 s** scan to payload (714 before; GPU
  61 s) - UKF 121 s (94 k steps/s over the whole brain), field estimate 77 s (float64), TractCloud
  58 s. CPU against GPU: field 0.07 / 0.5 mm (median / 99th), fibers 41,898 / 41,895, tract mix r
  0.994, single-draw labels agreeing 0.757 on the 31,145 streamlines tracked from the same seed voxel
  (matching by seed index paired different seeds - each run's seeds come from its own mask - and gave
  0.39). For scale, TractCloud's own single draws on one tractogram agree 0.82-0.88 (tract mix r
  0.989-0.998): the tract mix is within TractCloud's own spread; per streamline, the CPU and GPU
  pipelines differ beyond it, by the tracking and field differences on top of the draw.

## 2026-10-02 The CPU path's other two stages: TractCloud 10x, the field estimate 2.3x

- **TractCloud on the CPU** (58 s): the network is all of it (features and context 0.3 s), and 67 %
  of the network is conv5 (512 → 1024 channels, 1x1). PyTorch's CPU 1x1 convolution without oneDNN
  (macOS builds) runs at 16 GFLOPS; the same product through BLAS (Accelerate) at 755: 1,024 → 21 ms
  per 1,024 streamlines. `_tractcloud.MatmulDGCNN`: TractCloud's weights, every 1x1 convolution with
  its BatchNorm folded into one matrix product, each edge convolution W [f_j - x_i; x_i] split as
  Wa f_j + (Wb - Wa) x_i (the neighbors' term once per point, gathered), the max over neighbors taken
  before the monotone terms (exact). **50.7 → 5.0 s.** In float64 it equals upstream's forward to
  1.5e-13 (algebraically exact); in float32 to 6e-5 typically, more where float32 flips a near-tie in
  the graph layers' nearest neighbors. PAT16: all 31,993 tract labels and top clusters identical; the
  rank field's ranks identical for 99.99 % of streamlines, gaps 99.86 %, tails 98.6 %. The CPU
  Labeler's default; the GPU keeps upstream's forward.
- **The field estimate on the CPU** (77 s float64; 61 s float32): trilinear sampling and its gradient
  are 47 %, and the full-resolution levels 6-9 - motion held - 52 of 59 s. `interp="linear_pe"`: at
  those levels the b0s are moved once, then sampled linearly along the phase-encoding axis (two gathers
  a voxel). **26 s** in float32 (the GPU's trilinear: 27.6 s). Its field is 0.12 / 0.81 mm (median /
  99th) from the GPU's, against topup 0.30 / 2.51 mm, r 0.962 (GPU 0.27 / 2.36, 0.964); against the
  T1 - the arbiter - no worse: brain 0.36 / 0.88 / 1.57, tumor 0.27 / 0.80 / 1.34, margin 0.38 / 0.92
  / 2.13 mm, against the GPU's 0.38 / 0.94 / 1.73, 0.28 / 0.85 / 1.67, 0.38 / 0.96 / 2.45 (within the
  measure's precision, ~0.46 mm at the brain's 99th). The CPU pipeline's setting; the GPU keeps
  trilinear.
- **The pipeline on the CPU, end to end** (`cpu_timing.json`): **155 s** scan to payload (714 at the
  start of the day, 259 after the tracker; the GPU 61 s): field estimate 27.5 s, UKF 120 s (96 k
  steps/s, 77 % of the total now), TractCloud 5.0 s, the rest 3 s. CPU against GPU: field 0.12 /
  0.81 mm, fibers 41,910 / 41,895, tract mix r 0.996 (within TractCloud's own 0.989-0.998), single-draw
  labels 0.749 on 30,706 same-seed streamlines (TractCloud's own single draws: 0.82-0.88).

## 2026-10-02 The CPU tracker on a many-core x86 machine (Modal, CPU only)

`modal_cpu_scaling.py` → `modal_cpu_scaling.json`: one container, `cpu=32` (the container saw 48
vCPUs; CPU model not exposed), 32 GiB, PAT16 on the CPU path, ~7 minutes, about $0.20. The function's
returned dict did not unpickle locally (no torch in the Modal client's environment); the numbers are
transcribed from its log, and it returns JSON text now.
- **Tracking, the whole brain** (float32, fast, batch 1,024): 8 workers 80.2 s (143 k steps/s),
  16 50.2 s (228 k), **32 38.5 s (297 k)**, 64 46.1 s (248 k: more workers than vCPUs). Fibers
  identical across 8, 16, 32 and 64 workers (41,941; the M2's CPU run 41,910, its GPU run 41,895: x86
  vector math rounds otherwise). 297 k steps/s is 3.1x the M2's 8 workers (96 k) and 70 % of the M2's
  Metal kernel (428 k); scaling from 8 to 32 workers 2.1x.
- **The pipeline at 32 workers**: 147 s scan to payload - UKF 38.5 s, TractCloud 5.0 s, but the field
  estimate 101.8 s against 33.7 s for the same estimate earlier in the same container (same code and
  input); unexplained, possibly contention on a shared host. With the earlier estimate the pipeline
  would be ~84 s. Not rerun (frugal); worth a second look before quoting an x86 end-to-end number.
- **The 101.8 s estimate, explained** (`modal_cpu_field_timing.json`, a second container): the field
  estimate on that machine by threads, fresh process: 8 → 43.3 s, **16 → 36.5 s**, 32 → 41.3 s, 48 →
  72.5 s; and after a 32-worker tracking pool, torch's thread count in the parent read **48** although
  32 had been set, and the estimate took 80.7 s. So the first run's pipeline estimate ran on all 48
  vCPUs (likely 24 physical cores), after its pools. Not reproduced on macOS (the count stays as set
  through pools); likely the Linux build's OpenMP runtime. Fix: each CPU stage sets its own threads
  (`_pipeline.threads`), the estimate min(16, cores) (`ESTIMATE_THREADS`).
- **The pipeline again, threads set per stage** (`modal_cpu_pipeline.json`, a third container):
  **127.6 s** - estimate 43.3 s (16 threads), UKF 68.1 s, TractCloud 7.6 s, preparation 4.8 s (numba
  compiling `_median` in a fresh container: its on-disk cache is per machine). This container was
  slower across the board: tracking at 32 workers took 38.5, 46.7 and 68.1 s in the three containers
  (1.8x), TractCloud 5.0, 4.7 and 7.6 s. Modal's CPU type is not chosen or visible here; quote x86
  numbers as ranges. Scan to payload on 32 x86 cores: roughly 90-130 s (the M2's CPU: 155 s).

## 2026-10-02 The pipeline without numba, dipy, pynrrd, rankfield or numcodecs

- **The mask in torch** (`_median.py`): numba's sliding-window histogram (0.2 s, a 137 MB dependency
  pinning numpy) replaced by the exact torch version - every voxel's 9^3 window as an unfold view, its
  median by slabs, scipy's "reflect" padding by index arithmetic - and DIPY's Otsu by its own dozen
  lines (scikit-image's). `median_check.json`: identical to `dipy.segment.mask.median_otsu` on all 14
  volumes on the CPU and on the GPU, thresholds identical, the filter identical to SciPy's on the random
  volumes; 3.5 s (CPU) / 3.0 s (MPS) against DIPY's 10.3. The mask runs on the pipeline's device.
- **The payload out of the default path**: `_pipeline.run(encode=False)` by default (rankfield,
  numcodecs and `_geometry` only with `encode=True`, the format work's); the TRX's rank-field arrays
  only with `rank_field=True`, encoded inside `_trx.write` (`trx_check.py` asks for both: all checks
  pass again). pynrrd only where NRRD files are written (`_prep.prep`, `_ukf_torch.load`).
- **Checked on every path**: `dependency_check.py` → `dependency_check.json` runs the pipeline with
  numba, dipy, nrrd, rankfield, numcodecs and sklearn unimportable (probes allowed: torch._dynamo
  probes for optional packages), GPU and CPU paths, TRX written: no blocked module loaded; fibers as
  before on each path (GPU 41,895, CPU 41,910); scan to labels 64.1 s (GPU), 157.9 s (CPU). On Modal
  with an image of torch (CPU wheel), numpy, scipy and nibabel only (`modal_cpu_pipeline.json`): the
  CPU path in **92.2 s** on 32 x86 cores (estimate 38.2 s at 16 threads, UKF 49.2 s, TractCloud
  3.3 s, mask 0.7 s; no numba compile).
- **The 12-patient cohort again, on the GPU path with the torch mask**: all 12 reproduce their committed
  results exactly (timings aside). The mask costs 3.1 s against 0.65 (median); the other stages ran
  6-8 % slower this time (the laptop, not the change: UKF 27.2 → 29.4 s, estimate 29.4 → 31.2 s).

## 2026-10-02 The M2's GPU path again: the CPU speedups do not transfer

GPU timings on the M2 move with what the display is doing (WindowServer shares the GPU), so variants
were timed alternately in one session:
- field estimate: trilinear 33.5 / 30.3 s, `linear_pe` 30.6 / 28.6 s (~7 %; 2.3x on the CPU) - MPS's
  grid_sample is already efficient;
- TractCloud: upstream 4.9 / 4.9 / 4.2 s, `MatmulDGCNN` 5.6 / 4.6 / 4.7 s (none; 10x on the CPU) -
  MPS's convolutions are not the CPU's slow path;
- the mask: MPS 3.1 s, CPU 3.6 s.
- one grid_sample call for all eight b0s instead of a loop: the sampling itself bit-identical (CPU
  float64, MPS float32), 17 % faster on the CPU, 5 % on MPS. But the **whole estimate is not**: the
  gradient sums the volumes in another order, L-BFGS takes another path, and the field ends up to
  56 Hz different somewhere (MPS; 64 Hz on the CPU) - the estimate's path sensitivity again, from
  summation order alone. Not adopted (small gain, and it would move the committed results).
The GPU path stays ~60-65 s: estimate ~29 s, UKF ~27 s. Remaining levers: tracking on the GPU and the
CPU at once (the CPU's 96 k steps/s beside the GPU's 428 k), the Metal kernel's inverses split
(5-8 % of tracking), a Metal kernel for the mask (3 s), and the estimate's convergence.

## 2026-10-02 The field estimate's convergence: why it moves, and what fixing it costs

`susc_convergence.py` (`results/susc_convergence*.json`; estimate(diagnostics=True) logs each level's
iterations and field): two runs on PAT16's b0s, as they are and plus noise of SD 0.01, on the GPU.
- **L-BFGS never converges.** Every level stops at its cap (15, 30, 60 iterations at iter_scale 3) -
  and still at 50-200 with iter_scale 10: in float32 the cost's rounding keeps the line search finding
  "improvements", so the tolerances never fire. The result is wherever the cap falls.
- **The runs diverge at full resolution.** Coarse levels 1-5 agree within 0.2 mm (99th); at level 6,
  the first at full resolution, the two jump to 0.42 mm deep in the brain, ending at 0.65 (deep) /
  0.84 (edge). Motion, estimated only at the coarse levels, differs by up to 0.26 and is then frozen.
- **More iterations alone make it worse**: iter_scale 10 fits better (SSD 256 → 196) and settles the
  motion (0.085), but topup's schedule leaves the fine levels almost unregularized (lambda 5e-10,
  1e-11), and the field grows local features that differ between runs: edge 99th 2.3 mm, max 17.8. The
  capped iterations were the regularization.
- **What moves**: with converged, regularized fits (iter_scale 10, lam_scale 1e4) only 70 of 94,518
  brain voxels differ by > 2 mm, 59 of them one blob at the orbitofrontal base (b0 signal 68 against the
  brain's 202): signal lost in both phase-encoding directions, where only the regularization can set
  the field.
- **Settings, PAT16** (stability deep / edge, median / 99th / max mm; against topup median / 99th; s):
  current (3, 1) 0.067/0.65/1.5, 0.069/0.84/3.1, 0.27/2.36, 33; (10, 1e4) 0.021/0.21/3.4,
  0.022/0.52/12.0, 0.22/2.12, 104; **(10, 1e5) 0.029/0.27/0.77, 0.027/0.35/1.31**, 0.23/2.60, 102;
  (10, 1e6) 0.042/0.45/1.8, 0.051/0.65/2.6, 0.24/2.63, 102; (5, 1e4) 0.061/0.67/2.7, 0.069/0.86/4.3,
  0.23/2.51, 52; **(5, 1e5) 0.050/0.35/1.04, 0.059/0.56/2.1**, 0.23/2.50, 52. Against the T1 (99th:
  brain / tumor / margin): current 1.73 / 1.67 / 2.45; (5, 1e5) 1.60 / 1.08 / 1.71 (second run 1.63 /
  1.07 / 1.88); (10, 1e5) 1.59 / 1.20 / 1.87.
- **On three more patients** (`susc_stability.py` → `results/susc_stability/`; current → (5, 1e5);
  stability deep / edge 99th; T1 tumor / margin 99th): PAT13 0.27 → 0.25, 0.41 → 0.44; T1 2.25 → 2.40,
  1.96 → 2.14 (and the tumor's median 0.89 → 1.21: worse) - PAT23 0.30 → 0.19, 0.51 → 0.33; T1
  1.72 → 1.63, 1.36 → 1.40 - PAT14 0.44 → 0.38, 0.46 → 0.47; T1 0.45 → 0.45, 0.45 → 0.48.
  **Not uniformly better**: stronger regularization steadies the field where the data are weak (PAT16's
  dropout, PAT23) and blunts it where the true field turns sharply (PAT13's skull base). PAT16 is the
  extreme case; the other three were already stable at the current settings. Not adopted.

### What the literature says about it (2026-10-02)

- **topup itself has no convergence criterion** (its user guide: "At present there is no proper
  convergence criterion implemented in topup. Instead a fixed number of iterations is used"), and
  b02b0.cnf's values were "found to be useful for registering a set of good quality b=0 images", with no
  further justification. So the capped iterations are topup's design too - but topup's minimizer is
  Gauss-Newton with an explicit Hessian (--minmet 0; scaled conjugate gradient optional), which gets
  far closer to a minimum in its 5-20 iterations than our L-BFGS does in the same count. Our
  "faithful" port changed the optimizer, and the schedule was tuned for the other one.
- **HySCO** (Ruthotto et al. 2012, Phys Med Biol, doi:10.1088/0031-9155/57/18/5715; HySCO2 in SPM's
  ACID toolbox; Macdonald & Ruthotto 2018, J Math Imaging Vis, ADMM with proven convergence on the
  non-convex problem): Gauss-Newton on the same physical model with a "tailored nonlinear
  regularization functional" that keeps the transformation diffeomorphic - it penalizes the
  intensity-modulation factor 1 +/- d_pe u approaching zero, exactly the pile-up and dropout regions
  where our runs diverge - giving "meaningful, i.e. diffeomorphic, geometric transformations,
  independent of the actual choice of the regularization parameters".
- **SuCor** (Chigurupati & Garyfallidis, arXiv 2603.16758, March 2026): per phase-encoding column, the
  displacement as the Wasserstein-2 barycentre between the two polarities' profiles (closed form by
  quantile matching), then a bending-energy fit whose strength is set by the Morozov discrepancy
  principle - the regularized field deviates from the raw one by 1.5 x the background noise (MAD) -
  so no tuned lambda. HCP: mutual information with T1 0.341 against topup's 0.317, 12 s on one CPU
  core against 55 min, but LR-RL consistency below topup's ("residual per-column variability"), and
  the authors name the same smoothness-against-fidelity tension we measured. No dropout handling.
- **DR-BUDDI** (Irfanoglu et al. 2015, NeuroImage, doi:10.1016/j.neuroimage.2014.11.042): uses the DWIs
  and a structural image to guide the registration and does not force exact up/down symmetry, robust
  where motion, ghosting and low SNR break the model; better in the brainstem.
- **Evaluations** (Graham et al. 2017, PLoS One, doi:10.1371/journal.pone.0185647; Gu & Eklund 2019,
  Front Neuroinform, doi:10.3389/fninf.2019.00076): reversed phase-encoding methods correct best; the
  LR-vs-AP difference is a usable but imperfect proxy; none of the common methods models the
  susceptibility field's interaction with head motion.
- **Uncertainty**: field-map approaches have used per-voxel confidence (from phase unwrapping) to
  control the deformation's smoothness locally - the adaptive regularization idea, from the field-map
  side. No reversed-phase-encoding method found that reports per-voxel uncertainty.

## 2026-10-02 The field estimate by Gauss-Newton: converged, steadier, closer to the T1, faster

`_susc.gauss_newton()` (estimate(optimizer="gn")): Levenberg-Marquardt Gauss-Newton on the same cost,
matrix-free - the residual's Jacobian from the sampled images' derivative by position (autograd,
pointwise), the B-spline basis and the Jacobian factor's derivative, plus the rigid parameters at the
motion levels - each step by preconditioned conjugate gradients (the Gauss-Newton matrix's diagonal,
separably), stopping on the cost's relative decrease. Gradient and Gauss-Newton product checked against
finite differences in float64 (exact to 6 digits at the full-resolution levels, trilinear, linear and
cubic; within trilinear's kinks at the motion levels).
- **Plain Gauss-Newton converges where L-BFGS did not, and folds.** PAT16: SSD 33 against 135 at level
  1, 178 against 256 at the end; the two runs (as is / plus noise 0.01) agree to 0.16 mm deep (99th;
  L-BFGS 0.65) - but outside the brain the field runs to 1760 Hz with 13,242 folded voxels (a polarity's
  Jacobian 1 -/+ readout dh/dpe below zero; topup 0, L-BFGS 13): with nothing to fit, the cost falls by
  pushing signal around.
- **HySCO's anti-folding penalty** (fold_phi: x^4 / (1 - x^2) of the displacement's derivative along
  the phase-encoding axis, weighted as the bending energy by the current SSD; estimate(fold=10)) removes
  every fold, in the brain and out. Weight 1 left PAT16 less stable (deep 99th 0.47); 10 is used.
  The field outside the brain still drifts further than L-BFGS's or topup's (34-75 mm against 15-20) -
  harmless to the brain so far, to be addressed.
- **topup's own iteration counts suffice** (iter_scale 1): it is the schedule tuned for this optimizer.
- **Speed** (PAT16, M2): 116 s plain; the bending energy as one contraction by per-axis Gram matrices
  (exact); stopping at a relative decrease of 1e-4 (from level 7 on, iterations moved the cost 1e-4
  and the field 0.03 voxel at the 99th - a few voxels outside the brain kept them going); conjugate
  gradients to 0.1, at most 30, tested every 5th (each test a GPU sync): 27 s. The subsampled levels
  are launch-bound on the GPU (22 ms a CG step on a grid 8x smaller than full resolution's 16 ms; 3x
  faster on the CPU), so estimate(coarse_device="cpu") runs them there: **18 s**. Against the
  tight-tolerance field: 0.08 mm deep, 0.27 edge (99th) - the size of the run-to-run spread.
  CPU only, float32: 22.5 s trilinear, 27.6 s linear_pe - with one sampling per iteration instead of
  L-BFGS's many evaluations, linear_pe is no longer faster, and trilinear gives the GPU's field (0.08
  mm deep, against linear_pe's 0.38).
- **Four patients** (`susc_stability.py`, `results/susc_stability/*_gn_fast.json`; L-BFGS → GN fast,
  fold 10, iter_scale 1, coarse levels on the CPU; stability deep / edge 99th; T1 brain / tumor /
  margin 99th; s): PAT16 0.65/0.84 → 0.06/0.16, 1.73/1.67/2.45 → 1.63/1.33/1.90, 33 → 18 - PAT13
  0.27/0.41 → 0.04/0.07, 1.46/3.35/3.41 → 1.40/2.76/2.87, 34 → 23 - PAT23 0.30/0.51 → 0.08/0.16,
  2.49/3.62/5.20 → 2.27/3.60/3.92, 32 → 20 - PAT14 0.44/0.46 → 0.08/0.12, 1.69/0.54/0.62 →
  1.56/0.39/0.67, 31 → 17. No folds. Better or equal everywhere but PAT14's margin (0.05 mm).
- Not yet the default: the noise-0.01 rerun measures the optimizer's sensitivity, not repeatability
  from independent data. Next: split-half and held-out tests from the scan's own b0s (6 AP spread over
  the 15-minute scan, 2 PA), on all 12 patients, then the default and the cohort rerun. The default
  L-BFGS path still reproduces the cohort's field exactly.

## 2026-10-02 Repeatability from independent b0s: Gauss-Newton predicts better, is no steadier at the edge

The noise-0.01 rerun measured the optimizer's sensitivity, not repeatability. `susc_held_out.py`
(`results/susc_held_out/`, `susc_held_out_summary.py`) uses each scan's own b0s - 6 AP spread over the
~15-minute series (volumes 0, 1, 26, 51, 76, 101), 2 PA back to back just before it: fits A = {AP0,
PA0}, B = {AP1, PA1} (one TR apart: independent noise, little motion), C = {AP last, PA1}; split-half
(A's field against B's), held-out (each fit corrects the other pair: RMS of corrected AP minus PA,
relative to the brain's mean), drift (A against C, carried by the full fit's motion). Twelve patients,
medians (`summary.json`):
- **Held-out residual: Gauss-Newton better on all 12**, 0.132 against L-BFGS's 0.145 (deep 0.120 /
  0.131, edge 0.138 / 0.153); uncorrected ~0.5. It is still 2-3x the same-polarity mismatch (AP0
  against AP1, ~0.05 on still patients): model error, not noise, limits the correction.
- **Split-half: no steadier.** Deep 99th 0.76 against 0.78 mm, edge 1.34 against 1.21, deep max 2.96
  against 2.68; drift (15 min) deep 1.32 / 1.33, edge 2.31 / 2.03. With two volumes a fit, Gauss-
  Newton's closer fit carries the noise at the edge. The noise-0.01 test (4-10x) overstated it.
- **No folds** (L-BFGS: 27 brain voxels over the 12 A-fits).
- **PAT08 moves within volumes**: AP0 against AP1, one TR apart, 0.24 (others ~0.05), worst in
  alternating even slices - interleaved acquisition, the head moving between slice groups, which a
  rigid per-volume model cannot follow. Its residual is 0.23 for every estimator. PAT23's mismatch is
  0.12.
- **Regularization by cross-validation** (`*_lam.json`, `summary_lam.json`; GN, topup's lambda x1e3 /
  1e4 / 1e5): the held-out residual rises with lambda on 11 of 12 (0.1318 → 0.1324 / 0.1325 / 0.1333),
  and the split-half barely moves (edge 99th 1.34 → 1.30 / 1.21 / 1.24; deep max no better): smoother
  fields do not remove the edge's variance - dropout, motion between slice groups and two-volume fits
  are likelier sources. topup's lambda is the cross-validated choice; a Morozov lambda is no longer a
  priority. x1e4 would buy L-BFGS's edge repeatability for 0.5 % of the residual.

## 2026-10-02 Gauss-Newton is the default: the cohort again

`_susc.estimate`'s defaults are now the held-out test's choice: optimizer "gn", fold 10, topup's
iteration counts and lambda, the gauss_newton() tolerances of "by Gauss-Newton", the subsampled levels
on the CPU under "mps" (coarse_device "auto"). The CPU path drops linear_pe: trilinear in float32,
22.6 s, the GPU's field to 0.03 mm (brain 99th; L-BFGS's two paths differed by 0.38). L-BFGS stays as
optimizer="lbfgs" (iter_scale 3 by default there); susc_stability.py's [iter_scale, lam_scale] configs
and susc_held_out.py's "lbfgs" mean it. The new defaults reproduce the tuned run bit for bit.
`cohort.py` on the 12 patients (`results/cohort/`, `cohort_summary.md`; the L-BFGS fields kept as
derived/<sub>/cohort_fields_lbfgs.npz), L-BFGS → Gauss-Newton, medians:
- against the T1, 99th percentile: brain 1.71 → 1.62 mm (better on 10 of 12), tumor 1.65 → 1.39
  (8 of 12), tumor margin 1.75 → 1.58 (10 of 12). Largest gains where L-BFGS did worst: PAT23's
  margin 5.20 → 3.85, PAT13's tumor 3.35 → 2.61, PAT16's margin 2.45 → 1.97. Worse: PAT25's tumor
  2.38 → 2.76, PAT19's 2.93 → 3.04, PAT05's 1.08 → 1.21 (margin 1.08 → 1.26), PAT14's margin
  0.62 → 0.72;
- the noise-0.01 rerun at the margin's 99th 0.33 → 0.07 mm (all 12); the half-split T1 fit 0.40 → 0.39
  (it measures the T1 check's precision more than the field's);
- field estimate 31.2 → 17.8 s, scan to labels 71 → 56 s (PAT16: 17.5 s, 53 s).

## 2026-10-02 The open items after the switch: where the edge varies, PAT25, the drift, PAT08

- **The edge's split-half variance is the skull base's lost signal.** PAT16 and PAT23, fits A / B:
  of the edge voxels whose two fields differ by > 1 mm, 90-93 % have one polarity below half the
  brain's median signal (19-25 % of the edge overall) and 95-97 % lie in the brain's lowest quarter
  (36-37 %), where the displacement's derivative along the phase encoding is 0.23-0.53 (edge overall
  0.07): pile-up and dropout, where the data do not set the field. Gauss-Newton fits it sharper than
  L-BFGS did (PAT23 edge 99th 2.02 against 1.09 mm).
- **Penalties against it, by the held-out test** (`*_reg.json`; medians over 12, against the default):
  fold 30 / 100 - held-out 0.1335 / 0.1361 against 0.1318 (worse on 11 / 12 of 12), split-half edge
  99th 1.17 / 1.04 against 1.34; a first-derivative ("membrane") penalty, new as
  estimate(membrane=...), which unlike the bending energy charges a linear ramp - 1e-4: held-out
  0.1318 (better on 9 of 12), edge 1.27, drift outside the brain 39 mm (median max); 1e-3: 0.1326,
  1.11, 27 mm. A trade, no free win; the default stays (membrane off) - a cohort rerun would buy the
  edge 0.1-0.3 mm at the held-out's expense.
- **The drift outside the brain** is far from it (voxels > 20 mm displaced: median 42 mm from the
  brain, in the lowest slices - neck, face, sinuses); within 5 mm of the brain the 99th percentile is
  11.7 mm against L-BFGS's 11.0 (PAT16) and 9.6 (PAT23). Linear ramps where nothing constrains the field,
  which bending energy does not charge; membrane 1e-3 halves it. Harmless to the brain as measured.
- **PAT25's tumor against topup** (`cohort/PAT25_topup.json`; FSL topup as a reference, cohort_topup.py
  now with --reuse, empty-region-safe, and the kept L-BFGS fields): Gauss-Newton's field is closer to
  topup's than L-BFGS's was (tumor 99th 2.12 against 2.98 mm; brain 1.98 against 3.06). Against the T1,
  99th, brain / margin / tumor: Gauss-Newton 1.91 / 2.03 / 2.76, L-BFGS 2.37 / 2.11 / 2.41, topup
  2.02 / 2.19 / 1.95. The tumor (1,057 voxels, skull base, bright edema, 5-13 mm displaced) is where
  the three disagree; elsewhere Gauss-Newton is the closest of the three. The same data-limited region.
- **PAT08 is not slice-group motion** (correcting the entry above). Its consecutive b0s differ evenly
  across slices (even/odd ratio 1.00; the alternation index 0.35-0.41 within the others' 0.13-0.47),
  and its estimated motion between them (0.3-0.4 mm, 0.14 deg) is PAT16's. The background noise is the
  same for all patients (SD ~4.5); the difference sits in the brightest b0 decile (54 % of its energy;
  CSF), worst at the ventricles (slices 26-34), no N/2 ghost: CSF flow or pulsation, 4x the others'.
  Physiology the model has no term for; every estimator sees it alike, and the same-polarity mismatch
  calibrates for it.
