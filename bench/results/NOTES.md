# Tractography incubation: measured results

The journal as it was written in medseg (`bench/tractography`), names and paths unchanged. Since
tractline (2026-10-02) the modules are the package's: `_pipeline` → `tractline.pipeline`, `_susc` →
`susceptibility`, `_prep` → `prep`, `_median` → `mask`, `_resample` → `resample`, `_t1check` →
`t1check`, `_trx` → `trx`, `_data` → `data`, `_ukf_torch` → `ukf`, `_ukf_metal` / `_ukf_triton` /
`_ukf_triton_block` → `ukf_metal` / `ukf_triton` / `ukf_triton_block`, `_tractcloud` →
`labelers.tractcloud`; scripts and `results/` are under `bench/`. The format work (M0-M4, the payload,
`_geometry`, `_groups`, the viewers, `dmc_field.py`) and the albula comparisons stayed in medseg with
their result files; their entries below are kept as the record.

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
  displacement as the Wasserstein-2 barycenter between the two polarities' profiles (closed form by
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

## 2026-10-02 Gauss-Newton on 32 x86 cores (Modal, CPU only): no faster there

`modal run bench/modal_cpu_scaling.py --what field` (`results/modal_cpu_field_timing.json`,
48 vCPU visible): the estimate by threads 8 / 16 / 32 / 48 - 48.9 / 40.2 / 44.6 / 51.5 s (16 best, as
before: ESTIMATE_THREADS stays); 41.6 s after a tracking pool. The pipeline end to end: **95.1 s** scan
to labels (estimate 38.5 s, UKF 50.7, TractCloud 4.1), against 92.2 s with L-BFGS (estimate 38.2):
on this machine Gauss-Newton costs what L-BFGS did - its gain here is accuracy, not time. On the M2
the same estimate takes 22.6 s on the CPU and 18 s with the GPU; the x86 run's per-step cost (small
tensor contractions, many syncs into torch's thread pool) is the likely difference, not measured.

## 2026-10-02 Can the rank field be rebuilt from a TRX's top-k? (for the record; not the plan)

Decided instead: tractline's TRX stays plain, for compatibility only (tracts as groups, tract
probabilities); rankfield, when it returns, is wired into the exporter and encodes from the labeler's
in-memory log-probabilities, as `encode_payload` does - no format arrays or format dependencies in the
TRX. The test, on 50,000 streamlines of HCP 101006's TractCloud log-probabilities (float16, as the
pipeline holds them), at the pipeline's field (keep "clip", depth 6, clip 8, temperature 1): from
each streamline's top 6 classes alone (stable order, ties to the lower index) the rank and gap-byte
planes are identical to the full encode; the tail (dropped mass) is not (96 % differ) - it needs the
sum over all 1,600 classes; with that one number stored, 49,997 of 50,000 tails match, 3 differ by up
to 21 / 65,535 (not chased). Deeper fields, the "shell" rule and other tail temperatures would need
the full log-probabilities.

## 2026-10-02 The pipeline on CUDA (Modal: A10, A10G, L40S); HARDI on three paths

`pipeline.run(device="cuda")`: the field estimate in float32 on the GPU (all levels), the Triton block
kernel for the tracker, TractCloud on CUDA. `modal_gpu_pipeline.py` (PAT16, two runs per container,
results/modal_gpu_pipeline_*.json), `gpu_pipeline_compare.py` (against the M2, gpu_pipeline_compare.json).
- **Time, PAT16, scan to labels, steady state:** A10 16.5 s (field 4.8, UKF 8.9, TractCloud 1.2), A10G
  17.5 s, L40S 12.2 s (9.6 with TF32) - against the M2's 53 s and 32 x86 cores' 94 s. The first run in
  a container pays the Triton compile when the Volume's cache misses (~200 s; cached: ~20 s).
- **TF32.** PyTorch lets cuDNN run float32 convolutions as TF32 by default on Ampere and newer.
  `pipeline.exact_float32` now turns it (and matmul's) off for the pipeline's stages on CUDA and
  restores it after: on the same card it moves PAT16's Other fraction by ~1.2 points (A10 62.6 →
  61.5 %, L40S 64.0 → 62.8 %) and the field by up to 1 mm; it costs nothing on the A10, 2.6 s on the
  L40S (the field estimate's convolutions). With it off, TractCloud on CUDA labels a tractogram exactly
  as the M2 does: the A10's 42,169 fibers, 32,268 of 32,268 labels identical.
- **Determinism.** Within a card type every run is bit-identical (two containers x two runs on an A10:
  one field hash, one fiber hash). Modal's "A10" lands on an A10 or an A10G; the two differ in
  float32 rounding (42,169 against 42,150 fibers), as an L40S does.
- **The Other fraction is sensitive on PAT16.** Fields agree with the M2's within the estimate's own
  rounding sensitivity (deep 99th 0.05-0.08 mm), fibers within ~20 of 42,170, but the share TractCloud
  labels Other spans 58-63 %: M2 (Metal) 58.1 %, A10G 59.9 %, A10 61.5 %, L40S 62.8 %. On the A10's own
  field: the float64 tracker (CPU) 62.2 %, Triton 61.5 %, Metal 58.8 % - the field moves it 0.7 points,
  the tracker's float32 rounding the rest, and Metal is the farthest from float64. Open: the label
  noise floor on PAT16 (the scan's own noise, as ukf_noise_floor.py measured fibers) - whether 3-4
  points of Other is within it, and if not, why Metal's rounding leans one way.
- **HARDI on three paths** (`hardi_paths.py`, `modal_hardi_paths.py`; results/hardi_paths_*.json; no
  correction - no reversed pair; prep, UKF, TractCloud): M2 Metal 105.3 s (UKF 93.9), L40S 39.7 s (UKF
  36.5; first run 49.9), 32 x86 cores 119.5 s (UKF 110.4). 78,404-78,417 fibers; Other 76.6-77.4 % -
  on this scan the paths agree within 0.8 points. Repeat runs bit-identical on the M2 and the L40S.

## 2026-10-02 PAT16's Other fraction: TractCloud's draw, not the tracker (`label_noise_floor.py`)

The M2's field held fixed; `results/label_noise_floor.json`.
- **TractCloud's own draws** (one Metal tractogram, context draws 0-9): Other 58.1, 59.2, 55.5, 56.2,
  **66.2**, 56.3, 56.4, 62.2, 55.9, 56.9 % - a single draw's aggregate moves by 11 points with the
  fibers unchanged.
- **Metal against float64, the scan as acquired:** 42,170 / 42,171 fibers, Other 58.1 / 58.8 %; matched
  by seed, 88.8 % of 32,228 labeled streamlines agree, the flips go both ways (1,460 Metal-Other /
  float64-tract, 1,686 the reverse), the flipping fibers' lengths do not differ (median 0.0 mm).
- **Rounding alone:** float64 at batch 1,024 and 4,096 - fibers not bit-identical, Other 58.8 % both.
- **The scan's noise** (wild bootstrap of the corrected tracker input, 4 replicates, each tracked by
  Metal and float64): Other 57.8-61.0 % (Metal), 57.9-59.9 % (float64); Metal minus float64 +1.1,
  +1.1, -1.0, -0.5 points - the sign changes, no bias. Caveat: the b0s' residuals (SD 67, against 6.2
  for the DW volumes) carry the 15-minute series' motion and CSF fluctuation, so the replicates perturb
  the b0s by the scan's variability, more than thermal noise.
- **So** the 58-63 % across cards and kernels ("The pipeline on CUDA") is inside both floors: any small
  change in the fibers acts as a new context draw. The earlier reading - Metal leaning one way - does
  not hold. The pipeline's single-draw labels are what is unstable (M0's 28 % per-streamline, now
  ±5 points in the aggregate on a clinical scan). Next: how labels and the aggregate settle with the
  number of draws (log-probabilities averaged), and what the pipeline should use.

## 2026-10-02 Why TractCloud's labels move: the shared global context, and 80 where the model was trained with 500

`label_draws.py` → `label_draws.json`, `label_kglobal.py` → `label_kglobal.json`; PAT16, the M2's field.
- **The global context drives the aggregate.** Each draw's context is each streamline's 20 nearest
  neighbors within a random 10 % subsample (its own) and 80 random streamlines (ONE set shared by every
  streamline of the draw). Over a 5 x 5 grid of local x global seeds the Other fraction's SD is 4.3
  points along the global axis and 0.1 along the local one: the shared 80 streamlines condition the
  whole tractogram's labels at once.
- **Ensembling over draws helps slowly** (two disjoint ensembles of K draws, Metal tractogram;
  per-streamline agreement, cluster probabilities averaged / tract mass averaged / majority vote): K 1
  85.7 / 86.6 / 85.7 %, K 3 90.5 / 91.5 / 89.4, K 5 92.4 / 93.2 / 91.9, K 10 94.9 / 95.4 / 94.2. Metal
  against float64 (seed-matched) 88.8 % at K 1, 94.2 % (cluster mean) / 94.9 % (mass) at K 10 - the same
  as one tractogram's self-consistency: the trackers differ by draw noise. The mass rule labels ~3-4
  points more Other than the cluster rule.
- **The released model was trained with k_global = 500** (TrainedModel/cli_args.txt; the paper's w =
  500) and local neighbors from all of a 10,000-streamline brain (k_ds_rate 1.0); upstream's packaged
  inference overrides to k_global = 80, k_ds_rate = 0.1, and this pipeline copied it. At 500 (one Metal
  tractogram, draws 0-9): Other 50.6-51.6 % per draw (SD 0.3 points; at 80: 55.5-66.2 %, SD 3.5);
  single-draw agreement 93-94 % (at 80: 75-88 %); disjoint 5-draw ensembles 97.5 % (at 80: 89.7 %);
  8.8 s a draw on the M2 (3.9). The two settings' 5-draw ensembles agree on 86.1 % of streamlines: 80
  is noisier AND shifted (+6-7 points of Other).
- **Accuracy is not settled here:** PAT16 has no reference labels; HCP 101006's file carries FreeSurfer
  region labels per point, not tract labels. TractCloud's labeled test subjects (TrainData_800clu800ol,
  of which only HCP_mass_center.npy was kept) would settle 80 against 500.

## 2026-10-02 The labeler at the trained context (k_global 500, 10,000 local candidates)

`labelers.tractcloud.Labeler` now runs TractCloud with the context the released model was trained with:
500 global streamlines, local neighbors among min(1, 10,000 / n) of the streamlines (the training
density); several draws average their cluster probabilities (was: a majority vote). `upstream=True`
restores upstream's inference context (80, 10 %) and reproduces the old labels exactly. Checked: one
draw = label_context.py's (Other 50.59 %), 5 draws = its ensemble (50.68 %); the CPU network agrees
with the GPU on every streamline. Time per draw on the M2 9.1 s (was 3.9); CPU 15.5 s.
- **Local setting** (`label_context.py`, k_global 500): 10 %, the training density (31 % here) and all
  streamlines give the same stability (Other SD 0.3 points, 5-draw ensembles 97.5-97.6 %) and agree on
  99.4-99.6 % of streamlines: only the global context matters.
- **Across trackers** (`label_trained_check.py`, PAT16, one draw, seed-matched Metal / float64): 93.9 %
  agree (at 80: 88.8 %); Other 50.6 / 51.1 %.
- **Another scan, HARDI** (healthy, 2 mm, 150 directions): Other over 5 draws 72.8-73.8 % (SD 0.4; at 80:
  75.0-78.3 %, SD 1.3); two single draws agree 93.9 % (at 80: 86.1 %). Smaller than PAT16's at 80, the
  same direction (-4 points of Other at 500).
- **Across cards** (`modal_gpu_pipeline.py`, `gpu_pipeline_compare.py`): Other M2 50.59 %, A10 50.60 %,
  L40S 50.93 %; tract mix r against the M2 0.9993 / 0.9992 (at 80: 0.987-0.997, Other 58-63 %). Scan to
  labels, steady: A10 17.9 s, L40S 10.2 s (TractCloud 2.0 / 1.3 s), M2 57.5 s. The L40S this time had
  49 GB (the last, 46 GB) and gave 42,169 fibers (42,161): card variants differ in rounding.
- **Ron's albula-diffusion port** (`tractcloud/tractcloud.ts`, read on GitHub) seeds its draws (mulberry32,
  fixed seed) after measuring the same instability (79 % kept their tract between draws on PAT16), and
  keeps upstream's 80 / 10 % (`model/make-model.py`): repeatable, but one arbitrary draw of an unstable
  setting; any change in the streamlines is a new draw.

## 2026-10-02 RapidParc as a second labeler: identical to its package

`labelers/rapidparc.py`: RapidParc's inference (github.com/MedVisBonn/RapidParc v1.0.4, BSD-3) written
here, ~100 lines of torch, no new dependencies (its safetensors read directly); the weights its release
v1.0.0 publishes (DATA/RapidParc: `rapidparc`, `hemiaug`; 6.6 MB each, sha256 as its package checks).
Its package pins pandas, scikit-learn, matplotlib, seaborn, openpyxl and an obsolete `pathlib` backport,
and its `__init__` imports its training and test code; labeling needs torch and the weights. Its
1,600 -> 43 mapping and tract names are TractCloud's (checked identical), so it returns the same Labels.
The context: shuffled groups of 2,000 streamlines, each re-scaled to [-1, 1] inside the network, as in
training (no training/inference mismatch). `labelers/base.py` holds what labelers share (Labels, the
40 mm cut).
- **Against its package** (`modal_rapidparc_check.py` → `rapidparc_check_modal.json`; both in one Modal
  container, torch 2.14.1; PAT16's A10 tractogram, 32,268 streamlines of 40 mm or more): for both
  models and seeds 0 and 42, the 1,600-cluster argmax identical on 32 x86 cores and on an A10G, every
  streamline. Ours 0.7-0.8 s a draw on the CPU, 0.3 s on the A10G (its package 1.2-3.2 / 0.6-2.4 s,
  reloading the model each call).
- PAT16's Other fraction with RapidParc: 49.7-49.8 % across seeds and models (TractCloud at 500: 50.6 %).
- A slip on the way: reading an .npz array inside a loop (`z["points"][a:b]` per streamline) re-reads the
  whole array each time - the first Modal run spent 8 minutes there before it was stopped.

## 2026-10-02 The labelers compared: 12 patients, and accuracy on TractCloud's own test split

**On the 12 patients** (`labeler_compare.py` → `labeler_compare.json`; each tracked once, the same
tractogram labeled four ways, seeds 0 and 1; medians): agreement between seeds - TractCloud at 80 84.1 %,
at 500 93.5 %, RapidParc 97.2 %, hemiaug 97.2 % (worst patient 81.4 / 90.9 / 96.1 / 96.2); Other's change
between seeds 2.3 / 0.4 / 0.04 / 0.09 points; Other 61.2 / 53.4 / 51.8 / 52.1 %; seconds a draw on the M2
4.2 / 9.6 / 0.8 / 0.7. Between labelers: TractCloud 500 and RapidParc agree on 83.5 % of streamlines
(min 76.2 %), tract mix r 0.990; RapidParc and hemiaug 93.6 %, r 0.999. (The cohort rerun with TractCloud
at 500 was stopped after 5 patients for this; its results were restored to the committed ones.)

**On TractCloud's test split** (`accuracy_tractcloud_test.py` → `accuracy_tractcloud_test.json`; its
release's TrainData_800clu800ol/test.pickle: 20 HCP subjects x 10,000 streamlines, ORG-atlas cluster
labels; each subject labeled as one tractogram; tract accuracy / macro F1 over 43 classes, seed 0):
- TractCloud at upstream's inference context (80, 10 %): **86.55 % / 82.74 %** (cluster accuracy 60.2 %).
- TractCloud at the trained context (500, all 10,000): **92.01 / 89.92** (seed 1 91.89 / 89.76; cluster
  76.7 %) - the TractCloud paper's numbers as RapidParc's paper tabulates them, 92.12 / 90.22.
- RapidParc **94.48 / 93.23** (seed 1 94.50 / 93.30; cluster 84.6 %) - its paper: 94.44 / 93.2.
  hemiaug 94.45 / 93.29 - its paper: 94.43 / 93.18.
- Seeds agree (tract): 87.4 / 95.5 / 98.4 / 98.2 %. Seconds a subject a draw: 1.3 / 3.2 / 0.13 / 0.12.
So both implementations reproduce their papers; upstream's packaged inference context costs TractCloud
5.5 points of accuracy and 7.2 of macro F1 on its own test data - not only noisier, more often wrong. On
healthy in-domain data RapidParc is 2.5 points more accurate than TractCloud at its best, steadier, ~25x
faster. RapidParc's paper on lesioned brains (hemispherotomy): both methods far lower (72-77 %), hemiaug
ahead of TractCloud by 7.5 points of accuracy post-surgery.

**Memory in a batch** (`memory_batch.py` → `memory_batch_{as_is,cleanup}.json`; four patients in one
process, the labelers loaded once): live GPU tensors stay at 0.03 GB - no leak - but PyTorch's MPS
allocator keeps freed blocks cached: the driver's memory 5.9 → 7.8 GB, the footprint 8.2 GB (and 13 GB
in labeler_compare's first run, 4 labelers x 2 seeds, which then swapped). gc + torch.mps.empty_cache
after each patient: 0.1 GB, footprint 0.3-2.3 GB, 4 % more time. `pipeline.release_memory(device)` does
it; `run()` calls it by default. TractCloud at 500 was the peak (batches of 1,024 streamlines: 5.9 GB);
its Labeler now takes batches of 256 - identical labels, as fast, 2.1 GB.

## 2026-10-02 RapidParc is the default; consistent across platforms

`pipeline.default_labeler(device)` = `labelers.rapidparc.Labeler` ("rapidparc" model); `run(s, None, ...)`
uses it. The shared 43-class scheme ships as package data (`labelers/scheme_43.json`, from RapidParc's
release files, identical to TractCloud's mapping), so the default path imports no TractCloud code:
`dependency_check.py` (TractCloud's `tractcloud` and `vtk` now blocked too) passes on the M2's GPU and CPU
paths, no blocked module loaded (59.1 s / 166.7 s). The label stage's timer is now "label" (was
"tractcloud"). TractCloud stays available (`labelers.tractcloud.Labeler`, at its trained context).
- **PAT16 on five platforms** (`modal_gpu_pipeline.py --gpus A10,L40S,CPU`, `gpu_pipeline_compare.py
  --m2-cpu`; against the M2's GPU path: fibers / labeled / Other / tract mix r / field deep 99th):
  M2 GPU 42,170 / 32,264 / 49.88 % / - / -, 56.5 s; M2 CPU 42,176 / 32,259 / 49.86 % / 0.99985 / 0.03 mm,
  166 s; A10 42,169 / 32,268 / 49.77 % / 0.99983 / 0.08 mm, 16.6 s; L40S 42,161 / 32,248 / 49.98 % /
  0.99990 / 0.07 mm, 12.4 s; 32 x86 cores 42,165 / 32,250 / 49.69 % / 0.99989 / 0.07 mm, 95 s. Other
  spans 0.3 points, the tract mix r >= 0.9998 (TractCloud at 500: r 0.9992; at upstream's 80: Other
  58-63 %). What remains is the trackers' float32 rounding; on identical fibers the labeler matches
  across CPU, Metal and CUDA.
- **A swap, and a guard.** The first five-platform run swapped the M2 (8 workers at 1.9 GB): its script
  had no `if __name__ == "__main__":` guard, and the CPU tracker's spawned workers re-import the main
  script - each re-ran the comparison's GPU pipeline. Guarded, the workers are ~240 MB each. The tracker
  now refuses to spawn from an unguarded script (`ukf._require_main_guard`: a clear error first; REPL and
  notebooks unaffected).

## 2026-10-03 The cohort with RapidParc: the new reference

`cohort.py` on the 12 patients (one process each, ~4.6 min a patient), labeling by the default
RapidParc; against the committed reference (Gauss-Newton, TractCloud at upstream's 80): **the field and
every T1 measure identical** (labeler-independent, as they should be); what changed is what labels:
- Other (ours): 0.56-0.68 → 0.50-0.62 (PAT16 0.583 → 0.498).
- Tract mix between the scan as acquired and as corrected: median r 0.9967 → 0.9968.
- **Tract centers moved by correction** (per tract, median / 90th / max mm): medians 1.9-3.0 → 1.4-1.9,
  maxima down sharply (PAT23 20.9 → 5.9, PAT07 17.6 → 5.8, PAT20 18.0 → 8.6). Part of the old movement
  was TractCloud's label noise - streamlines changing tract between the two arms' runs - not correction.
- `cohort_summary.md`: margin vs T1 99th 4.78 → 1.58 mm (unchanged), scan to labels 54 s (field 18 s,
  UKF 29 s, RapidParc ~1 s).
- `trx_check.py` with RapidParc's labels: all checks pass, both variants (42,170 streamlines float32 zip
  23.9 MB; 32,264 labeled, float16 directory 11.3 MB).

## 2026-10-03 An adversarial review of the code and documents, and the fixes

Four reviewers (the field estimate and preprocessing; the tracker and pipeline; the labelers; the
documents and bench) read the repository for defects. Fixed:
- **The field estimate**: PCG returns zero for a zero right-hand side (was 0/0); the Marquardt test is
  `not f1 < f0`, so a NaN step is never accepted; `estimate` refuses fewer than two distinct
  phase-encoding directions and non-finite b0s (identical b0s now give a finite zero field).
- **Preprocessing**: `prepare` names a missing shell or missing b0s; integers outside int16's range are
  stored as float32 (uint16 scans were wrapping).
- **The tracker**: the main-guard check (`ukf.require_main_guard`, was `_require_main_guard`) parses the
  script (an `if __name__ == "__main__":` at module level, either order or `in` a tuple; a comment no
  longer passes) and runs before any work; `pipeline.run` and `track` call it on the CPU path.
  `TRACTOGRAPHY_SHARE=0` (blocks sent as copies, for systems without shared memory) now sends copies -
  torch's pickling had moved each block's whole base tensor to shared memory anyway; the same fibers as
  shared memory on PAT16.
- **The labelers**: per-streamline lengths (one NaN point made every later streamline's length NaN in
  the cumulative sum; an empty streamline took a neighbor's length); empty Labels when nothing is 40 mm or longer (both labelers crashed);
  RapidParc pads the last group as often as needed (fewer than 1,000 streamlines failed) - checked at
  n = 1, 7, 999, 1,000, 2,500; `draws` may be any iterable (a generator was consumed by `len`); several
  draws are averaged in the log domain (`base.LogMean`; float32 probabilities underflowed to -inf);
  TractCloud refuses fewer than its k (20; 200 with `upstream=True`) streamlines.
- **TractCloud's context, as trained** (`trained_context`): the previous Labeler passed k_ds_rate
  10,000 / n to upstream's `RealDataDataset`, which subsamples *within consecutive ~10,000-streamline
  index chunks* - slabs of the brain in seed order - so above 10,000 streamlines each streamline saw
  candidates only from its own slab. Training's context is the k nearest among all of a 10,000-streamline
  brain; `trained_context` draws one set of 10,000 candidates from the whole tractogram. At 10,000
  streamlines the two give the same context (9,999 of 10,000 rows; one tie order). On the test split
  nothing changes (accuracy 92.01 / 86.55 / RapidParc 94.48 / hemiaug 94.45, all as before, now through
  the Labeler's own path); on PAT16 (32,264 labeled) the trained context's Other share is 50.62 %
  (draw 0; was 50.59 %), draws 0 and 1 agree on 92.6 %; `upstream=True` still gives 58.09 %.
- **TRX**: `trx.write` refuses to replace a directory that is not a TRX (it deleted whatever directory was at the path).
- **The documents and bench**: `labelers.md` and the README rewritten to what was built; `pipeline.md`'s
  table from the committed cohort; `results/README.md` says which committed results were made under an
  older default (L-BFGS, TractCloud); twelve bench scripts honor `$TRACTOGRAPHY_DATA`; `modal_gpu_pipeline`'s
  CPU function mounts its volume; `run_pipeline.py` imports TractCloud only when asked.

The cohort's PAT16 rerun after the fixes is identical to the committed result (UKF throughput aside).

Corrections to earlier entries:
- "RapidParc ... identical to its package": on Modal's CPU and A10G, yes; on the M2's GPU one streamline of
  32,264 differs by cluster in one of four runs (hemiaug, seed 0), every tract identical.
- "The cohort with RapidParc": the tract centers' maxima went down for most patients, not all - PAT13's
  rose (8.3 → 9.5 mm, EmC); PAT08 15.8 → 10.0, PAT25 11.2 → 8.7.
- The "500" TractCloud runs on clinical tractograms before today (`labeler_compare.json`,
  `label_kglobal.json`, `label_trained_check.json`, `modal_gpu_pipeline_*_tc500.json`, the 500 rows of the
  2026-10-02 entries) used the chunked local context above, not exactly the training's; the global
  context - the larger effect - was as trained. On PAT16 the difference is 0.03 points of Other.
- `upstream=True` reproduces upstream's inference exactly only when every streamline is 40 mm or more:
  upstream's package labels every streamline; ours cuts first, so the context is drawn from the kept ones.

## 2026-10-03 A documentation review: the package's documents and the bench's records

Two more reviewers read the documents against the code and the records against their results. Fixed:
- **The package**: `pipeline.py` states the subject `run(s, ...)` takes (its fields, and that the DWI's
  own b0s come first), the `device="mps"` and `shell=2800.0` defaults (also in the README), that
  `stats["seed_voxel"]` is (k, j, i), the CUDA path; `trx.write` names a missing `logp` (was a torch
  dtype error); empty Labels carry logp only when asked; importing `labelers.tractcloud` no longer leaves
  a stub `vtk` in `sys.modules` (it hid the real one for the rest of the process) and says what it does
  to `sys.path` and the environment; TractCloud's minimum count, its checked commit (94de627) and the
  `.[triton]` extra's platform are documented; `resample.py`'s and `susceptibility.py`'s descriptions
  corrected (the cost is a mean, not a sum).
- **Licenses**: `mask.otsu` is DIPY's function verbatim (which DIPY took from scikit-image), so
  `THIRD_PARTY_NOTICES.md` now carries DIPY's BSD notice beside RapidParc's, and no longer says nothing
  is copied.
- **The bench**: `label_draws.py` and `label_noise_floor.py` pin TractCloud to upstream's context, as
  their committed results were made (with the trained context now the default, `label_draws.py` would
  have fed an 80-streamline global context to a network loaded for 500); eight scripts honor
  `$TRACTOGRAPHY_DATA`; usage lines and docstrings of `modal_ukf_triton.py` (the block kernel is the
  default), `gpu_pipeline_compare.py`, `modal_gpu_pipeline.py` (`--tf32`, `--save-fibers` and fetching
  its file), `modal_infer_opt.py` (two entry points; a record), `modal_rapidparc_check.py` (an A10G),
  `modal_cpu_scaling.py`, `run_pipeline.py`, `ukf_bench.py`, `ukf_step_bench.py`; the README's index
  and data layout (the Modal Volume's contents, FSL's and RapidParc's reference environments).
- **Provenance** (`results/README.md`): `pat16_seeding*.json` and `pat16_topup_compare*.json` carry
  TractCloud's labels; `cohort/PAT23_topup.json`, `cpu_timing.json` and `modal_cpu_scaling.json` were made
  with L-BFGS, `cohort/PAT25_topup.json` with Gauss-Newton; `susc_stability/PAT16_gn.json` holds an L-BFGS
  configuration beside the Gauss-Newton one.
- **Not for a public repository**: a Modal app id (`modal_cpu_scaling.json`) and a home-directory path
  (`ukf32.json`; `ukf32_compare.py` now writes it relative to `$TRACTOGRAPHY_DATA`) removed. Both remain in
  the history, for the audit before the repository goes public.

Corrections - committed files that later runs overwrote, so the numbers above are in NOTES only:
- `modal_cpu_field_timing.json` now holds the Gauss-Newton run ("Gauss-Newton on 32 x86 cores":
  48.9 / 40.2 / 44.6 / 51.5 s, 41.6 s after a pool, the thread count then 32). The L-BFGS run's 43.3 /
  36.5 / 41.3 / 72.5 s and 80.7 s at 48 threads after a pool ("The 101.8 s estimate, explained") are
  not in any committed file.
- `modal_cpu_pipeline.json` holds the run from the package layout (commit 6e36ef8: 93.9 s; estimate
  37.1 s, UKF 50.9 s), not the 127.6 s or 92.2 s runs quoted above.
- `accuracy_tractcloud_test.json`'s timings are the 2026-10-03 rerun's (TractCloud at 500: 2.93 s a
  subject and draw; 3.2 s in the first run); its accuracies are unchanged.
- The 49.9 % Other of `pipeline.md`'s PAT16 row is one draw (`gpu_pipeline_compare.json`'s M2 run); the
  cohort's 49.8 % is five draws averaged.

## 2026-10-03 Otsu's threshold, rewritten

`mask.otsu` was DIPY's function verbatim, which DIPY had itself copied from scikit-image; it is now
written here from Otsu's definition (the split maximizing n0 n1 (mu0 - mu1)^2), keeping DIPY's
conventions - each bin stands for its upper edge, the threshold is the lower edge of the lower class's
last bin - so the mask still matches DIPY's. Checked: the same threshold as `dipy.segment.threshold.otsu`
on 9,000 random cases (normal, integer, bimodal and gamma data; 256, 64 and 17 bins); `median_check.py`
again identical on all 14 volumes (the 12 patients, PAT16 corrected by topup and by ours), CPU and GPU -
its result as committed, timings aside. `THIRD_PARTY_NOTICES.md` credits DIPY as an implementation again;
RapidParc's scheme file is the one item redistributed.

## 2026-10-03 uv first

The checkout is a uv project: `uv sync` builds its environment (Python 3.12, `.python-version`, the
version every result here was made with; `uv.lock` committed), and `uv run bench/<script>.py` runs any
script. What the bench needs beyond the package is the `bench` dependency group (trx-python, dipy,
pynrrd, vtk, matplotlib, scikit-learn), installed by default and never in the package's metadata; the
`triton` extra is Linux-only. `rapidparc_check.py` runs RapidParc's package in an isolated uv
environment (`uv run --isolated --with RapidParc==1.0.4 --with torch==2.14.1`) instead of a hand-made
venv; `modal_ukf_triton.py` compares through `uv run --project`; the Modal images install with
`uv_pip_install`, torch pinned at 2.14.1 (what every committed Modal result used); the modal CLI is a uv
tool. Usage lines say `uv run`. Checked: the lock resolves the same versions as the environment the
results were made in (torch 2.14.1, numpy 2.5.3, scipy 1.18.1, nibabel 5.4.2, dipy 1.12.1, ...), the
environment synced entirely from uv's cache, and through `uv run` the PAT16 cohort run is identical to
the committed result (UKF throughput aside) and `rapidparc_check.py` reproduces its committed result.

## 2026-10-03 The bench's dependencies: dev by default, vtk and scikit-learn opt-in

The `bench` group became uv's conventional `dev` group (installed by `uv sync` / `uv run` without
configuration): pynrrd, dipy, trx-python, matplotlib. vtk (0.5 GB, three scripts that read VTK
tractography files: `ukf_compare.py`, `ukf_bench.py`, `resample_check.py`) and scikit-learn (one macro F1
in `accuracy_tractcloud_test.py`) are opt-in groups: `uv run --group vtk ...`, `uv run --group sklearn ...`.
The default environment went from 1.3 GB to 785 MB (torch is 533 MB of it). Checked: `cohort_summary.py`
runs; each group installs on request; the optional TractCloud labeler labels with vtk not installed.
- dipy is opt-in too (`--group dipy`): DIPY as the mask's reference (`median_check.py`) and the
  bootstrap's spherical-harmonic basis (`_bootstrap.py`, for `label_noise_floor.py` and
  `ukf_noise_floor.py`). `susc_check.py` masks with `tractline.mask.median_otsu` instead of DIPY's: on its
  input (topup's mean unwarped b0, float64) the two masks are identical, 94,505 voxels. `dev` is now
  pynrrd, trx-python and matplotlib.

## 2026-10-03 The package needs numpy and torch; a broken CPU path fixed

**Dependencies.** Nothing on the pipeline's path uses scipy or nibabel: the pipeline takes and returns
arrays (reading NIfTI is the caller's), and only `t1check.py` (a measurement) uses scipy, while its
nibabel import was unused. Required now: numpy, torch; scipy is the `t1check` extra (t1check names it
when missing); both stay in the bench's `dev` group (the loader, most measurements). `dependency_check.py`
blocks scipy and nibabel once the subject is loaded (the loader uses them): on PAT16 the GPU path
(49.4 s) and the CPU path (144.6 s) give the committed counts (42,170 / 32,264; 42,176 / 32,259), no
blocked module loaded. A wheel installed alone (numpy, torch) imports every pipeline module and labels.

**A bug the check found.** `pipeline.track` (and so `run`) failed on the CPU with UnboundLocalError since
the review's fixes (884fcc0, in v0.1.0): the main-guard check added at the top of `track` reads `os`,
and a function-local `import os` further down made `os` local to the whole function. The local imports
are gone (the module imports os). The cohort and the GPU checks run on "mps" and never reached it; the
review's CPU checks called `ukf.track` directly. ruff's F823 flags this pattern.

## 2026-10-03 Tests and continuous integration

`tests/` (pytest, ~40 s on the M2) and ruff (pyflakes rules) in the `dev` group; `.github/workflows/ci.yml`
runs both on Linux on every push, fetching RapidParc's weights. The pipeline is tested on a synthetic
phantom (a 32 x 44 x 20 block at 2 mm, a 4 x 4-voxel band of fibers along y, 30 directions at b = 2800,
a distortion-free reversed pair): on the M2's CPU 786 fibers, identical across runs and worker counts,
the field under 1 Hz, over 95 % of the fibers' points within a voxel of the band. Both ways the CPU path's UnboundLocalError (above)
would have been caught: ruff's F823, and the test once it runs the default worker count - with
`workers=2` the guard's `workers or os.cpu_count()` never read `os`, so the first version of the test
passed with the bug in place.

Found by the tests: RapidParc's normalization divides by each axis's extent over a group, so a lone
straight or planar streamline gave 0/0 and NaN probabilities (RapidParc's package does the same). An
axis with no extent now divides by 1; with every extent positive the arithmetic is RapidParc's, and
`rapidparc_check.py` reproduces its committed result. `trx.write` checks for `logp` before anything else.
`prep.prepare` no longer warns on the b0 rows' 0/0 (it was discarded). Unused imports and variables
removed from the bench and the package (`ruff --fix`), and upstream's unused TRACT_NAMES no longer
imported by the TractCloud labeler.

## 2026-10-03 What the correction needs from the acquisition: polarity and readout time

For a DICOM front end, whose phase-encoding polarity and readout time are vendor-private (Philips states
no polarity at all): `readout_polarity.py` on the 12 patients (`results/readout_polarity/`,
`readout_polarity_summary.py`), the field estimate's defaults on the M2's GPU, the DWI's displacement
compared in the brain (99th percentile of |difference|):
- **Absolute polarity does not matter**: every sign flipped gives the same displacements, 0.0 mm on all
  12 (topup's model: the field's sign flips with them; the smoothing is quadratic in it).
- **Readout time barely matters**: scaled by 0.8 or 1.25 (both series), the displacement moves at most
  0.20 mm on 11 patients, against 6.2-9.8 mm of correction - the field is fitted in Hz and the smoothing
  weight acts on it (k^2 = 0.64 to 1.56), but the fit hardly responds. PAT26 at 1.25: 0.92 mm, because that
  run ended at a lower cost (193.6 against the base's 199.8): a different and better solution the scaled
  problem's path reached, not a readout effect - the default fit on PAT26 stops short of it (open: worth a
  look at the estimator, e.g. restarts or the coarse levels).
- **Relative polarity is visible in the images**: labeled opposite, the true pair gives a field whose
  displacement's 99th percentile is 6.2-9.8 mm; a control - the AP b0s split in two halves, one polarity
  truly, labeled opposite - gives 0.7-1.7 mm. Pair over control: 4.7-12.8x, median 7.1x (smallest PAT25).
  The first level's cost ratio separates less well (PAT25 0.08 against 0.11, PAT29 0.11 against 0.12): the
  displacement is the statistic. The control is the easy case - halves of one series, seconds apart; two
  same-polarity series minutes apart would show more motion and drift, and are not in ds001226.
So a front end needs the phase-encoding axis (standard DICOM, 0018,1312), the relative polarity (readable
from the images, with the caveat above, or from the headers when they state it), and a readout time,
nominal if need be; the absolute polarity it does not.

## 2026-10-03 Polarity from images across separate series: not shown (preliminary, 2 subjects)

OpenNeuro ds005123 (CC0, Siemens Prisma; `fetch_ds005123.py` reads per subject the diffusion series' two
leading b0s by streaming, and the acq-dwi spin-echo field maps, 107 MB for 12 subjects), two subjects
through `polarity_separate_series.py` (displacement 99th percentile in the brain, mm, sub-10317 / sub-10369):
true pair (diffusion b0s + PA field map) 10.5 / 18.7; **control, separate series (diffusion b0s + the
same-polarity AP field map, ~7 min later, labeled opposite) 9.5 / 11.6**; control within one series
1.0 / 1.0; the field maps' own pair (AP + PA) 5.8 / 6.2, and the diffusion+PA pair's field differs from it
by 9.4 / 13.3. So the earlier result (`readout_polarity.py`: polarity readable from the images) holds
within a series but is not shown across series; here the diffusion b0s and the same-polarity field map
differ in more than distortion - their correlation in the brain is 0.64, below the opposite-polarity field
maps' 0.79-0.82, and no shift along j improves it. Cause open: the sequences' contrast (multiband HYDI
diffusion against a spin-echo field map, same TE/TR/readout by their sidecars), the streamed extraction,
or the acquisition. Until resolved, relative polarity comes from the headers, not the images.

Correction: the estimate already scales the volumes to a common mean (topup's --scale=1,
susceptibility.py); scaling the series again in the script changes nothing (`--scale on` / `off` agree).

## 2026-10-03 GE DICOM headers through dcm2niix's sidecar-only mode

`dcm2niix -b o` (v1.0.20260416, Homebrew; sidecars only, no images) on dcm2niix's own GE validation sets,
cloned into $TRACTOGRAPHY_DATA/dicom (dcm_qa_polar cda2bc7, dcm_qa_ge e92133a, dcm_qa_trt 6d7467a; ~270 MB
fetched, ~1 GB on disk): 12,357 DICOM files, 35 series, 7 s, no NIfTI written. Every series gets a signed
`PhaseEncodingDirection` (j / j-), a `TotalReadoutTime` (15-122 ms: acceleration and partial Fourier move it
eightfold - a nominal value is only safe when both series of a pair share the protocol) and a
`ShimSetting`; the 28 series with a reference sidecar in the sets match it exactly on all four fields.
GE's shim is three values, (0043,1002-1004), the linear X/Y/Z shim gradients - plain private values, read
straight; Siemens records eight (linear and second order), so a GE comparison sees only a linear re-shim.
dcm_qa_polar's epi_pepolar series alternate polarity volume by volume inside one series: dcm2niix splits
them, the reversed volumes as series number + 1000 (6 and 1006, 8 and 1008).

## 2026-10-03 A residual check before correcting: catches every bad pair, refuses some good ones

albula-diffusion's planning refuses a pair when its fit leaves more than half of the two scans'
difference. `pair_residual_check.py` applies that rule to our estimate (left = ||corrected mean A -
corrected mean B|| / ||mean A - mean B|| in the brain; motion not applied):
- **Bad pairs, all refused** (24 of 24): ds001226's AP b0s in halves labeled opposite, left 0.74-1.33;
  ds005123's diffusion b0s with the same-polarity field map labeled opposite, 0.70-0.94; and the
  **re-shimmed pair** (diffusion b0s with the reversed PA field map, ShimSetting differing on all 12
  subjects), 0.61-0.78, 12 of 12 refused - the case the image-based polarity test could not catch.
- **Good pairs, mostly accepted**: ds005123's two field maps (one shim), left 0.34-0.54, median 0.44, but
  sub-10581 1.24 (its field normal, 6.7 mm: the scans moved between); ds001226's AP/PA, 0.32-0.77, median
  0.46, refused on 4 of 12 - PAT25 0.77 and PAT29 0.65 the two with the most b0 motion (2.7, 2.1 mm),
  PAT19 0.61, PAT08 0.51. 6 of 24 good pairs refused at 0.5.
The bad pairs' lowest (0.61) and the good pairs' highest (1.24) overlap, and the overlap is motion: the
check counts what motion moved as difference left, and our estimate fits motion but the check does not
apply it. A motion-aware residual (the estimate's own rigid motion applied before comparing) is the
next step; the refusal threshold to be set from it. albula-diffusion does not model motion, so its check
would meet the same false refusals, and its field would carry the motion in those patients.
- **With motion applied** (every b0 unwarped by the estimate's own model - fitted rigid motion, field,
  Jacobian - the series' means compared): good pairs 0.16-0.26 (ds001226) and 0.26-0.45 (ds005123), all
  under 0.5; but the re-shimmed pairs drop to 0.38-0.47 and the mislabeled ones to 0.44-0.59, inside or
  beside the good range - the fitted motion explains part of a wrong pair's mismatch (bad fits needed more
  motion, median 1.1 mm against 0.6, overlapping). The AP-halves control drops to 0.17-0.53, but its field
  is 0.7-1.7 mm (harmless if applied); the dangerous bad pairs are the ones with large fields (7-22 mm).
  The good range also differs by dataset (noise, number of b0s), so no fixed threshold is principled.
  Conclusion: header checks decide (the sidecar's ShimSetting caught all 12 re-shims exactly; polarity and
  protocol likewise); the residual without motion is a warning that flags every dangerous pair here and
  the moving good ones, for review, not a refusal.

## 2026-10-03 The BIDS reader: the correcting series found, checked, or refused with the reason

`tractline.bids.load(dwi)` (the `bids` extra: nibabel, scipy) reads a BIDS diffusion series - DICOM through
dcm2niix's output - and the series that measure its field: B0FieldIdentifier/B0FieldSource, else IntendedFor
field maps, else a diffusion series in the same folder phase-encoded the other way (or given). The diffusion
series' own b0s come first, so the field is in its frame; a partner on another grid is resampled by the
scanner's coordinates (cubic), as the bench's loader does. Checked before use, from today's measurements: the
same phase-encoding axis (within 10 degrees in the world) and both polarities; ShimSetting identical when both
state it (else "not verified", said); echo time within 1 ms and voxels within 0.01 mm; a readout time stated,
or the diffusion series' assumed only when the protocols match (said). A failed check, or no partner: the
pipeline tracks the series as acquired (`Correction.applied` False, the reason in `note`); `pipeline.run`
skips the field stages and the timer counts them as 0. After a correction it reports `residual_left` (the
pair's difference left, as `pair_residual_check.py` measures it, now through this one function), a warning
above `RESIDUAL_WARN` 0.5, never a refusal: on the 24 good pairs 0.27-0.64 and 1.19 once (sub-10581), four above
0.5, all from motion (PAT19, PAT25, PAT29, sub-10581); on the 36 bad pairs 0.53-1.25, all above.
- ds001226: on all 12 patients `bids.load` hands the pipeline exactly the bench loader's arrays (dwi, bval,
  bvec, affine, b0s - including the four resampled - pe_vectors, readout); PAT16 through `run_pipeline.py
  --bids`: 42,170 streamlines, 32,264 labeled (the cohort's), residual 0.33, no warning; PAT03 refused (its
  second series is phase-encoded along i) and tracked as acquired, the reason naming that series.
- ds005123: all 12 refused - the IntendedFor field maps were acquired under another shim.
- `run_pipeline.py --bids PATH [--shell B]` (default shell: the one with the most volumes); 11 tests on
  synthetic BIDS sessions (`tests/test_bids.py`).

## 2026-10-03 Release review (0.2.0): four reviewers, and what changed

Four read-only reviewers attacked the BIDS reader, the pipeline's new paths, release readiness and the tests.
Fixed:
- **The residual** (`pipeline.residual_left`): it judged noise against noise - on the undistorted phantom
  0.98, with the motion/re-shim warning. It now returns None ("not judged", said in the note) when the
  two groups differ by less than twice what their within-group variation predicts; each volume is
  corrected with its own polarity and readout before the group means (each group's first readout was
  used for all); a failure or a non-finite value no longer fails the run (None, with a warning).
  Recalibrated (`pair_residual_check.json`): judged good pairs 0.27-0.64 (3 above 0.5: PAT19, PAT25, PAT29),
  judged bad ones 0.53-0.96, all above; not judged 7 - sub-10581 twice (its series moved 2.5-3.8 mm within
  themselves; the bad one of the two is still refused by the shim check) and 5 AP-halves controls (their
  fields ~1 mm, harmless).
- **`can_correct`** requires both polarities; an uncorrected Correction's note always says "not corrected";
  `Timer.total` is strict again (a misspelled stage raises) and a stage that did not run is recorded by
  `Timer.skip` as 0 s.
- **`bids.load`**: a partner covering less than 99 % of the diffusion series' brain is refused (a half slab
  gave a 484 Hz field on the undistorted phantom); a diffusion series without b0s is refused, a partner
  without them dropped; GRE field maps (phasediff, magnitude, fieldmap) named by B0FieldIdentifier are
  refused as unsupported (a phasediff map had been used as a b0); BIDS inheritance inside a BIDS dataset
  (sidecars, .bval/.bvec), only same-named files outside one; voxel sizes compared per world axis (sorted
  sizes hid a 2.5 mm against 2 mm voxel along phase encoding); a readout assumed only when the matrix along
  phase encoding and the acceleration also match; EstimatedTotalReadoutTime used and said as estimated; the
  diffusion series' own run preferred; a failing partner dropped (and said) when the rest still cover both
  polarities; same-folder partners found by world axis; unusable PhaseEncodingDirection values, missing
  .bval, a ".nii" in a folder name, IntendedFor without an extension - clean refusals or matches, not crashes;
  a missing nibabel/scipy names `tractline[bids]`. Still identical to the bench loader on all 12 ds001226
  patients; all 12 ds005123 subjects still refused.
- **Tests**: the review's mutation run found the suite caught 8 of 33 deliberate breaks - the phantom is
  undistorted, so sign, Jacobian, readout and gradient-flip breaks gave the same answers. Added
  `tests/test_correctness.py` (a pair distorted by a known 40 Hz field with two readout times: the field
  recovered, the residual, the displacement's sign; prepare's gradients in RAS for three storage
  orientations; the tracked shell; a TRX round trip; RapidParc's rows back to their fibers without weights;
  reversed and cross-protocol BIDS partners; exact_float32; the Timer) - 23 of the 25 missed breaks fail it -
  and 16 BIDS tests for the findings above. 70 tests, none skipped locally; CI now syncs the dipy group (the
  mask's DIPY check runs), lists skips, and fails when the weights fetch fails.
- **Release**: version 0.2.0, single-sourced from `__version__`; the README's install lines at v0.2.0; the
  sdist holds the package, tests and user documents only; THIRD_PARTY_NOTICES credits ds005123, dcm2niix and
  its validation sets, FSL, and says what the HCP-derived results are (four files of aggregate numbers).
- **The repository was public**, not private as planned, since its creation on 2026-10-02 (0 forks, stars or
  watchers); kept public by the owner's decision; the README's "private" line replaced by a research-use
  statement.

Corrections to earlier entries: "A residual check before correcting" (its 0.61, "6 of 24", "24 of 24")
and the BIDS-reader entry's ranges are superseded by the numbers above; "Polarity from images across
separate series" left the cause open - it is the re-shim (ShimSetting differs on all 12 subjects); the
"10 mm field" was the same-polarity field map labeled opposite (7-16 mm on 12 subjects), while the
re-shimmed reversed pair gave 9-22 mm against the field maps' own 5-7 mm - the documents now say so.

## 2026-10-05 Stage times on three scans, five platforms

`timing_matrix.py` (here) + `modal_timing_matrix.py` (an L4, an A10, an L40S, a 32-worker CPU container) →
`timing_matrix_{mps,l4,a10,l40s,cpu}.json`. The pipeline as it runs (`pipeline.run`, RapidParc one draw), two
runs a scan in one process; below the second (steady state), seconds. "read" is the scan into memory, outside
the total. Scans: Stanford HARDI (no pair, b = 2000, 150 directions); ds001226 PAT16 (corrected, b = 2800, 50
directions); ds005123 sub-10317 (Prisma, 120x120x69 at 2 mm, 145 volumes; the full series fetched into
`ds005123_full/`, 191 MB; its field maps refused - another shim - so tracked as acquired, b = 3250, 30
directions). And the correction alone on ds005123 with the shim check off (`ds005123_forced`): TIMING ONLY,
the field is not the series' (residual_left 0.664, above the 0.5 warning).

| HARDI | M2 | L4 | A10 | L40S | CPU 32 |
|---|---|---|---|---|---|
| read / prep / load | 0.4 / 4.0 / 1.3 | 1.2 / 0.8 / 2.1 | 1.0 / 0.6 / 1.7 | 1.1 / 0.4 / 1.5 | 1.1 / 1.2 / 0.9 |
| UKF / label | 110.9 / 1.1 | 67.6 / 1.6 | 71.8 / 1.2 | 36.0 / 1.0 | 103.6 / 2.2 |
| **total** (first run) | **117.3** (120.5) | **72.1** (81.1) | **75.3** (90.2) | **38.8** (52.5) | **107.9** (110.2) |

| PAT16 | M2 | L4 | A10 | L40S | CPU 32 |
|---|---|---|---|---|---|
| field estimate / apply | 26.1 / 1.6 | 8.0 / 1.0 | 5.0 / 0.8 | 3.4 / 0.8 | 37.2 / 0.6 |
| prep / load | 3.2 / 0.3 | 0.5 / 0.6 | 0.4 / 0.5 | 0.2 / 0.4 | 1.0 / 0.2 |
| UKF / label | 28.8 / 0.7 | 11.2 / 0.9 | 9.1 / 0.7 | 4.3 / 0.6 | 50.4 / 1.8 |
| **total** (first run) | **60.7** (54.9) | **22.2** (27.8) | **16.4** (30.6) | **9.6** (21.9) | **91.1** (91.2) |

| ds005123 | M2 | L4 | A10 | L40S | CPU 32 |
|---|---|---|---|---|---|
| prep / load | 5.5 / 0.3 | 0.8 / 0.6 | 0.6 / 0.5 | 0.3 / 0.5 | 1.7 / 0.3 |
| UKF / label | 39.5 / 1.1 | 18.7 / 1.6 | 13.7 / 1.3 | 7.3 / 1.0 | 70.4 / 2.3 |
| **total** (first run) | **46.5** (54.9) | **21.7** (402.4) | **16.1** (275.2) | **9.0** (266.3) | **74.7** (73.8) |
| forced correction: estimate / apply | 32.4 / 5.4 | 8.5 / 2.5 | 5.7 / 2.0 | 3.8 / 1.9 | 65.8 / 1.2 |

- **The Triton kernel compiles once per gradient count.** ds005123's first run took 4.4-6.7 min on each GPU,
  nearly all of it compiling for its 33 volumes; cached on the Volume after, per architecture. A deployment
  meets this once per new protocol: worth pre-warming or a kernel not specialized on the count.
- **The M2 numbers are provisional.** The same HARDI work as `hardi_paths_mps.json` (17,909,995 steps) took
  111 s against 94 s, and PAT16's estimate 18.7 then 26.1 s between runs. The machine (Mac14,15, fanless) was
  not idle - Photos' mediaanalysisd ~80 % and Spotlight ~40 % of a core, seen just after; to rerun idle.
- **Tracking time follows the gradient count**: on the L40S ds005123 (33 volumes) and HARDI (160) take about
  the same steps, 7.3 s against 36.0 s.
- **The CPU path's slow stage is the field estimate** (37 s PAT16, 66 s ds005123): more than any GPU's whole run.
- **The answers agree**: fiber counts within 0.03 % across platforms; the three CUDA GPUs give bit-identical
  tractograms on the uncorrected scans (HARDI, ds005123), slightly different on PAT16 (the float32 field differs
  per GPU). The CPU container showed 48 CPUs to the process; 32 worker processes; its CPU model unreported.

## 2026-10-06 The Triton kernel compiled 6-8 copies a scan: one now, and warm() for none

The 61 copies of `ukf_step_block` in the Volume's Triton cache differed, within one NPAD (the gradient count's
power of two: ds005123's 30 directions 32, PAT16's 50 64, HARDI's 150 256), only in Triton's default
specialization of integer arguments - a copy for each that equals 1 or is divisible by 16. B (the active fibers)
and step pass through all three cases, and the grid (nk, nj, ni) adds its own: 6-8 copies a scan, 33-50 s each.
`ukf_step_block` now takes `do_not_specialize` for its seven integers (one copy per NPAD and GPU architecture),
and `ukf_triton_block.warm()` compiles every NPAD (16-512) without running. `modal_timing_matrix.py --cache cold`
(a new, empty cache per GPU) → `timing_matrix_{l4,a10,l40s}_nospec.json`; `--prewarm` → `timing_matrix_l4_prewarm.json`.
UKF seconds, first run / second:

| | L4 before | L4 cold | L4 warmed | A10 before | A10 cold | L40S before | L40S cold |
|---|---|---|---|---|---|---|---|
| HARDI (NPAD 256) | 76.2 / 67.6 | 124.8 / 68.5 | 72.9 / 72.2 | 86.3 / 71.8 | 108.5 / 70.6 | 49.1 / 36.0 | 67.6 / 34.1 |
| PAT16 (64) | 15.8 / 11.2 | 57.9 / 11.3 | 11.0 / 11.1 | 22.5 / 9.1 | 41.7 / 9.2 | 16.0 / 4.3 | 31.9 / 4.6 |
| ds005123 (32) | **399.3** / 18.7 | **65.3** / 18.6 | **18.6** / 18.5 | **272.7** / 13.7 | **46.2** / 14.0 | **264.4** / 7.3 | **35.9** / 8.5 |

("before": yesterday's runs, the shared cache already holding HARDI's and PAT16's copies, not ds005123's.)
- **A cold compile is now one copy: 27-56 s** (first run minus second: L40S 27-34, A10 32-38, L4 47-56 s, growing
  with NPAD), against 4.4-6.7 min for ds005123's 8. warm()'s per-NPAD times on the L4 are 1.5-7 s above the
  in-run ones: another container (5 % slower on HARDI's steady state; compiling is CPU work), and the first
  compile in an empty cache also builds Triton's launcher.
- **warm() on the L4: 351 s for all six** (16: 48.6, 32: 48.2, 64: 53.4, 128: 58.2, 256: 63.2, 512: 79.5 s);
  after it no first run compiles (each within 1 s of its second). Once per GPU architecture and Triton
  version, into TRITON_CACHE_DIR. **In parallel** (`warm()` now spawns one process per NPAD, up to the CPUs:
  compiling is single-threaded CPU work), `timing_matrix_l4_prewarm_parallel.json`: **75 s wall** on the L4
  (8 CPUs), each NPAD 49.8-69.0 s as alone - the slowest, 512, sets it; then ds005123's first run 20.7 s
  against its second 18.6 s (the process's first launch; a compile would be 46 s or more). Into the full
  cache again (`_rewarm_{parallel,serial}.json`): nothing recompiled - one after another 3.0 s (2.3 s the
  first, then 0.0-0.4 s), in parallel 11.6 s (3.7-5.7 s a process to start torch and the GPU). warm() belongs
  once per machine or image, not at every start: a run loads its kernel from the cache by itself.
- **Steady state unchanged** (within 0.3 s, but the L40S's ds005123 8.5 against 7.3 s; the A10 and L4 show
  no such cost on it, so likely container variation).
- **The tractograms are bit-identical** to the specialized kernel's on every GPU and scan, but PAT16 on the
  L40S: 42,161 fibers against 42,169 - the field estimate (torch, not this kernel) differs between L40S
  containers of the same card type (46 GB, driver 580.95.05), as the 2026-10-02 entry saw (42,169 / 42,161);
  HARDI and ds005123, which have no estimate, are identical there too.

Correction to 2026-10-05: "compiles once per gradient count" - it compiled 6-8 copies per (NPAD, grid
divisibility), so a new grid recompiled even at a cached NPAD; now one per NPAD.

## 2026-10-06 The CPU path: a faster tracker step, a compiled one opt-in, the field estimate 1.8x

**Where the time went** (`cpu_profile.py`, `modal_cpu_profile.py` → `cpu_profile_{m2_before,m2_after,x86}.json`; PAT16, the
tracker in one process and one thread - a worker's setting - on every 24th seed). The M2: 23.7 k half-fiber steps/s, the
filter's algebra 43 %, H inside it 22 %, interp 21 %, ~9,500 operator calls a step; a step's fixed cost 2.2 ms, so steps with
fewer than 64 half-fibers live (half of a block's steps: each runs to its longest fiber) took 17 %. x86 (Modal, 48 vCPUs
visible): 14.5 k, interp 34 %, the fixed cost ~6 ms, those steps 30 %. The filter carried every measurement twice (Z 21 x
2N: the N gradients and their negatives, identical) through zh, Pxz, Ht, I and i, ~47 % of it. A vectorized 10x10 Cholesky
and inverse lost to torch's batched LAPACK at every batch size (2-14x): not the lever.

**The fast step (`ukf._ukf_step_fast`, `_interp_half`, `_H_half`; `track(fast=True)` - the CPU pipeline's - is now this).** The
N measurements once with the filter's sums doubled (as the Metal and Triton kernels), the weights applied first and each
contraction one batched product, interp's 27 voxels in one gather and one weighted product. `ukf_cpu_tune.py` →
`ukf_cpu_tune_{m2,x86,x86_compile}.json` (its "fast" there is the step before this one), one thread, same container each:
M2 24.1 → 31.6 k (1.31x), x86 21.6 → 30.8 k (1.43x). Against float64 (every 20th seed, `ukf_cpu_check.py`'s measures) as
before within noise: M2 all points within 0.1 mm 0.902 → 0.902, ends within 0.06 mm 0.934 → 0.937, density r 0.9939 → 0.9941
(x86 0.885 / 0.927 / 0.9936 → 0.888 / 0.929 / 0.9928). Bit for bit the prototype it came from. The whole pipeline, PAT16
tracking: M2 8 workers 124 s → 89-96 s (`cpu_compare_m2.json`); x86 32 workers, one container, 39.4-40.2 → 34.5-35.4 s,
ds005123 49.8-51.9 → 41.2-43.2 s (`cpu_compare.json`). Fewer gains with many workers than in one process (below).

**The compiled step: 2-3x in one process, slower with workers - opt-in (`TRACTLINE_COMPILE=1`), off by default.** The fast
step through torch.compile (Inductor's C++): one process M2 46-48 k (2.0x), x86 62.2 k (2.9x), as close to float64 (0.897 /
0.931 / 0.9941; x86 0.903 / 0.934 / 0.9941); the compile 11-31 s on the M2, 60-84 s on x86. With workers, three faults found
and fixed: Inductor bakes the compiling process's thread count into each kernel (`#pragma omp parallel num_threads(4)`:
8 workers ran 32 threads, load 35) - now `cpp.threads` 1; a module-level cache in `_interp_half` was a guard that failed on
first use, and sizes 0/1 and the step limit were specialized - every worker recompiled 2-3 times (2-4 s each warm, 10-20 s
cold) - now one graph (neighbors in D, the step limit outside, a lone half-fiber padded to two). What stays: each spawned
worker pays torch.compile's start - tracing, loading the cached code - ~8 s on the M2, ~100 s on 32 x86 cores at once.
`cpu_scaling.json` (x86, half of PAT16's seeds): uncompiled 109 / 168 / 184 k steps/s at 8 / 16 / 32 workers, compiled
52 / 50 / 39 k; the whole pipeline's PAT16 tracking compiled 104-167 s against the tuned step's 35 s (`cpu_compare.json`).
Rescuing it would take workers that do not trace: AOTInductor, or fork on Linux. Not done.

**The M2's CPU at 4+ workers is limited by the machine, not the step.** On a quarter of PAT16's seeds, total steps/s at 1 / 2 /
4 / 8 workers: uncompiled 35 / 55 / 89 / 110 k, compiled 50 / 62 / 91 / 86 k - equal at 4 workers; batches of 256, 512 and
1,024 all 75-95 k. Memory bandwidth or the fanless Air's power and heat; not separated. Its CPU numbers move 15-30 % with
both (PAT16 tracking with today's code 126 s once against 89-96 s three hours earlier).

**Refilling blocks as fibers stop** (instead of each block running to its longest fiber): simulated on PAT16's half-fiber step
counts with the fast step's costs (1.0 ms a step + 24 us a half-fiber), 16 workers on half the seeds: today 11.9 s, the best
refill (16 streams of interleaved seeds, up to 1,024 live) 9.7 s, the bound 8.6 s - ~1.2x, and results would depend on the
stream count. Deferred.

**The field estimate, 1.8x** (`susceptibility.py`). Profiled on the M2's CPU (float32, 8 threads, PAT16: 22.2 s): H 11.9 s,
6.7 s of it element-wise on (V, X, Y, Z) arrays; sample 6.1 s; levels 6-7 13.5 s. Changed:
- J'PJ through the sums over volumes: the field part is Bs'(W1 f + W12 g + sum Ac e) + dmats'(W12 f + W3 g + sum Bc e), the
  motion part M_v'(Ac_v f + Bc_v g + e_v - mean e) - W the diagonal's W1, W2 / 2, W3, Ac and Bc the volumes' A and Bj
  centered, e_v = M_v dm_v; (X, Y, Z) arrays and batched products instead of the (V, X, Y, Z) residual and (V-1, X, Y, Z, 6)
  temporaries. Exact (random inputs, float64: 2e-16 to 7e-16 of the old's);
- with motion held (levels 6-9), the moved grid in grid_sample's coordinates built once a level, and the derivative taken by
  the phase-encoding shift alone (the only component read there);
- each level's logged ssd before and after through sample() instead of unwarp() (10 % of the time; the field bit for bit).
PAT16: M2 CPU float32 22.6 → 12.3 s, float64 38.6 → 21.1 s; x86 32 cores, one container, 29.4-34.3 → 18.8 s
(`cpu_compare_pat16_today.json`); the M2's GPU path 17.4 → 13.5 s after the first change (its levels 1-5 run on the CPU).
**The field moves by the estimator's own rounding noise.** Displacement differences, mm, median / 99th / max, deep brain |
edge: the old code float32 at 4 against 8 threads (summation order only) 0.002 / 0.057 / 0.86 | 0.002 / 0.206 / 1.09; new
against old, float32 0.002 / 0.064 / 0.87 | 0.002 / 0.206 / 1.08, float64 0.003 / 0.078 / 1.03 | 0.003 / 0.126 / 0.59; and
against old float64, the new float32 0.002 / 0.055 / 0.30 | 0.002 / 0.107 / 0.57 where the old float32 was 0.002 / 0.066 /
0.88 | 0.002 / 0.217 / 1.24. The motion-held changes alone moved it less (0.017 / 0.042 at the 99th). **Not yet run: the
12-patient cohort (`cohort.py`) against the T1** - started and stopped (the laptop was needed); to run before relying on it.

**The CPU pipeline end to end, PAT16**: x86 32 cores, one container, 69.3-76.0 s → 55.0 s (estimate 18.8, tracking 33.7);
the M2's CPU 154 s → 151 s once (estimate 24.0 → 17.2 s, tracking 126 s that run - see above).

Correction to 2026-10-05: the M2 column, provisional there, rerun idle (`timing_matrix_mps.json`): HARDI 100.2 s (UKF 94.5),
PAT16 49.2 s (48.8 first run), ds005123 50.2 s (48.8), ds005123's forced correction 36.0 s - the committed numbers again;
the slower ones were the machine's load (Photos' mediaanalysisd, Spotlight). These predate the field-estimate change.


## 2026-10-06 The full timing table after the CPU changes; the cohort against the T1 again

`timing_matrix.py` (the M2: MPS, then the CPU path at 8 workers, right after the cohort - the laptop warm, load ~4) and
`modal_timing_matrix.py --prewarm` (an L4, an A10, an L40S, a 32-worker x86 container) → `timing_matrix_<platform>_1006.json`;
`timing_table.py --tag 1006 --before ""` made the tables ("before": 2026-10-05's runs, the M2's MPS rerun of this morning).
Second runs, seconds; "first run" its total.

**Stanford HARDI (no correction; 150 dirs, b 2000)**

| stage, s | M2 GPU (MPS) | M2 CPU, 8 workers | L4 | A10 | L40S | x86 CPU, 32 workers |
|---|---|---|---|---|---|---|
| field estimate | - | - | - | - | - | - |
| field apply | - | - | - | - | - | - |
| prep | 3.66 | 3.38 | 1.04 | 0.59 | 0.26 | 1.88 |
| load | 1.2 | 0.98 | 3.53 | 1.72 | 1.05 | 1.33 |
| ukf | 93.12 | 243.37 | 75.67 | 70.12 | 34.49 | 91.36 |
| label | 1.0 | 1.99 | 1.6 | 1.13 | 1.07 | 2.4 |
| total | **99.0** | **249.7** | **81.8** | **73.6** | **36.9** | **97.0** |
| first run | 98.7 | 248.5 | 82.5 | 75.3 | 40.0 | 114.6 |
| before (second run) | 100.2 | - | 72.1 | 75.3 | 38.8 | 107.9 |

**ds001226 PAT16 (corrected; 50 dirs, b 2800)**

| stage, s | M2 GPU (MPS) | M2 CPU, 8 workers | L4 | A10 | L40S | x86 CPU, 32 workers |
|---|---|---|---|---|---|---|
| field estimate | 9.34 | 12.4 | 7.47 | 4.21 | 10.42 | 24.93 |
| field apply | 0.84 | 0.71 | 1.03 | 0.8 | 1.5 | 0.58 |
| prep | 3.05 | 3.38 | 0.46 | 0.37 | 0.14 | 1.02 |
| load | 0.24 | 0.24 | 0.64 | 0.5 | 0.23 | 0.23 |
| ukf | 26.65 | 87.67 | 11.29 | 9.11 | 4.59 | 42.18 |
| label | 0.66 | 1.28 | 1.02 | 0.7 | 0.66 | 1.57 |
| total | **40.8** | **105.7** | **21.9** | **15.7** | **17.5** | **70.5** |
| first run | 40.6 | 109.1 | 23.7 | 16.4 | 18.2 | 79.7 |
| before (second run) | 49.2 | - | 22.2 | 16.4 | 9.6 | 91.1 |

**ds005123 sub-10317 (field maps refused; 30 dirs, b 3250)**

| stage, s | M2 GPU (MPS) | M2 CPU, 8 workers | L4 | A10 | L40S | x86 CPU, 32 workers |
|---|---|---|---|---|---|---|
| field estimate | - | - | - | - | - | - |
| field apply | - | - | - | - | - | - |
| prep | 5.44 | 5.84 | 0.82 | 0.64 | 0.23 | 1.8 |
| load | 0.24 | 0.24 | 0.72 | 0.47 | 0.33 | 0.36 |
| ukf | 39.01 | 127.23 | 18.57 | 13.82 | 8.39 | 57.72 |
| label | 1.04 | 2.09 | 1.64 | 1.14 | 1.35 | 2.28 |
| total | **45.7** | **135.4** | **21.8** | **16.1** | **10.3** | **62.2** |
| first run | 46.5 | 134.0 | 23.1 | 16.2 | 9.7 | 62.1 |
| before (second run) | 50.2 | - | 21.7 | 16.1 | 9.0 | 74.7 |

ds005123's correction forced past the shim check (timing only; estimate + apply): M2 GPU (MPS) 13.49 + 3.82 s; M2 CPU, 8 workers 19.23 + 4.25 s; L4 7.65 + 5.54 s; A10 4.29 + 1.96 s; L40S 5.83 + 2.1 s; x86 CPU, 32 workers 39.09 + 1.25 s.


- **The field estimate**: PAT16 M2 GPU 17.4 → 9.3 s, A10 5.0 → 4.2, L4 8.0 → 7.5, L40S 3.4 → 3.2 (a second container,
  `timing_matrix_l40s_1006_repeat.json`: PAT16 9.7 s in all; the first container's 10.4 s came with NVIDIA driver 610.57.04
  against 580.95.05 - the old code there 5.7 s, `estimate_compare_l40s.json` - a host, not the code), x86 32 cores 37.2 → 18.8-24.9
  (two containers), the M2's CPU 23.4 → 12.4. ds005123's forced estimate M2 GPU 31.8 → 13.5 s.
- **No first run compiles**: ds005123's first runs 266-402 s on 2026-10-05 → 9.7-23.1 s (one copy per gradient size, the shared
  Volume cache pre-warmed: `warm()` 44.9 s wall on the L40S, 49.0 the A10, 77.7 the L4 - sm_89 and sm_86 now hold all six sizes).
- **GPU tracking unchanged** (the kernels did not change): HARDI's L4 75.7 against 67.6 s is the container (the same kernel ran
  68.5-72.2 s in two others this morning).
- **The CPU path**: the x86 container PAT16 91.1 → 70.5 s, ds005123 74.7 → 62.2 s, HARDI 107.9 → 97.0 s (different containers;
  the same container's comparison: `cpu_compare*.json`); the M2's CPU PAT16 105.7 s (cool, alternating with the committed code,
  this morning: 105.5 / 120.3 against 148.8 / 152.3 s).
- **The field on the GPU, new against old** (`modal_estimate_compare.py` → `estimate_compare_{l40s,a10}.json`, PAT16's brain,
  mm, median / 99th / max): L40S deep 0.003 / 0.062 / 1.06, edge 0.003 / 0.12 / 0.59; A10 0.002 / 0.063 / 1.07, 0.003 / 0.16 / 1.08 -
  the CPU's rounding-noise scale (above); all voxels' 1.2-1.5 / 8-10 mm are outside the brain, where nothing holds the field.
  Each version is bit for bit across processes on a GPU.

**The 12-patient cohort against the T1** (`cohort.py` on the M2, the pipeline's default path; `cohort/*.json`,
`cohort_summary.md`): unchanged. Medians over 12, committed → now: the tumor margin's residual against the T1 at the 99th
1.58 → 1.59 mm, the brain's 1.62 → 1.60, half-split precision 0.39 → 0.38, validation r / slope 0.85 / 0.74 → 0.86 / 0.74,
tract mix r 0.9968 → 0.9967; per patient the 99th percentiles move by -0.05 to +0.09 mm either way. Scan to labels 54 → 42 s
(the estimate 18 → 9 s, UKF 29 → 28 s).
- **HARDI on the CPU, committed against today's code** (no correction: the tracker alone). x86 32 workers, one container
  (`modal_cpu_compare.py --only hardi` → `cpu_compare_hardi.json`): tracking 104.6 / 105.1 → 76.8 / 78.4 s (1.35x), total
  109.3 / 109.8 → 81.2 / 82.9 s. The M2's CPU, 8 workers, one run each (`cpu_compare_m2_hardi.json`): tracking 344.2 → 242.7 s
  (1.42x), total 350.7 → 249.0 s. Tracking is 93-97 % of HARDI's time on every platform, 81-94 % of ds005123's, 47-83 % of
  PAT16's (the field estimate 12-35 %).

## 2026-10-06 GPU tracking: the time outside the kernel was the seed fit; larger blocks; fibers bit for bit

`gpu_track_profile.py` (the M2, Metal) and `modal_gpu_track_profile.py` (an L4, an A10, an L40S; Triton) →
`gpu_track_profile_{mps,l4,a10,l40s}.json`: every kernel step timed with the device synchronized, against the half-fibers
it carried, on the three scans as acquired, at blocks of 50,000 (the pipeline's) and 200,000 half-fibers. The kernels
themselves were last tuned on 2026-10-01; this is the loop around them. The CUDA files hold the runs after the seed fit's
change below (with "seeds_s"; the runs before it are quoted here, their files overwritten); the M2's, the run before it.

- **A fiber's result does not depend on its batch**: every scan, every GPU, the fibers bit for bit the same at both block
  sizes (each fiber's arithmetic in its own lanes, reductions in a fixed order). Block size and refilling are speed only.
- **The M2 kernel at full load** (16 k+ half-fibers live): 1.52 / 1.95 / 4.33 us a half-fiber step for 30 / 50 / 150
  gradients (~0.9 us + 23 ns a gradient; HARDI's 231 k steps/s is 2026-10-01's one-step figure). The whole runs lose 18-27 %
  to it: a step with under 1,024 live costs 1.7-2.7 ms whatever it carries (1.7-8.6 s a scan), 1k-16k live 10-40 % more per
  half-fiber, and the loop around the kernel.
- **On CUDA the loop around the kernel was mostly the seed fit.** "loop" (wall less kernel steps) for HARDI: L40S 23.5 s
  against a 12.4 s kernel, L4 34.9 / 37.7, A10 21.9 / 50.3, the M2 8.6 / 87.5. ukf.seeds() fitted each seed's tensor by a
  batched SVD least squares (gelsd) on the host - one per mask voxel, the same 2N x 6 design matrix every time; on Modal's
  container CPUs that was ~20-30 s for HARDI's 156 k voxels. Now the design's pseudo-inverse once, one product
  (`ukf._seed_tensor`): the same seeds on all three scans, initial states within 2.3e-15 (float64), the GPU fibers bit for
  bit (PAT16 on the M2; HARDI and ds005123 on the CUDA GPUs against their earlier hashes). seeds() on the M2's CPU: PAT16
  0.94 → 0.36 s, HARDI 4.18 → 1.65, ds005123 1.82 → 0.55. HARDI's tracking, blocks of 50,000: L40S 35.9 → 16.8 s, L4 72.7
  → 42.4, A10 72.2 → 53.8 (the seed fit now 1.8-2.8 s there).
- **Blocks of 200,000** (`pipeline.GPU_BATCH`, the pipeline's GPU paths now): 2-17 % on top (L40S PAT16 4.33 → 3.52 s,
  HARDI 16.8 → 15.2; the M2 HARDI 96.1 → 88.6) for ~0.4 GB of device memory. PAT16 through the M2's pipeline:
  tracking 26.7 → 24.7 s, scan to labels 40.8 → 38.8 s, its fibers bit for bit this morning's.
- **What is left**: on the M2 18-27 % to tails, an under-filled middle and per-step loop costs - a refilled block (one
  stream kept full) would recover most and could not change a fiber; on the L40S the kernel's full-load estimate for PAT16
  is 2.2 s of 3.5. Inside the kernel, HARDI's per-gradient work (2026-10-01: no single hotspot).
- **Refilled, not blocked** (`ukf._track_stream`; `track(stream=True)`, the GPU kernels' default; capacity `batch`, the
  pipeline's `GPU_BATCH` 200,000): one stream topped up in seed order when a quarter of its places are free, on steps that
  are multiples of the recording interval - every age then has the same remainder, so the survivors record together and
  the step limit (the kernels get step 0) is checked only on the step a cohort reaches it: the per-step work is the
  block's. Fibers bit for bit the blocks' (every scan, the M2 and three CUDA GPUs). PAT16's tenth of the seeds on the M2:
  blocks of 2,000 16.9 s, a stream of 2,000 8.7 s; one block of 50,000 and a stream of 50,000 both 7.04 s (a first
  version that tracked each half-fiber's age and recording slot every step was 21 % slower there: ten more launches a
  step on MPS). Whole scans, the M2, tracking (`gpu_track_profile_mps.json`; blocks: `_mps_blocks.json`): PAT16 27.5 (blocks
  of 50,000) / 25.5 (200,000) → 24.3 s, HARDI 96.1 / 88.6 → 81.4, ds005123 40.7 / 35.6 → 32.1 - within 4-8 % of the kernel's
  full-load time; capacity 100,000 the same. CUDA (`gpu_track_profile_{l4,a10,l40s}.json`; blocks: `_blocks.json`, another
  container each): ds005123 L40S 5.98 → 4.72 s, A10 10.91 → 10.47, L4 13.38 → 12.96; PAT16 fits one block (95 k half-fibers)
  and HARDI (203 k) had only a small second block to lose - within the containers' 5-10 %.
- **The launchers' constants cached per tractography** (Q's diagonal and the voxel sizes were read back with .cpu() every
  step - each a wait for the device). Part of the above; not measured apart.
- **Seeding on the CUDA device, tried and left out**: the fit in float64 there (0.1-0.8 s instead of 0.4-2.4 s) moved
  ±2 fibers and 0.03 % of the steps on every scan and GPU (`_gpuseeds.json`) - the device's float64 rounding in the
  signal's interpolation - and the seeds would no longer be every platform's. The fit stays on the host.
- **Not quick**: the M2's mask (`prep`: 3.0 s of PAT16, 5.4 s of ds005123 on MPS; four exact 9^3 medians, DIPY's
  median_otsu voxel for voxel - the CPU no faster); the field estimate's coarse levels on CUDA (3.5 of the L40S's 4.8 s;
  launch-bound, the host CPUs there too slow to take them as the M2's does).


## 2026-10-06 The timing table: every step, every scan, every processor (today's code)

`timing_matrix.py` (the M2: MPS, then the CPU at 8 workers) and `modal_timing_matrix.py` (an L4, an A10, an L40S, a
32-worker x86 container; the Triton cache warm) → `timing_matrix_<platform>_1006b.json`; `timing_table.py --tag 1006b`.
The second run of each (the first within a few percent: nothing compiles). The M2 is the 15-inch MacBook Air (10-core GPU,
8 CPU cores, 16 GB); the x86 container showed 48 vCPUs, its model not; Modal's containers vary 5-10 % run to run (the
L40S's PAT16 estimate 3.2-4.9 s today). In the M2 GPU column the estimate's subsampled levels run on its CPU, by design.

**Stanford HARDI (no correction; 150 dirs, b 2000)**

| step, seconds | M2 GPU (MPS) | M2 CPU, 8 workers | L4 | A10 | L40S | x86 CPU, 32 workers |
|---|---:|---:|---:|---:|---:|---:|
| field estimate | n/a | n/a | n/a | n/a | n/a | n/a |
| field apply | n/a | n/a | n/a | n/a | n/a | n/a |
| prep (shell, gradients, mask) | 3.68 | 3.46 | 0.77 | 0.61 | 0.25 | 1.06 |
| load (tracker input) | 1.02 | 1.01 | 2.06 | 2.22 | 1.02 | 1.40 |
| tracking (UKF) | 81.05 | 243.92 | 40.40 | 51.93 | 15.31 | 75.99 |
| labeling (RapidParc) | 1.03 | 1.88 | 1.58 | 1.21 | 1.10 | 1.90 |
| total | **86.8** | **250.3** | **44.8** | **56.0** | **17.7** | **80.3** |

**ds001226 PAT16 (corrected; 50 dirs, b 2800)**

| step, seconds | M2 GPU (MPS) | M2 CPU, 8 workers | L4 | A10 | L40S | x86 CPU, 32 workers |
|---|---:|---:|---:|---:|---:|---:|
| field estimate | 9.36 | 12.25 | 6.91 | 4.32 | 4.93 | 20.79 |
| field apply | 0.88 | 0.72 | 1.01 | 0.81 | 0.59 | 0.45 |
| prep (shell, gradients, mask) | 3.06 | 3.38 | 0.44 | 0.36 | 0.14 | 1.78 |
| load (tracker input) | 0.25 | 0.24 | 0.55 | 0.52 | 0.22 | 0.23 |
| tracking (UKF) | 24.30 | 87.82 | 8.85 | 7.74 | 3.44 | 42.33 |
| labeling (RapidParc) | 0.67 | 1.30 | 0.94 | 0.76 | 0.64 | 1.32 |
| total | **38.5** | **105.7** | **18.7** | **14.5** | **10.0** | **66.9** |

**ds005123 sub-10317 (field maps refused; 30 dirs, b 3250)**

| step, seconds | M2 GPU (MPS) | M2 CPU, 8 workers | L4 | A10 | L40S | x86 CPU, 32 workers |
|---|---:|---:|---:|---:|---:|---:|
| field estimate | n/a | n/a | n/a | n/a | n/a | n/a |
| field apply | n/a | n/a | n/a | n/a | n/a | n/a |
| prep (shell, gradients, mask) | 5.48 | 5.86 | 0.78 | 0.64 | 0.24 | 1.63 |
| load (tracker input) | 0.26 | 0.24 | 0.61 | 0.51 | 0.37 | 0.44 |
| tracking (UKF) | 32.09 | 126.64 | 12.75 | 10.51 | 4.84 | 52.80 |
| labeling (RapidParc) | 1.05 | 2.07 | 1.58 | 1.26 | 1.34 | 1.84 |
| total | **38.9** | **134.8** | **15.7** | **12.9** | **6.8** | **56.7** |

**ds005123: the correction alone, forced past the shim check (timing only: this field is not the series')**

| step, seconds | M2 GPU (MPS) | M2 CPU, 8 workers | L4 | A10 | L40S | x86 CPU, 32 workers |
|---|---:|---:|---:|---:|---:|---:|
| field estimate | 13.33 | 19.55 | 7.21 | 4.35 | 4.80 | 31.22 |
| field apply | 4.76 | 4.18 | 2.50 | 1.99 | 1.47 | 0.94 |



- **Tracking is most of every uncorrected scan** (HARDI 76-97 %, ds005123 71-94 %) and 40-84 % of PAT16, whose field
  estimate is the rest's largest part (17-49 %); the M2's mask (prep) is 3-6 s on either of its paths.
- **Refill's own share** (`gpu_track_profile.py --blocks`, the same process: one refilled stream of 200,000 against blocks of
  200,000, fibers bit for bit; `gpu_track_profile_<gpu>_stream_vs_blocks.json`), the time saved: the M2 HARDI 2.7 %, PAT16 0,
  ds005123 3.4 %; L40S 7.6 / 1.8 / 4.2 %; A10 0.9 / 0.3 / 2.5 %; L4 -0.3 / -0.3 / 3.6 % (PAT16's 95 k half-fibers fit one
  block). Correction to the entry above: its M2 figures (HARDI 88.6 → 81.4 s, ds005123 35.6 → 32.1 s against blocks of
  200,000) also carry the seed fit's change; refill alone is the 0-3.4 % here.

## 2026-10-07 The pipeline's ensemble: uncertainty from every stage, and a rank field to hold it

`ensemble_pilot.py` → `ensemble_pilot_thermal.json`, `ensemble_pilot_residual.json`, `ensemble_pilot_added.json`;
`src/tractline/ensemble.py`;
`docs/uncertainty-ensemble.md`. PAT16 on the M2 (MPS), RapidParc. The reference is the pipeline as it runs today: the
scan as acquired, labeled with draw 0. Each of 16 replicates resamples every stage:

1. a wild bootstrap of the scan as acquired, both series' b0s included (`ShellWildBootstrap`: SH order 2/4/6 for the
   700/1200/2800 shells' 16/30/50 directions);
2. the field re-estimated from the replicate and applied;
3. the replicate tracked with Metal;
4. the result labeled with draw r.

The ensemble is the reference plus replicates 1-10 (R = 11); replicates 11-16 are held out as fresh runs. Streamlines
are matched by seed voxel. About 40 s per run.

Three b0 noise settings:
- **thermal:** b0 residuals scaled by 0.122, to the DW residual RMS of 8.46;
- **residual:** the b0s' own residuals, SD 69.5, which carry motion and pulsation;
- **added** (run after the two, see "Open" below): each b0 as acquired plus the thermal residuals, sign-flipped.

Eight draw-only runs label the reference tractogram with other draws.

**Per streamline (thermal; residual in brackets)**

| | |
|---|---|
| reference streamlines whose tract a held-out run changes | 34.0 % (36.5 %) |
| the same, labeler draw alone | 2.9 % |
| replicate against replicate | 38.1 % (39.0 %) |

The scan's noise, carried through the field and the tracker, moves about ten times more labels than the labeler's
draw does.

AUROC for predicting that change:

| predictor | AUROC |
|---|---|
| single run: TRX tract probability | 0.646 (0.646) |
| single run: TRX tract margin | 0.646 (0.646) |
| ensemble: mixture tract probability | **0.802** (0.814) |
| ensemble: vote fraction | 0.796 (0.808) |
| ensemble mixture decoded from its rank field (depth 6, keep clip) | 0.792 (0.803) |

Calibration against P(a held-out run keeps the reference's tract):

| predictor | Brier | ECE |
|---|---|---|
| single run | 0.295 (0.317) | 0.278 (0.303) |
| ensemble mixture | **0.169** (0.168) | **0.030** (0.025) |
| ensemble vote fraction | 0.172 (0.170) | 0.042 (0.036) |

- **One run's probabilities are badly overconfident against the pipeline's own variability.** Streamlines stated at
  ≥ 0.99 kept their tract in 73.5 % of held-out runs; the ensemble mixture's ≥ 0.99 kept it in 96 %.
- **Consensus.** The ensemble's consensus differs from the reference's label for 19.6 % (23.1 %) of streamlines. A
  held-out run agrees with the consensus 69.6 % of the time, against 66.0 % for the reference.
- **Paths are the larger part.** Matched paths are a median MDF of 7.8 mm apart (90th percentile 33.9 mm). Most of
  that is fibers ending at different points: where the two lengths are within 5 %, the median is 2.4 mm, as in
  `ukf_noise_floor`. A run's own MDF from the reference predicts its label change at 0.785.

**Votes in a rank field: the depth cut leaves them out**

At depth 6 with keep clip, the kept clusters hold every run's winner for only 74.8 % of streamlines; vote tails reach 6
of 11. Regrouped votes (SLF I-III, CC 1-7 merged) are exact for that same 74.8 % and short elsewhere. Winning clusters
are many: across 11 runs a streamline won in a median of about 5 distinct clusters, 1 in only 4.8 %, all within 6 in
77.9 %.

So vote planes aligned to the field's ranks are not enough for streamlines. Storing each run's winning cluster outright
(R × uint16, 22 B at R = 11, beside the field's 19 B) gives exact votes for any grouping.

**Per voxel (DWI grid 96 × 96 × 60, tau = 1 streamline per run)**

- 92,166 voxels have a tract.
- A held-out run's winner (a tract, or NONE) agrees with the ensemble's density field winner in 67.1 % of them, against
  60.4 % for the reference run's own map.
- The consensus's vote fraction is calibrated: Brier 0.170, ECE 0.039, and stated 1.0 is observed 0.966.
- The field encoded with rankfield (44 classes, depth 6, keep shell) is 13 B/voxel with exact winners. Its
  rank-aligned vote planes hold every run's winner in 91.6 % of tract voxels.
- Other wins most voxels (it is half the streamlines).

**Was open: the replicates' fields were not the reference's. Fixed with the b0s kept as acquired**

- In thermal mode, every replicate's field had a 99th-percentile |field| of 183-186 Hz against the reference's 154 Hz,
  and residual_left of 0.30 against 0.33.
- In residual mode the field sat at 143-148 Hz and residual_left at 0.36.
- The cause: the b0s were not exchangeable. A replicate's b0 was the six b0s' mean plus noise, so the motion between
  them was gone (thermal) or scrambled voxel by voxel (residual), and the field estimate responds to that.
- **added** (`--b0-noise added`, records in `ensemble/PAT16/added/`, no draw-only runs) keeps each b0 as acquired, the
  reversed pair's too, and adds the thermal residuals (scale 0.122, RMS 8.46) with flipped signs. A replicate's b0 thus
  carries twice the noise variance of the scan's.
- The 16 replicates' fields now bracket the reference's: p99 |field| 153.4-155.3 Hz (mean 154.3, 4 of 16 below
  154.0), residual_left 0.331-0.333 against 0.331. The doubled b0 noise shows only as that +0.3 Hz and +0.001.
  The voxel field's positions now rest on fields drawn about the scan's own.

Every metric moves within run-to-run spread (thermal / residual / added):

| | thermal | residual | added |
|---|---|---|---|
| held-out change of the reference's tract | 34.0 % | 36.5 % | 33.2 % |
| replicate against replicate (all 120 pairs) | 37.9 % | 38.7 % | 37.8 % |
| AUROC: TRX probability / ensemble mixture / vote fraction | 0.646 / 0.802 / 0.796 | 0.646 / 0.814 / 0.808 | 0.648 / 0.796 / 0.790 |
| AUROC: mixture decoded from the rank field | 0.792 | 0.803 | 0.785 |
| ECE: single run / mixture / vote fraction | 0.278 / 0.030 / 0.042 | 0.303 / 0.025 / 0.036 | 0.270 / 0.032 / 0.043 |
| single run stated ≥ 0.99, kept | 73.5 % | 71.0 % | 74.4 % |
| consensus differs from the reference | 19.6 % | 23.1 % | 18.3 % |
| MDF median / 90th (mm) | 7.8 / 33.9 | 9.3 / 35.1 | 7.4 / 33.4 |
| regrouped votes exact at depth 6 | 74.8 % | 73.5 % | 75.2 % |
| voxels: consensus / reference map agree with a held-out winner | 67.1 / 60.4 % | 66.8 / 59.7 % | 67.1 / 60.8 % |
| voxels: ECE of the consensus vote fraction | 0.039 | 0.039 | 0.039 |

Labels barely registered the field mismatch. Thermal's 30 Hz excess moved no metric beyond the spread between modes.
Residual mode, with the noisiest b0s, had the most label change and consensus disagreement; added has the least.

- **The reference is closer to a replicate (33 %) than two replicates are to each other (38 %)** in all three modes, so
  the field is not the cause. Explained the next day (2026-10-08): the bootstrap is centered on the scan's own model
  estimate.

Not measured: accuracy (no reference tract labels for PAT16). One patient. The bootstrap's sign flips are symmetric
around a smooth fit and do not reproduce magnitude noise at low signal.

## 2026-10-07 Are votes needed beside the mixture? No, not for a labeler with probabilities

`ensemble_votes_test.py` → `ensemble_votes_test_added.json`, `ensemble_votes_test_thermal.json`. These use the
ensemble pilot's records with no new runs: the ensemble is r000-r010 (R = 11), r011-r016 are held out, and there are
about 177,000 (streamline, held-out run) pairs. Intervals are 95 % from a block bootstrap: 200 resamples over 148
blocks of 10-voxel cubes of seed voxels. The pass mark was fixed before running: votes are needed when votes minus
field6 exceeds 0.01 AUROC with its interval above 0. Numbers are for **added**; thermal agrees to within 0.007 everywhere.

**1. Regrouping after labeling**

AUROC for a held-out run changing the reference's group. The predictors:
- single: one run's group probability;
- field6: the mixture as the web export encodes it (depth 6, keep clip), group mass decoded;
- margin6: the same field's group margin;
- full: the mixture with nothing cut;
- votes: exact vote fraction;
- votes6: vote fraction from rank-aligned planes.

| grouping | change | single | field6 | margin6 | full | votes | votes6 | votes - field6 |
|---|---|---|---|---|---|---|---|---|
| tracts | 33.2 % | 0.648 | 0.785 | 0.758 | **0.796** | 0.790 | 0.782 | +0.005 (0.003-0.006) |
| SLF, CC merged | 32.7 % | 0.644 | 0.784 | 0.757 | **0.794** | 0.788 | 0.780 | +0.005 (0.003-0.006) |
| tracts, outliers folded | 37.7 % | 0.630 | 0.792 | 0.770 | **0.796** | 0.791 | 0.788 | -0.001 (-0.002-0.000) |
| clusters, outliers folded | 61.2 % | 0.606 | 0.783 | 0.761 | **0.783** | 0.777 | 0.777 | -0.006 (-0.006 - -0.005) |

- **The mixture beats votes in every grouping**: full minus votes is -0.005 to -0.006, with intervals excluding 0.
- **Votes beat the depth-6 field only where the depth cut loses mass** (tracts: field6 is 0.011 below full), and by
  less than the pass mark.
- **Calibration (ECE):** full 0.032-0.037, field6 0.034-0.040, votes 0.043-0.060, single 0.27-0.50.
- **rankfield's group margin (decode_groups) predicts worse than the group mass decoded from the same bytes**, by 0.02
  in every grouping. For confidence, read mass, not margin.

**Deeper ranks close the gap that votes only narrow.** The mixture encoded deeper, at tract level (outliers folded):

| | AUROC | bytes per streamline |
|---|---|---|
| depth 6 clip | 0.785 (0.792) | 19 |
| depth 8 clip | 0.792 (0.795) | 25 |
| depth 12 clip | 0.796 (0.796) | 37 |
| full | 0.796 | |
| depth 6 shell | 0.774 | 19 |
| depth 12 shell | 0.795 | 37 |

Votes at 22 B (per-run winners, R = 11) reach 0.790, below depth 8 at 25 B.

**2. Beyond the mixture.** Within deciles of the full mixture, votes barely separate change: conditional AUROC is
0.516-0.519 for vote fraction and 0.509-0.535 for vote entropy, where 0.5 is nothing. A logistic model (5-fold by
block) given logit(full) plus vote fraction and entropy lowers log-loss from 0.516 to 0.507 at tract level, 0.540 to
0.527 outliers folded, and 0.544 to 0.543 at clusters, against 0.636-0.668 with the intercept alone. This is a small,
real gain: two confidently disagreeing runs and two unsure ones do differ, slightly.

**3. Runs.** AUROC of full / field6 / votes, tract level:

| R | full | field6 | votes |
|---|---|---|---|
| 2 | 0.705 | 0.702 | 0.639 |
| 4 | 0.753 | 0.750 | 0.727 |
| 6 | 0.774 | 0.770 | 0.759 |
| 8 | 0.786 | 0.779 | 0.776 |
| 11 | 0.796 | 0.785 | 0.790 |

ECE of full falls from 0.150 at R = 2 to 0.032 at R = 11. Votes trail most at small R, where their 1/R steps are
coarse. Nothing has levelled off by 11; more runs still pay.

**4. Voxels: summed votes are not merged votes, and it hardly matters**

| merge | voxels | votes differ | Brier, ECE: summed | Brier, ECE: merged first |
|---|---|---|---|---|
| CC 1-7 | 24,240 | 3,228 (13 %) | 0.138, 0.054 | 0.136, 0.058 |
| SLF I-III | 12,474 | 1,370 (11 %) | 0.135, 0.053 | 0.134, 0.056 |
| superficial (Sup-*) | 38,565 | 9,031 (23 %) | 0.176, 0.055 | 0.168, 0.047 |

The second column counts voxels where the merged group's summed tract votes differ from votes recomputed after
merging the densities. Against held-out runs' merged-first winners, calibration differs little, by most for the
superficial merge, the largest one. The rank-aligned vote planes regroup voxels only approximately, as expected:
a voxel's winner is a density argmax, not the group of one class's win.

**Reading.** For a labeler that gives probabilities, the mixture is the uncertainty to store. It fits rankfield's
format as it is; it needs depth 8 or more rather than 6, or a keep rule that spends depth on mass. Votes add a sliver
of information (log-loss, 2 %) and exact regrouping for streamline labels, but at equal bytes they predict worse than
a deeper mixture. Votes remain the only representation where runs give hard labels without probabilities: raters,
atlases, other software. Their place in rankfield is that case, not this one.

Caveats:
- The ensemble's calibration is close to guaranteed here: its held-out runs come from the same bootstrap. The single
  run's miscalibration is the finding.
- One patient, RapidParc, stability not accuracy.

## 2026-10-08 Why the scan is closer to a replicate than replicates are to each other: the bootstrap is centered on it

`ensemble_exchange.py` → `ensemble_exchange.json`; `_bootstrap.py` gains `leverage`, `flip` and b0 mode "flipped".
PAT16, M2 (MPS), 8 replicates per variant, the reference `ensemble/PAT16/added/r000`.

In the 2026-10-07 ensemble, a replicate changes 33 % of the reference's labels and another replicate's 38 %. Two
candidates, both built into the bootstrap:

- **Noise level.** In squared signal distance, two replicates are 1.31× farther apart than either is from the scan in
  the DW volumes, and 2× in the b0s. In the DW volumes this comes from the leverage correction (h = 0.38 / 0.50 / 0.56
  for SH order 2 / 4 / 6 on 16 / 30 / 50 directions: σ²(2 − h) against 2σ²). In the b0s it is the "added" noise.
- **Whitening.** Per-voxel sign flips break up residuals that are spatially structured in the scan.

Variants (labels changed and median MDF: reference vs replicate / replicate vs replicate):

| variant | DW residuals | b0s | signs | labels | MDF mm | field p99 Hz | residual_left |
|---|---|---|---|---|---|---|---|
| added (r001-r008) | leverage-scaled | as acquired + noise | per voxel | 33.2 / 38.0 % | 7.4 / 10.3 | 153.6-155.3 | 0.331-0.333 |
| lev0 | raw | as acquired + noise | per voxel | 27.7 / 32.1 % | 4.7 / 6.9 | 153.9-154.7 | 0.331-0.332 |
| member | raw | residual flipped in place | per voxel | 27.9 / 32.3 % | 4.7 / 6.9 | 155.4-157.3 | 0.326-0.328 |
| member-block | raw | residual flipped in place | per volume | 30.0 / 34.4 % | 5.8 / 8.2 | 154.4-161.2 | 0.310-0.338 |

The reference's field: 154.0 Hz, residual_left 0.331.

In `member` the scan is itself one of the replicates' sign patterns (all +1): in signal it is exactly as far from a
replicate as two replicates are from each other (squared distance 74.9 / 75.0 DW, 58.4 / 58.7 b0). The gap stays
(ratio 0.86, against 0.87 in `added`), so the noise level is not the cause. Keeping the spatial structure
(`member-block`) keeps it too (0.87), so whitening is not the cause either. The gap shows in paths as much as in
labels, so it is the tracker, not the labeler.

**The cause: the scan is the center of the replicate cloud.** A least-squares residual is orthogonal to the fit's
space in every voxel. A pipeline that reads the signal through a similar model (UKF's per-voxel tensor fit) sees
fit + r about as it sees the fit alone. Sign-flipped residuals, per voxel or per volume, are no longer orthogonal,
and that moves the model estimate. Two more runs from `lev0`'s fit, b0s as acquired:

| pair | labels changed | median MDF mm |
|---|---|---|
| reference vs fit (no residual) | 10.1 % | 0.54 |
| reference vs anti (fit − r: every sign flipped, 4× a replicate's squared distance from the scan) | 10.8 % | 0.90 |
| fit vs anti | 11.1 % | 0.91 |
| reference / fit / anti vs a replicate | 27.7 / 27.8 / 27.6 % | 4.7 |

The farthest sign pattern in signal tracks like the scan. The scan, its fit and its reversed residual sit at one
point, the same distance from every replicate. The ~10 % among them is mostly the labeler's draw (0 / 101 / 102)
plus 0.5-0.9 mm of path change.

**What this means for the ensemble.** As with any bootstrap, the replicates spread around the scan's own estimate and
cannot see that estimate's own error. A held-out replicate's change of the reference's label (33 %) is therefore
not what a fresh scan of the patient would do. A fresh scan carries its own noise, like a second replicate, so
replicate against replicate (38 %) is the better estimate. Calibration against held-out replicates, measured from the
reference, is somewhat optimistic for real rescans. Scoring held-out replicates against a replicate's labels instead
of the reference's would measure it without that centering.

Side results:

- **Keep the leverage correction.** Raw residuals give two replicates 2σ²(1 − h) apart, under the 2σ² between two
  scans, and change about 5 points fewer labels. `added` stays the default.
- **The b0 noise does not move labels.** `lev0` and `member` match.
- **Don't flip the b0s in place.** "flipped" pushes every field above the reference's (155.4-157.3 Hz). Keep "added".
- **Per-volume signs move more** than per-voxel signs: 30 % against 28 % labels, and the field spreads to 161 Hz.
  Spatially coherent perturbations matter more to the field and the tracker. The scan's real noise is likely
  spatially correlated (parallel imaging, interpolation), so per-voxel flips may understate it. Not resolved here.

## 2026-10-08 The ensemble's confidence against rescans: a double bootstrap, and a recalibration

`ensemble_double.py` → `ensemble_double.json`. PAT16, M2 (MPS), 52 runs. Follows the entry above: the pilot scored
its confidence against held-out replicates, which spread around the scan's own estimate, not against rescans.

**Set-up.** The double bootstrap gives the ensemble a scan whose rescans we have.

- **Pseudo-scans:** four first-level replicates (`added/r011`-`r014`). Each was rebuilt from its seed and rerun; all
  four reproduce the stored record exactly.
- **Second level:** each pseudo-scan is bootstrapped again ("added", the reference's mask), 12 runs. Runs 1-10 plus
  the pseudo-scan make its ensemble (R = 11, as the pilot). Runs 11-12 are held out for the pilot's centered scoring.
- **Rescans:** the other 15 first-level replicates. They are draws about the same center (the real scan's estimate)
  as the pseudo-scan, as real rescans are draws about the truth.

**The double bootstrap reproduces the pilot, then shows what it missed** (scored against centered held-out runs /
against rescans):

| | centered | rescans |
|---|---|---|
| labels changed | 33.4 % | 38.1 % (replicate vs replicate: 38.0 %) |
| AUROC: TRX probability / mixture / vote fraction | 0.648 / 0.799 / 0.792 | 0.641 / 0.780 / 0.773 |
| ECE: TRX probability / mixture / vote fraction | 0.271 / 0.029 / 0.041 | 0.318 / 0.067 / 0.080 |
| stated ≥ 0.99, kept: TRX / mixture | 74.1 % / 96.2 % | 69.3 % / 93.0 % |
| voxels: consensus / own map agree with the other run | 67.5 / 61.1 % | 65.8 / 58.8 % |
| voxels: ECE of the consensus vote fraction | 0.038 | 0.054 |

The centered column matches the pilot's numbers (33.2 %, 0.80, ECE 0.03 / 0.27), so the set-up is sound. Against
rescans the change rate is the replicate-vs-replicate rate, as the centering argument predicts. The ensemble is still
far better than one run: AUROC 0.78 against 0.64, and ECE 0.07 against 0.32. But its ECE more than doubles, and it is
overconfident at the top. Per pseudo-scan, the change rates (37.7-38.3 %) and ECEs (0.065-0.069) barely vary.

**Recalibration for rescans.** Fitted on three pseudo-scans, scored on the fourth (Brier, ECE averaged over the four
held-out pseudo-scans):

| | Brier | ECE |
|---|---|---|
| mixture as stated | 0.188 | 0.067 |
| mixture, Platt on log-odds (a = −0.12, b = 0.50) | 0.186 | 0.049 |
| **mixture, isotonic on log-odds** | **0.182** | **0.0025** |
| vote fraction as stated | 0.193 | 0.080 |
| **votes, a table per count** | 0.189 | **0.0026** |
| voxel consensus vote fraction as stated | 0.179 | 0.054 |
| **voxel votes, a table per count** | **0.176** | **0.0024** |

- One temperature is not enough. The mixture is underconfident in the middle (stated 0.74, kept 0.81) and
  overconfident at the top (stated 0.997, kept 0.948). An isotonic map fixes both.
- A first attempt pooled the held-out predictions and got an ECE of exactly 0. With equal-sized groups, a table
  fitted on the other groups averages back to the pooled rate, which hides the error. Scores are now per held-out
  group.
- **Votes table (R = 11):** P(rescan keeps the tract | the pseudo-scan's tract won v of 11) is 0.13, 0.23, 0.31, 0.39,
  0.46, 0.52, 0.59, 0.66, 0.73, 0.81 and 0.92 for v = 1-11. Unanimity leaves a 1-in-12 chance that a rescan changes
  the label.
- **Voxel table:** P(rescan's winner = consensus | v of 11) runs from 0.11 at v = 0 to 0.95 at v = 11.

**Applied to the real scan's ensemble** (`added/r000` + r001-r010; the maps were fitted on pseudo-scans only):

- Mean stated P(a rescan keeps the tract): 0.688 as is, 0.621 isotonic, 0.606 by votes. Replicate against replicate
  keeps 0.620, so the recalibrated mean matches a rate it was not fitted to.
- Streamlines stated ≥ 0.9: 28 % as is, 14 % recalibrated. Stated ≥ 0.99: 15 % as is, none recalibrated. The isotonic
  map tops out at 0.945.

Caveats:

- One patient, one protocol. The maps belong to this pipeline at R = 11, and other protocols or sites would need their
  own fit. A double bootstrap per scan costs about 4 × 12 extra runs.
- The pseudo-scans' b0s carry the first level's added noise as well as their own. That should not matter: b0 noise did
  not move labels (entry above).
- Everything rests on the bootstrap's noise being the scan's. Per-voxel sign flips may understate spatially
  correlated noise (entry above). Only test-retest scans can check that, and the repository has none.


## 2026-10-08 The bootstrap against real rescans: it understates them, and correlated signs close part of the gap

`ensemble_retest.py` → `ensemble_retest_<pair>.json`. New: `fetch_ds003416.py`, `_masivar.py`, `_register.py`;
`_bootstrap.py` gains `flip="smooth"` and handles a single reversed b0. M2 (MPS), 124 pipeline runs (40, 42 and 42 per
pair).

**Data.** MASiVar (OpenNeuro ds003416, CC0, Cai et al. MRM 2021): healthy adults scanned repeatedly on a Philips Achieva
3T. 261 MB was fetched. Two subjects, three reference scans, five rescans:

| pair | protocol | reference A | rescans |
|---|---|---|---|
| cIIIs01-2.5 | 2.5 mm, b1000 ×40 + b2000 ×56, 16 forward b0s | s1Bx1 scan 2 | same session (scan 4) |
| cIIIs01-2.1 | 2.1 × 2.1 × 2.2 mm, same shells and b0s | s1Bx3 scan 1 | same session (scan 2); another session (s1Bx1 scan 1) |
| cIs1 | 2.5 mm, 38 slices, b1000/2000/3000 ×96 (3 of 5 shells), 3 forward b0s | s1Ax1 scan 1 | same session (scan 2); next day (s1Ax2) |

Each scan is one series per shell, phase-encoded "j", plus a short "j-" series: three b1000s and one b0. Every scan,
on both subjects, has that one reversed b0. The pipeline gets the shells joined, and the field is estimated from the
forward b0s plus the reversed one. Three quirks, handled in `_masivar.py`:

- The sidecars state no readout time, so a nominal 0.05 s is used throughout.
- Each series' trailing b = 0 volume is a real b0 (0.99–1.0 of the others), not a Philips trace image.
- Philips stores a different scale per series. All series are put on one series' units.

The cIIIs01 scans track b = 2000 and cIs1 tracks b = 3000. The labeler draws a random subset of streamlines as context;
every run here uses the same draw (0), so only the data's share of label change is counted.

Terms used below:
- **replicate:** a wild-bootstrap copy of the scan, its residual noise re-signed (`_bootstrap.ShellWildBootstrap`);
- **signs:** how the noise is re-signed. Per voxel: each voxel and volume its own sign. Per volume: one sign per volume.
  Smooth: patches of one sign a few voxels across within each slice;
- **seed keys:** streamlines matched across runs by the voxel they were seeded from (about 90 % match);
- **MDF:** mean direct-flip distance between two paths, mm;
- **voxel winner:** the class with the most streamlines in a voxel (Other included);
- **field p99:** the 99th percentile of the estimated off-resonance field's magnitude, Hz.

**Matching across scans.** A rescan's seeds are not A's voxels.

- Nearest-path matching (MDF) cannot tell rescans from replicates. On cIIIs01-2.5, among matches within 5 mm, a run
  against itself (each streamline to its nearest other one) changes 9.8 % of labels. Replicate against replicate
  changes 9.7 %, and the rescan 10.8 %. The nearest path is mostly a neighbor, so the measure sits at its floor
  (`ensemble_retest.py analyze --nearest`).
- What is used: the rescan's corrected mean b0 is registered rigidly to A's (dipy, mutual information). Its corrected
  DWI is then resampled onto A's grid (cubic, b-vectors turned by the rotation) and tracked there, so seed keys match.
  Head motion between scans: 0.2–1.0 mm and 0.2–0.5° within a session (NCC 0.96–0.97); 2.7 mm / 3.2° and 5.6 mm /
  10.5° between sessions (NCC 0.92, 0.85).
- Resampling smooths the noise. As a control, A is sent onto the rescan's grid and back, and so are four replicates.
  - A against itself resampled changes 12–18 % of labels.
  - With per-voxel and smooth signs, replicate pairs that are both resampled change 0.4–1.1 points fewer labels than
    native pairs (per-volume signs: −0.1 to +0.7).
  - The rescan scored against resampled A matches the rescan against A.
  - So interpolation does not inflate the rescans.
- Voxel winners need no matching. Their rescan figures below are interpolation-free: the rescan is tracked on its own
  grid and its streamlines are moved into A's space.

**Labels changed (seed keys), replicate against replicate (8 replicates, 7 pairs, "added" b0s), against real rescans:**

| pair | per-voxel signs | smooth (in-plane, FWHM 4 voxels) | per-volume signs | rescan, same session | rescan, other session |
|---|---|---|---|---|---|
| cIIIs01-2.5 | 31.5 % (31.3–31.7) | 35.0 % (34.5–35.4) | 36.5 % (33.3–38.4) | 36.3 % | — |
| cIIIs01-2.1 | 32.2 % (32.0–32.4) | 34.9 % (34.7–35.1) | 35.8 % (33.5–37.7) | 36.9 % | 39.1 % |
| cIs1 | 39.6 % (39.2–40.2) | 40.7 % (40.2–41.0) | 41.6 % (40.0–42.2) | 49.9 % | 47.5 % |

Median path distance between seed-matched streamlines (MDF, mm), in the same column order:

- cIIIs01-2.5: 9.7 / 12.7 / 13.7 against 13.1.
- cIIIs01-2.1: 10.5 / 12.9 / 13.9 against 14.0 and 15.9.
- cIs1: 13.2 / 14.3 / 15.0 against 21.5 and 19.3.

**Voxel winners agree** (where either run has a tract; Other counts as a class):

| pair | per-voxel / smooth / per-volume | rescan, same session | rescan, other session |
|---|---|---|---|
| cIIIs01-2.5 | 71.0 / 69.7 / 69.0 % | 64.8 % (67.7 % on A's grid) | — |
| cIIIs01-2.1 | 73.3 / 72.8 / 72.6 % | 69.7 % (70.8 %) | 65.1 % (70.1 %) |
| cIs1 | 62.1 / 62.0 / 61.7 % | 56.9 % (57.6 %) | 53.0 % (55.2 %) |

**Field p99** (nominal readout):

| pair | A | per-voxel | smooth | per-volume | rescans |
|---|---|---|---|---|---|
| cIIIs01-2.5 | 155 | 146–156 | 138–151 | 123–145 | 125 |
| cIIIs01-2.1 | 182 | 161–189 | 156–200 | 155–177 | 199 (same session), 170 (other) |
| cIs1 | 114 | 106–112 | 105–114 | 105–114 | 95 (same session), 159 (next day) |

**What it shows:**

- **The per-voxel bootstrap understates a real rescan on every pair.**
  - Labels: by about 5 points within a session on cIIIs01 and 7 between sessions; by 8–10 on cIs1.
  - Paths: 3–8 mm shorter median distance than a rescan's.
  - Voxel agreement: 4–9 points higher than a rescan's.
- **The residuals are correlated in-plane, much less across slices.** Lag-1 correlation along x, y and z of the
  diffusion-weighted residuals:
  - cIIIs01-2.5: 0.53, 0.40, 0.09.
  - cIIIs01-2.1: 0.47, 0.31, 0.08.
  - cIs1: 0.30, 0.11, 0.05.

  The b0 residuals are correlated across slices too (z: 0.14–0.42; cIs1 0.42). Residuals hold model misfit as well as
  noise. The in-plane pattern is likely the 2D EPI reconstruction's. Per-voxel signs remove it.
- **On cIIIs01, keeping that correlation closes the within-session gap.**
  - Smooth signs: 35.0 % against the rescan's 36.3 % (2.5 mm), and 34.9 % against 36.9 % (2.1 mm).
  - Per-volume signs: 36.5 % and 35.8 %. They keep all of the correlation but spread widely between pairs
    (33–38 %), with only one sign per volume.
  - Path distances match the rescans too (12.7–13.9 mm against 13.1–14.0).
  - On cIIIs01-2.5, per-volume signs also spread the field to 123–145 Hz, bracketing the rescan's 125. Per-voxel signs
    stayed within 146–156.
- **Neither closes the between-session gap, nor the voxel gap.**
  - The one other-session rescan (cIIIs01-2.1) moves 39.1 % of labels: 3–4 points above smooth and per-volume signs,
    7 above per-voxel.
  - Voxel agreement moves at most 2 points toward the rescans'.
  - Another session brings head position, slice placement and registration error, which a noise bootstrap does not
    hold. Its field (170 Hz against A's 182) moved less than the same-session rescan's did (199), so the field alone
    does not explain the gap.
- **cIs1 stays 6–10 points short with every variant.**
  - Its residuals are less correlated (0.30 / 0.11), so the signs matter less there.
  - Its rescans move 19–21 mm against 13–15.
  - It differs from cIIIs01 in several ways at once: 38 slices (95 mm, the cortex cut at the top), 3 forward b0s against
    16, tracking at b = 3000, and 96 directions per shell.
  - Its replicates' fields barely vary (105–114 Hz), while its rescans give 95 and 159. A field resting on 3 forward b0s
    is a likely cause. That was not tested.

**For the ensemble:**

- **Use spatially correlated signs.** On cIIIs01, `flip="smooth"` (in-plane, FWHM 4) matches a same-session rescan
  within 1.3–2.0 points, with a steady spread. `flip="volume"` matches on average, but each pair is noisier.
- **Even so, the ensemble's confidence is optimistic for a rescan in another session.**
  - By about 3–4 points of label change on the one such rescan of cIIIs01, and more on a protocol like cIs1.
  - The double bootstrap's recalibration (entry above) was fitted on per-voxel replicates at 38 % change. It should be
    refitted with smooth signs and read as a same-session map.

Caveats:

- Two subjects, five rescans, one Philips scanner. (Corrected 2026-10-09: two Philips Achieva 3T scanners at one site,
  cIIIs01 on scanner B and cIs1 on scanner A.)
- The labeler's draw is held at 0, so these rates are the data's share only. The pilot's 38 % included draws.
- The pairs share their reference with the replicates. Replicate against replicate is therefore the comparison; A
  against a replicate is centered on A and changes less (27.7–37.4 %).
- FWHM 4 was chosen from the residual correlation. It was not fitted.
- An earlier, separate analysis of a third subject (cIIIs04) read 33.0 % for replicate against replicate and 37.8 % for
  a same-session rescan. It had no registration and gave the rescan another labeler draw, but it points the same way.
  Its data stays in DATA/ds003416/sub-cIIIs04 and `_masivar.py` loads it.

## 2026-10-08 Smooth signs on PAT16, and soft labels: labels change because paths change

`ensemble_pilot.py` and `ensemble_double.py` gain `--flip` and `--fwhm` (records in `added-smooth/`, `double-smooth/`;
`ensemble_pilot_added-smooth.json`, `ensemble_double_smooth.json`); `ensemble_double.py` now computes the
replicate-vs-replicate keep rate (was a constant) and the scan's residual correlation. New: `ensemble_soft.py` →
`ensemble_soft.json`. PAT16, M2 (MPS), 69 runs.

**The double bootstrap refitted with smooth in-plane signs (FWHM 4 voxels).** The entry above showed per-voxel signs
understate a same-session rescan on MASiVar's cIIIs01, and smooth signs match it there.

Terms: the **mixture** is a streamline's tract probabilities averaged over the ensemble's runs; the **vote fraction**
is the share of runs whose winning tract is its tract. **AUROC** says how well a stated confidence ranks the
streamlines whose tract changes (0.5 chance, 1 perfect). **ECE**, the expected calibration error, is the average gap
between stated and observed probability over bins of the statement. **Isotonic** and **votes table** are the two
recalibration maps (entry above). Rows marked "pilot" come from `ensemble_pilot.py`; rows marked "double" from
`ensemble_double.py`.

PAT16's diffusion-weighted residuals are spatially correlated too, more along y than x. Lag-1 correlation along x, y, z:
0.28, 0.43, 0.03 (MASiVar cIIIs01-2.5: 0.53, 0.40, 0.09).

| PAT16 | per-voxel signs | smooth signs |
|---|---|---|
| pilot: held-out run changes the reference's label | 33.2 % | 35.1 % |
| pilot: AUROC mixture / one run | 0.796 / 0.648 | 0.788 / 0.644 |
| replicate against replicate keeps the tract | 62.0 % | 59.9 % |
| double: change against rescans | 38.1 % | 40.3 % |
| double: ECE as stated, mixture / one run | 0.067 / 0.318 | 0.076 / 0.339 |
| double: held-out ECE, isotonic / votes table / voxel table | 0.0025 / 0.0026 / 0.0024 | 0.0031 / 0.0027 / 0.0019 |
| votes table: P(keep \| 11 of 11) | 0.915 | 0.901 |
| real scan: mean P(rescan keeps), isotonic / votes | 0.621 / 0.606 | 0.595 / 0.580 |
| real scan: share stated ≥ 0.9, isotonic | 14.2 % | 10.4 % |
| replicates' field p99 (reference 154.0 Hz) | 153.4–155.3 | 153.0–154.6 |

- **Smooth signs add 2 points of change on PAT16**, against 1.1–3.5 on MASiVar. The recalibrated confidence drops by
  2–3 points, and fewer streamlines reach 0.9.
- **Ranking (AUROC) and the recalibrated ECE (0.002–0.003) are the same as with per-voxel signs.**
- **PAT16's fields do not spread with smooth signs.** On MASiVar cIIIs01-2.1 they do (156–200 Hz against 161–189 per
  voxel; on cIIIs01-2.5, 138–151 against 146–156). PAT16 has 8 b0s from both phase-encoding polarities, against
  MASiVar's one reversed b0 per scan, so its field is likely better pinned.
- **From here the bootstrap's signs are smooth.** The maps above are the ones to use. `added/` and `double/` stay for
  comparison.

**Soft labels do not steady the output.** `ensemble_soft.py` reads the MASiVar retest records two ways: each
streamline as its winning tract (hard), or as its tract probabilities carried forward (soft).

| | streamlines, hard → soft | voxel tract mix, hard → soft | tract streamline-density overlap, hard → soft |
|---|---|---|---|
| cIIIs01-2.5, rescan | 36.3 → 35.3 % | 0.188 → 0.176 | 0.513 → 0.522 |
| cIIIs01-2.1, rescan same / other session | 36.9 / 39.1 → 35.8 / 38.0 % | 0.17–0.19 → 0.16–0.17 | 0.50–0.52 → 0.51–0.53 |
| cIs1, rescans | 47.5–49.9 → 46.2–48.6 % | 0.24 → 0.23 | 0.39–0.40 → 0.40–0.41 |

Columns:
- Streamlines, seed-matched. Hard: the share whose winning tract changed. Soft: the mean total variation distance
  between the two scans' tract-probability vectors (0 the same, 1 disjoint).
- Voxel tract mix: the total variation between the two scans' tract mix in each voxel, density-weighted.
- Tract streamline-density overlap: per named tract, the sum of minima over the sum of maxima of its streamline-density
  maps, mass-weighted (1 identical).

- **The labeler is rarely unsure.** The top tract averages 0.93–0.94, and 0.4–0.5 % of streamlines have it below 0.5.
- **Labels move because paths move.** One seed's streamline runs 13–21 mm apart (median MDF) on a rescan and reaches
  other anatomy. Carrying one run's probabilities forward cannot carry that.
- **The ensemble's mixture can,** because it pools over runs whose paths differ. That is the next entry's test.
- **Tract maps are unstable too:** a named tract's streamline-density map overlaps its rescan's by about half.

**Where this leaves the noise model:**
- the bootstrap with smooth signs matches a same-session rescan's spread on cIIIs01;
- what it cannot hold (another session's position, slice placement and registration, and a field resting on few
  b0s) is measured on real rescans from here.

## 2026-10-08 The ensemble scored on real rescans: it ranks well and states too much; a double bootstrap's map fits one of two subjects

`ensemble_retest.py rescans` (new) → an ensemble for every MASiVar rescan, 8 replicates with smooth signs (FWHM 4),
each corrected on the rescan's grid, resampled onto the reference's and tracked there (40 runs, M2).
`ensemble_coverage.py` → `ensemble_coverage_smooth.json`.

Each scan's ensemble (its run and 8 replicates, R = 9) is scored against the other scan's single run, both ways. That
makes 10 scorings over the five rescans. The labeler's draw is 0 throughout. Terms as in the entry above.

**Streamlines** (seed keys; the statement is P(the other scan keeps this streamline's tract)):

| | change | AUROC one run / mixture | ECE as stated: one run / mixture | mean stated P(keep), mixture |
|---|---|---|---|---|
| cIIIs01-2.5, same session | 36.3 % | 0.65 / 0.78 | 0.30 / 0.082–0.086 | 0.72 |
| cIIIs01-2.1, same session | 36.9 % | 0.65 / 0.78 | 0.30 / 0.081–0.083 | 0.71 |
| cIIIs01-2.1, other session | 39.1 % | 0.65 / 0.77 | 0.32–0.33 / 0.092–0.105 | 0.70–0.71 |
| cIs1, both rescans | 47.5–49.9 % | 0.65–0.66 / 0.73–0.74 | 0.40–0.43 / 0.15–0.18 | 0.67–0.69 |
| all 10 scorings | 42.0 % | 0.65 / 0.76 | 0.35 / 0.120 | 0.70 |
| held-out replicate (centered, 3; R = 8) | 32.7 % | 0.66 / 0.80 | 0.26 / 0.035 | 0.70 |

The centered row scores an ensemble of A and 7 replicates against the eighth, the pilot's scoring. It agrees with the
pilot's numbers (AUROC 0.80, ECE 0.03), on another subject and scanner.

- **On real rescans the ensemble still ranks well:** AUROC 0.73–0.78, against 0.65 for one run.
- **Its mean stated P(keep) exceeds the observed keep rate** by 8–10 points on cIIIs01 and 15–18 on cIs1. One run's
  probability is off by 30–43 (ECE).

**Recalibrated, leaving one pair out:**

| held-out pair | ECE as stated | isotonic, fitted on the other pairs' rescans | PAT16's double-bootstrap map, carried over | observed keep / PAT16 map's mean |
|---|---|---|---|---|
| cIIIs01-2.5 | 0.084 | 0.029 | **0.004** | 0.637 / 0.641 |
| cIIIs01-2.1 | 0.090 | 0.039 | **0.011** | 0.620 / 0.631 |
| cIs1 | 0.167 | 0.081 | 0.091 | 0.513 / 0.603 |

- **On cIIIs01, the double bootstrap's map transfers.** It was fitted on PAT16 (Siemens Trio, smooth signs) with no
  real rescan. On cIIIs01's three real rescans (Philips, another subject and protocol) it brings ECE to 0.004–0.011,
  and its mean statement is within a point of the observed keep rate.
  - It beats an isotonic map fitted on MASiVar's own rescans, likely because those include cIs1's.
  - The two cIIIs01 pairs share a subject, so holding one out is not a fully independent test.
- **On cIs1 every map fails.** It keeps 51 % of labels where every map states about 60 %. That is the protocol the entry
  above singles out (38 slices, 3 forward b0s).
- **Voxels** (named tracts; Other excluded): the vote fraction's ECE is 0.063–0.085 as stated. A table per vote count
  fitted on the other pairs brings it to 0.012–0.022.

**Envelopes.** A tract's envelope at level L is its highest-probability voxels holding L of its summed probability.
P(v, k) is the share of the 9 runs in which tract k wins voxel v. Coverage is the share of the other scan's voxels of
that tract inside the envelope, pooled over the 42 named tracts and weighted by those voxels. Other is excluded: it
is about half of all streamlines and would dominate the pool.

| stated level | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | 0.995 |
|---|---|---|---|---|---|---|
| rescans (10 scorings) | 0.32 | 0.46 | 0.53 | 0.60 | 0.64 | 0.68 |
| held-out replicate (3) | 0.39 | 0.54 | 0.61 | 0.68 | 0.72 | — |

One run's own tract covers 33 % of the rescan's (41 % of a held-out replicate's).

- **Envelopes are far too small for named tracts, even against the bootstrap's own runs.**
  - A voxel no run gave to a tract cannot be in its envelope.
  - Winner maps give each voxel to one class, so tracts that overlap, or share a voxel with Other, lose it.
  - At R = 9, even the widest envelope holds about two thirds of where a rescan puts the tract.
- **Recalibration of the level only reaches low targets.** Leaving one pair out:
  - A target of 0.5 uses the stated 0.71–0.80 and holds 0.56 on the held-out cIIIs01 pairs (cIs1 0.41).
  - No target of 0.8 or more is reached by any measured level. The widest holds 0.71–0.73 on cIIIs01 and 0.62 on cIs1.
- **Envelopes need more runs, or maps that let tracts overlap.** Neither was tested. Only R = 9 was run.

**Ensemble outputs are steadier between scans than single runs.**

| | single runs differ | ensembles differ |
|---|---|---|
| per seed (total variation of tract probabilities) | 0.36–0.39 (cIIIs01), 0.48–0.50 (cIs1) | 0.26–0.27, 0.35–0.37 |
| named winner territories overlap (Jaccard: sum of minima over sum of maxima) | 0.22–0.24, 0.15–0.16 | 0.36–0.39, 0.28–0.29 |

The ensemble's winner territories overlap their rescan's 1.6–1.9 times as much as one run's do, though still under
half. These are winner territories; the tract streamline-density overlap of the entry above is a different measure
(0.39–0.53 for single runs).

What this suggests:

- **The ensemble's ranking held on all five real rescans.**
- **Its recalibration can come from the double bootstrap** on a protocol like cIIIs01's (16 forward b0s): one healthy
  subject, three rescans. On cIs1's it did not.
- **Envelopes for named tracts stop near two thirds of a rescan's tract at R = 9.** They cannot yet be stated at 80 or
  90 %.

## 2026-10-08 Where and which: position holds on a rescan, a region's main named tract mostly holds, a streamline's name does not

`ensemble_where_which.py` → `ensemble_where_which_smooth.json`. Existing records only. Pairs: MASiVar's reference
against its rescans (on its grid), and replicate pairs with smooth signs, MASiVar's and PAT16's.

Spheres of 8 mm radius:
- MASiVar: 300 centers on dense white matter (voxels with at least 20 streamlines);
- PAT16: centers drawn from the tumor's margin (0–10 mm outside the mask), at least 4 mm apart (879).

PAT16's tumor mask (ds001226, on the T1) is put on the DWI grid by registering the T1 rigidly to the corrected mean b0
(`_register.py`, mutual information): 2,922 voxels (45.6 mL). Its outline sits on the b0's bright temporal lesion in
all three planes.

Measures:
- **density:** histogram intersection of all streamlines' voxel density (sum of minima over sum of maxima; 1
  identical);
- **direction:** density-weighted |cos| between the two runs' principal directions per voxel (voxels with at least 5
  streamlines in both);
- **density per tract:** each named tract's own intersection, mass-weighted;
- **sphere's main named tract and named mix:** over the streamlines visiting a sphere, the named tract with the most of
  them, and the total variation between the two runs' named-tract mixes. Other is excluded from these.

Other, the labeler's class for streamlines it gives no tract, is about half of all streamlines. It is the commonest
class in 88–99 % of spheres (median share 0.56–0.58). With Other counted as a tract, the "same main class" figure is
97–100 % and says little.

| | density, all | direction \|cos\| | density per tract | spheres: same main named tract | spheres: named mix total variation |
|---|---|---|---|---|---|
| cIIIs01-2.5 rescan / replicates | 0.765 / 0.779 | 0.970 / 0.972 | 0.497 / 0.514 | 89.0 / 88.6 % | 0.097 / 0.084 |
| cIIIs01-2.1 rescan same, other session / replicates | 0.767, 0.755 / 0.787 | 0.971, 0.967 / 0.974 | 0.507, 0.483 / 0.526 | 90.3, 82.3 / 88.3 % | 0.082, 0.100 / 0.074 |
| cIs1 rescans / replicates | 0.633–0.647 / 0.748 | 0.937–0.942 / 0.967 | 0.377–0.386 / 0.467 | 80.7–82.3 / 88.4 % | 0.126–0.127 / 0.091 |
| PAT16 replicates, tumor margin (spheres) | 0.697 | 0.972 | — | 86.1 % | 0.106 |

For PAT16 the density and direction are the spheres' medians; over the brain they are 0.763 and 0.976.

- **Position holds.**
  - Where white matter is and how much passes: 0.75–0.79 of the density is shared on cIIIs01, 0.63–0.65 on cIs1.
  - Its local direction: |cos| 0.94–0.97, about 14–20°.
- **A region's main named tract mostly holds:** the same in 89–90 % of spheres on a same-session rescan of cIIIs01,
  82 % on the other session, and 81–82 % on cIs1. The named mix moves by about 0.1.
- **A single streamline's name holds less:** 50–64 % keep their tract on a rescan (entries above).
- **Split by tract, density is shared much less** (0.48–0.53 against 0.75–0.79 for all streamlines). The per-tract
  figure is an average of each tract's own overlap, so it is not a split of the all-streamline figure.
- **Rescans are worse than replicates** on density, direction and the named mix. On the main named tract, same-session
  rescans of cIIIs01 match their replicates.
- **At PAT16's tumor margin, shared density is lower** than in MASiVar's dense white-matter spheres (0.70 against
  0.79–0.81). The sphere selection differs, and the cause was not tested.
- **For a question about a region,** which named tracts pass near it is more reproducible than any single
  streamline's name. It is still not stable: on a rescan, about one sphere in ten changes its main named tract within
  a session, and one in five across sessions or on cIs1.

## 2026-10-08 Where the paths part: mostly far from the seed, and only modestly more often at crossing fibers

`ensemble_crossings.py` → `ensemble_crossings_smooth.json`. Per ensemble (MASiVar's references with 8 smooth replicates,
PAT16's 17 runs), 4,000 seeds that every run labeled. Requiring every run to label the seed favors stable seeds.

Method:
- Each run's path is followed outward from the seed by arc length.
- A side diverges where the median distance of the other runs' points to the reference's exceeds 4 mm.
- Crossings: dipy's CSA ODF peaks on the tracked shell (relative threshold 0.5, 25° apart). Two or more peaks is a
  crossing.

| | sides diverging before 80 mm | divergence distance, diverging sides (quartiles) | crossing share at divergence / before it | crossing share, brain |
|---|---|---|---|---|
| cIIIs01-2.5 | 47 % | 30 / 40 / 52 mm | 73.1 / 60.7 % | 82.9 % |
| cIIIs01-2.1 | 48 % | 30 / 40 / 52 mm | 72.6 / 62.3 % | 84.2 % |
| cIs1 | 53 % | 24 / 34 / 46 mm | 85.7 / 76.3 % | 93.3 % |
| PAT16 | 46 % | 30 / 40 / 52 mm | 77.3 / 62.9 % | 89.2 % |

- **A seed's versions stay together a long way.** About half the sides diverge before 80 mm, at a median 34–40 mm from
  the seed. The rest end, or stay within 4 mm, to 80 mm.
- **Divergence points are crossings 1.1–1.2 times as often as the path before them,** a modest excess with no interval
  or test.
- **This crossing map cannot carry a sharper statement.** CSA peaks at this threshold call 83–93 % of the brain
  "crossing". A divergence point is also placed where the spread already exceeds 4 mm, after its cause.
- **"Identity decided at this crossing" is not grounded well enough to show.** It stays out of the output unless a
  sharper fiber model (CSD, multi-shell) and the spread's onset change this.

## 2026-10-08 PAT16 at the tumor: what one run says, and what 17 say

`ensemble_pat16_report.py` → `ensemble_pat16_report_smooth.json` and DATA/ensemble/PAT16/report_smooth.npz.
`ensemble_pat16_viewer.py` (+ `pat16_viewer.html`) builds a one-page viewer from them.

The ensemble is PAT16's 17 runs with smooth signs; r000 is the pipeline as it runs today, on the scan as acquired. The
tumor is 45.6 mL, temporal (entry above).

Definitions:
- a tract's voxels are those it wins with at least 2 streamlines;
- it **enters** the tumor in a run when 3 or more of its voxels (0.05 mL) lie inside the mask. Smaller amounts appear
  in mL but do not count;
- envelopes use P(v, k) over the 17 runs, with winners of at least 1 streamline, as calibrated in the entry above.

The levels are carried over from MASiVar's rescans (R = 9, named tracts): a target of 0.5 uses the stated 0.764. No
measured level reached 0.9, so the widest (0.995) is shown. On MASiVar that level held 0.68 of a rescan's tract; at
R = 17 its coverage was not measured.

The rows are the 11 of the 34 tracts within 20 mm that enter in every run, enter in some runs, or are the arcuate and
corticospinal tracts nearby. Abbreviations:
- ILF: inferior longitudinal fasciculus; MdLF: middle longitudinal fasciculus; AF: arcuate fasciculus;
- Sup-F, Sup-T: superficial frontal and temporal;
- SF, SO: striato-frontal and striato-occipital; TF: thalamo-frontal;
- CC1: corpus callosum segment 1; CB: cingulum bundle; CST: corticospinal tract.

| tract | today (r000): mL inside | 17 runs: P(enters) | mL inside, median (range) | P(within 5 mm) | widest envelope (stated 0.995): mL inside / within 5 mm |
|---|---|---|---|---|---|
| ILF | 1.05 | 1.00 | 1.12 (0.89–1.64) | 1.00 | 4.50 / 11.42 |
| Sup-F | 0.91 | 1.00 | 0.91 (0.77–1.23) | 1.00 | 3.83 / 9.59 |
| SF | 0.39 | 1.00 | 0.47 (0.30–0.55) | 1.00 | 4.97 / 12.19 |
| SO | 0.20 | 1.00 | 0.20 (0.06–0.34) | 1.00 | 1.33 / 2.11 |
| MdLF | 0.16 | 0.94 | 0.16 (0.03–0.27) | 1.00 | 2.91 / 7.81 |
| Sup-T | 0.11 | 0.76 | 0.08 (0.02–0.20) | 1.00 | 2.45 / 5.47 |
| CC1 | 0.03 | 0.71 | 0.05 (0.00–0.17) | 1.00 | 0.98 / 3.64 |
| TF | 0.05 | 0.29 | 0.02 (0.00–0.11) | 1.00 | 0.83 / 3.30 |
| CB | 0.00 | 0.35 | 0.03 (0.00–0.11) | 1.00 | 1.36 / 3.06 |
| AF | 0.00 | 0.00 | 0.00 (0.00–0.03) | 0.71 | 0.94 / 2.28 |
| CST | 0.02 | 0.00 | 0.00 (0.00–0.03) | 0.47 | 0.19 / 0.70 |

- **Some tracts enter the tumor mask in all 17 runs:** ILF, superficial frontal, striato-frontal, striato-occipital.
  For those, one run gets the amount roughly right, though the runs span −36 % to +46 % of the median.
- **For others, one run's yes or no is close to a coin toss.**
  - TF enters in 5 of 17 runs, CB in 6, CC1 in 12.
  - Today's run puts TF inside (3 voxels), CC1 at 2 voxels (below the 3-voxel rule) and CB outside.
- **The CST.**
  - Its voxels come within 5 mm in 8 of 17 runs. A single run cannot tell whether this patient's CST is in the 47 %
    or the 53 %.
  - Today's run has 1 CST voxel inside (0.02 mL), below the rule.
  - The CST holds 3.6 % (3.0–4.2 %) of the streamline-voxel visits in the 0–10 mm margin in every run, with no
    streamline threshold.
- **Envelopes at the widest measured level are wide** (ILF: 11.4 mL within 5 mm, against 3.8 mL for today's run), and
  on MASiVar they still held only two thirds of a rescan's tract.
- **Caveats:**
  - PAT16 has no rescan.
  - The streamline recalibration comes from its double bootstrap, which matched cIIIs01's real rescans.
  - The envelope levels come from MASiVar's pooled coverage at R = 9, cIs1 included.
  - None of this is clinical validation: nothing was checked against stimulation mapping or histology. These numbers
    measure how reproducible the pipeline's output is, not where the fibers run.

## 2026-10-08 Review of today's entries: corrections

Two reviewers checked today's entries, the ensemble doc and the PAT16 viewer against the result files. Neither had
written them.

**Corrections to the analysis:**
- **The sphere mix and envelope coverage had pooled Other in.** Other is about half of all streamlines and the commonest
  class in 88–99 % of spheres.
  - Redone over the 42 named tracts:
    - `ensemble_where_which.py` reports named-tract figures beside the with-Other ones.
    - `ensemble_coverage.py` restricts voxel calibration and coverage to named tracts, and stores the fine coverage curve.
  - The same main tract in a sphere fell from 98–100 % to 81–90 %.
  - A stated 0.9 envelope's coverage fell from 0.79 to 0.60, and the widest's from 0.89 to 0.68.
  - The PAT16 report's levels and the viewer follow the corrected curve.

**Corrections to the text:**
- Every MASiVar scan has one reversed b0. What sets cIs1 apart is 3 forward b0s against 16, 38 slices, b = 3000
  tracking and 96 directions per shell.
- The 199 Hz field belonged to the same-session rescan.
- Ranges previously taken from one pair are now given for all pairs.
- Sup-F's widest envelope is 3.83 / 9.59 mL.
- cIs1's per-voxel field range is 106–112 Hz.
- The run count is 124.
- The divergence medians are conditional on diverging.
- "Enters" is defined (3 voxels), and today's CC1 (2 voxels) does not count.
- Headings and conclusions resting on one subject or one patient now say so.

**Viewer:** a research-use notice; plain control names; the single run labeled as the current pipeline; every tract
that enters in any run listed; real-rescan figures kept apart from PAT16's replicate figures; and interpolated levels.

## 2026-10-09 Compact run records, and presence maps: outlines that hold 80 % of a rescan's tract

**Compact run records.** `tractline.compact` (new) ports the tractography incubation's storage (medseg,
tractography-incubation branch: `_geometry.py`, `_pipeline.encode_payload`). `RunRecord.save(compact=True)` writes
format 2; `RunRecord.load` reads both formats.

- **Geometry:** every vertex of every labeled streamline, rounded to a 0.05 mm grid, stored as 2nd-order predictive
  residuals (constant-velocity prediction) through blosc zstd. Lossless on the grid, so every vertex is within 0.025 mm.
- **Labels:** the 1,600 cluster log-probabilities as a rank field (keep "clip", depth 12, tail at T = 1), decoded back
  to dense log-probabilities with the tail spread evenly over the classes not kept.
- **Checks:**
  - On a stored run (41,595 streamlines), depth 12 changes tract probabilities by a median 0.0002 and at most 0.005.
    Tract argmax is the same in 99.998 % of streamlines.
  - The labels themselves are stored exactly.
  - `bench/compact_check.py` re-encodes cIIIs01-2.5's 18 smooth-sign runs and scores the pair both ways. Every ensemble
    metric matches the dense records to four decimals (mixture AUROC 0.7797 / 0.7798, ECE, voxel ECE, coverage).
    Only the single run's labeler AUROC moves (0.654 → 0.653).
- **Size:**
  - A record with full geometry on cIIIs04 (64,378 streamlines, 3.8 M vertices, steps of 1.8 mm) is 7.7 MB.
    Geometry is 5.8 MB of that (1.5 B per vertex), the field 0.8 MB.
  - A format 1 record (20-point paths, dense float16 probabilities) is 147–230 MB.
- `numcodecs` joins the `rankfield` dependency group. The package still needs numpy and torch only, and
  `test_dependencies.py` now blocks both.

**Presence maps.** `ensemble_coverage.py --maps presence --min-count T` (new): a run's voxel map gives tract k every voxel
where it has at least T streamlines, so tracts may overlap. These are the voxel rank field's classes above NONE at
tau = T. Winner maps give each voxel to its densest class only (Other included), so tracts lose voxels where they
overlap.

Same 10 scorings of MASiVar's rescans as the entry "The ensemble scored on real rescans" (R = 9, smooth signs, named
tracts; coverage is the share of the rescan's voxels of a tract inside the ensemble's envelope):

| maps | stated 0.5 / 0.8 / 0.9 | widest (0.995) | one run's tract covers the rescan's | voxel ECE | winner-territory overlap, cIIIs01: single run / ensemble |
|---|---|---|---|---|---|
| winner | 0.32 / 0.53 / 0.60 | 0.68 | 0.33 | 0.074 | 0.22–0.24 / 0.36–0.39 |
| presence ≥ 1 | 0.44 / 0.70 / 0.80 | 0.90 | 0.63 | 0.045 | 0.48–0.50 / 0.64–0.65 |
| presence ≥ 2 | 0.44 / 0.69 / 0.78 | 0.88 | 0.63 | 0.050 | 0.50–0.52 / 0.63–0.64 |
| presence ≥ 5 | 0.42 / 0.66 / 0.75 | 0.86 | 0.64 | 0.061 | 0.50–0.53 / 0.60–0.63 |

Levels recalibrated leaving one pair out (presence ≥ 2):

| target | cIIIs01-2.5 | cIIIs01-2.1 | cIs1 |
|---|---|---|---|
| 0.8 | 0.86 (stated 0.93) | 0.86 (stated 0.945) | 0.72 (stated 0.89) |
| 0.9 | 0.91 (widest) | 0.91 (widest) | 0.82 (stated 0.985) |

- **Letting tracts overlap is what makes the outlines usable.**
  - As stated, a 90 % envelope now holds 78–80 % of a rescan's tract, against 60 %.
  - Recalibrated, a target of 80 % holds 84–86 % on cIIIs01, where winner maps could not reach it at all.
  - On cIIIs01, 90 % is reachable at the widest level.
- **cIs1 stays about 10 points short** with every map, as in every entry above.
- **One run's map already covers twice as much** (0.63 against 0.33). Winner maps understated even the single run.
- **The threshold matters little between 1 and 5 streamlines.** Presence ≥ 2 is the default from here: no single stray
  streamline marks a voxel.
- **Winner mode reproduces the committed results** (`ensemble_coverage_smooth.json`): keys renamed, values within
  ±0.0001.
- **Still open:**
  - A held-out subject: cIIIs04 (running, entry below when done) will test the presence calibration on a subject it was
    not fitted on.
  - The PAT16 report and both viewers still draw winner maps.

## 2026-10-09 A held-out subject: the calibration holds on cIIIs04, and more runs help less after 13

`ensemble_retest.py` with the new pair cIIIs04-2.1 (MASiVar cohort III, subject 04, 2.1 × 2.1 × 2.2 mm, b1000 ×40 +
b2000 ×56, 16 forward b0s). The reference is s1Bx1 scan 1; its rescans are s1Bx1 scan 2 (same session), and s1Bx3 and
s1Bx5 scan 1 (other sessions). Every scan has a 17-run ensemble (smooth signs, FWHM 4). 71 runs on the M2, all as
compact records: 540 MB. `ensemble_coverage.py --maps presence --min-count 2` (with `--reps`, `--tag`) →
`ensemble_coverage_smooth_presence2_with_cIIIs04.json` and `..._cIIIs04_R{5,9,13,17}.json`; `ensemble_where_which.py`
now includes the pair.

**Held out.** All four pairs are scored at R = 9; leaving one pair out, cIIIs04 is fitted on the three earlier pairs
only (cIIIs01 ×2, cIs1), and it is a subject none of the maps saw.

| cIIIs04, held out | as stated | recalibrated |
|---|---|---|
| streamline ECE (6 scorings) | 0.087 | isotonic on the other pairs 0.021; **PAT16's double-bootstrap map 0.0066** |
| mean stated P(keep), observed 0.623 | 0.709 | isotonic 0.602; PAT16's map 0.629 |
| voxel ECE (presence ≥ 2) | 0.038 | table on the other pairs 0.009 |
| envelope coverage, target 0.8 | 0.73 (stated 0.8) | 0.84 (stated 0.925) |
| envelope coverage, target 0.9 | 0.82 (stated 0.9) | 0.92 (the widest, 0.995) |

Per rescan (A's ensemble scored on the rescan / the rescan's on A):

| rescan | change | AUROC mixture / one run | ECE mixture as stated | coverage at stated 0.9 |
|---|---|---|---|---|
| same session | 35.4 % | 0.790 / 0.646 | 0.063 / 0.067 | 0.84 / 0.83 |
| session 3 | 38.2 % | 0.773 / 0.638 | 0.091 / 0.091 | 0.82 / 0.82 |
| session 5 | 39.7 % | 0.765 / 0.639 | 0.106 / 0.104 | 0.81 / 0.80 |

- **The double bootstrap's map transfers to a second healthy subject.** It was fitted on PAT16 (Siemens, simulated
  rescans only). On cIIIs04's real rescans its ECE is 0.0066, and its mean statement is within a point of the observed
  keep rate. This is as on cIIIs01 (0.004–0.011, entry "The ensemble scored on real rescans"), and better than a map
  fitted on MASiVar's own rescans.
- **The presence-map envelope levels, fitted on the other pairs, hold on cIIIs04:** a target of 0.8 holds 0.84, 0.9
  holds 0.92.
- **As before, another session changes more than the same session** (35 → 38–40 %), and the as-stated ECE grows with
  it.

**More runs** (cIIIs04, presence ≥ 2, the first R runs of each 17-run ensemble; the 6 rescan scorings):

| R | coverage at stated 0.5 / 0.8 / 0.9 | streamline ECE / AUROC (mixture) | voxel ECE | mean stated P(keep) |
|---|---|---|---|---|
| 5 | 0.45 / 0.70 / 0.79 | 0.112 / 0.754 | 0.069 | 0.734 |
| 9 | 0.46 / 0.73 / 0.82 | 0.087 / 0.775 | 0.038 | 0.709 |
| 13 | 0.47 / 0.74 / 0.83 | 0.077 / 0.784 | 0.028 | 0.700 |
| 17 | 0.47 / 0.74 / 0.84 | 0.072 / 0.789 | 0.023 | 0.694 |

- **Every measure improves with R, with most of the gain by R = 9–13.** From 13 to 17 runs: coverage +0.01, AUROC
  +0.005, voxel ECE −0.005.
- **The as-stated confidence falls with R toward the observed keep rate (0.62),** so a recalibration map belongs to
  an R. The maps here were fitted at R = 9 (MASiVar) and R = 11 (PAT16's double bootstrap).

**Where and which, on cIIIs04:**
- The main named tract in an 8 mm sphere is the same on 90 % of spheres on the same-session rescan, 90 % on session
  3, and 86 % on session 5 (replicate pairs 88 %).
- The named mix moves by 0.07–0.08.
- Density shared: 0.75–0.78 (all tracts) against 0.50–0.53 per tract.
- Direction: |cos| 0.97.

Caveats: three healthy subjects in all on two Philips scanners of one model at one site (MASiVar: cIIIs01 and cIIIs04 on
scanner B, cIs1 on scanner A), one Siemens patient; R beyond 17 untested;
no ground truth beyond reproducibility.

## 2026-10-09 Landmark sites on cIIIs04: the ensemble and the rescans where tracts cross and where one runs alone

`ensemble_landmarks.py` (CPU, about 80 s; no pipeline runs) reads cIIIs04-2.1's compact records: A's 17-run ensemble
(A, smooth/r001..r016) and the three rescans' single runs on A's grid. It writes `ensemble_landmarks_smooth.json` and
`landmarks.npz` beside the records. `ensemble_landmarks_3d.py` draws them, and the viewer is published (private) at
https://claude.ai/artifact/WnTmmSCx5hTrvDe1m9C67m.

Terms:
- **Presence:** a tract with at least 2 streamlines in a voxel.
- **Other:** the labeler's class for streamlines it assigns to no named tract. It is the most frequent label at every
  site.
- **Mode:** the tract most of A's 17 runs give a seed.
- **Stated:** the mode's share of the 17 runs, read as P(another run gives the mode).

**Sites, without a T1.** Each site is found from the ensemble itself, on presence shares (or mean counts) smoothed by
a Gaussian of σ = 1 voxel:
- **centrum semiovale, left and right:** corticospinal, callosal (CC3–CC5) and superior longitudinal (SLF-II,
  SLF-III, arcuate) fibers all present; the largest of the three shares' minimum.
- **temporal stem, left and right:** the most posterior voxel where the uncinate and inferior occipito-frontal (IOFF)
  fasciculi both have a smoothed share of at least 0.6. Their best overlap alone falls in the frontal lobe.
- **corticospinal tract alone, left and right:** the CST's share minus the largest other named tract's share,
  maximized.
- **genu and splenium:** CC1's and CC7's densest voxel (mean streamline count) within 3 mm of the midline. A second
  callosal segment is present at both (CC2, CC6).

The names are approximate. At each site the analysis takes:
- the voxel rank field from all 17 runs: classes by mean streamline count, those of 2 or more; the table below shows
  the top 5;
- the seeds whose full streamline in A passes within 4 mm and that all 17 runs labeled, 464–1056 per site.

The callosal and SLF classes that define the left centrum semiovale site are present there in every run but are not
among its top 5 by count.

| site | all 17 runs agree | stated | replicate (leave one out): gives the mode / stated | rescan gives the mode: same session / s3 / s5 | rescan's tract among the 17 runs' (s5) | rank field at the site (mean streamlines) |
|---|---|---|---|---|---|---|
| centrum semiovale, L | 0.19 | 0.75 | 0.72 / 0.75 | 0.69 / 0.64 / 0.61 | 0.90 | Other 31, AF 21, Sup-F 12, SLF-III 8, CST 6 |
| centrum semiovale, R | 0.17 | 0.74 | 0.71 / 0.74 | 0.69 / 0.64 / 0.61 | 0.92 | Other 51, SLF-II 37, SLF-III 13, Sup-F 12, CC4 6 |
| temporal stem, L | 0.24 | 0.76 | 0.74 / 0.76 | 0.70 / 0.67 / 0.64 | 0.89 | Other 116, IOFF 21, UF 6 |
| temporal stem, R | 0.20 | 0.75 | 0.73 / 0.75 | 0.70 / 0.67 / 0.62 | 0.89 | Other 160, IOFF 35, EmC 9, UF 3, EC 2 |
| CST alone, L | 0.17 | 0.81 | 0.79 / 0.81 | 0.77 / 0.75 / 0.76 | 0.94 | Other 41, CST 16 |
| CST alone, R | 0.12 | 0.74 | 0.71 / 0.75 | 0.70 / 0.67 / 0.68 | 0.93 | Other 41, CST 13 |
| genu | 0.18 | 0.77 | 0.74 / 0.77 | 0.70 / 0.70 / 0.70 | 0.94 | Other 35, CC1 15, CC2 14 |
| splenium | 0.06 | 0.69 | 0.66 / 0.69 | 0.58 / 0.59 / 0.59 | 0.91 | Other 33, CC7 24, CC6 9 |

Abbreviations: AF arcuate fasciculus; Sup-F superficial frontal; SLF superior longitudinal fasciculus; CST
corticospinal tract; CC1–CC7 corpus callosum segments, front to back; IOFF inferior occipito-frontal fasciculus; UF
uncinate fasciculus; EmC extreme capsule; EC external capsule.

**The statement is not inflated by choosing the mode on the same runs.**
- Leaving one run out, the other 16 runs' mode has nearly the same share: 0.764 against 0.764 in-sample over the
  whole brain.
- The left-out run gives that mode in 0.740, so even a replicate falls 2.4 points short, 1.4–3.7 at the sites. A
  share is a noisy estimate of a probability.

**A real rescan falls further short.**
- Whole brain, the rescan gives the mode in 0.703 (same session), 0.674 (session 3) and 0.661 (session 5), against
  0.764 stated: 6.1, 9.0 and 10.4 points short.
- So about 4–8 points of the gap are the rescan's own, beyond what a replicate shows.
- At the sites, the stated value minus the rescan's is:
  - 4–11 points for the same session;
  - 6–11 for session 3;
  - 4–15 for session 5.
- At session 5 the gap is largest where tracts cross: 12–15 points at the centrum semiovale and temporal stem, against
  4–7 at the CST-alone sites and the genu.
- The splenium is the least stable site. Only 6 % of its seeds have all 17 runs agreeing, split between CC7 and CC6,
  and it falls about 10 points short in every session.

**Runs rarely all agree.** Only 6–24 % of the seeds near a site get the same tract in all 17 runs. Other is the mode
for most seeds at every site (511 of 565 at the left CST-alone site), so the site names describe a minority of the
streamlines there.

**The rescan's tract is nearly always one the 17 runs gave.**
- Over all seeds: 89–94 % at the sites (session 5) and 92–95 % over the whole brain.
- Over the seeds whose label changed: 79–86 % over the whole brain.

**Runner-up.** For each streamline, the labeler's second choice is the tract with the second-highest probability
(cluster probabilities summed per tract). Over the 44,717 seeds labeled in all 17 runs:
- When a rescan changes a streamline's tract (35–39 %), the new tract is A's runner-up in 48–51 % of cases.
- When A's runs split, the most frequent tract other than run A's own is A's runner-up in 50 %.
- Out of 42 alternatives, that is informative: the runner-up is the right kind of alternative.
- Its probability is not useful as a statement. Its median is 0.002, though the runs split on 80 % of seeds.
- The ensemble's set of tracts covers more of the changes (79–86 %) than the single runner-up does (about 50 %).

**Voxel runner-up.** In each voxel, the named tracts are ranked by mean streamline count over the 17 runs, leaving
out Other. The viewer draws where each shown tract comes second (`rank2` in `landmarks.npz`).

**Envelopes per tract.** The 80 % envelope uses the stated level 0.925. That level was fitted at R = 9 on the other
subjects and is applied here to 17-run votes without refitting. Its thresholds are at least 2–4 of 17 runs, by tract.

For the nine tracts drawn, which were chosen because the sites concern them, the envelope holds 85–92 % of each
rescan's tract voxels. Examples: CST 0.85–0.89, CC4 0.88–0.92, UF 0.89–0.90, ILF 0.85–0.89. This is not compared with
the pooled 0.84 (R = 9, all 42 tracts).

**The viewer** (`ensemble_landmarks_3d.py`, template `landmarks_3d.html`, 5.2 MB) shows the whole brain:
- per tract, presence shells: the core (at least 9 of 17 runs), the 80 % envelope, the widest (any run), and where it
  is the named runner-up;
- dots where a chosen rescan has a tract outside its 80 % envelope;
- the 8 sites. At a site, shells and dots are cropped to 30 mm. The fans are full streamlines (compact records) from
  all 17 runs, colored by tract or by each run's runner-up, with the rescans' versions in white. Up to 16 split seeds
  (mode in at most 60 % of runs, some named) and 16 agreeing seeds are drawn.

A reviewer agent checked the first draft against the JSON. Its corrections are in, including the leave-one-out
baseline it asked for.

Caveats:
- one healthy subject;
- sites found from the tracts themselves, not from anatomy;
- Other is left out of the shells;
- dots outside an envelope may come from registration error or real scan-to-scan change as well as from a tight
  envelope.

## 2026-10-09 Endpoints against FastSurfer's anatomy: streamlines consistent with their bundle's rule keep their name more often

**Anatomy.**
- FastSurfer (VINN, `fastsurfer:asegdkt`, 2.5.4) ran through haversack on Modal: `tools/ranked_emit_modal.py
  --source openneuro`, which fetches the T1 on the worker. It took 84 s of compute per subject.
- Each subject's 79-class ranked field became a duckn store (about 7 MB) and was restored to aseg + DKT labels
  (lateralized) on the T1 grid. All of it is in DATA/fastsurfer/. The stores were built with `uv run --isolated --extra
  fastsurfer --extra duckn` in haversack, because its .venvs/fastsurfer predates the labelfield pin.
- Registration to the corrected mean b0, rigid:
  - PAT16: tumor_on_dwi()'s cached transform;
  - cIIIs04: FastSurfer's conformed T1 to A_meanb0, with `bench/_register.py`.
  - Both were checked by eye (`bench/endpoints_overlay.py` → `ensemble_endpoints_registration.png`): the ventricles
    and the cortical edge sit on the b0's.

**Method** (`bench/ensemble_endpoints.py` → `ensemble_endpoints.json`).
- **End regions:** each end's region is the nearest cerebral gray-matter label within 4 mm. The cerebellar cortex is
  excluded, since basal ends otherwise reached across the tentorium.
- **Field-of-view edge:** an end within 3 mm of the DWI's inferior face, with no gray nearby, is "fov-edge". The field
  of view cuts the brainstem: 86 % of CST streamlines end within 3 mm of it.
- **Side:** an end's side is taken from the midsagittal plane, not from its nearest label's hemisphere.
- **Deep structures:** the thalamus and striatum count as entered along the path. cIIIs04 uses the full streamlines;
  PAT16 uses 20-point paths, which can miss a short passage.
- **Rules** (in the script) by kind:
  - pair: one end in each set, on one side;
  - within: both ends in the two sets' union, for the superficial pair classes, whose DKT lobe borders are uncertain;
  - callosal: both ends in the segment's own cortex, on opposite sides;
  - via: the thalamic and striatal classes, which the ORG atlas names by the structures they pass. The path enters the
    structure and one end lies in the lobe, on its side;
  - cingulum: one end in cingulate, parahippocampal or entorhinal cortex or the hippocampus, the other there or in
    the adjacent medial cortex, on one side.
- **Unscored:** an AF, UF or IOFF streamline with an end in the insula. Also the capsules, corona radiata, PLIC, MdLF,
  SO, the cerebellar classes and Other.

**An independent review revised the first pass** (the agent's report is in this session; the first pass is in
a7c07a9). The first pass:
- tested the thalamic and striatal classes on where tracking stops;
- accepted any left-right cortical pair as callosal, and any medial pair as cingulum;
- let basal ends take cerebellar labels;
- failed the CST at the field-of-view edge;
- missed standard terminations: the AF's rostral middle frontal and ventral postcentral, the UF's superior and
  rostral middle frontal (DKT has no frontal pole), and the SLF-II's pars opercularis and triangularis;
- gave the IOFF precentral, paracentral and caudal middle frontal ends, which it should not have.

The first pass's sentence that the thalamic classes are "misnamed" corticopontine courses is withdrawn. The atlas
class may include them.

**Consistent share** (17 runs pooled, scored named streamlines):
- **cIIIs04:** 0.66 (0.57 without the superficial classes); each rescan 0.65–0.67.
- **PAT16:** 0.59 (0.53).

| bundle | cIIIs04 | PAT16 |
|---|---|---|
| AF | 0.94 | 0.72 |
| IOFF | 0.84 | 0.79 |
| UF | 0.78 | 0.76 |
| ILF | 0.77 | 0.69 |
| CST | 0.76 | 0.68 |
| SLF-II / SLF-III / SLF-I | 0.73 / 0.71 / 0.47 | 0.49 / 0.61 / 0.61 |
| TF / TO / TP / TT | 0.68 / 0.58 / 0.50 / 0.48 | 0.66 / 0.48 / 0.50 / 0.59 |
| SF / SP | 0.77 / 0.46 | 0.47 / 0.52 |
| CB | 0.22 | 0.37 |
| CC1–CC7 | 0.51, 0.39, 0.47, 0.31, 0.19, 0.42, 0.39 | 0.34, 0.42, 0.32, 0.30, 0.19, 0.28, 0.43 |
| superficial (Sup-*) | 0.56–0.94 | 0.51–0.84 |

- **The long association bundles are mostly consistent** (AF, IOFF, UF, ILF, CST at 0.76–0.94 on cIIIs04).
- **The cingulum (0.22) and callosal segments (0.19–0.51) are mostly not consistent with segment- and
  structure-specific rules.** Whether that reflects the tracking, the labeler's grouping or rules stricter than the
  atlas's own clusters is not settled.
- **PAT16 is lower for the AF and SLF-II,** both near its temporal tumor. FastSurfer is trained on healthy brains and
  its labels near the tumor are unchecked.

**Consistency and the rescan** (cIIIs04, seeds labeled in all 17 runs, A's streamline scored):

| rescan | keeps its name: consistent / inconsistent (n) | within bundles, weighted (bundles; negative) | gives the mode: consistent-run mode / all-run mode |
|---|---|---|---|
| same session | 0.60 / 0.47 (11,038 / 6,298) | +0.11 (28; SP, TP, CC2, CC3) | 0.55 / 0.70 |
| session 3 | 0.57 / 0.44 | +0.11 (28; SLF-I, TP, CC3, CC5, Sup-T) | 0.53 / 0.67 |
| session 5 | 0.56 / 0.41 | +0.13 (28; CC2) | 0.52 / 0.66 |

- **A streamline consistent with its bundle's rule keeps its name on a rescan 12–15 points more often pooled, and
  11–13 points within bundles.** A tenth to a sixth of the pooled gap is which bundles are involved. In a few bundles
  (mostly callosal) the gap is negative.
- **Voting only among consistent runs predicts the rescan worse,** 0.52–0.55 against 0.66–0.70. This is expected by
  construction: it discards votes from a process that a rescan reproduces. It shows that the pipeline reproduces its
  own inconsistent courses, not that consistency and reproducibility are independent.
- **Of the inconsistent streamlines, 41–47 % keep the same name on a rescan.**

**What this adds.** Beside the ensemble's (corrected) agreement, a second number per streamline or region: whether
its course is consistent with its bundle's endpoint rule. It is a weak anatomical check, not a measure of accuracy.
The rules are a first pass, revised once, and their permissive or strict errors move the shares by tens of points
(the CC rule alone moved CC1 from 0.93 to 0.51).

Caveats:
- The rules have not been reviewed by a human anatomist.
- 4 mm nearest-gray lookup.
- One healthy subject and one patient.
- PAT16 uses 20-point paths, and its labels near the tumor are unchecked.
- The runner-up and rank information in FastSurfer's store is not used yet.

## 2026-10-13 The cIIIs04 viewer refocused on the CST, AF and UF, with FastSurfer's anatomy

A clinical collaborator suggested focusing on the corticospinal tract, arcuate fasciculus and uncinate fasciculus.
`ensemble_landmarks.py` and `ensemble_landmarks_3d.py` were rerun; the viewer's link is unchanged
(https://claude.ai/artifact/WnTmmSCx5hTrvDe1m9C67m).

**What changed.**
- **Tracts:** the three are drawn by default; the others are listed but off.
- **New sites:** the posterior limb of the internal capsule, left and right, from FastSurfer. Each is the white-matter
  voxel within 5 mm of both the thalamus and the lentiform nucleus that lies nearest the centroid of all such voxels.
  - They land at (−10, 17, 41) and (22, 16, 40) mm, between each thalamus and lentiform nucleus and behind the
    thalamus's front.
  - They replace the tract-defined "CST alone" sites. The other sites are unchanged.
- **Anatomy surfaces:** FastSurfer's sensorimotor cortex, language cortex, orbitofrontal cortex, thalamus, lentiform
  nucleus, brainstem and lateral ventricles. They are carried onto the DWI by the T1-to-DWI transform and cropped with
  the shells.
- **Endpoint coloring:** each fan version can be colored by the endpoint check (ensemble_endpoints.py's rules): 1,584
  consistent, 517 not, 1,369 unscored.
- **Focus panel:** per tract, as in the table below.

**The focus tracts** (`focus tracts` in ensemble_landmarks_smooth.json; A's streamlines named so, among seeds labeled
in all 17 runs):

| | CST | AF | UF |
|---|---|---|---|
| streamlines | 443 | 646 | 426 |
| stated (the runs' share giving it) | 0.40 | 0.74 | 0.59 |
| a fresh run keeps it | 0.36 | 0.73 | 0.56 |
| rescan keeps it: same session / s3 / s5 | 0.27 / 0.30 / 0.33 | 0.67 / 0.63 / 0.57 | 0.47 / 0.51 / 0.40 |
| changed names go to (s5) | Other 75 %, TF 5 % | Other 60 %, Sup-PT 9 % | Other 83 %, SF 5 % |
| 80 % outline holds the rescan's bundle (s5) | 0.85 | 0.86 | 0.90 |
| ends consistent with the rule | 0.76 | 0.94 | 0.78 |

- **The CST's name is the least stable of the three per streamline,** though the collaborator expects it to be the
  most reliable. Three quarters of the changes go to Other. The ensemble's own statement (0.40) already flags this.
- **Its territory is steady:** the 80 % outline holds 85 % of a rescan's CST.
- So for the CST, voxel-level outlines answer the clinical question better than streamline names. The likely cause is
  the labeler sorting corticospinal streamlines between CST and Other; it has not been checked.

## 2026-10-13 Why CST streamlines become Other: their lower end turns into the cerebellum at the pons

`ensemble_cst_other.py` → `ensemble_cst_other.json`. It compares cIIIs04's run A with a replicate (smooth/r001), the
same-session rescan and the session 5 rescan, for A's streamlines named CST (482–507, seeds labeled in both).

| | replicate | same session | session 5 |
|---|---|---|---|
| keeps CST | 0.33 | 0.26 | 0.31 |
| Other: another cluster's outlier twin / a non-anatomical cluster / its own cluster's twin | 0.35 / 0.18 / 0.02 | 0.36 / 0.20 / 0.02 | 0.29 / 0.20 / 0.02 |
| another named tract | 0.11 | 0.15 | 0.18 |
| MDF between versions, median: kept / became Other | 3.0 / 19.9 mm | 4.0 / 20.9 mm | 4.7 / 21.0 mm |
| upper end moved: kept / Other | 1.7 / 3.8 mm | 2.2 / 4.0 mm | 2.5 / 10.4 mm |
| lower end moved: kept / Other | 3.9 / 32.1 mm | 5.0 / 34.7 mm | 7.7 / 35.6 mm |
| the Other version's lower end in the cerebellum (A's version, same seeds) | 0.36 (0.06) | 0.37 (0.07) | 0.39 (0.07) |
| fits the CST endpoint rule: A's version / the Other version | 0.76 / 0.20 | 0.75 / 0.22 | 0.75 / 0.22 |
| labeler's P(CST) in A, median: kept / became Other | 0.959 / 0.945 | 0.957 / 0.941 | 0.961 / 0.934 |

- **It is the path, not the labeler.**
  - A's versions are good CST courses: precentral or postcentral cortex to the brainstem, and 75 % fit the rule
    whether or not they later change.
  - The labeler is equally sure of both groups (P(CST) about 0.94–0.96).
  - On the other run, the cortical end stays within a few millimeters, but the lower end moves 32–36 mm.
  - About 37 % of the Other versions end in the cerebellum, against 6–7 % of A's: they cross from the pons into the
    cerebellum (the middle cerebellar peduncle) instead of descending. Others stop early or take another brainstem
    course.
  - Only 20–22 % of the Other versions still fit the CST rule. The labeler calls them Other because they are no longer
    CST-shaped, which is right.
- **It is not the outlier twin.** Only 2 % go to their own cluster's outlier twin (cluster + 800). Most go to twins of
  non-CST clusters, or to the atlas's non-anatomical clusters.
- **A replicate does the same as a rescan** (keeps 0.33 against 0.26–0.31). So the cause is tracking variability at
  the pons, where descending corticospinal fibers cross transverse pontine fibers, not anything particular to
  rescanning. This crossing is a known difficulty for CST tractography.
- **"Unlabeled" lower ends** (about half of A's) lie outside FastSurfer's labeled brain at the bottom of the field of
  view, around the lower brainstem.

What it suggests:
- **Constrain the descent anatomically.** A CST path should not enter the cerebellum: FastSurfer gives the
  cerebellum, brainstem and ventral DC. The constraint can stop such a path or penalize turning into the middle
  cerebellar peduncle.
- **Report the CST at the voxel level.** Its outline holds 85 % of a rescan's bundle.
- **Test a tracker setting at the pons** that keeps to the superior–inferior fiber population where two cross.

## 2026-10-13 Pooling sessions: three scans' ensembles predict a fourth scan almost as stated, with no correction

`ensemble_pooled.py` → `ensemble_pooled.json`. cIIIs04's four scans (A and the same-session, session 3 and session 5
rescans) each have a 17-run ensemble on A's grid. Leaving each scan out in turn, its own run is predicted by:
- **single:** each other scan's ensemble alone, averaged over the three;
- **pooled:** the three other ensembles together, 51 runs, each scan weighted equally.

Envelope levels are as stated, with no recalibration. Streamline figures use the 33,228 seeds labeled in all 68 runs.

**All named tracts** (voxels pooled over the 42, weighted by the target's voxels), held-out scan A / same session /
session 3 / session 5:

| | single | pooled |
|---|---|---|
| widest outline holds | 0.95 / 0.95 / 0.94 / 0.94 | 0.99 / 0.99 / 0.98 / 0.98 |
| envelope stated 0.8 holds | 0.76 / 0.76 / 0.74 / 0.75 | 0.81 / 0.81 / 0.79 / 0.78 |
| envelope stated 0.9 holds | 0.86 / 0.86 / 0.85 / 0.85 | 0.90 / 0.90 / 0.88 / 0.88 |
| envelope stated 0.95 holds | 0.91 / 0.91 / 0.90 / 0.90 | 0.95 / 0.95 / 0.94 / 0.93 |
| streamlines, the mode: stated / observed | 0.77 / 0.66–0.68 | 0.72–0.73 / 0.69–0.72 |

**Focus tracts** (stated 0.8 and 0.9, ranges over the four held-out scans):

| | single 0.8 / 0.9 | pooled 0.8 / 0.9 | widest: single / pooled |
|---|---|---|---|
| CST | 0.73–0.79 / 0.86–0.89 | 0.77–0.83 / 0.86–0.92 | 0.94–0.96 / 0.98–0.99 |
| AF | 0.74–0.77 / 0.84–0.86 | 0.78–0.83 / 0.88–0.91 | 0.94–0.96 / 0.98–0.99 |
| UF | 0.75–0.79 / 0.86–0.89 | 0.79–0.82 / 0.89–0.91 | 0.93–0.95 / 0.98–0.99 |

- **Pooling sessions makes the stated levels hold without any correction map.**
  - An envelope stated as 80 % holds 78–81 %, 90 % holds 88–90 %, and 95 % holds 93–95 %. One scan's ensemble falls 4–5
    points short at each level.
  - The widest outline misses 1–2 % of the new scan's tract, against 5–6 %.
  - The streamline statement shrinks to a 0–4 point gap (from 8–10). The pooled ensemble states less (0.72 against
    0.77) and is right more often (0.69–0.72 against 0.66–0.68).
- **What the single-scan ensemble lacks is between-session variation.** Head position, slice placement, registration
  and the distortion field are what resampling one scan cannot produce.
- **Caveat: run count is confounded.** Pooled is 51 runs, single 17. On this subject, going from 13 to 17 runs gained
  about 0.01 in coverage (NOTES "A held-out subject"), so most of the 0.04–0.05 gain is likely the sessions, not the
  count; a single scan with 51 runs was not available to check.
- **Other caveats:**
  - one subject, one protocol;
  - for the A and same-session held-out scans, the pool includes a scan from the same session.

What it means:
- For this subject, three sessions' ensembles give honest uncertainty with no correction.
- For a patient with one scan, the honest route is to add a session term learned from test-retest data like this. The
  pooled-versus-single gap is the size of that term: about 4–5 points of coverage at each stated level, and a widest
  outline about 4 % too tight.

## 2026-10-13 Direction spread along the CST: flat, and adds nothing to the vote share

`ensemble_cst_direction.py` → `ensemble_cst_direction.json`. This is the simplest test of per-voxel, per-tract direction
statistics.

**Method.**
- For each of cIIIs04's 17 runs and each voxel, the CST streamlines' direction is the principal axis of their
  segment tangents (at least 3 segments).
- For each voxel where at least 9 runs have one, the spread is the mean angle between each run's direction and their
  common one (3,875 voxels).

**Where.**
- The spread is flat: median 4.5° (90th percentile 9.0°), and 3.0–5.4° in every 6 mm axial band from the brainstem to
  the cortex.
- Brainstem 4.5° against 4.6° elsewhere. There is no peak at the pons.

**Whether it predicts.** For A's CST streamlines, the spread along the path, against keeping the name on a rescan
(AUROC for a change):

| rescan | path mean spread | path 90th-percentile spread | 1 − vote share |
|---|---|---|---|
| same session | 0.59 | 0.59 | 0.68 |
| session 3 | 0.67 | 0.64 | 0.72 |
| session 5 | 0.66 | 0.66 | 0.74 |

Fitted on the same-session rescan and scored on session 5, the vote share alone gives AUROC 0.740, and vote share
plus spread gives 0.746. The two correlate at −0.41.

- **Why it fails: selection.** Statistics conditioned on the label see only the streamlines still named CST. The
  unstable ones veer into the cerebellum and are named Other (NOTES "Why CST streamlines become Other"), so they leave
  the class. What remains agrees in direction. Instability shows as streamlines leaving the bundle (presence, which
  the vote share carries), not as the bundle's direction wobbling.
- **Per-tract direction statistics are descriptive, not predictive.** They can still label crossings ("CST
  superior–inferior ±5°, runner-up SLF anterior–posterior") and drive glyphs. As a stability signal for the CST they
  do not earn a rank-field format extension.
- **A version that might predict** would compute direction over all streamlines from the tract's seeds whatever their
  final name, or look for two directions in one voxel. That overlaps the path-divergence measures already in hand.

Caveats: one tract, one subject.

## 2026-10-13 Grid-free presence: a density threshold keeps the calibration from 2 mm to 1 mm; the count rule does not

`ensemble_grid_check.py` → `ensemble_grid_check.json`.

**Method.**
- **Data:** cIIIs04's 68 runs (four scans × 17), focus tracts CST, AF and UF. Each run's streamlines are sampled every
  0.4 mm.
- **Grids:**
  - A's diffusion grid (2.1 × 2.1 × 2.2 mm);
  - a 1 mm grid in the subject's T1 space (FastSurfer's grid, cropped to the brain), reached through the T1-to-DWI
    transform.
- **Presence, two definitions:**
  - **density:** streamlines visiting a voxel over its mean cross-section (1.5 h²), smoothed by 1 mm, at least 0.3
    per mm²;
  - **count:** at least 2 streamlines per voxel, the earlier rule.
- **Test:** leave one scan out, as in "Pooling sessions". Coverage is averaged over the four held-out scans.

| coverage: widest / stated 0.5 / 0.8 / 0.9 / 0.95 | CST | AF | UF |
|---|---|---|---|
| 2 mm, density, single | 0.96 / 0.54 / 0.76 / 0.86 / 0.91 | 0.96 / 0.53 / 0.77 / 0.87 / 0.91 | 0.96 / 0.54 / 0.78 / 0.86 / 0.91 |
| 1 mm, density, single | 0.96 / 0.56 / 0.77 / 0.86 / 0.92 | 0.96 / 0.55 / 0.77 / 0.86 / 0.92 | 0.96 / 0.55 / 0.77 / 0.86 / 0.91 |
| 2 mm, density, pooled | 0.99 / 0.51 / 0.79 / 0.89 / 0.94 | 0.99 / 0.51 / 0.80 / 0.90 / 0.95 | 0.99 / 0.52 / 0.81 / 0.90 / 0.95 |
| 1 mm, density, pooled | 0.99 / 0.51 / 0.79 / 0.89 / 0.94 | 0.99 / 0.52 / 0.81 / 0.90 / 0.95 | 0.99 / 0.52 / 0.81 / 0.91 / 0.95 |
| 2 mm, count, single | 0.95 / 0.50 / 0.76 / 0.85 / 0.92 | 0.95 / 0.52 / 0.77 / 0.85 / 0.92 | 0.95 / 0.50 / 0.77 / 0.85 / 0.91 |
| 1 mm, count, single | 0.91 / 0.47 / 0.72 / 0.84 / 0.91 | 0.91 / 0.47 / 0.74 / 0.83 / 0.91 | 0.90 / 0.46 / 0.72 / 0.83 / 0.90 |

Tract volume (the held-out run's presence, mean):

| | CST | AF | UF |
|---|---|---|---|
| density, 2 mm / 1 mm | 56.6 / 52.9 mL | 55.0 / 50.9 mL | 39.2 / 36.2 mL |
| count, 2 mm / 1 mm | 69.2 / 19.3 mL | 65.4 / 21.3 mL | 47.8 / 13.4 mL |

- **Density makes presence grid-free.** Coverage at every stated level agrees within 1–2 points between 2 mm and 1
  mm, single and pooled, and the tract volume changes by about 7 %.
- **The count rule does not survive the grid change.** At 1 mm the volumes shrink to about a third, and single-scan
  coverage drops 3–5 points.
- **The pooled result holds on either grid:** stated 0.8 holds 0.79–0.81, and 0.9 holds 0.89–0.91.

Consequences:
- Store and report presence as density (streamlines per mm²) and compute fields on whichever grid a use needs; the
  calibrated levels carry over.
- The canonical space can be the T1's 1 mm grid, where the anatomy is.
- The rescans were still tracked on A's grid (resampled DWI). Tracking each session on its native grid and moving its
  streamlines into T1 space is the next step for clean pooling. It was not tested here.

## 2026-10-13 The ensemble as a duckn store: 18.7 MB for four scans' presence at 1 mm

`ensemble_store.py` (the new optional `store` group: zarr ≥ 3.3, and duckn at haversack's tag v0.6.2; Python 3.12+)
→ DATA/ensemble/cIIIs04/cIIIs04_ensemble.duckn.zip. The votes are cached beside the records (presence_t1.npz, 15 MB).

**Contents.** Per class (42 tracts and Other) and voxel of a 1 mm grid in the subject's T1 space (149 × 183 × 148,
FastSurfer's label grid cropped to the brain), with density presence (ensemble_grid_check.py):
- ensembles/<scan>/votes: each scan's 17-run ensemble (2.9 MB each);
- ensembles/<scan>/run: the scan's own run (0.45 MB each);
- pooled/votes: all 68 runs (5.0 MB).

**Layout.** It follows haversack's ranked stores, with the user's convention for anatomy: sharded with few blocks,
since a reader decompresses whole arrays.
- The store is zarr v3, built in a staging directory and packed into a zip with every entry stored uncompressed.
- Each array is (class, z, y, x): one chunk per class in one shard per array, blosc zstd 9 with bitshuffle.
- Geometry goes through duckn's own models: LPS mm, cell centering, a list axis for the class. Every group and array
  carries duckn metadata, and every node validates on read-back (DucknMetadata and DucknGroupMetadata).
- The `tractline` extension block (version 0.1) holds:
  - the class names and the presence definition;
  - per class and array, the vote counts where the envelopes of stated 0.5 / 0.8 / 0.9 / 0.925 / 0.95 start (CST in
    A's ensemble: 16 / 10 / 6 / 4 / 3 of 17 runs);
  - the coverage those levels held on held-out scans;
  - the T1-to-DWI transform and the DWI affine;
  - the FastSurfer store, by relative path, as sharing this T1 frame.

**Block layout** (one 17-run ensemble, 43 classes, 174 MB raw): the whole array decompresses in 0.04–0.08 s in every
layout tried.
- one block per class: 2.9 MB, written in 4.5 s at zstd 9, or 3.6 MB in 0.04 s at zstd 3;
- four blocks per class: 2.4 MB in 16 s;
- 64³ chunks with empty chunks skipped: 2.9 MB in 19 s.

**Revised the same day: FastSurfer's conformed grid is the canonical per-subject grid** (the user's suggestion).
- **What changed:** the store now uses FastSurfer's conformed lattice (256³, 1 mm, where its ranked anatomy is
  native), cropped to the brain. The crop is conformed (i, j, k) [54, 43, 27] to [204, 190, 211], 150 × 147 × 184
  voxels, about the same count as before, so the same cost (about 9 minutes for 68 runs).
- **The crop is recorded** in the `tractline` block, so FastSurfer's own arrays slice to the same voxels; the old
  route restored its labels onto the original T1 grid. The cache is now presence_conformed.npz, and the original-T1
  version is removed.
- **The store is 19.6 MB, and every node validates.**
- **Alignment check:** slicing FastSurfer's ranked store (parts/0/ranks, rank 1) with the recorded crop, the CST core
  (at least 9 of 17 runs) falls in cerebral white matter 46 %, brainstem 27 %, ventral DC 9 %, thalamus 5 % and
  outside the brain 4 %.

**Rank fields added to the store** (`rankfield/A`: A's 17 runs; `rankfield/pooled`: all 68).
- **Basis:** per voxel, the 43 classes ranked by mean streamline density over the runs, with "none" as a
  pseudo-class at the presence threshold (0.3 per mm²). Densities below a tenth of the threshold count as absent,
  which drops faint smoothing tails.
- **Encoding:** rankfield 0.4 (`rf.encode`, depth 6, keep shell), as FastSurfer's ranked anatomy is. Arrays: ranks,
  support and tail. The encoder's metadata is its own `ranked` extension beside `tractline`.
- **Check:** rank 1 is the densest class in every voxel (1.0 agreement, about 1.18 million voxels with a tract ranked
  first).
- **Size:** about 10 MB per rank field (ranks 3.3, support 5.4, tail 1.4 MB). The store is now 39 MB.
  - The size reflects real overlap. Above the floor, the median voxel holds 5 classes, the 90th percentile 10, and half
    hold more than 4.
  - Alternatives measured: no floor, 13.4 MB each (46 MB store); a floor of a third, 8.0 MB; depth 4, 7.7 MB (35 MB).
- **Cache:** presence_conformed.npz (votes, each scan's run, and the two mean-density fields in float16) is 102 MB.
