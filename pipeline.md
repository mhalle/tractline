# An optimized pipeline to the confidence-enhanced format

## 0. The pipeline as built (2026-10-02)

`_pipeline.py`, scan to payload, in memory, on the M2 laptop (16 GB). Measured on ds001226 PAT16
(`results/cohort/PAT16.json`; the 12-patient cohort in `results/cohort_summary.md`):

| stage | module | where | PAT16 |
|---|---|---|---|
| susceptibility field from the b0s + reversed pair (topup's model, Gauss-Newton) | `_susc.estimate` | GPU, float32; subsampled levels on the CPU | 17.5 s |
| apply it (cubic along phase encoding, Jacobian) | `_susc.apply` | CPU, float64 | 1.3 s |
| tracker input (b = 2800 shell, RAS gradients, median_otsu mask) | `_prep.prepare`, `_median` | CPU | 0.7 s |
| UKF two-tensor, ORG settings, the binary's seeds | `_ukf_torch.track` (Metal) | GPU, float32 steps | 27.0 s |
| TractCloud, one draw | `_tractcloud.Labeler` | GPU, float32 | 4.1 s |
| rank field + geometry (0.05 mm grid) | rankfield, `_geometry` | CPU | 0.9 s |
| **scan → payload** | | | **54 s; 3.0 MB for 32 k streamlines** |

For scale: FSL topup + applytopup take 665 s for the correction alone on the same machine; the whole
Stanford HARDI brain tracks in 135 s on the M2 and 78 s on an A10G (`_ukf_triton_block`). Against
the T1, the correction cuts the tumor margin's misplacement from 4.8 mm to 1.6 mm (99th percentile,
median of 12 patients). On the CPU alone the field takes 22.6 s, the same field as the GPU's (0.03 mm
at the brain's 99th percentile). Sections 1-5 below are the format work on the HCP tractogram that preceded it.

*2026-10-01. Measured on the HCP test subject (440,621 streamlines, 21.6 M vertices): an A10G on
Modal for the GPU, the M2 laptop and Modal CPUs for the rest. Every number below cites the
results file it came from. "Projected" marks a sum of measured stages that was not run end to end.*

## 1. What is possible

| stage | as run (upstream code) | optimized, measured | how |
|---|---|---|---|
| DWI → tractogram (UKF two-tensor) | Slicer binary (CPU) | **135 s** whole HARDI brain on the M2 (Metal), 78 s on an A10G (Triton) | `_ukf_torch` + kernels, float32 steps below the scan's noise (§0) |
| read the tractogram | 3.6 s (VTP) | ~0 s (TRX, memory-mapped) | `convert.json` |
| 15-point features | 2.4 s (CPU) | small on the GPU (not timed) | arc-length resampling, vectorized |
| context (kNN in 10 k chunks + 80 global) | 2.8-3.4 s (CPU) | **0.39 s** (GPU, same random draws; 99.89 % identical context) | `infer_opt.json` |
| TractCloud inference | 28.0 s (fp32, cudnn **off**, inputs copied per batch) | **3.28 s** exact-equivalent (fp32 + cudnn + `torch.compile`; 5 tract labels differ, all at margin < 0.5) · **1.83 s** fp16 (1.0 % of tract labels differ) | `infer_opt.json`, `infer_opt_exact.json` |
| field encode (rankfield) | 5.0 s (v0.3.10, 8 CPU) | **0.45 s** on the GPU, from the logits in place | `timing_gpu.json` |
| geometry encode | — | 0.13 s (numba + zstd 5) | `convert.json` |
| field + tables compress | — | 0.3 s (zstd 9) | `convert.json` |
| **tractogram → payload, one member** | **about 40 s** | **about 5 s projected** (fp32), about 3.5 s (fp16) | sums of the rows |
| **five members (the ensemble)** | about 200 s | **about 20 s projected** | members share features and kNN distances |

The one change that matters most is a setting: **upstream disables cuDNN on CUDA**
(`TractCloudPipeline.run_on_polydata`). Turning it back on is 3.4× (28.0 → 7.65 s, 2 cluster
labels differ). `torch.compile` doubles that again in exact fp32. Moving the context to the GPU
saves the per-batch host-to-device copy (about 8 GB per run). fp16 halves the rest but changes
labels; see §3.

## 2. The stage layout

```
tractogram (TRX, mmap) ─┬─▶ geometry encode (CPU, numba)  ──────────────────────────────┐
                        └─▶ features ─▶ context ─▶ inference ×R ─▶ field encode ×R      │
                            (GPU-resident from here; only 19-byte planes leave)          ▼
                                                                    store: geometry + field[R] + provenance
```

- **One GPU residency.** Features, context, inference and the field encode stay on the device.
  1.4 GB of fp16 logits per member never leave it; 8.4 MB of planes do.
- **Geometry runs beside it, on the CPU,** from the same memory-mapped arrays.
- **Members share their work.** All R runs use the same features, and the full kNN distance
  computation only needs the per-chunk random subsets redrawn. Only the draws and the forward
  pass repeat.

## 3. What to defer, and what cannot be

The rule: store what lets a decision be made later. Bake in only what cannot be undone, and then
record it (§4).

**Deferred by the format already:**
- the hard label, the grouping (cluster, tract, category, custom merges), the outlier threshold
  and the confidence threshold (M1, M1b);
- the display subset and the level of detail (chunks, `order`);
- geometry precision, up to the declared grid.

**Worth deferring that the current pipeline bakes in:**
- **The ensemble.** Store R members on a leading axis (5 × 5.5 MB) instead of one run. Instability,
  vote counts and mean margins become read-time computations, and "which run" stops being a
  decision.
- **The 40 mm length filter.** `wm_preprocess_all.py -l 40` deletes short fibers before anything
  else sees them. Keep every streamline with its length as per-streamline data, mark those under
  40 mm unclassified (TractCloud was trained on ≥ 40 mm), and make the cut a view filter.
- **Per-vertex UKF measures** (FA, tensors). `_pp` drops them by default. They are per-vertex data
  under the same predictive coding, an opt-in layer, not a deletion.
- **Field depth.** 6 is the measured trade-off; 8 halves the mass-margin cost for 3.3 B more per
  streamline compressed. Record which.

**Not deferrable, and therefore recorded:**
- **the tractography** (parameters, binary, seeds);
- **the model** (weights);
- **the context draws** (seed, RNG, draw order);
- **the numerics.** fp32 + cudnn + compile changes 5 tract labels in 440 k, all near ties. fp16
  changes 4,395 (1.0 %), 84 % of them at margin < 0.5 (6.4 % of all streamlines are), where the
  field already marks doubt. That is 28× smaller than run-to-run instability (27.9 %), but it is a
  baked decision. **Default to exact fp32** (3.28 s); allow fp16 only with the mode in the record.

## 4. Provenance: the same methods as rankfield and haversack

The pattern already used: rankfield's `meta` block declares everything a reader needs and refuses
what it does not know; seeds are explicit; the device is recorded because it changes bytes
(`format.md`, "Encoding is reproducible across devices, except the tail's last unit"); versions are
pinned by tag; the incubation records upstream commits. For the pipeline, one record per store,
JSON, beside the arrays:

```json
{
 "pipeline": {"name": "tractcloud-rankfield", "version": "…", "commit": "…"},
 "inputs": {
  "dwi": {"sha256": "…", "bvals_sha256": "…", "bvecs_sha256": "…"},
  "mask": {"sha256": "…", "method": "dipy median_otsu (computed by _median.py, identical)", "params": {"median_radius": 4, "numpass": 4}}
 },
 "correction": {"model": "FSL topup's (b02b0.cnf schedule), reimplemented (_susc.py)", "interp": "trilinear estimate, cubic application",
                "motion_convention": "first volume of each acquisition: no phase-encoding translation", "device": "mps", "dtype": "float32",
                "inputs": {"b0_ap": 6, "b0_pa": 2, "readout_s": 0.0266}, "seconds": "…"},
 "tractography": {
  "tool": "UKFTractography", "source": "github.com/pnlbwh/ukftractography", "commit": "2d2b661",
  "binary": {"package": "34627-macosx-amd64-UKFTractography-git2d2b661-2025-06-02.tar.gz",
             "sha512": "a2d0ebdd09cdfb87…", "host": "Slicer 5.12.3 (34627)", "arch": "x86_64 via Rosetta"},
  "parameters": {"numTensor": 2, "seedingThreshold": 0.1, "stoppingFA": 0.08, "stoppingThreshold": 0.06,
                 "recordLength": 1.8, "seedsPerVoxel": "…", "…every flag, defaults resolved…": null},
  "nondefault": ["seedingThreshold", "stoppingFA", "stoppingThreshold", "recordLength"],
  "seconds": "…"
 },
 "classifier": {
  "tool": "TractCloud", "commit": "94de627", "weights_sha256": "…", "classes": 1600,
  "context": {"k": 20, "k_global": 80, "k_ds_rate": 0.1, "chunk": 10000, "rng": "numpy legacy MT19937",
              "draw_order": ["global randint", "per-chunk choice"]},
  "numerics": {"device": "NVIDIA A10G", "torch": "2.14.1+cu130", "dtype": "float32", "cudnn": true,
               "tf32": false, "compiled": true, "batch": 4096},
  "members": [{"seed": 0}, {"seed": 1}, {"seed": 2}, {"seed": 3}, {"seed": 4}]
 },
 "field": {"rankfield": "<the meta block, verbatim: version, keep, depth, clip, curve, tail_temperatures, …>",
           "encoded_on": "cuda"},
 "geometry": {"quantum_mm": 0.0409, "origin": ["…"], "predictor": "second-order", "codec": "blosc zstd 5 bitshuffle",
              "max_error_mm": 0.0205, "sha256_of_grid_positions": "…"},
 "binding": {"nb_streamlines": 440621, "order": "tractogram", "geometry_sha256": "…"},
 "deferred": ["labels", "grouping", "outlier threshold", "confidence threshold", "length filter", "ensemble aggregation"]
}
```

The rules carried over:
- **Declare every parameter, with defaults resolved.** "Default" changes between versions; UKF's
  `recordLength` default (0.9) is not what this tractogram used (1.8).
- **Hash binaries, not just versions.** The Slicer extension server publishes SHA-512s, so a
  prebuilt UKF can be named exactly.
- **Record the device and the numerics whenever they can change bytes**: GPU vs CPU tails, cudnn
  on or off, fp16.
- **Bind derived data to its source by content hash** (the field to the geometry, the geometry to
  the tractogram), so nothing is silently applied to the wrong thing.
- **List what was deferred**, so a reader knows which decisions are still theirs.

## 5. Open

- **The field estimate's repeatability.** Gauss-Newton fixed the optimizer's sensitivity (negligible
  input noise: 0.07 mm at the margin's 99th, was 0.33 under L-BFGS) and predicts held-out b0s better on
  all 12 patients; fits from independent pairs of b0s still differ by ~1.3 mm at the brain's edge
  (99th), and stronger regularization does not help (NOTES 2026-10-02). Motion between slice groups
  (PAT08) is outside the model.
- **PAT23's frontal base.** Under a 104 cm³ meningioma, topup and our correction agree with each
  other and both disagree with the T1 by 4-8 mm in 4 % of the margin; which is wrong is open.
- **Upstream.** Enabling cudnn in TractCloud's own pipeline is a one-line change worth proposing
  to its authors: 3.4×, 2 labels in 440 k.
