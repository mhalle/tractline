# A modular, accelerated state-of-the-art pipeline: design

Written 2026-10-24 on `ismrm2027` (not committed). It covers how tractline grows from one fixed path (field → UKF →
RapidParc) into a pipeline of interchangeable stages. Each stage gets an accelerated implementation validated against an
established tool. Nothing here changes today's default. Motion and eddy correction already have their own design
(`docs/gpu-motion-eddy-design.md` on `motion-eddy`), and this doc adopts its integration plan (its sections 5.1 and 5.2)
rather than repeating it.

## Summary

- **Stages are slots with interchangeable implementations.** Each implementation states the artifacts it reads and writes,
  its options, its devices, whether it is stochastic, and the tool it is validated against (its oracle). A run is a
  frozen `PipelineConfig`, one entry per slot. The config generalizes the motion-eddy design's section 5.2.
- **Presets are named configs, named by what they contain and versioned, never by status** (no "classic",
  "mainstream", "state of the art": those words go stale).
  - `ukf-rapidparc@1` is today's path, bit for bit. The alias `default` points to it, and only the owner moves the
    alias.
  - `msmt-ifod2-act-sift2-tractseg@1` follows today's MRtrix/QSIPrep recipe: MP-PCA → Gibbs removal → field →
    motion/eddy → bias correction → MSMT-CSD → iFOD2 + ACT → SIFT2 → TractSeg → tractometry.
  - `msmt-ptt-act-sift2-recobundles@1` builds on what step 6 measured as most accurate on the phantom: PTT + ACT,
    there on a single-shell CSD. The rest of the recipe (MSMT, SIFT2, RecoBundles) has not been tested with it.

  Any slot can be overridden.
- **The approach: oracle first, then accelerate.** Each new slot first gets an *oracle implementation*, a thin wrapper that
  runs the established tool in a subprocess. Then it gets our accelerated implementation, validated against that oracle
  at two levels:
  - the outputs agree;
  - the spread agrees: rescan agreement and the ensemble's spread are the same within tolerance.

  The wrapper stays available as the reference.
- **The config hash keys the calibration maps.** It covers every slot's implementation, its version and its options. It
  covers no runtime details. A calibration map applies only to runs with its hash and its protocol, as maps already
  refuse a mismatched protocol.
- **One GPU substrate serves most stages:**
  - a composed-warp resampler;
  - batched per-voxel fits;
  - streamline kernels that share an FOD sampler and a tissue sampler;
  - CNN inference in torch;
  - sparse streamline-by-voxel and streamline-by-fixel operations.

  Metal serves the M2, Triton serves CUDA, and plain torch is the CPU fallback and the float64 reference.
- **Effort.** The foundation takes 1–2 weeks. The minimum of the two new presets takes about 2.5–3.5 months. The full set
  takes 4–6 months. The order is set by what step 6 measured: tracking and tissue first.

## 1. What exists, as stages

| slot | today | module | where it runs |
|---|---|---|---|
| ingest | BIDS series, field-map pairing and refusals | `bids` | CPU |
| denoise | opt-in MP-PCA or Patch2Self (dipy, its own process) | `denoise` | CPU |
| field | topup's model, Gauss-Newton; apply with a cubic interpolation along PE and the Jacobian | `susceptibility` | GPU (estimate), CPU (apply) |
| motion_eddy | prototype on `motion-eddy` (Metal kernels, MI anchor) | `bench/motion_eddy/` there | GPU |
| mask | median_otsu, exactly as dipy's | `mask`, inside `prep` | GPU or CPU |
| model | the UKF's two-tensor model, internal to the tracker; one shell (`shell=`) | `prep`, `ukf` | — |
| track | UKF two-tensor, ORG settings | `ukf`, `ukf_metal`, `ukf_triton_block` | GPU |
| bundles | RapidParc (default), TractCloud | `labelers` | GPU |
| output | TRX | `trx` | CPU |
| QC | residual left by the field; T1 alignment check | `pipeline.residual_left`, `t1check` | CPU |
| uncertainty | the whole-pipeline wild-bootstrap ensemble, records, votes, envelopes | `ensemble`, `bench/_bootstrap.py`, `bench/calibration_*` | as the stages |

The bench already runs these outside the library: DIPY's CSD + PTT + ACT, tissue maps from FastSurfer or a T1, and
TractSeg in an isolated environment (`bench/_benchmark_dipy.py`, `bench/_tractseg_run.py`, `bench/robust_benchmark.py`).
These are the first oracle wrappers. They need to move behind the stage interface, not be written again.

**What step 6 measured (docs/robustness-plan.md).** On the ISMRM 2015 phantom, PTT + ACT with white-matter seeding
scores mean F1 0.583 against UKF's 0.412, with VS 48.9 % against 27.5 %. Seeding does not explain the gap. On rescans,
TractSeg's masks agree best (Dice 0.89–0.96 against UKF's 0.73–0.81), but its masks are about three times larger.
RapidParc on PTT's streamlines is off its training domain: the tract mix differs by a total variation of 0.149.
DIPY's PTT took about 190 s per scan on the CPU, against UKF's 50 s on the M2's GPU. These results set the priorities
in section 7.

## 2. Architecture

### 2.1 The scan state and its artifacts

A run carries one `Scan` state, a mapping from artifact names to values. Stages never mutate it. Each stage returns new
artifacts, and the runner merges them. The conventions are the ones `pipeline.py` states today, extended:

| artifact | shape, type | grid and convention |
|---|---|---|
| `dwi` | (X, Y, Z, G) float32 | the DWI's grid; as acquired until a correction replaces it |
| `bval`, `bvec` | (G,), (3, G) | FSL convention; `bvec` is rotated once motion is estimated |
| `affine`, `vox` | 4×4, (3,) | voxel → RAS mm; voxel sizes as the header states |
| `b0s`, `pe` | (X, Y, Z, V); PE vectors, readout times | as `pipeline.py`'s subject fields |
| `field_hz` | (X, Y, Z) | Hz on the DWI's grid |
| `motion`, `eddy` | (G, 6), (G, k) | per volume; the motion-eddy design's definitions |
| `noise_sigma` | (X, Y, Z) or None | from MP-PCA; used by later fits and by the bootstrap |
| `mask` | (X, Y, Z) bool | the brain mask on the DWI's grid |
| `t1`, `t1_affine` | (X', Y', Z') | the T1's own grid |
| `seg` | (X', Y', Z') int | the segmentation's labels (FreeSurfer LUT), on the T1's grid |
| `t1_to_dwi` | 4×4 (+ optional PE displacement) | world → world, rigid; `t1check`'s residual displacement optional |
| `tissue` | (X, Y, Z, 5) float32 | 5TT partial volumes on the DWI's grid: cortical GM, deep GM, WM, CSF, pathological (MRtrix's order) |
| `response` | per tissue, per shell | the response functions |
| `fod` | (X, Y, Z, n_sh) per tissue | real SH, MRtrix's basis; the WM FOD for tracking |
| `peaks`, `fixels` | (X, Y, Z, k, 3) amplitudes; fixel index | FOD lobes |
| `tensor` | (X, Y, Z, 6), FA, MD | DTI, for QC and tractometry |
| `tractogram` | list of (n, 3) RAS mm, seed order | plus `seed_voxel`, per-streamline stats |
| `weights` | (n,) float32 or None | SIFT2's cross-section multipliers |
| `labels` | `Labels` or None | per streamline: tract, log-probabilities |
| `bundle_masks` | (B, X, Y, Z) bool or float | TractSeg-style voxel masks, with the bundle names |
| `profiles` | (B, nodes, metrics) | tractometry |
| `records` | provenance, timings, warnings, QC per stage | see 2.4 |

New artifacts carry their grid explicitly: a `Volume` type of (array, affine) for anything not on the DWI's grid. This
avoids the silent-grid mistakes the bench guards against by hand today. Moving an artifact between grids is always an
explicit stage step that is recorded, never implicit.

### 2.2 The stage interface

```python
@dataclass(frozen=True)
class Implementation:
    slot: str                        # "denoise", "gibbs", "field", "motion_eddy", "bias", "mask", "anatomy",
                                     # "register_t1", "tissue", "model", "track", "filter", "bundles", "tractometry"
    name: str                        # "mppca", "mrtrix-dwidenoise", "ptt", "ifod2", ...
    version: str                     # bumped whenever its outputs change; part of the config hash
    requires: frozenset[str]         # artifacts read (an "a|b" alternative allowed)
    provides: frozenset[str]         # artifacts written
    options: Mapping[str, Option]    # allowed keys, types, ranges, defaults; unknown keys raise at config time
    devices: tuple[str, ...]         # ("mps", "cuda", "cpu")
    stochastic: bool                 # takes a seed from the run context
    scope: str                       # "replicate": rerun per ensemble replicate; "scan": once per scan (anatomy)
    oracle: str | None               # the implementation it is validated against ("mrtrix-ifod2"), or "external" itself
    status: str                      # "oracle" | "experimental" | "validated"
    run: Callable[[Scan, Mapping, RunContext], StageResult]
```

- **`RunContext`** holds the device, workers, `Timer`, scratch directory and external tool paths. It also holds the
  replicate's seed. Each stochastic stage gets its own substream: (replicate, slot) → a counter-based generator (Philox),
  so the same replicate always reproduces and adding a stage does not shift any other stage's random numbers.
- **`StageResult`** holds the new artifacts, plus a record: what ran, its time, any warnings, its QC numbers, and a short
  note line. This is today's `Correction.note` and `warnings`, made general.
- **Skip semantics.** "auto" means a stage runs only when its inputs exist, as the field correction does today without a
  reversed pair. The skip and its reason go into the record, never a silent no-op. A stage marked required fails the run
  before any work when its inputs cannot exist, for example ACT without a T1 or a tissue fallback.
- **Process isolation.** dipy's OpenMP runtime and torch's cannot share a process on macOS. An implementation that needs
  dipy, MRtrix, ANTs or FreeSurfer runs through one helper, `external.run(tool, inputs, outputs)`. The helper writes
  NIfTI or npz to scratch, runs the tool in its own environment with `python -I` or the binary, and reads the outputs
  back. This is today's `denoise` and `_tractseg_run` pattern, made common.

### 2.3 The config, presets and the hash

```python
cfg = PipelineConfig.preset("msmt-ifod2-act-sift2-tractseg@1").with_(track={"impl": "ptt"}, bundles="recobundles")
corr, scan = run(subject, cfg, device="mps")             # run(**kwargs) keeps working: it builds a config from `default`
```

- **Frozen and validated at construction.** It checks every implementation name, option key and value. It also checks
  that the DAG is satisfied: each `requires` is met by ingest or by an earlier slot's `provides`, and each alternative is
  resolved. All of this happens before any work, as `tracker=` is checked today.
- **Slot order is fixed.**

  ingest → denoise → gibbs → field → motion_eddy → apply → bias → mask → (anatomy → register_t1 → tissue) → model →
  track → filter → bundles → tractometry → output

  Denoising and Gibbs removal come first, on the data as acquired, before any interpolation, as both methods assume.
  `apply` is one resample per volume, composing the field, motion and eddy (the motion-eddy design's `apply_volumes`).
  It is not a user-facing slot.
- **Two hashes.**
  - **The config hash** is the SHA-256 of the canonical JSON of {slot: (impl, version, options)}. It covers what changes
    the outputs. Calibration maps key on it. The existing `pipeline settings sha256` in the calibration map becomes this
    hash.
  - **The runtime record** is kept apart: device, workers, batch sizes, library versions and the git state. It is
    recorded but not hashed. The Metal and Triton kernels are not bit-identical, and that difference belongs in a
    cross-device check, not in a key. Decision 6 below asks whether the device goes into the key instead.
- **Presets.**

  | slot | `ukf-rapidparc@1` (`default`) | `msmt-ifod2-act-sift2-tractseg@1` | `msmt-ptt-act-sift2-recobundles@1` |
  |---|---|---|---|
  | denoise | — | MP-PCA | MP-PCA |
  | gibbs | — | subvoxel shift | subvoxel shift |
  | field | auto | auto | auto |
  | motion_eddy | — | quadratic | quadratic |
  | bias | — | N4 | — |
  | mask | median_otsu | SynthStrip | SynthStrip |
  | anatomy / tissue | — | FastSurfer → 5TT (hsvs-like) | FastSurfer → 5TT |
  | model | UKF two-tensor, one shell | MSMT-CSD (dhollander responses) | MSMT-CSD (single-shell CSD fallback) |
  | track | UKF, ORG settings | iFOD2 + ACT, GM-WM interface seeding | PTT + ACT, WM seeding |
  | filter | — | SIFT2 | SIFT2 |
  | bundles | RapidParc | TractSeg | RecoBundles |
  | tractometry | — | 100-node profiles | 100-node profiles |

  - **The naming rule.** A name lists the parts that define the recipe: fiber model, tracker, constraint, filter and
    bundler. Preprocessing is listed in the preset file, not in the name. The defaults of the model and tracker
    (`ukf-rapidparc`: UKF's own two-tensor model) are left out.
  - **Presets are immutable.** Once released, a preset's contents never change. Any change, even an option's default
    or an implementation's version, is a new number (`@2`). Old numbers stay loadable, so a calibration map's preset
    can always be rebuilt. The config hash remains the identity; the name is a readable label, and maps record both.
  - **Why the names describe contents.** The recipe step 6 found best and today's MRtrix recipe will both be
    superseded. A name like `msmt-ifod2-act-sift2-tractseg@1` stays true when that happens; "mainstream" or "best"
    would not.
  - **Without a T1,** the presets fall back to `tissue: "fa-t1-classes"` (the bench's FA/T1 classifier) or to no ACT, and
    the record says which ran. The fallback is a different config, so it has its own hash.
  - There is no separate fast preset: `ukf-rapidparc@1` is already the fast one.
- **Calibration maps per preset.** A map is learned per (config hash, protocol). Changing any slot in a preset means a
  new map. That is the cost of options, and it belongs in the data product's statement: without a matching map, the
  product says "run-to-run only", as today.

### 2.4 Provenance

Each run writes its config (`to_dict`), the config hash, the runtime record, and every stage's record in order.
`Correction`, `Tractogram` and `Labels` stay as views on the `Scan` for compatibility. The ensemble's `runs.json` and
the bench's `pipeline_settings()` both store `cfg.to_dict()` and its hash, in place of today's hand-written dicts.

### 2.5 The ensemble over a modular pipeline

- **Replicate scope.** DWI-derived slots (`scope="replicate"`) rerun on every bootstrap replicate. T1-derived slots
  (`scope="scan"`: anatomy, register_t1 and the T1 side of tissue) run once per scan and are shared across replicates.
  A per-scan option resamples them too, for example to put registration jitter into the envelope. Today's rescans
  re-derive their masks, and an option keeps that behavior available.
- **Content-addressed cache.** Each stage's output is keyed by (input artifact hashes, implementation, version,
  options). A 17-run ensemble then pays for FastSurfer once.
- **Matching across runs.** Seed-voxel matching (`StreamlineEnsemble`) needs one deterministic streamline per seed
  voxel, which UKF has. Probabilistic trackers (iFOD2, PTT) have no seed-to-streamline identity, so their ensembles
  use the voxel summaries (`VoxelEnsemble`: densities, votes, envelopes), which work for any tractogram. Bundle masks
  vote per voxel directly. SIFT2 weights enter the density as weighted counts, an option that changes the hash.
- **Replicates with motion.** These follow the motion-eddy design (R0 today, R1 forward-model bootstrap with motion on).
- **Cost.** An ensemble costs R × the per-run time. The per-run target for every preset is ≤ 3 min on the M2, so a
  17-run ensemble finishes within an hour. This target, not single-run speed, is what makes the GPU work worth doing.

## 3. The GPU substrate

These are five shared pieces, so that each stage is mostly glue:

1. **Composed-warp resampler.** One sample per output voxel through a composed map (rigid ∘ field ∘ eddy, or T1 → DWI),
   trilinear or cubic B-spline, with an optional Jacobian. It serves `apply`, register_t1, tissue onto the DWI grid,
   the TractSeg MNI round trip, and the bootstrap's forward push. The field's cubic interpolation along PE is its 1-D
   case.
2. **Batched per-voxel fits.** Weighted least squares, constrained least squares and small eigensolvers over millions
   of voxels: DTI, SH, CSD's non-negativity, the MSMT block system, MP-PCA's per-patch eigendecomposition. In torch
   with batched `linalg`. Where MPS lacks an op (for example `eigh`), there is a Metal kernel or a measured CPU
   fallback.
3. **Streamline kernels.** The UKF's Metal and Triton kernels already set the pattern: one thread per half-fiber,
   blocks stepping until their longest fiber stops, float32 steps from float64 seeds. PTT and iFOD2 add:
   - a shared FOD sampler (trilinear SH interpolation, then amplitude on a direction set or along a probing arc);
   - a shared ACT sampler (5TT partial volumes, the stop, back-track and valid-end rules);
   - a counter-based RNG per streamline.
4. **CNN inference.** RapidParc, TractCloud, TractSeg, FastSurfer and SynthStrip are all torch or portable to it. They
   run in-process on MPS or CUDA where their dependencies allow, and in their own environment otherwise.
5. **Sparse streamline × voxel and streamline × fixel operations.** Segment-voxel intersection lengths, tangent-to-fixel
   assignment, and weighted sums. They serve SIFT2, density maps (the ensemble's presence), tractometry and TDI-based
   QC. The pieces are torch sparse COO/CSR, built once per tractogram.

Each kernel gets a float64 torch reference and a check script (as `metal_check.py` and `median_check.py` do today),
and a determinism test.

## 4. Per-slot plan

Effort is one engineer's weeks for the accelerated implementation, after its oracle wrapper. The oracle wrappers take
1–3 days each, most of them adapted from the bench.

| slot | options | oracle | our algorithm, acceleration | status | effort |
|---|---|---|---|---|---|
| denoise | `mppca` (radius), `patch2self` | MRtrix `dwidenoise`; dipy | today's dipy wrapper. GPU MP-PCA: patches unfolded as in `mask.py`, batched covariance and `eigh`, MP threshold. Exact equality to dipy's patch weighting is a check, not a goal. Keep the correlated-noise caveat (NOTES 2026-10-17). | wrapper exists | 1–2 |
| gibbs | `subvoxel-shift` (Kellner 2016) | MRtrix `mrdegibbs` | per-slice FFT, a set of subvoxel shifts, local minimal oscillation. `torch.fft` on any device. The record states the partial-Fourier caveat. | new | 0.5–1 |
| field | `tractline` (today), `topup` (external) | FSL topup | done. 6 of 8 scans agree with topup (calibration_topup_check); 32-direction series caveat | validated | — |
| motion_eddy | `rigid`, `linear`, `quadratic` | FSL eddy | the motion-eddy design; its sections 5.1–5.2 are this doc's integration plan for that slot | prototype | per its plan |
| bias | `n4` | ANTs `N4BiasFieldCorrection` (MRtrix `dwibiascorrect ants`) | N4's histogram sharpening plus a B-spline fit of the log-bias, on the mean b0, as a batched least-squares solve. Applied to all volumes. It matters for MSMT-CSD and intensity normalization, not for UKF (signal / b0) | new | 1–2 |
| mask | `median-otsu` (today), `synthstrip` | FreeSurfer `mri_synthstrip`; MRtrix `dwi2mask` | SynthStrip is a CNN; its released weights run in torch on MPS or CUDA. On the mean b0, or on the T1 carried over | wrapper to write | 0.5–1 |
| anatomy | `fastsurfer`, `synthseg` | FastSurfer, FreeSurfer | the networks themselves (torch, GPU). Labels only, no surfaces. The bench's `tissue_fs` uses FastSurfer already | bench | 1 |
| register_t1 | `rigid-ngf`, `bbr`, `scanner` | ANTs, FSL `epi_reg` (BBR) | rigid, with normalized gradient fields or MI, multi-resolution, on the resampler. `t1check`'s joint rigid + PE B-spline is an option. MI from the motion-eddy branch's `mi_torch` | partly in `t1check` | 1–2 |
| tissue | `5tt-seg` (from labels), `fa-t1-classes` (no labels) | MRtrix `5ttgen freesurfer` / `hsvs` | label → class tables and partial volumes by supersampling, as the bench's `tissue_fs`, on the resampler | bench | 0.5 |
| model | `ukf-internal` (today), `dti`, `csd` (single-shell), `msmt-csd`; responses `tournier`, `dhollander` | MRtrix `dwi2response`, `dwi2fod` | response selection on the CPU (heuristics over a few thousand voxels). FOD: per-voxel SH with non-negativity on a direction set. A batched active-set or projected-Newton solver on the GPU. MSMT as the stacked multi-tissue system | dipy CSD in the bench | 2–4 |
| track | `ukf` (today), `ptt`, `ifod2`, `sd-stream`; ACT on or off; seeding `wm`, `gmwmi`, `dynamic`, `ukf-seeds` | MRtrix `tckgen`; dipy PTT | the streamline kernels on the shared FOD and ACT samplers. iFOD2: candidate arcs and rejection sampling, per streamline. PTT: parallel transport frames, per streamline. dipy's GPUStreamlines (CUDA) is prior art to study (license and Metal portability to check) | UKF done; PTT bench (CPU, ~190 s) | 3–5 |
| filter | `sift2`, `none` | MRtrix `tcksift2` | the fixel map (FMLS lobes), segment → fixel assignment (sparse), per-streamline weights by regularized line search, iterated. Sparse matrix-vector products on the GPU | new | 3–4 |
| bundles | `rapidparc` (today), `tractcloud`, `tractseg`, `recobundles` | TractSeg; dipy RecoBundles; pyAFQ | RapidParc and TractCloud as today. TractSeg in-process in torch (weights are Apache-2.0), peaks to MNI via the resampler. RecoBundles: streamline-to-atlas registration (SLR) plus MDF pruning, batched distances on the GPU. Each labeler declares its training domain. A run off that domain gets a warning and the Other-share check | partly done | 1–2 |
| tractometry | `profiles` (nodes, weighting), metrics from `tensor` or FOD | pyAFQ | per-bundle core, Mahalanobis-weighted node means. Sparse operations; seconds | new | 1 |
| output | TRX (+ weights, labels, masks, profiles) | — | TRX gains per-streamline weights; NIfTI for masks and maps | partly done | 0.5 |

Status words: "validated" means it passed both validation levels (section 5). A slot's default never changes without
the owner's decision and a new calibration map.

## 5. Validation

Each accelerated implementation is checked against its oracle on a fixed panel, at two levels.

**Level 1, output agreement.** The tolerances are set per artifact before running:

| artifact | measure |
|---|---|
| denoised / degibbsed DWI | relative RMS difference in the brain; residual-noise σ |
| bias field | correlation and max relative difference in the brain |
| masks, segmentations | Dice; the boundary's 95th-percentile distance |
| registration | displacement at the brain edge, 99th percentile |
| responses, FODs | response relative error; FOD angular correlation; peak angular error and amplitude |
| tractograms | density-map correlation; bundle presence Dice per tract; length distribution (KS) |
| SIFT2 weights | correlation; weighted density against the FOD (fixel-wise) |
| bundles | Dice per bundle; per-streamline agreement where defined |
| profiles | node-wise difference relative to rescan variance |

**Level 2, spread agreement.** The same data goes through the oracle path and ours:
- across rescans (MASiVar cohort III, ON-Harmony), the rescan Dice and the per-tract presence agreement;
- across bootstrap replicates, the ensemble's spread.

They must agree within the rescan-to-rescan floor. An implementation that matches outputs but narrows the spread would
miscalibrate the maps, so this level is not optional.

**Downstream, per preset.** Each preset is scored on:
- the ISMRM 2015 phantom (F1, VS, IB, overreach, as step 6);
- the calibration gap and the calibrated coverage on cohort III, with its own map;
- the per-run time on the M2 and on CUDA.

**Panel.** ds001226 (PAT16 and the 12 patients), MASiVar cohorts II and III, ON-Harmony, the ISMRM 2015 phantom,
Stanford HARDI, and the SHORELine and DW-POSSUM simulations for motion. All are already on disk or fetched before.
MRtrix, ANTs, FastSurfer and SynthStrip are new installs (decision 4).

## 6. Compatibility with today's code

- `ukf-rapidparc@1` reproduces today's `run()` bit for bit. That is the foundation phase's acceptance test, on PAT16 and on
  cIIIs04's stored record (as `calibration_runs.py verify` checks today).
- `run(s, labeler, timer, ..., denoise=, tracker=, shell=, motion_eddy=)` keeps its signature. It translates its
  keywords into a config built from `default`. The bench and the ensemble keep working. New code passes `config=`.
- Today's modules become implementations, with no change to their numerics:

  | module | implementation |
  |---|---|
  | `susceptibility` | `field:tractline` |
  | `mask.median_otsu` | `mask:median-otsu` |
  | `prep` (shell selection) | `model:ukf-internal` option `shell` |
  | `ukf` | `track:ukf` |
  | `labelers.*` | `bundles:rapidparc`, `bundles:tractcloud` |
  | `denoise` | `denoise:mppca`, `denoise:patch2self` |

- The bench's oracle code (`_benchmark_dipy.py`, `_tractseg_run.py`, `tissue_fs`) moves into
  `tractline/external/` as oracle implementations. The bench then calls the library.

## 7. Order, effort and gates

The order is set by what was measured, not by today's common recipe. Tracking and tissue made the large phantom
difference, so they come first.

| phase | content | effort | gate |
|---|---|---|---|
| A. Foundation | `Implementation`, `PipelineConfig`, `Scan`, hashes, `external.run`, cache; today's modules as implementations; `ukf-rapidparc@1` and `default`; `run()` translation | 1–2 wk | `ukf-rapidparc@1` bit for bit on PAT16 and cIIIs04 |
| B. Anatomy | T1 ingest; FastSurfer (oracle, then in-process); 5TT; register_t1 on the resampler; SynthStrip | 2–3 wk | Dice and edge displacement vs ANTs/FreeSurfer; `tissue_fs` reproduced |
| C. Fiber model | DTI; CSD and MSMT-CSD on the GPU; dhollander responses | 2–4 wk | level 1 vs MRtrix on the panel |
| D. Tracking | FOD and ACT samplers; GPU PTT (oracle: dipy's, already run); then iFOD2 (oracle: MRtrix) | 3–5 wk | level 1 and level 2; phantom score ≥ the oracle's |
| E. Bundles | TractSeg in-process; RecoBundles; labeler domain checks | 1–2 wk | Dice vs the oracle; Other-share check |
| F. Preprocessing extras | Gibbs; GPU MP-PCA; N4 | 2–3 wk | level 1 and level 2 |
| G. SIFT2 and tractometry | fixels, SIFT2, profiles, weighted TRX | 3–5 wk | weights vs `tcksift2`; profiles vs pyAFQ |
| H. Presets calibrated | an ensemble and map per preset on cohort III (8 subjects × 17 runs each) | compute: about a day per preset at ≤ 3 min per run | the gap and coverage per preset |

- **Phase A, as built (2026-10-24, branch `modular-pipeline`; NOTES entry of that date).**
  - `tractline.config` (`Option`, `Implementation`, registry, `PipelineConfig`, presets) and `pipeline.execute` / `Run`.
  - Six slots: denoise, field, mask, model, track, bundles.
  - The mask was split out of `prep.prepare` with no change in output.
  - `ukf-rapidparc@1` is bit for bit on PAT16 and matches cIIIs04's stored record.
  - Deferred to phase B, where they are first needed: the `Scan` artifact state (requirements are between slots for
    now), `external.run`, and the cache.
  - Finished the same day: the calibration runners run through `pipeline.execute` (`calibration_runs.run_scan`) and
    record `cfg.to_dict()` in place of the hand-written settings. New cards (schema 3) also match on the config hash.
    A cross-device test (the devices run the same pipeline) and two real-scan checks. Per streamline the devices differ
    by about a fifth of the ensemble's spread; at the ensemble level, where maps act, they do not (decision 6).
- **Totals.** A–E (both new presets, without SIFT2) take about 2.5–3.5 months. The full set takes 4–6 months.
  Motion/eddy runs on its own track (its merge plan, after 28 Oct) and slots in after A.
- **Parallel work.** Phases B, C and E touch separate modules and can run as separate sessions once A has landed. D
  needs C's FODs and B's tissue maps.

### 7.1 Phase C plan: single-shell CSD (2026-10-24)

**What the measurements say first.** In the step 6 benchmark, DIPY's CSD fit plus peaks took 6.4 s on cIIIs04 (CPU,
parallel), and its PTT tracking took 193.5 s. The fit is not the cost. Phase C's value is a torch implementation inside
the pipeline: no DIPY subprocess (the OpenMP clash), FODs on the GPU where the tracker reads them, and one code path for
every ensemble replicate. GPU speed is a secondary goal. This shortens phase C to about 1–1.5 weeks. The time goes to
phase D, where the 190 s is.

**The oracle.** DIPY 1.12's `ConstrainedSphericalDeconvModel` and `auto_response_ssst`, with the bench's settings
(`_benchmark_dipy.py csd`): response from FA > 0.7 within 10 voxels of the center; `sh_order_max` 8 (6 below 45
directions); lambda 1, tau 0.1; `small_sphere` for the constraint; peaks on `default_sphere` (3 peaks, relative
threshold 0.1, separation 25°). DIPY's docstring states the algorithm (Tournier et al. 2007, as DIPY implements it).
We write our own from that description and credit both, as with topup.

**The algorithm, batched.** Per voxel:
- z = Xᵀs, then an initial f from P = XᵀX (with μI if needed);
- a threshold tau × the mean amplitude (from the order-0 coefficient);
- the negative set taken from the order-4 part first;
- iterate: Q = P + B_Hᵀ B_H over the negative directions, Cholesky solve, recompute the set, stop when it repeats
  (at most 50 iterations).

Batched, the negative set is a 0/1 weight per constraint direction: Q = P + Bᵀ diag(w) B for all voxels of a chunk at
once (einsum), then a batched Cholesky. Voxels that have converged are frozen. The work is a chunk × directions × n_sh²
product per iteration. It runs in float64 on the CPU and CUDA. MPS has no float64, so there it runs in float32, with the
difference measured against float64. Threshold comparisons can flip on borderline voxels, so agreement is to a
tolerance, not bit for bit.

**Pieces.**
1. `tractline.sh`: the real SH basis and the order-zero mean. It works in DIPY's descoteaux07 convention (what the PTT
   oracle reads). The stored FOD's basis is chosen and documented here, with conversions to MRtrix's basis tested.
2. The response: our own small DTI fit (weighted least squares) over the ROI, the FA threshold, and the mean
   eigenvalues and S0, recorded.
3. The batched CSD fit and FOD sampling on a sphere.
4. Peaks: the FOD on `default_sphere`, local maxima over the sphere's edges, the relative threshold and the 25°
   suppression, batched.
5. The model slot's `csd` implementation (options: `sh_order` auto/6/8, `lambda`, `tau`, the response's `fa_threshold`
   and `roi_radius`). The `track:ukf` implementation keeps requiring `model:ukf-two-tensor`, so `csd` first becomes
   usable with `track:ptt` (phase D). Until then it is called directly and checked by a bench script.

**Level 1 checks against DIPY**, on cIIIs04 A, PAT16, the phantom and Stanford HARDI:
- the response (eigenvalues and S0, relative difference);
- per voxel: the FOD's SH correlation and amplitude difference, the iteration counts, the share of voxels with the
  same negative set at the end;
- peaks: the count agreement, the angular error of matched peaks (median and 95th percentile), the amplitude ratio;
- time against DIPY's 6.4 s.

Level 2 waits for phase D: it needs the tracking on top.

**Status (2026-10-24, uncommitted at first).** `tractline.sh`, `tractline.csd` and `bench/csd_check.py` are done.
- **Our fit reproduces DIPY's exactly.** On cIIIs04 and PAT16, with DIPY's constraint sphere, our FODs equal DIPY's
  (correlation 1.0, L2 difference 0.0 to five decimals), and the response agrees to about 1e-8.
- **Our own sphere** is a 181-axis hemisphere made by repulsion. It differs from DIPY's (relative L2 difference 0.22–0.26
  at the 99th percentile in fiber voxels; 84–88 % of first peaks within 5 degrees). DIPY's own sphere, rotated at random,
  differs from itself as much. The constraint samples the sphere, and where it samples moves the FOD: a property of the
  method, worth knowing for the uncertainty work.
- **MPS (float32) against CPU (float64):** equal to five decimals.
- **Time, idle M2:**
  - cIIIs04: ours 4.7 s on MPS, 7.6 s on the CPU; DIPY 6.5 s for the fit and peaks in 8 processes;
  - PAT16: 3.6 s, 5.7 s and 4.9 s.
  So there is no real speedup yet, as planned: the gain is in-process FODs.
- **On MPS,** `cholesky_solve` and `solve_triangular` are about 50 times slower than an LU solve, so MPS solves by LU.
- **After the review:** the first step is float64 on the host on every device, and a near-singular X^T X (condition
  number above 1e8, e.g. fewer directions than coefficients) runs on the CPU in float64. The docstring states the
  provenance (DIPY's code read to match its behavior; our own batched implementation).

### 7.2 Phase D plan: GPU PTT with ACT (2026-10-24)

**The oracle,** as step 6 ran it (`_benchmark_dipy.py csd --track`): DIPY 1.12's `ptt_tracking` on the CSD's SH.
- **Settings:** step 0.5 mm; max angle 10°; probe length 1.5 mm; probe quality 7; probe count 1; PMF threshold 0.1;
  `default_sphere`; 20–250 mm; valid streamlines only.
- **ACT:** include = GM + background + the field of view's faces; exclude = CSF.
- **Seeds:** uniform in WM ≥ 0.5; seeds and tracking from fixed generators.
- **Cost:** 193.5 s for cIIIs04's tracking on 8 CPU processes.

The method, read in DIPY's code (`propspeed.pyx`, `tractogen.pyx`, `pmf.pyx`, `stopping_criterion.pyx`, the
interpolation) to understand it, and from Aydogan & Shi 2021. Our own implementation follows, crediting both.
- **State per half-streamline:** position (voxel coordinates); a frame (T, N1, N2); curvatures (k1, k2) in the normal
  plane; the last accepted likelihood.
- **Propagation:** a second-order parallel-transport step. Position += s T + s²/2 (k1 N1 + k2 N2), in voxel units through
  1/voxel size. The tangent and N2 are updated, then the frame is re-orthonormalized. Curvatures below 1e-4 are raised
  to it unless both are; then the step is straight.
- **Candidates:** (k1, k2) uniform in a disk of radius 1 / r_min, where r_min = (step / mean voxel size) / 2 /
  sin(max angle / 2).
- **The probe:** quality - 1 sub-steps of probe length / (quality - 1) along the candidate's arc. The PMF is read along the
  local tangent at each sub-step, and values at or below the threshold count as 0. The likelihood is (1 / (quality ×
  count)) × (the last value + the probe's sum), to the data-support exponent.
- **Rejection sampling:** 10 candidates estimate the maximum, which is then doubled. Up to 100 tries follow, and the first
  with u × max < L is accepted. If none is accepted, the half stops.
- **Initialization:** the same, with a random N2 perpendicular to the seed direction (N1 = N2 × T). The seed direction is the PMF's
  largest peak at the seed.
- **Bidirectional:** the backward half starts from the seed along the negated first forward segment, with fresh state
  and the same random stream continued.
- **The PMF value** along a direction: the SH, trilinearly interpolated, evaluated at the sphere vertex nearest the
  direction (|cos|), not at the direction itself. Negative values are not clipped; the threshold removes them.
- **Trilinear conventions** (PMF and ACT): voxel centers at integers; valid within [-0.5, n - 0.5), with the edge voxel
  repeated; outside is OUTSIDEIMAGE.
- **ACT:** include > 0.5 gives ENDPOINT (tested first); exclude > 0.5 gives INVALIDPOINT; outside the image is a valid
  end. A streamline is valid only when both halves end validly. The stopping point is not kept.
- **Random numbers:** DIPY seeds each streamline from its seed's coordinates. Ours: a counter-based hash of (seed index,
  half, step, draw), so a streamline's path does not depend on batch, device order or thread count.

**Our implementation.**
1. **The FOD sampler** (`tractline.fields`): the PMF on a vertex set, per masked voxel, computed once
   (B @ coeffs). The vertex set is our own hemisphere(362) with antipodes, or DIPY's for the oracle check. A lookup is a
   gather of 8 corners × the nearest vertex, plus trilinear weights. An option evaluates the SH exactly along the
   direction, with no vertex quantization: not DIPY's behavior, so off by default for the oracle preset.
2. **The ACT sampler:** include and exclude maps (float32, on the device) with the same trilinear conventions; the
   tissue maps from the bench's `tissue_fs` / `tissue_t1` (FastSurfer labels or the T1 classes) until phase B makes them
   in the library.
3. **The tracker,** first as a vectorized torch prototype. All half-streamlines of a batch step in lockstep, and finished
   ones are masked. Per step, the 10 estimating candidates and a first block of 10 tries are scored at once (a probe is
   6 sub-steps × one lookup); the rare halves still unaccepted try again. Then a Metal kernel, and a Triton one, as the
   UKF has: one thread per half-streamline, the same samplers, the same counter-based random numbers.
4. **Frames:** the b-vectors in image axes with FSL's x flip for a positive determinant, as `prep.py` and the bench
   apply it. Directions are in the voxel axes' mm frame, steps through 1/voxel size, as DIPY's.
5. **The slot:** `track:ptt` (options as above) requires `model:csd` and a `tissue` input. It needs the `Scan` artifact
   state or a narrow stand-in, which comes due here.

**Checks.**
- **Level 1, exact pieces:** the propagator against the formulas (float64); the likelihood of a given candidate; the
  ACT decision at given points; the trilinear conventions at the edges; the RNG's independence of batching.
- **Level 1, the tracker as a whole:** a probabilistic tracker cannot match DIPY's paths, since the random streams
  differ. It matches distributions instead: on the phantom, F1, VS, IB and overreach against step 6's DIPY run (0.583,
  48.9 %, 338); on cIIIs04, the density map's correlation, length histograms, the share of valid streamlines, RapidParc's
  tract mix, and each against DIPY run twice with different seeds (DIPY's own spread is the tolerance).
- **Level 2:** the rescan agreement of step 6 (CST, AF, UF), and then an ensemble with its own calibration map.
- **Target time:** under 30 s for cIIIs04's seeds on the M2 (DIPY: 193.5 s), so that a 17-run ensemble with PTT
  stays under about 15 min of tracking.

**Status (2026-10-24).**
- `tractline.fields` (the samplers) equals DIPY's own functions.
- `tractline.ptt` (the torch prototype) tracks a synthetic band on the CPU and MPS.
- Reading DIPY's PTT found three unit inconsistencies (NOTES 2026-10-24): a probe voxel_size² too long, a curvature bound
  v̄ times too high, and points shifted one step along the half's starting direction.
- **Decided 2026-10-24 (owner):** the tracker works in real units, its only behavior. Reproducing DIPY's units is a
  private validation switch (`Tracker(..., _dipy_compat=True)`), used by the tests and the oracle comparisons and never
  by a config, preset or calibration map.
- **Still to run:** the phantom in both. Step 6's PTT advantage was measured with DIPY's units; how much of it survives
  in real units decides whether PTT earns its preset.

## 8. Licensing and the source policy

The policy already set for FSL applies to every tool here:
- read the source to understand the method;
- write our own implementation;
- credit the idea in docstrings;
- never copy, and never translate mechanically.

| tool | license | use here |
|---|---|---|
| MRtrix3 | MPL-2.0 | oracle (external); methods read and credited |
| DIPY, incl. RecoBundles, PTT | BSD-3 | oracle; today's denoise wrapper |
| TractSeg | Apache-2.0 | oracle; weights in-process |
| FastSurfer | Apache-2.0 | oracle; networks in-process |
| ANTs | Apache-2.0 | oracle (external) |
| SynthStrip, SynthSeg | to check (FreeSurfer's license vs the repositories' own) | oracle; weights in-process if permitted |
| FSL (topup, eddy, XTRACT) | FSL license (non-commercial) | external oracle only |
| pyAFQ | BSD-2 | oracle |
| GPUStreamlines (dipy) | to check | prior art for the GPU trackers |

Running a network's released weights in-process is using the tool, not reimplementing it. It is allowed where the
weights' license permits, and recorded as such in the implementation's record.

## 9. Decisions for the owner

1. **When.** *Decided 2026-10-24:* phase A started at once, on its own branch (`modular-pipeline`, from `ismrm2027`),
   rather than after the deadline. Merging waits for the `ismrm2027` and `motion-eddy` merges.
2. **`PipelineConfig` with motion/eddy.** This answers the motion-eddy design's open decision. Recommended: introduce
   the config in phase A, and wire `motion_eddy` as its first new slot.
3. **First tracker.** *Decided 2026-10-24:* GPU PTT + ACT first (measured, oracle in hand), then iFOD2. The order is
   C (single-shell CSD, DIPY as oracle) → D (GPU PTT + ACT on the bench's tissue maps) → B (anatomy, with MRtrix and
   ANTs as oracles) → MSMT-CSD and iFOD2. Nothing before B needs a download.
4. **Installs and downloads.** MRtrix3, ANTs, FastSurfer weights, SynthStrip weights, and GPUStreamlines to read. Each
   needs your approval before downloading.
5. **Labelers off their domain.** Options: retrain RapidParc on PTT and iFOD2 tractograms, or rely on
   tractogram-agnostic bundling (RecoBundles, TractSeg) for those presets. Recommended: agnostic bundling first, with
   retraining as a later study.
6. **The hash's scope.** *Decided 2026-10-24:* the device stays out of the key.
   - **Per streamline, the devices differ.** On cIIIs04, a single run on the CPU against MPS gives 93.8 % the same tract
     and 31 % of paths more than 1 mm apart. That is about a fifth of a bootstrap replicate's difference.
   - **At the ensemble level, where maps act, they do not differ.** cIIIs04's whole 17-run ensemble was run again on the
     CPU (`bench/device_ensemble.py`) and judged against the same rescans with the cohort III map (fitted without
     cIIIs04). Calibrated coverage is within 0.003 of the MPS ensemble's at every target and for both terms, about a
     tenth of the spread between subjects. The shapes' coverage agrees to three decimals and their size within 0.003. The
     point statement's calibrated ECE differs more (0.023 against 0.018, one rescan).
   - **Limits.** One subject; the CPU against MPS; CUDA not run.

7. **Calibration budget.** Which presets get maps (about a day of compute each on the M2, more with motion/eddy on).
   Recommended: `ukf-rapidparc@1` (existing) and `msmt-ptt-act-sift2-recobundles@1` first.
8. **Preset naming.** Content names with an immutable version (section 2.3). Alternatives: dated recipes
   (`recipe-2026.10`), which age honestly but say nothing about contents, or short opaque codes with a lookup table.
