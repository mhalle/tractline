# TractCloud × rankfield: the tractography experiment

*Plan for a self-contained demo: an offline Python pipeline plus a PicoGL browser viewer. No 3D Slicer, no SlicerLive. Renderer design is left to the implementer; this document specifies what the viewer must show and what data it receives.*

*Written 2026-10-01. Incubated in medseg `bench/tractography/` on branch `tractography-incubation` (never merged into main, as with `vessels-incubation`); data and captured fields stay outside the repo (`_data.py`: `~/tmp/data/tractography`). Revision 2. The rankfield facts below were checked against rankfield v0.3.10 (format 0.4). The TractCloud facts were checked against TractCloud `94de627` (main, 2026-10-01) during M0, with three corrections.*

## The claim

TractCloud labels each streamline by taking an argmax over 1600 classes and throws the rest of the field away. If we keep that field as a rankfield, we can:

1. **Store it cheaply.** About 19 bytes per streamline instead of about 3.2 KB. Be modest here: a naive top-6 (fp16 values plus uint16 indices) is about 24 B. The case for rankfield is its decode semantics and tooling more than the byte count.
2. **See where tract assignment is ambiguous.** Use group margins instead of hard labels.
3. **Predict instability from a single run.** Streamlines whose tract label changes across repeated inference runs should be the ones with a low tract-level margin *in the first run alone*. That margin should also predict changes better than a one-byte cluster-confidence score. This is the result that justifies the whole idea, and it costs almost nothing to measure.
4. **Defer decisions** that are now fixed at inference: hard assignment, grouping granularity (cluster → tract → category), and the outlier threshold. The viewer makes these live controls.

Claim 3 is the headline. If it fails, we learn that early and cheaply (milestone M1). It can fail in two ways, and each should be reported as a result:

- The margin does not predict instability at all.
- It predicts instability, but no better than the one-byte baseline. Then storing the field is not justified.

## Background

**TractCloud** (`github.com/SlicerDMRI/TractCloud`, `src/tractcloud/`; checked at `94de627`):

- Each streamline is resampled to 15 equally spaced points in RAS (`extract_ras_features`), then shifted so the subject's mean matches the HCP atlas center (`center_tractography`). There is no registration.
- Context for each streamline: its k = 20 nearest streamlines, plus k_global = 80 streamlines drawn at random from the whole brain. **Neither random draw is seeded**, so labels vary between runs. Both draws come from numpy's global stream: the global sample first, then the kNN draws.
  - *Correction:* the kNN search is NOT over a 10 % subsample of the whole brain. The brain is split into chunks of about 10 k streamlines in file order, and each streamline's neighbors are searched in a 10 % random draw (`k_ds_rate = 0.1`) from its own chunk, about 1 k streamlines. The original training repo does the same (`RealData_PatchData` → `cal_local_feat` on the chunk). So the context depends on file order, and a subsample must not be run as if it were a whole brain.
  - The model was trained with k_global = 500 and `k_ds_rate = 1.0` (`cli_args.txt`). Both the CLI and the paper's test script use 80 and 0.1 at inference.
- The model is a DGCNN whose output is `F.log_softmax` over **1600 classes**: the 800 ORG atlas clusters (0–799) plus an outlier twin for each (800–1599).
- `run_inference` keeps only `pred.data.max(1)[1]`. **This is the line to patch.**
- `tract_mapping.py` maps clusters to 42 tracts in 5 categories (Association, Projection, Commissural, Cerebellar, Superficial). Outlier classes map to "Other."
  - *Correction:* "Other" also holds 289 plausible clusters (0–799) that the atlas leaves unannotated, so it is a group of 1089 classes, not just the outliers.
- Log-softmax equals the logits minus a per-streamline constant. rankfield stores only differences between classes, so the log-probabilities can be encoded directly.

**Test data** (`TestData.tar.gz`, 1.5 GB, from TractCloud GitHub release v1.0.0). It contains four subjects, all `.vtp`:

| Cohort | File | Notes |
|---|---|---|
| HCP (adult) | `HCP/101006_ukf_pp_with_region.vtp` | UKF tractography. **440,621 streamlines, 21.6 M points** (about 49 per streamline). Point arrays: `region_label`, `region_mask`. Roughly ±65 × −98..70 × ±66 mm. |
| dHCP (neonate) | `dHCP/sub-CC00069XX12_ses-26300_pp.vtp` | 738 MB |
| PPMI | `PPMI/3104_pp.vtp` | 612 MB |
| ABCD (child) | `ABCD/sub-000_..._b3000_pp.vtp` | 562 MB |

**Use the HCP subject.** Adding a second cohort (dHCP or ABCD) later would show whether ambiguity changes with age.

**rankfield** (`github.com/mhalle/rankfield`, v0.3.10, format 0.4; checked):

- Install from a pinned tag, since the project is alpha: `pip install "rankfield[torch,store] @ git+https://github.com/mhalle/rankfield.git@v0.3.10"`.
- `encode(logits, *, depth=6, clip=8, keep="shell"|"clip", tail_temperatures=(1.0,), memory_budget=...)` takes `(K, Z, Y, X)` floating-point logits; fp16 is accepted.
- **`keep="clip"` is the rule to use here.** It keeps the winner, then the non-winners within `clip`, closest first, with no grid neighbors involved.
  - Each streamline's record therefore depends only on its own logits.
  - `ranks[1]` is the true runner-up, except at a tie at the depth cut.
  - The default `keep="shell"` assumes a voxel grid and does not fit here.
- Rank dtype is uint16 when K + 1 > 255, which applies here.
- Decoders, which need only numpy, except `decode_groups`, which uses torch:
  - `decode_groups(code, groups)` gives the **best-class** group margin `max_{c∈S} l_c − max_{c∉S} l_c`, floored at `−clip`. It does not use the tail.
  - `probabilities(code, T)` gives the kept classes' probabilities, renormalized using the stored tail. The tail is the total probability of everything dropped, as a single number with no class attribution.
  - `margin`, `deficit` and `tail_at` complete the set.
- Storage: `rankfield.store.write_parts` / `read_parts` read and write zarr v3. They are useful for the offline archive. The web export uses its own raw binary (Step 5).
- No JavaScript decoder was found in rankfield, haversack, labelfield or duckn, although rankfield's comments mention one. The decode is a 256-entry lookup table plus `exp`, so port it rather than wait for it.

## Two margins, defined once

The plan uses two group margins. Both are computed in the Step 2 reference, and M1 decides which one the viewer leads with.

| | Definition | What it means | From the rankfield |
|---|---|---|---|
| **Best-class margin** `m̂_S` | `max_{c∈S} l_c − max_{c∉S} l_c` | Positive exactly for the group containing the argmax cluster, so it agrees with TractCloud's hard label | `decode_groups`. Exact up to the byte step, with a `−clip` floor where nothing outside S was kept |
| **Mass margin** `m_S` | `LSE_{c∈S} l_c − LSE_{c∉S} l_c` | Log-odds of the group as a whole. It is large when probability is spread across sibling clusters of one tract. | From `probabilities()`, counting the tail against S. This gives a lower bound. |

**Winning tract** means the tract of the argmax cluster, which is TractCloud's label. The mass margin of that tract can be negative when another tract holds more total probability. Record where that happens: those are ambiguous streamlines by definition.

The **plausibility margin** takes both forms too. Here S is clusters 0–799 and its complement is clusters 800–1599.

## Pipeline (offline, Python)

The pipeline is a small CLI (`capture`, `analyze`, `encode`, `export`). Each stage reads the previous stage's files, so stages can be rerun independently.

### Step 1: Capture the field

- Install TractCloud with `pip install -e .`. Fetch the model with its own downloader (`model_data.py`) or by hand from the release (`TrainedModel.tar.gz`).
- Patch, or wrap, `run_inference` to return the full `(N_s, 1600)` log-softmax as float16, alongside the argmax. Here N_s is the number of streamlines; "depth" below always means rankfield planes.
- **Run R = 5 times with explicit seeds** for the global sample and the kNN subsample. This turns the hidden nondeterminism into a deliberate ensemble. Save each run as `logp_run{r}.npy`, `(N_s, 1600)` fp16, about 1.4 GB each. Keep these files offline; they never go to the browser.
- Compute: use a GPU on Modal for the full subject. TractCloud's README reports about 33 s of GPU inference per 500 k streamlines and about 17 times longer on CPU. For iteration on CPU, use a fixed 100 k-streamline subsample (also seeded).
- Record the device for each run. Logits from CPU and GPU runs differ in the last bits, and so does rankfield's tail (see Risks).

### Step 2: Derive per-streamline quantities (plain numpy, before rankfield)

Do this first, without rankfield, so it serves as the reference.

- `cluster_label`: argmax. Ties go to the lower index, which is numpy's and rankfield's convention.
- `tract_label`: the cluster mapped to a tract through `tract_mapping.py`.
- For the winning tract:
  - best-class margin `m̂` and mass margin `m`;
  - the runner-up tract under each definition;
  - a flag where the mass winner is not the label winner.
- Plausibility margin, in both forms.
- **Baselines.** Each costs one or two bytes and needs no field:
  - `p_max`: the winning cluster's probability;
  - the cluster-level margin, winning cluster against the runner-up cluster.
- **Instability**, across the R runs:
  - whether the tract label changes from run 0;
  - the number of distinct tract labels;
  - the per-run margins (mean and spread).

### Step 3: Analysis (the M1 figure)

**Target:** the tract label changes in at least one of runs 1–4 relative to run 0.

**Primary predictors, all from run 0 only.** This is how TractCloud is used in practice: you run it once.

- tract best-class margin `m̂`;
- tract mass margin `m`;
- baseline: `p_max`;
- baseline: the cluster-level margin.

**Secondary:** the mean margin across all R runs. Report it as an upper reference only. It uses the ensemble a real user would not have.

**Outputs:**

- The fraction of streamlines whose tract label changes, by margin bin, with one curve per predictor.
- The AUROC of each predictor. **The headline number is the tract-margin AUROC minus the best baseline's AUROC.**
- The tract pairs that account for most changes. Expected: AF vs SLF-III, TF vs CR-F, neighboring superficial tracts, CC subdivisions.
- One or two static plots (matplotlib), plus a small JSON of summary numbers. The viewer embeds these.

### Step 4: Encode with rankfield

```python
lg = torch.from_numpy(logp_run0)            # (N_s, 1600) fp16
lg = lg.T[:, :, None, None]                 # (1600, N_s, 1, 1), a view: streamlines on Z
code = rankfield.encode(lg, keep="clip", depth=6, clip=8.0, tail_temperatures=(1.0,))
```

- **Put streamlines on Z.** The encoder splits work into memory-bounded batches along Z, at about 18 B per class per voxel. That is about 29 KB per streamline at K = 1600, or about 37 k streamlines per batch at the default 1 GiB `memory_budget`. If streamlines go on X, the whole subject lands in one batch of about 12 GB.
- **Order does not matter.** Under `keep="clip"` a record depends only on its own streamline, so encode in capture order and permute for export.
- **Depth:** start at 6. Report fidelity, mean kept planes and the tail distribution at depths 4, 6 and 8. Planes past the clip are sentinels and cost almost nothing after compression.
- **Byte curve:** keep the default log curve. It is finer than a uniform byte below about 1 logit, where instability lives. Its range of 64 is mostly unused under a clip of 8, and that does no harm.
- **Temperatures:** write T = 1 only. Tails at other temperatures can be added later, but only by re-encoding, and only if M1 makes the mass margin the lead and a temperature control is wanted.
- **Fidelity check** against the Step 2 reference:
  - Cluster and tract labels: these should be **exact**. `ranks[0] − 1` is the argmax, with the same tie rule.
  - Best-class margins from `decode_groups`: the error should be within the byte step, except where the true margin is below `−clip`. Report the error distribution and the count of floored streamlines.
  - Mass margins from `probabilities()` with the tail counted against S: these are lower bounds. Report the error distribution against the tail value, and the fraction of streamlines whose sign differs from the reference.
  - Plausibility margins: check both forms in the same way.
- **Size:** the uncompressed figure is 6 × 2 B ranks + 5 B gaps + 2 B tail ≈ 19 B per streamline, or about 8.4 MB for the HCP subject. Report the compressed store size beside it. Compare with:
  - dense fp16: about 1.4 GB;
  - naive top-6 (fp16 values plus uint16 indices): about 24 B per streamline;
  - argmax only: 2 B per streamline;
  - argmax plus `p_max`: 3 B per streamline.

### Step 5: Export for the web

Geometry and field go in separate files, so the viewer can regroup without reloading geometry.

- **Geometry.**
  - 21.6 M points is too much for the browser. Resample each streamline to a fixed or capped point count; start at 16–24 points.
  - Quantize to int16 at 0.01 mm about the volume center.
  - Shuffle streamlines deterministically and split them into chunks of a few percent each, so the first n chunks are a uniform random sample. That lets the viewer load progressively.
  - Each chunk holds `u32 lineCount, u32 pointCount, u32 offsets[lineCount+1], i16 xyz[pointCount*3]`, or a fixed stride if the point count is fixed.
- **Field.** Ship the rankfield ranks, gaps and tail per streamline, permuted into the same shuffled order. Ship the field itself, not precomputed margins: live regrouping needs it, and it is small.
- **Precomputed extras**, one byte or a bitfield per streamline: the instability count across runs, and the run-to-run spread of the margin M1 selected.
- **Tables:**
  - The rankfield level table, as 256 float32 values written from Python with `rankfield.levels(meta)`. The JS port indexes this table and never recomputes the curve, so it matches Python bit for bit.
  - The `clip`.
  - Cluster → tract → category maps, tract colors and full names.
  - The summary JSON from Step 3.
- **Manifest** (`manifest.json`): the chunk list, sizes and counts; center and scale; the rankfield meta block (depth, clip, curve, keep rule, temperatures written); and table file names.
- **Container:** static files. Keep a TRX export (per-streamline data arrays alongside streamlines) as an optional second target, so the result works with other tools.

## Viewer (PicoGL): requirements only

Rendering approach is the implementer's choice. Lines with per-vertex attributes are sufficient; capsule tubes and ray-marching are not required.

**Decode on the client.** Port the level-table decode, both group-margin definitions and the plausibility margin to JS. Recompute them whenever the grouping changes, then upload per-streamline attributes to a buffer or texture. At 100 k streamlines × 6 entries this takes milliseconds.

Two rules the port must keep:

- **A support byte of 0 means the class is absent.** It is not `levels[0]`, which is 64 logits under the default curve. A winner whose runner-up byte is 0 leads by at least `clip`.
- **Best-class levels are floored at `−clip`**, as `decode_groups` floors them. The mass margin counts the tail against S.

Validate the port against Python on a fixed set of streamlines. Decoding through the shared table should make the best-class margins match exactly.

**Display modes:**

1. *Hard labels.* Color by tract, as TractCloud outputs today. This is the baseline.
2. *Margin.* Tract color, with opacity (or saturation) set by the group margin, so ambiguous fibers fade. Default to whichever margin won M1, with the other available as a toggle.
3. *Plausibility.* Color by plausibility margin, with the outlier threshold as a slider.
4. *Instability.* Highlight streamlines whose label changed across runs, overlaid on mode 2. This is the visual counterpart of the M1 figure.

**Controls:**

- Margin definition (best-class or mass), margin threshold and opacity curve.
- Outlier threshold.
- Grouping: cluster, tract or category, plus custom merges (for example, SLF I + II + III). Margins recompute live.
- Tract visibility, toggled per tract and per category.
- Load fraction, for progressive chunks.
- *Temperature: dropped.* It does not change argmax or best-class margins. Reconsider it only if the mass margin leads and extra tails were written in Step 4.

**Panels:**

- The Step 3 plots and summary numbers, including the AUROC gain over the baseline.
- A size panel comparing dense, naive top-6, argmax plus `p_max`, and rankfield, at the encoded depth.
- On hover or click of a streamline: winning tract, runner-up, both margins, plausibility and instability count.
- A short statement of the margin conventions: best-class floored at `−clip`; mass margin as a lower bound with the tail counted against S.

**Constraints:**

- Static hosting, no server.
- Must work at 100 k streamlines on a laptop integrated GPU. Full-subject loading is a stretch goal.
- Orbit camera with standard anatomical view presets.

## Milestones

| | Deliverable | Exit test |
|---|---|---|
| M0 | Field capture on a 100 k subsample, R = 5, plus the Step 2 numpy reference (both margins and both baselines) | Shapes match, and run 0's labels match unpatched TractCloud with the same seeds |
| M1 | Instability vs margin figure; AUROC table from run 0 alone | The tract margin predicts tract-label change monotonically **and** beats the `p_max` / cluster-margin baseline by a stated amount. Otherwise, a documented negative result, which stops the project here. M1 also picks the margin the viewer leads with. |
| M2 | rankfield encode plus fidelity report at depths 4, 6 and 8 | Labels exact. Best-class margins within the byte step outside the floor. Mass-margin error reported against the tail. Sizes reported. |
| M3 | Full-subject capture on Modal, plus web export | Manifest and chunks load; byte counts match |
| M4 | PicoGL viewer, modes 1–2, JS decode validated against Python | Live margin-threshold change on 100 k streamlines |
| M5 | Modes 3–4, regrouping, panels | All controls work; regrouping recomputes margins client-side |

## Risks and open questions

- **Data-use terms. Check before hosting anything.**
  - The test subjects come from HCP, dHCP, PPMI and ABCD, each of which has its own data-use agreement.
  - ABCD and PPMI in particular restrict redistribution, and even derived streamlines may count as redistribution.
  - Default to a local or private demo until this is confirmed. HCP is the most likely candidate for public release.
- **TractCloud license.** It uses the Slicer license. Patching and running it is fine, but confirm the terms before shipping the model or its outputs.
- **Coordinates.** The model sees recentered coordinates. The viewer should display the original RAS.
- **The field may not beat one byte.** If `p_max` predicts tract instability as well as the tract margin, the field is not worth storing for this purpose. Grouping is where the field should earn its keep: a streamline split between sibling clusters has a low cluster confidence and a stable tract. M1 is designed to show whether that happens.
- **The tail is unattributed.** rankfield stores the dropped probability as one number, so the mass margin is a bound, not an exact value. If the tail is large for many low-margin streamlines, raise the depth or the clip and measure again; don't assume.
- **CPU and GPU results differ.** A store written on a GPU is not byte-identical to one written on a CPU: the uint16 tail can differ by one unit. Ranks and gaps are exact given the same logits. The logits themselves also differ between devices. Don't compare M0 (CPU) and M3 (GPU) outputs byte for byte; compare labels and margins within tolerance.
- **The ensemble mixes two sources of randomness.** The global context sample and the kNN subsample both vary. Optionally, vary them separately to see which drives instability; this is useful to the TractCloud authors.

## Handoff notes for the implementer

- Start at M0 on CPU with the 100 k subsample. Nothing downstream is worth building until M1 holds.
- The rankfield API in this document was checked against v0.3.10: `encode` (including `keep="clip"`), `decode_groups`, `probabilities`, `margin`, `tail_at`, `levels` and `store`. Pin that tag, or recheck the API if you move to a later one, because the format is still alpha.
- The TractCloud facts in Background were checked at `94de627` during M0. The model needs `HCP_mass_center.npy`, which ships only in `TrainData_800clu800ol.tar.gz` (166 MB); keep that one file.
- M0 builds the context over the whole brain with upstream `RealDataDataset` unchanged, under `np.random.seed`, and runs the model on the 100 k targets only. A target's label is then exactly what a full-brain run with that seed gives it (`capture.py`).
- There is no JS decoder to reuse. Port the decode from `decode.py` (`_field`, `decode_groups`, `probabilities`). Test it against Python using the level table exported in Step 5.
