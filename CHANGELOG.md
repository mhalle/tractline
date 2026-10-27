# Changelog

Releases are git tags (`uv add git+https://github.com/mhalle/tractline --tag vX.Y.Z`); `main` is the newest
release and `dev` the work in progress (`docs/development.md`, "Branches and releases"). The experiments
behind each change, with their numbers and later corrections, are in `bench/results/NOTES.md`.

## 0.5.2 - 2026-10-27

- **The README rewritten** for a first-time reader: what tractline does, a quick start with `uvx` and
  `uv tool install`, what it reads (DICOM through dcm2niix), the steps, timings, versions, use from Python
  (`uv add ... --tag`) and the documentation. The developer material moved to `docs/development.md` (setup, data,
  tests, branches and releases, the package's modules) and the bench's catalog to `bench/README.md`.
- **Fixed: `pipeline.run(trx=...)` into a folder that does not exist** failed when writing the TRX, after the whole
  run; the folder is now made before the work starts (and by `trx.write` itself).

## 0.5.1 - 2026-10-27

- **`tractline docs`**: the version, every command's help, the README and the QC guide as one Markdown text (or one of
  them: `readme`, `guide`, `help`), for reading or for an LLM. The wheel carries both documents (`tractline/_docs`), so
  it works from `uvx` and `pip` installs.
- **From DICOM**: the QC guide (section 2) and the README give a tested dcm2niix recipe (`uvx dcm2niix@1.0.20260724`),
  with the file naming that lets tractline pair a reversed series, and the rename for a spin-echo field map.

## 0.5.0 - 2026-10-27

- **Head motion and eddy current correction on CUDA, with fused kernels** (`tractline.motion_eddy_triton`): Triton
  versions of the Metal kernels (the Gauss-Newton moments, the cost, the predictions, the forward resample and the
  final resample), chosen automatically on CUDA when Triton is installed (`Options.backend`: `triton`). On an NVIDIA A10,
  ON-Harmony 16974's correction takes 3.4 s against 94 s for the plain torch path (27 s on an M2), with 4.2 GiB of
  device memory, and agrees with the M2's to about 0.003 mm and 0.001 degrees. The kernels compile on first use (about
  80 s), cached by Triton; none of their arguments is specialized on, so one copy serves every scan size. The memory
  check before a run covers CUDA too (the free device memory, with a budget measured on the A10).
- **`tractline warm [series]`**: on CUDA, compiles the tracker's kernels and the motion and eddy correction's ahead of
  the first run, into Triton's cache (`TRITON_CACHE_DIR`): for a series' protocol, for AP/PA phase encoding with and
  without a field (the default), or every combination (`--all`); `motion_eddy_triton.warm()` from Python.
- `tractline run --motion-eddy --gibbs` on the A10: 36 s from scan to labels (the M2: 153 s).

## 0.4.0 - 2026-10-26

- **`tractline run --gibbs`**: Gibbs ringing removed first (the `gibbs` slot, experimental), from the series as
  acquired and from the reversed phase-encoding b0s on their own grid; the slice axis from the sidecar's
  `SliceEncodingDirection` (`pipeline.slice_axis_of`). Its time and b0 counts are in the JSON summary, and the scan's
  total includes it. `pipeline.run(gibbs=)` and `config_for(gibbs=)` turn the slot on from Python.

## 0.3.0 - 2026-10-26

New methods are opt-in. The default pipeline (preset `default`, `ukf-rapidparc@1`, config hash 28d01a1d...) runs
0.2.0's methods; its output differs only as the speed-ups at the end of this list say. Methods marked experimental are checked against their oracles but have been used on few scans.

- **The `tractline` command** (`tractline.cli`, click): `tractline run` takes one dMRI series (NIfTI, BIDS) to labeled
  UKF tractography as TRX, with options for the correction, the shell, the labeler, the device, QC files, the
  corrected series and a JSON summary; `tractline fetch` gets the labelers' weights ahead of a run. It tracks the
  shell with the most volumes unless `--shell` says otherwise (the default preset tracks b = 2800, so the command's
  config hash differs from the preset's on other shells). It runs without a
  checkout: `uvx --from "tractline[cli] @ git+https://github.com/mhalle/tractline@v0.3.0" tractline run ...`.
  Progress goes to standard error and errors end in one line (`TRACTLINE_DEBUG=1` shows the traceback).
- **Weights fetched on first use** (`tractline.fetch`): RapidParc's from its release (6.6 MB a model) and TractCloud's
  code and model from its GitHub (about 42 MB: four modules at a pinned commit, the v1.0.0 model, and the atlas
  centroid streamed out of the training-data archive), each checked against a pinned sha256 and kept in a user cache
  (`TRACTLINE_CACHE`; `TRACTLINE_OFFLINE=1` refuses to download). A checkout's `$TRACTOGRAPHY_DATA` copies are used
  when present. `--labeler tractcloud` no longer needs a checkout; importing `labelers.tractcloud` fetches what is
  missing.
- **A configurable pipeline** (`tractline.config`, `pipeline.execute`, `pipeline.config_for`): eight slots (`denoise`,
  `gibbs`, `field`, `motion_eddy`, `mask`, `model`, `track`, `bundles`), each off or one implementation with checked
  options. A config hashes to sha256 over its canonical JSON; slots that are off are left out, so a new slot does not
  change an existing config's hash, and runtime guards (`Option(hashed=False)`) are not hashed. A run records its
  config and its runtime choices beside it.
- **Head motion and eddy current correction** (`motion_eddy` slot, `tractline.motion_eddy`; experimental): per volume,
  a rigid motion and a polynomial along the phase-encoding axis (linear, or quadratic as eddy's `--flm`), each volume
  registered to a leave-one-out prediction from its shell, fitted by Gauss-Newton on the GPU (Metal kernels on Apple
  GPUs). The susceptibility field is composed into the same map, so each volume is resampled once, and the b-vectors
  are rotated with the head. Outlier slices (signal loss) are replaced by the prediction, as eddy's `--repol`, or
  flagged; corrupt, empty or runaway volumes are left out of the fit and of tracking. QC files: `write_qc` (TSV, JSON)
  and `plot_qc`. With eddy given the same field on ON-Harmony 16974 (one low-motion scan), the composed correction is
  nearer eddy's than the correction applied in sequence in all 55 volumes compared. The CPU runs the same model (its
  kernels checked against the GPU's) but has not been run end to end on a real scan; an estimated 12-13 minutes for
  105 volumes. Not yet tested on CUDA. Needs scipy >= 1.15 (the `bids` or `cli` extra).
- **Gibbs ringing removal** (`gibbs` slot, `tractline.gibbs`; experimental): DIPY's subvoxel-shift method (Kellner et
  al. 2016) in torch, in one process.
- **Denoising** (`denoise` slot, `tractline.denoise`): DIPY's Patch2Self or MP-PCA, run in a subprocess, before or
  after the correction (`order`). Needs DIPY, and scikit-learn for Patch2Self.
- **CSD and PTT** (`model: csd`, `track: ptt`; experimental): single-shell constrained spherical deconvolution in
  torch, matching DIPY's fit, and parallel transport tractography, checked against DIPY's in a compatibility mode and
  run in real units (DIPY mixes voxels and millimeters); a Metal kernel on Apple GPUs. Its default stopping and
  seeding (ACT, white matter) need tissue maps from the caller (`tissue=`).
- **Opt-in settings from the robustness study**: RapidParc's context groups keyed by seed (`grouping`:
  `hash-sorted`, `hash-bucket`), and a UKF tie-break between the two tensors (`ukf.track(tie_angle=, tie_sw2=)`).
- **Uncertainty prototypes**: `tractline.ensemble` (several runs of the pipeline with every stage resampled, summarized
  per streamline and per voxel) and `tractline.compact` (compact run records; needs numcodecs and rankfield, a
  dependency group of the repository, not an extra). The calibration studies built on them
  are in `bench/` and `docs/data-product.md`.
- **Changed**:
  - The mask is its own timed stage (`pipeline.STAGES`; `pipeline_stages(prefix, cfg)` follows the config), with the
    same voxels as before; `pipeline.brain_mask` and `pipeline.correct_motion_eddy` are new.
  - New parameters: `pipeline.run(tracker=, denoise=, denoise_order=, config=, motion_eddy=)`, `pipeline.track(tracker=)`,
    `label(draw=, grouping=)`, `prepare(mask=)`, `Timer(stream=)`; `ukf.track(observe=, compile=)`, and
    `stream=True` its GPU default (the same fibers).
  - `pipeline.execute` returns a `Run`, which still unpacks as `corr, tg, labels`.
  - `Correction` gains `denoise`, `gibbs`, `bvec`, `motion_eddy` and `provenance`; `applied` means the field was
    applied, so it is False for a motion and eddy correction without a field. `bids.Subject` gains `b0_native`.
  - click is a dependency; new extras `cli` (nibabel, scipy >= 1.15, matplotlib) and `plot` (matplotlib); the `bids`
    extra now asks for scipy >= 1.15.
- **Documentation**: a guide to running tractline with head motion and eddy current correction and reading its QC
  (`docs/motion-eddy-qc-guide.md`); the designs of the configurable pipeline and of the correction
  (`docs/sota-pipeline-design.md`, `docs/gpu-motion-eddy-design.md`); American spelling throughout.
- **GPU tracking faster, the fibers bit for bit the same**: the seed tensors fitted through one pseudo-inverse instead
  of a least squares per seed on the host (HARDI's tracking on an L40S 35.9 → 16.8 s, an L4 72.7 → 42.4 s); one stream
  of up to 200,000 half-fibers topped up as they stop instead of blocks of 50,000 run down to their longest fiber
  (`ukf.track(stream=True)`, the GPU default; with the seed fit, the M2's HARDI 96 → 81 s, ds005123 41 → 32 s - refill
  alone 0-8 % against blocks of 200,000; ~0.4 GB of device memory);
  the kernels' launchers no longer read constants back from the device every step.
- **The field estimate ~1.8x faster** on every device (PAT16, the M2's CPU: 22.6 → 12.3 s; 32 x86 cores 29-34 → 19 s):
  the Gauss-Newton matrix applied through sums over the volumes, the sampling grid built once a level where motion is
  held. The same mathematics; the field moves by the estimator's own rounding noise (as much as changing the thread
  count moved it before), and the 12 patients' agreement with the T1 is unchanged (margin 99th 1.58 → 1.59 mm). The
  M2's default path, scan to labels: 54 → 42 s (median of 12).
- **The CPU tracker's fast step changed** (`ukf.track(fast=True)`, the CPU pipeline's): the N gradients once with the
  filter's sums doubled, batched products, interp's 27 voxels in one gather - 1.3-1.4x in one process, as close to
  float64 as before; the CPU pipeline's tracking 1.1-1.4x (PAT16 on 32 x86 cores 39 → 34 s).
- **`ukf.track(compile=True)`**: the fast step through torch.compile (needs a C++ compiler). 2-3x in one process, but
  each worker pays torch.compile's start, and with 8 or more workers it was slower: the pipeline uses it only with
  `TRACTLINE_COMPILE=1`. `stats["compile_s"]`: its time.
- **The Triton tracker compiles one copy per gradient-count size** instead of 6-8 (its integer arguments no longer
  specialized): a new protocol's first CUDA run pays 27-56 s instead of up to 7 min. `ukf_triton_block.warm()`
  compiles all six sizes ahead (75 s on an L4, in parallel).
- **This changelog**; the README warns that 0.1.0's CPU path fails.

## 0.2.0 - 2026-10-03

- **BIDS input, with the series that correct it found and checked** (`tractline.bids.load`, the `bids`
  extra: nibabel, scipy). DICOM arrives through dcm2niix's output. The correcting series are found from the
  sidecars (`B0FieldIdentifier`/`B0FieldSource`, `IntendedFor`, or a diffusion series in the same folder
  phase-encoded the other way) or named (`partners=`), and used only when they share the diffusion
  series' phase-encoding axis, cover both polarities, were acquired under the same shim (`ShimSetting`),
  match its protocol, cover its brain, and have a readout time stated or safely assumed; otherwise the
  reason is given. Follows BIDS's inheritance principle inside a BIDS dataset.
  `bench/run_pipeline.py --bids PATH [--partners ...]`.
- **Correction is optional.** Without a valid pair the series is tracked as acquired:
  `Correction.applied` is False and `Correction.note` says why; `field_hz`, `motion` and `displacement_mm`
  are then None (they were always arrays).
- **A residual check after correcting** (`pipeline.residual_left`, `Correction.residual_left`,
  `Correction.warnings`): how much of the pair's difference the field leaves; above `RESIDUAL_WARN` (0.5)
  a warning - motion between the series, or series that do not share one field - never a refusal; not
  judged (None) when the series differ by little more than their own variation.
- **Each b0's own readout time** (`b0_readout_s` on the subject) when the series that measure the field
  differ.
- **The package requires only numpy and torch.** scipy and nibabel are no longer dependencies: scipy is
  the `t1check` extra, and reading NIfTI is the caller's or `tractline.bids`'s. `triton` (the `triton`
  extra) installs on Linux only.
- **uv first**: a uv project with `uv.lock` and `.python-version` (3.12); the bench's packages are
  dependency groups (`dev` by default; `dipy`, `vtk`, `sklearn` opt-in), never in the package's metadata.
- **Fixed: `pipeline.run(device="cpu")` crashed** with UnboundLocalError (a function-local `import os`
  shadowed the module's), from 0.1.0's review fixes on.
- **Fixed: RapidParc on a lone straight or planar streamline** gave NaN probabilities (0/0 in its
  normalization); an axis with no extent now divides by 1, the arithmetic otherwise RapidParc's.
- **`Timer.skip`**: a stage that did not run is recorded as 0 s; `Timer.total` still raises on a name it
  does not know.
- **Tests and CI**: 70 tests, among them a pair distorted by a known field (the correction recovers it),
  the package with numpy and torch alone, the BIDS reader's refusals; ruff; GitHub Actions on Linux.
- **Docs**: `CHANGELOG.md`; `docs/duckn-proposal.md` and `docs/dcm2niix-dmri-map.md` (what a duckn store
  would need to carry for tractline, and where dcm2niix keeps each vendor's diffusion facts); the README's
  "Your own data". Ron Kikinis is acknowledged as a contributor.

## 0.1.0 - 2026-10-03

The first release, as the tractography work left its incubation repository.

**Known problem:** `pipeline.run(device="cpu")` fails with UnboundLocalError. Use 0.2.0.

- **The pipeline, scan to labeled streamlines** (`pipeline.run`): susceptibility correction from a
  reversed phase-encoding pair - FSL topup's model, estimated by Gauss-Newton with HySCO's anti-folding
  penalty; UKF two-tensor tractography as the Slicer UKFTractography binary does it, on Metal, Triton
  (CUDA) or CPU worker processes; tract labels from RapidParc (the default; its inference written here,
  its released weights) or TractCloud (optional, at the context its model was trained with); optional
  TRX output with labels and tract probabilities.
- **Checked against the originals**: the tracker against the binary, the mask against DIPY, the
  correction against topup and the T1 on 12 tumor patients (OpenNeuro ds001226: the tumor margin's
  misplacement 4.8 mm to 1.6 mm, 99th percentile, median), the labelers on TractCloud's labeled test split
  (RapidParc 94.5 % tract accuracy, as its paper reports).
- Apache License 2.0; third-party credits in `THIRD_PARTY_NOTICES.md`.

## Before 0.1.0

Incubated in another repository from 2026-10-01 and moved here with its history (2026-10-02): the tracker
and its GPU kernels, the field estimate (L-BFGS, then Gauss-Newton), TractCloud's labeling and the
instability of its single context draw, RapidParc. The git history before `1814b62` and
`bench/results/NOTES.md` record it.
