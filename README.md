# tractline

Diffusion MRI to named white-matter tracts, in torch: susceptibility correction from a reversed
phase-encoding pair, UKF two-tensor tractography, and tract labeling, as one pipeline from the scan
to labeled streamlines, on an Apple GPU, a CUDA GPU or the CPU. The correction (FSL topup's model) and
the tracker (the Slicer UKFTractography algorithm, with Metal and Triton kernels) are our own
implementations, each checked against its original; labeling uses RapidParc (its inference written here,
its released weights; the default) or TractCloud (its model code and weights, the context built here).

Private while it is being tested. It was incubated in another (private) repository; this one keeps that
history.

## Install

With [uv](https://docs.astral.sh/uv/), from a checkout:

```
uv sync                       # the package (numpy, scipy, torch, nibabel) and the bench's (the dev group)
uv sync --extra triton        # + the tracker's kernel on CUDA (Linux with an NVIDIA GPU)
uv run bench/run_pipeline.py --sub PAT16
```

As a dependency of another project (the bench's groups stay behind - they are not in the package's metadata):

```
uv add "tractline @ git+https://github.com/mhalle/tractline@v0.1.0"
uv add "tractline[nrrd,triton] @ git+https://github.com/mhalle/tractline@v0.1.0"   # NRRD input/output, CUDA kernel
```

`pip install` works too, from the same URL. On Linux, PyPI's torch is the CUDA build (Triton included).

Data and weights live outside the repository, in `$TRACTOGRAPHY_DATA` (default `~/tmp/data/tractography`;
layout in `src/tractline/data.py`):
- **RapidParc's weights** (the default labeler): `RapidParc/rapidparc.safetensors` (and `hemiaug.safetensors`),
  6.6 MB each, from https://github.com/MedVisBonn/RapidParc/releases/tag/v1.0.0; checked against their
  sha256 when loaded.
- **TractCloud** (optional labeler): its code in `TractCloud/src` (github.com/SlicerDMRI/TractCloud), its
  weights `TrainedModel/` and `TrainData_800clu800ol/HCP_mass_center.npy` from its release v1.0.0.
- **The bench's data**: OpenNeuro ds001226 (BTC_preop, CC0) in `ds001226/`; the Stanford HARDI scan in
  `ukf/hardi/`; TractCloud's test split (`TrainData_800clu800ol/test.pickle`) and HCP test tractogram
  (`TestData/`) from its release. For `topup_ref.py` and `cohort_topup.py`: `fsl-env/`, FSL's topup
  (conda, from FSL's channel - FSL is not a Python package).

The bench needs more than the package: pynrrd, dipy, trx-python and matplotlib, the `dev` dependency
group, which `uv sync` and `uv run` install by default. Two heavy packages are opt-in groups, each for a
few scripts: `vtk` (0.5 GB; `ukf_compare.py`, `ukf_bench.py`, `resample_check.py`: VTK tractography files)
and `sklearn` (`accuracy_tractcloud_test.py`) - `uv run --group vtk bench/ukf_compare.py`. `rapidparc_check.py` runs RapidParc's
own package in an isolated environment uv builds for it (`uv run --with RapidParc==1.0.4`; its pins would not
fit the project's). The `modal_*.py` scripts run through the modal CLI, a uv tool (`uv tool install modal`;
`modal run bench/modal_gpu_pipeline.py`); their images install with uv, torch pinned at 2.14.1 (every committed
Modal result's). They use a Modal Volume `tractography-bench` for inputs and the Triton
compile cache; most scripts ship their inputs from `$TRACTOGRAPHY_DATA` in the image, and the Volume holds
the rest: `triton-cache/`, `ukf/hardi/` (`dwi.nhdr`, `mask.nrrd`), `TrainedModel/`,
`TrainData_800clu800ol/HCP_mass_center.npy`, `hcp/feat.npy`, `variants/f64.npz`, as each script's docstring
names them; `modal_gpu_pipeline.py --save-fibers` writes to its `tractline/`.

## The package (`src/tractline`)

`pipeline.py` runs it in memory: correct → track → label, then optionally TRX. Its docstring states the
subject it takes (the fields `run(s, ...)` reads) and the conventions every module follows (array
layouts, units, devices). Two defaults to know: `device="mps"` (pass `"cuda"` or `"cpu"` elsewhere) and
`shell=2800.0` (ds001226's b-value; pass the scan's own). On the CPU the tracker's workers share their
input through shared memory; `TRACTOGRAPHY_SHARE=0` sends copies instead, for systems without it. The default path needs numpy, scipy,
torch, nibabel and RapidParc's weights, nothing more - no TractCloud code (`bench/dependency_check.py`). A
script using the CPU path needs an `if __name__ == "__main__":` guard: the tracker spawns worker processes,
which re-import the script (the tracker checks, and refuses without one).

| module | stage |
|---|---|
| `susceptibility.py` | `estimate` (FSL topup's model by Gauss-Newton, with HySCO's anti-folding penalty; GPU or CPU), `apply`, `displacement_mm` |
| `prep.py`, `mask.py` | the tracker's input: one shell, gradients in RAS, DIPY's `median_otsu` mask (exactly, in torch) |
| `ukf.py`, `ukf_metal.py`, `ukf_triton_block.py` | UKF two-tensor tractography as the Slicer binary does it; the Metal (Apple) and Triton (CUDA) kernels for the steps (`ukf_triton.py`, the unrolled first attempt, compiles too slowly to use) |
| `labelers/rapidparc.py` | RapidParc labels - the default |
| `labelers/tractcloud.py`, `resample.py` | TractCloud labels, at the context its model was trained with (optional) |
| `labelers/base.py`, `labelers/scheme_43.json` | what labelers share: `Labels`, the 40 mm cut, the 43-class scheme |
| `trx.py` | optional output: the tractogram as TRX, with tract labels and probabilities |
| `t1check.py` | measurement, not pipeline: the distortion left against the T1 |
| `data.py` | where data and weights live |

## The bench (`bench/`)

Run from the repository root with `uv run` (it syncs the environment first), e.g. `uv run bench/cohort.py --sub PAT16`.
Results and the running journal are in `bench/results/` (`NOTES.md`, newest entries last). Some committed
results predate a change of default (the field estimate's optimizer, the labeler); `bench/results/README.md`
says which, and what re-running their scripts would now produce.

- `run_pipeline.py --sub PATnn [--trx [PATH]] [--labeler rapidparc|hemiaug|tractcloud]`: the pipeline on
  one subject; `trx_check.py` reads the TRX back with trx-python and checks it.
- `cohort.py --sub PATnn`: one ds001226 patient (`_ds001226.py`) - the pipeline, the scan as acquired
  for comparison, both against the T1 and tumor; `cohort_summary.py` tabulates `results/cohort/`;
  `cohort_topup.py` adds FSL topup as a second reference. Changes to the pipeline are checked against the
  committed 12-patient cohort.
- Against the originals: `ukf_compare.py` / `modal_ukf_track.py` (the Slicer binary),
  `ukf_metal_check.py` and `modal_ukf_triton.py` (the GPU kernels against `ukf`), `ukf32_compare.py`,
  `ukf_noise_floor.py` and `modal_ukf_labels.py` (float32 against the scan's noise),
  `median_check.py` (DIPY), `susc_check.py` and `t1_alignment.py` (topup, the T1; PAT16),
  `rapidparc_check.py` / `modal_rapidparc_check.py` (RapidParc's package).
- The labelers: `accuracy_tractcloud_test.py` (accuracy on TractCloud's labeled test split),
  `labeler_compare.py` (12 patients, four labelers), `label_noise_floor.py`, `label_draws.py`,
  `label_context.py`, `label_kglobal.py`, `label_trained_check.py` (TractCloud's context and stability),
  `memory_batch.py` (memory across patients in one process).
- The field estimate: `susc_held_out.py` (+ `susc_held_out_summary.py`: split-half, held-out prediction
  and drift from each scan's own b0s, 12 patients - the test that picks its settings),
  `susc_stability.py`, `susc_convergence.py`.
- Speed: `cpu_timing.py`, `modal_cpu_scaling.py` (x86, CPU only), `modal_gpu_pipeline.py` (+
  `gpu_pipeline_compare.py`: the pipeline on CUDA GPUs and CPUs against the M2), `hardi_paths.py` +
  `modal_hardi_paths.py` (the HARDI brain on the M2, an L40S and a Modal CPU container), `ukf_cpu_check.py`.
- Helpers: `_ds001226.py` (the patients' loader), `_fibercmp.py` (fibers compared by seed), `_bootstrap.py`
  (bootstrap replicates of a scan).

Records kept because committed results came from them (their inputs are not all reproducible from this
repository): `pat16_prep.py`, `susc_apply.py`, `topup_ref.py`, `pat16_topup_compare.py`, `pat16_seeding.py`,
`mac_labels.py`, `ukf_bench.py`, `ukf_step_bench.py`, `modal_ukf_step.py`, `compare_variant.py`,
`t1_alignment_figure.py`, `resample_check.py` and `modal_infer_opt.py` (TractCloud inference; both need
`hcp/feat.npy`, made by a script that stayed in the incubation repository), `ukf_hotspots.cc` (the binary's
profile, `results/ukf_hotspots.json`). `t1_alignment.py` and `median_check.py` read files those records
made (`susc_apply.py`, `pat16_prep.py`).

## Documents (`docs/`)

`pipeline.md` (the pipeline as built, its timings, what is open), `labelers.md` (the labelers),
`prior-art.md` (the incubation's survey: streamline storage, the confidence of streamline parcellation,
compact soft output - some of its references point to format work that is not in this repository).

## Not here

The compact format work (the rank field, predictive geometry, the viewers, DeepMultiConnectome's field
test) stayed in the incubation repository; when it returns it will be wired into an exporter from the
labeler's in-memory probabilities, not through the TRX.

## Contributors

Ron Kikinis contributed to tractline.

## Licenses

Apache License 2.0 (`LICENSE`): all of tractline's code. The originals it implements, and the one file it
redistributes (RapidParc's tract scheme, with its notice), are credited in `THIRD_PARTY_NOTICES.md`.
