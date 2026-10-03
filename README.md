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

```
pip install -e .              # numpy, scipy, torch, nibabel
pip install -e ".[nrrd]"      # NRRD input/output for the tracker
pip install -e ".[triton]"    # the tracker's kernel on CUDA
```

Data and weights live outside the repository, in `$TRACTOGRAPHY_DATA` (default `~/tmp/data/tractography`;
layout in `src/tractline/data.py`):
- **RapidParc's weights** (the default labeler): `RapidParc/rapidparc.safetensors` (and `hemiaug.safetensors`),
  6.6 MB each, from https://github.com/MedVisBonn/RapidParc/releases/tag/v1.0.0; checked against their
  sha256 when loaded.
- **TractCloud** (optional labeler): its code in `TractCloud/src` (github.com/SlicerDMRI/TractCloud), its
  weights `TrainedModel/` and `TrainData_800clu800ol/HCP_mass_center.npy` from its release v1.0.0.
- **The bench's data**: OpenNeuro ds001226 (BTC_preop, CC0) in `ds001226/`; the Stanford HARDI scan in
  `ukf/hardi/`; TractCloud's test split (`TrainData_800clu800ol/test.pickle`) and HCP test tractogram
  (`TestData/`) from its release.

The bench needs more than the package: trx-python, dipy, pynrrd, vtk, matplotlib, scikit-learn, modal (for
the `modal_*.py` scripts, which also use a Modal Volume `tractography-bench` for inputs and the Triton
compile cache), and FSL for the topup references.

## The package (`src/tractline`)

`pipeline.py` runs it in memory: correct → track → label, then optionally TRX. Its docstring states the
conventions every module follows (array layouts, units, devices). The default path needs numpy, scipy,
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

Run from the repository root with the package installed, e.g. `python bench/cohort.py --sub PAT16`.
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
  `modal_hardi_paths.py` (the HARDI brain on the M2, an L40S and 32 x86 cores), `modal_infer_opt.py`
  (TractCloud inference), `ukf_cpu_check.py`.

Records kept because committed results came from them (their inputs are not all reproducible from this
repository): `pat16_prep.py`, `susc_apply.py`, `topup_ref.py`, `pat16_topup_compare.py`, `pat16_seeding.py`,
`mac_labels.py`, `ukf_bench.py`, `ukf_step_bench.py`, `modal_ukf_step.py`, `compare_variant.py`,
`t1_alignment_figure.py`, `resample_check.py` (needs `hcp/feat.npy`, made by a script that stayed in the
incubation repository).

## Documents (`docs/`)

`pipeline.md` (the pipeline as built, its timings, what is open), `labelers.md` (the labelers),
`prior-art.md` (the incubation's survey: streamline storage, the confidence of streamline parcellation,
compact soft output - some of its references point to format work that is not in this repository).

## Not here

The compact format work (the rank field, predictive geometry, the viewers, DeepMultiConnectome's field
test) stayed in the incubation repository; when it returns it will be wired into an exporter from the
labeler's in-memory probabilities, not through the TRX.

## Licenses

Apache License 2.0 (`LICENSE`). What tractline takes from others, and on what terms, is in
`THIRD_PARTY_NOTICES.md`.
