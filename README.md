# tractline

Diffusion MRI to named white-matter tracts, in torch: susceptibility correction from a reversed
phase-encoding pair, UKF two-tensor tractography, and tract labeling, as one pipeline from the scan
to labeled streamlines. Each stage is our own implementation of a published method, checked against
the original: FSL topup's model, the Slicer UKFTractography binary, and RapidParc (the default labeler)
or TractCloud.

Private while it is being tested. It was incubated in medseg (`bench/tractography` on the
`tractography-incubation` branch); this repository keeps that history.

## Install

```
pip install -e .            # numpy, scipy, torch, nibabel
pip install -e ".[nrrd]"    # NRRD input/output for the tracker
```

The default labeler needs RapidParc's released weights (`RapidParc/`: `rapidparc.safetensors`, 6.6 MB,
from github.com/MedVisBonn/RapidParc v1.0.0), read from the data directory (`tractline.data`:
`$TRACTOGRAPHY_DATA`, default `~/tmp/data/tractography`). TractCloud, optional, also needs its code and
weights there. `docs/labelers.md` describes the labelers.

## The package (`src/tractline`)

`pipeline.py` runs it in memory: correct → track → label, then optionally TRX, on an Apple GPU
(`mps`), a CUDA GPU or the CPU. Its docstring states
the conventions every module follows (array layouts, units, devices). The default path needs numpy,
scipy, torch, nibabel and RapidParc's weights, nothing more - no TractCloud code (`bench/dependency_check.py`).
A script using the CPU path needs an `if __name__ == "__main__":` guard: the tracker spawns workers.

| module | stage |
|---|---|
| `susceptibility.py` | `estimate` (FSL topup's model by Gauss-Newton, with HySCO's anti-folding penalty; GPU or CPU), `apply`, `displacement_mm` |
| `prep.py`, `mask.py` | the tracker's input: one shell, gradients in RAS, DIPY's `median_otsu` mask (exactly, in torch) |
| `ukf.py`, `ukf_metal.py`, `ukf_triton_block.py` | UKF two-tensor tractography as the Slicer binary does it; the Metal (Apple) and Triton (CUDA) kernels for the steps (`ukf_triton.py`, the unrolled first attempt, compiles too slowly to use) |
| `labelers/tractcloud.py`, `resample.py` | TractCloud labels and log-probabilities (at the context the model was trained with) |
| `labelers/rapidparc.py` | RapidParc labels - the default (its released weights; the same 43-class scheme) |
| `labelers/base.py` | what labelers share: `Labels`, the 40 mm cut |
| `trx.py` | optional output: the tractogram as TRX, with tract labels and probabilities |
| `t1check.py` | measurement, not pipeline: the distortion left against the T1 |
| `data.py` | where data and weights live |

## The bench (`bench/`)

Run from the repository root with the package installed, e.g. `python bench/cohort.py --sub PAT16`.
Results and the running journal are in `bench/results/` (`NOTES.md`, newest entries last).

- `run_pipeline.py --sub PATnn [--trx [PATH]]`: the pipeline on one subject; `trx_check.py` reads the
  TRX back with trx-python and checks it.
- `cohort.py --sub PATnn`: one ds001226 patient (`_ds001226.py`) - the pipeline, the scan as acquired
  for comparison, both against the T1 and tumor; `cohort_summary.py` tabulates `results/cohort/`;
  `cohort_topup.py` adds FSL topup as a second reference. Every change to the pipeline is checked by
  reproducing the 12-patient cohort.
- Against the originals: `ukf_compare.py` / `modal_ukf_track.py` (the Slicer binary),
  `ukf_metal_check.py` and `modal_ukf_triton.py` (the GPU kernels against `ukf`), `ukf32_compare.py`,
  `ukf_noise_floor.py` and `modal_ukf_labels.py` (float32 against the scan's noise),
  `resample_check.py` (TractCloud's features), `median_check.py` (DIPY), `susc_check.py` and
  `t1_alignment.py` (topup, the T1; PAT16).
- The field estimate: `susc_held_out.py` (+ `_summary`: split-half, held-out prediction and drift from
  each scan's own b0s, 12 patients - the test that picks its settings), `susc_stability.py`,
  `susc_convergence.py`.
- Speed: `cpu_timing.py`, `modal_cpu_scaling.py` (x86, CPU only), `modal_gpu_pipeline.py` (+
  `gpu_pipeline_compare.py`: the pipeline on CUDA GPUs against the M2), `hardi_paths.py` +
  `modal_hardi_paths.py` (the HARDI brain on the M2, an L40S and 32 x86 cores), `modal_infer_opt.py`
  (TractCloud inference), `ukf_cpu_check.py`.

Records kept because committed results came from them: `pat16_prep.py`, `susc_apply.py`,
`topup_ref.py`, `pat16_topup_compare.py`, `pat16_seeding.py`, `mac_labels.py`, `ukf_bench.py`,
`ukf_step_bench.py`, `modal_ukf_step.py`, `compare_variant.py`, `t1_alignment_figure.py`.

## Documents (`docs/`)

`pipeline.md` (the pipeline as built, its timings, what is open), `labelers.md` (the labeling stage as
a swappable component), `prior-art.md` (the incubation's survey: streamline storage, the confidence of
streamline parcellation, compact soft output).

## Not here

The compact format work (the rank field, predictive geometry, the viewers, DeepMultiConnectome's
field test, and the incubation's original plan for it) stays in medseg; when it returns it will be wired into an exporter from the labeler's
in-memory probabilities, not through the TRX.
