# Tractography incubation

Diffusion MRI to named white-matter tracts with confidence, in the rank-field format. Data lives
outside the repo (`_data.py`: `$TRACTOGRAPHY_DATA`, default `~/tmp/data/tractography`); results and
the running journal are in `results/` (`results/NOTES.md`, newest entries last). `pipeline.md`
describes the pipeline and its timings; `plan.md` and `prior-art.md` are the original plan and survey.

## The pipeline

`_pipeline.py` runs it, scan to payload, in memory: correct → track → label → encode. Its docstring
states the conventions every module below follows (array layouts, units, devices).

| module | stage |
|---|---|
| `_ds001226.py` | a subject: the DWI, the reversed-phase-encoding b0s, the T1, the tumor mask |
| `_susc.py` | susceptibility correction: `estimate` (FSL topup's model, GPU), `apply`, `displacement_mm` |
| `_prep.py`, `_median.py` | the tracker's input: one shell, gradients in RAS, DIPY's `median_otsu` mask (exactly, ~100x faster) |
| `_ukf_torch.py`, `_ukf_metal.py` | UKF two-tensor tractography as the Slicer binary does it; the Metal kernel for the steps |
| `_tractcloud.py`, `_resample.py` | TractCloud labels and log-probabilities |
| `_geometry.py` (+ rankfield) | the payload: predictive geometry, the rank field |
| `_t1check.py` | measurement, not pipeline: the distortion left against the T1 |

`_ukf_triton_block.py` is the CUDA counterpart of the Metal kernel (`_ukf_triton.py`, the unrolled
first attempt, compiles too slowly to use).

## Running and checking it

- `cohort.py --sub PATnn`: one ds001226 patient - the pipeline, the scan as acquired for comparison,
  both against the T1 and tumor; `cohort_summary.py` tabulates `results/cohort/`; `cohort_topup.py`
  adds FSL topup as a second reference.
- Component checks, each against its reference: `ukf_compare.py` / `modal_ukf_track.py` (the Slicer
  binary), `ukf_metal_check.py` and `modal_ukf_triton.py` (the GPU kernels against `_ukf_torch`),
  `ukf32_compare.py`, `ukf_noise_floor.py` and `modal_ukf_labels.py` (float32 against the scan's noise),
  `resample_check.py` (upstream's features), `median_check.py` (DIPY), `susc_check.py` and
  `t1_alignment.py` (topup, the T1; PAT16).

## The format work (rankfield on TractCloud)

M0-M3 on the HCP test subject: `capture.py`, `modal_capture.py`, `reference.py`, `analyze.py`,
`regroup.py`, `_groups.py`, `encode.py`, `export.py` (+ `export_check.mjs`, `viewer/`, `decode/`),
`features.py`; encodings and speed: `geometry_bench.py`, `trako_compare.py`, `convert_bench.py`,
`encode_timing.py`, `modal_infer_opt.py`, `modal_timing.py`, `modal_context_variant.py`;
`dmc_field.py` (DeepMultiConnectome's 13,695-class head).

## Records, superseded

Kept because committed results came from them; new work uses the pipeline modules instead.

| script | superseded by |
|---|---|
| `pat16_prep.py` | `_prep.prepare` (it now calls `_prep.prep`) |
| `susc_apply.py` | `_susc.apply` |
| `topup_ref.py` (PAT16; wrote the b0s through the int16 header, quantizing them by up to 0.03) | `cohort_topup.py` |
| `pat16_topup_compare.py`, `mac_pipeline.py`, `mac_labels.py` | `_pipeline.py`, `cohort.py` |
| `pat16_seeding.py`, `albula_compare.py`, `albula_fw_compare.py`, `albula/` | (questions answered: NOTES) |
| `ukf_bench.py`, `ukf_step_bench.py`, `modal_ukf_step.py`, `compare_variant.py`, `modal_tc_hardi.py` | (early timings and variants) |
| `t1_alignment_figure.py` | (PAT16's figure) |
