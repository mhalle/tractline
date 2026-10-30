# The bench

The scripts behind the measurements in `bench/results/NOTES.md`, the running journal (newest entries last). The list
below names the main families; each script's docstring says what it measures, and the NOTES entries say which
script made which result. Run them
from the repository root with `uv run`, which syncs the environment first: `uv run bench/cohort.py --sub PAT16`. Their
data lives in `$TRACTOGRAPHY_DATA` (`docs/development.md`, "Data"); some need an opt-in dependency group, named in
their docstrings and in `docs/development.md`.

Results are in `bench/results/`. Some committed results predate a change of default (the field estimate's optimizer,
the labeler); `bench/results/README.md` says which, and what re-running their scripts would now produce.

- `run_pipeline.py --sub PATnn [--trx [PATH]] [--labeler rapidparc|hemiaug|tractcloud]`: the pipeline on
  one subject; `trx_check.py` reads the TRX back with trx-python and checks it.
- `cohort.py --sub PATnn`: one ds001226 patient (`_ds001226.py`): the pipeline, the scan as acquired
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
- Uncertainty: `ensemble_pilot.py` (the pipeline run with every stage resampled: the scan bootstrapped, its field,
  tracking, the draw; summarized per streamline and per voxel by `tractline.ensemble` and encoded with rankfield;
  `docs/uncertainty-ensemble.md`); `ensemble_votes_test.py` (whether votes are needed beside the mixture: regrouping,
  information beyond it, runs, voxel merges, with block-bootstrap intervals).
- The field estimate: `susc_held_out.py` (+ `susc_held_out_summary.py`: split-half, held-out prediction
  and drift from each scan's own b0s, 12 patients: the test that picks its settings),
  `susc_stability.py`, `susc_convergence.py`, `readout_polarity.py` (+ `_summary`: what the correction needs to
  know of the acquisition: absolute polarity, readout time, relative polarity from the images).
- Speed: `cpu_timing.py`, `modal_cpu_scaling.py` (x86, CPU only), `modal_gpu_pipeline.py` (+
  `gpu_pipeline_compare.py`: the pipeline on CUDA GPUs and CPUs against the M2), `hardi_paths.py` +
  `modal_hardi_paths.py` (the HARDI brain on the M2, an L40S and a Modal CPU container), `ukf_cpu_check.py`,
  `timing_matrix.py` + `modal_timing_matrix.py` (stage times on HARDI, PAT16 and ds005123: the M2, an L4, an A10,
  an L40S and a 32-worker CPU container), `cpu_profile.py` + `modal_cpu_profile.py` (where the CPU tracker's and
  the field estimate's time goes), `ukf_cpu_tune.py` (tracker-step variants: speed and agreement with float64),
  `modal_cpu_compare.py` (the CPU pipeline, committed against working copy, in one x86 container; worker scaling),
  `modal_estimate_compare.py` (the field estimate, committed against working copy, per GPU), `timing_table.py` (the
  cross-platform table from timing_matrix's results), `gpu_track_profile.py` + `modal_gpu_track_profile.py` (the GPU
  tracker's steps against the fibers they carry; block sizes compared bit for bit).
- Head motion and eddy current correction: `motion_eddy/` (its own README): the prototype, the comparisons with eddy
  and TORTOISE, the field referee, the CUDA runner (`modal_cuda.py`) and the comparison of two runs' parameters
  (`compare_records.py`).
- Calibration and robustness: `calibration_*.py` (per-protocol calibration of the ensemble's confidence, across
  protocols, scanners and sites), `robust_*.py`, `gibbs_check.py` (the opt-in robustness steps;
  `docs/robustness-plan.md`).
- The configurable pipeline: `dependency_check.py`, `device_agreement.py`, `csd_check.py`, `ptt_compare.py`
  (`docs/sota-pipeline-design.md`); data fetchers `fetch_ds*.py`.
- PTT without a T1 (NOTES 2026-10-29): `ptt_stopping_phantom.py` (the stopping rules on the ISMRM phantom, count-matched,
  and run to run), `ptt_stopping_scan.py` (on one scan given by path, beside UKF), `ptt_fa_anatomy.py` (where paths end
  against FastSurfer's labels; FA by tissue and shell), `ptt_fa_auto.py` (the automatic threshold against FastSurfer's
  cortex, and its repeatability), `ptt_auto_onharmony.py` (the automatic threshold out of sample: ten ON-Harmony
  scans, FastSurfer run locally; NOTES 2026-10-30).
- The brain mask (NOTES 2026-10-27 and 10-28): `skullstrip_check.py` and `skullstrip_reference_modal.py` (the SynthStrip
  network against its authors' release), `mask_study.py`, `mask_compare_scan.py` (the masks on one scan given by
  path), `cst_thalamus_outside.py` and `cst_traced.py` (CST through the thalamus, named streamlines outside the brain,
  CST by its endpoints).
- Helpers: `_ds001226.py` (the patients' loader), `_fibercmp.py` (fibers compared by seed), `_bootstrap.py`
  (bootstrap replicates of a scan).

Records kept because committed results came from them (their inputs are not all reproducible from this
repository): `pat16_prep.py`, `susc_apply.py`, `topup_ref.py`, `pat16_topup_compare.py`, `pat16_seeding.py`,
`mac_labels.py`, `ukf_bench.py`, `ukf_step_bench.py`, `modal_ukf_step.py`, `compare_variant.py`,
`t1_alignment_figure.py`, `resample_check.py` and `modal_infer_opt.py` (TractCloud inference; both need
`hcp/feat.npy`, made by a script that stayed in the repository where tractline began, before this one), `ukf_hotspots.cc` (the binary's
profile, `results/ukf_hotspots.json`). `t1_alignment.py` and `median_check.py` read files those records
made (`susc_apply.py`, `pat16_prep.py`).

## On Modal

The `modal_*.py` scripts run through the modal CLI, a uv tool (`uv tool install modal`;
`modal run bench/modal_gpu_pipeline.py`); their images install with uv, torch pinned at 2.14.1 (every committed
Modal result's). They use a Modal Volume `tractography-bench` for inputs and the Triton
compile cache; most scripts ship their inputs from `$TRACTOGRAPHY_DATA` in the image, and the Volume holds
the rest: `triton-cache/`, `ukf/hardi/` (`dwi.nhdr`, `mask.nrrd`), `TrainedModel/`,
`TrainData_800clu800ol/HCP_mass_center.npy`, `hcp/feat.npy`, `variants/f64.npz`, as each script's docstring
names them; `modal_gpu_pipeline.py --save-fibers` writes to its `tractline/`.
