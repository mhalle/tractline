# What the committed results were made with

`NOTES.md` is the journal: each entry says what was run and what it found, as of its date. The result
files are what those runs wrote. Two defaults changed during the work, so some files were made under an
older one, and re-running their script today writes something different under the same name.

**The field estimate's optimizer**: L-BFGS until 2026-10-02 ("Gauss-Newton is the default" in NOTES),
Gauss-Newton since. Made with L-BFGS: `susc_check*.json`, `susc_convergence.json` and
`susc_convergence_it*.json`, `susc_stability/PAT13.json`, `PAT14.json`, `PAT23.json` (no suffix),
`t1_alignment.json`, `pat16_topup_compare*.json`, `cohort/PAT23_topup.json`, `cpu_timing.json`,
`modal_cpu_scaling.json`, the `lbfgs` configurations in `susc_held_out/`. Gauss-Newton: files with `gn` in
their name and `gn` configurations - except `susc_stability/PAT16_gn.json`'s first configuration
("iter_scale 3, lam_scale 1"), which is L-BFGS beside the `optimizer gn` one - and
`cohort/PAT25_topup.json` (with the L-BFGS field beside it as `ours_lbfgs`). `topup_ref.json` is FSL's
topup alone. A script run today uses Gauss-Newton; `susc_held_out.py`, `susc_stability.py` and
`susc_convergence.py` take `{"optimizer": "lbfgs"}` configurations, `susc_check.py` does not.

**The labeler**: TractCloud at upstream's inference context (k_global 80) until 2026-10-02, TractCloud at
its trained context (500) briefly, RapidParc since ("RapidParc is the default"). Made with TractCloud:
`cpu_timing.json`, `modal_cpu_scaling.json`, `modal_cpu_field_timing.json`, `modal_cpu_pipeline.json`
(older still: some also predate the repository's split and carry the removed payload's timings),
`hardi_paths_*.json`, `mac_labels.json`, `ukf_labels.json`, `infer_opt*.json`, `dependency_check_tc500.json`,
`pat16_seeding*.json` and `pat16_topup_compare*.json` (their tract shares; 80),
`modal_gpu_pipeline_*_tc500.json` (500), `*_k80.json` and `*_tf32on.json` / `*_c0_tf32off.json` /
`*_c1_tf32off.json` (80; the `_k80` A10 file ran on an A10G). Their timers' label stage is named
`tractcloud`; it is `label` now. The TractCloud experiments (`label_*.json`, `labeler_compare.json`,
`accuracy_tractcloud_test.json`, `memory_batch_*.json`) name their labelers themselves, except
`label_noise_floor.json` and `label_draws.json`: TractCloud at upstream's 80 (their scripts now record it).

**Current** (RapidParc, Gauss-Newton): `cohort/PATnn.json` and `cohort_summary.*` (2026-10-03; the
`_topup` files are above),
`dependency_check.json`, `gpu_pipeline_compare.json`, `modal_gpu_pipeline_{a10,l40s,cpu}.json`,
`trx_check.json`, `rapidparc_check*.json`.

**Labeler-independent** (the tracker, the mask, the kernels): `ukf*.json` except `ukf_labels.json`,
`median_check.json`, `resample.json`.

Some timings were overwritten by later runs of the same script; NOTES keeps the earlier numbers: see
"Corrections" in its 2026-10-03 documentation-review entry.
