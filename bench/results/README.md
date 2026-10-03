# What the committed results were made with

`NOTES.md` is the journal: each entry says what was run and what it found, as of its date. The result
files are what those runs wrote. Two defaults changed during the work, so some files were made under an
older one, and re-running their script today writes something different under the same name.

**The field estimate's optimizer**: L-BFGS until 2026-10-02 ("Gauss-Newton is the default" in NOTES),
Gauss-Newton since. Made with L-BFGS: `susc_check*.json`, `susc_convergence.json` and
`susc_convergence_it*.json`, `susc_stability/PAT13.json`, `PAT14.json`, `PAT23.json` (no suffix),
`t1_alignment.json`, `pat16_topup_compare*.json`, `topup_ref.json`, the `lbfgs` configurations in
`susc_held_out/`. Files with `gn` in their name, or `gn` configurations, are Gauss-Newton. A script run
today uses Gauss-Newton unless given `{"optimizer": "lbfgs"}`.

**The labeler**: TractCloud at upstream's inference context (k_global 80) until 2026-10-02, TractCloud at
its trained context (500) briefly, RapidParc since ("RapidParc is the default"). Made with TractCloud:
`cpu_timing.json`, `modal_cpu_scaling.json`, `modal_cpu_field_timing.json`, `modal_cpu_pipeline.json`
(older still: some also predate the repository's split and carry the removed payload's timings),
`hardi_paths_*.json`, `mac_labels.json`, `ukf_labels.json`, `infer_opt*.json`, `dependency_check_tc500.json`,
`modal_gpu_pipeline_*_tc500.json` (500), `*_k80.json` and `*_tf32on.json` / `*_c0_tf32off.json` /
`*_c1_tf32off.json` (80; the `_k80` A10 file ran on an A10G). Their timers' label stage is named
`tractcloud`; it is `label` now. The TractCloud experiments (`label_*.json`, `labeler_compare.json`,
`accuracy_tractcloud_test.json`, `memory_batch_*.json`) name their labelers themselves.

**Current** (RapidParc, Gauss-Newton): `cohort/*.json` and `cohort_summary.*` (2026-10-03),
`dependency_check.json`, `gpu_pipeline_compare.json`, `modal_gpu_pipeline_{a10,l40s,cpu}.json`,
`trx_check.json`, `rapidparc_check*.json`.

**Labeler-independent** (the tracker, the mask, the kernels): `ukf*.json`, `median_check.json`,
`resample.json`, `pat16_seeding*.json`.
