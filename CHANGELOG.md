# Changelog

Releases are git tags (`uv add "tractline @ git+https://github.com/mhalle/tractline@vX.Y.Z"`). The experiments
behind each change, with their numbers and later corrections, are in `bench/results/NOTES.md`.

## Unreleased

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
