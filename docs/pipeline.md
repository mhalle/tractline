# The pipeline

## The pipeline as built (2026-10-02)

`tractline.pipeline`, scan to labeled streamlines, in memory, on the M2 laptop (16 GB). Measured on ds001226 PAT16
(`bench/results/cohort/PAT16.json`; the 12-patient cohort in `bench/results/cohort_summary.md`):

| stage | module | where | PAT16 |
|---|---|---|---|
| susceptibility field from the b0s + reversed pair (topup's model, Gauss-Newton) | `susceptibility.estimate` | GPU, float32; subsampled levels on the CPU | 17.2 s |
| apply it (cubic along phase encoding, Jacobian) | `susceptibility.apply` | CPU, float64 | 1.1 s |
| tracker input (b = 2800 shell, RAS gradients, median_otsu mask) | `prep.prepare`, `mask` | GPU | 3.1 s |
| UKF two-tensor, ORG settings, the binary's seeds (+ 0.2 s loading) | `ukf.track` (Metal) | GPU, float32 steps | 27.3 s |
| RapidParc, one draw (the default labeler) | `labelers.rapidparc.Labeler` | GPU, float32 | 0.7 s |
| **scan → labels** | | | **49.6 s; 42,170 streamlines, 32,264 labeled (40 mm or more), 49.9 % Other** |

For scale: FSL topup + applytopup take 665 s for the correction alone on the same machine. Against the
T1, the correction cuts the tumor margin's misplacement from 4.8 mm to 1.6 mm (99th percentile, median
of 12 patients). On the M2's CPU alone the field takes 22.6 s, the same field as the GPU's (0.03 mm at
the brain's 99th percentile). On CUDA (`device="cuda"`: the field in float32 on the GPU, the Triton
tracker, float32 without TF32) the pipeline takes 16.6 s on an A10 and 12.4 s on an L40S, steady state;
on a Modal CPU container (48 logical x86 cores) 95 s, the field 41 s of it (`gpu_pipeline_compare.json`).

| Stanford HARDI, track and label (no reversed pair) | M2 (Metal) | L40S | 32 x86 cores |
|---|---|---|---|
| seconds (UKF) | 105 (94) | 40 (37) | 120 (110) |

(Steady state, `hardi_paths_*.json`; labeled with TractCloud at k_global 80, 1-7 s of the total -
RapidParc takes about 1 s on the M2.)

## The labeler

On TractCloud's own test split RapidParc is the most accurate (94.5 % against
TractCloud's 92.0 % at its trained context, 86.6 % at upstream's inference context), the steadiest
(seeds agree on 98 % of tracts) and ~25x faster; on the 12 patients TractCloud at 500 and RapidParc agree
on 83.5 % of streamlines. RapidParc is the default since 2026-10-02; across the M2 (GPU, CPU), an A10,
an L40S and x86 CPUs PAT16's Other share spans 0.3 points and the tract mix agrees to r >= 0.9998 (NOTES,
"RapidParc is the default"). One TractCloud draw is not a stable label (PAT16's Other share 55.5-66.2 %
over context draws at upstream's 80); `draws=` averages several draws' probabilities for either labeler.

## Open

- **The field estimate's repeatability.** Gauss-Newton fixed the optimizer's sensitivity (negligible
  input noise: 0.07 mm at the margin's 99th, was 0.33 under L-BFGS) and predicts held-out b0s better on
  all 12 patients; fits from independent pairs of b0s still differ by ~1.3 mm at the brain's edge
  (99th), almost all where the signal is lost at the skull base - the data do not determine the field
  there; stronger regularization trades held-out accuracy for it (NOTES 2026-10-02).
- **PAT23's frontal base.** Under a 104 cm³ meningioma, topup and our correction agree with each
  other and both disagree with the T1 by 4-8 mm in 4 % of the margin; which is wrong is open.
- **Upstream.** Enabling cudnn in TractCloud's own pipeline is a one-line change worth proposing
  to its authors: 3.4×, 2 labels in 440 k.
