# The pipeline

## The pipeline as built (2026-10-02)

`tractline.pipeline`, scan to labeled streamlines, in memory, on the M2 laptop (16 GB). Measured on ds001226 PAT16
(`bench/results/cohort/PAT16.json`; the 12-patient cohort in `bench/results/cohort_summary.md`):

| stage | module | where | PAT16 |
|---|---|---|---|
| susceptibility field from the b0s + reversed pair (topup's model, Gauss-Newton) | `susceptibility.estimate` | GPU, float32; subsampled levels on the CPU | 17.5 s |
| apply it (cubic along phase encoding, Jacobian) | `susceptibility.apply` | CPU, float64 | 1.3 s |
| tracker input (b = 2800 shell, RAS gradients, median_otsu mask) | `prep.prepare`, `mask` | CPU | 0.7 s |
| UKF two-tensor, ORG settings, the binary's seeds | `ukf.track` (Metal) | GPU, float32 steps | 27.0 s |
| TractCloud, one draw | `labelers.tractcloud.Labeler` | GPU, float32 | 4.1 s |
| **scan → labels** | | | **53 s; 32 k labeled streamlines** |

For scale: FSL topup + applytopup take 665 s for the correction alone on the same machine; the whole
Stanford HARDI brain tracks in 135 s on the M2 and 78 s on an A10G (`ukf_triton_block`). Against
the T1, the correction cuts the tumor margin's misplacement from 4.8 mm to 1.6 mm (99th percentile,
median of 12 patients). On the CPU alone the field takes 22.6 s, the same field as the GPU's (0.03 mm
at the brain's 99th percentile); on 32 x86 cores (Modal) 38.5 s, the pipeline 95 s. On CUDA
(`device="cuda"`: the field in float32 on the GPU, the Triton tracker, float32 without TF32) the
pipeline takes 16.5 s on an A10 and 12.2 s on an L40S, steady state.

| Stanford HARDI, track and label (no reversed pair) | M2 (Metal) | L40S | 32 x86 cores |
|---|---|---|---|
| seconds (UKF) | 105 (94) | 40 (37) | 120 (110) |

## Open

- **TractCloud's Other fraction on PAT16 moves with float32 rounding:** 58-63 % across the M2, A10,
  A10G and L40S (float64: 62.2 %); its noise floor on this scan is not measured (NOTES 2026-10-02, "The
  pipeline on CUDA").
- **The field estimate's repeatability.** Gauss-Newton fixed the optimizer's sensitivity (negligible
  input noise: 0.07 mm at the margin's 99th, was 0.33 under L-BFGS) and predicts held-out b0s better on
  all 12 patients; fits from independent pairs of b0s still differ by ~1.3 mm at the brain's edge
  (99th), almost all where the signal is lost at the skull base - the data do not determine the field
  there; stronger regularization trades held-out accuracy for it (NOTES 2026-10-02).
- **PAT23's frontal base.** Under a 104 cm³ meningioma, topup and our correction agree with each
  other and both disagree with the T1 by 4-8 mm in 4 % of the margin; which is wrong is open.
- **Upstream.** Enabling cudnn in TractCloud's own pipeline is a one-line change worth proposing
  to its authors: 3.4×, 2 labels in 440 k.
