# The pipeline

## The pipeline as built (2026-10-03)

`tractline.pipeline`, scan to labeled streamlines, in memory, on the M2 laptop (16 GB). Measured on ds001226 PAT16
(`bench/results/cohort/PAT16.json`; the 12-patient cohort in `bench/results/cohort_summary.md`):

| stage | module | where | PAT16 |
|---|---|---|---|
| susceptibility field from the b0s + reversed pair (topup's model, Gauss-Newton) | `susceptibility.estimate` | GPU, float32; subsampled levels on the CPU | 17.2 s |
| apply it (cubic along phase encoding, Jacobian) | `susceptibility.apply` | CPU, float64 | 1.1 s |
| tracker input (b = 2800 shell, RAS gradients, median_otsu mask) | `prep.prepare`, `mask` | GPU | 3.1 s |
| UKF two-tensor, ORG settings, the binary's seeds (+ 0.2 s loading) | `ukf.track` (Metal) | GPU, float32 steps | 27.3 s |
| RapidParc, one draw (the default labeler) | `labelers.rapidparc.Labeler` | GPU, float32 | 0.7 s |
| **scan → labels** | | | **49.6 s; 42,170 streamlines, 32,264 labeled (40 mm or more), 49.9 % Other (one draw)** |

For scale: FSL topup + applytopup take 665 s for the correction alone on the same machine. Against the
T1, the correction cuts the tumor margin's misplacement from 4.8 mm to 1.6 mm (99th percentile, median
of 12 patients). On the M2's CPU alone the field takes 22.6 s, the same field as the GPU's (0.03 mm at
the brain's 99th percentile). On CUDA (`device="cuda"`: the field in float32 on the GPU, the Triton
tracker, float32 without TF32) the pipeline takes 16.6 s on an A10 and 12.4 s on an L40S, steady state;
on a Modal CPU container (`cpu=32`; 48 CPUs visible to the process) 95 s, the field 41 s of it
(`gpu_pipeline_compare.json`; the Other share's 49.9 % is its M2 run, the cohort's five-draw average 49.8 %).

| Stanford HARDI, track and label (no reversed pair) | M2 (Metal) | L40S | Modal CPU (`cpu=32`) |
|---|---|---|---|
| seconds (UKF) | 105 (94) | 40 (37) | 120 (110) |

(Steady state, `hardi_paths_*.json`; labeled with TractCloud at k_global 80, 1-7 s of the total -
RapidParc takes about 1 s on the M2.)

## Input, and when the correction runs

`tractline.bids` reads a BIDS diffusion series (DICOM through dcm2niix) and finds the series that measure its field:
B0FieldIdentifier/B0FieldSource, else IntendedFor field maps, else a diffusion series in the same folder
phase-encoded the other way. It corrects only when they share the diffusion series' phase-encoding axis, cover
both polarities, were acquired under the same shim (ShimSetting, when stated), with a matching protocol and a
readout time stated or safely assumed; otherwise it tracks the series as acquired and says why. On ds001226 it
hands the pipeline exactly the bench loader's arrays (all 12 patients; PAT03, whose second series is
phase-encoded left-right, is refused with that reason); on ds005123 it refuses all 12 subjects' field maps,
acquired under another shim. After a correction the pipeline reports how much of the pair's difference the
field leaves (`residual_left`), a warning above 0.5 - motion between the series, or series that do not share
one field - never a refusal. The evidence for each check: NOTES 2026-10-03 (absolute polarity irrelevant,
readout time nearly so within a protocol; a re-shimmed pair gave a confident, wrong 10 mm field; the residual
flags every bad pair tested and some moving good ones).

## The labeler

On TractCloud's own test split RapidParc is the most accurate (94.5 % against
TractCloud's 92.0 % at its trained context, 86.6 % at upstream's inference context), the steadiest
(seeds agree on 98 % of tracts) and ~25x faster; on the 12 patients TractCloud at 500 and RapidParc agree
on 83.5 % of streamlines. RapidParc is the default since 2026-10-02; across the M2 (GPU, CPU), an A10,
an L40S and a Modal CPU container PAT16's Other share spans 0.3 points and the tract mix agrees to r >= 0.9998 (NOTES,
"RapidParc is the default"). One TractCloud draw is not a stable label (PAT16's Other share 55.5-66.2 %
over context draws at upstream's 80). The pipeline labels with one draw; called directly, either labeler
averages several draws' probabilities (`labeler(fibers, draws=range(5))`).

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
