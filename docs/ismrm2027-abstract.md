<!-- ISMRM 2027 abstract draft (due 2026-10-28, 23:59 UTC), written 2026-10-19 on branch ismrm2027.
Sources for every number: bench/results/ismrm2027/stats.json (held-out ECE and coverage, 95 % intervals over
subjects), bench/results/calibration_card_masivar_cIII.json, bench/results/calibration_transfer_cIs1.json,
bench/results/NOTES.md (2026-10-07, 2026-10-08, 2026-10-13 "Pooling sessions", Robustness steps 4 and 5),
docs/data-product.md. References and novelty wording: docs/ismrm2027-references.md.
Word counts (script, superscripts and Markdown removed) are at the end of this file. -->

# Calibrated uncertainty for automated tract identification from one diffusion scan, tested on rescans and a phantom

## Impact statement

Tract maps from one diffusion scan can carry probabilities that hold on a real rescan once a protocol-specific
recalibration map, learned from test-retest data, is applied. These probabilities describe reproducibility; anatomical
accuracy needs its own term.

## Synopsis

**Motivation:** Automated tractography reports white matter tracts without saying how they would change on a rescan.

**Goals:** Calibrated uncertainty for tract identification from one diffusion MRI scan.

**Approach:** We ran the whole pipeline (distortion correction, UKF tractography, RapidParc labeling) on 17 bootstrap
replicates, learned a protocol-specific recalibration map from MASiVar test-retest subjects, and tested it
leave-one-subject-out, on another protocol, and on the ISMRM 2015 phantom.

**Results:** The map lowered expected calibration error from 0.07–0.10 to 0.008 and brought tract-envelope coverage to
target. It failed on another protocol; envelopes missed 17–18 points of phantom ground truth.

## Introduction

Automated tract identification from diffusion MRI is read as a measurement, yet on our test-retest protocol a rescan
renames about 35–40 % of streamlines. Bootstrap tractography resamples preprocessed data and refits the local
model;<sup>1–4</sup> whole pipelines have been perturbed numerically.<sup>5,6</sup> To our knowledge, tract-label
probabilities have not been calibrated against real rescans: earlier checks used same-session repeats at the
orientation level<sup>7</sup> or simulated truth.<sup>8,9</sup> We ask whether a whole-pipeline bootstrap is
calibrated for rescans, whether a protocol-specific recalibration map makes it so, and what it says about accuracy.

## Methods

**Data.** MASiVar cohort III (OpenNeuro ds003416):<sup>10</sup> eight healthy adults, one Philips Achieva 3T,
b = 1000 ×40 and 2000 ×56 s/mm², 2.1 × 2.1 × 2.2 mm, 16 b0s, one reversed-phase-encoding b0. All subjects have a
same-session rescan; five have nine other-session rescans. Rescans were registered rigidly to the reference and
tracked on its grid.

**Pipeline.** Per run: the susceptibility field from the b0 pair (topup's model<sup>11</sup>), two-tensor
UKF<sup>12</sup> with ORG-atlas settings<sup>13</sup> at b = 2000, and RapidParc<sup>14</sup> labels (TractCloud's<sup>15</sup>
42 tracts plus Other). The ensemble is the scan plus 16 wild-bootstrap replicates (R = 17), each through the whole
pipeline. Because residuals are correlated within slices, signs are drawn as smooth in-plane patches
(FWHM 4 voxels), after spatially dependent wild bootstraps.<sup>16,17</sup>

**Statements.** Streamline: P(a rescan gives the seed the ensemble's modal tract), stated as its vote share. Presence
envelope: per tract, the highest-vote voxels holding a stated share of the vote mass; coverage is the share of the
rescan's tract inside, pooled over tracts. Point: the main named tract within 8 mm.

**Recalibration map.** Monotone maps (isotonic for vote shares; an inverted coverage curve for envelopes), with
separate same- and other-session terms, evaluated leave-one-subject-out by expected calibration error
(ECE)<sup>18</sup> and coverage at targets 0.8, 0.9 and 0.95; 95 % intervals are bootstrapped over held-out subjects.
Comparators: as stated, and a double-bootstrap map fitted without rescans on a patient scan (PAT16).

**Transfer.** The map was applied to MASiVar cIs1: one subject, another scanner, 2.5 mm,
b = 1000/2000/3000 ×96, 3 forward b0s, with same-session and next-day rescans.

**Phantom.** ISMRM 2015 phantom<sup>19</sup> (b = 1000 ×32), 2023 ROI scoring<sup>20</sup> rebuilt with scilpy 1.5.0:
envelopes of 14 bundles against the ground truth's presence subsampled to each run's streamline count, beside a
bias-free ensemble of 17 ground-truth subsamples.

## Results

The ensemble was overconfident against rescans (Fig. 2). Held out, streamline ECE fell from 0.070 [0.065, 0.075] to
0.008 [0.007, 0.010] (same session) and from 0.097 [0.092, 0.102] to 0.008 [0.005, 0.010] (other session); PAT16's
map gave 0.019 and 0.027. Point ECE fell from 0.038 and 0.059 to 0.014 and 0.022.

As stated, envelopes covered 2–5 points less than their level. With the map, coverage at 0.8/0.9/0.95 was 0.803
[0.797, 0.810], 0.904 [0.898, 0.909], 0.954 [0.949, 0.959] (same session) and 0.802 [0.797, 0.806], 0.902
[0.897, 0.907], 0.952 [0.947, 0.954] (other session); every subject was within 1.2 points below to 1.7 above target
(Figs. 3, 5).

On cIs1 the protocol match refused the map. Applied anyway, it lowered streamline ECE (0.165/0.138 to 0.096/0.043),
but envelopes fell 6–12 points short.

On the phantom, envelopes covered 0.650, 0.724 and 0.778 of the truth at 0.8, 0.9 and 0.95, against 0.818, 0.904 and
0.951 for the bias-free ensemble: 17–18 points short (Fig. 4).

## Discussion

**Precision and accuracy need separate terms.** The map adds a few points of coverage; the phantom's
systematic error is several times larger, and no rescan reveals it. Tractography's systematic errors are
well documented;<sup>19,21,22</sup> here they appear as a coverage gap against ground truth. Each statement must name its
event, a rescan or the anatomy; our accuracy term is provisional and not combined with the map.

**Maps are per protocol.** Most of the correction is within-session overconfidence; the other-session term adds
0.7–1.5 points of coverage, consistent with MASiVar's variability ordering.<sup>10</sup> cIs1 stayed miscalibrated, so
each protocol needs its own test-retest data, and scans without a matching map report run-to-run variability only.

**Limits.** One scanner protocol, healthy adults, eight subjects (five for other sessions), one phantom at b = 1000,
R fixed at 17, statements pooled over tracts.

## Conclusion

A protocol-specific recalibration map learned from eight test-retest subjects makes single-scan tract statements
calibrated for held-out rescans on that protocol. Claims about anatomy need a separate accuracy term.

## Acknowledgements

[PLACEHOLDER: grants, code and data links.] MASiVar data: OpenNeuro ds003416 (CC0).

## Figures

**Preview figure** (no caption): `bench/results/ismrm2027/preview.png`.

**Figure 1** (`fig1_pipeline.png`). Whole-pipeline bootstrap ensemble. One scan and 16 wild-bootstrap replicates with
spatially smooth sign patterns each pass through susceptibility-field estimation, UKF tractography (b = 2000) and
RapidParc labeling (42 tracts plus Other), giving 17 votes per seed, voxel and point. If a protocol-specific
recalibration map matches the scan's protocol, the votes become probabilities for a same-session or other-session
rescan; otherwise the output is labeled run-to-run only.

**Figure 2** (`fig2_streamline_calibration.png`). Held-out reliability of the streamline statement, P(a rescan gives the
ensemble's tract), leave-one-subject-out: (a) same-session rescans, 8 subjects; (b) other-session rescans, 5 subjects.
Thin lines: subjects. As stated (orange): ECE 0.070 and 0.097. With the protocol-specific recalibration map (blue):
0.008 in both. PAT16's double-bootstrap map, fitted without rescans (dashed): 0.019 and 0.027. Dotted: identity.

**Figure 3** (`fig3_envelope_coverage.png`). Presence-envelope coverage of the rescan's tract, pooled over 42 tracts,
held out: (a) same session; (b) other session. Circles: subjects at targets 0.8, 0.9 and 0.95. As stated, envelopes
cover 2–5 points less than their level (0.777/0.869/0.921 and 0.760/0.852/0.906). At the map's levels, coverage is
0.803/0.904/0.954 and 0.802/0.902/0.952, every subject within 1.2 points below to 1.7 above target.

**Figure 4** (`fig4_phantom_gap.png`). ISMRM 2015 phantom, 14 bundles with a label counterpart, 2023 ROI scoring.
Coverage of the ground truth's presence, subsampled to each run's streamline count. The pipeline's 17-run labeled
envelopes (orange) cover 0.650, 0.724 and 0.778 at stated 0.8, 0.9 and 0.95; a bias-free ensemble of ground-truth
subsamples (green) holds its levels (0.818, 0.904, 0.951). Gap: 17–18 points. Dots: bundles. One simulated brain; no
interval.

**Figure 5** (`fig5_cst_envelopes.png`). One subject's corticospinal tract (cIIIs04), from a single scan. Envelopes at
the recalibrated levels for targets 0.8 and 0.95 (stated 0.83 and 0.98, from the fold that excludes this subject)
cover 0.80 and 0.97 of the same-session rescan's tract (yellow outline). Axial and coronal slices. Per-tract coverage
was not validated; shown for illustration.

## References

1. Jones DK, Pierpaoli C. Confidence mapping in diffusion tensor magnetic resonance imaging tractography using a bootstrap approach. Magn Reson Med. 2005;53(5):1143-1149.
2. Whitcher B, Tuch DS, Wisco JJ, et al. Using the wild bootstrap to quantify uncertainty in diffusion tensor imaging. Hum Brain Mapp. 2008;29(3):346-362.
3. Jones DK. Tractography gone wild: probabilistic fibre tracking using the wild bootstrap with diffusion tensor MRI. IEEE Trans Med Imaging. 2008;27(9):1268-1274.
4. Jeurissen B, Leemans A, Jones DK, et al. Probabilistic fiber tracking using the residual bootstrap with constrained spherical deconvolution. Hum Brain Mapp. 2011;32(3):461-479.
5. Kiar G, de Oliveira Castro P, Rioux P, et al. Comparing perturbation models for evaluating stability of neuroimaging pipelines. Int J High Perform Comput Appl. 2020;34(5):491-501.
6. Kiar G, Chatelain Y, de Oliveira Castro P, et al. Numerical uncertainty in analytical pipelines lead to impactful variability in brain networks. PLoS One. 2021;16(11):e0250755.
7. Campbell JSW, MomayyezSiahkal P, Savadjiev P, et al. Beyond crossing fibers: bootstrap probabilistic tractography using complex subvoxel fiber geometries. Front Neurol. 2014;5:216.
8. Consagra W, Ning L, Rathi Y. Neural orientation distribution fields for estimation and uncertainty quantification in diffusion MRI. Med Image Anal. 2024;93:103105.
9. Manzano-Patrón JP, Deistler M, Schröder C, et al. Uncertainty mapping and probabilistic tractography using simulation-based inference in diffusion MRI: a comparison with classical Bayes. Med Image Anal. 2025;103:103580.
10. Cai LY, Yang Q, Kanakaraj P, et al. MASiVar: multisite, multiscanner, and multisubject acquisitions for studying variability in diffusion weighted MRI. Magn Reson Med. 2021;86(6):3304-3320.
11. Andersson JLR, Skare S, Ashburner J. How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging. NeuroImage. 2003;20(2):870-888.
12. Malcolm JG, Shenton ME, Rathi Y. Filtered multitensor tractography. IEEE Trans Med Imaging. 2010;29(9):1664-1675.
13. Zhang F, Wu Y, Norton I, et al. An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. NeuroImage. 2018;179:429-447.
14. Bisten J, von Bornhaupt V, Grün J, et al. RapidParc: a global-context transformer for parallel, accurate, and lesion-robust tractogram parcellation. Imaging Neurosci. 2026;4:IMAG.a.1168.
15. Xue T, Chen Y, Zhang C, et al. TractCloud: registration-free tractography parcellation with a novel local-global streamline point cloud representation. In: MICCAI 2023. Lecture Notes in Computer Science. Springer; 2023:409-419.
16. Shao X. The dependent wild bootstrap. J Am Stat Assoc. 2010;105(489):218-235.
17. Kurisu D, Kato K, Shao X. Gaussian approximation and spatially dependent wild bootstrap for high-dimensional spatial data. J Am Stat Assoc. Published online 2023. doi:10.1080/01621459.2023.2218578
18. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning. PMLR. 2017;70:1321-1330.
19. Maier-Hein KH, Neher PF, Houde JC, et al. The challenge of mapping the human connectome based on diffusion tractography. Nat Commun. 2017;8(1):1349.
20. Renauld E, Théberge A, Petit L, et al. Validate your white matter tractography algorithms with a reappraised ISMRM 2015 Tractography Challenge scoring system. Sci Rep. 2023;13(1):2347.
21. Thomas C, Ye FQ, Irfanoglu MO, et al. Anatomical accuracy of brain connections derived from diffusion MRI tractography is inherently limited. Proc Natl Acad Sci U S A. 2014;111(46):16574-16579.
22. Schilling KG, Nath V, Hansen C, et al. Limits to anatomical accuracy of diffusion tractography using modern approaches. NeuroImage. 2019;185:1-11.

## Counts

Counted with a script (Markdown, comments and superscript numbers removed; words are whitespace-separated tokens, so
numbers and intervals count as words). Limits from the 2027 call, with a margin of about 5 %:

| part | count | limit |
|---|---|---|
| title | 114 characters | about 125 |
| impact statement | 36 words | 40 |
| synopsis (with its four labels) | 95 words | 100 |
| body (Introduction 88, Methods 295, Results 159, Discussion 136, Conclusion 28) | 706 words (711 with the five headings) | 750 |
| Figure 1 caption | 475 characters | 500 |
| Figure 2 caption | 418 characters | 500 |
| Figure 3 caption | 403 characters | 500 |
| Figure 4 caption | 429 characters | 500 |
| Figure 5 caption | 352 characters | 500 |

Notes for submission:
- The figure files still say "card" (Fig. 1 box "Calibration card"; Figs. 2, 3 and the preview legend "With card";
  Fig. 5 title "card-calibrated"). The text says "protocol-specific recalibration map"; the figures need relabeling.
- The other-session rescans are not all later than the reference (cIIIs01's is an earlier session), so the text says
  "other-session", not "later-session".
- The phantom envelopes are at the stated levels (no map exists for the phantom's protocol); the gap is quoted against
  the bias-free ensemble, as in NOTES step 4.
- Point ECE, the cIs1 figures and the phantom figures have no intervals (ranges or single point estimates only).
