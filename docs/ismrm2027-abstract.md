<!-- ISMRM 2027 abstract draft (due 2026-10-28, 23:59 UTC), branch ismrm2027; revised 2026-10-19 after an independent
review. Sources for every number: bench/results/ismrm2027/stats.json (held-out ECE and coverage, 95 % intervals over
subjects), bench/results/calibration_card_masivar_cIII.json, bench/results/calibration_transfer_cIs1.json,
bench/results/calibration_learning_curve.json, bench/results/NOTES.md (2026-10-07, 2026-10-08, 2026-10-13 "Pooling
sessions", Robustness steps 4 and 5), docs/data-product.md. References and novelty wording:
docs/ismrm2027-references.md. Word counts (script; superscripts, figure tags and Markdown removed) are at the end. -->

**Open choices for the user and co-authors:** (1) the final title (the one below is the reviewer's first alternative);
(2) which 5 figures to take from the figure pool at the end, then numbering the [F-…] tags in the body; (3) the
preview figure (preview_coverage or preview_streamline); (4) the categories, proposed 710 Diffusion: Tractography (primary) and 709 Diffusion:
Simulation/Validation (secondary); (5) the code and data link in the Acknowledgements.

# Rescan-calibrated uncertainty for automated tract identification from one diffusion scan, and its limits on a phantom

## Impact statement

From one diffusion scan, tract envelopes can state how much of a rescan's tract they will cover, once a
protocol-specific recalibration map learned from test-retest data is applied. This calibrates reproducibility only;
anatomical accuracy needs a separate term.

## Synopsis

**Motivation:** Automated tractography does not state how tracts change on rescans.

**Goals:** One-scan tract identification with uncertainty calibrated for a rescan.

**Approach:** We ran UKF tractography and RapidParc labeling on the scan and 16 bootstrap replicates (R = 17), fitted a
protocol-specific recalibration map on MASiVar test-retest data, and tested it leave-one-subject-out, on another
protocol and a phantom.

**Results:** Held out by subject, mapped tract envelopes met their rescan coverage targets within 0.5 points but did not
transfer across protocols. On a phantom, unmapped envelopes covered 17–18 points less count-matched truth than a
bias-free ensemble.

## Introduction

Automated tract identification is read as a measurement, yet on this protocol a rescan renames 35–40% of streamlines.
Bootstrap tractography resamples preprocessed data and refits the local model;<sup>1–4</sup> whole pipelines have been
perturbed numerically.<sup>5,6</sup> To our knowledge, no study has propagated data-noise resampling through field
re-estimation, tracking and labeling, or calibrated tract labels against real rescans; earlier checks used same-session
repeats at the orientation level<sup>7</sup> or simulated truth.<sup>8,9</sup> We do both and test on a phantom.

## Methods

**Data.** MASiVar cohort III (OpenNeuro ds003416):<sup>10</sup> eight healthy adults on one Philips Achieva 3T; b = 1000
×40 and 2000 ×56 s/mm²; 2.1×2.1×2.2 mm; 16 b0s and one reversed-phase-encoding b0. All have same-session rescans; five
have nine other-session rescans.

**Pipeline.** Each run estimates the susceptibility field from the b0 pair (topup's model<sup>11</sup>), tracks with
two-tensor UKF<sup>12</sup> (ORG-atlas settings<sup>13</sup>) at b = 2000, and labels with RapidParc<sup>14</sup> into
TractCloud's<sup>15</sup> 42 tracts plus Other. The ensemble runs the whole pipeline on the scan and 16 wild-bootstrap
replicates (R = 17). Residuals are correlated within slices, so signs are smooth in-plane patches (FWHM 4 voxels),
following the spatially dependent wild bootstrap.<sup>16,17</sup>

**Ensemble statements.** The primary statement is a presence envelope: per tract, the highest-vote voxels holding a
stated share of the vote mass (presence: ≥0.3 streamlines per mm²). Its coverage is the share of the rescan's tract
inside, pooled over tracts. Two diagnostics: the probability that a rescan gives a seed the ensemble's modal tract, and
the main named tract within 8 mm of points on a 4-voxel lattice. "As stated" means the raw vote share.

**Recalibration map.** Monotone stated-to-observed maps, separate for same- and other-session rescans, were evaluated
leave-one-subject-out by coverage at targets 0.8, 0.9 and 0.95 and by expected calibration error<sup>18</sup> (ECE;
seven stated-probability bins). 95% intervals resample held-out subjects, without refitting folds. A learning curve
refits it on k training subjects.

**Transfer.** MASiVar cIs1: one subject, another scanner, 2.5 mm, b = 1000/2000/3000 ×96, 3 forward b0s, tracked at b =
3000; same-session and next-day rescans.

**Phantom.** ISMRM 2015 phantom<sup>19</sup> (b = 1000 ×32), 2023 ROI scoring<sup>20</sup> rebuilt from scilpy 1.5.0.
Envelopes of 14 bundles at stated levels (no map exists for this protocol) were compared with ground-truth presence at
each run's streamline count and with a bias-free ensemble of 17 ground-truth subsamples.

## Results

As stated, envelopes fell 2–5 points short. Held out by subject, the map brought coverage at 0.8/0.9/0.95 to 0.803
[0.797, 0.810], 0.904 [0.898, 0.909], 0.954 [0.949, 0.959] (same session) and 0.802 [0.797, 0.806], 0.902 [0.897,
0.907], 0.952 [0.947, 0.954] (other session), every subject within −1.2 to +1.7 points [F-coverage, F-cst]. From k = 1,
coverage error was +0.002 to +0.005 (as stated, −0.023 to −0.048) [F-learning].

Diagnostics: streamline ECE fell from 0.070 [0.065, 0.075] to 0.008 [0.007, 0.010] (same session) and from 0.097 [0.092,
0.102] to 0.008 [0.005, 0.010] (other session) [F-streamline]; point ECE fell from 0.038 and 0.059 to 0.014 and 0.022.
Unanimous streamlines kept their label on only 91–93% of rescans.

On cIs1, the map partly improved streamline ECE (0.165 → 0.096, 0.138 → 0.043), but envelopes stayed 6–12 points short.

On the phantom, envelopes at 0.8/0.9/0.95 covered 0.650/0.724/0.778 of the truth, 17–18 points below the bias-free
ensemble's 0.818/0.904/0.951 [F-phantom]. Truth-assisted, scorer-assigned envelopes fell 12–13 points short of their own baseline, so part of the gap is label mapping.

## Discussion

**Precision and accuracy need separate terms.** The phantom's systematic error, invisible to rescans, is several times
the map's few-point correction. Such errors are well documented;<sup>19,21,22</sup> we measure them as a coverage gap,
kept separate from the map. Each statement names its event: rescan or anatomy.

**Maps are per protocol.** Most correction is within-session; the other-session term adds 0.7–1.5 points, consistent
with MASiVar.<sup>10</sup> On cIs1 (one subject, no interval), protocol, scanner and tracked shell are confounded. One
training subject removes most miscalibration, so on this protocol a handful of volunteers suffices.

**Limits.** One protocol; healthy adults; 5–8 subjects per interval; one phantom at b = 1000; R fixed; only rescans are
resampled; streamline events need a rescan streamline from the seed (95–99%); large tracts dominate coverage; replicates
carry ~1.5× the scan's noise variance.

## Conclusion

Held out by subject, a protocol-specific recalibration map makes single-scan tract envelopes cover a rescan's tract at
their stated levels on this protocol. Anatomical claims need a separate accuracy term.

## Acknowledgements

[PLACEHOLDER: grants; code and data links.] MASiVar data: OpenNeuro ds003416 (CC0).

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

## Figure pool

Up to five are chosen from these, plus the preview. Each caption is at most 500 characters. Files are in
bench/results/ismrm2027/; the captions describe the figures as being relabeled.

**[F-pipeline]** (`fig1_pipeline.png`). The scan and 16 wild-bootstrap replicates (smooth sign patches, FWHM 4 voxels)
each pass through susceptibility-field estimation, UKF tractography at b = 2000 and RapidParc labeling (42 tracts plus
Other): R = 17 votes per seed, voxel and point. If a protocol-specific recalibration map, validated leave-one-subject-out
on 8 test-retest subjects, matches the scan's protocol, votes become rescan probabilities; otherwise the output is
run-to-run only.

**[F-streamline]** (`fig2_streamline_calibration.png`). A diagnostic: held-out reliability of P(a rescan gives the seed
the ensemble's tract), leave-one-subject-out, for (a) same-session rescans, 8 subjects, and (b) other-session rescans, 5
subjects. Thin lines: subjects. As stated (orange): ECE 0.070 and 0.097. With the map (blue): 0.008 in both. Gray: a
double-bootstrap map fitted on a patient scan with no rescan (PAT16), 0.019 and 0.027. Dotted: identity.

**[F-coverage]** (`fig3_envelope_coverage.png`). Held-out coverage of the rescan's tract by presence envelopes, pooled
over 42 tracts, against target coverage: (a) same session; (b) other session. Circles: subjects at targets 0.8, 0.9
and 0.95. As stated, envelopes cover 2–5 points less than their target (0.777/0.869/0.921 and 0.760/0.852/0.906). At
the map's levels, coverage is 0.803/0.904/0.954 and 0.802/0.902/0.952, every subject within −1.2 to +1.7 points.

**[F-phantom]** (`fig4_phantom_gap.png`). ISMRM 2015 phantom, 14 bundles, 2023 ROI scoring. Coverage of ground-truth
presence subsampled to each run's streamline count. Envelopes at stated levels (no map applies to this protocol;
orange) cover 0.650, 0.724 and 0.778 at 0.8, 0.9 and 0.95; a bias-free ensemble of ground-truth subsamples (green)
holds its levels (0.818, 0.904, 0.951). Part of the 17–18-point gap is label mapping. Dots: bundles. One simulated
brain; no interval.

**[F-cst]** (`fig5_cst_envelopes.png`). One subject's corticospinal tract (cIIIs04), from a single scan, axial and
coronal. Envelopes at the map's levels for targets 0.8 and 0.95 (stated 0.83 and 0.98, from the fold that excludes this
subject) cover 0.80 and 0.97 of the same-session rescan's tract (yellow outline). One tract, for illustration;
per-tract coverage was not validated.

**[F-learning]** (`learning_curve.png`). Held-out calibration against the number of training subjects k, the map
refitted on every set of k of the other subjects; k = 0 is as stated. Same session, k = 1 to 7; other session, k = 1 to
4. Envelope coverage error at targets 0.8, 0.9 and 0.95 is +0.002 to +0.005 from k = 1 on (as stated, −0.023 to
−0.048). Streamline ECE falls from 0.011 at k = 1 to 0.008. Lines: means; bands: 10th–90th percentiles over held-out
subjects and training sets.

**[F-preview]** (`preview_coverage.png` or `preview_streamline.png`; ISMRM previews carry no caption). Held-out
envelope coverage of a rescan's tract (the headline), or the streamline reliability diagnostic, as stated and with
the protocol-specific recalibration map.

## Counts

Counted with a script (Markdown, comments, superscript numbers and [F-…] tags removed; words are whitespace-separated
tokens, so numbers and intervals count as words). Limits from the 2027 call; target margin about 5 %:

| part | count | limit |
|---|---|---|
| title | 117 characters | about 125 |
| impact statement | 38 words | 40 |
| synopsis (with its four labels) | 95 words | 100 |
| body (Introduction 76, Methods 303, Results 166, Discussion 132, Conclusion 30) | 707 words (712 with the five headings) | 750 |
| captions: F-pipeline 451, F-streamline 406, F-coverage 415, F-phantom 437, F-cst 343, F-learning 448, F-preview 164 characters | | 500 each |

Notes:
- ECE uses seven bins of stated probability (edges 0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1; calibration_card.ECE_BINS),
  not bins by vote count; the map itself is fitted per vote count (k = 1–17).
- The learning-curve coverage error with the map is +0.0015 to +0.0045 over every k, term and target (written +0.002 to
  +0.005); its point ECE eases from 0.018 to 0.014 (same session), not quoted for space.
- "Other-session", not "later-session": cIIIs01's other-session rescan is from an earlier session.
- The cIs1 refusal is not reported as a finding: it follows from the protocol check by construction.
- PAT16's double-bootstrap map (an R = 11 vote table, fitted on a patient with no rescan) is now only in the
  F-streamline caption.
- No 95 % intervals for point ECE, the learning curve, cIs1 or the phantom (single estimates or ranges only).
