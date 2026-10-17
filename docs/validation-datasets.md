# Validation datasets with a known ground truth

Survey, 2026-10-17; not yet independently reviewed. It serves step 5 of `docs/robustness-plan.md` ("Accuracy").

The robustness work measures agreement with real rescans (MASiVar). A tracker that agrees with itself can still be wrong in
the same way every time. This page lists the tractography challenges and benchmarks that have a ground truth, says what each
ground truth is, and judges which ones tractline's stages can be scored on.

**Sources.** Facts come from the primary papers (PubMed Central full text), the challenge sites and the data repositories
(Zenodo, Mendeley Data, OpenNeuro, Synapse), read on 2026-10-17. Links marked "live" answered HTTP 200 that day; that does
not mean the files were downloaded. Anything not confirmed in a primary source is marked **UNVERIFIED**.

**Already cited.** `docs/prior-art-uncertainty.md` cites Maier-Hein 2017 (ref. 64) and mentions Grisot 2021. Nothing else
here appears in `docs/prior-art.md` or `docs/prior-art-uncertainty.md`.

## Summary table

"Rev-PE" means a reversed phase-encoding b0, or a field map. In the last four columns, "yes" means the stage can be scored
against that ground truth, "partly" means with caveats, and "no" means it cannot.

| Benchmark (year) | Ground truth | Data | Rev-PE | Metrics, code | Access today | UKF | 42-tract labeler | Field correction | Ensemble calibration |
|---|---|---|---|---|---|---|---|---|---|
| **ISMRM 2015 Tractography Challenge** | 25 bundles hand-segmented from an HCP tractogram, simulated with Fiberfox | human-brain geometry; 2 mm, b = 1000, 32 directions; SNR ≈ 20; spikes, ghosts, motion, B0 distortion | **yes**: an AP b0 for the PA series, plus a field map | VS/IS/VB/IB, overlap, overreach, F1; scilpy scripts (ROI-based 2023 scoring) | Zenodo, CC BY 4.0, open; Tractometer live | **yes** | **partly**: about 8 of the 42 tracts have a counterpart; AF is a known *false* bundle | **yes**, and the field map can score the field itself (UNVERIFIED that it is the simulated field) | **yes**: voxel envelopes against the bundle masks; Fiberfox can redraw the noise |
| **DiSCo** (CDMRI/MICCAI 2021) | 12,196 simulated axon-like strands between 16 ROIs; Monte Carlo signal | 40³ voxels of a 1 mm³ substrate; four shells (1000–13191), 90 directions each, 4 b0s; Rician SNR 10–50 | no; no artifacts | Pearson r of connectivity matrices, AUC of binary connectivity, valid-weight fraction | Mendeley Data v2, CC BY 4.0, open | **yes**, after rescaling the header (v2 ships 1 mm and 2 mm versions) | no: not brain anatomy | no | **yes**: connection presence over 3 phantoms × 120 ROI pairs × 5 SNRs |
| **IronTract** (MICCAI 2019, rounds to 2021) | tracer (Lucifer Yellow, Fluorescein) in two macaques, same brains as the dMRI | ex vivo, 4.7 T, 0.7 mm; 1 b0 + 514 DWIs (Cartesian, b ≤ 40,000); HCP-like b = 6000/12000 resampled | no: Fomblin, no susceptibility | partial ROC AUC (FPR ≤ 0.3) over several thresholds; round-2 scripts on GitHub | registration, QMENTA platform; rounds "completed" (current access UNVERIFIED) | partly (ex vivo, macaque) | no | no | **yes**: thresholded presence maps give the ROC directly |
| **3D VoTEM** (ISBI 2018) | macaque tracer atlas; squirrel-monkey BDA tracer (histology); a physical phantom with traced bundles | macaque 0.25 mm, b = 4900; squirrel monkey 0.3 mm, b = 1200, 31 directions; phantom 2.25 × 2.25 × 2.5 mm, b = 1000/2000, 96 directions, 7 repeats on 2 scanners | **phantom: AP and PA**; animals ex vivo, none stated | sensitivity, specificity, Youden; overlap, overreach, Dice | Synapse teams (registration); site live; "freely available" per the paper | partly (phantom yes; animals ex vivo) | no | **phantom: yes** (real scanner pairs) | **yes**: phantom repeats are real rescans with known bundles |
| **Fiber Cup** (MICCAI 2009) and the Tractometer (2013) | physical phantom of acrylic fibers, one coronal-like slab, 16 connections | 3 mm, 3 slices of 64 × 64; b = 650/1500/2000, 64 directions; 2 repeats; DWI SNR 9.1/2.6/1.1 | not described | Fillard: 16 seeds, distances; Côté: VC/IC/NC/VB/IB/ABC | DWIs at tractometer.org (live); original site gone (HTTP 410) | partly (3 slices) | no | no | weak |
| **HARDI reconstruction** (ISBI 2012, 2013) | simulated orientations (2013: a Phantomas-style phantom, ground-truth peaks) | 2013: 50³ volume; DTI 32 dirs b = 1200, HARDI 64 dirs b = 3000, DSI 515; SNR 10/20/30 | no | per-voxel fiber count and angular error; `compute_local_metrics.py` | 2013 site live | orientations only | no | no | orientations only |
| **TractoInferno** (2022, silver standard) | RecobundlesX bundles from an ensemble of 4 trackers, manually QC'd | 284 human scans, 6 sites, 3 T, single shell ≈ b = 1000 | not all subjects (topup skipped) | bundle volume against the reference | OpenNeuro ds003900, CC0 | agreement only | agreement only (30 bundles incl. AF, UF, PYT) | no | no |

Searches found no new tractography ground-truth challenge at MICCAI, ISBI or ISMRM from 2022 to 2026 (see "Newer
challenges"). That is a search result, not proof that none exists.

## ISMRM 2015 Tractography Challenge

**Paper.** Maier-Hein KH et al. The challenge of mapping the human connectome based on diffusion tractography. *Nat Commun*
2017;8:1349. doi:10.1038/s41467-017-01285-x. Rescoring: Renauld E, Théberge A, Petit L, Houde J-C, Descoteaux M. Validate
your white matter tractography algorithms with a reappraised ISMRM 2015 Tractography Challenge scoring system. *Sci Rep*
2023;13:2347. doi:10.1038/s41598-023-28560-w.

**Ground truth.**
- An expert segmented 25 bundles by hand from a whole-brain global tractogram of one HCP subject. The bundles cover more
  than 70 % of the white matter (Zenodo record text).
- The bundles: CC, cingulum, fornix, anterior and posterior commissures, optic radiation, ICP, MCP, SCP, POPT, CST, FPT,
  ILF, UF and SLF, with left and right where they apply.
- Each tract was smoothed by simulating it alone with Fiberfox and re-tracking it.
- The DWIs were then simulated from the bundles with Fiberfox's four-compartment model (Neher et al., *MRM*
  2014;72:1460–70, doi:10.1002/mrm.25045).
- **There is no arcuate fasciculus in the ground truth.** The AF, IFOF, FAT, MdLF, SFOF and VOF were *invalid* bundles found
  by over 80 % of submissions. The paper says the AF "was generated from ILF and SLF crossing streamlines".
- 2023 changes:
  - looping and broken streamlines were removed from the ground truth;
  - CST, POPT and FPT were merged into one "brainstem projection system", leaving **21 scored bundles**;
  - ILF and OR were thinned to separate them;
  - segmentation for scoring uses hand-drawn ROIs instead of RecoBundles.

**Data.**

| Version (Zenodo) | Contents | Distortion data |
|---|---|---|
| Challenge data, [10.5281/zenodo.572345](https://doi.org/10.5281/zenodo.572345) | 90 × 108 × 90 at 2 mm isotropic; one b0 + 32 directions at b = 1000; T1-like image. Artifacts: complex Gaussian noise (SNR ≈ 20 against white-matter b0), 10 spikes, N/2 ghosts, motion in volumes 6, 12 and 24, and B0 distortion from a real field map registered to the subject. PE posterior-anterior; TE 108 ms; dwell time 1 ms | `ISMRM_2015_Tracto_challenge_data_with_reversed_phase_v1_0.zip` holds `Diffusion_WITH_REVERSEPHASE.nii.gz` (the AP b0 is the second volume, per the Tractometer processed-data page), `fmap_Hz.nii.gz` and `fmap_RadPerSec.nii.gz` |
| b = 3000 version, [10.5281/zenodo.1007149](https://doi.org/10.5281/zenodo.1007149) | one b0 + 64 directions; "other simulation parameters are the same" | `param.ffp_FMAP.nrrd`; no reversed b0 listed |
| High-resolution version, [10.5281/zenodo.579933](https://doi.org/10.5281/zenodo.579933) | 1.25 mm; b = 1000/2000/3000, 90 directions each, 18 b0s | README states no artifacts |

- Also on Zenodo:
  - the ground-truth bundles (TRK, TCK, VTK);
  - an artifact-free ground-truth DWI;
  - `FilesForSimulation_v1.zip`, the inputs to re-run Fiberfox.
- Submissions: [10.5281/zenodo.840086](https://doi.org/10.5281/zenodo.840086), 96 tractograms, about 39 GB.
- **Caveat.** The Tractometer DWI page warns that the gradient x components are wrongly signed for some tools ("flipped
  ODFs", visible in the CC).
- **UNVERIFIED:**
  - the total readout time needed for a topup-style correction, which neither page states (only the 1 ms dwell time is
    given);
  - whether `fmap_Hz` is exactly the field used in the simulation.

**Metrics.**
- Original:
  - valid and invalid connections (VC, IC) and no connections (NC);
  - valid and invalid bundles (VB, IB);
  - overlap (OL), the share of ground-truth voxels reached;
  - overreach (OR), voxels outside the bundle, normalized by bundle volume.
- 2023 renames VC to "valid streamlines" (VS) and IC + NC to IS, and adds F1 (Dice).
- Code:
  - the 2023 ROI scoring is `scil_score_ismrm_Renauld2023.sh` on tractometer.org, which calls scilpy's
    `scil_score_tractogram.py`, `scil_score_bundles.py` and `scil_tractogram_math.py` (scilpy 1.5.0 recommended);
  - the newer equivalent is `scil_tractogram_segment_with_ROI_and_score`;
  - ROIs and masks: `scoring_data_Renauld2023.zip` at scil.usherbrooke.ca, 241 MB, live;
  - the 2015 RecoBundles scorer: github.com/scilus/ismrm_2015_tractography_challenge_scoring
    ([10.5281/zenodo.810130](https://doi.org/10.5281/zenodo.810130)).

**Access.**
- All Zenodo records are open under CC BY 4.0; the scil.usherbrooke.ca mirrors are live.
- The old `tractometer.org/ismrm_2015_challenge/` URL is gone (404); the pages now live under `tractometer.org/ismrm2015/`.

**Findings.**
- 96 submissions came from 20 groups.
- Mean 21 of 25 valid bundles were found (median 23).
- Submissions held a mean of 88 ± 58 invalid bundles, more than four times the valid ones. 41 of these occurred in most
  submissions.
- On average 54 ± 23 % of streamlines were valid; OL was 36 ± 16 % and OR 29 ± 26 %.
- Bundle-level precision was 23 ± 9 % and recall 85 ± 15 %.
- Tracking on the ground-truth orientation field itself still made invalid bundles, so the ambiguity lies in orientation
  data, not in noise.
- Renauld 2023 confirmed these conclusions with the new scoring:
  - scoring the ground truth against itself gives Dice 92 %, not 100 %;
  - a TractoFlow run with topup scored better than every 2015 entry. The site's erratum gives F1 up to 58 %, not the
    paper's 64–65 %;
  - without the reversed b0, the same run's Dice fell to 57 %.

**Fit for tractline.**
- **UKF:**
  - Yes. b = 1000 with 32 directions is a clinical single shell the two-tensor model handles, but it is far from MASiVar's
    b = 2800 shell.
  - The b = 3000 / 64-direction variant is closer, but it has no reversed b0.
  - Scoring is independent of our labeler: the ROI scorer assigns streamlines to the ground-truth bundles itself.
- **Labeler:**
  - Partly. The anatomy is one HCP subject, the labelers' own training domain, in a plausible orientation.
  - Ground-truth bundles with an ORG counterpart:

    | Ground truth | ORG tracts |
    |---|---|
    | CST | CST (merged into the brainstem projection system in 2023, so compare against the 2015 masks) |
    | UF | UF |
    | ILF | ILF |
    | cingulum | CB |
    | SLF | SLF-I to III |
    | CC | CC1 to CC7 |
    | ICP, MCP | ICP, MCP |

  - Fornix, commissures, OR and SCP have no ORG tract.
  - FPT and POPT map only loosely (CR-F, CR-P, CPC).
  - Every streamline labeled AF, EC, EmC, IOFF, MdLF or superficial is a false positive by construction. That makes a sharp
    test of the "reproducibly wrong" question: how much AF does tractline produce, and how stable is it?
- **Field correction:**
  - Yes. A PA series with an AP b0, the same axis and polarity pair the BIDS reader requires. The readout time must be
    supplied (UNVERIFIED value).
  - The provided field map could score our field against the truth (UNVERIFIED that it is the simulated field).
  - The data also has motion, which `docs/pipeline.md` lists as not corrected, and spikes and ghosts, for which the
    pipeline has no step.
- **Ensemble calibration:**
  - Yes:
    - per-voxel presence envelopes for the mappable tracts against the ground-truth masks;
    - the ensemble's per-streamline confidence against VS/IS.
  - With `FilesForSimulation_v1.zip` and Fiberfox (MITK-Diffusion, GitHub, last push 2026-02), new noise draws of the same
    phantom would act as simulated rescans with a known truth. That combines the MASiVar question with accuracy. It is
    UNVERIFIED that the current Fiberfox reproduces the 2015 simulation exactly.

## DiSCo: Diffusion-Simulated Connectivity (CDMRI/MICCAI 2021)

**Papers.**
- Rafael-Patino J, Girard G, Truffet R, Pizzolato M, Caruyer E, Thiran J-P. The diffusion-simulated connectivity (DiSCo)
  dataset. *Data Brief* 2021;38:107429. doi:10.1016/j.dib.2021.107429.
- Girard G, Rafael-Patiño J, et al. Tractography passes the test: Results from the diffusion-simulated connectivity (DiSCo)
  challenge. *NeuroImage* 2023;277:120231. doi:10.1016/j.neuroimage.2023.120231.

**Ground truth.**
- Each phantom is 12,196 tubular strands with diameters of 1.4–4.2 µm. They were generated with the Numerical Fiber
  Generator, meshed, and connect 16 ROIs on the surface of a 1 mm sphere.
- The signal comes from Monte Carlo spin simulation (MC/DC, 10⁹ particles).
- Provided:
  - the strand centerlines (TRK/TCK) with their diameters and end-ROI pairs;
  - 16 × 16 connectivity matrices (strand count and cross-sectional area);
  - intra-strand volume fraction, ROI labels and the mesh.
- The challenge used three phantoms (training, validation, test). About 22 % of the 120 ROI pairs are connected.

**Data.**
- 40 × 40 × 40 voxels; 25 µm physical voxels, with 50 µm also provided.
- 4 b0s + 360 DWIs, 90 directions on each of b = 1000, 1925, 3094 and 13191 s/mm².
- Noiseless, plus Rician noise at SNR 10, 20, 30, 40 and 50.
- No artifacts, no reversed PE, no field map. The challenge paper lists artifacts and clinical resolutions as future work.

**Metrics.**
- Teams were ranked by Pearson r between the submitted and true connectivity matrices.
- Also reported: the fraction of valid connectivity weight, and ROC AUC on binary connectivity.
- The training phantom shipped with an evaluation script (challenge data page).

**Access.**
- Mendeley Data [10.17632/fgf86jdfg6.2](https://doi.org/10.17632/fgf86jdfg6.2) (v2, 2022-01-31, CC BY 4.0, open; three
  phantoms; images also given at 1 mm and 2 mm "isotropic"). v1 is 10.17632/fgf86jdfg6.1.
- Challenge description: [10.5281/zenodo.4733450](https://doi.org/10.5281/zenodo.4733450) (CC BY-ND 4.0).
- The hardi.epfl.ch challenge page asked for registration in 2021; the Mendeley copy needs none.

**Findings.**
- 14 teams submitted 111 matrices.
- Each team's best r ranged 0.874–0.973 (mean 0.950).
- The top three by r all used parallel transport tractography; the most accurate binary result used deterministic RK4 with
  radial DSI.
- Specific false-positive and false-negative connections recurred across all methods.
- The authors say the substrate is simpler than a brain: two teams had no false positives at all.

**Fit for tractline.**
- **UKF:** Yes. Use the b = 3094 shell (90 directions), close to our b = 2800, and the 1 mm header version from v2.
  - The ORG stopping thresholds assume in vivo brain FA and diffusivity; DiSCo's are different. Thresholds may need a check
    (UNVERIFIED how UKF behaves here).
- **Labeler:** No. A sphere of 16 ROIs is not brain anatomy.
- **Field correction:** No. There is no distortion and no reversed PE.
- **Ensemble calibration:** Yes, and cheaply.
  - Score connection presence: 3 phantoms × 120 pairs × 5 SNRs.
  - Score voxel presence against strand-density maps.
  - The volume is 64,000 voxels, so a 17-run ensemble costs seconds.

## IronTract (MICCAI 2019; round 2 results at ISMRM 2021)

**Paper.** Maffei C, Girard G, Schilling KG, et al. Insights from the IronTract challenge: Optimal methods for mapping brain
pathways from multi-shell diffusion MRI. *NeuroImage* 2022;257:119327. doi:10.1016/j.neuroimage.2022.119327.

**Ground truth.**
- Anterograde/bidirectional tracer, dextran-conjugated, in two male rhesus macaques, in the same brains as the dMRI:
  - training: Lucifer Yellow in the frontal pole;
  - validation: Fluorescein in ventrolateral prefrontal cortex.
- Tissue was cut in 50 µm sections. Tracer was charted along the axon trajectories, not only at the endpoints.

**Data.**
- Ex vivo on a 4.7 T Bruker, 3D EPI, 0.7 mm isotropic, 96 × 96 × 112.
- 1 b0 + 514 DWIs on a Cartesian q-space grid, b ≤ 40,000 s/mm²; 48 h per brain.
- Resampled by NUFFT to the HCP lifespan shells, scaled ×4 for fixed tissue (b = 6000 and 12000).
- SNR at b0: 64 (training) and 51 (validation).
- The brains were in Fomblin "to eliminate susceptibility artifacts". No reversed PE.

**Metrics.**
- Voxelwise TPR and FPR against the tracer mask, at several thresholds per team.
- Ranked by partial AUC over FPR 0–0.3 (maximum 0.3), in two categories: any sampling scheme, and the HCP scheme only.
- Round 2 standardized pre- and post-processing: github.com/chiaramff/IronTract.

**Access.**
- Data on the QMENTA platform after registering at irontract.mgh.harvard.edu (site live; both rounds "completed").
- The paper says the challenge "remains open". Current download and scoring status: UNVERIFIED.
- No license is stated.

**Findings.**
- Carefully tuned analyses of the HCP scheme matched the slower Cartesian scheme.
- Simple pre- and post-processing improved most methods.
- Fanning and branching, more than crossing, were the hardest configurations.
- The best-performing orientation method was RUMBA-SD. The winning pipeline added denoising, Gibbs-ringing correction and
  motion/eddy correction.

**Fit for tractline.**
- **UKF:** Partly. The b = 6000 ex vivo shell is the closest match. Ex vivo diffusivities and a single b0 are far from
  in vivo ORG settings.
- **Labeler:** No. Macaque.
- **Field correction:** No.
- **Ensemble calibration:** Yes. A presence envelope thresholded at several levels is exactly the multi-threshold ROC the
  challenge scores, and a voxelwise reliability diagram against the tracer mask is direct. This is the only real-tissue
  axon ground truth among the open challenges.

## 3D VoTEM (ISBI 2018; the squirrel-monkey histology)

The 3D VoTEM challenge ran at **ISBI 2018** (April 4), not ISMRM 2018.

**Papers.**
- Schilling KG, Nath V, Hansen C, et al. Limits to anatomical accuracy of diffusion tractography using modern approaches.
  *NeuroImage* 2019;185:1–11. doi:10.1016/j.neuroimage.2018.10.029.
- Schilling KG, Daducci A, Maier-Hein K, et al. Challenges in diffusion MRI tractography – Lessons learned from international
  benchmark competitions. *Magn Reson Imaging* 2019;57:194–209. doi:10.1016/j.mri.2018.11.014.
- The squirrel-monkey histology study: Schilling KG, Gao Y, Stepniewska I, et al. Anatomical accuracy of standard-practice
  tractography algorithms in the motor system – A histological validation in the squirrel monkey brain. *Magn Reson
  Imaging* 2019;55:7–25. doi:10.1016/j.mri.2018.09.004.

**Ground truth and data.**

| Sub-challenge | Ground truth | Data |
|---|---|---|
| Macaque (Thomas et al. 2014) | tracer-derived pathways from precentral gyrus and V4v; region-to-region connectivity | ex vivo, 0.25 mm; 7 b0s + 114 at b = 4900; SNR ≈ 40; TORTOISE-preprocessed |
| Squirrel monkey | BDA tracer in M1 of the same brain, charted on histological sections; per-voxel fiber-density maps and 71 histology-defined ROIs | ex vivo, 9.4 T, 300 µm; b = 1200, 31 directions; 10 averages; SNR ≈ 25 |
| Physical phantom (Synaptive Medical) | 16 bundles, 2–6 mm across, traced by hand on a T1 image and registered to dMRI | two Philips 3 T scanners; 2.25 × 2.25 × 2.5 mm; b = 1000 and 2000, 96 directions each; **AP and PA** phase encoding "to allow pre-processing" for susceptibility; 7 repeats per scanner |

**Metrics.**
- ROI level: sensitivity, specificity and Youden's J.
- Voxel level: overlap, overreach and Dice.

**Access.**
- Synapse: register, then join the "VoTEM Phantom", "VoTEM Squirrel Monkey" or "VoTEM Macaque" team.
- The NeuroImage paper states all data for the three sub-challenges are freely available and submissions stay open. The
  VoTEM site is live; whether the Synapse projects still serve files is UNVERIFIED. No license is stated.

**Findings.**
- 176 submissions came from 9 groups.
- Median Youden J was 0.21 and 0.30 (macaque PCG, V4v) and 0.37 (squirrel-monkey M1); the best were 0.56, 0.58 and 0.67.
- Every method traded sensitivity against specificity; none escaped the trade-off.

**Fit for tractline.**
- **UKF:** Phantom yes; animals partly (ex vivo).
- **Labeler:** No. The phantom's bundles are not brain tracts.
- **Field correction:** The phantom, yes, with real AP/PA pairs from real scanners.
- **Ensemble calibration:** Yes, and uniquely so. The phantom has 7 × 2 real repeated scans on two scanners with known
  bundles: real rescans, real noise and a ground truth at once. It is the closest physical analog of MASiVar with a known
  answer.

## Fiber Cup (MICCAI 2009) and the Tractometer (2013)

**Papers.**
- Fillard P, Descoteaux M, Goh A, et al. Quantitative evaluation of 10 tractography algorithms on a realistic diffusion MR
  phantom. *NeuroImage* 2011;56:220–34. doi:10.1016/j.neuroimage.2011.01.032.
- Côté M-A, Girard G, Boré A, Garyfallidis E, Houde J-C, Descoteaux M. Tractometer: towards validation of tractography
  pipelines. *Med Image Anal* 2013;17:844–57. doi:10.1016/j.media.2013.03.009.

**Ground truth.**
- A physical phantom of hydrophobic acrylic fibers in a 2D frame, mimicking a coronal slice. It has crossing, kissing,
  splitting and curving bundles. It was built by the procedure of Poupon et al. 2008 and 2010.
- 16 connections (Schilling 2019 *MRI* review). The number of bundles is UNVERIFIED; Côté's figures number at least seven.
- The ground truth was withheld during the contest and released afterward. Its current download location is UNVERIFIED:
  the original site, lnao.fr, returns HTTP 410.

**Data.**
- 3 T Tim Trio, 3 mm isotropic, 3 slices of 64 × 64.
- b = 650, 1500 and 2000, 64 directions each, plus one b0. Two acquisitions and their average.
- DWI SNR 9.1, 2.6 and 1.1 by b-value; b0 SNR ≈ 15.8.
- No reversed PE is described.

**Metrics.**
- Fillard: distances (spatial, tangent, curvature) from 16 given seeds.
- Côté introduced the global metrics later reused for ISMRM 2015: VC, IC, NC, VB, IB and average bundle coverage (ABC).

**Access.** DWIs and the WM mask at tractometer.org/fibercup/data (live). The site asks users to cite three papers.

**Findings.**
- At high SNR, ODF models with streamline tracking work.
- At low SNR, spatial smoothness priors help (Fillard).
- Averaging, sharp ODFs and careful seeding matter (Côté).
- Côté also found deterministic tracking makes fewer invalid tracts than probabilistic.

**Fit for tractline.** A two-dimensional slab is a weak test for UKF. There is nothing for the labeler, the field
correction or the envelopes. It is of historical interest; the ISMRM 2015 phantom superseded it.

## HARDI reconstruction challenges (ISBI 2012 and 2013)

**Paper (2012).** Daducci A, Canales-Rodríguez EJ, Descoteaux M, et al. Quantitative comparison of reconstruction methods
for intra-voxel fiber recovery from diffusion MRI. *IEEE Trans Med Imaging* 2014;33:384–99. doi:10.1109/TMI.2013.2285500.

**Ground truth.**
- 2012: two synthetic datasets with known fiber configurations.
- 2013: a simulated phantom, with ground-truth orientations shipped as `ground-truth-peaks.nii.gz`.
  - Its signal model is CHARMED-like: a Gaussian extra-axonal compartment and a Söderman cylinder intra-axonal one.
  - Each voxel is split into 9 × 9 × 9 sub-voxels, an approach adapted from the Numerical Fiber Generator (organizer
    slides).
  - Phantomas (Caruyer et al., ISMRM 2014; github.com/ecaruyer/phantomas, BSD) is the open library behind this kind of
    phantom.
  - That the 2013 phantom has 27 bundles at 1 mm is UNVERIFIED (secondary sources only).

**Data (2013).**
- 50 × 50 × 50 volume.
- Three schemes: DTI (32 directions, b = 1200), HARDI (64 directions, b = 3000) and DSI (515 samples, b < 12000).
- Each at SNR 10, 20 and 30, plus one custom scheme per team on request.

**Metrics.**
- Per voxel: correct number of fiber populations and angular error.
- Evaluation script: `compute_local_metrics.py`.
- 2013 also aimed to judge local models through connectivity, with the Tractometer; the details are UNVERIFIED.

**Access.** The 2013 site is live (hardi.epfl.ch/static/events/2013_ISBI/). No license is stated.

**Findings.** 20 methods were compared in 2012. Each had strengths and weaknesses that depended on acquisition and SNR;
no single winner emerged.

**Fit for tractline.** Only the orientation stage can be tested: UKF's two tensor directions against the ground-truth peaks
on the HARDI scheme. That check is narrow but cheap. There is nothing for the labeler or the field correction.

## TractoInferno, and other silver standards

**Paper.** Poulin P, Theaud G, Rheault F, et al. TractoInferno – A large-scale, open-source, multi-site database for machine
learning dMRI tractography. *Sci Data* 2022;9:725. doi:10.1038/s41597-022-01833-1.

**Reference ("silver standard").**
- Ensemble tractography from four algorithms (deterministic, probabilistic, PFT and surface-enhanced), bundled by
  RecobundlesX with label fusion, with manual QC.
- 30 bundles, including AF, UF, PYT (pyramidal tract), SLF, ILF, IFOF and CC parts.
- This is agreement with a well-run pipeline, not anatomy.

**Data.**
- 284 healthy human scans from 6 sites on 3 T scanners, single shell near b = 1000.
- T1s, DWIs, fODFs and masks.
- Topup was skipped because "not all reverse b0 images were available".

**Access.** OpenNeuro ds003900, snapshot 1.1.1, license CC0 (checked through the OpenNeuro API).

**Other silver standards worth knowing:**
- TractSeg's 72 reference tracts on 105 HCP subjects. Zenodo record 1477956, 44 GB, CC BY-NC 4.0 (tracts only; HCP DWIs
  need an HCP account). TractSeg is already cited in `docs/prior-art-uncertainty.md`.
- A caution against any human "ground truth" for bundle definitions: Schilling KG et al. Tractography dissection
  variability: What happens when 42 groups dissect 14 white matter bundles on the same dataset? *NeuroImage*
  2021;243:118502. doi:10.1016/j.neuroimage.2021.118502. Protocol-to-protocol differences exceeded subject-to-subject ones.

**Fit for tractline.** These give agreement only: our AF, UF and CST against RBX's or TractSeg's, at scale and across sites.
They cannot answer the "reproducibly wrong" question. They are a cross-check for the labeler's Other share and bias, not
for accuracy.

## Other phantoms and resources in use

- **Fiberfox** (MITK-Diffusion; Neher et al. 2014, doi:10.1002/mrm.25045). It simulates k-space acquisition with artifacts:
  noise, ghosts, spikes, motion, eddy currents and B0 distortion.
  - It generated the ISMRM 2015 data; the GitHub repo is active (last push 2026-02-24).
  - It could also simulate a phantom from an ORG-labeled HCP tractogram. The ground truth would then contain all 42 tracts,
    and an AP/PA pair could be added. That would make the labeler scorable tract by tract. It is a build-our-own option;
    the effort is UNVERIFIED.
- **Diffantom** (Esteban O, et al. *Front Neuroinform* 2016;10:4, doi:10.3389/fninf.2016.00004). Whole-brain phantoms
  derived from HCP data.
- **BigMac** (Howard AFD, et al. *Nat Commun* 2023;14:4320, doi:10.1038/s41467-023-39916-1).
  - One macaque with in vivo and postmortem dMRI (up to 1000 directions, b up to 10 ms/µm²), polarized light imaging and
    histology, co-registered.
  - It supports a microscopy-informed connectome rather than a tracer ground truth.
  - The full data needs a data-sharing agreement through the Digital Brain Bank.
- **Physical phantoms reviewed:** Drobnjak I, Neher P, Poupon C, Sarwar T. Physical and digital phantoms for validating
  tractography and assessing artifacts. *NeuroImage* 2021;245:118704. doi:10.1016/j.neuroimage.2021.118704.
- **Earlier community challenges with no anatomical ground truth:**
  - MICCAI DTI Challenge 2011–2015: neurosurgical CST cases judged by experts; dti-challenge.org did not answer.
  - ISMRM 2017 TraCED: scan-rescan reproducibility, one subject on two scanners; its site returns 404.

## Newer challenges (2022–2026)

- **MEMENTO** is *not* a tractography challenge. It tested how diffusion *signal* representations generalize to unseen
  acquisitions, on human MASSIVE data and mouse double-diffusion-encoding data. De Luca A, et al. *NeuroImage*
  2021;240:118367. doi:10.1016/j.neuroimage.2021.118367. Data at github.com/PROVIDI-Lab/MEMENTO.
- **CDMRI-QuantCONN** appears in the MICCAI 2023 challenge list as an acquisition-harmonization challenge. Its ground truth
  and data were not checked (UNVERIFIED).
- The International Society for Tractography's first meeting (IST 2025, Bordeaux; proceedings arXiv:2602.12410) announced no
  challenge or ground-truth dataset on its listing page; the full PDF was not read.
- Searches for tractography ground-truth challenges at MICCAI, ISBI or ISMRM from 2022 to 2026 found none beyond the
  DiSCo results paper (2023) and the ISMRM 2015 rescoring (2023).

## Recommendation

**First, the ISMRM 2015 phantom, the version with the reversed-phase b0, scored with the 2023 ROI scorer.**
- It is the only benchmark where all four of tractline's stages run on one dataset:
  - the field correction, on a real AP/PA pair geometry with a provided field map to check the field;
  - UKF, on a clinical single shell;
  - the labeler, on human HCP anatomy in its own training domain, for about 8 mappable tracts, with AF as a built-in false
    positive;
  - the ensemble's envelopes, against ground-truth masks.
- Its scoring is open, scriptable (scilpy) and independent of our labeler. It is CC BY 4.0, a few hundred MB and needs no
  registration.
- Fiberfox can redraw the noise to make simulated rescans with a known truth.
- Three cautions:
  - flip the gradient x sign;
  - find or derive the readout time;
  - read results as one subject at b = 1000. Then check the b = 3000 variant without the field step.

**Second, DiSCo v2 for the tracker and the calibration of connection presence.**
- It is tiny, CC BY 4.0, open, and has five SNR levels and three phantoms.
- It isolates noise-to-tracker behavior, which the robustness work says dominates the spread. It answers whether lower
  spread came with more or fewer false connections.
- It tests nothing upstream (field) or downstream (labeler).

**Next, if real tissue or real rescans with known paths are needed:**
- the VoTEM Synaptive phantom: real AP/PA scans, 7 repeats on each of 2 scanners;
- then IronTract's tracer ROC for the envelopes.
Both need registration, and their availability today is UNVERIFIED.
