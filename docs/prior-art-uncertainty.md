<!-- A search on 2026-10-09 by a delegated agent, spot-checked: Campbell et al. 2014 and Yu et al. 2024 (the 0.5-voxel seed ball) read in full; Brown et al. is the MICCAI CDMRI 2013 workshop, published by Springer in 2014. The project's labeler is RapidParc, on TractCloud's 42-tract + Other scheme. Paper titles keep their published spelling. Summary in prior-art.md, section 4. -->

# Prior-art search: bootstrap ensembles of a full tractography pipeline vs. real test-retest rescans

Date of search: 2026-10-09. Databases: PubMed (via the NCBI E-utilities connector), web search (standard and extended), arXiv, bioRxiv/medRxiv, publisher and lab pages. Bibliographic details come from PubMed records unless noted otherwise. "Verified" means I read at least the abstract, and for some papers the full text, in this session. Quotes are kept under 15 words.

---

## 1. Bottom line, claim by claim

### Claim 1: does the bootstrap's spread of tractography output match real rescans? Per-voxel sign flips understate it because noise is spatially correlated in-plane. Smooth sign patches match a same-session rescan, and between-session variability is larger.

**Verdict: partially anticipated. The core findings appear novel.**

- **Closest prior work: Campbell et al. 2014 (Front Neurol).** This is the only study I found that checks bootstrap *coverage* against real repeated scans. One subject was scanned four times in one session without repositioning. Each scan in turn served as the bootstrap source, and the authors checked whether the fODF peaks of the other three scans fell inside the residual-bootstrap 68 % and 95 % intervals. Agreement was good in voxels with FA > 0.3. In low-FA voxels the bootstrap underestimated the variability. They blamed SH overfitting and sub-voxel misregistration. The paper does not discuss spatial noise correlation.
  - Differences from the project: their check is at the level of voxel fiber orientations, not tractography outputs (labels, paths, tract maps). It covers same-session repeats only. They do not identify spatial noise correlation as the cause.
- **Other partial precedents:**
  - Zhu et al. 2011 found that the within-scan wild-bootstrap variability was about the same size as intra-site and inter-site variability. That result is at the tensor level, from a traveling subject.
  - Vorburger et al. 2012 measured how *precise* bootstrap uncertainty estimates were over 10 sessions. They did not check coverage.
  - Roine et al. 2019 used the residual bootstrap as a stand-in for intrasubject variability of connectome metrics and repeated their experiments on HCP test-retest data. That is connectome level, with no coverage check.
  - Whitcher 2008, Jones 2008 and Haroon 2009 compared wild or residual bootstrap against the repetition bootstrap, using repeats from the same session.
  - Yuan et al. 2008 showed theoretically that wild and repetition bootstraps fail for the principal direction of isotropic, oblate or multi-fiber tensors.
- **On noise correlation:**
  - Spatially correlated or non-stationary noise after parallel-imaging reconstruction is well documented (Robson 2008, Dietrich 2008), and so is its effect on PCA denoising (Henriques et al. 2023).
  - I found **no dMRI paper** that names spatial noise correlation as a reason the wild or residual bootstrap understates rescan variability.
  - I found no dMRI paper that proposes spatially coherent ("patch" or "block") sign flipping.
  - The nonlocal bootstrap (Yap 2013/2014) uses spatial self-similarity, but to borrow residuals from elsewhere in the image, not to preserve correlation.
  - Statistical precedent for keeping dependence intact by flipping signs on whole clusters does exist. Examples are the cluster bootstrap-t / wild cluster bootstrap (Cameron, Gelbach & Miller 2008) and whole-image sign flipping in neuroimaging permutation inference (Winkler 2014). Cite these as the methodological basis.
- **Conservative wording:** "Bootstrap uncertainty has been compared with within-session repeats at the level of fiber orientation (Campbell 2014) and tensor metrics (Zhu 2011). To our knowledge it has not been compared with rescans at the level of tractography output, and spatial noise correlation has not been identified as the cause of under-dispersion."

### Claim 2: whole-pipeline resampling, including re-estimation of the distortion field

**Verdict: no direct precedent found. The idea is anticipated only in spirit.**

- Every dMRI bootstrap paper I found resamples data that has *already* been preprocessed, then refits only the local model. This covers Jones 2003/2005/2008, Lazar 2005, Berman 2008, Haroon 2009, Jeurissen 2011, Vorburger 2013 and Gu 2019. Gu et al. 2019 used the wild bootstrap to *compare* preprocessing pipelines, but did not push resampled data back through them.
- Kiar et al. 2020/2021 perturbed an *entire* dMRI connectome pipeline. Their perturbations were Monte Carlo Arithmetic, operating-system variation and small injected noise. This is computational instability, not a data-noise resampling model, and they did not compare it with rescans or examine distortion correction.
- Several studies show that distortion correction strongly affects tractography: Irfanoglu 2012, Gui 2008, Gu & Eklund 2019, and Begnoche 2022, which found changes far from the high-distortion regions. None of them propagates noise uncertainty through re-estimating the field.
- The Fuzzy Fibers review (Schultz et al.) and Dell'Acqua et al. 2025 say that uncertainty builds up across the pipeline, but neither quantifies it this way.
- **Conservative wording:** "We found no prior work that propagates data-noise resampling through susceptibility-field re-estimation into tractography and tract labels."

### Claim 3: per-streamline label change on a rescan, matched by seed voxel after rigid registration (about 35–50 % of streamlines change tract)

**Verdict: the measurement appears novel, but the matching method is partially anticipated. Yu et al. 2024 must be cited.**

- **Yu et al. 2024 (Proc SPIE 12926; arXiv 2403.18992)** introduced an "epsilon ball seeding" metric:
  - For each streamline in registered tractogram A, find the streamlines in B that pass within a 0.5-voxel ball around its seed.
  - Resample to 100 points and take the minimum MDF distance.
  - They applied it to **scan-rescan** pairs (BLSA). For scan-to-rescan, the mean of medians was about 3.2 mm.

  This is very close to the project's seed-matched streamline correspondence, but it measures **geometric** distance, not label agreement.
- **Zhang et al. 2019 (HBM)** is the closest reproducibility study for the *same family of pipeline*: two-tensor UKF tractography plus the ORG fiber-clustering atlas, on three test-retest datasets with 255 subjects. Reproducibility was measured at the cluster and tract level (wDice and FA relative difference), not per streamline.
- **Schilling et al. 2021 (42 groups)** ran streamline-based agreement analyses *across segmentation protocols on identical streamlines*. That is per-streamline label agreement, but across methods rather than across rescans.
- TractCloud (Xue et al. 2023: 42 tracts plus an "other" class, i.e. 43 classes), DeepWMA (Zhang 2020) and SupWMA report per-streamline accuracy on labeled atlas data and consistency across datasets. **I found no test-retest per-streamline label analysis for any of them.**
- I found no published figure for the fraction of streamlines that change tract between rescans.

### Claim 4: calibration (ECE, AUROC) of per-streamline and per-voxel tract probabilities against real rescans, plus a double-bootstrap recalibration that transferred from a Siemens scan to a Philips subject

**Verdict: appears novel in tractography and tract segmentation.**

- Lucena et al. 2022 is the closest. They calibrated deep-learning tract-segmentation probabilities (test-time dropout and augmentation, then volume-based calibration). The reference was manual or automatic ground-truth masks, not rescans, and they reported coverage of the true *volume*, not ECE against repeat scans.
- Kebiri et al. (arXiv 2307.02223) flagged inaccurate segmentations using an uncertainty score, evaluated against Dice. They did not test calibration.
- In probabilistic tractography, validation has been binary or ROC-based, against tracers (Grisot 2021) or phantoms (Maier-Hein 2017). It has not been calibration. Campbell 2014's interval-coverage check is the nearest thing to a calibration test, and it applies to orientations only.
- Extended web searches for "reliability diagram", "Brier score", "ECE" and "conformal" combined with tractography or tract segmentation found nothing in this domain. Conformal methods exist for glioma and other segmentation, but not for tracts.
- The double bootstrap is classical statistics (Beran 1987 prepivoting; Hall & Martin 1988 bootstrap iteration). Applying it to recalibrate tractography and showing it transfers across vendors appears new.

### Claim 5: coverage of real rescan tracts by ensemble "envelopes", and the finding that winner-take-all maps prevent honest envelopes while presence maps fix it

**Verdict: the coverage test against rescans appears novel. The envelope idea itself has precedent.**

- Brown, Booth & Hamarneh 2014 (CDMRI) built **tract confidence regions** from sampled tract curves between two regions. The region is meant to contain the *mean* tract at a chosen confidence level. They validated it on synthetic data, on consistency across subjects (12 subjects) and on patients. There was no rescan coverage test, and the region targets the mean path, not mass coverage.
- Brecheisen et al. 2013 introduced "illustrative confidence intervals" around fiber bundles for neurosurgeons, as a visualization.
- Reid et al. 2020 measured how many streamlines are needed for reproducible *binarized* trackmaps across repeated probabilistic runs. That is stochasticity of the algorithm, not rescans.
- Besseling 2012 and Boukadi 2019 report test-retest Dice of thresholded tracts, but did not test envelope coverage.
- Overlapping, multi-label tract masks are standard (TractSeg outputs one mask per tract). The *specific* argument that winner-take-all voxel maps make it impossible to build mass-calibrated envelopes did not appear in anything I found.
- The surgical literature discusses fixed safety margins and their sources of error (Yang et al. 2021 review), but not envelopes with empirical coverage.

### Related topics

- **Probability semantics:** That a tractography "connection probability" measures precision, not the probability that a connection exists, is well established. Cite Jones 2008 (Cortex), Morris, Embleton & Parker 2008 (null connection maps), Jbabdi & Johansen-Berg 2011, Jones, Knösche & Turner 2013, Sotiropoulos & Zalesky 2019 and Maier-Hein 2017. The project's contribution is a calibration test, not this observation.
- **Regional vs streamline-level reproducibility:** Bundle- and voxel-level test-retest work is extensive (TraCED, Heiervang, Besseling, Boukadi, Cousineau, Zhang 2019, MASiVar, Schilling 2021). Streamline-level work is rare (Yu 2024 for geometry; Schilling 2021 across protocols).
- **Compact storage:** Predictive geometry coding of streamlines is **already done**:
  - TRAKO (Haehn et al. 2020) uses Draco, which combines quantization with prediction schemes, and also compresses per-vertex and per-streamline properties.
  - zfib (Presseau 2015) uses linearization, quantization and encoding.
  - QFib (Mercier 2020) uses unit-vector quantization.
  - TRX (Rheault et al., OHBM 2022 abstract) is a community container format with data per streamline and per group.

  I found no tractography precedent for top-k or ranked *soft-label* storage. Storing truncated soft labels is common practice in knowledge distillation, but I could not verify a specific citation for it (see UNVERIFIED).

---

## 2. Annotated bibliography

### A. Bootstrap and probabilistic tractography uncertainty

1. **Jones DK. Determining and visualizing uncertainty in estimates of fiber orientation from diffusion tensor MRI.** *Magn Reson Med* 2003;49(1):7–12. doi:10.1002/mrm.10331.
   - Uses a repetition bootstrap on repeated DTI acquisitions to make "cone of uncertainty" maps of the principal eigenvector, and relates them to anisotropy.
   - Level: voxel orientation. Data: in vivo with repeated measurements; no between-session test.
2. **Jones DK, Pierpaoli C. Confidence mapping in diffusion tensor MRI tractography using a bootstrap approach.** *Magn Reson Med* 2005;53(5):1143–9. doi:10.1002/mrm.20466.
   - Uses the bootstrap to run any deterministic tracker probabilistically, and introduces the "tract-propagator".
   - Argues that the bootstrap covers all sources of variability without a model (repetition bootstrap, within one session).
   - Level: streamline and pathway.
3. **Jones DK, Travis AR, Eden G, Pierpaoli C, Basser PJ. PASTA: pointwise assessment of streamline tractography attributes.** *Magn Reson Med* 2005;53(6):1462–7. doi:10.1002/mrm.20484.
   - Shows bootstrap distributions of FA, trace and orientation uncertainty at each point along a streamline. A visualization tool.
4. **Lazar M, Alexander AL. Bootstrap white matter tractography (BOOT-TRAC).** *NeuroImage* 2005;24(2):524–32. doi:10.1016/j.neuroimage.2004.08.050.
   - Repetition-bootstrap tractography. Its dispersion matched an analytic error model except where tracts branch.
   - Also used to estimate connection probabilities. Level: streamline dispersion.
5. **Whitcher B, Tuch DS, Wisco JJ, Sorensen AG, Wang L. Using the wild bootstrap to quantify uncertainty in diffusion tensor imaging.** *Hum Brain Mapp* 2008;29(3):346–62. doi:10.1002/hbm.20395.
   - Introduces the wild bootstrap to DTI (one measurement per direction). Compared with the regular bootstrap in Monte Carlo simulation and in vivo.
   - Level: voxel FA and eigenvector. The residuals are resampled voxel by voxel.
6. **Jones DK. Tractography gone wild: probabilistic fibre tracking using the wild bootstrap with diffusion tensor MRI.** *IEEE Trans Med Imaging* 2008;27(9):1268–74. doi:10.1109/TMI.2008.922191.
   - Wild-bootstrap tractography from a single acquisition, which gave results comparable to regular (repeated-data) bootstrap tracking.
   - This is the closest methodological ancestor of the project's per-replicate tracking.
7. **Berman JI, Chung S, Mukherjee P, Hess CP, Han ET, Henry RG. Probabilistic streamline q-ball tractography using the residual bootstrap.** *NeuroImage* 2008;39(1):215–22. doi:10.1016/j.neuroimage.2007.08.021.
   - Residual bootstrap with a spherical-harmonic q-ball fit. Validated by simulation, with in vivo tracking of CST and CC.
8. **Haroon HA, Morris DM, Embleton KV, Alexander DC, Parker GJM. Using the model-based residual bootstrap to quantify uncertainty in fiber orientations from Q-ball analysis.** *IEEE Trans Med Imaging* 2009;28(4):535–50. doi:10.1109/TMI.2008.2006528.
   - The model-based residual bootstrap closely matched the conventional (repeated-data) bootstrap for fiber orientations. Level: voxel.
9. **Jeurissen B, Leemans A, Jones DK, Tournier J-D, Sijbers J. Probabilistic fiber tracking using the residual bootstrap with constrained spherical deconvolution.** *Hum Brain Mapp* 2011;32(3):461–79. doi:10.1002/hbm.21032.
   - Residual-bootstrap CSD tractography, assessed by Monte Carlo simulation and on clinical data. No rescans.
10. **Campbell JSW, MomayyezSiahkal P, Savadjiev P, Leppert IR, Siddiqi K, Pike GB. Beyond crossing fibers: bootstrap probabilistic tractography using complex subvoxel fiber geometries.** *Front Neurol* 2014;5:216. doi:10.3389/fneur.2014.00216. **Most relevant to claim 1.**
    - Checked residual-bootstrap 68 % and 95 % intervals of fODF peaks against three other same-session repeats (one subject, four scans, no repositioning; full text read).
    - Coverage was good where FA > 0.3. The bootstrap underestimated variability at low FA.
11. **Vorburger RS, Reischauer C, Boesiger P. BootGraph: probabilistic fiber tractography using bootstrap algorithms and graph theory.** *NeuroImage* 2013;66:426–35. doi:10.1016/j.neuroimage.2012.10.058.
    - A graph-based tractography method whose edge weights come from wild-bootstrap cones of uncertainty. The authors report high repeatability.
12. **Behrens TEJ et al. Characterization and propagation of uncertainty in diffusion-weighted MR imaging.** *Magn Reson Med* 2003;50(5):1077–88. doi:10.1002/mrm.10609.
    - Bayesian posteriors on fiber orientation, propagated into a "probability of connection" (probtrackx).
    - Uncertainty comes from the model only: local noise, no preprocessing, no rescans.
13. **Behrens TEJ, Johansen-Berg H, Jbabdi S, Rushworth MFS, Woolrich MW. Probabilistic diffusion tractography with multiple fibre orientations: what can we gain?** *NeuroImage* 2007;34(1):144–55. doi:10.1016/j.neuroimage.2006.09.018.
    - Multi-fiber extension, using automatic relevance determination, of the Bayesian framework above.
14. **Yap P-T, An H, Chen Y, Shen D. Uncertainty estimation in diffusion MRI using the nonlocal bootstrap.** *IEEE Trans Med Imaging* 2014;33(8):1627–40. doi:10.1109/TMI.2014.2320947.
    - Borrows information from self-similar nonlocal regions. It agrees better with Monte Carlo ground truth than the residual bootstrap does, in silico.
    - It does not model spatially correlated noise or compare against rescans.
    - Related: the 2013 IPMI version (doi:10.1007/978-3-642-38868-2_33) and the "wild nonlocal bootstrap" applied to tractography (CDMRI 2013; doi:10.1007/978-3-319-02475-2_13).
15. **Gruen J, van der Voort G, Schultz T. Model averaging and bootstrap consensus-based uncertainty reduction in diffusion MRI tractography.** *Comput Graph Forum* 2023;42(1):217–30. doi:10.1111/cgf.14724.
    - Uses bootstrap replicates to *reduce* uncertainty by building a consensus fiber model, rather than to describe it.
    - Verified from the Eurographics library record and abstract only.

### B. Test-retest reproducibility of tractography and bundles

16. **Heiervang E, Behrens TEJ, Mackay CE, Robson MD, Johansen-Berg H. Between session reproducibility and between subject variability of diffusion MR and tractography measures.** *NeuroImage* 2006;33(3):867–77. doi:10.1016/j.neuroimage.2006.07.037.
    - 8 subjects × 3 sessions, probabilistic tractography.
    - Between-session CV was under 5 % for tract-mean FA. Tract volume was less reproducible than FA and MD.
17. **Vollmar C et al. Identical, but not the same: intra-site and inter-site reproducibility of fractional anisotropy measures on two 3.0T scanners.** *NeuroImage* 2010;51(4):1384–94. doi:10.1016/j.neuroimage.2010.03.046.
    - 9 subjects on two nominally identical scanners.
    - CV of FA in probabilistic tracts was about 50 % higher than in the seed ROIs, and inter-site CV was above intra-site CV. Level: ROI and tract mean.
18. **Besseling RMH et al. Tract specific reproducibility of tractography based morphology and diffusion metrics.** *PLoS One* 2012;7(4):e34125. doi:10.1371/journal.pone.0034125.
    - 9 subjects, two sessions, probabilistic CSD. Inter-session Dice of tract segmentations was 0.70–0.92.
    - Volume and tract-density metrics were the least reproducible.
19. **Nath V, Schilling KG, et al. Tractography reproducibility challenge with empirical data (TraCED): the 2017 ISMRM diffusion study group challenge.** *J Magn Reson Imaging* 2020;51(1):234–49. doi:10.1002/jmri.26794.
    - 46 algorithm entries × 16 tracts on scan-rescan data from one subject on two scanners from the same vendor.
    - Measured ICC and Dice. Reproducibility of the top submissions was 0.87–0.97 across sessions and scanners. Level: voxel or tract.
20. **Zhang F, Wu Y, Norton I, Rathi Y, Golby AJ, O'Donnell LJ. Test-retest reproducibility of white matter parcellation using diffusion MRI tractography fiber clustering.** *Hum Brain Mapp* 2019;40(10):3041–57. doi:10.1002/hbm.24579.
    - Uses the same family of pipeline (two-tensor UKF plus the ORG atlas) on 255 subjects from three test-retest datasets.
    - Measured wDice and FA relative difference at the parcel and tract level. **No per-streamline correspondence.**
    - The UKF/ORG details come from search-result snippets of the full text. The abstract alone confirms only the datasets and metrics.
21. **Cai LY et al. MASiVar: Multisite, multiscanner, and multisubject acquisitions for studying variability in diffusion weighted MRI.** *Magn Reson Med* 2021;86(6):3304–20. doi:10.1002/mrm.28926.
    - 319 scans, separating intrasession, intersession, interscanner and intersubject variability for DTI, NODDI, bundle segmentation and connectomes.
    - Variability rose consistently from intrasession to intersession to interscanner to intersubject. This independently supports the project's finding that same-session < between-session.
22. **Schilling KG, Tax CMW, Rheault F, et al. Fiber tractography bundle segmentation depends on scanner effects, vendor effects, acquisition resolution, diffusion sampling scheme, diffusion sensitization, and bundle segmentation workflow.** *NeuroImage* 2021;242:118451. doi:10.1016/j.neuroimage.2021.118451.
    - Ranks sources of bundle variation; the effect of the segmentation workflow was larger than all the others.
    - Tractography adds variability beyond noise, scanner and acquisition effects. Level: bundle volume, shape and microstructure.
23. **Schilling KG, Rheault F, et al. Tractography dissection variability: what happens when 42 groups dissect 14 white matter bundles on the same dataset?** *NeuroImage* 2021;243:118502. doi:10.1016/j.neuroimage.2021.118502.
    - 57 protocols applied to the same streamlines.
    - Volume-based and **streamline-based** agreement: variability across protocols was larger than within-protocol or across-subject variability.
24. **Rheault F et al. Tractostorm: the what, why, and how of tractography dissection reproducibility.** *Hum Brain Mapp* 2020;41(7):1859–74. doi:10.1002/hbm.24917.
    - Rater variability in virtual dissection: Dice about 0.77 within raters and about 0.65 between raters.
    - Follow-up: Tractostorm 2, *HBM* 2022;43:2134–47, doi:10.1002/hbm.25777.
25. **Boukadi M et al. Test-retest reliability of diffusion measures extracted along white matter language fiber bundles using HARDI-based tractography.** *Front Neurosci* 2019;12:1055. doi:10.3389/fnins.2018.01055.
    - 18 subjects scanned one week apart, using anatomically constrained probabilistic CSD. Bundle overlap and ICC of tractometry.
26. **Cousineau M et al. A test-retest study on Parkinson's PPMI dataset yields statistically significant white matter fascicles.** *NeuroImage Clin* 2017;16:222–33. doi:10.1016/j.nicl.2017.07.020.
    - Used test-retest reproducibility as a filter for which of 50 fascicles to keep for tractometry.
27. **Reid LB, Cespedes MI, Pannek K. How many streamlines are required for reliable probabilistic tractography?** *NeuroImage* 2020;211:116646. doi:10.1016/j.neuroimage.2020.116646.
    - Reproducibility of binarized trackmaps and microstructure across *repeated tractography runs on the same data* (40 HCP subjects).
    - Shows that the method of binarization matters. This is a precedent for envelope or threshold reliability, but from algorithmic stochasticity, not rescans.
28. **Roine T et al. Reproducibility and intercorrelation of graph theoretical measures in structural brain connectivity networks.** *Med Image Anal* 2019;52:56–67. doi:10.1016/j.media.2018.10.009.
    - Used the residual bootstrap as a surrogate for intrasubject variability of connectome metrics, then repeated the experiments on HCP test-retest data (44 subjects). Level: connectome.
29. **Yu T, Li Y, Kim ME, …, Schilling KG, Landman BA. Tractography with T1-weighted MRI and associated anatomical constraints on clinical quality diffusion MRI.** *Proc SPIE* 2024;12926. doi:10.1117/12.3006286 (arXiv:2403.18992). **Closest to claim 3's matching method.**
    - "Epsilon ball seeding": seed-matched streamline correspondence (0.5-voxel ball, 100-point resampling, minimum MDF) between registered tractograms.
    - Applied to diffusion **scan-rescan** pairs as a reference variability (full text read). Geometry only, not labels.
30. **Dell'Acqua F et al. The tractographer's dilemma: understanding sources of variability in tractography.** *Brain Struct Funct* 2025. doi:10.1007/s00429-025-02964-y.
    - Narrative review of sources of variability through the pipeline. Useful framing citation; abstract only.

### C. Streamline-level labelers and label agreement

31. **Xue T, Chen Y, Zhang C, Golby AJ, Makris N, Rathi Y, Cai W, Zhang F, O'Donnell LJ. TractCloud: registration-free tractography parcellation with a novel local-global streamline point cloud representation.** MICCAI 2023 (LNCS); doi:10.1007/978-3-031-43993-3_40; arXiv:2307.09000.
    - Training data has 43 classes (42 tracts plus "other"). Reports per-streamline accuracy on labeled atlas data, and tract identification rate and distance to atlas on five external datasets.
    - **No test-retest evaluation** (full text checked on ar5iv). This is probably the labeler family the project uses; confirm in the project's own code.
32. **Zhang F et al. Deep white matter analysis (DeepWMA): fast and consistent tractography segmentation.** *Med Image Anal* 2020;65:101761. doi:10.1016/j.media.2020.101761.
    - Streamline CNN for 54 tracts, with 90.99 % per-fiber accuracy on training data and consistency of tract detection across 597 scans. No test-retest analysis of per-streamline labels.
33. **Wasserthal J, Neher P, Maier-Hein KH. TractSeg — fast and accurate white matter tract segmentation.** *NeuroImage* 2018;183:239–53. doi:10.1016/j.neuroimage.2018.07.070.
    - Segments tracts directly from fODF peaks. The abstract does not discuss overlap; that TractSeg outputs one mask per tract, with overlap allowed, is from my knowledge of the method.
    - Relevant to the presence vs winner-take-all point in claim 5.

### D. Calibration and uncertainty quantification in tract segmentation

34. **Lucena O, Borges P, Cardoso J, Ashkan K, Sparks R, Ourselin S. Informative and reliable tract segmentation for preoperative planning.** *Front Radiol* 2022;2:866974. doi:10.3389/fradi.2022.866974.
    - 3D U-Net, with test-time dropout for model uncertainty, test-time augmentation for data uncertainty, and volume-based calibration.
    - Volume variance correlated with segmentation error, and calibrated volumes were more likely to contain the ground-truth volume. Reference was ground-truth labels, not rescans; no ECE against repeat scans.
35. **Kebiri H, Gholipour A, Bach Cuadra M, Karimi D. Direct segmentation of brain white matter tracts in diffusion MRI.** arXiv:2307.02223 (2023).
    - Segments tracts from DWI and proposes an uncertainty score for detecting failures (judged against Dice). No calibration and no rescans.
    - Verified from the arXiv abstract. The search engine linked a PMC version (PMC10350097), which I could not open.
36. **Brown CJ, Booth BG, Hamarneh G. Uncertainty in tractography via tract confidence regions.** In: *Computational Diffusion MRI and Brain Connectivity* (MICCAI CDMRI 2013), Springer 2014. doi:10.1007/978-3-319-02475-2_12.
    - Builds a volumetric confidence region that should contain the mean tract at a given level, from sampled tract curves.
    - Validated on synthetic data, consistency across 12 subjects, and MS and tumor cases. **No rescan coverage test.** Full text read.
37. **Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks.** ICML 2017; arXiv:1706.04599.
    - Standard reference for ECE and temperature scaling.
38. **Beran R. Prepivoting to reduce level error of confidence sets.** *Biometrika* 1987;74(3):457–68. doi:10.1093/biomet/74.3.457. **Hall P, Martin MA. On bootstrap resampling and iteration.** *Biometrika* 1988;75(4):661–71. doi:10.1093/biomet/75.4.661.
    - The statistical basis for double (iterated) bootstrap calibration.
    - Verified through bibliographic records and secondary descriptions; I could not open the full texts.

### E. Noise correlation and the bootstrap

39. **Robson PM, Grant AK, Madhuranthakam AJ, Lattanzi R, Sodickson DK, McKenzie CA. Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions.** *Magn Reson Med* 2008;60(4):895–907. doi:10.1002/mrm.21728.
    - Parallel imaging gives spatially varying noise amplification.
    - The "pseudo multiple replica" method simulates noise through the real reconstruction, using the measured noise correlation between receiver channels. This is the principled alternative to i.i.d. voxel sign flips.
40. **Dietrich O, Raya JG, Reeder SB, Ingrisch M, Reiser MF, Schoenberg SO. Influence of multichannel combination, parallel imaging and other reconstruction techniques on MRI noise characteristics.** *Magn Reson Imaging* 2008;26(6):754–62. doi:10.1016/j.mri.2008.02.001.
    - Noise statistics depend on how coils are combined and on the parallel-imaging reconstruction, so simple noise models are invalid in general.
    - The paper is about the noise *distribution*, not spatial correlation as such.
41. **Henriques RN, Ianuş A, Novello L, Jovicich J, Jespersen SN, Shemesh N. Efficient PCA denoising of spatially correlated redundant MRI data.** *Imaging Neurosci* 2023;1:1–26. doi:10.1162/imag_a_00049 (bioRxiv 10.1101/2023.03.29.534707).
    - Says spatially correlated noise affects most modern MRI acquisitions and badly degrades MP-PCA denoising.
    - Verified from the bioRxiv abstract. The published version's page returned 403, but its title and DOI appear in search results.
42. **Veraart J et al. Denoising of diffusion MRI using random matrix theory.** *NeuroImage* 2016;142:394–406. doi:10.1016/j.neuroimage.2016.08.016.
    - MP-PCA, which assumes i.i.d. noise. Relevant if the project denoised before computing residuals.
43. **Yuan Y, Zhu H, Ibrahim JG, Lin W, Peterson BS. A note on the validity of statistical bootstrapping for estimating the uncertainty of tensor parameters in diffusion tensor images.** *IEEE Trans Med Imaging* 2008;27(10):1506–14. doi:10.1109/TMI.2008.926069.
    - Theoretical and simulation limits: wild and repetition bootstraps cannot quantify principal-direction uncertainty for isotropic, oblate or multi-tensor voxels.
44. **Chung S, Lu Y, Henry RG. Comparison of bootstrap approaches for estimation of uncertainties of DTI parameters.** *NeuroImage* 2006;33(2):531–41. doi:10.1016/j.neuroimage.2006.07.001.
    - Introduces the residual bootstrap and the repetition bootknife for DTI. Simulation only.
45. **Zhu T, Liu X, Connelly PR, Zhong J. An optimized wild bootstrap method for evaluation of measurement uncertainties of DTI-derived parameters in human brain.** *NeuroImage* 2008;40(3):1144–56. doi:10.1016/j.neuroimage.2008.01.016.
    - Recommends at least 1000 iterations and HC2/HC3 residual scaling. Validated against Monte Carlo simulation, not rescans.
46. **Zhu T et al. Quantification of accuracy and precision of multi-center DTI measurements: a diffusion phantom and human brain study.** *NeuroImage* 2011;56(3):1398–411. doi:10.1016/j.neuroimage.2011.02.010.
    - Wild-bootstrap within-scan variability was about the same size as intra-site and inter-site variability, in a phantom and a traveling human at three sites. Level: tensor metrics.
47. **Vorburger RS, Reischauer C, Dikaiou K, Boesiger P. In vivo precision of bootstrap algorithms applied to diffusion tensor imaging data.** *J Magn Reson Imaging* 2012;36(4):979–86. doi:10.1002/jmri.23733.
    - Repeatability of bootstrap uncertainty estimates over 10 sessions; ICC above 0.96 for the uncertainty values. This measures precision of the estimator, not coverage.
48. **O'Gorman RL, Jones DK. Just how much data need to be collected for reliable bootstrap DT-MRI?** *Magn Reson Med* 2006;56(4):884–90. doi:10.1002/mrm.21014.
    - Simulation: at least about 500 bootstraps and 5 or more repeats are needed for a reliable repetition bootstrap.
49. **Cameron AC, Gelbach JB, Miller DL. Bootstrap-based improvements for inference with clustered errors.** *Rev Econ Stat* 2008;90(3):414–27. doi:10.1162/rest.90.3.414.
    - Cluster bootstrap-t procedures that respect correlated errors within clusters. Statistical precedent for flipping signs on patches rather than single voxels.
    - Verified from the abstract. The phrase "wild cluster bootstrap" is the usual later name; I did not confirm it from the text.
50. **Winkler AM, Ridgway GR, Webster MA, Smith SM, Nichols TE. Permutation inference for the general linear model.** *NeuroImage* 2014;92:381–97. doi:10.1016/j.neuroimage.2014.01.060.
    - Sign-flipping under symmetric errors (randomise). Flipping whole images keeps spatial dependence intact, which is a neuroimaging precedent for coherent sign flips.

### F. Uncertainty propagation from preprocessing (distortion correction)

51. **Andersson JLR, Skare S, Ashburner J. How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging.** *NeuroImage* 2003;20(2):870–88. doi:10.1016/S1053-8119(03)00336-7.
    - The reversed phase-encode (blip-up/down) approach that topup is based on. The field is estimated deterministically, with no uncertainty output.
52. **Irfanoglu MO, Walker L, Sarlls J, Marenco S, Pierpaoli C. Effects of image distortions originating from susceptibility variations and concomitant fields on diffusion MRI tractography results.** *NeuroImage* 2012;61(1):275–88. doi:10.1016/j.neuroimage.2012.02.054.
    - Compared tractography from RL and AP phase-encode acquisitions of the same subjects. EPI distortion changed trajectories substantially, and correction improved consistency.
53. **Gu X, Eklund A. Evaluation of six phase encoding based susceptibility distortion correction methods for diffusion MRI.** *Front Neuroinform* 2019;13:76. doi:10.3389/fninf.2019.00076.
    - Simulation-based comparison of six correction methods. Results differ by method. No propagation into tractography uncertainty.
54. **Begnoche JP, Schilling KG, Boyd BD, Cai LY, Taylor WD, Landman BA. EPI susceptibility correction introduces significant differences far from local areas of high distortion.** *Magn Reson Imaging* 2022;92:1–9. doi:10.1016/j.mri.2022.05.016.
    - Correction (Synb0 inside PreQual) changed FA throughout the brain, not only near air–tissue interfaces.
55. **Gui M et al. A tractography comparison between turboprop and spin-echo echo-planar diffusion tensor imaging.** *NeuroImage* 2008;42(4):1451–62. doi:10.1016/j.neuroimage.2008.05.066.
    - Residual EPI distortion after correction lowered *inter-session* tractography reproducibility near field inhomogeneities. Indirect support for treating the distortion field as a source of uncertainty.
56. **Kiar G, de Oliveira Castro P, Rioux P, Petit E, Brown ST, Evans AC, Glatard T. Comparing perturbation models for evaluating stability of neuroimaging pipelines.** *Int J High Perform Comput Appl* 2020;34(5):491–501. doi:10.1177/1094342020926237. **Kiar G et al. Numerical uncertainty in analytical pipelines lead to impactful variability in brain networks.** *PLoS One* 2021;16(11):e0250755. doi:10.1371/journal.pone.0250755.
    - Perturbation of a whole dMRI connectome pipeline: targeted noise, Monte Carlo Arithmetic and OS variation.
    - This is the closest "whole-pipeline perturbation" precedent, but it models computational rather than measurement noise and has no rescan comparison.
57. **Gu X, Eklund A, Özarslan E, Knutsson H. Using the wild bootstrap to quantify uncertainty in mean apparent propagator MRI.** *Front Neuroinform* 2019;13:43. doi:10.3389/fninf.2019.00043.
    - Uses the wild bootstrap to compare preprocessing pipelines; the bootstrap is applied after preprocessing.
58. **Cai LY et al. PreQual: an automated pipeline for integrated preprocessing and quality assurance of diffusion weighted MRI images.** *Magn Reson Med* 2021;86(1):456–70. doi:10.1002/mrm.28678.
    - The preprocessing pipeline behind MASiVar's distributed preprocessed data. Relevant because the project reruns distortion correction itself.

### G. Probability semantics in probabilistic tractography

59. **Morris DM, Embleton KV, Parker GJM. Probabilistic fibre tracking: differentiation of connections from chance events.** *NeuroImage* 2008;42(4):1329–39. doi:10.1016/j.neuroimage.2008.06.012.
    - Visitation "probability" decays with distance. Proposes null connection maps (Poisson statistics) to separate real connections from distance effects.
60. **Jones DK. Studying connections in the living human brain with diffusion MRI.** *Cortex* 2008;44(8):936–52. doi:10.1016/j.cortex.2008.05.002.
    - Review that sets deterministic against probabilistic tracking. Commonly cited for the point that probabilistic output reflects precision, not anatomical truth.
    - Abstract verified; that specific point is from memory of the text.
61. **Jbabdi S, Johansen-Berg H. Tractography: where do we go from here?** *Brain Connect* 2011;1(3):169–83. doi:10.1089/brain.2011.0033.
    - Critical review: tractography is indirect, inaccurate and hard to quantify.
62. **Jones DK, Knösche TR, Turner R. White matter integrity, fiber count, and other fallacies: the do's and don'ts of diffusion MRI.** *NeuroImage* 2013;73:239–54. doi:10.1016/j.neuroimage.2012.06.081.
    - Warns against over-interpreting streamline counts and similar quantities.
63. **Sotiropoulos SN, Zalesky A. Building connectomes using diffusion MRI: why, how and but.** *NMR Biomed* 2019;32(4):e3752. doi:10.1002/nbm.3752.
    - Limits of connectome reconstruction and of connection weighting.
64. **Maier-Hein KH et al. The challenge of mapping the human connectome based on diffusion tractography.** *Nat Commun* 2017;8:1349. doi:10.1038/s41467-017-01285-x.
    - Simulated ground truth: most algorithms recover 90 % of true bundles but produce more invalid than valid bundles. Accuracy is limited even where reproducibility is high.
65. **Grisot G, Haber SN, Yendiki A. Diffusion MRI and anatomic tracing in the same brain reveal common failure modes of tractography.** *NeuroImage* 2021;239:118300. doi:10.1016/j.neuroimage.2021.118300.
    - Voxel-wise comparison with tracer in the same macaque brain (ROC-type sensitivity and specificity). Tracer validation is ROC-based, not calibration.
66. **Schultz T, Vilanova A, Brecheisen R, Kindlmann G. Fuzzy fibers: uncertainty in dMRI tractography.** arXiv:1307.3271 (book chapter, 2013/2014).
    - Review of sources of uncertainty (noise, artifacts, model, partial volume, preprocessing and parameters), probabilistic tractograms, and visualization.
    - The arXiv page does not name the book; it is commonly cited as being in *Scientific Visualization: Uncertainty, Multifield, Biomedical, and Scalable Visualization* (Springer, 2014), which I did not verify.
67. **Brecheisen R, Platel B, ter Haar Romeny BM, Vilanova A. Illustrative uncertainty visualization of DTI fiber pathways.** *Vis Comput* 2013;29(4):297–309. doi:10.1007/s00371-012-0733-9.
    - "Illustrative confidence intervals" around bundles, based on fiber distance and connection probability, evaluated with three neurosurgeons.
    - Verified from the TU Delft publication record; the Springer page was paywalled.
68. **Yang JY-M, Yeh C-H, Poupon C, Calamante F. Diffusion MRI tractography for neurosurgery: the basics, current state, technical reliability and challenges.** *Phys Med Biol* 2021;66(15). doi:10.1088/1361-6560/ac0d90.
    - Neurosurgical review covering error propagation through the pipeline and safety margins. The figure of about 5 mm for margins came from a search-result summary, not the abstract.

### H. Tractogram compression and soft-label storage

69. **Presseau C, Jodoin P-M, Houde J-C, Descoteaux M. A new compression format for fiber tracking datasets.** *NeuroImage* 2015;109:73–83. doi:10.1016/j.neuroimage.2014.12.058.
    - zfib: linearization, quantization and encoding, with a user-set maximum error. Over 96 % compression at 0.1 mm error.
70. **Haehn D, Franke L, Zhang F, Cetin Karayumak S, Pieper S, O'Donnell LJ, Rathi Y. TRAKO: efficient transmission of tractography data for visualization.** MICCAI 2020, LNCS 12267:322–32; arXiv:2004.13630.
    - glTF container compressed with **Draco** (quantization plus prediction), including per-vertex and per-fiber properties. Over 28× reduction.
    - **This anticipates predictive geometry coding of streamlines.**
71. **Mercier C, Rousseau S, Gori P, Bloch I, Boubekeur T. QFib: fast and efficient brain tractogram compression.** *Neuroinformatics* 2020;18(4):627–40. doi:10.1007/s12021-020-09452-0.
    - Unit-vector quantization of step directions; individual streamlines can be decompressed.
72. **Rheault F, Houde J-C, Descoteaux M. Visualization, interaction and tractometry: dealing with millions of streamlines from diffusion MRI tractography.** *Front Neuroinform* 2017;11:42. doi:10.3389/fninf.2017.00042.
    - Handling compressed streamlines (collinear points removed) without biasing tractometry.
73. **Rheault F et al. TRX: a community-oriented tractography file format.** OHBM 2022 abstract.
    - Header, positions and offsets, plus data per vertex, per streamline and per group; memory-mappable.
    - Verified only from the ResearchGate listing and the TRX spec documentation. Probably not peer reviewed.
74. **Takemura H, Caiafa CF, Wandell BA, Pestilli F. Ensemble tractography.** *PLoS Comput Biol* 2016;12(2):e1004692. doi:10.1371/journal.pcbi.1004692.
    - "Ensemble" in the sense of combining algorithms and parameters. Worth citing to avoid a terminology clash with the project's bootstrap "ensemble".

### Other supporting citations (verified)

- **Malcolm JG, Shenton ME, Rathi Y.** Filtered multitensor tractography. *IEEE Trans Med Imaging* 2010;29(9):1664–75. doi:10.1109/TMI.2010.2048121. The original UKF tractography paper.
- **Yendiki A et al.** Automated probabilistic reconstruction of white-matter pathways in health and disease using an atlas of the underlying anatomy. *Front Neuroinform* 2011;5:23. doi:10.3389/fninf.2011.00023. TRACULA. Its abstract does not report test-retest; I did not find a TRACULA test-retest paper in this search.

---

## 3. Gaps and suggested citations for a write-up

**Gaps the project appears to fill. These are not anticipated by anything found; phrase them as "to our knowledge".**

1. Validating a single-scan bootstrap ensemble against real rescans *at the level of tractography output*: labels, paths and tract maps, with same-session and between-session separated.
2. Showing that i.i.d. voxel-wise wild-bootstrap sign flips understate rescan variability because DWI noise is spatially correlated in-plane, and fixing it with spatially smooth sign patches.
   - Campbell 2014 saw under-dispersion at low FA and attributed it to overfitting and misregistration. The project offers a different, testable mechanism. Cite Campbell explicitly and discuss the difference.
3. Resampling the whole pipeline, including susceptibility-field re-estimation, inside each replicate.
4. Per-streamline tract-label instability on a rescan: 35–50 % of seed-matched streamlines change tract.
5. Calibration (ECE, AUROC) of streamline and voxel tract probabilities against rescans, and double-bootstrap recalibration that transfers across vendors.
6. Empirical coverage of probability-mass envelopes on rescans, and the case against winner-take-all voxel maps for building envelopes.

**Must-cite prior work (closest anticipations), by claim:**

| Claim | Must cite |
|---|---|
| 1 | Campbell 2014; Zhu 2011; Whitcher 2008; Jones 2008; Yuan 2008; Yap 2014; Robson 2008; Henriques 2023; Cameron et al. 2008; Winkler 2014 |
| 2 | Irfanoglu 2012; Andersson 2003; Gu & Eklund 2019; Begnoche 2022; Kiar 2020/2021; Schilling 2021 (NeuroImage 242) |
| 3 | Yu et al. 2024 (seed-ball matching, scan-rescan); Zhang 2019 HBM (UKF + ORG test-retest); Schilling 2021 (42 groups, streamline-based agreement); Xue 2023 TractCloud; Zhang 2020 DeepWMA |
| 4 | Lucena 2022; Kebiri 2023; Guo 2017; Beran 1987; Hall & Martin 1988; Campbell 2014 (coverage of orientation intervals) |
| 5 | Brown, Booth & Hamarneh 2014; Brecheisen 2013; Reid 2020; Besseling 2012; Yang 2021 |
| Probability semantics | Morris 2008; Jones 2008 (Cortex); Jbabdi & Johansen-Berg 2011; Jones, Knösche & Turner 2013; Maier-Hein 2017 |
| Data | Cai 2021 (MASiVar); Cai 2021 (PreQual); Nath 2020 (TraCED) |
| Compression | Presseau 2015; Haehn 2020 (TRAKO/Draco); Mercier 2020 (QFib); TRX (Rheault 2022 abstract) |

**Wording cautions:**

- Do not claim to be first to compare bootstrap uncertainty with repeated scans. Campbell 2014 and the repetition-bootstrap papers did this at the voxel and orientation level.
- Do not claim predictive coding of streamline geometry is new. TRAKO/Draco does it. The new part, if any, is storing ensemble labels as ranked soft labels alongside the geometry.
- Seed-matched streamline correspondence across scan-rescan was used by Yu et al. 2024. The new part is *label* agreement and its link to ensemble probabilities.
- MASiVar itself reports that intrasession < intersession < interscanner variability. Present the project's between-session > same-session finding as consistent with MASiVar, extended to tractography labels and coverage.
- If the labeler is TractCloud, its 42 tracts plus "other" match the training labels described in Xue et al. 2023.

**Suggested follow-up searches (not completed):**

- Full text of Zhang 2019 HBM, for any fiber-level analysis. I could only read the abstract and search snippets.
- Recent IST 2025 proceedings (arXiv:2602.12410), which may contain abstracts on uncertainty or calibration in tractography.
- Kwon, Amorosino, Pestilli 2026 (bioRxiv; doi:10.64898/2026.07.30.741597), an ensemble tract-segmentation benchmark under domain shift. The abstract mentions no calibration against rescans, but check the full text.
- Semantic Scholar forward citations of Campbell 2014 and Jones 2008 ("Tractography gone wild"), to catch any later coverage-against-rescan work. I chained only through PubMed and web search.

**UNVERIFIED:**

- An Aja-Fernández et al. 2014 paper on noise estimation in parallel MRI (GRAPPA/SENSE) is often cited for spatial noise correlation. I did not retrieve or confirm it; Robson 2008 and Henriques 2023 are used instead. PubMed did list Rabanillo-Viloria et al. 2018 (*MRM*, doi:10.1002/mrm.27469), on exact g-factor maps for 3D GRAPPA with correlated k-space noise, as a possible substitute; I saw its abstract but did not annotate it above.
- A knowledge-distillation citation for storing top-k soft labels. Shen & Xing (arXiv:2112.01528, "A Fast Knowledge Distillation Framework for Visual Recognition") is often cited for this, but the abstract does not confirm top-k storage. Do not cite it for that without checking the full text.
- Kiar et al. 2020: whether the "targeted noise" model was compared with session-to-session variability. The abstract does not say.
- Siddiqui et al. 2021, "A progressive approach for uncertainty visualization in diffusion tensor imaging" (*Comput Graph Forum*, doi:10.1111/cgf.14317). Seen only as a search-result title and snippet, which mentions wild-bootstrap ensembles.

---

## 4. Search log

**PubMed (NCBI E-utilities connector): search_articles, get_article_metadata, lookup_article_by_citation, get_full_text_article**

Seed lookups by citation:
- Nath 2020 JMRI 51:234; Whitcher 2008 HBM; Jones 2003 and 2008; Jones & Pierpaoli 2005; Lazar 2005; Berman 2008; Heiervang 2006; Vollmar 2010; Behrens 2003 and 2007; Maier-Hein 2017; Presseau 2015; Jbabdi 2011; Jones 2008 Cortex; Winkler 2014; Veraart 2016; Andersson 2003; O'Gorman 2006.

Queries:
- `Lucena tract segmentation informative reliable`
- `test-retest reproducibility white matter parcellation tractography fiber clustering`
- `wild bootstrap diffusion` (23 hits, all screened)
- `(bootstrap[Title] OR "Monte Carlo"[Title]) AND (diffusion[Title]) AND (repeated OR "test-retest" OR repetition) AND (uncertainty OR variability)`
- `bootstrap[Title/Abstract] AND diffusion[Title/Abstract] AND (reproducibility[Title/Abstract] OR "test-retest"[Title/Abstract]) AND (tractography[Title/Abstract] OR tensor[Title/Abstract])`
- `("residual bootstrap" OR "wild bootstrap" OR "repetition bootstrap") AND ("test-retest" OR "scan-rescan" OR "repeated scans" OR "between-session" OR "inter-session")` (0 hits)
- `nonlocal bootstrap diffusion MRI uncertainty`
- `residual bootstrap constrained spherical deconvolution probabilistic tracking`
- `Jones DK[Author] AND (bootstrap OR uncertainty) AND (fiber OR fibre) AND (2003[pdat] OR 2005[pdat])`
- `MASiVar multisite multiscanner variability diffusion`
- `Schilling bundle segmentation scanner vendor effects acquisition workflow`
- `Tractography dissection variability ... 42 groups ...`
- `Tractostorm tractography dissection reproducibility`
- `Yendiki A[Author] AND (reliability OR reproducibility OR "test-retest") AND tractography`
- `TractSeg fast and accurate white matter tract segmentation`
- `Deep white matter analysis DeepWMA ...`
- `(tractography OR "tract segmentation") AND (calibration OR "conformal prediction" OR "expected calibration error")` (26 hits; the first 20 screened, none relevant)
- `("test-retest" OR "scan-rescan") AND (TractSeg OR RecoBundles OR TRACULA OR "bundle segmentation" OR "tract segmentation") AND (Dice OR overlap)`
- `(uncertainty OR probabilistic) AND tractography AND (test-retest OR scan-rescan) AND (bootstrap OR ensemble)`
- `Kiar G[Author] AND (numerical OR perturbation) AND (diffusion OR connectome OR pipelines)`
- `(susceptibility distortion correction) AND tractography AND (uncertainty OR reproducibility OR variability)`
- `(preprocessing OR pipeline) AND (diffusion MRI) AND tractography AND (variability OR reproducibility) AND (bundle OR tract) AND (distortion correction OR eddy)`
- `EPI susceptibility correction introduces significant differences far from local areas of high distortion`
- `Evaluation of six phase encoding based susceptibility distortion correction methods for diffusion MRI`
- `Rheault F[Author] AND (reproducibility OR Tractostorm OR variability) AND tractography`
- `Morris DM AND probabilistic fibre tracking differentiation of connections from chance events`
- `(Jones DK[Author]) AND (fallacies OR "fiber count") AND white matter integrity`
- `Sotiropoulos SN[Author] AND Zalesky A[Author] AND connectomes`
- `Ensemble tractography Takemura`
- `Malcolm JG[Author] AND unscented Kalman filter tractography`
- `compression tractography streamlines file format OR tractogram compression`
- `Dietrich O[Author] AND noise characteristics parallel imaging multichannel`
- `Aja-Fernández S[Author] AND noise AND (GRAPPA OR parallel) AND correlation` (returned PMIDs but I did not read the specific 2014 paper)
- `Robson PM[Author] AND g-factor pseudo multiple replica`
- `How many streamlines are required for reliable probabilistic tractography`
- `Diffusion MRI tractography for neurosurgery: the basics ...`
- `Campbell JS[Author] AND bootstrap probabilistic tractography complex subvoxel fiber geometries`
- `Cousineau ... PPMI`; `Boukadi ... language fiber bundles`; `Besseling tract specific reproducibility`
- `Benchmarking generalizability deep learning white matter tract segmentation`
- `Tractography with T1-weighted MRI and associated anatomical constraints ...`
- `PreQual ...`
- `TractoFlow ...` (found but not used)

**Web search (standard and extended), each followed by fetching the primary source:**
- wild bootstrap and spatially correlated noise, underestimation, test-retest (extended)
- g-factor and noise correlation in parallel imaging DWI
- tract segmentation uncertainty, calibration and ECE (extended)
- conformal prediction for tractography and bundle segmentation (extended)
- TractCloud 42 tracts and test-retest
- per-streamline label test-retest for DeepWMA-type labelers (extended)
- tractography uncertainty ensemble bootstrap vs scan-rescan, 2024–2025 (extended); this surfaced Campbell 2014
- uncertainty propagation from susceptibility distortion correction / topup (extended)
- Bayesian SDC posterior uncertainty (extended)
- tractography uncertainty neurosurgery safety margins (extended)
- probabilistic tract-map threshold coverage on rescans (extended)
- tractography "credible region" / confidence set / HDR (extended); this surfaced Brown et al. 2014
- probabilistic tractography calibration, reliability diagrams, Brier score (extended)
- streamline correspondence across sessions by seed (extended); this surfaced Yu et al. 2024
- white matter tract segmentation with MC dropout, ensembles and reliability diagrams (extended)
- streamline classification uncertainty with an "Other" class (extended)
- Presseau zfib; TRAKO/Draco; TRX format (standard and extended)
- tractogram compression with predictive coding, 2022–2024
- Cameron, Gelbach & Miller 2008; Beran 1987 / Hall & Martin 1988
- Gruen et al. 2023 CGF; Brecheisen et al. 2013

**Direct fetches (arXiv, ar5iv, publisher and lab PDFs):**
- arXiv 2307.09000 and its ar5iv HTML (TractCloud)
- arXiv 2307.02223 (Kebiri et al.)
- arXiv 2606.09893 (Tractogram foundation model, 2026; no rescan or calibration content in the abstract)
- arXiv 1307.3271 (Fuzzy Fibers)
- arXiv 2004.13630 and its ar5iv HTML (TRAKO)
- arXiv 2303.14371
- arXiv 1706.04599
- arXiv 2112.01528
- Frontiers 10.3389/fneur.2014.00216 (Campbell, full text)
- bioRxiv 10.1101/2023.03.29.534707 (Henriques)
- SFU PDF for Brown et al. (full text, via pdftotext)
- arXiv PDF 2403.18992 (Yu et al., full text, via pdftotext)

**Blocked (403, paywall or browser check):** medRxiv, MIT Press (Imaging Neuroscience), Wiley (HBM full text), PMC article pages, Springer.
