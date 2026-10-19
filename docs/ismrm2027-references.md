<!-- Literature check for the ISMRM 2027 abstract (due 2026-10-28), done 2026-10-19 by a delegated agent on branch
ismrm2027. Every entry in section 1 was checked against a primary record this session: PubMed (E-utilities), Crossref,
Europe PMC or arXiv full text, or the publisher's or dataset's own page. Section 4 says which. Quotes are under 15 words.
This file supersedes the "must-cite" lists in prior-art-uncertainty.md and prior-art.md for the abstract; those files
were not edited. -->

# ISMRM 2027 abstract: verified references, novelty wording, search log

The four claims checked:

1. A whole-pipeline bootstrap ensemble for tractography and automated tract labeling. It applies a wild bootstrap with
   spatially smooth sign patterns, re-estimates the susceptibility field from the resampled b0s, tracks and labels
   each replicate, and reports per-streamline and per-voxel (presence-envelope) uncertainty.
2. Per-voxel sign flips understate test-retest variability, because the noise is spatially correlated within each
   slice. Smooth sign patches close the same-session gap.
3. A single scan's ensemble is overconfident against real rescans. A per-protocol calibration card learned from
   MASiVar cohort III (8 subjects) and evaluated leave-one-subject-out makes the probabilities calibrated for
   same-session and later-session rescans: streamline ECE falls from 0.07–0.10 to 0.008, and envelope coverage holds
   from 0.5 to 0.95.
4. On the ISMRM 2015 phantom, scored with the 2023 ROI rescoring, rescan-calibrated envelopes under-cover the ground
   truth by 17–18 points against a count-matched target, while a bias-free ensemble holds its levels. The gap is
   systematic error and needs its own term.

**Counts.** 66 references verified (section 1). 9 items are UNVERIFIED (section 3). Nothing found anticipates any
claim in full. Three findings change the wording:

- The statistical idea behind claim 2 already exists: a wild bootstrap with spatially dependent multipliers
  (Shao 2010; Kurisu, Kato & Shao 2024). Cite it as the method's basis. Do not call smooth sign patterns new.
- Bootstrap intervals under-covering a known truth has been shown at the ODF level in simulation (Consagra et al.
  2024). Campbell et al. 2014 showed under-dispersion against same-session repeats at low FA.
- The RapidParc paper's first author is **Bisten J**, with von Bornhaupt V second. The two contributed equally.
  The brief and `docs/labelers.md` both put von Bornhaupt first.

---

## 1. Reference list (verified only; ISMRM style)

The abstract will set the superscript order later. The numbers here are for lookup only. The groups follow the claims.
DOIs are included for checking and can be dropped in the abstract.

**Bootstrap uncertainty in dMRI and tractography**

1. Jones DK. Determining and visualizing uncertainty in estimates of fiber orientation from diffusion tensor MRI. Magn Reson Med. 2003;49(1):7-12. doi:10.1002/mrm.10331
2. Jones DK, Pierpaoli C. Confidence mapping in diffusion tensor magnetic resonance imaging tractography using a bootstrap approach. Magn Reson Med. 2005;53(5):1143-1149. doi:10.1002/mrm.20466
3. Whitcher B, Tuch DS, Wisco JJ, et al. Using the wild bootstrap to quantify uncertainty in diffusion tensor imaging. Hum Brain Mapp. 2008;29(3):346-362. doi:10.1002/hbm.20395
4. Jones DK. Tractography gone wild: probabilistic fibre tracking using the wild bootstrap with diffusion tensor MRI. IEEE Trans Med Imaging. 2008;27(9):1268-1274. doi:10.1109/TMI.2008.922191
5. Berman JI, Chung S, Mukherjee P, et al. Probabilistic streamline q-ball tractography using the residual bootstrap. NeuroImage. 2008;39(1):215-222. doi:10.1016/j.neuroimage.2007.08.021
6. Yuan Y, Zhu H, Ibrahim JG, et al. A note on the validity of statistical bootstrapping for estimating the uncertainty of tensor parameters in diffusion tensor images. IEEE Trans Med Imaging. 2008;27(10):1506-1514. doi:10.1109/TMI.2008.926069
7. Haroon HA, Morris DM, Embleton KV, et al. Using the model-based residual bootstrap to quantify uncertainty in fiber orientations from Q-ball analysis. IEEE Trans Med Imaging. 2009;28(4):535-550. doi:10.1109/TMI.2008.2006528
8. Jeurissen B, Leemans A, Jones DK, et al. Probabilistic fiber tracking using the residual bootstrap with constrained spherical deconvolution. Hum Brain Mapp. 2011;32(3):461-479. doi:10.1002/hbm.21032
9. Zhu T, Hu R, Qiu X, et al. Quantification of accuracy and precision of multi-center DTI measurements: a diffusion phantom and human brain study. NeuroImage. 2011;56(3):1398-1411. doi:10.1016/j.neuroimage.2011.02.010
10. Vorburger RS, Reischauer C, Dikaiou K, Boesiger P. In vivo precision of bootstrap algorithms applied to diffusion tensor imaging data. J Magn Reson Imaging. 2012;36(4):979-986. doi:10.1002/jmri.23733
11. Vorburger RS, Reischauer C, Boesiger P. BootGraph: probabilistic fiber tractography using bootstrap algorithms and graph theory. NeuroImage. 2013;66:426-435. doi:10.1016/j.neuroimage.2012.10.058
12. Campbell JSW, MomayyezSiahkal P, Savadjiev P, et al. Beyond crossing fibers: bootstrap probabilistic tractography using complex subvoxel fiber geometries. Front Neurol. 2014;5:216. doi:10.3389/fneur.2014.00216
13. Yap PT, An H, Chen Y, Shen D. Uncertainty estimation in diffusion MRI using the nonlocal bootstrap. IEEE Trans Med Imaging. 2014;33(8):1627-1640. doi:10.1109/TMI.2014.2320947
14. Roine T, Jeurissen B, Perrone D, et al. Reproducibility and intercorrelation of graph theoretical measures in structural brain connectivity networks. Med Image Anal. 2019;52:56-67. doi:10.1016/j.media.2018.10.009
15. Gu X, Eklund A, Özarslan E, Knutsson H. Using the wild bootstrap to quantify uncertainty in mean apparent propagator MRI. Front Neuroinform. 2019;13:43. doi:10.3389/fninf.2019.00043
16. Consagra W, Ning L, Rathi Y. Neural orientation distribution fields for estimation and uncertainty quantification in diffusion MRI. Med Image Anal. 2024;93:103105. doi:10.1016/j.media.2024.103105
17. Manzano-Patrón JP, Deistler M, Schröder C, et al. Uncertainty mapping and probabilistic tractography using simulation-based inference in diffusion MRI: a comparison with classical Bayes. Med Image Anal. 2025;103:103580. doi:10.1016/j.media.2025.103580
18. Takemura H, Caiafa CF, Wandell BA, Pestilli F. Ensemble tractography. PLoS Comput Biol. 2016;12(2):e1004692. doi:10.1371/journal.pcbi.1004692

**Whole-pipeline perturbation and distortion correction**

19. Kiar G, de Oliveira Castro P, Rioux P, et al. Comparing perturbation models for evaluating stability of neuroimaging pipelines. Int J High Perform Comput Appl. 2020;34(5):491-501. doi:10.1177/1094342020926237
20. Kiar G, Chatelain Y, de Oliveira Castro P, et al. Numerical uncertainty in analytical pipelines lead to impactful variability in brain networks. PLoS One. 2021;16(11):e0250755. doi:10.1371/journal.pone.0250755
21. Andersson JLR, Skare S, Ashburner J. How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging. NeuroImage. 2003;20(2):870-888. doi:10.1016/S1053-8119(03)00336-7
22. Smith SM, Jenkinson M, Woolrich MW, et al. Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage. 2004;23(Suppl 1):S208-S219. doi:10.1016/j.neuroimage.2004.07.051
23. Irfanoglu MO, Walker L, Sarlls J, et al. Effects of image distortions originating from susceptibility variations and concomitant fields on diffusion MRI tractography results. NeuroImage. 2012;61(1):275-288. doi:10.1016/j.neuroimage.2012.02.054
24. Gu X, Eklund A. Evaluation of six phase encoding based susceptibility distortion correction methods for diffusion MRI. Front Neuroinform. 2019;13:76. doi:10.3389/fninf.2019.00076
25. Begnoche JP, Schilling KG, Boyd BD, et al. EPI susceptibility correction introduces significant differences far from local areas of high distortion. Magn Reson Imaging. 2022;92:1-9. doi:10.1016/j.mri.2022.05.016

**Correlated noise, and resampling that keeps dependence**

26. Robson PM, Grant AK, Madhuranthakam AJ, et al. Comprehensive quantification of signal-to-noise ratio and g-factor for image-based and k-space-based parallel imaging reconstructions. Magn Reson Med. 2008;60(4):895-907. doi:10.1002/mrm.21728
27. Henriques RN, Ianuş A, Novello L, et al. Efficient PCA denoising of spatially correlated redundant MRI data. Imaging Neurosci. 2023;1:imag-1-00049. doi:10.1162/imag_a_00049
28. Gao C, Yang Q, Kim ME, et al. Characterizing patterns of diffusion tensor imaging variance in aging brains. J Med Imaging. 2024;11(4):044007. doi:10.1117/1.JMI.11.4.044007
29. Cameron AC, Gelbach JB, Miller DL. Bootstrap-based improvements for inference with clustered errors. Rev Econ Stat. 2008;90(3):414-427. doi:10.1162/rest.90.3.414
30. Shao X. The dependent wild bootstrap. J Am Stat Assoc. 2010;105(489):218-235. doi:10.1198/jasa.2009.tm08744
31. Kurisu D, Kato K, Shao X. Gaussian approximation and spatially dependent wild bootstrap for high-dimensional spatial data. J Am Stat Assoc. Published online 2023. doi:10.1080/01621459.2023.2218578 (print issue: see UNVERIFIED item 2)
32. Winkler AM, Ridgway GR, Webster MA, et al. Permutation inference for the general linear model. NeuroImage. 2014;92:381-397. doi:10.1016/j.neuroimage.2014.01.060
33. Bowring A, Telschow F, Schwartzman A, Nichols TE. Spatial confidence sets for raw effect size images. NeuroImage. 2019;203:116187. doi:10.1016/j.neuroimage.2019.116187

**Pipeline components: tracker, labeler, comparators**

34. Malcolm JG, Shenton ME, Rathi Y. Filtered multitensor tractography. IEEE Trans Med Imaging. 2010;29(9):1664-1675. doi:10.1109/TMI.2010.2048121
35. Reddy CP, Rathi Y. Joint multi-fiber NODDI parameter estimation and tractography using the unscented information filter. Front Neurosci. 2016;10:166. doi:10.3389/fnins.2016.00166
36. Zhang F, Wu Y, Norton I, et al. An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. NeuroImage. 2018;179:429-447. doi:10.1016/j.neuroimage.2018.06.027
37. Zhang F, Cetin Karayumak S, Hoffmann N, et al. Deep white matter analysis (DeepWMA): fast and consistent tractography segmentation. Med Image Anal. 2020;65:101761. doi:10.1016/j.media.2020.101761
38. Xue T, Chen Y, Zhang C, et al. TractCloud: registration-free tractography parcellation with a novel local-global streamline point cloud representation. In: MICCAI 2023. Lecture Notes in Computer Science. Springer; 2023:409-419. doi:10.1007/978-3-031-43993-3_40
39. Bisten J, von Bornhaupt V, Grün J, et al. RapidParc: a global-context transformer for parallel, accurate, and lesion-robust tractogram parcellation. Imaging Neurosci. 2026;4:IMAG.a.1168. doi:10.1162/IMAG.a.1168
40. Wasserthal J, Neher P, Maier-Hein KH. TractSeg - fast and accurate white matter tract segmentation. NeuroImage. 2018;183:239-253. doi:10.1016/j.neuroimage.2018.07.070
41. Smith RE, Tournier JD, Calamante F, Connelly A. Anatomically-constrained tractography: improved diffusion MRI streamlines tractography through effective use of anatomical information. NeuroImage. 2012;62(3):1924-1938. doi:10.1016/j.neuroimage.2012.06.005
42. Girard G, Whittingstall K, Deriche R, Descoteaux M. Towards quantitative connectivity analysis: reducing tractography biases. NeuroImage. 2014;98:266-278. doi:10.1016/j.neuroimage.2014.04.074
43. Aydogan DB, Shi Y. Parallel transport tractography. IEEE Trans Med Imaging. 2021;40(2):635-647. doi:10.1109/TMI.2020.3034038
44. Tournier JD, Calamante F, Connelly A. Robust determination of the fibre orientation distribution in diffusion MRI: non-negativity constrained super-resolved spherical deconvolution. NeuroImage. 2007;35(4):1459-1472. doi:10.1016/j.neuroimage.2007.02.016
45. Garyfallidis E, Brett M, Amirbekian B, et al. Dipy, a library for the analysis of diffusion MRI data. Front Neuroinform. 2014;8:8. doi:10.3389/fninf.2014.00008

**Test-retest data and reproducibility of tracts and labels**

46. Cai LY, Yang Q, Kanakaraj P, et al. MASiVar: multisite, multiscanner, and multisubject acquisitions for studying variability in diffusion weighted MRI. Magn Reson Med. 2021;86(6):3304-3320. doi:10.1002/mrm.28926
47. Zhang F, Wu Y, Norton I, et al. Test-retest reproducibility of white matter parcellation using diffusion MRI tractography fiber clustering. Hum Brain Mapp. 2019;40(10):3041-3057. doi:10.1002/hbm.24579
48. Yu T, Li Y, Kim ME, et al. Tractography with T1-weighted MRI and associated anatomical constraints on clinical quality diffusion MRI. Proc SPIE Int Soc Opt Eng. 2024;12926:129262B. doi:10.1117/12.3006286
49. Nath V, Schilling KG, Parvathaneni P, et al. Tractography reproducibility challenge with empirical data (TraCED): the 2017 ISMRM diffusion study group challenge. J Magn Reson Imaging. 2020;51(1):234-249. doi:10.1002/jmri.26794
50. Schilling KG, Tax CMW, Rheault F, et al. Fiber tractography bundle segmentation depends on scanner effects, vendor effects, acquisition resolution, diffusion sampling scheme, diffusion sensitization, and bundle segmentation workflow. NeuroImage. 2021;242:118451. doi:10.1016/j.neuroimage.2021.118451
51. Schilling KG, Rheault F, Petit L, et al. Tractography dissection variability: what happens when 42 groups dissect 14 white matter bundles on the same dataset? NeuroImage. 2021;243:118502. doi:10.1016/j.neuroimage.2021.118502
52. Rheault F, De Benedictis A, Daducci A, et al. Tractostorm: the what, why, and how of tractography dissection reproducibility. Hum Brain Mapp. 2020;41(7):1859-1874. doi:10.1002/hbm.24917
53. Rheault F, Schilling KG, Valcourt-Caron A, et al. Tractostorm 2: optimizing tractography dissection reproducibility with segmentation protocol dissemination. Hum Brain Mapp. 2022;43(7):2134-2147. doi:10.1002/hbm.25777

**Calibration and uncertainty in segmentation**

54. Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian binning. Proc AAAI Conf Artif Intell. 2015;29(1). doi:10.1609/aaai.v29i1.9602 (page range left out: see UNVERIFIED item 3)
55. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of the 34th International Conference on Machine Learning. PMLR. 2017;70:1321-1330.
56. Jungo A, Reyes M. Assessing reliability and challenges of uncertainty estimations for medical image segmentation. In: MICCAI 2019. Lecture Notes in Computer Science. Springer; 2019:48-56. doi:10.1007/978-3-030-32245-8_6
57. Mehrtash A, Wells WM, Tempany CM, et al. Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. IEEE Trans Med Imaging. 2020;39(12):3868-3878. doi:10.1109/TMI.2020.3006437
58. Angelopoulos AN, Bates S. Conformal prediction: a gentle introduction. Found Trends Mach Learn. 2023;16(4):494-591. doi:10.1561/2200000101
59. Lucena O, Borges P, Cardoso J, et al. Informative and reliable tract segmentation for preoperative planning. Front Radiol. 2022;2:866974. doi:10.3389/fradi.2022.866974
60. Lucena O, Lavrador JP, Irzan H, et al. Assessing informative tract segmentation and nTMS for pre-operative planning. J Neurosci Methods. 2023;396:109933. doi:10.1016/j.jneumeth.2023.109933
61. Brown CJ, Booth BG, Hamarneh G. Uncertainty in tractography via tract confidence regions. In: Computational Diffusion MRI and Brain Connectivity (MICCAI CDMRI 2013). Mathematics and Visualization. Springer; 2014:129-138. doi:10.1007/978-3-319-02475-2_12

**Phantom ground truth and the limits of accuracy**

62. Maier-Hein KH, Neher PF, Houde JC, et al. The challenge of mapping the human connectome based on diffusion tractography. Nat Commun. 2017;8(1):1349. doi:10.1038/s41467-017-01285-x
63. Renauld E, Théberge A, Petit L, et al. Validate your white matter tractography algorithms with a reappraised ISMRM 2015 Tractography Challenge scoring system. Sci Rep. 2023;13(1):2347. doi:10.1038/s41598-023-28560-w
64. Thomas C, Ye FQ, Irfanoglu MO, et al. Anatomical accuracy of brain connections derived from diffusion MRI tractography is inherently limited. Proc Natl Acad Sci U S A. 2014;111(46):16574-16579. doi:10.1073/pnas.1405672111
65. Schilling KG, Nath V, Hansen C, et al. Limits to anatomical accuracy of diffusion tractography using modern approaches. NeuroImage. 2019;185:1-11. doi:10.1016/j.neuroimage.2018.10.029
66. Rheault F, Poulin P, Valcourt Caron A, et al. Common misconceptions, hidden biases and modern challenges of dMRI tractography. J Neural Eng. 2020;17(1):011001. doi:10.1088/1741-2552/ab6aad

Corrections to details given in the older docs or in the brief:

- **ACT (Smith 2012):** the DOI is 10.1016/j.neuroimage.2012.06.005, per PubMed.
- **ORG atlas (Zhang 2018):** the DOI is 10.1016/j.neuroimage.2018.06.027, pages 429–447.
- **RapidParc:** Bisten is first author, sharing first authorship with von Bornhaupt (equal-contribution note in the full
  text). PubMed and Crossref both put Bisten first. Cite it as "Bisten J, von Bornhaupt V, et al.", not "von
  Bornhaupt et al."
- **Haroon 2009:** published online in 2008; the issue is 2009;28(4).
- **Yu 2024:** the SPIE article number is 129262B, from PubMed.
- **Reddy & Rathi 2016:** this paper describes the unscented *information* filter with NODDI, not plain two-tensor UKF.
  For "two-tensor UKF" cite Malcolm 2010 [34]. Cite [35] only if the shipped UKF build is that variant.
- **Kiar 2020:** it is indexed in PubMed (PMID 32831546). Its perturbation models are targeted noise, Monte Carlo
  Arithmetic and operating-system variation, all applied to a connectome pipeline.
- **MASiVar cohort III:** "8 adult subjects all scanned on one scanner", 1–6 sessions each, two scans per session
  (OpenNeuro ds003416 README). The abstract's 8 subjects match.

### What each key reference shows (checked this session)

| # | Shows | Source checked |
|---|---|---|
| 12 Campbell 2014 | Residual-bootstrap 68 % and 95 % intervals of fODF peaks were checked against the other three of four same-session repeats (one subject, no repositioning). Correspondence was "very good" where FA > 0.3. At low FA the bootstrap "underestimates the scan-rescan repeatability", which the authors put down to SH-order overfitting and slight misregistration. Spatial noise correlation is not discussed | Europe PMC full text |
| 48 Yu 2024 | "Epsilon ball seeding": for each streamline in A, the streamlines of B that pass within 0.5 voxel of its seed; 100-point resampling; minimum MDF. Scan-to-rescan gives a mean of medians of 3.16 mm, used as a reference for SDStream variability. Geometry only, no labels | arXiv full text |
| 19, 20 Kiar 2020, 2021 | Whole-pipeline perturbation of a structural connectome pipeline by targeted noise, Monte Carlo Arithmetic and OS variation (2020), and MCA throughout (2021). Stability ranged from all digits significant to 0–1. Numerical, not a measurement-noise model; no rescan comparison and no distortion-field step | PubMed abstracts |
| 4 Jones 2008 | Wild-bootstrap tracking from one acquisition, comparable to regular bootstrap tracking in vivo | PubMed abstract |
| 3 Whitcher 2008 | Wild bootstrap introduced for DTI with one observation per direction; compared with the regular bootstrap in simulation and in vivo | PubMed abstract |
| 7 Haroon 2009 | The model-based residual bootstrap with q-ball "closely" matched the conventional (repetition) bootstrap | PubMed abstract |
| 8 Jeurissen 2011 | Residual-bootstrap CSD tractography; accuracy and precision from Monte Carlo simulation; clinical data; no rescans | PubMed abstract |
| 10 Vorburger 2012 | Precision of four bootstrap methods over 10 sessions (ICC > 0.96 for the uncertainty values). This is repeatability of the estimator, not coverage | PubMed abstract |
| 11 Vorburger 2013 | BootGraph: graph-based tractography with edge weights from wild-bootstrap cones of uncertainty; "high repeatability" | PubMed abstract |
| 9 Zhu 2011 | Wild-bootstrap within-scan variability was "of the same magnitude" as intra- and inter-site variability; tensor level, phantom and one traveling subject | PubMed abstract |
| 14 Roine 2019 | Residual bootstrap used for intrasubject variability of connectome metrics; replicated on HCP test-retest (44 subjects); no coverage test | PubMed abstract |
| 15 Gu 2019 | Wild bootstrap for MAP-MRI metrics; used to compare preprocessing pipelines. The bootstrap is applied after preprocessing | PubMed abstract |
| 16 Consagra 2024 | In simulation, the residual-bootstrap pointwise 95 % ODF intervals show "severe under-coverage" with sparse sampling (M = 10), measured by empirical coverage proportion against known truth | arXiv full text |
| 17 Manzano-Patrón 2025 | SBI trained in silico gives "calibrated" posteriors on simulations. SBI and MCMC tractography agree at or above scan-rescan reproducibility. Calibration is against simulation, not rescans | PubMed abstract |
| 39 Bisten 2026 (RapidParc) | Transformer labeler on TractCloud's 42 tracts + Other; 94.44 % accuracy on the test set. An appendix figure shows streamlines assigned to more than one class over 20 runs (context draws), mostly at the Other boundary. That is algorithmic variability, not rescans; no calibration | Europe PMC full text |
| 38 Xue 2023 (TractCloud) | Registration-free streamline parcellation in subject space using local and global streamline context | arXiv abstract, Crossref |
| 46 Cai 2021 (MASiVar) | 319 scans; variability increases from intrasession to intersession to interscanner to intersubject, across DTI, NODDI, bundles and connectomes | PubMed abstract, dataset README |
| 62 Maier-Hein 2017 | Most submissions recover 90 % of ground-truth bundles but produce many more invalid than valid bundles, half of them systematic across groups | PubMed abstract |
| 63 Renauld 2023 | New ROI-based segmentation and scoring for the ISMRM 2015 phantom; 96 submissions rescored; cleaner ground-truth tractogram | PubMed abstract |
| 36 Zhang 2018 (ORG) | 800-cluster atlas, 58 deep tracts plus superficial clusters; false-positive clusters annotated | PubMed abstract |
| 47 Zhang 2019 | Test-retest of fiber-clustering parcellation with the curated atlas, 255 subjects in 3 datasets; wDice and FA difference at parcel and tract level. That the tracker was UKF is not stated in the abstract (UNVERIFIED item 6) | PubMed abstract |
| 30, 31 Shao 2010; Kurisu 2024 | Dependent wild bootstrap: multipliers are themselves dependent so that resampling keeps the data's dependence. Shao did this for time series. Kurisu et al. extend it to space, generating the multipliers from an auxiliary random field: the SDWB multipliers "are spatially dependent" | Kurisu: arXiv full text; Shao: abstract via institutional record |
| 29 Cameron 2008 | Cluster bootstrap-t procedures for few clusters with within-cluster correlated errors | NBER abstract (summary) |
| 32 Winkler 2014 | Sign-flipping permutation under symmetric errors (randomise) | PubMed abstract |
| 33 Bowring 2019 | Spatial confidence sets with empirical coverage, from a wild t-bootstrap that multiplies each subject's whole residual image by one Rademacher sign. This keeps spatial dependence. Coverage is tested in 3D simulations | Europe PMC full text |
| 26, 27, 28 Robson 2008; Henriques 2023; Gao 2024 | Parallel imaging gives spatially varying noise amplification (Robson). Spatially correlated noise affects "most modern MRI acquisitions" (Henriques). DTI "exhibits spatially varying and correlated noise" (Gao) | PubMed and bioRxiv abstracts |
| 6 Yuan 2008 | Wild and repetition bootstraps cannot quantify principal-direction uncertainty for isotropic or oblate tensors | PubMed abstract |
| 23, 24, 25 Irfanoglu 2012; Gu & Eklund 2019; Begnoche 2022 | EPI distortion changes tractography substantially; correction methods differ; correction changes FA far from high-distortion areas. None propagates noise through field estimation | PubMed abstracts |
| 59, 60 Lucena 2022, 2023 | 3D U-Net with test-time dropout and augmentation, then volume-based calibration against ground-truth labels; no ECE and no rescans (2022 full text checked). In 2023, uncertainty correlates with distance to nTMS responses | Europe PMC full text; PubMed abstract |
| 56 Jungo & Reyes 2019 | Voxel-wise uncertainty is well calibrated at dataset level but miscalibrated at subject level, measured by ECE and reliability diagrams | arXiv full text |
| 57 Mehrtash 2020 | FCNs for segmentation are overconfident; ensembling improves calibration | PubMed abstract |
| 55 Guo 2017 | Modern networks are miscalibrated; temperature scaling. Defines ECE in the body, crediting Naeini 2015 | PMLR page, arXiv full text |
| 61 Brown 2014 | A volumetric region that contains the *mean* tract at a given confidence; synthetic and real data; no rescan coverage | SFU full text |
| 64, 65 Thomas 2014; Schilling 2019 | Against tracer and phantom ground truth, no method was highly accurate: accuracy has inherent limits | PubMed abstracts |
| 18 Takemura 2016 | "Ensemble tractography" means combining tracking algorithms and parameters. This clashes with "ensemble" for bootstrap replicates | PubMed abstract |

---

## 2. Claims, closest prior work, safe wording

| Claim | Closest prior work | Status | Safe novelty wording |
|---|---|---|---|
| **1. Whole-pipeline bootstrap ensemble** (smooth-sign wild bootstrap, susceptibility field re-estimated per replicate, tracking and labeling per replicate, per-streamline and per-voxel uncertainty) | Bootstrap tractography that resamples already-preprocessed data and refits only the local model [2–5, 8, 11, 13]. Whole-pipeline *numerical* perturbation, with no data-noise model [19, 20]. Bootstrap used to *compare* preprocessing pipelines, not run through them [15]. Label consistency over random context draws in the labeler's own paper [39]. Envelope-like regions for the mean tract [61] | Not anticipated as a whole. Every part has precedent (wild bootstrap tracking; field estimation from reversed b0s [21]) | "Bootstrap tractography has resampled preprocessed data and refit the local model;^{2–4,8} whole pipelines have been perturbed numerically.^{19,20} To our knowledge, data-noise resampling has not been propagated through susceptibility-field re-estimation, tracking and automated tract labeling." Do not say "ensemble tractography" without separating the term from Takemura et al.^{18}; "bootstrap ensemble" or "resampled-pipeline ensemble" is safer |
| **2. Per-voxel sign flips understate test-retest variability; smooth sign patches close the same-session gap** | Under-dispersion against same-session repeats at low FA, put down to overfitting and misregistration [12]. Wild bootstrap of the same size as site variability, tensor level [9]. Spatially correlated, non-stationary noise [26–28]. Validity limits of the wild bootstrap [6]. **Statistical precedent for the fix: dependent and spatially dependent wild bootstrap [30, 31]**, cluster bootstrap [29], whole-image sign flips [32, 33] | Partly anticipated. The resampling idea is established statistics (SDWB). The dMRI application, the attribution to in-plane correlation and the rescan test were not found | "Spatially dependent multipliers are an established remedy for dependent errors;^{30,31} we found no dMRI study that attributes bootstrap under-dispersion against rescans to in-plane noise correlation^{12} or applies dependent sign patterns to diffusion data." Avoid "novel sign patches" and "first wild bootstrap with spatial structure" |
| **3. Single-scan ensemble overconfident against rescans; per-protocol calibration card (MASiVar cohort III, LOSO); ECE 0.07–0.10 → 0.008; envelope coverage 0.5–0.95** | Interval coverage against same-session repeats, orientation level [12]. Bootstrap ODF intervals under-cover a simulated truth [16]. SBI posteriors calibrated in silico and compared with scan-rescan reproducibility [17]. Tract-segmentation probabilities calibrated against labels by volume, no ECE, no rescans [59, 60]. ECE and subject-level miscalibration in segmentation [55–57]. Variability ordering in MASiVar [46]. Label and tract test-retest without probabilities [47–51] | Not anticipated. Calibration of tractography or tract-label probabilities against real rescans was not found, nor was a per-protocol recalibration map | "To our knowledge, the calibration of tractography and tract-label probabilities against real rescans has not been measured: earlier checks used same-session repeats at the orientation level^{12} or simulated truth.^{16,17} A per-protocol map learned from test-retest subjects^{46} and validated leave-one-subject-out makes them calibrated for same- and other-session rescans." Use "to our knowledge"; do not say "first calibrated tractography", since Manzano-Patrón et al. use "calibrated" for in-silico posteriors^{17} |
| **4. Phantom: rescan-calibrated envelopes under-cover ground truth by 17–18 points; a bias-free ensemble holds its levels; systematic error needs a separate term** | Systematic invalid bundles and limited accuracy against ground truth [62, 64–66]. Rescored phantom [63]. Confidence regions for the mean tract, synthetic validation [61]. Spatial confidence sets with simulated coverage [33]. Bootstrap under-coverage of ODFs against simulated truth [16] | Not anticipated. The bias-versus-variance point is well known qualitatively [62, 64, 66]. The quantitative coverage gap of rescan-calibrated envelopes against phantom truth, and its split into a separate term, were not found | "Tractography's systematic errors on this phantom are well documented.^{62,63} We found no study that tests rescan-calibrated uncertainty against a ground truth; on the rescored phantom,^{63} it under-covers by 17–18 points, so precision and accuracy need separate terms." Avoid implying that bias is a new observation |

Notes for the text:

- MASiVar itself finds intrasession < intersession variability [46]. Present the same-session and other-session result
  as consistent with it.
- Seed-ball matching across rescans is Yu et al.'s [48]. If the abstract mentions seed-matched label change, credit it.
- "Calibration" and "ECE" need [55] or [54]. ECE as a metric for segmentation calibration is in [56, 57].
- If space is short, the minimum set is 4, 12, 16, 19, 21, 31, 39, 46, 48, 55, 62 and 63, plus 34 for the tracker.

---

## 3. UNVERIFIED

1. **TractCloud LNCS volume number.** Search results credit a BibSonomy record with vol. 14227. Springer's page was
   blocked and dblp did not answer. Crossref confirms the DOI, pages 409–419 and the MICCAI 2023 book. Reference 38
   gives no volume.
2. **Kurisu, Kato & Shao print citation**, vol. 119, no. 547, pp. 1820–1832 (2024). This comes from the RePEc and
   IDEAS listing via a search summary. Crossref gives only the 2023 online version.
3. **Naeini 2015 page range.** Crossref and PubMed give none. 2901–2907 is from memory and is left out of reference
   54. The paper and DOI are verified.
4. **Cameron, Gelbach & Miller 2008 and the name "wild cluster bootstrap".** The abstract (NBER summary) says
   "cluster bootstrap-t". That the paper introduces the *wild* cluster bootstrap was not confirmed from the text.
   Cite it for cluster-level resampling only.
5. **Shao 2010 multiplier construction** (Gaussian multipliers with kernel covariance). Only secondary sources and
   Kurisu et al.'s description were read; the full text was not.
6. **Zhang 2019 HBM tracker.** That it used two-tensor UKF is not in the abstract. The full text was not open on
   Europe PMC or the PMC OAI service.
7. **Kwon, Amorosino & Pestilli 2026** (bioRxiv doi:10.64898/2026.07.30.741597, "Benchmarking generalizability in
   deep learning-based white matter tract segmentation"). Only a Sciety summary was seen, and the authors were not
   confirmed. It proposes an ensemble (EWMTS) for domain shift; the summary mentions no calibration or rescans.
   Do not cite it.
8. **Mossina & Friedrich 2025** (MICCAI 2025, LNCS pp. 78–88, doi:10.1007/978-3-032-04965-0_8): conformal prediction
   sets for segmentation by morphological dilation. Bibliographic details are confirmed by Crossref; the content only
   by a search summary. It would be the citation for "conformal segmentation", if one is needed.
9. **From the older docs, still open and not needed for claims 1–4:** Aja-Fernández et al. 2014 on parallel-imaging
   noise; Siddiqui et al. 2021 (CGF); top-k soft-label citations; the book that holds the Fuzzy Fibers chapter.

Checked but not cited: DDEvENet (arXiv 2409.07020) is gray-matter parcellation, not tracts. Karimi et al. 2024
(Imaging Neurosci) gives voxel posteriors for dMRI models. Hainline et al. 2019 (MRI) uses deep nets for GFA bias and
variance. FIESTA (Dumais 2023) and Young 2024 (tractfinder) do segmentation without calibration. In the IST 2025
proceedings (arXiv 2602.12410), one abstract bounds the binomial variance of connection probability; none is on
bootstrap calibration or rescan coverage.

---

## 4. Search log (2026-10-19)

**Bibliographic verification**
- PubMed via the NCBI connector (`lookup_article_by_citation`, `get_article_metadata`) and E-utilities
  (esearch/esummary/efetch): Campbell 2014; Jones 2003, 2008; Jones & Pierpaoli 2005; Whitcher 2008; Berman 2008;
  Yuan 2008; Haroon 2009; Jeurissen 2011; Zhu 2011; Vorburger 2012, 2013; Yap 2014; Roine 2019; Gu 2019; Gu & Eklund
  2019; Cai 2021; Maier-Hein 2017; Renauld 2023; Kiar 2020, 2021; Malcolm 2010; Reddy & Rathi 2016; Wasserthal 2018;
  Smith 2012; Girard 2014; Aydogan & Shi 2021; Andersson 2003; Smith 2004; Rheault 2020 (HBM), 2022, 2020 (J Neural
  Eng); Winkler 2014; Zhang 2018, 2019, 2020; Lucena 2022, 2023; Mehrtash 2020; Yu 2024; RapidParc; Robson 2008;
  Irfanoglu 2012; Begnoche 2022; Schilling 2019, 2021 (×2); Nath 2020; Tournier 2007; Garyfallidis 2014; Bowring
  2019; Gao 2024; Thomas 2014; Takemura 2016; Naeini 2015; Manzano-Patrón 2025; Kebiri 2023.
- Crossref REST API: Kiar 2020; TractCloud; Cameron 2008; Brown 2014; Jungo & Reyes 2019; Naeini 2015; Henriques
  2023; Dell'Acqua 2025; RapidParc (author order); Shao 2010; Kurisu 2023; Davidson & Flachaire 2008; Wu 1986;
  Mammen 1993; Sommerfeld 2018; Angelopoulos & Bates 2023; Mossina & Friedrich 2025; Thomas 2014; Schilling 2019;
  Consagra 2024; Rheault 2020 (J Neural Eng). Zenodo 572345 is not in Crossref.
- Full texts read and grepped: Campbell 2014 (Europe PMC), RapidParc (Europe PMC), Lucena 2022 (Europe PMC), Bowring
  2019 (Europe PMC), Yu 2024 (arXiv 2403.18992), Consagra 2024 (arXiv 2307.08138), Kurisu (arXiv 2103.10720), Guo
  2017 (arXiv 1706.04599), Jungo & Reyes (arXiv 1907.03338), Brown 2014 (SFU PDF), IST 2025 proceedings (arXiv
  2602.12410).
- Other pages: PMLR v70 (Guo); arXiv abstract pages (TractCloud, Consagra, Kurisu); NBER t0344 (Cameron); bioRxiv
  API (Henriques); OpenNeuro ds003416 README via GitHub (MASiVar cohorts).
- Not reachable: MIT Press (Imaging Neuroscience, RESTAT), Springer (TractCloud), Semantic Scholar abstracts
  (publisher elided), the Europe PMC full text for MASiVar and Zhang 2019, dblp (no response).

**Novelty searches (PubMed, E-utilities)**
- `(tractography OR "tract segmentation" OR "bundle segmentation" OR parcellation) AND (calibration OR calibrated OR "conformal prediction" OR "expected calibration error") AND 2022:2026[dp]`: 18 hits; only Lucena 2022 and Manzano-Patrón 2025 are relevant.
- `(tractography OR "tract segmentation" OR "bundle") AND (uncertainty OR ensemble OR bootstrap) AND ("test-retest" OR "scan-rescan" OR rescan OR repeatability)`, 2022 on: 13 hits; Manzano-Patrón 2025.
- `(tractography OR "bundle segmentation" OR "tract segmentation") AND uncertainty AND (deep OR "neural network" OR Bayesian OR ensemble) AND 2022:2026[dp]`: 9 hits; Lucena 2023, Karimi 2024, Kebiri 2023.
- `(RecoBundles OR TractSeg OR "white matter analysis" OR TractCloud OR DeepWMA OR XTRACT OR TRACULA) AND (uncertainty OR probability OR confidence) AND 2021:2026[dp]`: 10 hits; none on label uncertainty.
- `("wild bootstrap" OR "residual bootstrap" OR "block bootstrap") AND (diffusion OR DWI OR MRI) AND (spatial OR correlated OR correlation)`, 2015 on: 8 hits; none on spatially correlated multipliers.
- `("susceptibility" OR topup OR "distortion correction") AND (uncertainty OR bootstrap) AND diffusion AND (field OR tractography)`: 8 hits, none relevant. Variants on Bayesian or uncertain distortion fields: 0 hits.
- `bootstrap[tiab] AND tractography[tiab] AND 2018:2026[dp]`: 6 hits, none relevant.
- `tractography phantom ground truth uncertainty coverage`; `wild bootstrap spatial fMRI`; `tractography systematic bias uncertainty calibration accuracy`: 0, 2 and 0 hits, none relevant.
- `TractCloud Xue`: found TractCloud-FOV (Chen et al., HBM 2025;46(5):e70201), with no uncertainty content.

**Forward citations (Semantic Scholar API, rate-limited)**
- Campbell 2014 (the first 100 citing papers, 2019 on): no coverage-against-rescan work.
- Jones 2008 "Tractography gone wild" (the first 100): Consagra 2023/2024, Manzano-Patrón 2025, Gruen 2022/2023
  (consensus), visualization papers; none tests calibration against rescans.
- Whitcher 2008 (the first 100): Gao 2024 (correlated noise), Harms 2023, µGUIDE, Hainline 2019, "Diffusion tensor
  estimation with uncertainty calibration" (2021, voxel level, not followed up); none on spatially dependent
  resampling.

**Web searches (standard and extended), primary sources then fetched**
- Conformal prediction for white matter tract segmentation, 2025: none for tracts; Mossina & Friedrich 2025 (general
  segmentation).
- Bootstrap tractography calibrated against test-retest, 2025–2026: none.
- Wild bootstrap, spatially correlated noise, patch sign flips, dMRI: none in dMRI; then the statistics literature,
  which led to Shao 2010 and Kurisu et al.
- Kwon, Amorosino & Pestilli 2026 benchmark: Sciety summary only.
- IST 2025 proceedings: the PDF was grepped for bootstrap, calibration, conformal, rescan, uncertainty and ensemble.
- arXiv: tractography uncertainty, calibration, test-retest, streamline classification: DDEvENet, Consagra; nothing on
  label calibration.
- Susceptibility-field uncertainty propagated into tractography: none.
- Uncertainty in automated bundle labeling (RecoBundles, TractSeg MC dropout): TractSeg's repository offers an MC-dropout
  uncertainty map (repository text, not peer reviewed); Lucena 2022; BundleSeg; none calibrated against rescans.
- ISMRM abstracts 2024–2026 on bootstrap tractography coverage: none found by search. The proceedings archive was not
  browsed directly.
- Bias and variance on the ISMRM 2015 phantom: Maier-Hein 2017, Rheault 2020 (J Neural Eng); no coverage test of
  uncertainty against the phantom truth.

**Limits.** This was not a systematic review. The ISMRM 2024–2026 proceedings were not browsed abstract by abstract.
Semantic Scholar paging stopped at 100 citing papers per seed paper because of rate limits.
