# Prior art: tractography geometry, parcellation confidence, compact soft output

*Compiled 2026-10-01 from three literature searches (primary sources: papers, specs, code).
Each claim links its source. "Unverified" marks what the searches could not confirm. Numbers
derived here rather than reported are marked "derived".*

## 1. Streamline geometry: storage and compression

**Formats.**
- **TRX** ([spec](https://github.com/tee-ar-ex/trx-spec/blob/master/specifications.md)) stores
  float16/32/64 RAS positions, offsets, and dpv/dps/groups/dpg in a zip or a folder. It has no
  quantization, prediction, levels of detail or spatial index.
  - Published as an OHBM 2022 abstract, with a 2023 PDF on ResearchGate (Rheault et al.). A
    journal version is unverified.
  - A [2026 benchmark repo](https://github.com/tee-ar-ex/trx-manuscript-2026-benchmark) compares
    load and save times across languages and commits no results yet.
- **BIDS BEP046** (tractography): the draft mandates `_tractogram.trx`, in an open PR
  ([#2333](https://github.com/bids-standard/bids-specification/pull/2333)). TRX is the interop
  target whatever else is built.
- **DSI Studio `.tt.gz`** ([TinyTrack](https://github.com/frankyeh/DSI-Studio/blob/master/libs/tracking/tract_model.cpp))
  is the closest prior art in practice, as shipped code with no paper.
  - Positions are fixed point at 1/32 voxel, with an int32 first point per streamline and
    **first-order int8 deltas**, then gzip.
  - A delta over ±127 is split by inserting points, which changes the vertex count, so it is
    not lossless.
  - NiiVue reads it in the browser.
- `.tck`, `.trk`, Camino and NiBabel store raw floats with no compression. VTK XML allows
  integer arrays and a compressor.

**Compression papers.**

| method | idea | reported size and error |
|---|---|---|
| Presseau et al., [NeuroImage 2015](https://doi.org/10.1016/j.neuroimage.2014.12.058) (.zfib) | linearization (point removal), quantization, DCT/wavelet, arithmetic coding | > 96 % at 0.1 mm max error; [ISMRM 2014](https://cds.ismrm.org/protected/14MProceedings/PDFfiles/2590.pdf): 96.7-99.6 % at 0.5 mm max |
| Rheault et al., [Front. Neuroinform. 2017](https://doi.org/10.3389/fninf.2017.00042) | linearization only, now dipy's `compress_streamlines` | up to 85 % of points removed at 0.1 mm; point-based ROI queries then miss streamlines |
| QFib, Mercier et al., [Neuroinformatics 2020](https://doi.org/10.1007/s12021-020-09452-0) | constant step; each direction quantized relative to the previous one (8-bit Fibonacci or 16-bit octahedral) | about 1 B/vertex at 0.1 mm steps, 18 µm mean / 47 µm max (iFOD1); error drifts |
| Fiblets, Schertzer et al., [CGF 2022](https://perso.telecom-paristech.fr/boubek/papers/Fiblets/Fiblets_lowres.pdf) | QFib in 60-point blocks for the GPU, re-anchored | about 1.3 B/vertex, 5 µm mean / 17 µm max |
| TRAKO, Haehn et al., [MICCAI 2020](https://arxiv.org/abs/2004.13630) | glTF + Draco point cloud: 14-bit quantization, difference prediction, attributes | paper: 10-28×, 0.08-0.15 mm mean (its table sizes contradict its vertex counts). **Measured here** on the HCP tractogram at matched grids: raw Draco 3.27 / 2.54 / 2.17 B/vertex at 14 / 12 / 11 bits, 1.38-1.62× the predictive encoding; decodes 4-4.7× slower in JS (`trako.json`, `decode_js.json`) |
| dictionary / prototype methods (Alexandroni 2017, Zimmerman Moreno 2016, Gori 2016) | sparse codes, prototypes | lossy at the mm level |

No tractogram-compression paper from 2021-2026 was found.

**Adjacent formats.**
- **Zarr Vectors** ([Allen Institute draft](https://github.com/AllenInstitute/zarr_vectors),
  active 2026-09) makes streamlines a first-class geometry in Zarr, with spatial chunks, a
  fragment index, cross-chunk links and multiscale "metavertices". Its codec is Blosc(zstd) with
  byte shuffle, with **no prediction**. It is the container half of the "from scratch" design,
  already drafted.
- **Neuroglancer** [skeletons](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/skeletons.md):
  float32, sharding, no LOD. Its [multiresolution meshes](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md)
  have an octree LOD and quantized positions.
- **COPC** ([copc.io](https://copc.io/)): integer coordinates, a clustered octree LOD, read by
  range requests.
- **glTF [EXT_meshopt_compression](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_meshopt_compression/README.md)**:
  zigzag deltas that work on line strips. Draco has no polylines.
- **Spatial queries:** WMQL uses an AABB tree
  ([PMC4940319](https://pmc.ncbi.nlm.nih.gov/articles/PMC4940319)); Fast Streamline Search
  ([2022](https://doi.org/10.1007/s12021-022-09590-7)). Apart from Zarr Vectors' chunks, no
  format stores a spatial index.

**Where geometry_bench.py stands.** Nobody reports **second-order prediction, lossless at a
stated grid, with modern entropy coding**. On the same HCP tractogram at 0.05 mm, under the same coder
(bitshuffle, zstd level 9), first-order int8 deltas (the `.tt.gz` idea) take 2.08 B/vertex.
Second-order prediction takes **1.55 B/vertex**, 25 % less (`geometry.json`). Per millimeter of streamline that is about 0.9 B/mm (derived).
QFib and Fiblets are about 10-13 B/mm at 0.1 mm steps (derived), but that comparison is unfair in
both directions: their data is 18× denser along each streamline and their coders are lossy. A
fair comparison needs one tractogram encoded by both. Only lossy methods with 10-20× coarser
tolerance, such as Presseau's at 0.5 mm max, are smaller.

**Gaps nobody covers:** lossless or grid-exact predictive coding; benchmarks at the 1.5-2 mm
steps of UKF and deterministic trackers; quantized or predictive positions in TRX and BEP046;
predictive coding of per-vertex attributes; per-streamline random access under entropy coding;
a standard LOD and spatial index for tractograms.

## 2. Confidence and reproducibility of streamline parcellation

- **TractCloud** ([MICCAI 2023](https://arxiv.org/abs/2307.09000)) states its random global
  context in the paper (w = 500; the CLI uses 80). Neither the paper, nor TractCloud-FOV
  ([HBM 2025](https://arxiv.org/abs/2502.20637)), nor TractFM
  ([2026](https://arxiv.org/abs/2606.09893)), nor the repo's issues discuss run-to-run variability
  or confidence. The inference path does not call the `fix_seed` helper the repo contains. **The
  28 % tract and 69 % cluster change rates have not been reported.**
- **Closest prior work: PETParc / RapidParc** (von Bornhaupt, Bisten, ... Schultz;
  [arXiv 2025](https://arxiv.org/abs/2503.07104),
  [Imaging Neuroscience 2026](https://doi.org/10.1162/IMAG.a.1168)).
  - The model also uses 1600 clusters mapped to 42 tracts plus outliers, with random context
    from sub-tractograms.
  - Across 20 random partitions, aggregate accuracy varies by a standard deviation below 0.001,
    which hides per-streamline flips.
  - Appendix Fig. A.4, "Cluster stability", counts per-streamline class changes over 20 runs. It
    finds the unstable streamlines are mostly those sent to "Other" at least once, at the
    inlier/outlier boundary. That matches M1's finding that 79 % of change events involve "Other".
  - It offers **no single-run predictor**. Its figure numbers were not extractable (unverified).
- **Soft membership in classic tractography:**
  - WMA/ORG outlier removal computes a per-fiber "fiber probability", a Gaussian affinity to
    atlas fibers. It is cluster-level and is not used to predict instability.
  - Maddah 2008 (EM mixture) and Wassermann 2010 (Gaussian-process tracts) are probabilistic.
  - O'Donnell & Westin 2007 assign hard to the nearest centroid, but test embedding stability
    over random samples.
- **Uncertainty in tract segmentation is all volumetric:** TractSeg MC dropout; Lucena 2022
  ([dropout + TTA](https://pmc.ncbi.nlm.nih.gov/articles/PMC10365092/)); Kebiri 2023
  ([EMD-based](https://arxiv.org/abs/2307.02223)). No streamline-level calibration, conformal or
  evidential work was found.
- **Reproducibility studies** (WMA test-retest, [Zhang 2019](https://lmi.bwh.harvard.edu/publications/test-retest-reproducibility-white-matter-parcellation-using-diffusion-mri-tractography);
  Schilling 2021 [a](https://doi.org/10.1016/j.neuroimage.2021.118502)/[b](https://www.biorxiv.org/content/10.1101/2021.03.17.435872v1))
  measure acquisition, protocol and workflow variability, not stochastic inference.
- **Uncertainty visualization** (Brecheisen 2013, "Fuzzy Fibers", the Schultz & Vilanova survey)
  concerns tracking uncertainty. No viewer was found that colors streamlines by classification
  confidence.
- **Hierarchical aggregation:** in this field the cluster-to-tract map only collapses argmax
  labels. In general ML the idea is established:
  - **Hierarchical Selective Classification** (Goren, Galil & El-Yaniv,
    [NeurIPS 2024](https://arxiv.org/abs/2405.11533)) takes a node's probability as the sum of
    its leaf softmaxes, climbs the hierarchy until confident, and calibrates the threshold with a
    PAC guarantee. That is the mass margin used for abstention.
  - Deng et al., "Hedging your bets" ([CVPR 2012](https://ieeexplore.ieee.org/document/6248086)).

**Where this work stands:** the measurement (28 %/69 % instability in TractCloud) and the
single-run tract-level predictor (AUROC 0.875 against 0.65) were not found anywhere. The
aggregation method is known in general ML; applying it here and validating it against
run-to-run instability is the contribution. RapidParc is the paper to cite and contrast.

### Focused check, 2026-10-02: run-to-run variability and context ensembling

A second search, aimed at whether the instability findings are new (web; OpenAlex and Semantic
Scholar citations of TractCloud; no Google Scholar; some citing papers paywalled):
- **TractCloud's own papers and repository** (MICCAI 2023 and its reviews, TractCloud-FOV HBM 2025,
  the two GitHub issues) do not discuss randomness at inference. The paper chooses the global
  streamlines at random (w = 500); the 10 % local subsample is the code's, not the paper's.
- **The only direct precedent is PETParc / RapidParc** (arXiv 2503.07104; Imaging Neuroscience 2026,
  doi 10.1162/IMAG.a.1168). PETParc Table 2: accuracy SD 8.0e-4 / 9.7e-4 over 20 runs on a TractCloud
  test subject. RapidParc Fig. A.4 ("Cluster stability", 20 runs): the share of streamlines given the
  same class in every run rises with the context size - about 86 % at 100, 94 % at 500, 98 % at 5,000
  (read off the chart; whether "class" is cluster or tract is unverified) - and the inconsistent ones
  were sent to "Other" at least once in ~84-92 % of cases. No ensembling.
- **TractFM** (arXiv 2606.09893) ensembles over 30 geometric transforms of the tractogram (test-time
  augmentation), not over context draws; its downstream predictions barely move across 10K-streamline
  subsets. Test-time dropout/augmentation exist for voxel-wise tract segmentation (Lucena et al.,
  Front. Radiol. 2022).
- **Numerical variability** in neuroimaging pipelines is established: Glatard et al. 2015 (operating
  systems), Kiar et al. 2020 (IJHPCA) and 2021 (PLOS ONE: Monte Carlo arithmetic, connectomes), Salari
  et al. 2021, Gonzalez Pepe et al. 2023 (CNN inference); CPU/GPU tractography: Hernandez-Fernandez et al.
  2019, Kim et al. 2022 (bedpostx); GPU nondeterminism in segmentation: Renard et al. 2020. No paper was
  found measuring TF32's effect on medical-imaging results, or a small input perturbation amplified
  through a random-context classifier.
- **Where this work stands now:** known - random context makes labels vary, at the "Other" boundary
  (RapidParc); numerical perturbations propagate. Apparently new - TractCloud's instability measured
  (28 % tract / 69 % cluster labels within 5 runs, far above RapidParc's ~2-6 % at comparable context
  sizes); the aggregate Other fraction moving ~11 points across draws on a pathological scan (against
  PETParc's stable aggregates on a healthy subject); hardware and TF32 acting as a fresh draw; ensembling
  over context draws and per-streamline tract confidence. Confidence: high that RapidParc is the only
  direct precedent; moderate on the rest (paywalls, no Google Scholar).

## 3. Storing soft output compactly

- **The closest designs are in LLM distillation:** top-K logits plus one residual mass.
  - "Ghost Token" in Sparse Logit Sampling ([ACL 2025](https://arxiv.org/abs/2503.16870)); also
    [Ryskulov 2026](https://arxiv.org/html/2608.03796) and [ReTaCo 2026](https://arxiv.org/html/2609.39275).
  - Sparse Logit Sampling shows that top-K **is biased**: the student learns inflated
    probabilities. Its unbiased fix samples tokens by importance and stores counts, about 12
    tokens at 3 B each.
  - DistillKit ([Arcee](https://github.com/arcee-ai/DistillKit)) fits a polynomial to the sorted
    log-probability curve.
  - None stores gaps measured from the winner, and none targets decisions made later.
- **Shen et al. 2026** ([arXiv 2607.07050](https://arxiv.org/html/2607.07050v4)) found that
  top-32 held 99.99 % of the mass yet contained the decision token on 0.4 % of prompts: "mass is
  not the same as decision support". This is **rankfield's group-blind depth cut, found
  independently**. Their fixes are restoring the omitted decision token, or storing the union of
  two supports.
- **Medical imaging** stores probabilities densely:
  - nnU-Net and TotalSegmentator write float32 `.npz`.
  - DICOM SEG FRACTIONAL uses 8 bits per segment, sparse only by omitting empty frames
    ([C.8.20.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.2.html)).
  - FSL probabilistic atlases store each class densely at 0-100.
  - OME-Zarr labels must be integers.
  - ReLabel ([CVPR 2021](https://ar5iv.labs.arxiv.org/html/2101.05022)) stores top-5 per
    location for ImageNet, the closest dense-map analog.
  - **No standard accepts top-k soft labels** (DICOM, OME-Zarr, COG/STAC, LAS, TRX). TRX and COG
    can carry the planes as generic typed arrays without their meaning.
- **Quantization:** Adler, Tang & Polyanskiy ([2022](https://arxiv.org/abs/2205.03752)) show an
  arcsinh compander is minimax-optimal for KL loss. That is the theory behind a nonuniform level
  table; rankfield's table is tuned for decision margins rather than KL.
- **Margins:** top-1 minus top-2 is a studied confidence score ([Liang 2024](https://arxiv.org/abs/2405.05160)).
  The best-outside-group margin has no established name. It is the Crammer-Singer margin on a
  coarsened label space (synthesis, unverified as a named quantity).

**Fixes for rankfield's two measured weaknesses suggested by the literature:**

| weakness | fix | cost |
|---|---|---|
| the depth cut is group-blind, so the best-outside-group margin reads high | store each declared coarse group's best class beyond the top N (Shen 2026's restoration) | about 3 B per declared group, the "declared partitions" idea |
| | store top-k per level of a hierarchy (SALT, Mortier) | more planes |
| | **report the margin as an interval**: an unstored class trails by at least the last stored gap (planes full) or the clip, so the true margin lies between that bound and the decoded value | none; decode only |
| the single tail makes the mass margin a loose bound | a residual "ghost" mass per declared coarse group | 2 B per group |
| | an importance-sampled tail after the exact top N, for an unbiased estimate | about 3 B per sample |
| | split the tail uniformly over the unstored classes, as a point estimate between the bounds | none; decode only |

The two decode-only rows can be tried in the viewer without changing the format.

## 4. The ensemble against real rescans: what is new (search of 2026-10-09)

The annotated search is in [prior-art-uncertainty.md](prior-art-uncertainty.md): 74 papers, each checked at least to
its abstract, with the search log. Claims as the project's NOTES make them, from 2026-10-08 on:

| Claim | Closest prior work | Verdict |
|---|---|---|
| The bootstrap's spread of *tractography output* against real rescans; per-voxel signs understate it because noise is correlated in-plane; smooth sign patches match a same-session rescan | Campbell et al. 2014 (Front Neurol 5:216): residual-bootstrap 68/95 % intervals of fODF peaks against same-session repeats; good above FA 0.3, under-dispersed below it, put down to overfitting and misregistration. Zhu et al. 2011, tensor level | Partly anticipated, at the orientation level. Output level, the noise-correlation cause and the patch-sign fix not found. Patch signs rest on the wild cluster bootstrap (Cameron et al. 2008) and whole-image sign flips (Winkler et al. 2014) |
| Resampling the whole pipeline, the susceptibility field re-estimated in every replicate | Kiar et al. 2020/2021 perturb whole pipelines numerically; every dMRI bootstrap found resamples preprocessed data | Not found |
| Per-streamline label change on a rescan (35–50 %) | Yu et al. 2024 (arXiv 2403.18992): streamlines matched within a 0.5-voxel ball of the seed across scan and rescan, for *geometry*; Zhang et al. 2019 (HBM): UKF + ORG test-retest at tract level | The measurement is not found; seed matching across rescans is Yu et al.'s |
| Calibration (ECE, AUROC) of tract probabilities against rescans; a double-bootstrap recalibration that transfers across vendors | Lucena et al. 2022: tract-segmentation probabilities calibrated against labels, not rescans; the double bootstrap is Beran 1987, Hall & Martin 1988 | Not found in tractography or tract segmentation |
| Envelope coverage of rescans; winner-take-all maps prevent honest envelopes, presence maps do not | Brown, Booth & Hamarneh (CDMRI 2013): tract confidence regions, never tested on rescans; Brecheisen et al. 2013, visualization | The rescan coverage test and the winner-map argument not found |

Wording to keep:
- Comparing a bootstrap with repeated scans is not new (Campbell 2014; the repetition-bootstrap papers).
- Predictive coding of streamline geometry is not new (TRAKO/Draco). The ranked soft labels beside it are.
- Seed-matched correspondence across rescans is Yu et al.'s; the label agreement and its link to the ensemble's
  probabilities are the project's.
- MASiVar itself reports intrasession < intersession < interscanner variability. The project's
  same-session / other-session difference agrees with it.
- "Ensemble tractography" already names something else (Takemura et al. 2016: tractograms combined over tracking
  parameters). A write-up needs another name for the resampled-pipeline ensemble.
- The search was not a systematic review. Its suggested follow-ups (the full text of Zhang 2019, Semantic Scholar
  forward citations of Campbell 2014 and Jones 2008, the IST 2025 proceedings) were not done.
