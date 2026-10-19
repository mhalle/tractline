# A tractography data product that states its uncertainty: a proposal

Draft, 2026-10-19. This is step 7 (synthesis) of `docs/robustness-plan.md`, written after steps 1–6, and revised after
an independent review.

**For research use only.** Nothing here has been validated for clinical decisions. Every statement below concerns
healthy adults on one research protocol, one simulated phantom, or one patient scan with no rescan. None of it has been
checked against stimulation mapping, histology or surgical outcome.

Sources:
- `docs/robustness-plan.md`: steps 1–6 and their outcomes;
- `docs/uncertainty-ensemble.md` and `docs/validation-datasets.md`;
- `bench/results/NOTES.md`, cited by date and title, through 2026-10-19, "Robustness step 6";
- `bench/results/calibration_card_masivar_cIII.json` (the card) and `bench/results/robust_benchmark.json` (step 6).

Every number is taken from these sources. Where a figure is derived here, the text says so.

## The question

Can tractline provide a more robust data product for tractography, one that avoids, or at least makes explicit, the
uncertainty and lack of determinism in the pipeline?

It follows an earlier idea: treat the pipeline as a single unit rather than as separate modules. The pipeline is field
correction, UKF tracking and RapidParc labeling, run in memory. One pass takes about 90 s on the M2's GPU for a cohort
III scan (below).

## The answer first

- **Yes for regional claims about a rescan, on one protocol. Not yet for claims about the anatomy.**
  - The product is an ensemble: 17 runs of the whole pipeline on bootstrap replicates of one scan.
    - Its statements are regional: where a named tract is, and which tract dominates near a point.
    - Each statement names the event its probability refers to.
  - A calibration card for the scan's protocol makes those statements honest about a same-session or a later rescan,
    for held-out healthy subjects (step 5).
    - Today the only card covers MASiVar cohort III on one Philips scanner.
    - Every other scan falls back to "run-to-run only", which is not a probability of any real event.
  - The anatomy is not calibrated. On one simulated phantom, the envelopes covered 17–18 points less of the truth than
    a bias-free ensemble did (step 4). That term is recorded, not applied.
- **Treating the pipeline as one unit is what the results support.**
  - On this protocol, nearly all of the spread is the DWI's noise reaching the tracker. The stages' effects do not
    add, so a per-stage "uncertainty budget" would misdescribe it.
  - Ensembles that carry the scan's own DWI noise through the tracker rank a rescan's changes well. That holds with or
    without the field re-estimated (AUROC 0.77–0.80 either way). Cheap perturbations do not rank them well (0.54–0.70).
  - Ranking is not calibration: the ensemble's stated values remain overconfident until the card is applied.
- **Some nondeterminism can be removed. With UKF, most of what matters cannot.**
  - Two opt-ins cut renames under a rounding-level input change from 3.96 to 0.58 %, at no measured cost in accuracy on
    the phantom: a seed-keyed labeler grouping and a UKF tie-break. They are candidate defaults; the decision is yours.
  - A rescan still renames about 35–40 % of streamlines on this protocol. Every change to UKF's settings or to the
    input that reduced that lost accuracy on the phantom: a stiffer tracker, or Patch2Self denoising.
  - Patch2Self's phantom loss was in a different regime: 33 volumes, against cIIIs04's 112.
  - So the product states the disagreement rather than hiding it, and never presents a single streamline's name as a
    finding.
- **The method behind the regional claims is the larger lever, and the choice is open** (step 6; one subject, one
  phantom).
  - **PTT + ACT (CSD), as a pipeline, is more accurate on the phantom.** This is DIPY's probabilistic tracker with
    anatomical stopping and white-matter seeding.
    - It scores mean F1 0.583 against UKF's 0.412: the largest accuracy difference measured in this work.
    - Seeding controls rule out the seeds as the cause. The tracker and its stopping rule were changed together, so
      their effects are not separated.
    - On cIIIs04's rescans, PTT + ACT is slightly steadier than UKF at matched streamline counts, by margins of
      0.01–0.07. That rests on one subject and one run per scan, and PTT + ACT reused A's tissue maps for seeding and
      for ACT on every rescan. Only the CST part depends on a post hoc fix for ACT's edge rule; the AF and UF leads hold
      as run.
  - **TractSeg's masks agree best across rescans, on their own terms.**
    - Presence Dice is 0.89–0.96 for the CST, AF and UF, against UKF's 0.73–0.81.
    - Part of that is how they were run: one registration for all four scans, a deterministic model, binary masks
      compared with single-run density presence, and a much larger AF.
    - Its accuracy on the phantom, off its training domain, cannot be ranked against the trackers'. A mask is a
      different kind of object.
  - **Whichever method supplies the regional claims needs its own ensemble design and its own card.** RapidParc was
    trained on UKF tractograms, and it labels PTT's streamlines off its domain.
- **The results are not tied to a bad protocol, but the numbers are protocol-specific.**
  - The cohort III protocol is not worse than the other data here.
  - The mechanisms recur on a Siemens patient scan and on the phantom.
  - The magnitudes and the card do not transfer. On MASiVar's cIs1 protocol, rescans change 47–50 % of names, and a
    map that fit cohort III did not fit cIs1.

## Glossary

| term | meaning |
|---|---|
| UKF | Today's tracker: two-tensor unscented Kalman filter tractography, deterministic given its input. |
| RapidParc | Today's labeler. It assigns each streamline to one of 42 named tracts or Other, using the ORG atlas's clusters. It labels each streamline in the context of 2,000 others, and was trained on UKF tractograms. |
| ORG atlas | The fiber-clustering atlas whose 800 clusters, and their outlier twins, define RapidParc's 43 classes. |
| Other | RapidParc's class for streamlines it gives no named tract. It holds about half of all streamlines. |
| CSD | Constrained spherical deconvolution: a model of fiber orientations in each voxel. |
| PTT | Parallel transport tractography (DIPY), a probabilistic tracker; here it runs on a CSD fit. |
| ACT | Anatomically constrained stopping from tissue maps. A path is kept only if it ends validly: in gray matter, outside the brain, or at the image's faces. |
| TractSeg | A neural network that segments 72 bundles directly from fiber peaks, with no tracking. It was trained on HCP data. |
| A | A subject's reference scan. For cIIIs04 it is compared with three rescans: "within" (the same session), session 3 and session 5. |
| cIIIs04, cIs1 | MASiVar (OpenNeuro ds003416) subjects: cohort III subject 04 on scanner B, and cohort I subject 1 on scanner A. |
| PAT16 | A patient from ds001226 with a temporal tumor, scanned on a Siemens Trio; no rescan. |
| R | The number of runs in an ensemble: here A plus 16 bootstrap replicates (R = 17). |
| presence | A tract is present in a voxel when its streamline density is at least 0.3 per mm² after 1 mm smoothing. |
| envelope at stated level L | The highest-vote voxels that hold a share L of the tract's vote mass. L is a level, not a probability, until a card maps it. |
| core, widest | The voxels at least half the runs give the tract (≥ 9 of 17), and the voxels any run gives it. |
| Vorob'ev shape | The vote cut whose volume is closest to a run's mean tract volume: an outline with an honest size. |
| landmark sites | Eight points on cIIIs04: the centrum semiovale and temporal stem on each side, the posterior limbs of the internal capsule, the genu and the splenium. |
| layers A, B1, B2, C | The product's claims: A, structure; B1, tract presence; B2, composition near a point; C, single streamlines (below). |
| epsilon | A rounding-level perturbation: the input × (1 + 10⁻⁶ u). |
| ECE | Expected calibration error: the average gap between stated and observed probability. |
| AUROC | How well a score ranks the cases that change: 0.5 is chance, 1 is perfect. |
| F1 | The phantom scorer's Dice of each recovered bundle with its ground truth, averaged over all 21 bundles; a missing bundle counts 0. |
| Dice | The overlap of two voxel sets, 2 \|A ∩ B\| / (\|A\| + \|B\|). |
| TV | Total variation distance between two tract mixes: 0 means the same, 1 means disjoint. |
| mps | PyTorch's Metal backend on the M2's GPU. |
| card | A per-protocol calibration file (step 5) that maps an ensemble's statistics to probabilities about a rescan. |

## What we learned

### One run's confidence is wrong; an ensemble ranks well but states too much

- **A fresh run changes about a third of the labels.** On PAT16, a held-out run of the pipeline changes 34.0 % of the
  reference's tract labels, with the labeler's draw resampled. The draw alone changes 2.9 % (NOTES 2026-10-07, "The
  pipeline's ensemble").
- **One run's probability is badly calibrated.**
  - Its ECE is 0.278, against 0.030 for an 11-run ensemble's mixture.
  - Streamlines that a single run calls ≥ 0.99 keep their tract in 73.5 % of held-out runs.
  - The ensemble ranks the changes at AUROC 0.802; one run ranks them at 0.646 (same entry).
- **The bootstrap is centered on the scan.** So held-out replicates measure the spread around this scan's estimate,
  not that estimate's own error (NOTES 2026-10-08, "Why the scan is closer to a replicate").
- **Per-voxel sign flips understate a real rescan,** because the noise is correlated in-plane.
  - Smooth signs (FWHM 4 voxels) close the same-session gap on cIIIs01 to 1.3–2.0 points.
  - They do not close the gap to another session (NOTES 2026-10-08, "The bootstrap against real rescans").
- **On real rescans, the ensemble ranks well and is overconfident** (NOTES 2026-10-08, "The ensemble scored on real
  rescans").
  - It ranks the changes at AUROC 0.73–0.78, against 0.65 for one run.
  - Its mean stated P(keep) exceeds the observed rate by 8–10 points on cIIIs01 and by 15–18 on cIs1.

### What repeats on a rescan, and what does not

| claim | how well it repeats | source |
|---|---|---|
| where white matter is (density shared) | 0.75–0.79 (cIIIs01, cIIIs04); 0.63–0.65 (cIs1) | NOTES 2026-10-08, "Where and which"; 2026-10-09, "A held-out subject" |
| its local direction (\|cos\|) | 0.94–0.97 (about 14–20°) | same |
| the main named tract within 8 mm | 82–90 % of spheres (cohort III, by session); 81–82 % (cIs1) | same |
| a named tract's density, per tract | 0.48–0.53 | same |
| a single streamline's name | kept on 50–64 % of rescans | same |

- **Labels change because paths change** (NOTES 2026-10-08, "Smooth signs on PAT16, and soft labels").
  - The labeler is rarely unsure: its top tract averages 0.93–0.94.
  - Carrying its full probabilities forward lowers rescan change by about a point.
  - A seed's streamline runs 13–21 mm apart on a rescan, and reaches other anatomy.
- **The CST's names are the least stable, but its territory is steady.**
  - A rescan keeps the name of 27–33 % of A's CST streamlines. Yet the 80 % outline holds 85 % of a rescan's CST
    (NOTES 2026-10-13, "The cIIIs04 viewer refocused").
  - The cause is the path at the pons. About 37 % of the versions that become Other turn into the cerebellum, against
    6–7 % of A's versions.
  - A replicate does the same as a rescan (NOTES 2026-10-13, "Why CST streamlines become Other").

### Regional shapes: presence, density, and an honest size

- **Presence maps, which let tracts overlap, make envelopes usable.** As stated, a 90 % envelope holds 78–80 % of a
  rescan's tract, against 60 % with winner maps (NOTES 2026-10-09, "Compact run records, and presence maps").
- **Density presence held across grids, on one subject** (NOTES 2026-10-13, "Grid-free presence"):
  - for cIIIs04's three focus tracts, at uncalibrated levels, coverage agrees within 1–2 points between the 2 mm and
    1 mm grids;
  - the count rule loses about two thirds of the volume at 1 mm.
- **The Vorob'ev shape gives an honest size.**
  - The majority core underestimates a repeat scan's tract volume by 5–9 %. The Vorob'ev shape is unbiased
    (0.99–1.05), with coverage about equal to precision (NOTES 2026-10-13, "A volume-matched core").
  - Across eight held-out subjects, its size is 0.99–1.03 (step 5).
- **Pooling sessions made the stated levels hold on cIIIs04 without any correction** (NOTES 2026-10-13, "Pooling
  sessions").
  - Three scans' ensembles (51 runs) cover 78–81 % at a stated 80 %, and 88–90 % at 90 %. One scan's 17 runs fall 4–5
    points short.
  - That entry read the shortfall as missing between-session variation, with the run count confounded.
  - Step 5's eight subjects revise this. Most of the correction a single scan needs is the bootstrap's overconfidence
    within a session; the other-session term adds only 2–3 points of stated probability.
- **The ensemble fits in a small store.** Four scans' presence at 1 mm, with each scan's 17-run votes (2.9 MB each), is
  a 19.6 MB duckn store; rank fields add about 10 MB each (NOTES 2026-10-13, "The ensemble as a duckn store").

### Where the spread comes from: one source, not a budget

NOTES 2026-10-17, "Where the ensemble's spread comes from" (cIIIs04 A, 17 runs per arm, labeler draw 0):

| arm | what varies | renamed vs A | share of the full arm's presence spread | AUROC on the rescans |
|---|---|---|---|---|
| full | the DWI's noise, and the field re-estimated | 31.6 % | 1 | 0.80 / 0.78 / 0.77 |
| frozen | the DWI's noise; A's own field | 31.1 % | 0.99 | 0.80 / 0.78 / 0.77 |
| fieldonly | the field from noisy b0s only | 16.3 % | 0.65 | 0.70 / 0.68 / 0.67 |
| draw | the labeler's draw | 3.0 % | 0.12 | 0.54 |
| epsilon | a 10⁻⁶ relative nudge of the input | 4.0 % | 0.19 | 0.55 |

- **The DWI's noise reaching the tracker is the spread.**
  - Holding the field fixed lowers label change by 0.5 points and presence spread by under 2 %.
  - It does not change the ranking of the rescans' changes.
  - With 16 forward b0s, the field varies by about 0.1 mm.
- **The stages do not add.** Frozen's 0.99 plus fieldonly's 0.65 is far above 1: every perturbation draws on one pool
  of fragile seeds.
- **No region belongs to the field.**
  - Its share is flat across field strength.
  - The frozen arm dominates even in the brainstem and cerebellum.
  - A per-region map of which stage the uncertainty comes from would show one source nearly everywhere.
- **Cheap perturbations are not a substitute for the bootstrap.** Ensembles of nudges, draws or field noise state
  0.89–0.98, against 0.61–0.67 observed.

### UKF's settings and the input: steps 1–4

These results are for cIIIs04 and its three rescans (within, session 3, session 5). The phantom is the ISMRM 2015
challenge, under the 2023 ROI scoring. Sources: NOTES 2026-10-17, "Robustness step 1" to "Robustness step 4", and the
plan's outcomes.

| configuration | epsilon: renamed | a rescan renames | phantom mean F1 | tract mix TV vs today (17 runs) | reading |
|---|---|---|---|---|---|
| today | 3.96 % | 35.4 / 38.2 / 39.7 % | 0.412 (replicates 0.404–0.413) | — | |
| hash-sorted grouping (step 1) | 1.61 % | 35.5 / 38.0 / 39.8 % | not scored alone | 0.0006 | free; fixes the labeler's floor only |
| tie-break, both swaps (step 2b) | 3.24 %; 0.58 % with hash-sorted | 35.8 / 38.3 / 39.7 % | 0.411 (with hash-sorted) | 0.002 | accuracy-neutral; reproducibility only |
| seeding threshold 0.2 (step 2) | 3.12 % | 35.4 / 38.1 / 39.9 % | not scored | 0.043 | a third fewer streamlines; no rescan gain |
| step length 0.6 (step 2) | 3.71 % | 31.8 / 34.0 / 36.5 % | 0.330 | 0.070 | steadier, less accurate |
| Rs 0.1 (step 2) | 3.62 % | 28.8 / 31.4 / 34.1 % | 0.314 | 0.113 | steadier, less accurate; AF ×0.43 |
| Patch2Self (step 3) | 4.07 % | 31.3 / 33.7 / 36.2 % | 0.278 | 0.034 | steadier; less accurate on the phantom, in another regime |
| MP-PCA, radius 2 / 3 (step 3) | not run | 35.5 / 35.2 % (within only) | not scored | 0.003 / 0.005 (A) | no change in rescan agreement |

- **Within UKF, steadier is not truer here.**
  - Step length 0.6 and Rs 0.1 lose F1 in 20 of 21 bundles.
  - Patch2Self loses the most. About a third of its loss comes from having fewer streamlines (step 4).
  - The phantom's regime is unlike cIIIs04's: 33 volumes on one shell and a DW SNR of about 7, against 112 volumes
    over two shells. So that loss does not transfer as it stands, but the warning does.
- **Every gain in rescan agreement came with a changed tractography.** Over-smoothing explains the tracker settings'
  gains as well as better tracking does (step 2).
- **Neither of two bootstrap designs makes a denoised pipeline honest** (step 3). A denoised pipeline would need its
  own card.

### Another tracker and a segmentation model: step 6

Source: NOTES 2026-10-19, "Robustness step 6", and `robust_benchmark.json`. Two alternatives to today's UKF were run
once per scan, with no ensemble:
- **PTT + ACT (CSD), as a pipeline:**
  - DIPY's PTT on a single-shell CSD fit, with DIPY's defaults;
  - ACT stopping from FastSurfer's tissue maps on cIIIs04, and from an FA/T1 classifier on the phantom;
  - white-matter seeding, and RapidParc labels.
- **TractSeg 2.10:** masks of 72 bundles from CSD peaks registered to its MNI template, with no tracking. Eighteen of
  them map to RapidParc's tracts.

**On the phantom** (the 2023 ROI scoring; means over all 21 bundles):

| run | streamlines | valid bundles | VS | mean OL / OR | mean F1 | count-matched Dice, run / UKF |
|---|---|---|---|---|---|---|
| UKF (today) | 61,637 | 20 | 27.48 % | 0.444 / 0.479 | 0.412 | — |
| PTT + ACT | 61,454 | 19 | 48.88 % | 0.532 / 0.244 | 0.583 | 0.618 / 0.521 |
| UKF, seeded only in PTT's white matter | 41,578 | 20 | 29.81 % | 0.398 / 0.455 | 0.398 | 0.511 / 0.504 |
| UKF (today), subsampled to that count | 41,578 | 20 | 27.41 % | 0.389 / 0.453 | 0.390 | 0.500 / 0.499 |
| PTT + ACT, seeded at UKF's seed points | 29,231 | 19 | 48.34 % | 0.442 / 0.209 | 0.532 | 0.589 / 0.501 |
| PTT + ACT, subsampled to that count | 29,231 | 19 | 48.97 % | 0.443 / 0.202 | 0.537 | 0.597 / 0.506 |
| PTT + ACT, invalid ends kept | 80,247 | 19 | 48.44 % | 0.565 / 0.258 | 0.595 | 0.629 / 0.520 |

- **PTT + ACT, as a pipeline, is the largest accuracy lever measured.**
  - Its F1 is 0.17 above UKF's, at about the same streamline count, and far outside UKF's replicate band.
  - Steps 1–4's options moved F1 by −0.13 to 0.00. The tie-break was neutral, and the seeding threshold was not
    scored.
  - PTT's F1 is higher in 19 of the 21 bundles.
- **The seeding controls rule out the seeds.**
  - Swapping seeds moves either tracker by less than 0.01 at a matched count.
  - Keeping PTT's invalid ends moves its VS by half a point.
  - PTT and ACT were changed together and are not separated: there was no PTT run without ACT, and no UKF run with it.
- **The phantom Dice is the less firm comparison.**
  - The target is each ground-truth bundle's traversal mask.
  - The trackers give density presence, whose Dice rises with the streamline count. Hence the count-matched column,
    where PTT + ACT leads 0.618 to 0.521.
  - TractSeg's binary masks score 0.551 on the same 14 bundles, and 0.505 over its 20 mapped bundles. A mask is a
    different kind of object, closer to the target's kind, so TractSeg's accuracy cannot be ranked against the
    trackers' from this figure.
  - F1 and the count-matched Dice are the firmer comparisons.
- **Labels.** RapidParc names 64.35 % of PTT's valid streamlines that have a counterpart as that counterpart, against
  57.59 % of UKF's.
- **Published figures are not comparable.** The scoring here is a reconstruction of the 2023 wrapper.

**On cIIIs04's rescans** (presence Dice of A's focus tracts with each rescan: within / session 3 / session 5):

| | UKF (today) | PTT, as run | PTT, the scan's own face (post hoc) | TractSeg |
|---|---|---|---|---|
| CST | 0.78 / 0.78 / 0.75 | 0.66 / 0.86 / 0.84 | 0.85 / 0.86 / 0.86 | 0.96 / 0.94 / 0.94 |
| AF | 0.81 / 0.79 / 0.77 | 0.82 / 0.81 / 0.80 | 0.82 / 0.81 / 0.81 | 0.95 / 0.94 / 0.93 |
| UF | 0.79 / 0.76 / 0.73 | 0.73 / 0.73 / 0.68 | 0.73 / 0.73 / 0.68 | 0.93 / 0.92 / 0.89 |
| landmark sites kept (of 8) | 6 / 6 / 5 | 7 / 8 / 7 | not recomputed | 8 / 8 / 7 (among 18 tracts) |
| CST volume, A / within / s3 / s5 (mL) | 51.8 / 56.8 / 59.8 / 58.1 | 87.2 / 46.3 / 89.3 / 77.2 | 87.2 / 83.1 / 89.3 / 88.0 | 58.6 / 58.7 / 59.3 / 58.9 |
| AF / UF volume, A (mL) | 53.6 / 41.8 | 44.7 / 16.6 | 44.7 / 16.6 | 162.8 / 28.6 |

Count-matched per tract: for each scan and tract, both methods' streamlines are subsampled to the smaller of the two
counts (5 draws):

| | CST: within / s3 / s5 | AF | UF |
|---|---|---|---|
| UKF, matched to PTT as run | 0.776 / 0.783 / 0.750 | 0.807 / 0.785 / 0.769 | 0.663 / 0.667 / 0.639 |
| PTT, as run | 0.756 / 0.793 / 0.802 | 0.822 / 0.811 / 0.805 | 0.735 / 0.732 / 0.684 |
| UKF, matched to PTT with the scan's own face | 0.775 / 0.783 / 0.750 | 0.806 / 0.785 / 0.768 | 0.670 / 0.667 / 0.643 |
| PTT, the scan's own face | 0.807 / 0.793 / 0.814 | 0.822 / 0.811 / 0.806 | 0.726 / 0.732 / 0.677 |

- **PTT + ACT's CST swing was an artifact of ACT's valid-end rule, not instability.**
  - Resampled onto A's grid, the within rescan leaves A's lowest slice without data in all 54 of its brainstem voxels;
    session 5 leaves 48 empty.
  - A descending path then stops a voxel short of the image's face, the only valid end under ACT, and is dropped.
  - Seeding and the tissue maps' registration were ruled out.
  - Making the scan's own data edge a valid end restores the CST: 83.1 mL on the within rescan and 88.0 mL on
    session 5.
  - That rule is post hoc: it was chosen after seeing the result. UKF has no valid-end rule and needs none. The
    benchmark's records keep the original rule, and the sites were not recomputed under it.
- **At matched counts, PTT + ACT is slightly steadier than UKF, within named limits.**
  - With the scan's own face, it leads by 0.01–0.06 on the CST, 0.02–0.04 on the AF and 0.03–0.07 on the UF. As run,
    the CST difference is −0.02 to +0.05; the AF and UF leads hold as run. Only the CST part depends on the post hoc
    rule.
  - The limits: one subject, one run per scan, margins of 0.01–0.07, and PTT + ACT reused A's tissue maps for seeding
    and for ACT on every rescan. The RNG check below varied the seeds, not the maps.
  - The unmatched table overstated its CST lead (PTT has 3.3 times UKF's CST streamlines) and reversed its UF result
    (UKF has about 2.5 times PTT's UF streamlines). Presence Dice rises with streamline count.
- **PTT + ACT's spread between runs on one scan is about its spread between scans.**
  - Reruns of A with other RNGs agree at 0.868–0.870 (CST), 0.828–0.833 (AF) and 0.736–0.755 (UF).
  - Rescans with the scan's own face agree at 0.850–0.863, 0.804–0.833 and 0.652–0.745, whether or not the RNGs are
    shared.
  - The CST's rescan figures sit just below its reruns on A (0.850–0.863 against 0.866–0.870); the AF and UF ranges
    overlap.
  - Most of PTT + ACT's disagreement between scans is its own sampling, which an ensemble of its runs would average.
- **TractSeg agrees best across rescans, on its own terms.**
  - Its Dice is 0.89–0.96, and its focus volumes move by at most 3.5 mL between scans.
  - But the comparison is not symmetric:
    - one registration, A's to MNI, served all four scans. Each rescan's registration onto A remains in its rescan
      agreement, as for every method; only the registration to MNI is shared and so cannot differ between scans;
    - it is deterministic given its input;
    - its binary learned masks are compared with the trackers' single-run density presence;
    - its sites are judged among 18 coarser tracts, against RapidParc's 42;
    - its masks are large (the AF 162.8 mL, three times UKF's), and a large mask agrees with itself more easily.
  - The shared-inductive-bias caution applies to it too: shapes learned from training data repeat whether or not they
    are where this subject's fibers run.
- **The comparisons are asymmetric in other ways.** PTT reuses A's tissue maps and seed points on every rescan, while
  UKF re-derives its mask and seeds from each scan.
- **The domain.**
  - RapidParc was trained on UKF tractograms. On A, PTT's tract mix differs from UKF's by a TV of 0.149; its Other
    share is 0.484 against 0.526, and RapidParc's mean top probability 0.923 against 0.941.
  - TractSeg was trained on HCP data (multi-shell, 270 directions, 1.25 mm, MRtrix's CSD peaks), so it is off its
    domain on both datasets here.

### A calibration card makes a single scan honest about a rescan: step 5

Sources: NOTES 2026-10-18, "Robustness step 5", and the card's `held-out evaluation (leave one subject out)`. The data
are MASiVar cohort III on scanner B: eight healthy subjects, each with a 17-run ensemble and a same-session rescan.
Five of them also have rescans in other sessions (9 in all).

| held out, leaving one subject out | same session (8) | other session (5) |
|---|---|---|
| streamline ECE: as stated / card | 0.070 / 0.008 | 0.097 / 0.008 |
| point ECE (main named tract within 8 mm): as stated / card | 0.038 / 0.014 | 0.059 / 0.022 |
| envelope coverage at target 0.5 / 0.8 / 0.9 / 0.95, card | 0.501 / 0.803 / 0.904 / 0.954 | 0.501 / 0.802 / 0.902 / 0.952 |
| the same, as stated | 0.535 / 0.777 / 0.869 / 0.921 | 0.529 / 0.760 / 0.852 / 0.906 |

- **No held-out subject is badly off,** and the fold maps are stable.
- **Most of the correction is the bootstrap's overconfidence within a session.** The other-session term adds 2–3 points
  of stated probability, and 0.7–1.5 points of coverage, on top of the same-session correction.
- **The card is about as honest as pooling three sessions.**
  - Its envelopes are only 2–5 % larger at matched coverage at 0.9, and 5–8 % larger at 0.95.
  - The pooled references are imperfect. They come from one or two subjects, two to three sessions, and include a
    same-session scan for the within targets.
- **Even a unanimous seed's name holds on only 93 % (same session) and 91 % (other session) of rescans.** No
  streamline reaches 0.99.
- **The 0.5 row is not a blind result.** The level grid was moved to start at 0.2 after the held-out results were seen.

### Accuracy needs a ground truth, and the envelopes are not about the anatomy: step 4

Source: NOTES 2026-10-17, "Robustness step 4". The target is the ISMRM 2015 phantom's own density presence,
count-matched to each run.

| stated level | 0.8 | 0.9 | 0.95 |
|---|---|---|---|
| labeled 17-run envelopes cover | 0.650 | 0.724 | 0.778 |
| a bias-free ensemble of ground-truth subsamples covers | 0.818 | 0.904 | 0.951 |

- **The gap is 17–18 points.** Sparsity is not the cause, since the bias-free ensemble has the same sparsity.
- **Part of the gap is the label mapping.**
  - Envelopes built from the scorer's truth-assisted assignment, rather than from RapidParc's labels, fall 12–13
    points short of their own baseline.
  - RapidParc names 57.6 % of UKF's valid streamlines as their counterpart.
  - Of the 275 streamlines it calls AF (a tract the phantom does not contain), the scorer calls 147 valid SLF.

## The product

### Which method supplies the regional claims: open after step 6

Everything below describes today's UKF ensemble, the only method with an ensemble and a card. Step 6 offers two other
candidates:

| method | phantom | cIIIs04 rescans, matched counts | what it would need |
|---|---|---|---|
| UKF + RapidParc (today) | F1 0.412 | baseline | in place: ensemble, card |
| PTT + ACT (CSD) + RapidParc | F1 0.583; count-matched Dice 0.618 against 0.521 | slightly steadier than UKF, by 0.01–0.07 (one subject; the CST part needs the post hoc edge rule) | an edge rule, an ensemble design, a card, and a check of RapidParc's labels |
| TractSeg | Dice 0.551, masks (not comparable) | Dice 0.89–0.96, on its own terms | an uncertainty model (one mask per scan), its own registration per scan, a card, and an in-domain check |

- **Whichever method is chosen, the ensemble and the card are rebuilt for it.** The card's settings hash names today's
  tracker and labeler; neither transfers.
- **RapidParc's training domain is a constraint.**
  - It was trained on UKF tractograms.
  - On the phantom, it labels PTT's streamlines correctly more often (64.35 % against 57.59 %), so the shift is not
    plainly worse.
  - Still, a PTT product needs evidence that the labels hold on real scans, or a labeler trained on such tractograms.
- **A PTT ensemble would average PTT's own sampling as well as the data's noise.** Its bootstrap design and card must
  be built for that.
- **A segmentation model has no ensemble yet.**
  - P(a rescan has the tract here) would need replicates run through it, each with its own registration.
  - Its tract definitions, its AF above all, are not ORG's.
- **Layer C exists only for tractography.** A segmentation-based product has no streamline names to misread.

### The unit: one ensemble of the whole pipeline

- **R = 17 runs of the pipeline as one unit.**
  - The runs are the scan as acquired, and 16 wild-bootstrap replicates. The replicates use smooth in-plane signs
    (FWHM 4 voxels) and keep the b0s as acquired, with added noise.
  - Each replicate's field is re-estimated and applied; the result is tracked, then labeled
    (`docs/uncertainty-ensemble.md`; the card's `pipeline` → `settings`).
- **Not a stage-by-stage budget.** The product reports the ensemble's total, because the stages do not add and their
  sum would overstate it. Per-stage arms remain validation experiments, not per-scan outputs.

### Cost, on one basis

All times are on the M2 laptop. UKF and RapidParc run on its GPU (Metal); PTT and TractSeg run on its 8 CPU cores.

| method | one run, correction included | a 17-run ensemble | source |
|---|---|---|---|
| UKF + RapidParc | 87–91 s (medians, cohort III); 89.3 s for cIIIs04 A | about 25 min (derived: 17 × about 90 s) | step 5; `robust_benchmark.json` |
| PTT + ACT + RapidParc | 3.5–5.6 min | about 1–1.5 h | NOTES 2026-10-19, "Robustness step 6" |
| TractSeg | 94–121 s for a rescan; 213.5 s for A, including a one-time 94 s registration | no ensemble designed | `robust_benchmark.json` |

- The CPU and GPU times are not a like-for-like speed comparison.
- Most of the gain from more runs comes by 9–13 runs (NOTES 2026-10-09, "A held-out subject"). But a card belongs to
  one R, and this card's R is 17.

### Layers and claims

Each layer says what it claims, how well that repeats, and in what form it is stated.

| layer | claim | how stated | evidence |
|---|---|---|---|
| A. structure | where white matter is, and its local direction | descriptive maps, with the protocol's measured rescan agreement beside them | density shared 0.75–0.79, \|cos\| 0.94–0.97 (cohort III) |
| B1. tract presence | where a named tract lies | envelopes at a target coverage (via the card); the Vorob'ev shape for a volume or extent | held out: targets of 0.8 / 0.9 / 0.95 met within about 1–2 points |
| B2. composition near a point | the main named tract within 8 mm | a calibrated probability (via the card) | point ECE 0.014 / 0.022, held out |
| C. single streamlines | a streamline's tract name | diagnostic only; never a finding | unanimous seeds hold on 91–93 % of rescans |

- **Layer A has no card statement.**
  - Its rescan agreement was measured on cohort III and cIs1, but it was not calibrated per voxel.
  - It is stated as a protocol-level figure, not as a per-voxel probability.
- **Layer B1: the grid.**
  - The card was fitted on A's diffusion grid.
  - Stating B1 on FastSurfer's conformed 1 mm grid, where the anatomy is, relies on the grid-free check. That check
    covers one subject and three tracts, at uncalibrated levels.
  - Until the card is refit or checked on the 1 mm grid, the product states B1 on the diffusion grid, or says it has
    moved the statement.
  - The grid should become a `validity` → `match` field.
- **Layer B1: shapes and coverage.**
  - A quoted volume or extent uses the Vorob'ev shape: held-out size 0.99–1.03, with coverage and precision both 0.80
    in the same session.
  - Envelope coverage is pooled over the 42 named tracts and dominated by large tracts. Per-tract statements were not
    validated.
- **Layer B2: the card's point statement** is the main named tract within 8 mm, for points on a lattice.
  - The product may also list the other tracts the runs gave there. The evidence for that list is per seed, on one
    subject: a rescan's tract was among the 17 runs' tracts for 92–95 % of cIIIs04's seeds (NOTES 2026-10-09, "Landmark
    sites on cIIIs04").
  - The list is therefore descriptive at the point level, not calibrated.
  - The labeler's own runner-up is not shown. It is the new tract in about half of the changes, but its probability
    (median 0.002) is no guide.
- **Layer C** is kept for diagnosis, and for comparison with earlier entries. The card maps it and marks it diagnostic
  only.
- **Other holds about half of all streamlines,** and it is left out of the named-tract statements. A display must say
  so.

### Every probability names its reference event

| event | what the statement means | covers | does not cover | source of the numbers |
|---|---|---|---|---|
| another run ("run-to-run only") | the share of this pipeline's runs on this scan that give it: a level, not a probability of any real event | the DWI's noise through the field, tracker and labeler, as the bootstrap models it | everything a real rescan adds (below); between-session change; the anatomy | the ensemble's shares, as stated |
| a same-session rescan | P(a rescan minutes later gives this) | the bootstrap's noise, plus what a real rescan adds: its own noise realization, not centered on this scan's estimate; small head motion (0.15–0.64 mm, 0.07–1.1° in step 5's pairs); and its resampling onto the reference grid | another session; the anatomy | the card's same-session term |
| a rescan in another session | P(a rescan at another visit, on the same scanner, gives this) | the above, plus a new head position (1.5–6.0°, 2.7–15.6 mm in step 5's pairs), slice placement and registration | the session interval (not checked); another scanner or site; the anatomy | the card's other-session term |
| the anatomy | P(the true tract is there) | not calibrated | | the provisional phantom term, shown separately, never applied |

- **Under "run-to-run only," the numbers are shown as levels.**
  - An envelope's stated level is a share of the vote mass. On the phantom, a bias-free 0.5 envelope covered 0.766 of
    the truth.
  - A streamline or point share is the fraction of runs that agree.
  - Even about runs these are slightly optimistic: a left-out replicate gives the mode in 0.740 of cases where 0.764
    is stated.
  - About rescans they are much more optimistic: a rescan gives the mode in 0.661–0.703 (NOTES 2026-10-09, "Landmark
    sites on cIIIs04").
- **A rescan in the card's sense includes its resampling onto the reference scan's grid,** which the reference does not
  undergo (step 5).

### The calibration card is required for calibrated statements

The card (`tractline.calibration_card/2`, `calibration_card_masivar_cIII.json`) maps a scan's 17-run statistics to
calibrated statements about a rescan.

- **What it holds:**
  - per term (same session, other session): isotonic maps for the streamline mode and for the main named tract within
    8 mm, indexed by the vote count k of 17;
  - the envelope's stated level for each target coverage;
  - the expected coverage, precision and size of the core, the Vorob'ev shape, the widest outline and the envelopes;
  - the held-out evaluation, per subject and per fold.
- **What it checks before it applies** (`validity` → `match`):
  - the protocol: 112 volumes (16 b0s, 40 at b1000, 56 at b2000); the 112 × 112 × 54 matrix; voxels within 0.01 mm
    of 2.1429 × 2.1429 × 2.2; phase encoding j / j-; one reversed b0 at 0.85–1.05 of the forward b0s;
  - R = 17;
  - the pipeline settings' sha256 and the labeler weights' sha256;
  - the terms.
- **What `match` does not check:**
  - **Scanner and site.** The other-session term depends on them, yet a scan from another scanner running the same
    protocol would pass. Proposed new fields: scanner (manufacturer and model) and site.
  - **The grid** the statements are made on (above). Proposed as a field.
  - **The population.** The card's domain is "healthy adults", which no field in a scan can verify, and pathology is
    not machine-checkable. Proposed: a population that the user must declare. If the user declares a patient, or
    declares nothing, the product falls back.
- **Without a match, the product says "run-to-run only"** and shows levels, as above.
  - Today, every scan outside MASiVar cohort III on scanner B falls back. PAT16 falls back, and so does any clinical
    scan.
  - The product does not borrow a card from another protocol. PAT16's double-bootstrap map fit cohort III, but not
    cIs1 (below).
- **Any change to the pipeline needs its own card.**
  - The current card is for today's defaults: index grouping, UKF defaults, no denoising, labeler draw 0, the mps
    device.
  - Adopting either reproducibility opt-in, or PTT + ACT with ACT's edge rule, means refitting the card. The step 5
    queue was 165 new runs.

### The accuracy term: provisional, and never stacked on the card

- **What it is** (`accuracy (provisional)`, `"apply": false`).
  - On the ISMRM 2015 phantom, the gaps between the labeled envelopes and a bias-free ensemble are 0.168, 0.180, 0.176
    and 0.172 at stated 0.8, 0.9, 0.925 and 0.95.
  - Beside it are recorded: mean F1 0.412, 20 of 21 bundles valid, and 57.6 % of valid streamlines labeled as their
    counterpart.
  - Part of the gap is the label mapping, since the truth-assisted envelopes fall 12–13 points short.
- **Why it is not applied:**
  - The gaps were measured at the uncalibrated stated levels, not at the card's. So they cannot be added to, or stacked
    on, the card's maps, which are about rescans.
  - It comes from one simulated brain at b = 1000 with 32 directions, with its own noise (DW SNR about 7.6), readout
    and labels. That is unlike this protocol.
- **How the product shows it:** as a separate sentence beside any regional claim, never as an adjusted probability.
  For example: "On one simulated brain, on another protocol, this pipeline's envelopes covered 17–18 points less of
  the true tract than a bias-free ensemble would."
- **What would turn it into a calibration:** phantoms or tracer data on a matched protocol, several realizations, and
  envelopes scored at the card's levels. `docs/validation-datasets.md` names the candidates: the VoTEM phantom's real
  repeated scans and IronTract's tracer ROC. It marks their availability as unverified.

### Reproducible by construction

- **Bit for bit per draw, on one machine.**
  - Replicate r uses seed r, and the labeler's draw is fixed. Rebuilding a run reproduces its record (NOTES 2026-10-17,
    "Where the ensemble's spread comes from").
  - `calibration_runs.py verify` reran cIIIs04's A at 53a0e2e. It gave the same 64,378 streamlines, clusters and mask,
    with every vertex within the records' 0.025 mm quantization (step 5).
- **The reproducibility opt-ins: candidate defaults, your decision.**
  - **Hash-sorted grouping** (step 1) keys RapidParc's context groups to each seed.
    - Epsilon renames fall from 3.96 to 1.61 %.
    - Accuracy on TractCloud's test split, draw-to-draw agreement and the tract mix do not change.
  - **The UKF tie-break** holds both tensor swaps while the two directions are within 0.5° (`tie_angle=0.5,
    tie_sw2=True`; step 2b).
    - Paths more than 5 mm apart under epsilon fall from 3.42 to 1.16 % over the whole brain.
    - With hash-sorted grouping, renames fall to 0.58 %.
  - **Costs:**
    - the tract mix shifts by a TV of 0.004 (A), and 1.3 % of long seeds are lost;
    - the rule overrides UKF's designed sw2 alternation at near ties;
    - on the phantom, the pair is accuracy-neutral (F1 0.411 against 0.412).
  - **Limits:**
    - neither opt-in changes rescan agreement;
    - agreement across backends (Metal, CPU, CUDA) was not measured, and the Triton kernels carry the rule untested;
    - one subject.
- **A nondeterministic stage to avoid or pin.** Patch2Self's random sketch alone renames 8.9 % of streamlines between
  seeds 0 and 1 (step 3). PTT has RNGs of its own. If any such stage is used, its seeds are part of the record and of
  the card.
- **The store.**
  - The ensemble goes into a duckn store, as built for cIIIs04 (NOTES 2026-10-13, "The ensemble as a duckn store").
    The store holds:
    - per-class votes;
    - the scan's own run;
    - rank fields;
    - a `tractline` extension block with the envelope vote counts, the coverage they held, and the T1-to-DWI transform.
  - This proposal adds to that block the card's ID, the event and grid of each stored statement, and the pipeline's
    commit and settings hash, so a reader can tell which statements are calibrated, and against what.
  - The compact run records (about 7.7 MB a run, with full geometry) stay beside it for diagnosis.

### Cross-stage flags, revised

The earlier proposal asked for flags where a stage's output is undetermined. The results support these:

- **Endpoints against anatomy: a flag, not a filter** (NOTES 2026-10-09, "Endpoints against FastSurfer's anatomy").
  - Streamlines whose ends are consistent with their bundle's rule, in FastSurfer's labels, keep their name on a
    rescan 12–15 points more often.
  - But voting only among consistent runs predicts a rescan worse: 0.52–0.55 against 0.66–0.70.
  - The rules have not been reviewed by an anatomist.
- **The CST's lower end turning into the cerebellum** is a specific, measured failure (above). It is worth flagging per
  streamline and per region.
- **The field: flag it from the acquisition, not from the ensemble.**
  - With 16 forward b0s, the replicates' fields vary by about 0.1 mm, too little to flag anything.
  - On cIs1 (3 forward b0s), the replicates' fields stayed at 105–114 Hz, while the rescans gave 95 and 159 Hz (NOTES
    2026-10-08, "The bootstrap against real rescans").
  - So the bootstrap's field spread does not reveal a poorly determined field. The number of b0s and the card's
    protocol check do.
  - The residual check is a warning, not a refusal (NOTES 2026-10-03, "A residual check before correcting"). On this
    protocol, every correction is above its 0.5 warning level (A 0.514), so its threshold needs setting per protocol
    before it can flag anything here.
- **The edge of the data.**
  - The field of view cuts the brainstem: 86 % of CST streamlines end within 3 mm of the inferior face (NOTES
    2026-10-09, "Endpoints against FastSurfer's anatomy").
  - The CST's lower end is therefore always near an edge.
  - Under ACT, an empty edge slice dropped half of a rescan's CST (step 6). A flag should mark tracts that reach the
    data's edge.
- **Outside the card's domain:** a protocol, scanner or site mismatch, a declared patient, or a pipeline option without
  its card. This is the flag that matters most.
- **Not flags:**
  - where paths part at crossings (the crossing detector calls 83–93 % of the brain crossing; NOTES 2026-10-08, "Where
    the paths part");
  - direction spread along the CST: descriptive only, AUROC 0.746 with the vote share against 0.740 without (NOTES
    2026-10-13, "Direction spread along the CST");
  - the labeler's runner-up probability.

### Avoiding uncertainty where it can be avoided

- **Report at the level that repeats.** For the CST, the outline (85 % of a rescan's bundle) answers a clinical
  question better than streamline names (27–33 % kept).
- **Pool sessions when a patient has them.** Pooled ensembles cover as stated with no correction, but they are only
  2–5 % tighter than the card's single-scan envelopes at 0.9 (step 5). Whether to use them is your decision.
- **An anatomical constraint at the pons** (proposed, not built). A CST path should not enter the cerebellum.
  FastSurfer gives the cerebellum, brainstem and ventral DC (NOTES 2026-10-13, "Why CST streamlines become Other").
  ACT is one general form of anatomical stopping; it was not tested at the pons.
- **A second pass guided by the ensemble** (proposed, not built).
  - Any such change is a new tractography, so it needs its own card.
  - It must be scored on the phantom, because the stiffer tracker and Patch2Self show that rescan agreement can rise
    while accuracy falls.

### What not to claim

- A single streamline's tract name as a finding.
- An envelope or probability as a statement about the anatomy.
- Calibrated probabilities for a scan whose protocol, scanner, settings or R do not match a card.
- A "run-to-run only" level as a probability.
- Per-tract calibrated coverage. Coverage was validated only pooled over tracts; the focus tracts' 0.898 / 0.904 at
  target 0.9 is a scratch figure.
- Anything in a tumor, edema or other pathology from a card fitted on healthy adults.
  - PAT16's 17 runs separate tracts that enter the tumor mask in every run from those that enter in about a third,
    which one run cannot do (NOTES 2026-10-08, "PAT16 at the tumor").
  - That is reproducibility, not validation against stimulation mapping or histology.
- A stage-by-stage breakdown of the uncertainty for a given scan.
- That a lower spread means a better result.
- That TractSeg's tracts are ORG's. Its AF is about three times UKF's AF volume on cIIIs04.

## What UKF's settings could not fix

- **A rescan renames about 35–40 % of streamlines on this protocol, and 47.5–49.9 % on cIs1.**
  - On cIIIs04, the rates are 35.4 / 38.2 / 39.7 %.
  - Across eight held-out subjects, a same-session rescan gives the ensemble's mode for 69.3 % of seeds.
  - Over the five subjects with other-session rescans, it gives the mode for 66.6 % (step 5).
- **It is the paths.**
  - A rescan moves 72–78 % of A's streamlines by more than 5 mm (a bootstrap run moves 65 %).
  - Those streamlines are renamed 45–48 % of the time, under every labeler grouping (step 1).
  - Real rescans differ by noise along the whole path. The tie-break only removes the tracker's amplification of
    rounding (step 2b).
- **Every UKF setting or denoiser that reduced it lost accuracy on the phantom:**
  - step length 0.6: renames 3.2–4.3 points lower, F1 0.330;
  - Rs 0.1: 5.6–6.8 points lower, F1 0.314;
  - Patch2Self: 3.5–4.5 points lower, F1 0.278, in the phantom's 33-volume regime;
  - against today's 0.412.

  A shared inductive bias raises test-retest agreement whether or not it matches the truth.
- **A change of method moves both numbers** (step 6).
  - PTT + ACT, as a pipeline, is more accurate on the phantom. At matched counts it is slightly steadier than UKF on
    rescans, by 0.01–0.07: one subject, one run per scan, A's tissue maps reused on every rescan, and, for the CST
    only, a post hoc fix to ACT's edge rule.
  - TractSeg's masks repeat at Dice 0.89–0.96, though that comparison favors it in several ways.
  - Neither method was measured per streamline across rescans. Neither has an ensemble or a card.
- **So the product's answer is to state the disagreement, calibrated, at the level where it is small enough to use:
  regions and envelopes, not names.** Which method produces the regions is a separate choice, now open.

## Protocol dependence

### The protocols in these results

| data | scanner | voxels | diffusion | forward / reversed b0s | tracked shell | what was measured |
|---|---|---|---|---|---|---|
| MASiVar cohort III | Philips Achieva 3T, scanner B | 2.1 × 2.1 × 2.2 mm | b1000 ×40 + b2000 ×56 | 16 / 1 | 2000 | the stage split, steps 1–3 and 6, the card (8 subjects) |
| MASiVar cIs1 | Philips Achieva 3T, scanner A | 2.5 mm, 38 slices | b1000 / 2000 / 3000 ×96 (3 of 5 shells) | 3 / 1 | 3000 | one subject, two rescans |
| PAT16 (ds001226) | Siemens Trio | 2.5 mm | b700 / 1200 / 2800, 16 / 30 / 50 directions | 8 from both polarities | 2800 | replicates only, no rescan |
| ISMRM 2015 phantom | Fiberfox simulation | 2 mm | b1000 ×32 | 1 / 1 | 1000 | accuracy against a ground truth (steps 4 and 6) |

Sources: NOTES 2026-10-07; 2026-10-08 ("The bootstrap against real rescans"; "Smooth signs on PAT16"); steps 4–6.

### Is the cohort III protocol especially bad? Not worse than the others here

- **It is the richest acquisition here for the field.** It has 16 forward b0s, and its field varies by about 0.1 mm
  across replicates. It has 96 directions over two shells, at 2.1 mm in-plane.
- **Its run-to-run spread is like PAT16's.**
  - A run renames 31.6 % of A's streamlines on cIIIs04, at labeler draw 0.
  - A held-out run changes 34.0 % of PAT16's labels (35.1 % with smooth signs), with labeler draws included, on
    another vendor's three-shell scan.
  - The draw alone accounts for about 3 %.
- **cIs1 is worse on every measure.**
  - Its rescans change 47.5–49.9 % of names, against 35–40 %, and every map fails on it.
  - It differs in several ways at once: tracking at b = 3000, 2.5 mm voxels, 38 slices that cut the cortex, 3 forward
    b0s and another scanner. No single cause can be named.

**Weaknesses of the cohort III protocol:**
- **In-plane correlated noise.**
  - A's DW residuals correlate with their neighbor at lag 1 by 0.42 / 0.24 / 0.08 along the three axes, and by −0.13
    at lag 2 along the second axis (step 3). cIIIs01, on the same protocol, gives 0.47 / 0.31 / 0.08 at lag 1.
  - The negative lag-2 value suggests the reconstruction: a k-space filter or interpolation. The sidecars do not say,
    so the source is unconfirmed.
  - The correlation is why per-voxel bootstrap signs understated a rescan.
  - It is one of two reasons MP-PCA at radius 2 removed almost nothing. The other is that the patch's 125 voxels barely
    exceed the 112 volumes.
  - PAT16's residuals are correlated too (0.28 / 0.43 / 0.03), so this is not peculiar to Philips.
- **The field of view cuts the brainstem.** 86 % of CST streamlines end within 3 mm of the inferior face. That matters
  for the CST's lower end, and for any tracker with valid-end rules: step 6's edge artifact came from it.
- **Smaller points:**
  - the sidecars state no readout time, so a nominal 0.05 s is used. The field's Hz scale is therefore nominal, which
    is harmless within the protocol;
  - there is only one reversed b0.

### The mechanisms look general

| mechanism | seen on |
|---|---|
| one run's probability is overconfident; an ensemble ranks a fresh run's changes much better | PAT16 (AUROC 0.80 against 0.65), cohort III, cIs1 (0.73–0.74 against 0.65–0.66) |
| labels change because paths change; the labeler is rarely unsure | cohort III, cIs1, PAT16 |
| the bootstrap is centered on the scan and understates a rescan | PAT16 (double bootstrap), cohort III, cIs1 |
| in-plane correlated noise | cohort III, cIs1, PAT16 |
| regions repeat better than names | cohort III, cIs1, PAT16 (replicates) |
| steadier UKF variants are not more accurate | the phantom (another acquisition) against cIIIs04's rescans |
| the DWI's noise through the tracker dominates; stages do not add | cIIIs04 only |

The last row rests on one subject. It may differ where the field rests on few b0s, as on cIs1; that was not tested.

### The magnitudes and the card are per protocol

- **PAT16:** a fresh run changes about 34 % of labels.
  - Its double-bootstrap map, fitted with no real rescan, fit cohort III's rescans: ECE 0.004–0.011 on cIIIs01 and
    0.0066 on cIIIs04.
  - Sources: NOTES 2026-10-08, "The ensemble scored on real rescans"; 2026-10-09, "A held-out subject".
- **cIs1:** rescans change 47–50 % of names.
  - That same map states about 0.60 where 0.51 is observed (ECE 0.091).
  - Its envelopes stay about 10 points short with every map (NOTES 2026-10-09, "Compact run records, and presence
    maps").
- **The phantom** is a different acquisition: b = 1000, 32 directions, a DW SNR of about 7. Patch2Self removed more
  than the estimated noise there, and the tracker settings' effects may differ with the shell.
- **The card's other-session term** depends on the scanner and site as well as the protocol. It was fitted on one
  scanner at one site.

### The next test: a second protocol

The question is whether a card can be built and held out on another protocol, and how far its maps differ from this
one's.

- **cIs1, on disk.**
  - It has one subject, with a same-session and a next-day rescan, scored at R = 9 with older records.
  - It would test the few-b0 field, and give a first look at another protocol's maps.
  - But a card needs more subjects on that protocol, and how many MASiVar has was not checked here.
- **Zhang et al. 2019** (HBM; `docs/prior-art-uncertainty.md`).
  - It used UKF tractography with the ORG atlas, the same family of pipeline, on three test-retest datasets with 255
    subjects.
  - Whether the data can be obtained, and what their protocols are, was not checked.
- **HCP Test-Retest.**
  - It was used for connectome-level bootstrap work (Roine et al. 2019, `docs/prior-art-uncertainty.md`).
  - Its DWIs need an HCP account (`docs/validation-datasets.md`), and its protocol was not reviewed here.
  - It would also be TractSeg's own domain.

## Open work and limits

- **One scanner and protocol for the card:** MASiVar cohort III on scanner B.
  - Other protocols, scanners and sites are untested, and cIs1 shows the maps need not transfer.
  - The card's `match` checks neither scanner nor site.
- **Healthy adults only.**
  - The card has 8 subjects. Five have rescans in other sessions: 9 rescans, 4 of them one subject's.
  - The session interval was not checked.
- **One simulated phantom at b = 1000.** The accuracy term is provisional, protocol-mismatched and not applied.
- **Patients and pathology are untested.** PAT16's tumor analysis measures reproducibility only, and FastSurfer's
  labels near the tumor are unchecked.
- **Cross-backend reproducibility was not measured.**
  - The card ran on one device (mps).
  - Aggregates are known to hold across five platforms: PAT16's Other share spans 0.3 points (NOTES 2026-10-02,
    "RapidParc is the default").
  - Per-streamline agreement across platforms was not measured.
- **Per-tract card statements were not validated.** Coverage is pooled over tracts and dominated by large ones.
- **The card's grid** is A's diffusion grid. Moving its statements to 1 mm rests on a one-subject check.
- **R is fixed at 17** for this card; another R needs a refit.
- **Step 6 rests on one subject and one phantom, with one run per scan.**
  - ACT's edge rule is post hoc. It was checked on Dice and volumes, not on the sites, and it is not in the
    benchmark's records.
  - PTT and ACT are not separated.
- **Next:**
  - a PTT + ACT ensemble and its card, with a bootstrap design suited to a probabilistic tracker;
  - RapidParc on CSD tractograms: whether its labels hold, or a labeler trained on such tractograms;
  - TractSeg in its own domain, on HCP-like data, with a registration per scan and an uncertainty model;
  - a second protocol for whichever method is kept.
- **Not built:**
  - the pons constraint;
  - an ensemble-guided second pass;
  - tracking each session on its native grid for pooling;
  - the store's card, event and grid fields;
  - the card's proposed scanner, site, grid and population fields.

## Decisions that are yours

- Whether hash-sorted grouping and the UKF tie-break become defaults. Each would need the card refit.
- Whether to adopt ACT's edge rule (the scan's own data edge as a valid end), if PTT + ACT is pursued.
- Which method supplies the regional claims: UKF, PTT + ACT, or a segmentation model.
- Which comes next: a card on a second protocol, or a PTT ensemble and its card.
- R, given the cost: about 25 min for 17 UKF runs and 1–1.5 h for 17 PTT runs on the M2, with most of the gain by
  9–13 runs. A card belongs to one R.
- Whether the product pools sessions when a patient has more than one scan.
- The residual check's threshold, per protocol.
- Whether the product ships layer C at all, even as a diagnostic.
- The name of the "card" (logged 2026-10-19, open). It is our shorthand for a protocol-specific recalibration map
  learned from test-retest scans, packaged with its validity domain, match check, provenance and held-out evaluation.
  "Calibration card" is not a term in the literature (it borrows loosely from model cards and datasheets); the
  technique is standard post-hoc (isotonic) and quantile recalibration. The ISMRM 2027 abstract uses "protocol-specific
  recalibration map". Decide later whether the repository keeps "card" or renames it; the name may suggest itself as the
  work goes on.
