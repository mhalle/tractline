# The pipeline's uncertainty, kept: an ensemble and a rank field to hold it

Draft, 2026-10-07. A prototype on the `uncertainty-ensemble` branch: `src/tractline/ensemble.py`,
`bench/ensemble_pilot.py`, measured in `bench/results/NOTES.md` (2026-10-07, "The pipeline's ensemble").

## The problem

A single run of tractline gives each streamline the labeler's probabilities and nothing else. TRX holds
two numbers per streamline: the tract's probability and its margin over the next tract. The labeler's
probabilities do not include:

- the scan's noise;
- the susceptibility field estimated from that noise;
- the tracker's path through the corrected scan;
- the labeler's own draw (RapidParc's shuffle; TractCloud's context).

Each of these moves labels, and the first three also move paths. The noise floors measured on PAT16
(NOTES 2026-10-02) show that a fresh run of the pipeline changes labels that one run's probabilities
called confident.

## The ensemble

Run the pipeline R times with every stage resampled:

1. a wild-bootstrap replicate of the scan as acquired, with each shell fitted at the SH order its
   directions allow, and each b0 (the reversed pair's too) kept as acquired with thermal-level noise added,
   so the motion between b0s that the field estimate sees is the scan's own;
2. the field re-estimated from that replicate and applied;
3. the replicate tracked;
4. the result labeled with its own draw.

From each run, keep a `RunRecord`: the labeled streamlines keyed by seed voxel (one seed per mask voxel,
so a seed voxel names the same streamline in every run whose mask holds it), their winning clusters and
cluster log-probabilities, 20-point paths, and per-voxel streamline counts per tract.

## Two rank fields

rankfield stores, per element, the kept classes as ranks, how far each trails the winner as a byte,
and the dropped mass as a tail. The ensemble gives it two kinds of element.

**Per streamline** (the web demo's shape, `(n, 1, 1)`). The field encodes the *mixture*: the log of the
mean cluster probability over the runs that labeled the seed. This captures total uncertainty, not one
run's.

**Per voxel** (rankfield's own shape, `(Z, Y, X)`), with 43 classes plus NONE. The logits are the log of
each tract's mean streamline count per voxel, and NONE is held at log(tau). A tract wins where it is the
densest and denser than tau, so:

- its margin against NONE is log(density / tau), and its zero level set is the tract's iso-density
  surface at tau;
- its margin against another tract is where the two cross.

This is the field that carries *position* uncertainty (scan, field, tracker), which the per-streamline
field cannot. It restores onto any grid (the T1, a navigation volume) through rankfield's restore.

## Votes: tested, and not needed here

Votes are each run's winning class, counted. The case for them was:

- **Regrouping.** A group's votes are its members' votes summed, exactly, when a run's label is the group of its
  winning class (a streamline here). Margins read from a depth-cut field are not exact under regrouping.
- **Disagreement.** Votes separate runs that confidently disagree from runs that are all unsure, where the mixture
  shows the same number for both.

`bench/ensemble_votes_test.py` tested this on the pilot's records (NOTES 2026-10-07, "Are votes needed"). For a
labeler that gives probabilities, the answer is no:

- **The mixture predicts better.** It beats votes at predicting a fresh run's change in every grouping tried: tracts,
  merged tracts, outliers folded, clusters. The margin is 0.005-0.006 AUROC, intervals excluding zero, and the mixture
  is better calibrated too.
- **Where the rank field loses, depth fixes it.** Encoded at depth 6 the mixture loses 0.011 AUROC to the depth cut at
  tract level. Depth 8 (25 B per streamline) recovers most of that and depth 12 all of it; votes (22 B) recover less.
- **Votes add little beyond the mixture.** Conditional AUROC within deciles of the mixture is 0.51-0.53, and log-loss
  improves by about 2 %.
- **Voxel votes do not regroup exactly.** A voxel's winner is a density argmax, so merging tracts can change it: 11-23 %
  of a merge's voxels differ. Calibration barely moves.

**For rankfield**

- Store the ensemble's mixture as an ordinary field; nothing in the format changes.
- Spend more depth on it (8 or more), or use a keep rule that keeps mass.
- Read confidence as group mass (`probabilities`), not `decode_groups`' margin, which predicted 0.02 worse here.

**Where votes do belong** is where runs give hard labels and no probabilities: several raters, multi-atlas fusion,
other software run repeatedly. There the vote counts are the only uncertainty there is. Rank-aligned planes (`votes`
(N, ...) uint8, `vote_tail`, `runs`, with `sum(votes) + vote_tail == runs`) or per-run winners (`winners` (R, ...)) are
the two layouts the prototype tried. Rank-aligned planes held every winner in 75 % of streamlines and 92 % of tract
voxels at R = 11, falling as R grows. Per-run winners are exact at 2R B. A rule for resampling votes onto another grid
is still to be designed.

## What is measured, and what is not

See NOTES (2026-10-07) for the numbers. On PAT16 (R = 11, 6 held-out runs):

- **Change.** A fresh run of the pipeline changes 34 % of the reference's tract labels; the labeler's draw alone changes
  3 %.
- **Prediction.** The ensemble predicts which labels change at AUROC 0.80, against 0.65 for one run's TRX probability.
- **Calibration.** The ensemble's probabilities are calibrated (ECE 0.03 against 0.28). Labels a single run calls ≥ 0.99
  survive a fresh run 73 % of the time.
- **Voxels.** The density field's winner agrees with a fresh run's winner in 67 % of tract voxels, against 60 % for one
  run's map, and its vote fractions are calibrated (ECE 0.04).

The pilot asked four questions:

1. Does ensemble confidence predict a fresh run's change better than one run's probability does?
2. Is it calibrated where one run's probability is not?
3. Are votes needed beside the mixture? (Not for this labeler: above.)
4. How much does the voxel field's consensus beat a single run's map at predicting a fresh run?

It does not measure accuracy: there are no reference tract labels for these scans. Bootstrap replicates
cover only the variability that is modeled, and model bias is untouched. One patient. The ensemble's own
calibration is close to guaranteed, since its held-out runs come from the same bootstrap; the single run's
miscalibration is the finding.

**The b0s.** A replicate's b0s must keep the scan's motion. An earlier version replaced each b0 with the b0s'
mean plus noise. That removed the motion between b0s (thermal-scaled residuals) or scrambled it voxel by voxel
(the b0s' own residuals). The replicates' fields then sat at a 99th-percentile |field| of 184 Hz or 145 Hz,
against the reference's 154.

Keeping each b0 as acquired and adding thermal-level noise (`--b0-noise added`) doubles the b0 noise variance.
It puts the replicates at 153.4-155.3 Hz, around the reference's 154.0, with residual_left 0.331-0.333
against 0.331. Every streamline and voxel metric above stays within run-to-run spread: AUROC 0.80, ECE 0.03
against 0.27, voxel agreement 67 % against 61 %.

**The reference sits at the center of the replicates.** A replicate changes 33 % of the reference's labels,
but 38 % of another replicate's. This is not the noise level: it persists when the scan is made one of the
replicates' own sign patterns. Nor is it the per-voxel sign flips: it persists with per-volume signs. It is the
bootstrap's centering. A least-squares residual is orthogonal to the fit's space, so the tracker sees the scan
much as it sees the fit alone. The fit, the scan and the fit minus its residual (every sign flipped) differ by
only 10 % of labels and under 1 mm, against 28 % and 4.7 mm to any replicate (NOTES 2026-10-08).

So held-out replicates measure the spread around this scan's estimate, not that estimate's own error.

**Calibrated for rescans: a double bootstrap.** Some replicates are made pseudo-scans and bootstrapped again to
build their own ensembles. The other first-level replicates serve as their rescans: they spread about the same
center, as real rescans spread about the truth.

Scored this way, the pilot's numbers come back for the centered scoring. Against rescans:

- **Change rate:** 38 %.
- **AUROC:** 0.78 for the ensemble against 0.64 for one run.
- **ECE:** the mixture's rises from 0.03 to 0.07, against 0.32 for one run. Its ≥ 0.99 streamlines keep their
  tract on 93 % of rescans.

A map fitted on the other pseudo-scans brings held-out ECE to 0.0025. It is either isotonic on the mixture's
log-odds or a table per vote count, and the same table works for voxels. Applied to the real scan, it gives a mean
P(rescan keeps the tract) of 0.62, the replicate-vs-replicate rate. Under it no streamline reaches 0.99: eleven
unanimous runs still mean a 1-in-12 chance that a rescan changes the label (NOTES 2026-10-08).

**Checked against real rescans.** MASiVar (OpenNeuro ds003416) scans healthy adults repeatedly on one Philips
scanner. Three reference scans with five rescans in all were used:
- subject 1 (cIIIs01) at 2.5 mm, with one same-session rescan;
- subject 1 at 2.1 mm, with one same-session and one other-session rescan;
- subject 2 (cIs1), 38 slices and 3 forward b0s against subject 1's 16, with one same-session and one next-day rescan.

Each rescan is registered to its reference scan and resampled onto its grid, so a seed voxel means the same place in
both. Resampling alone changes some labels. Running the same resampling on the reference and on its replicates showed it
does not inflate the rescan figures. The labeler's random choice of context is held fixed, so only the data differ.

The first three columns compare two simulated replicates of the same scan, made with three ways of re-signing its
noise. The last two compare the scan with a real rescan. Share of streamlines whose tract changed:

| | replicates, per-voxel signs | replicates, smooth signs | replicates, per-volume signs | rescan, same session | rescan, other session |
|---|---|---|---|---|---|
| subject 1, 2.5 mm | 31.5 % | 35.0 % | 36.5 % | 36.3 % | — |
| subject 1, 2.1 mm | 32.2 % | 34.9 % | 35.8 % | 36.9 % | 39.1 % |
| subject 2 | 39.6 % | 40.7 % | 41.6 % | 49.9 % | 47.5 % |

- **Per-voxel sign flips understate a rescan.** Noise in neighboring voxels within a slice is correlated (lag-1
  correlation 0.1–0.5), likely from the scanner's image reconstruction; across slices it is under 0.1 in the
  diffusion-weighted volumes. Per-voxel signs remove that correlation.
- **Signs drawn as smooth patches within each slice close the same-session gap on subject 1** to 1–2 points
  (`flip="smooth"`, FWHM 4 voxels). Per-volume signs do as well on average, but each pair varies more.
- **Subject 1's one other-session rescan changes 39.1 %,** 3–4 points more than smooth or per-volume signs. Across all
  rescans, voxel agreement stays 3–9 points below the simulated pairs. Another session brings a new head position,
  slice placement and registration error, which re-signing noise cannot produce.
- **Subject 2 stays 6–10 points short with every variant.** Its field rests on 3 forward b0s, and its 38 slices cut
  the cortex. Its replicates' fields stay at 105–114 Hz while its rescans give 95 and 159 Hz.

Refitted with smooth signs on PAT16, the double bootstrap's ranking (AUROC 0.77) and recalibration (held-out ECE
0.002–0.003) are unchanged. Its confidence drops by 2–3 points, to a mean P(rescan keeps the tract) of 0.60 against
0.62.

Keeping each streamline's full tract probabilities, instead of only its top tract, barely helps: label change on a
rescan falls by about 1 point. The labeler is rarely unsure (top tract 0.93–0.94 on average; 0.4–0.5 % of
streamlines below 0.5). Labels change because the path from a seed moves, a median 13–21 mm on average along its
length, and reaches other anatomy. The ensemble, which pools over runs whose paths differ, is what can carry that
(NOTES 2026-10-08, "Smooth signs on PAT16, and soft labels").

**Scored on real rescans.** Each MASiVar scan was given its own ensemble of 9 runs, then scored against each scan it
was paired with, both ways (10 scorings). AUROC measures how well a stated confidence ranks the streamlines whose
tract changes (0.5 chance, 1 perfect). ECE is the average gap between stated and observed probability.

- **The ranking held on all five rescans:** AUROC 0.73–0.78, against 0.65 for one run's labeler probability.
- **The stated probabilities are too high:** 8–11 points on subject 1 and 15–18 on subject 2 (ECE).
- **PAT16's double-bootstrap map, carried over unchanged, matched subject 1's three rescans:** ECE 0.004–0.011, with
  the stated mean within about a point of the observed. The map was fitted with smooth signs on PAT16's Siemens scan,
  which has no rescan. It did not fit subject 2: it states 0.60 where 0.51 is observed.
- **Ensemble outputs are steadier between scans than one run's.** Tract territories overlap 1.6–1.9 times as much
  (0.28–0.39 against 0.15–0.24), though still under half.
- **Envelopes for named tracts are far too small.** A tract's envelope at a level is its highest-agreement voxels
  holding that share of its votes. As stated, a 90 % envelope holds 60 % of where the rescan put that tract. Every
  voxel the nine runs gave the tract holds about 68 %. Raising the stated level reaches a real 50 %, but nothing at 80 %
  or above. This counts each voxel for the one class with the most streamlines; maps that let tracts overlap, and more
  runs, were not tested.

**What repeats, and what does not.** Where white matter is (0.63–0.79 of the density shared), and its local direction
(|cos| 0.94–0.97, about 14–20°), hold on a rescan. In a sphere of 8 mm radius, the main named tract is the same on
81–90 % of rescans. Half of all streamlines fall in the labeler's Other class and are left out of this count. A single
streamline keeps its tract name on only 50–64 % of rescans ("Where and which").

Across simulated replicates, about half of a seed's paths stay within 4 mm of each other for 80 mm. The rest part at a
median 34–40 mm from the seed. Parting points fall in "crossing" voxels only 1.1–1.2 times as often as the path before
them. The crossing detector calls 83–93 % of white matter crossing, though, so this cannot show whether crossings drive
the change ("Where the paths part").

For PAT16, 17 runs separate tracts the pipeline places inside the tumor mask in every run (ILF, superficial frontal,
striato-frontal, striato-occipital) from those it places there in only about a third of runs (thalamo-frontal,
cingulum). A single run cannot tell these apart. The corticospinal tract comes within 5 mm in 8 of 17 runs. This
measures how reproducible the output is, not whether it is anatomically right: nothing here was checked against
stimulation mapping or histology ("PAT16 at the tumor"; `bench/ensemble_pat16_viewer.py` draws it).

**Presence maps and a held-out subject (NOTES 2026-10-09).** Letting tracts overlap in the voxel maps (a tract holds a
voxel where it has at least 2 streamlines; the voxel rank field's classes above NONE) makes the envelopes usable. As
stated, a 90 % envelope holds 78–82 % of a rescan's tract, against 60 % with winner maps. With levels recalibrated on
three earlier pairs, a third healthy subject (cIIIs04, held out) gets 84 % for an 80 % target and 92 % for 90 %.

PAT16's double-bootstrap map, fitted with no real rescan, brings that subject's streamline ECE to 0.0066. More runs
improve every measure, with most of the gain by 9–13 runs. The as-stated confidence falls with the number of runs, so
a recalibration map belongs to a run count.

What the reported confidence should be:

- the ensemble's agreement, adjusted to predict a same-session rescan using simulated rescans with correlated noise.
  That adjustment matched real rescans for one healthy subject whose protocol pins the field well (16 b0s), and failed
  for the second;
- fitted per protocol, since it depends on the pipeline and on the number of runs, and checked on test-retest scans
  for any new protocol;
- regional answers first: which named tracts pass near a point, and in what proportion. Envelopes cannot yet be stated
  at 80 or 90 % with 9 runs;
- read as a same-session confidence, and as untested in edema and tumor tissue. Against a scan from another session
  it is still optimistic by a few points, more on protocols whose field rests on few b0s.
