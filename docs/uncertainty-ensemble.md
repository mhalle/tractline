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

**The reference sits at the centre of the replicates.** A replicate changes 33 % of the reference's labels,
but 38 % of another replicate's. This is not the noise level: it persists when the scan is made one of the
replicates' own sign patterns. Nor is it the per-voxel sign flips: it persists with per-volume signs. It is the
bootstrap's centring. A least-squares residual is orthogonal to the fit's space, so the tracker sees the scan
much as it sees the fit alone. The fit, the scan and the fit minus its residual (every sign flipped) differ by
only 10 % of labels and under 1 mm, against 28 % and 4.7 mm to any replicate (NOTES 2026-10-08).

So held-out replicates measure the spread around this scan's estimate, not that estimate's own error. A fresh
scan's change is better estimated replicate against replicate (38 %), and the pilot's calibration is somewhat
optimistic for real rescans. Scoring held-out runs against a replicate's labels, not the reference's, would
remove the centring.
