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

## The plane rankfield lacks: votes

Margins and probabilities read from a depth-cut field do not survive regrouping after the fact; this is
rankfield's group-blind depth cut (NOTES 2026-10-01, M2). Votes do survive it. Each run's label is the
group of its winning class, so a group's votes are its members' votes summed, exactly.

The prototype tried two ways to store them beside `ranks`, `support` and `tail`.

**Votes aligned to the ranks**

| array | shape | dtype | meaning |
|---|---|---|---|
| `votes` | (N, Z, Y, X) | uint8 | runs in which the class at rank j won; 0 where the rank is absent |
| `vote_tail` | (Z, Y, X) | uint8 | runs whose winner is not among the kept classes |
| `runs` | (Z, Y, X) or scalar | uint8 | runs that labeled this element |

These satisfy `sum(votes) + vote_tail == runs`. They are 7 B per element, and regrouping from them is exact where
`vote_tail == 0`.

- **Voxels (44 classes, keep shell):** enough. The tail is zero in 92 % of tract voxels.
- **Streamlines (1,600 clusters, depth 6, keep clip):** not enough. The tail is zero for only 75 % of streamlines,
  because winning clusters scatter: across 11 runs, a median of about 5 distinct clusters won the same seed.

**Each run's winner outright**

| array | shape | dtype | meaning |
|---|---|---|---|
| `winners` | (R, Z, Y, X) | uint16 (uint8 at <= 255 classes) | class + 1 that run r chose; 0 = the run did not label this element |

These are 2R B per streamline (22 B at R = 11, beside the field's 19 B) or R B per voxel. Votes for any grouping are
exact, and the order of the runs is kept, so the runs can be resampled again later. This is the recommendation for
streamlines.

**Recording the ensemble.** Either way, the block records how the ensemble was made: R, which stages were resampled,
the bootstrap's settings, and the draws.

**Keep rule.** A keep rule that stores every class that won any run, before filling the remaining depth by gap, would
make the rank-aligned votes exact. At the streamline level it would need depth of order R.

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
3. Do vote planes regroup exactly at depth 6?
4. How much does the voxel field's consensus beat a single run's map at predicting a fresh run?

It does not measure accuracy: there are no reference tract labels for these scans. Bootstrap replicates
cover only the variability that is modeled, and model bias is untouched. One patient.

**The b0s.** A replicate's b0s must keep the scan's motion. An earlier version replaced each b0 with the b0s'
mean plus noise. That removed the motion between b0s (thermal-scaled residuals) or scrambled it voxel by voxel
(the b0s' own residuals). The replicates' fields then sat at a 99th-percentile |field| of 184 Hz or 145 Hz,
against the reference's 154.

Keeping each b0 as acquired and adding thermal-level noise (`--b0-noise added`) doubles the b0 noise variance.
It puts the replicates at 153.4-155.3 Hz, around the reference's 154.0, with residual_left 0.331-0.333
against 0.331. Every streamline and voxel metric above stays within run-to-run spread: AUROC 0.80, ECE 0.03
against 0.27, voxel agreement 67 % against 61 %.

**Open.** The reference is still closer to a replicate (33 % labels changed) than two replicates are to each
other (38 %), in every b0 mode. A candidate is the per-voxel sign flips, which whiten DW residuals that are
structured in the scan as acquired. A block bootstrap would test it.
