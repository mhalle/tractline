# A tracker and labeler more robust to noise: the plan

Draft, 2026-10-17. It follows "Where the ensemble's spread comes from" (NOTES 2026-10-17, commit 33bcdec).

## What we know

- **Almost all of the spread is the DWI's noise reaching the tracker.** The distortion field adds little on this
  protocol.
- **Small perturbations of any kind draw on the same pool of fragile seeds.** A 0.1 mm field wobble renames 16 % of
  streamlines.
- **RapidParc's index-based grouping sets a floor near 3 %.** It relabels streamlines whose paths did not move.
- **Earlier NOTES point to the tracker's hard decisions.** Most path distance between runs is streamlines ending at
  different points (2026-10-07). Paths part a median 34–40 mm from the seed, only modestly more often at crossings
  (2026-10-08).

## The goal, and the trap

The goal is better agreement with real rescans. Lower run-to-run variance alone is not the goal: a tracker that
ignored the data would be perfectly reproducible.

Every step is judged on cIIIs04's four scans (A and the within, session 3 and session 5 rescans), on:
- **rescan agreement:** the share of streamlines a rescan renames; presence Dice of CST, AF and UF; the main tract
  within 8 mm of the landmark sites;
- **honesty:** a 17-run ensemble's stated confidence against rescans, and its AUROC (today 0.80 / 0.78 / 0.77);
- **no visible bias:** tract volumes, lengths and the Other share, against today's.

Accuracy needs a ground truth, so it is the last step (a phantom), not something these steps can show. Every change is
opt-in; no default changes until you decide.

## How it runs: one step at a time, subagents, managed from this session

- **Sequential.** One step runs at a time, in this checkout, on a new branch `robustness` cut from
  `uncertainty-ensemble`. Since nothing runs in parallel, there are no worktrees, no extra environments, no Dropbox
  sync of copies, no merge conflicts, and the GPU and memory are never shared.
- **Each step is a subagent run in phases, with this session managing:**
  1. **Build.** The subagent writes the code and runs a smoke test (one run), then reports.
  2. **Run.** I launch the long queue as a background job and watch it. The heavy runs stay in one place, and nothing
     runs while a subagent is still editing.
  3. **Analyze and draft.** The same subagent, resumed with its context, analyzes the runs and drafts the NOTES entry.
  4. **Review.** A separate reviewer subagent checks the code, every number against the JSON, any overreach, and
     American English.
  5. **Your gate.** I show you the results and the review's findings before fixes are applied (your standing rule).
     Then I apply the fixes, commit, and say whether the next step's brief should change in light of the result.
- **Data guards:**
  - each step writes only under `DATA/ensemble/cIIIs04/robustness/<step>/` and never modifies existing records;
    scripts refuse to overwrite;
  - compact records only;
  - a free-disk check before each queue: stop below 20 GB (33 GB free today);
  - large intermediates are deleted at the end of the step.
- **The machine:** one tracking job at a time on the M2's GPU. The pending M2 timing rerun needs an idle machine, so
  it can run between steps if you want.

## The steps

### 1. Labeler: context groups that do not depend on the seed set

- **What:** an opt-in grouping for RapidParc that keeps each seed in the same context group whatever other seeds
  exist. For example, assign group = hash(seed key, draw) mod G, with G = ceil(n / 2000). Sorting by key alone is not
  enough, because the 2,000-wide cuts shift when n changes.
- **Measure:**
  - accuracy on TractCloud's test split (`bench/accuracy_tractcloud_test.py`; RapidParc 94.5 % today);
  - agreement across draws;
  - renames on near-identical paths when the epsilon arm is relabeled (today about 3 %);
  - the full arm's rescan AUROC.
- **Cost:** labeling only, from the existing compact records, with no tracking. About an hour; a few hundred MB.
- **Limit:** RapidParc labels in context, so group-mates whose paths changed still matter. This reduces the floor; it
  cannot remove it.
- **Outcome (2026-10-17, NOTES "Robustness step 1").**
  - The opt-in groupings `hash-sorted` and `hash-bucket` cut the epsilon floor from 4.0 % to 1.6 / 1.5 %, and renames
    on paths within 0.1 mm from about 3 % to 0.4–0.5 %. Accuracy, draws and the tract mix are unchanged.
  - Rescan agreement and AUROC do not move, because the paths change: 65–78 % of streamlines move more than 5 mm and
    are renamed 43–48 % of the time under every grouping.
  - Hash-sorted is the candidate default: its groups always hold 2,000; hash-bucket's shrink toward 1,000 for some n.
    The default stays `index` until you decide.

### 2. Tracker: which decision makes paths diverge, then a softer one

- **Diagnose.** Rerun a sample of a few thousand seeds on A against the epsilon and fieldonly inputs. Record, step by
  step, which stop condition fired (mask, signal below 0.06, FA below 0.08, step limit) and every switch between the
  two tensors. Classify each pair that diverges by more than 5 mm by its first event:
  - a different stopping step;
  - a different switch;
  - the mask, which is recomputed per run;
  - gradual drift with no discrete event.
- **Soften the main cause.** Prototype one opt-in variant aimed at the dominant cause, for example:
  - a fixed mask, if the mask dominates;
  - hysteresis on the thresholds, or tissue-based stopping from FastSurfer (ACT/CMC style), if stopping dominates;
  - a minimum stretch before a tensor switch, if switching dominates.
- **Score:**
  - the epsilon and fieldonly perturbations, rerun (16 runs each);
  - A and the three rescans;
  - a 17-run ensemble for honesty;
  - tract volumes and lengths, to catch bias.
- **Cost:** about 2–3 hours of tracking; under 1 GB.
- **Outcome (2026-10-17, NOTES "Robustness step 2").**
  - About three quarters of epsilon's diverging pairs part with no ≥5° event. That class may include swaps between
    near-identical tensors, which the 5° rule cannot see. Divergence comes mostly from low-FA seeds (FA below 0.2:
    69 % of it); that the filter amplifies fastest where its two tensors are degenerate is a hypothesis. The mask
    plays no part.
  - Rs 0.1 and step_length 0.6 cut rescan renames by 3–7 points and raise AUROC by at most 0.018 (one subject, no
    uncertainty). But each is a different, stiffer tractography with large tract-level changes (Rs 0.1: AF volume
    ×0.43, CST ×1.32), and over-smoothing explains the gains as well as better tracking does.
  - No tracker setting is a safe default. The defaults are unchanged; the variants stay opt-in through
    `pipeline.track(tracker=)`.
  - Next, step 2b: a minimum stretch before a switch, or a tie-break for near-equal tensors (never tried). Then score
    the variants on the ISMRM 2015 phantom (docs/validation-datasets.md) with its ROI scorer. Its ground truth has no
    AF, and its b = 1000 shell may not carry the Rs effect at b = 2800.
- **Outcome of step 2b (2026-10-17, NOTES "Robustness step 2b").**
  - Holding both tensor swaps while the two directions are within 0.5° (opt-in `tie_angle=0.5, tie_sw2=True`) cuts
    epsilon's whole-brain paths more than 5 mm apart from 3.42 to 1.16 %. With step 1's hash-sorted grouping,
    epsilon's renames fall from 3.96 to 0.58 %.
  - The cost is small in the aggregates, on one subject: tract mix total variation 0.004, 1.3 % of long seeds lost.
    It overrides UKF's designed sw2 alternation at those ties.
  - Holding only the first swap (`tie_angle` alone, the reviewer's hypothesis) does not help: 3.62 against 3.42 %
    today. So holding sw2 is necessary. Holding sw2 alone was not run. That the instability is sw2's alternation at
    ties where both FAs are at or below 0.2 is an inference.
  - Neither rule reduces fieldonly's sensitivity (both are slightly higher) or changes rescan agreement.
  - The both-swaps rule is a candidate default for reproducibility under rounding-level input perturbations;
    agreement across backends was not measured. The decision is yours; the defaults are unchanged.
  - Next candidate, not run: hysteresis on sw2's FA threshold, or a tie keyed on |fat1 − 0.2|.

### 3. Denoising before tracking

- **What:** MP-PCA (dipy, in a subprocess) before the field correction, on the series as acquired, opt-in.
  Patch2Self too if it is feasible. If step 2's variant helps, test denoising with it as well as without (not done:
  denoising was not combined with the tie-break or any tracker setting).
- **Measure:**
  - A and the three rescans;
  - a 17-run ensemble whose replicates carry the raw scan's noise into the denoiser;
  - the same criteria as above.
- **Cost:** about 1 hour; a few hundred MB.
- **Outcome (2026-10-17, NOTES "Robustness step 3").**
  - **Order.** Denoising was built before the field correction, not after: the forward series as acquired, its
    own b0s feeding the field from the denoised series, the reversed b0 as acquired. It is opt-in:
    `pipeline.run(denoise=)`, with dipy in a subprocess. The defaults are unchanged (the same SHA-256 through
    `P.run`). The after-correction order exists but was not scored.
  - **MP-PCA does not change rescan agreement at radius 2 or 3 (smoke tests, within rescan only).** At radius 2 its 125-voxel patch barely exceeds the 112 volumes,
    and the noise is correlated in-plane (source unconfirmed). Its sigma, 0.20, is about the integer rounding noise,
    against the bootstrap's residual RMS of 5.75, and it removes an RMS of 0.24. Radius 3 denoises somewhat (sigma
    median 2.08) and keeps only 64.9 % of today's CST streamlines as CST. Smoke tests only (A and the within rescan):
    35.5 % and 35.2 % renamed, against 35.4 %.
  - **Patch2Self** (13 s a scan) cuts rescan renames by 3.5–4.5 points (31.3 / 33.7 / 36.2 % against 35.4 / 38.2 /
    39.7 %). Reweighted to today's tract mix the gain shrinks to 3.6 / 3.9 / 2.7 points, about step 0.6's
    reweighted 2.6–3.6. It holds within every length tertile, so it is not the shift toward shorter streamlines. The
    landmark sites hold at 8 / 8 / 7 against 6 / 6 / 5. Fieldonly's paths more than 5 mm apart fall from 34.6 to
    27.1 %; with the field offset held fixed, a raw-sized wobble on the denoised DWI moves 30.7 % (4 runs), so about
    half is a less sensitive DWI and half a calmer field.
  - **Bias: smaller than the tracker settings' in the tract mix and the Other share (over the 17 runs: total
    variation 0.034, Other 0.538 against 0.531), not elsewhere.**
    - 5 % fewer labeled streamlines, and the median 3.5 mm shorter.
    - The AF: Dice 0.73 with today's (a rescan: 0.81), and ×0.84 in volume.
    - The CST's per-seed labels reshuffle (19 % of today's CST stays CST, the lowest of all), though its presence
      holds about where a rescan puts it (Dice 0.76; a rescan 0.78). The rescans rename A's CST more often on two
      of three.
  - **The sketch seed adds a nondeterministic stage.** Patch2Self version 3's random sketch alone renames 8.9 % and
    moves 17.1 % of paths more than 5 mm (seed 1 against 0, A only, partial): about twice and six times
    Patch2Self's own epsilon (4.07 and 2.99 %, index grouping). Steps 1–2b brought the floor to 0.58 % with both
    opt-ins, which this comparison did not use. The seed is pinned and recorded; only seeds 0 and 1 were compared,
    and they give different tractographies.
  - **Neither of two bootstrap designs makes its confidence honest; today's ensemble is overconfident too.** The raw bootstrap denoised per replicate states 0.794 against
    0.722 / 0.702 / 0.681 observed; the bootstrap about the denoised A states 0.811 against 0.728 / 0.703 / 0.680
    (gaps 0.07–0.13, today 0.06–0.10). AUROC is 0.792 / 0.781 / 0.769 and 0.790 / 0.775 / 0.755, against today's
    0.801 / 0.779 / 0.773. A denoised pipeline needs its own calibration from test-retest data, which a single scan
    cannot supply (as for between-session variation, NOTES 2026-10-13, "Pooling sessions").
  - **Reading.** Agreement with rescans improves by 3.5–4.5 points (2.7–3.9 at today's tract mix) at a modest
    tract-mix bias, with the costs above; whether as noise removal or a systematic change is open. A shared inductive bias raises test-retest
    agreement whether or not it is truth. The phantom (step 5) is needed to judge accuracy, and per-protocol
    calibration for honesty. The default is unchanged; the decision is yours.

### Reordered on 2026-10-17, after step 3

Every gain in rescan agreement so far (Rs 0.1, step 0.6, Patch2Self) changes what the pipeline finds. Only a ground
truth can tell better from merely steadier. And a single scan cannot calibrate its own confidence: under denoising,
neither of two bootstrap designs was honest, and today's ensemble is somewhat overconfident too. So accuracy comes
next, then a calibration step for single scans, and the tracker benchmark moves to last.

### 4. Accuracy on the ISMRM 2015 phantom

- **What:** the ISMRM 2015 Tractography Challenge data (Zenodo 10.5281/zenodo.572345, CC BY 4.0), the version with
  the reversed phase-encoding b0, so that the field correction runs as on real scans. Scored with the 2023 ROI scoring
  (Renauld et al. 2023; `scil_score_ismrm_Renauld2023.sh`, scilpy 1.5.0 in an isolated environment). Details and
  caveats are in `docs/validation-datasets.md`: the gradient x components need flipping, and the readout time is not
  stated.
- **Configurations:** today; the tie-break with hash-sorted grouping; Patch2Self; step 0.6; Rs 0.1.
- **Measure:**
  - the scorer's valid and invalid bundles, overlap, overreach and F1, per bundle and overall;
  - RapidParc's labels against the ground-truth bundles where a counterpart exists (about 8 of the 42 tracts). The
    phantom has no AF, but streamlines along its SLF may be labeled AF; the scorer's verdict, not the label, decides;
  - if cheap, a 17-run ensemble's presence envelopes against the ground-truth bundle masks: calibration against a
    known truth.
- **Caveats to keep in view:** b = 1000 with 32 directions, unlike the 2800 shell tracked on cIIIs04, so the effects
  of Rs and of denoising may not carry over; one simulated brain.

### 5. A calibration card for single scans

- **What:** learn, per protocol, the correction that turns a single scan's ensemble agreement into a calibrated
  probability of what a rescan shows: a session term, as in "Pooling sessions" (NOTES 2026-10-13). Fit it on MASiVar
  test-retest subjects and check it on held-out subjects, for today's pipeline and for any option that becomes a
  default (denoising needs its own card).
- **Measure:** stated against observed on held-out subjects and sessions, for streamlines and for regional claims
  (presence envelopes, the main tract near a point).
- **Output:** a card format the data product carries; without a matching card, the product says "run-to-run only".

### 6. Benchmark against other trackers

- **What:**
  - a probabilistic CSD tracker with tissue-based stopping (DIPY PTT + ACT; MRtrix iFOD2 only if it installs
    easily), labeled with RapidParc, with its Other share as the domain-shift check, since the labelers were trained on
    UKF;
  - TractSeg's bundle masks for CST, AF and UF, as the no-tracking baseline.
- **Measure:** regional measures only, since probabilistic streamlines cannot be matched by seed: presence Dice across
  the four scans, the landmark sites, volumes and run time; and the phantom score from step 4.
- **Cost:** DIPY on the CPU, a few hours. TractSeg on Modal, so no local GPU or disk.

### 7. Synthesis

- What reduces the spread without bias, what is accurate on the phantom, and what that means for the data product.
  Update `docs/uncertainty-ensemble.md` and the earlier proposal; settle which options become defaults (your call).

## Order and gates

1 → 2 → 2b → 3 (done) → 4 → 5 → 6 → 7, with your gate after each step. The order can change at a gate.
