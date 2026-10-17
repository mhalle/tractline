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

### 3. Denoising before tracking

- **What:** MP-PCA (dipy, in a subprocess) after the field correction and before tracking, opt-in. Patch2Self too if
  it is feasible. If step 2's variant helps, test denoising with it as well as without.
- **Measure:**
  - A and the three rescans;
  - a 9-run ensemble whose replicates carry the raw scan's noise into the denoiser;
  - the same criteria as above.
- **Cost:** about 1 hour; a few hundred MB.

### 4. Benchmark against other trackers

- **What:**
  - a probabilistic CSD tracker with tissue-based stopping (DIPY PTT + ACT; MRtrix iFOD2 only if it installs
    easily), labeled with RapidParc, with its Other share as the domain-shift check, since the labelers were trained on
    UKF;
  - TractSeg's bundle masks for CST, AF and UF, as the no-tracking baseline.
- **Measure:** regional measures only, since probabilistic streamlines cannot be matched by seed: presence Dice across
  the four scans, the landmark sites, volumes and run time.
- **Cost:** DIPY on the CPU, a few hours. TractSeg on Modal, so no local GPU or disk; it could run during an earlier
  step's queue, since it shares nothing local.

### 5. Synthesis, then accuracy

- **Synthesis:** what reduces the spread without bias, and what that means for the data product. Update
  `docs/uncertainty-ensemble.md` and the earlier proposal; settle which options become defaults (your call).
- **Accuracy:** a phantom with known paths (DiSCo, or the ISMRM 2015 Tractometer data) to check that the winners are
  not reproducibly wrong. This is planned after step 5, as its own step.

## Order and gates

1 → 2 → 3 → 4 → 5, with your gate after each step. The order can change at a gate:
- if step 2 shows stopping dominates, tissue-based stopping moves up;
- if steps 2–3 give large gains, step 4 may matter less.

Steps 1–4 take about a day of machine time.
