"""Does pooling several sessions' ensembles predict a new scan better than one scan's ensemble? (NOTES 2026-10-13,
"Pooling sessions").

    uv run --group rankfield bench/ensemble_pooled.py [--pair cIIIs04-2.1]

cIIIs04-2.1 has four scans on A's grid (A; the same-session, session 3 and session 5 rescans), each with a 17-run
ensemble: its own run (A.npz or <tag>_onA.npz) and 16 replicates (smooth/r###, smooth/<tag>_r###). Leaving each scan
out in turn, its own run (the scan as the pipeline runs it) is the target, predicted by
  - single: each other scan's 17-run ensemble alone (the three averaged);
  - pooled: the three other ensembles together, 51 runs, each scan weighted equally.
Voxels (presence: a tract with at least MIN_COUNT streamlines): for each named tract, the share of the target's voxels
inside the widest outline (any run) and inside the envelopes at stated levels (the highest-share voxels holding that
share of the tract's summed share, as ensemble_coverage.py); pooled over the 42 named tracts (weighted by the target's
voxels) and for the focus tracts (CST, AF, UF). Streamlines (seeds labeled in all 68 runs): the predicted share of the
most frequent tract against how often the target gives it. Writes results/ensemble_pooled.json.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import TRACT_NAMES, LUT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402

RUNS = 17
MIN_COUNT = 2
NAMED = E.N_TRACTS - 1
FOCUS = ("CST", "AF", "UF")
LEVELS = (0.5, 0.8, 0.9, 0.925, 0.95)


def thresholds(q, R, levels):
    """q: one tract's integer votes (0..R) per voxel -> for each level, the lowest vote count kept by its envelope."""
    counts = np.bincount(q, minlength=R + 1).astype(float)
    mass = counts * np.arange(R + 1)
    mass[0] = 0
    cum = np.cumsum(mass[::-1])[::-1]                                   # mass at or above each vote count
    total = cum[1] if R >= 1 else 0
    out = []
    for L in levels:
        ok = np.flatnonzero(cum >= L * total)
        out.append(int(ok.max()) if len(ok) else 1)
    return out


def voxel_scores(votes, R, target):
    """votes: (NAMED, nvox) integer votes out of R; target: (NAMED, nvox) bool -> per tract [voxels, widest, *levels]."""
    rows = {}
    for k in range(NAMED):
        t = target[k]
        n = int(t.sum())
        if not n:
            continue
        q = votes[k]
        th = thresholds(q, R, LEVELS)
        rows[TRACT_NAMES[k]] = [n, float((t & (q > 0)).sum() / n)] + [float((t & (q >= v)).sum() / n) for v in th]
    return rows


def summarize(rows, names=None):
    names = names or list(rows)
    n = np.array([rows[k][0] for k in names if k in rows], float)
    v = np.array([rows[k][1:] for k in names if k in rows])
    w = (v * n[:, None]).sum(0) / n.sum()
    return dict(zip(["widest"] + [f"envelope {L}" for L in LEVELS], [round(float(x), 3) for x in w]))


def main(args):
    t0 = time.time()
    subject, _, rescans, _, _ = PAIRS[args.pair]
    d = DATA / "ensemble" / subject / "retest" / args.pair
    scans = {"A": [d / "A.npz"] + [d / "smooth" / f"r{r:03d}.npz" for r in range(1, RUNS)]}
    for tag in rescans:
        scans[tag] = [d / f"{tag}_onA.npz"] + [d / "smooth" / f"{tag}_r{r:03d}.npz" for r in range(1, RUNS)]
    votes, target, keys, tracts = {}, {}, {}, {}
    for s, paths in scans.items():
        v = None
        keys[s], tracts[s] = [], []
        for i, p in enumerate(paths):
            r = E.RunRecord.load(p, logp=False)
            pres = r.counts[:NAMED] >= MIN_COUNT
            v = pres.astype(np.uint8) if v is None else v + pres
            if i == 0:
                target[s] = pres
            keys[s].append(r.keys); tracts[s].append(LUT[r.cluster])
            del r
        votes[s] = v
        print(f"{s} loaded {time.time() - t0:.0f}s", flush=True)

    # seeds labeled in all 68 runs: per scan, the tract each of its runs gave (m, 17)
    common = None
    for s in scans:
        for k in keys[s]:
            common = k if common is None else np.intersect1d(common, k)
    T = {}
    for s in scans:
        T[s] = np.stack([t[np.searchsorted(k, common)] for k, t in zip(keys[s], tracts[s])], 1)
        for k in keys[s]:
            assert (k[np.searchsorted(k, common)] == common).all()

    def frac(cols):                                                       # (m, 43) vote fractions over a set of scans
        f = np.zeros((len(common), E.N_TRACTS))
        for s in cols:
            c = np.zeros((len(common), E.N_TRACTS))
            np.add.at(c, (np.repeat(np.arange(len(common)), RUNS), T[s].ravel()), 1)
            f += c / RUNS
        return f / len(cols)

    res = {"pair": args.pair, "seeds labeled in all 68 runs": int(len(common)), "levels": LEVELS, "held out": {}}
    for h in scans:
        others = [s for s in scans if s != h]
        tgt_tract = T[h][:, 0]
        row = {}
        # voxels
        single = [voxel_scores(votes[s], RUNS, target[h]) for s in others]
        pooled = voxel_scores(sum(votes[s].astype(np.int16) for s in others), RUNS * len(others), target[h])
        row["voxels, all named tracts: single (mean of 3)"] = {k: round(float(np.mean([summarize(x)[k] for x in single])), 3) for k in summarize(single[0])}
        row["voxels, all named tracts: pooled"] = summarize(pooled)
        for name in FOCUS:
            row[f"voxels, {name}: single (mean of 3)"] = {k: round(float(np.mean([summarize(x, [name])[k] for x in single])), 3) for k in summarize(single[0], [name])}
            row[f"voxels, {name}: pooled"] = summarize(pooled, [name])
        # streamlines: the predicted share of the mode against the target giving it
        for label, cols in [("single (mean of 3)", [[s] for s in others]), ("pooled", [others])]:
            stated, observed = [], []
            for c in cols:
                f = frac(c)
                mode = f.argmax(1)
                stated.append(f.max(1).mean()); observed.append((tgt_tract == mode).mean())
            row[f"streamlines: stated / observed, {label}"] = [round(float(np.mean(stated)), 3), round(float(np.mean(observed)), 3)]
        res["held out"][h] = row
        print(h, json.dumps({k: v for k, v in row.items() if "all named" in k or "streamlines" in k}), flush=True)
    (HERE / "results/ensemble_pooled.json").write_text(json.dumps(res, indent=1))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="cIIIs04-2.1", choices=sorted(PAIRS))
    main(ap.parse_args())
