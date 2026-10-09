"""Does the ensemble's confidence hold on a real rescan? (NOTES 2026-10-08, "The ensemble scored on real rescans")

    uv run --group dipy bench/ensemble_coverage.py [--pairs cIIIs01-2.5 cIIIs01-2.1 cIs1] [--flip smooth]

From ensemble_retest.py's records (DATA/ensemble/<subject>/retest/<pair>/); no pipeline runs. Each scan of a pair
has an ensemble on the reference's grid: the reference A with <flip>/r001..r008, and each rescan <tag>_onA with
<flip>/<tag>_r001..r008 (`ensemble_retest.py rescans`). Each ensemble is scored against the other scan's single run,
in both directions (A's ensemble against the rescan, the rescan's against A), and against a held-out replicate of its
own (the bootstrap's centered scoring, for contrast).

Streamlines (seed keys): the stated P(the other scan keeps this streamline's tract) - one run's labeler probability,
the ensemble's mixture, its vote fraction - against what the other scan did: change rate, AUROC (a low stated P
predicting change), calibration (ECE, bins). Also how far apart the two scans' ensembles are, per seed, against how
far apart their single runs are (total variation between tract probability vectors; one run's: its winner, one-hot).

Voxels (each run's winner per voxel, tau 1): an ensemble's P(v, k) - the share of its runs in which tract k wins
voxel v - against the other scan's winner: calibration over every (voxel, tract) where either has a tract, and
envelope coverage. A tract's envelope at level L is its highest-probability voxels holding L of its summed
probability; were P calibrated, the other scan's voxels of that tract would fall inside it at about rate L. One run's
"envelope" is its own tract (P 0 or 1). Stability: per tract, the histogram intersection of the two scans' P maps
(sum of minima over sum of maxima), against their single runs'. Writes results/ensemble_coverage_<flip>.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402
from ensemble_pilot import auroc, calibration  # noqa: E402

OUT = HERE / "results/ensemble_coverage_{}.json"
ONEHOT = np.eye(E.N_TRACTS, dtype=np.float32)[LUT]                 # (1600, 43)
LEVELS = (0.5, 0.7, 0.8, 0.9, 0.95)
K = E.N_TRACTS + 1                                                  # 43 tracts + NONE


class Ensemble:
    """A scan's ensemble keyed on its own run: per seed the mixture's tract probabilities and the vote fraction for
    the run's tract; per voxel P(v, k) over the runs; the run's winner map."""
    def __init__(self, paths):
        ref = E.RunRecord.load(paths[0])
        S, V = E.StreamlineEnsemble(ref), E.VoxelEnsemble(ref.shape)
        for p in paths:
            rec = ref if p == paths[0] else E.RunRecord.load(p)
            S.add(rec); V.add(rec)
            del rec
        self.keys, self.tract = ref.keys, LUT[ref.cluster]
        rows = np.arange(S.n)
        self.p_single = (np.exp(ref.logp.astype(np.float32)) @ ONEHOT)
        self.p_mix = np.exp(S.mixture().numpy()) @ ONEHOT
        self.vfrac = S.votes()[rows, self.tract] / np.maximum(S.present, 1)
        self.P = V.vote.astype(np.float32) / V.runs                   # (44, nvox)
        self.win = V.winner(ref.counts)
        self.runs = V.runs
        self.shape = ref.shape


def winners(path):
    rec = E.RunRecord.load(path, logp=False)
    return rec.keys, LUT[rec.cluster], E.VoxelEnsemble(rec.shape).winner(rec.counts)


def match(keys_a, keys_b):
    order = np.argsort(keys_a)
    pos = np.searchsorted(keys_a[order], keys_b).clip(0, len(keys_a) - 1)
    hit = keys_a[order][pos] == keys_b
    return order[pos[hit]], np.flatnonzero(hit)


def coverage(P, win_other, level):
    """Per tract (named and Other; NONE excluded): the share of the other run's voxels of tract k inside the highest-
    probability region holding `level` of P[k]'s mass; mass-weighted over tracts (weights: the other run's voxels)."""
    num = den = 0.0
    for k in range(E.N_TRACTS):
        w = P[k]
        tot = w.sum()
        theirs = win_other == k
        if tot <= 0 or not theirs.any():
            continue
        order = np.argsort(-w, kind="stable")
        inside = np.zeros(len(w), bool)
        inside[order[: int(np.searchsorted(np.cumsum(w[order]), level * tot)) + 1]] = True
        inside &= w > 0
        num += (inside & theirs).sum()
        den += theirs.sum()
    return round(float(num / den), 4) if den else None


GRID = np.round(np.arange(0.5, 0.9951, 0.005), 3)


def coverage_curve(P, win_other, levels=GRID):
    """coverage() at many levels at once: one sort per tract."""
    num, den = np.zeros(len(levels)), 0.0
    for k in range(E.N_TRACTS):
        w = P[k]
        theirs = win_other == k
        if w.sum() <= 0 or not theirs.any():
            continue
        order = np.argsort(-w, kind="stable")
        ws, hs = w[order], theirs[order]
        cw, ch = np.cumsum(ws), np.cumsum(hs & (ws > 0))
        cut = np.searchsorted(cw, levels * cw[-1])                          # index of the last voxel inside
        num += ch[np.minimum(cut, len(ch) - 1)]
        den += theirs.sum()
    return num / den


def intersection(Pa, Pb, named=True):
    ks = range(E.N_TRACTS - 1) if named else range(E.N_TRACTS)
    mn = sum(float(np.minimum(Pa[k], Pb[k]).sum()) for k in ks)
    mx = sum(float(np.maximum(Pa[k], Pb[k]).sum()) for k in ks)
    return round(mn / mx, 4)


def onehot_P(win):
    P = np.zeros((K, len(win)), np.float32)
    P[win, np.arange(len(win))] = 1
    return P


def score(ens: Ensemble, other_keys, other_tract, other_win):
    """The ensemble's statements against another run (the other scan's, or a held-out replicate)."""
    ia, ib = match(ens.keys, other_keys)
    keep = other_tract[ib] == ens.tract[ia]
    rows = np.arange(len(ia))
    t = ens.tract[ia]
    stated = {"single run: labeler probability": ens.p_single[ia][rows, t], "ensemble: mixture": ens.p_mix[ia][rows, t],
              "ensemble: vote fraction": ens.vfrac[ia]}
    out = {"matched": round(len(ia) / len(ens.keys), 4), "change_rate": round(float(1 - keep.mean()), 4),
           "mean stated P(keep)": {n: round(float(p.mean()), 4) for n, p in stated.items()},
           "auroc predicting change": {n: auroc(-p, ~keep) for n, p in stated.items()},
           "calibration": {n: calibration(p, keep) for n, p in stated.items()}}
    either = (ens.P[:E.N_TRACTS].sum(0) > 0) | (other_win != E.NONE)
    Pv = ens.P[:E.N_TRACTS, either]
    hit = (other_win[either][None] == np.arange(E.N_TRACTS)[:, None])
    keepv = Pv > 0
    out["voxels"] = {"calibration: P(v, k) against the other run's winner (P > 0)": calibration(Pv[keepv], hit[keepv]),
                     "envelope coverage (ensemble)": {f"{L:g}": coverage(ens.P, other_win, L) for L in LEVELS},
                     "single run's tract covers the other's": coverage(onehot_P(ens.win), other_win, 1.0)}
    out["_arrays"] = {"p_mix": stated["ensemble: mixture"], "keep": keep, "Pv": Pv[keepv], "hitv": hit[keepv],
                      "curve": coverage_curve(ens.P, other_win)}
    return out


def recalibrate(rows, groups, flip):
    """Maps fitted on real rescans of the other pairs, scored on the held-out pair: isotonic on the mixture's log-odds
    (ensemble_double.isotonic), a table per voxel probability, and the envelope level that covered the target on the
    other pairs. Beside them, PAT16's double-bootstrap isotonic map (ensemble_double_<flip>.json), carried over as is."""
    from ensemble_double import isotonic
    from ensemble_pilot import logodds
    pairs = sorted(set(groups))
    pat = json.loads((HERE / f"results/ensemble_double_{flip}.json").read_text())["recalibration for rescans"]["mixture, isotonic on logit(p)"]
    pat_knots = (np.array(pat["knots: logit p"]), np.array(pat["knots: P"]))
    out = {}
    for g in pairs:
        tr = [r["_arrays"] for r, gg in zip(rows, groups) if gg != g]
        te = [r["_arrays"] for r, gg in zip(rows, groups) if gg == g]
        z_tr, y_tr = logodds(np.concatenate([a["p_mix"] for a in tr])), np.concatenate([a["keep"] for a in tr]).astype(float)
        z_te, y_te = logodds(np.concatenate([a["p_mix"] for a in te])), np.concatenate([a["keep"] for a in te]).astype(float)
        iso = isotonic(z_tr, y_tr)
        Pv_tr, h_tr = np.concatenate([a["Pv"] for a in tr]), np.concatenate([a["hitv"] for a in tr]).astype(float)
        Pv_te, h_te = np.concatenate([a["Pv"] for a in te]), np.concatenate([a["hitv"] for a in te]).astype(float)
        vals = np.unique(np.concatenate([Pv_tr, Pv_te]))
        table = {float(v): (h_tr[Pv_tr == v].mean() if (Pv_tr == v).any() else float(v)) for v in vals}
        curve_tr = np.mean([a["curve"] for a in tr], 0)
        curve_te = np.mean([a["curve"] for a in te], 0)
        env = {}
        for L in (0.5, 0.8, 0.9, 0.95):
            ok = np.flatnonzero(curve_tr >= L)
            use = GRID[ok[0]] if len(ok) else GRID[-1]
            env[f"{L:g}"] = {"stated level used": float(use), "reached on the other pairs": bool(len(ok)),
                             "held-out coverage": round(float(curve_te[np.flatnonzero(GRID == use)[0]]), 4),
                             "held-out coverage at the stated level": round(float(curve_te[np.flatnonzero(GRID == round(L, 3))[0]]), 4)}
        out[g] = {"streamlines: ECE as stated / isotonic (other pairs) / PAT16's map": [
                      calibration(1 / (1 + np.exp(-z_te)), y_te)["ece"], calibration(np.interp(z_te, *iso), y_te)["ece"],
                      calibration(np.interp(z_te, *pat_knots), y_te)["ece"]],
                  "streamlines: mean stated P(keep) as stated / isotonic / PAT16's map, against observed": [
                      round(float((1 / (1 + np.exp(-z_te))).mean()), 4), round(float(np.interp(z_te, *iso).mean()), 4),
                      round(float(np.interp(z_te, *pat_knots).mean()), 4), round(float(y_te.mean()), 4)],
                  "voxels: ECE as stated / table (other pairs)": [calibration(Pv_te, h_te)["ece"],
                                                                 calibration(np.array([table[float(v)] for v in Pv_te]), h_te)["ece"]],
                  "envelopes": env}
    out["coverage curve, all rescan scorings (level: coverage)"] = {f"{L:g}": round(float(c), 4) for L, c in
                                                                  zip(GRID, np.mean([r["_arrays"]["curve"] for r in rows], 0)) if round(L * 100) % 5 == 0}
    return out


def main(args):
    res = {"flip": args.flip, "pairs": {}}
    pooled = {"rescan": [], "held-out replicate": []}
    groups = []
    for pair in args.pairs:
        d = DATA / "ensemble" / PAIRS[pair][0] / "retest" / pair
        rescans = PAIRS[pair][2]
        reps = lambda prefix, n=8: [d / args.flip / f"{prefix}r{r:03d}.npz" for r in range(1, n + 1)]  # noqa: E731
        A = Ensemble([d / "A.npz"] + reps(""))
        A7 = Ensemble([d / "A.npz"] + reps("", 7))
        row = {"runs per ensemble": A.runs, "rescans": {}}
        row["A, held-out replicate r008 (centered)"] = score(A7, *winners(d / args.flip / "r008.npz"))
        pooled["held-out replicate"].append(row["A, held-out replicate r008 (centered)"])
        for tag in rescans:
            paths = [d / f"{tag}_onA.npz"] + reps(f"{tag}_")
            if not all(p.exists() for p in paths):
                continue
            B = Ensemble(paths)
            ka, ta, wa = winners(d / "A.npz")
            kb, tb, wb = winners(d / f"{tag}_onA.npz")
            r = {"A's ensemble, scored on the rescan": score(A, kb, tb, wb),
                 "the rescan's ensemble, scored on A": score(B, ka, ta, wa)}
            ia, ib = match(A.keys, B.keys)
            r["between the two scans"] = {
                "seeds: total variation, single runs (winners)": round(float((A.tract[ia] != B.tract[ib]).mean()), 4),
                "seeds: total variation, ensemble mixtures": round(float(0.5 * np.abs(A.p_mix[ia] - B.p_mix[ib]).sum(1).mean()), 4),
                "tract maps: intersection, single runs (named tracts)": intersection(onehot_P(A.win), onehot_P(B.win)),
                "tract maps: intersection, ensembles (named tracts)": intersection(A.P, B.P)}
            row["rescans"][tag] = r
            pooled["rescan"] += [r["A's ensemble, scored on the rescan"], r["the rescan's ensemble, scored on A"]]
            groups += [pair, pair]
            print(pair, tag, json.dumps({k: (v.get("change_rate"), v.get("auroc predicting change"), v["voxels"]["envelope coverage (ensemble)"])
                                         if "voxels" in v else v for k, v in r.items()}), flush=True)
            del B
        res["pairs"][pair] = row
    summ = {}
    for name, rows in pooled.items():
        if not rows:
            continue
        summ[name] = {"scorings": len(rows),
                      "change_rate": round(float(np.mean([r["change_rate"] for r in rows])), 4),
                      "mean stated P(keep)": {n: round(float(np.mean([r["mean stated P(keep)"][n] for r in rows])), 4) for n in rows[0]["mean stated P(keep)"]},
                      "auroc": {n: round(float(np.mean([r["auroc predicting change"][n] for r in rows])), 4) for n in rows[0]["auroc predicting change"]},
                      "ece": {n: round(float(np.mean([r["calibration"][n]["ece"] for r in rows])), 4) for n in rows[0]["calibration"]},
                      "voxel ece": round(float(np.mean([r["voxels"]["calibration: P(v, k) against the other run's winner (P > 0)"]["ece"] for r in rows])), 4),
                      "envelope coverage": {f"{L:g}": round(float(np.mean([r["voxels"]["envelope coverage (ensemble)"][f"{L:g}"] for r in rows])), 4) for L in LEVELS},
                      "single run's tract covers the other's": round(float(np.mean([r["voxels"]["single run's tract covers the other's"] for r in rows])), 4)}
    res["summary"] = summ
    if len(set(groups)) > 1:
        res["recalibration on real rescans (leave one pair out)"] = recalibrate(pooled["rescan"], np.array(groups), args.flip)
    for rows in pooled.values():
        for r in rows:
            r.pop("_arrays", None)
    Path(str(OUT).format(args.flip)).write_text(json.dumps(res, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS))
    ap.add_argument("--flip", default="smooth")
    main(ap.parse_args())
