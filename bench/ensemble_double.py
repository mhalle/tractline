"""The ensemble's confidence against rescans, by a double bootstrap (NOTES 2026-10-08, "the bootstrap is centred on it").

    uv run --group dipy --group rankfield bench/ensemble_double.py run [--subject PAT16]
    uv run --group dipy --group rankfield bench/ensemble_double.py analyze [--subject PAT16]

A bootstrap's replicates spread around the scan's own estimate, so the pilot's held-out runs, scored against the scan,
measure the spread around it and not its error: a held-out run changes 33 % of the scan's labels, a rescan-like draw
(replicate against replicate) 38 %. The double bootstrap gives the ensemble a scan whose rescans we have:

- pseudo-scans: first-level replicates (ensemble_pilot's added/r011..r014, rebuilt from their seeds and rerun as
  s000 - a check that the rebuild is the replicate);
- each pseudo-scan bootstrapped again (ShellWildBootstrap of the replicate itself, b0s "added", the reference's mask),
  `level2` runs: 1..build make its ensemble (with s000, as the pilot's), the rest are held out (the pilot's scoring);
- its rescans: the other first-level replicates r001..r016 - draws about the same centre (the real scan's estimate)
  as a pseudo-scan is, as real rescans are draws about the truth.

analyze: per streamline of each pseudo-scan, the single run's TRX probability, the ensemble mixture and the vote
fraction against (a) its held-out second-level runs (centred, as the pilot) and (b) its rescans; per voxel, the
consensus against both. Then a recalibration for rescans fitted on (b) - Platt scaling of the mixture's log-odds and a
table per vote count - scored leaving one pseudo-scan out, and applied to the real scan's ensemble (added/r000 +
r001..r010). Writes results/ensemble_double.json.
"""
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, OTHER

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_pilot import DEVICE, _winners_only, auroc, calibration, tract_mass, logodds  # noqa: E402
from ensemble_exchange import reference_mask  # noqa: E402

OUT = HERE / "results/ensemble_double.json"
PSEUDO = (11, 12, 13, 14)
FIRST = range(1, 17)


def level1(s, mask):
    from _bootstrap import ShellWildBootstrap
    return ShellWildBootstrap(s.dwi, s.bval, s.bvec, extra_b0s=s.b0s[..., len(s.b0_index):], b0_noise="added", mask=mask)


def as_subject(s, dwi, pa):
    return replace(s, dwi=dwi, b0s=np.concatenate([dwi[..., s.b0_index].astype(np.float64), pa], -1))


def run(args):
    from _ds001226 import load
    from _bootstrap import ShellWildBootstrap
    s = load(args.subject)
    labeler = P.default_labeler(DEVICE)
    grid = (s.affine, s.dwi.shape[:3])
    mask = reference_mask(s)
    wb1 = level1(s, mask)
    first = DATA / "ensemble" / args.subject / "added"
    for k in PSEUDO:
        d = DATA / "ensemble" / args.subject / "double" / f"r{k:03d}"
        d.mkdir(parents=True, exist_ok=True)
        log_path = d / "runs.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        dwi, pa = wb1.replicate(k)
        sk = as_subject(s, dwi, pa)

        def one(name, subj, draw):
            if (d / f"{name}.npz").exists():
                return
            timer, t0 = P.Timer(), time.time()
            with P.exact_float32(DEVICE):
                corr = P.correct(subj, timer, DEVICE)
                tg = P.track(subj, corr.dwi, timer, device=DEVICE)
                labels = labeler(tg.fibers, draws=(draw,), logp=True)
            rec = E.RunRecord.of(tg, labels, *grid)
            rec.save(d / f"{name}.npz")
            log[name] = {"draw": draw, "labeled": int(labels.keep.sum()), "other": round(float((labels.tract == OTHER).mean()), 4),
                         "residual_left": corr.residual_left,
                         "field_hz_p99_abs": round(float(np.quantile(np.abs(corr.field_hz), 0.99)), 2),
                         "total_s": round(time.time() - t0, 1)}
            if name == "s000":                          # the rebuild is the first-level replicate
                ref = E.RunRecord.load(first / f"r{k:03d}.npz", logp=False)
                log[name]["same_as_first_level"] = bool(np.array_equal(ref.keys, rec.keys) and np.array_equal(ref.cluster, rec.cluster))
            log_path.write_text(json.dumps(log, indent=1))
            print(f"r{k:03d}", name, json.dumps(log[name]), flush=True)
            P.release_memory(DEVICE)

        one("s000", sk, k)                                                  # the pilot's draw for r0k
        wb2 = ShellWildBootstrap(dwi, s.bval, s.bvec, extra_b0s=pa.reshape(*s.dwi.shape[:3], -1), b0_noise="added", mask=mask)
        log["bootstrap"] = wb2.noise_check(mask)
        for j in range(1, args.level2 + 1):
            name = f"s{j:03d}"
            if not (d / f"{name}.npz").exists():
                one(name, as_subject(s, *wb2.replicate(1000 * k + j)), 1000 * k + j)


# ----------------------------------------------------------------------------------------------- analysis

def platt(z, y, iters=50):
    """Logistic regression of y on (1, z) by Newton's method -> (a, b): P = sigmoid(a + b z)."""
    X = np.stack([np.ones_like(z), z], 1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        w += step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def isotonic(z, y, bins=200):
    """A non-decreasing map z -> P(y): pool-adjacent-violators over `bins` quantile bins of z -> knots (x, p), applied by
    interpolation (flat beyond the ends)."""
    edges = np.unique(np.quantile(z, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(edges) - 2)
    n = np.bincount(b, minlength=len(edges) - 1).astype(float)
    x = np.bincount(b, z, minlength=len(n)) / np.maximum(n, 1)
    m = np.bincount(b, y, minlength=len(n)) / np.maximum(n, 1)
    keep = n > 0
    x, m, n = x[keep], m[keep], n[keep]
    blocks = []                                                           # [mean, weight, x-weighted sum, count]
    for xi, mi, ni in zip(x, m, n):
        blocks.append([mi, ni, xi * ni])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, n2, s2 = blocks.pop()
            blocks[-1] = [(blocks[-1][0] * blocks[-1][1] + m2 * n2) / (blocks[-1][1] + n2), blocks[-1][1] + n2, blocks[-1][2] + s2]
    return np.array([bl[2] / bl[1] for bl in blocks]), np.array([bl[0] for bl in blocks])


def held_out(fit, apply, groups, *cols):
    """Fit on all groups but one, score on that one: Brier and ECE per held-out group, and their mean. (Pooling the
    held-out predictions instead hides the error: with groups of equal size, a per-value table fitted on the others
    averages back to the pooled rate exactly.)"""
    out = {}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        model = fit(*(c[tr] for c in cols))
        cal = calibration(apply(model, *(c[te] for c in cols[:-1])), cols[-1][te])
        out[f"r{g:03d}"] = {"brier": cal["brier"], "ece": cal["ece"]}
    out["mean"] = {k: round(float(np.mean([v[k] for v in out.values()])), 4) for k in ("brier", "ece")}
    return out


def vote_table(v, y, R):
    """P(y | v votes of R), per vote count (the overall rate where a count is absent)."""
    return np.array([y[v == c].mean() if (v == c).any() else y.mean() for c in range(R + 1)])


def per_pseudo(root, k, build):
    """One pseudo-scan's streamline predictors and its pairs against held-out second-level runs and rescans, and its
    voxel consensus scored the same way."""
    d = root / "double" / f"r{k:03d}"
    own = E.RunRecord.load(d / "s000.npz")
    S, V = E.StreamlineEnsemble(own), E.VoxelEnsemble(own.shape)
    for name in ["s000"] + [f"s{j:03d}" for j in range(1, build + 1)]:
        rec = own if name == "s000" else E.RunRecord.load(d / f"{name}.npz")
        S.add(rec); V.add(rec)
    t = LUT[own.cluster]
    rows = np.arange(S.n)
    p_single = tract_mass(own.logp)[rows, t]
    p_mix = tract_mass(S.mixture().numpy())[rows, t]
    votes = S.votes()[rows, t].astype(np.int64)
    pred = {"single": p_single, "mixture": p_mix, "votes": votes, "present": S.present.copy()}
    cons = V.winner(V.sum / V.runs)
    nvox = V.sum.shape[1]
    cvote = V.vote.astype(np.int64)[cons, np.arange(nvox)]
    inside = V.sum.max(0) > 0
    own_map = V.winner(own.counts)

    def score(names, folder):
        H = E.StreamlineEnsemble(_winners_only(own))
        vox = []
        for n in names:
            rec = E.RunRecord.load(folder / f"{n}.npz", logp=False)
            H.add(_winners_only(rec))
            w = V.winner(rec.counts)
            vox.append(((w == cons)[inside], (w == own_map)[inside], cvote[inside]))
        W = np.stack(H.winners)
        have = W >= 0
        keep = (LUT[np.maximum(W, 0)] == t[None])[have]
        return {"row": np.broadcast_to(rows, W.shape)[have], "keep": keep,
                "vox_cons": np.concatenate([v[0] for v in vox]), "vox_own": np.concatenate([v[1] for v in vox]),
                "vox_votes": np.concatenate([v[2] for v in vox])}

    held = [p.stem for p in sorted(d.glob("s*.npz")) if int(p.stem[1:]) > build]
    return pred, S.runs, {"centred (held-out second-level runs)": score(held, d),
                          "rescans (other first-level replicates)": score([f"r{r:03d}" for r in FIRST if r != k], root / "added")}


def analyze(args):
    root = DATA / "ensemble" / args.subject
    res = {"subject": args.subject, "pseudo_scans": [f"r{k:03d}" for k in PSEUDO], "build": args.build,
           "runs_logs": {f"r{k:03d}": json.loads((root / "double" / f"r{k:03d}" / "runs.json").read_text()) for k in PSEUDO}}
    data = {k: per_pseudo(root, k, args.build) for k in PSEUDO}
    R = data[PSEUDO[0]][1]
    targets = list(data[PSEUDO[0]][2])
    res["scoring"] = {}
    pooled = {}
    for tgt in targets:
        cols = {key: [] for key in ("single", "mixture", "vfrac", "votes", "keep", "pseudo", "vox_cons", "vox_own", "vox_votes", "vox_pseudo")}
        for k in PSEUDO:
            pred, _, sc = data[k]
            r = sc[tgt]["row"]
            cols["single"].append(pred["single"][r]); cols["mixture"].append(pred["mixture"][r])
            cols["votes"].append(pred["votes"][r]); cols["vfrac"].append(pred["votes"][r] / np.maximum(pred["present"][r], 1))
            cols["keep"].append(sc[tgt]["keep"]); cols["pseudo"].append(np.full(len(r), k))
            for key in ("vox_cons", "vox_own", "vox_votes"):
                cols[key].append(sc[tgt][key])
            cols["vox_pseudo"].append(np.full(len(sc[tgt]["vox_cons"]), k))
        c = {key: np.concatenate(v) for key, v in cols.items()}
        pooled[tgt] = c
        keep = c["keep"]
        res["scoring"][tgt] = {
            "pairs": int(len(keep)), "change_rate": round(float(1 - keep.mean()), 4),
            "change_rate per pseudo-scan": {f"r{k:03d}": round(float(1 - keep[c["pseudo"] == k].mean()), 4) for k in PSEUDO},
            "auroc predicting change": {n: auroc(-c[n], ~keep) for n in ("single", "mixture", "vfrac")},
            "calibration: P(keeps the pseudo-scan's tract)": {n: calibration(c[n], keep) for n in ("single", "mixture", "vfrac")},
            "voxels": {"agreement with consensus": round(float(c["vox_cons"].mean()), 4),
                       "agreement with the pseudo-scan's own map": round(float(c["vox_own"].mean()), 4),
                       "calibration: P(winner = consensus)": calibration(c["vox_votes"] / R, c["vox_cons"])}}
        print(tgt, json.dumps({k: res["scoring"][tgt][k] for k in ("change_rate", "auroc predicting change")}), flush=True)

    # recalibration for rescans, scored leaving one pseudo-scan out
    c = pooled["rescans (other first-level replicates)"]
    z, keep, v, g = logodds(c["mixture"]), c["keep"].astype(float), c["votes"], c["pseudo"]
    v_vox, keep_vox = c["vox_votes"], c["vox_cons"].astype(float)
    as_is = {"mixture": held_out(lambda *a: None, lambda m, z: sigmoid(z), g, z, keep),
             "vote fraction": held_out(lambda *a: None, lambda m, f: f, g, c["vfrac"], keep),
             "voxel vote fraction": held_out(lambda *a: None, lambda m, f: f / R, c["vox_pseudo"], v_vox, keep_vox)}
    platt_cv = held_out(platt, lambda ab, z: sigmoid(ab[0] + ab[1] * z), g, z, keep)
    iso_cv = held_out(isotonic, lambda k, z: np.interp(z, *k), g, z, keep)
    vote_cv = held_out(lambda v, y: vote_table(v, y, R), lambda t, v: t[v], g, v, keep)
    vox_cv = held_out(lambda v, y: vote_table(v, y, R), lambda t, v: t[v], c["vox_pseudo"], v_vox, keep_vox)
    a, b = platt(z, keep)
    iso = isotonic(z, keep)
    table, table_vox = vote_table(v, keep, R), vote_table(v_vox, keep_vox, R)
    nan_none = lambda t: [None if np.isnan(x) else round(float(x), 4) for x in t]  # noqa: E731
    res["recalibration for rescans"] = {
        "held-out pseudo-scan (Brier, ECE), as stated": as_is,
        "mixture, Platt: P = sigmoid(a + b logit(p))": {"a": round(a, 4), "b": round(b, 4), "held-out": platt_cv},
        "mixture, isotonic on logit(p)": {"knots: logit p": [round(float(x), 3) for x in iso[0]],
                                          "knots: P": [round(float(x), 4) for x in iso[1]], "held-out": iso_cv,
                                          "in-sample calibration": calibration(np.interp(z, *iso), keep)},
        "votes: P(keep | votes of R)": {"R": R, "table": nan_none(table), "held-out": vote_cv},
        "voxels: P(winner = consensus | consensus votes of R)": {"table": nan_none(table_vox), "held-out": vox_cv}}
    print(json.dumps({k: (v.get("held-out", v) if isinstance(v, dict) else v) for k, v in res["recalibration for rescans"].items()},
                     default=str)[:3000], flush=True)

    # applied to the real scan's ensemble (the pilot's: added/r000 + r001..r010)
    first = root / "added"
    ref = E.RunRecord.load(first / "r000.npz")
    S = E.StreamlineEnsemble(ref)
    for n in ["r000"] + [f"r{r:03d}" for r in range(1, args.build + 1)]:
        S.add(ref if n == "r000" else E.RunRecord.load(first / f"{n}.npz"))
    t = LUT[ref.cluster]
    rows = np.arange(S.n)
    p_mix = tract_mass(S.mixture().numpy())[rows, t]
    p_cal = np.interp(logodds(p_mix), *iso)
    p_platt = sigmoid(a + b * logodds(p_mix))
    p_vote = table[S.votes()[rows, t].astype(np.int64)]
    q = lambda p: {f">= {x}": round(float((p >= x).mean()), 4) for x in (0.5, 0.9, 0.99)}  # noqa: E731
    res["the real scan's ensemble"] = {
        "streamlines": int(S.n),
        "mean stated P(rescan keeps the tract)": {"mixture as is": round(float(p_mix.mean()), 4),
                                                  "mixture recalibrated (isotonic)": round(float(p_cal.mean()), 4),
                                                  "mixture recalibrated (Platt)": round(float(p_platt.mean()), 4),
                                                  "votes recalibrated": round(float(np.nanmean(p_vote)), 4)},
        "replicate vs replicate keep rate (NOTES 2026-10-08)": 0.62,
        "share of streamlines stated at": {"mixture as is": q(p_mix), "mixture recalibrated (isotonic)": q(p_cal),
                                   "votes recalibrated": q(p_vote)}}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["the real scan's ensemble"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--subject", default="PAT16")
    ap.add_argument("--level2", type=int, default=12)
    ap.add_argument("--build", type=int, default=10)
    args = ap.parse_args()
    run(args) if args.what == "run" else analyze(args)
