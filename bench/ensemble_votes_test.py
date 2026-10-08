"""Are votes needed beside the ensemble's mixture? Four tests on the ensemble pilot's records (no new runs).

    uv run --group rankfield bench/ensemble_votes_test.py [--b0-noise added] [--subject PAT16] [--boot 200]

The records are ensemble_pilot.py's, under DATA/ensemble/<subject>/<b0 noise>/: r000 the reference (the scan as
acquired, draw 0), r001-r016 replicates with every stage resampled. Ensemble: r000-r010 (R = 11). Held out:
r011-r016. A run's label under a grouping is the group of its winning cluster. The target is a held-out run
giving the reference's streamline another group than the reference's own.

1. Regrouping after labeling. For four groupings - the tracts as labeled; SLF I-III and CC 1-7 merged; outliers
   folded into their cluster's tract (cluster 800 + i is cluster i's outlier twin); clusters with outliers folded
   (800) - which confidence in the reference's group predicts change (AUROC):
     single  the reference run's own group probability (TRX's tract_probability at the tract grouping)
     field6  the mixture as the web export encodes it (rankfield depth 6, keep clip), its group mass decoded
     margin6 the same field's group margin (rankfield.decode_groups: the union's own margin)
     full    the mixture's group mass, nothing cut
     votes   the runs whose winner fell in the group, over the runs that labeled the seed (exact)
     votes6  the same from vote planes aligned with field6's ranks (short by the vote tail)
   with 95 % intervals from a block bootstrap (blocks: 10-voxel cubes of seed voxels, the streamlines' and runs'
   correlations kept inside a block), and the paired differences votes - field6 and votes - full.
2. Beyond the mixture. Within deciles of full, does the vote fraction or the votes' entropy across groups still
   separate change (conditional AUROC, deciles pooled by size)? And a logistic model, 5-fold by block: log-loss of
   logit(full) alone against logit(full) + vote fraction + vote entropy.
3. Runs. The ensemble rebuilt from r000..r(R-1), R = 2, 4, 6, 8, 11, against the same held-out runs: AUROC and ECE
   of full, field6 and votes, at the tract and the outlier-folded groupings.
4. Voxels. For merges of tracts (CC 1-7; SLF I-III; the superficial Sup-*), how often the summed votes of a merge
   differ from votes taken after merging the densities, and whether it matters: the merged group's predicted vote
   fraction both ways, against held-out runs' merged-first winners (Brier, ECE).

Pass mark, fixed before running: votes are needed for a grouping when votes - field6 AUROC exceeds 0.01 with its
interval above 0; deeper ranks would do instead when votes - full is within 0.01 there. Writes
results/ensemble_votes_test_<b0 noise>.json.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
import rankfield as rf
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES

HERE = Path(__file__).resolve().parent
ENSEMBLE = [f"r{k:03d}" for k in range(11)]
HELD = [f"r{k:03d}" for k in range(11, 17)]
BLOCK = 10                       # voxels a block's side


def groupings():
    slf = [TRACT_NAMES.index(t) for t in ("SLF-I", "SLF-II", "SLF-III")]
    cc = [TRACT_NAMES.index(f"CC{k}") for k in range(1, 8)]
    remap = np.arange(E.N_TRACTS); remap[slf] = slf[0]; remap[cc] = cc[0]
    _, remap = np.unique(remap, return_inverse=True)
    c = np.arange(1600)
    return {"tract": LUT, "merged (SLF, CC)": remap[LUT], "tract, outliers folded": LUT[c % 800],
            "cluster, outliers folded": c % 800}


# ------------------------------------------------------------------------------------------- statistics

class AUC:
    """Weighted AUROC (ties half) of one score for one target, fast under many weightings (the bootstrap)."""
    def __init__(self, score, positive):
        u, inv = np.unique(np.asarray(score, float), return_inverse=True)
        self.inv, self.pos, self.k = inv, np.asarray(positive, bool), len(u)

    def __call__(self, w):
        wp = np.bincount(self.inv, weights=w * self.pos, minlength=self.k)
        wn = np.bincount(self.inv, weights=w * ~self.pos, minlength=self.k)
        below = np.cumsum(wn) - wn
        P, N = wp.sum(), wn.sum()
        return float((wp * (below + 0.5 * wn)).sum() / (P * N)) if P > 0 and N > 0 else np.nan


def boot_weights(blocks, B, seed=0):
    """(B, pairs) multiplicities: blocks resampled with replacement, each pair weighted by its block's count."""
    u, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv] for _ in range(B)]).astype(float)


def summarize(point, boots):
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return [round(point, 4), round(float(lo), 4), round(float(hi), 4)]


def ece(p, hit, bins=10):
    p, hit = np.asarray(p, float), np.asarray(hit, float)
    e = np.minimum((p * bins).astype(int), bins - 1)
    return round(float(sum((e == b).mean() * abs(p[e == b].mean() - hit[e == b].mean()) for b in range(bins) if (e == b).any())), 4)


def logloss_cv(X, y, folds, iters=50):
    """Mean held-out log-loss of a logistic model (Newton's method, a small ridge), folds given per row."""
    X = np.c_[np.ones(len(X)), X]
    out = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        b = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X[tr] @ b))
            g = X[tr].T @ (p - y[tr]) + 1e-6 * b
            H = (X[tr] * (p * (1 - p))[:, None]).T @ X[tr] + 1e-6 * np.eye(len(b))
            step = np.linalg.solve(H, g)
            b -= step
            if np.abs(step).max() < 1e-8:
                break
        p = np.clip(1 / (1 + np.exp(-X[te] @ b)), 1e-9, 1 - 1e-9)
        out.append(-(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).mean() * te.sum())
    return round(float(sum(out) / len(y)), 5)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------- the ensemble

def group_mass(logp, lut, G):
    p = np.exp(np.asarray(logp, np.float64))
    out = np.zeros((len(p), G))
    np.add.at(out.T, lut, p.T)
    return out


def winners_matrix(ref, names, d):
    """(R, n) each run's winning cluster per reference streamline (-1 where the run did not label the seed)."""
    S = E.StreamlineEnsemble(ref)
    out = []
    for n in names:
        r = E.RunRecord.load(d / f"{n}.npz", logp=False)
        ri, qi = S.match(r)
        w = np.full(S.n, -1, np.int64); w[ri] = r.cluster[qi]
        out.append(w)
    return np.stack(out)


def mixture(ref, names, d, snapshots=()):
    """The mixture over runs `names` (log mean probability over the runs that labeled each seed), and at each R in
    snapshots the mixture of the first R."""
    S = E.StreamlineEnsemble(ref)
    snaps = {}
    for k, n in enumerate(names, 1):
        S.add(E.RunRecord.load(d / f"{n}.npz"))
        if k in snapshots:
            snaps[k] = S.mixture().numpy().copy()
    return S.mixture().numpy(), snaps


def encode(mix):
    n = len(mix)
    return rf.encode(torch.from_numpy(np.maximum(mix, -1e4).T.copy()).reshape(1600, n, 1, 1), depth=6, keep="clip")


def decoded_mass(code, lut, gref):
    ids, pr = rf.probabilities(code)
    ids, pr = ids.reshape(code.depth, -1), pr.reshape(code.depth, -1)
    return (np.where(ids >= 0, pr, 0) * (np.asarray(lut)[np.maximum(ids, 0)] == gref[None])).sum(0)


def decoded_margin(code, lut, gref, G):
    groups = [np.flatnonzero(lut == g).tolist() for g in range(G)]
    m = rf.decode_groups(code, groups).reshape(G, -1).cpu().numpy()
    return m[gref, np.arange(m.shape[1])]


def votes_for(W, lut, gref):
    """(n,) runs whose winner fell in the reference's group, (n,) runs that labeled the seed."""
    have = W >= 0
    hit = have & (np.asarray(lut)[np.maximum(W, 0)] == gref[None])
    return hit.sum(0), have.sum(0)


def vote_entropy(W, lut, G):
    have = W >= 0
    g = np.where(have, np.asarray(lut)[np.maximum(W, 0)], G)
    n = W.shape[1]
    c = np.zeros((n, G + 1)); np.add.at(c, (np.broadcast_to(np.arange(n), W.shape), g), 1)
    c = c[:, :G]
    p = c / np.maximum(c.sum(1, keepdims=True), 1)
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(1)


# ------------------------------------------------------------------------------------------- the tests

def run(args):
    d = DATA / "ensemble" / args.subject / args.b0_noise
    ref = E.RunRecord.load(d / "r000.npz")
    n = len(ref.keys)
    W = winners_matrix(ref, ENSEMBLE, d)                                  # (11, n)
    H = winners_matrix(ref, HELD, d)                                      # (6, n)
    mix, snaps = mixture(ref, ENSEMBLE, d, snapshots=(2, 4, 6, 8))
    snaps[11] = mix
    code = encode(mix)
    allv = np.zeros((1600, n), np.int64)
    for w in W:
        ok = w >= 0; np.add.at(allv, (w[ok], np.flatnonzero(ok)), 1)
    ranks = code.ranks.reshape(code.depth, -1)
    vp, vtail = E.vote_planes(ranks, allv, (W >= 0).sum(0))
    # pairs (streamline, held-out run) where the held-out run labeled the seed
    have = H >= 0
    rows = np.broadcast_to(np.arange(n), H.shape)[have]
    vox = np.stack([(ref.keys >> 2 * E.KEY_BITS) & 4095, (ref.keys >> E.KEY_BITS) & 4095, ref.keys & 4095], 1) // BLOCK
    block = np.ravel_multi_index(tuple(vox.T), (4096 // BLOCK + 1,) * 3)[rows]
    Wb = boot_weights(block, args.boot)
    _, bidx = np.unique(block, return_inverse=True)
    folds = np.random.default_rng(1).integers(0, 5, bidx.max() + 1)[bidx]
    res = {"subject": args.subject, "b0_noise": args.b0_noise, "ensemble": ENSEMBLE, "held_out": HELD, "pairs": int(have.sum()),
           "bootstrap": {"resamples": args.boot, "block_voxels": BLOCK, "blocks": int(len(np.unique(block)))},
           "vote_tail_zero": round(float((vtail == 0).mean()), 4)}

    # 1 and 2
    t1, t2 = {}, {}
    for name, lut in groupings().items():
        lut = np.asarray(lut); G = int(lut.max()) + 1
        gref = lut[ref.cluster.astype(np.int64)]
        changed = (lut[np.maximum(H, 0)] != gref[None])[have]
        single = group_mass(ref.logp, lut, G)[np.arange(n), gref]
        full = group_mass(mix, lut, G)[np.arange(n), gref]
        hits, runs = votes_for(W, lut, gref)
        votes = hits / np.maximum(runs, 1)
        gv = E.group_votes(ranks, vp, lut, G)                              # (G, n) from the planes
        votes6 = gv[gref, np.arange(n)] / np.maximum(runs, 1)
        pred = {"single": single, "field6": decoded_mass(code, lut, gref), "margin6": decoded_margin(code, lut, gref, G),
                "full": full, "votes": votes, "votes6": votes6}
        aucs = {k: AUC(-v[rows], changed) for k, v in pred.items()}
        point = {k: a(np.ones(len(rows))) for k, a in aucs.items()}
        boots = {k: np.array([a(w) for w in Wb]) for k, a in aucs.items()}
        t1[name] = {"groups": G, "change_rate": round(float(changed.mean()), 4),
                    "auroc": {k: summarize(point[k], boots[k]) for k in pred},
                    "votes - field6": summarize(point["votes"] - point["field6"], boots["votes"] - boots["field6"]),
                    "votes - full": summarize(point["votes"] - point["full"], boots["votes"] - boots["full"]),
                    "full - field6": summarize(point["full"] - point["field6"], boots["full"] - boots["field6"]),
                    "ece": {k: ece(pred[k][rows], ~changed) for k in ("single", "field6", "full", "votes")}}
        d1 = t1[name]["votes - field6"]; d2 = t1[name]["votes - full"]
        t1[name]["verdict"] = ("votes needed" if d1[0] > 0.01 and d1[1] > 0 else "votes not needed") + \
                              ("; deeper ranks would do" if d1[0] > 0.01 and abs(d2[0]) <= 0.01 else "")
        # 2: within deciles of full, the vote fraction and the votes' entropy
        ent = vote_entropy(W, lut, G)
        fr = full[rows]
        edges = np.unique(np.quantile(fr, np.linspace(0, 1, 11)))
        dec = np.clip(np.searchsorted(edges, fr, side="right") - 1, 0, len(edges) - 2)
        cond = {}
        for k, s in (("vote fraction", -votes[rows]), ("vote entropy", ent[rows])):
            num = den = 0.0
            for b in range(len(edges) - 1):
                m = dec == b
                if changed[m].any() and (~changed[m]).any():
                    num += m.sum() * AUC(s[m], changed[m])(np.ones(m.sum())); den += m.sum()
            cond[k] = round(num / den, 4) if den else None
        y = changed.astype(float)
        t2[name] = {"conditional auroc within deciles of full (0.5 = nothing beyond the mixture)": cond,
                    "logloss (5-fold by block)": {"logit(full)": logloss_cv(logit(fr)[:, None], y, folds),
                                                  "logit(full) + vote fraction + vote entropy": logloss_cv(
                                                      np.c_[logit(fr), votes[rows], ent[rows]], y, folds),
                                                  "intercept only": logloss_cv(np.zeros((len(y), 0)), y, folds)}}
        print(name, json.dumps(t1[name]), json.dumps(t2[name]), flush=True)
    res["1_regrouping"], res["2_beyond_the_mixture"] = t1, t2

    # 3: runs
    t3 = {}
    for name in ("tract", "tract, outliers folded"):
        lut = np.asarray(groupings()[name]); G = int(lut.max()) + 1
        gref = lut[ref.cluster.astype(np.int64)]
        changed = (lut[np.maximum(H, 0)] != gref[None])[have]
        rowsR = {}
        for R in sorted(snaps):
            m = snaps[R]
            hits, runs = votes_for(W[:R], lut, gref)
            p = {"full": group_mass(m, lut, G)[np.arange(n), gref], "field6": decoded_mass(encode(m), lut, gref),
                 "votes": hits / np.maximum(runs, 1)}
            rowsR[R] = {k: {"auroc": round(AUC(-v[rows], changed)(np.ones(len(rows))), 4), "ece": ece(v[rows], ~changed)}
                        for k, v in p.items()}
        t3[name] = rowsR
        print(name, json.dumps(rowsR), flush=True)
    res["3_runs"] = t3

    # 4: voxels
    recs = {k: E.RunRecord.load(d / f"{k}.npz", logp=False).counts for k in ENSEMBLE + HELD}
    V = E.VoxelEnsemble(ref.shape)
    merges = {"CC 1-7": [f"CC{k}" for k in range(1, 8)], "SLF I-III": ["SLF-I", "SLF-II", "SLF-III"],
              "superficial (Sup-*)": [t for t in TRACT_NAMES if t.startswith("Sup-")]}
    t4 = {}
    for name, members in merges.items():
        idx = [TRACT_NAMES.index(t) for t in members]
        remap = np.arange(E.N_TRACTS); remap[idx] = idx[0]
        _, remap = np.unique(remap, return_inverse=True)
        G = int(remap.max()) + 1
        g0 = remap[idx[0]]

        def merged_first(c):
            cg = np.zeros((G, c.shape[1]), np.float32); np.add.at(cg, remap, c)
            return np.where(cg.max(0) >= V.tau, cg.argmax(0), G)

        def summed(c):
            w = V.winner(c)
            return np.where(w == E.NONE, G, remap[np.minimum(w, E.N_TRACTS - 1)])
        a = np.stack([summed(recs[k]) == g0 for k in ENSEMBLE]).mean(0)
        b = np.stack([merged_first(recs[k]) == g0 for k in ENSEMBLE]).mean(0)
        hits = np.stack([merged_first(recs[k]) == g0 for k in HELD])          # (6, nvox)
        inside = (a > 0) | (b > 0) | hits.any(0)
        hit = hits[:, inside].ravel()
        pa, pb = np.tile(a[inside], len(HELD)), np.tile(b[inside], len(HELD))
        t4[name] = {"voxels (any vote or held-out win)": int(inside.sum()),
                    "voxels where summed votes differ": int((a != b)[inside].sum()),
                    "summed votes: brier, ece": [round(float(((pa - hit) ** 2).mean()), 4), ece(pa, hit)],
                    "merged-first votes: brier, ece": [round(float(((pb - hit) ** 2).mean()), 4), ece(pb, hit)]}
        print(name, json.dumps(t4[name]), flush=True)
    res["4_voxels"] = t4
    (HERE / f"results/ensemble_votes_test_{args.b0_noise}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="PAT16")
    ap.add_argument("--b0-noise", default="added")
    ap.add_argument("--boot", type=int, default=200)
    run(ap.parse_args())
