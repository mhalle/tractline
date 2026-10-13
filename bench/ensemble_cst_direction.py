"""Does the CST's direction disagree between runs where its streamlines are unstable? The simplest test of per-voxel,
per-tract direction statistics (NOTES 2026-10-13, "Direction spread along the CST").

    uv run --group dipy --group rankfield bench/ensemble_cst_direction.py

cIIIs04-2.1's 17-run ensemble (A, smooth/r001..r016; compact records, full streamlines). Per run and voxel (DWI grid),
the CST streamlines' direction there: the principal axis of their summed segment tangents' outer products (axial, sign
free), where at least MIN_SEGS segments of that run's CST pass. Per voxel where at least MIN_RUNS runs have one, the
spread: the mean angle between each run's direction and their common direction (the principal axis of the runs'
directions' summed outer products).
  - where: the spread's median per 6 mm axial band, with the share of the band's voxels FastSurfer labels brainstem
    (cIIIs04's label map, carried by the T1-to-DWI transform);
  - whether it predicts: for A's CST streamlines, the spread averaged over (and its 90th percentile along) the voxels
    each passes, against whether each rescan keeps the name CST: AUROC, beside the ensemble's own vote share (the
    share of the 17 runs naming the seed's streamline CST); and the two together, a logistic model fitted on the
    same-session rescan and scored on session 5.
Writes results/ensemble_cst_direction.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_endpoints import Anatomy, FS  # noqa: E402

RUNS, MIN_SEGS, MIN_RUNS, BAND_MM = 17, 3, 9, 6.0
CST = TRACT_NAMES.index("CST")
L = np.asarray(LUT)


def auroc(score, positive):
    """P(score of a positive > score of a negative), ties half."""
    s, y = np.asarray(score, float), np.asarray(positive, bool)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sv = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic(X, y, iters=50):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-np.column_stack([np.ones(len(X)), X]) @ w))


def run_directions(fibers, inv, shape):
    """CST fibers -> (voxel flat indices, unit principal directions) where at least MIN_SEGS segments pass."""
    P = np.concatenate(fibers)
    lens = np.array([len(f) for f in fibers])
    start = np.concatenate([[0], np.cumsum(lens)])
    a = np.concatenate([np.arange(s, e - 1) for s, e in zip(start[:-1], start[1:])])
    t = P[a + 1] - P[a]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    mid = (P[a + 1] + P[a]) / 2
    ijk = np.rint(mid @ inv[:3, :3].T + inv[:3, 3]).astype(int)
    ok = np.all((ijk >= 0) & (ijk < shape), 1)
    flat = np.ravel_multi_index(ijk[ok].T, shape)
    t = t[ok]
    vox, inv_idx, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    D = np.zeros((len(vox), 3, 3))
    for i in range(3):
        for j in range(3):
            D[:, i, j] = np.bincount(inv_idx, weights=t[:, i] * t[:, j], minlength=len(vox))
    keep = cnt >= MIN_SEGS
    _, vecs = np.linalg.eigh(D[keep])
    return vox[keep], vecs[:, :, 2]


def main():
    from _masivar import scan_files
    d = DATA / "ensemble" / "cIIIs04" / "retest" / "cIIIs04-2.1"
    img = nib.load(scan_files("cIIIs04", "s1Bx1", 1, None)["APP"][0][1])
    aff, shape = img.affine, img.shape[:3]
    inv = np.linalg.inv(aff)
    paths = [d / "A.npz"] + [d / "smooth" / f"r{r:03d}.npz" for r in range(1, RUNS)]
    nvox = int(np.prod(shape))
    dirs = np.zeros((RUNS, nvox, 3), np.float32)
    has = np.zeros((RUNS, nvox), bool)
    keys, tracts = [], []
    A = None
    for r, p in enumerate(paths):
        rec = E.RunRecord.load(p, logp=False, fibers=True)
        m = L[rec.cluster] == CST
        vox, vec = run_directions([rec.fibers[i] for i in np.flatnonzero(m)], inv, shape)
        dirs[r, vox], has[r, vox] = vec, True
        keys.append(rec.keys); tracts.append(L[rec.cluster])
        if r == 0:
            A = rec
        print(f"run {r}: {m.sum()} CST streamlines, {len(vox)} voxels", flush=True)
    nruns = has.sum(0)
    good = np.flatnonzero(nruns >= MIN_RUNS)
    V = dirs[:, good]                                                    # (R, n, 3)
    S = np.einsum("rni,rnj,rn->nij", V, V, has[:, good].astype(np.float32))
    _, ev = np.linalg.eigh(S)
    common = ev[:, :, 2]
    cosang = np.abs(np.einsum("rni,ni->rn", V, common)).clip(0, 1)
    ang = np.degrees(np.arccos(cosang))
    spread = np.where(has[:, good], ang, 0).sum(0) / has[:, good].sum(0)
    spread_vol = np.full(nvox, np.nan, np.float32)
    spread_vol[good] = spread
    res = {"voxels with the CST in at least 9 runs": int(len(good)),
           "spread (deg): median / 90th percentile": [round(float(np.median(spread)), 1), round(float(np.percentile(spread, 90)), 1)]}

    # where: per axial band, with FastSurfer's brainstem share
    anat = Anatomy("cIIIs04")
    t1_to_dwi = np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"]
    w = np.stack(np.unravel_index(good, shape), 1) @ aff[:3, :3].T + aff[:3, 3]
    lab = anat.raw(w @ np.linalg.inv(t1_to_dwi)[:3, :3].T + np.linalg.inv(t1_to_dwi)[:3, 3])
    z = w[:, 2]
    bands = []
    for lo in np.arange(np.floor(z.min() / BAND_MM) * BAND_MM, z.max(), BAND_MM):
        b = (z >= lo) & (z < lo + BAND_MM)
        if b.sum() >= 20:
            bands.append({"z (mm)": f"{lo:.0f} to {lo + BAND_MM:.0f}", "voxels": int(b.sum()),
                          "median spread (deg)": round(float(np.median(spread[b])), 1),
                          "brainstem share": round(float((lab[b] == 16).mean()), 2),
                          "cerebral white matter share": round(float(np.isin(lab[b], [2, 41]).mean()), 2)})
    res["by axial band (inferior first)"] = bands
    bs = lab == 16
    res["median spread (deg): brainstem / elsewhere"] = [round(float(np.median(spread[bs])), 1), round(float(np.median(spread[~bs])), 1)]
    for b in bands:
        print(b, flush=True)

    # whether it predicts: A's CST streamlines, their path's spread against keeping the name on each rescan
    mA = L[A.cluster] == CST
    rowsA = np.flatnonzero(mA)
    path_mean, path_p90 = np.full(len(rowsA), np.nan), np.full(len(rowsA), np.nan)
    for n, i in enumerate(rowsA):
        f = A.fibers[i]
        ijk = np.rint(f @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < shape), 1)
        s = spread_vol[np.unique(np.ravel_multi_index(ijk[ok].T, shape))]
        s = s[~np.isnan(s)]
        if len(s):
            path_mean[n], path_p90[n] = s.mean(), np.percentile(s, 90)
    keysA = A.keys[rowsA]
    # the ensemble's own vote share for these seeds (runs where the seed is labeled)
    votes, seen = np.zeros(len(rowsA)), np.zeros(len(rowsA))
    for k, t in zip(keys, tracts):
        pos = np.minimum(np.searchsorted(k, keysA), len(k) - 1)
        hit = k[pos] == keysA
        seen += hit
        votes += hit & (t[pos] == CST)
    share = votes / np.maximum(seen, 1)
    res["A's CST streamlines with a defined path spread"] = int((~np.isnan(path_mean)).sum())
    Xs = {}
    for tag in ("within", "session3", "session5"):
        R = E.RunRecord.load(d / f"{tag}_onA.npz", logp=False)
        pos = np.minimum(np.searchsorted(R.keys, keysA), len(R.keys) - 1)
        hit = (R.keys[pos] == keysA) & ~np.isnan(path_mean)
        keep = L[R.cluster[pos]] == CST
        y = keep[hit]
        res[f"{tag}: AUROC predicting a name change (higher score = change)"] = {
            "path mean spread": round(auroc(path_mean[hit], ~y), 3),
            "path 90th-percentile spread": round(auroc(path_p90[hit], ~y), 3),
            "ensemble: 1 - vote share": round(auroc(1 - share[hit], ~y), 3),
            "keeps CST (n)": [round(float(y.mean()), 3), int(hit.sum())]}
        Xs[tag] = (np.column_stack([share[hit], path_p90[hit]]), y.astype(float), share[hit])
        print(tag, res[f"{tag}: AUROC predicting a name change (higher score = change)"], flush=True)
    Xtr, ytr, _ = Xs["within"]
    Xte, yte, s_te = Xs["session5"]
    w_both, w_share = logistic(Xtr, ytr), logistic(Xtr[:, :1], ytr)
    res["fitted on the same-session rescan, scored on session 5: AUROC predicting keep, share only / share + spread"] = [
        round(auroc(predict(w_share, Xte[:, :1]), yte.astype(bool)), 3), round(auroc(predict(w_both, Xte), yte.astype(bool)), 3)]
    res["correlation of vote share with path 90th-percentile spread"] = round(float(np.corrcoef(Xte[:, 0], Xte[:, 1])[0, 1]), 3)
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("fitted", "correlation", "median spread", "spread"))}))
    (HERE / "results/ensemble_cst_direction.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
