"""Why do corticospinal streamlines turn into Other on another run or a rescan? (NOTES 2026-10-13, "Why CST
streamlines become Other").

    uv run --group dipy --group rankfield bench/ensemble_cst_other.py

cIIIs04-2.1's compact records: run A against a replicate (smooth/r001) and against the same-session and session 5
rescans (<tag>_onA), seeds labeled in both. For A's streamlines named CST:
  - what the other run gives: CST, Other or another tract; for Other, which kind of cluster - the outlier twin of A's
    own CST cluster (cluster + 800), another cluster's twin (and which tract that cluster belongs to), or one of the
    atlas's non-anatomical clusters (0-799, mapped to Other);
  - geometry: the mean distance between the two versions (MDF, 20-point paths), their lengths, and how far each end
    moved (the upper end: the highest point; the lower end: the lowest);
  - the labeler's own probability for CST in A (cluster probabilities summed over CST's 11 clusters);
  - with FastSurfer's labels (ensemble_endpoints.py): whether each version fits the CST endpoint rule, and the label
    at the other version's lower end.
Writes results/ensemble_cst_other.json.
"""
import collections, json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_endpoints import Anatomy, FS, ends_info, consistent  # noqa: E402

L = np.asarray(LUT)
OTHER, CST = len(TRACT_NAMES) - 1, TRACT_NAMES.index("CST")
REGIONS = {0: "unlabeled", 2: "cerebral white matter", 41: "cerebral white matter", 16: "brainstem", 7: "cerebellum",
           46: "cerebellum", 8: "cerebellum", 47: "cerebellum", 28: "ventral DC", 60: "ventral DC"}


def kind(ca, cb):
    if L[cb] != OTHER:
        return "CST" if L[cb] == CST else "another tract"
    if cb == ca + 800:
        return "Other: its own cluster's outlier twin"
    if cb >= 800:
        return "Other: another cluster's outlier twin"
    return "Other: a non-anatomical cluster"


def mdf(p, q):
    a = np.linalg.norm(p - q, axis=-1).mean(-1)
    return np.minimum(a, np.linalg.norm(p - q[:, ::-1], axis=-1).mean(-1))


def share(x):
    c = collections.Counter(x)
    return {k: round(v / max(len(x), 1), 3) for k, v in c.most_common()}


def main():
    from _masivar import scan_files
    d = DATA / "ensemble" / "cIIIs04" / "retest" / "cIIIs04-2.1"
    aff = nib.load(scan_files("cIIIs04", "s1Bx1", 1, None)["APP"][0][1]).affine
    anat = Anatomy("cIIIs04")
    to_t1 = np.linalg.inv(np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"])
    A = E.RunRecord.load(d / "A.npz", fibers=True)
    p_cst = np.exp(A.logp.astype(np.float32))[:, L == CST].sum(1)
    del A.logp
    res = {"CST clusters": np.flatnonzero(L == CST).tolist()}
    for tag, path in [("replicate", d / "smooth" / "r001.npz"), ("same-session rescan", d / "within_onA.npz"),
                      ("session 5 rescan", d / "session5_onA.npz")]:
        B = E.RunRecord.load(path, logp=False, fibers=True)
        _, ia, ib = np.intersect1d(A.keys, B.keys, return_indices=True)
        m = L[A.cluster[ia]] == CST
        ia, ib = ia[m], ib[m]
        ca, cb = A.cluster[ia], B.cluster[ib]
        k = np.array([kind(x, y) for x, y in zip(ca, cb)])
        keep, other = L[cb] == CST, L[cb] == OTHER
        dd = mdf(A.paths[ia], B.paths[ib])
        fa, fb = [A.fibers[i] for i in ia], [B.fibers[i] for i in ib]
        upper = lambda F: np.array([f[np.argmax(f[:, 2])] for f in F])     # noqa: E731
        lower = lambda F: np.array([f[np.argmin(f[:, 2])] for f in F])     # noqa: E731
        du = np.linalg.norm(upper(fa) - upper(fb), axis=1)
        dl = np.linalg.norm(lower(fa) - lower(fb), axis=1)
        fits = {}
        for who, F in (("A", fa), ("B", fb)):
            ra, rb, sa, sb, en = ends_info(F, anat, to_t1, aff)
            fits[who], _ = consistent(np.full(len(F), CST), ra, rb, sa, sb, en)
        lab_b = anat.raw(lower(fb) @ to_t1[:3, :3].T + to_t1[:3, 3])
        lab_a = anat.raw(lower(fa) @ to_t1[:3, :3].T + to_t1[:3, 3])
        med = lambda x: round(float(np.median(x)), 1)                       # noqa: E731
        twins = [TRACT_NAMES[L[c - 800]] for c in cb[other & (cb >= 800)]]
        r = {"A's CST streamlines": int(len(ia)), "the other run gives": share(k.tolist()),
             "Other's twins belong to clusters of": dict(list(share(twins).items())[:5]),
             "MDF between versions, median mm: kept CST / became Other": [med(dd[keep]), med(dd[other])],
             "upper end moved, median mm: kept / Other": [med(du[keep]), med(du[other])],
             "lower end moved, median mm: kept / Other": [med(dl[keep]), med(dl[other])],
             "length, median mm: A / B, became Other": [med([np.linalg.norm(np.diff(f, axis=0), axis=1).sum() for f in np.array(fa, object)[other]]),
                                                        med([np.linalg.norm(np.diff(f, axis=0), axis=1).sum() for f in np.array(fb, object)[other]])],
             "fits the CST rule: A kept / A became Other / B kept / B became Other": [
                 round(float(fits["A"][keep].mean()), 2), round(float(fits["A"][other].mean()), 2),
                 round(float(fits["B"][keep].mean()), 2), round(float(fits["B"][other].mean()), 2)],
             "lower end's label: B where it became Other": share([REGIONS.get(int(x), "other") for x in lab_b[other]]),
             "lower end's label: A, the same seeds": share([REGIONS.get(int(x), "other") for x in lab_a[other]]),
             "lower end's label: B where it kept CST": share([REGIONS.get(int(x), "other") for x in lab_b[keep]]),
             "labeler's P(CST) in A, median: kept / became Other": [round(float(np.median(p_cst[ia][keep])), 3),
                                                                     round(float(np.median(p_cst[ia][other])), 3)]}
        res[tag] = r
        print(tag, json.dumps(r), flush=True)
        del B
    (HERE / "results/ensemble_cst_other.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
