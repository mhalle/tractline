"""Does the envelopes' calibration survive a change of grid if presence is defined by streamline density? (NOTES
2026-10-13, "Grid-free presence").

    uv run --group dipy --group rankfield bench/ensemble_grid_check.py

cIIIs04-2.1's four scans, 17 runs each (compact records; full streamlines, in A's world mm), focus tracts CST, AF,
UF. Each run's streamlines of a tract are sampled every STEP_MM and voxelized on a grid, counting each streamline
once per voxel it visits:
  - dwi: A's diffusion grid (2.1 x 2.1 x 2.2 mm);
  - t1: a 1 mm grid in the subject's T1 space (FastSurfer's label grid, cropped to the brain), the streamlines carried
    by the T1-to-DWI transform.
Presence, two definitions:
  - density (grid free): the count over the voxel's mean cross-section (1.5 h^2 for a voxel of side h, the mean
    projected area of a cube), smoothed by a Gaussian of SMOOTH_MM, at least RHO streamlines per mm^2 (0.3: about 2
    streamlines in a 2.1 mm voxel, the earlier rule);
  - count (the earlier rule): at least 2 streamlines in the voxel, here also on the 1 mm grid, to show what a grid
    change does to it.
Leaving each scan out, its own run is the target: single (each other scan's ensemble alone, averaged) and pooled (the
three together). Coverage: the target's voxels inside the widest outline and the envelopes at stated levels.
Writes results/ensemble_grid_check.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402
from ensemble_pooled import thresholds, LEVELS  # noqa: E402
from ensemble_endpoints import Anatomy, FS  # noqa: E402

RUNS, STEP_MM, SMOOTH_MM, RHO = 17, 0.4, 1.0, 0.3
FOCUS = ("CST", "AF", "UF")
L = np.asarray(LUT)


def sample(fibers):
    """fibers -> points every STEP_MM or closer along each (P) and each point's streamline (O), vectorized"""
    if not fibers:
        return np.zeros((0, 3)), np.zeros(0, int)
    lens = np.array([len(f) for f in fibers])
    F = np.concatenate(fibers).astype(np.float64)
    start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    owner_pt = np.repeat(np.arange(len(fibers)), lens)
    a = np.setdiff1d(np.arange(len(F) - 1), start[1:] - 1)                 # segment starts within a streamline
    seg = F[a + 1] - F[a]
    k = np.maximum(np.ceil(np.linalg.norm(seg, axis=1) / STEP_MM).astype(int), 1)
    first = np.repeat(np.cumsum(k) - k, k)
    t = (np.arange(k.sum()) - first) / np.repeat(k, k)
    P = np.repeat(F[a], k, axis=0) + np.repeat(seg, k, axis=0) * t[:, None]
    O = np.repeat(owner_pt[a], k)
    last = start + lens - 1
    return np.vstack([P, F[last]]), np.concatenate([O, np.arange(len(fibers))])


class Grid:
    def __init__(self, name, affine, shape, to_grid_world=np.eye(4)):
        self.name, self.affine, self.shape = name, affine, tuple(int(s) for s in shape)
        self.inv = np.linalg.inv(affine) @ to_grid_world               # DWI world mm -> this grid's voxel
        self.h = np.sqrt((affine[:3, :3] ** 2).sum(0))
        self.area = 1.5 * float(np.prod(self.h)) ** (2 / 3)              # mean projected area of the voxel, mm^2

    def counts(self, P, O):
        """streamlines visiting each voxel (flat index -> count), each streamline once per voxel; P, O from sample()"""
        n = int(np.prod(self.shape))
        if not len(P):
            return np.zeros(n, np.uint16)
        ijk = np.rint(P @ self.inv[:3, :3].T + self.inv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < self.shape), 1)
        flat = np.ravel_multi_index(ijk[ok].T, self.shape)
        pair = np.unique(O[ok].astype(np.int64) * n + flat)
        return np.bincount(pair % n, minlength=n).astype(np.uint16)

    def density(self, c):
        """streamlines per mm^2 of a voxel's mean cross-section, smoothed by SMOOTH_MM"""
        return gaussian_filter((c.reshape(self.shape) / self.area).astype(np.float32), SMOOTH_MM / self.h)

    def presence(self, c):
        return {"density": (self.density(c) >= RHO).ravel(), "count": c >= 2}


def coverage(votes, R, target):
    t = int(target.sum())
    if not t:
        return None
    th = thresholds(votes, R, LEVELS)
    return [round(float((target & (votes > 0)).sum() / t), 3)] + [round(float((target & (votes >= v)).sum() / t), 3) for v in th]


def main():
    from _masivar import scan_files
    t0 = time.time()
    pair = "cIIIs04-2.1"
    subject, (ses, scan), rescans, shells, _ = PAIRS[pair]
    d = DATA / "ensemble" / subject / "retest" / pair
    img = nib.load(scan_files(subject, ses, scan, shells)["APP"][0][1])
    anat = Anatomy("cIIIs04")
    t1_to_dwi = np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"]
    ijk = np.argwhere(anat.lab > 0)
    lo, hi = np.maximum(ijk.min(0) - 5, 0), np.minimum(ijk.max(0) + 6, anat.lab.shape)
    crop = anat.affine.copy()
    crop[:3, 3] = anat.affine[:3, :3] @ lo + anat.affine[:3, 3]
    grids = [Grid("dwi", img.affine, img.shape[:3]), Grid("t1", crop, hi - lo, np.linalg.inv(t1_to_dwi))]
    scans = {"A": [d / "A.npz"] + [d / "smooth" / f"r{r:03d}.npz" for r in range(1, RUNS)]}
    for tag in rescans:
        scans[tag] = [d / f"{tag}_onA.npz"] + [d / "smooth" / f"{tag}_r{r:03d}.npz" for r in range(1, RUNS)]
    tk = [TRACT_NAMES.index(n) for n in FOCUS]
    votes, target = {}, {}                                                 # [grid][defn][scan] -> (3, nvox)
    for g in grids:
        votes[g.name] = {"density": {}, "count": {}}; target[g.name] = {"density": {}, "count": {}}
    for s, paths in scans.items():
        for i, p in enumerate(paths):
            rec = E.RunRecord.load(p, logp=False, fibers=True)
            tract = L[rec.cluster]
            samples = {k: sample([rec.fibers[j] for j in np.flatnonzero(tract == k)]) for k in tk}
            for g in grids:
                pres = [g.presence(g.counts(*samples[k])) for k in tk]
                for defn in ("density", "count"):
                    stack = np.stack([pp[defn] for pp in pres])
                    v = votes[g.name][defn]
                    v[s] = stack.astype(np.uint8) if s not in v else v[s] + stack
                    if i == 0:
                        target[g.name][defn][s] = stack
            del rec
        print(f"{s} done {time.time() - t0:.0f}s", flush=True)
    res = {"focus": FOCUS, "levels": LEVELS, "step_mm": STEP_MM, "smooth_mm": SMOOTH_MM, "rho per mm2": RHO,
           "grids": {g.name: {"shape": list(g.shape), "voxel mm": [round(float(x), 2) for x in g.h]} for g in grids},
           "coverage [widest, *levels], mean over held-out scans": {}}
    for g in grids:
        for defn in ("density", "count"):
            for mode in ("single", "pooled"):
                for ti, name in enumerate(FOCUS):
                    rows = []
                    for h in scans:
                        others = [s for s in scans if s != h]
                        tg = target[g.name][defn][h][ti]
                        if mode == "single":
                            cs = [coverage(votes[g.name][defn][s][ti], RUNS, tg) for s in others]
                            rows.append(np.mean([c for c in cs if c], 0))
                        else:
                            pooled = sum(votes[g.name][defn][s][ti].astype(np.int16) for s in others)
                            rows.append(np.array(coverage(pooled, RUNS * len(others), tg)))
                    key = f"{g.name} grid, {defn}, {mode}, {name}"
                    res["coverage [widest, *levels], mean over held-out scans"][key] = [round(float(x), 3) for x in np.mean(rows, 0)]
                    print(key, res["coverage [widest, *levels], mean over held-out scans"][key], flush=True)
        for defn in ("density", "count"):
            vox = {n: int(np.mean([target[g.name][defn][s][ti].sum() for s in scans])) for ti, n in enumerate(FOCUS)}
            res[f"{g.name} grid, {defn}: target voxels per tract (mean)"] = vox
            res[f"{g.name} grid, {defn}: target volume per tract, mL"] = {n: round(v * float(np.prod(g.h)) / 1000, 1) for n, v in vox.items()}
    (HERE / "results/ensemble_grid_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if "volume" in k}), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
