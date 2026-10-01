"""Comparing two tractographies made from the same seeds, fiber by fiber: {seed key: (n, 3) RAS points}.

Used by ukf32_compare.py (float32 against float64) and ukf_noise_floor.py (bootstrap replicates
against the scan as acquired), so the two report the same measures.
"""
import numpy as np


def q(x, ps=(0.5, 0.9, 0.99, 1.0)):
    return [float(f"{v:.3g}") for v in np.quantile(x, ps)] if len(x) else []


def density(fibs, i2r, dims_ijk):
    """Streamlines per voxel on the DWI grid (each fiber counted once per voxel it visits), flattened."""
    r2i = np.linalg.inv(i2r)
    m = np.zeros(dims_ijk, np.int32)
    for p in fibs:
        ijk = np.rint(p @ r2i[:3, :3].T + r2i[:3, 3]).astype(int)
        ijk = np.unique(ijk[(ijk >= 0).all(1) & (ijk < dims_ijk).all(1)], axis=0)
        m[tuple(ijk.T)] += 1
    return m.ravel().astype(float)


def compare(a: dict, b: dict, i2r, dims_ijk) -> dict:
    """a against b: which seeds each kept, and for the seeds both kept with the same point count, the
    largest point distance and the ends' distances (b's fiber taken in the orientation nearer a's);
    the length difference for every seed both kept; and the density maps' Pearson r."""
    both = sorted(set(a) & set(b))
    same = [k for k in both if len(a[k]) == len(b[k])]

    def oriented(k):
        r = b[k][::-1]
        return r if np.linalg.norm(a[k] - r, axis=1).max() < np.linalg.norm(a[k] - b[k], axis=1).max() else b[k]
    ob = {k: oriented(k) for k in same}
    dmax = np.array([np.linalg.norm(a[k] - ob[k], axis=1).max() for k in same])
    ends = np.concatenate([np.linalg.norm(a[k][[0, -1]] - ob[k][[0, -1]], axis=1) for k in same]) if same else np.array([])
    length = lambda f: np.linalg.norm(np.diff(f, axis=0), axis=1).sum()
    dlen = np.array([length(a[k]) - length(b[k]) for k in both])
    da, db = density(a.values(), i2r, dims_ijk), density(b.values(), i2r, dims_ijk)
    return {"both": len(both), "only_first": len(set(a) - set(b)), "only_second": len(set(b) - set(a)),
            "same_point_count": len(same), "max_point_distance_mm": q(dmax),
            "within_1e-3_mm": int((dmax < 1e-3).sum()), "within_0.1_mm": int((dmax < 0.1).sum()),
            "ends_within_0.06_mm_fraction_same_count": round(float((ends < 0.06).mean()), 4) if len(ends) else None,
            "length_diff_mm_abs": q(np.abs(dlen)), "density_r": round(float(np.corrcoef(da, db)[0, 1]), 5)}
