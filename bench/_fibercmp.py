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


def step_diff(a: dict, b: dict, vox) -> dict:
    """One step from identical inputs, a against b (dicts of advance()'s outputs as numpy): relative
    errors of state and covariance, position error in mm (vox: spacings in (k, j, i)), direction error,
    FA and mean-signal errors, and how many of each decision flipped."""
    rel = lambda u, v, ax: np.linalg.norm((u - v).reshape(len(u), -1), axis=1) / np.linalg.norm(v.reshape(len(v), -1), axis=1)
    ang = np.degrees(np.arccos(np.clip(np.abs((a["dir"] * b["dir"]).sum(1)), 0, 1)))
    d = {"state_rel_err": q(rel(a["state"], b["state"], 1)), "P_rel_err": q(rel(a["P"], b["P"], 1)),
         "position_err_mm": q(np.linalg.norm((a["x"] - b["x"]) * vox, axis=1)), "direction_err_deg": q(ang),
         "fa_abs_err": q(np.abs(a["fa"] - b["fa"])), "mean_signal_abs_err": q(np.abs(a["mean_signal"] - b["mean_signal"]))}
    for k in ("swap", "swap2", "inside", "stop"):
        d[f"{k}_flips"] = int((a[k] != b[k]).sum())
    flip = a["stop"] != b["stop"]
    if flip.any():                                                    # how close the float64 step was to a threshold
        d["stop_flips_ref_margin"] = {"fa_minus_0.08": q(b["fa"][flip] - 0.08, (0, 0.5, 1)),
                                      "mean_signal_minus_0.06": q(b["mean_signal"][flip] - 0.06, (0, 0.5, 1))}
    return d
