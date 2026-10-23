"""Rigid registration, mode 2 (calibration_cohortII.REGISTRATION): FA onto FA first (center of mass, translation,
rigid; dipy, mutual information), then a rigid mutual-information refinement of the mean b0s started from it. A process
of its own, as bench/_register.py (dipy's OpenMP and torch's cannot share one on macOS).

    python bench/_register_fa.py in.npz out.npz   # in: moving_fa, static_fa, moving_b0, static_b0, moving_affine, static_affine
                                                  # [, mode = 3 [, other_start]]

The FA maps and the mean b0s of a scan share its grid (moving_affine, static_affine). out.npz: to_static (4x4, moving
world -> static world), ncc (the mean b0s' normalized cross-correlation in the static's bright voxels, as
_register.py's), and the FA stage's own: init_to_static, init_ncc (the b0s' NCC there), fa_ncc_init, fa_ncc (the FA maps'
NCC in the static FA's mask at the FA stage and after the refinement).

Mode 3 (in.npz has mode = 3): multi-start. Each start is refined by the same rigid MI step on the mean b0s, and the
result with the best final b0 NCC is kept: (a) the mode 1 pipeline on the mean b0s (center of mass, translation,
rigid), (b) mode 2's FA stage, (c) if given, other_start: the other direction's transform inverted (in dipy's
convention, static world -> moving world, that is the other direction's moving -> static matrix itself). out.npz:
to_static, ncc, best (a, b or c), and per candidate start_to_static_<x>, to_static_<x>, ncc_<x>, fa_ncc_<x>.
"""
import sys
import numpy as np

KW = dict(level_iters=[1000, 100, 10], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])   # as _register.py


def ncc(static, moved, m=None):
    m = static > np.quantile(static, 0.99) / 3 if m is None else m
    a, b = static[m] - static[m].mean(), moved[m] - moved[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def register(moving_fa, static_fa, moving_b0, static_b0, moving_affine, static_affine):
    from dipy.align import affine_registration
    from dipy.align.imaffine import AffineMap
    _, M0 = affine_registration(moving_fa, static_fa, moving_affine=moving_affine, static_affine=static_affine,
                                pipeline=["center_of_mass", "translation", "rigid"], **KW)        # dipy's M: static world -> moving world
    res, M = affine_registration(moving_b0, static_b0, moving_affine=moving_affine, static_affine=static_affine,
                                 pipeline=["rigid"], starting_affine=M0, **KW)
    amap = lambda Mx, vol: AffineMap(Mx, static_b0.shape, static_affine, vol.shape, moving_affine).transform(vol)  # noqa: E731
    fm = static_fa > 0
    return {"to_static": np.linalg.inv(M), "ncc": ncc(static_b0, res), "init_to_static": np.linalg.inv(M0),
            "init_ncc": ncc(static_b0, amap(M0, moving_b0)), "fa_ncc_init": ncc(static_fa, amap(M0, moving_fa), fm),
            "fa_ncc": ncc(static_fa, amap(M, moving_fa), fm)}


def multi(moving_fa, static_fa, moving_b0, static_b0, moving_affine, static_affine, other_start=None):
    """Mode 3: the starts (a) b0 pipeline, (b) FA stage, (c) other_start (dipy's convention), each refined by a rigid MI
    step on the mean b0s; the best by final b0 NCC."""
    from dipy.align import affine_registration
    from dipy.align.imaffine import AffineMap
    b0 = dict(moving_affine=moving_affine, static_affine=static_affine)
    starts = {}
    _, starts["a"] = affine_registration(moving_b0, static_b0, pipeline=["center_of_mass", "translation", "rigid"], **b0, **KW)
    _, starts["b"] = affine_registration(moving_fa, static_fa, moving_affine=moving_affine, static_affine=static_affine,
                                         pipeline=["center_of_mass", "translation", "rigid"], **KW)
    if other_start is not None:
        starts["c"] = np.asarray(other_start, float)
    fm = static_fa > 0
    out = {}
    for x, M0 in starts.items():
        res, M = affine_registration(moving_b0, static_b0, pipeline=["rigid"], starting_affine=M0, **b0, **KW)
        out[f"start_to_static_{x}"] = np.linalg.inv(M0)
        out[f"to_static_{x}"] = np.linalg.inv(M)
        out[f"ncc_{x}"] = ncc(static_b0, res)
        out[f"fa_ncc_{x}"] = ncc(static_fa, AffineMap(M, static_fa.shape, static_affine, moving_fa.shape, moving_affine).transform(moving_fa), fm)
    best = max(starts, key=lambda x: out[f"ncc_{x}"])
    out.update(best=best, to_static=out[f"to_static_{best}"], ncc=out[f"ncc_{best}"])
    return out


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    args = [z[k] for k in ("moving_fa", "static_fa", "moving_b0", "static_b0", "moving_affine", "static_affine")]
    if "mode" in z.files and int(z["mode"]) == 3:
        out = multi(*args, other_start=z["other_start"] if "other_start" in z.files else None)
    else:
        out = register(*args)
    np.savez(sys.argv[2], **out)
