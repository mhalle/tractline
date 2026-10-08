"""Rigid registration (dipy, mutual information) in a process of its own: dipy's OpenMP runtime and torch's cannot both
load in one process on macOS (OMP Error #15), so bench/ensemble_retest.py calls this as a script.

    python bench/_register.py in.npz out.npz      # in: moving, moving_affine, static, static_affine

out.npz: to_static (4x4, moving world -> static world) and ncc (normalized cross-correlation in the static's bright
voxels, after).
"""
import sys
import numpy as np


def register(moving, moving_affine, static, static_affine):
    from dipy.align import affine_registration
    res, M = affine_registration(moving, static, moving_affine=moving_affine, static_affine=static_affine,
                                 pipeline=["center_of_mass", "translation", "rigid"], level_iters=[1000, 100, 10],
                                 sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])
    m = static > np.quantile(static, 0.99) / 3
    a, b = static[m] - static[m].mean(), res[m] - res[m].mean()
    return np.linalg.inv(M), float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))   # dipy's M: static world -> moving world


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    T, ncc = register(z["moving"], z["moving_affine"], z["static"], z["static_affine"])
    np.savez(sys.argv[2], to_static=T, ncc=ncc)
