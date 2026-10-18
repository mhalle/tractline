"""Denoising of a DWI before tracking, opt-in (docs/robustness-plan.md, step 3): dipy's MP-PCA or Patch2Self, in a
process of its own. dipy's OpenMP runtime and torch's cannot both load in one process on macOS (OMP Error #15; as
bench/_register.py), so this module imports neither at its top: the parent writes the series to a temporary
directory, runs `python -m tractline.denoise` on it, and reads the result back.

    d = denoise(dwi, bval, "mppca")            # Denoised: dwi (float32), sigma (MP-PCA's noise SD per voxel), info
    d = denoise(dwi, bval, "mppca:r3")         # a setting in the name: MP-PCA's patch radius (r), Patch2Self's sketch seed (s)

  mppca       dipy.denoise.localpca.mppca (Veraart et al. 2016; Manjon et al. 2013's patch weighting): every 5 x 5 x 5
              patch (PATCH_RADIUS 2: 125 samples, at least the 112 volumes of a MASiVar scan, as MRtrix's dwidenoise
              chooses its window) over all volumes at once, the components below the Marchenko-Pastur edge dropped.
              No mask: the whole field of view, as dwidenoise (dipy zeroes what a mask leaves out, and the tracking
              mask is computed after, from the denoised DWI). Run in `workers` slabs along the longest spatial axis,
              each slab with the 2 r slices its patches reach; the result is dipy's on the whole volume, bit for bit
              (each voxel's patches are summed in dipy's order; tests/test_denoise.py).
              Its noise estimate counts the patch's voxels as independent samples. At radius 2 the 125 voxels barely
              exceed MASiVar's 112 volumes, and noise correlated in-plane (MASiVar's Philips scans: the bootstrap's
              smooth signs, FWHM 4) leaves fewer effective samples still: there its sigma came out far below the noise
              and little was removed (NOTES 2026-10-17, "Robustness step 3"). A larger radius ("mppca:r3") gives it
              more samples.
  patch2self  dipy.denoise.patch2self (Fadnavis et al. 2020; dipy's version 3, which fits on a count sketch of the
              voxels, Fadnavis et al. 2024): each volume regressed on the others, voxel by voxel (patch radius 0, OLS),
              the b0s on the other b0s. It needs noise independent between volumes, not between voxels. The sketch
              draws from numpy's global generator, seeded here (SKETCH_SEED, or "patch2self:s<seed>") so a run repeats. Negative values
              clipped to 0 (dipy's shift_intensity, its default, shifts by min - min: nothing). It estimates no sigma.
"""
from __future__ import annotations

import json, os, subprocess, sys, tempfile, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

METHODS = ("mppca", "patch2self")
PATCH_RADIUS = 2
SKETCH_SEED = 0


@dataclass
class Denoised:
    dwi: np.ndarray              # (X, Y, Z, G) float32
    sigma: np.ndarray | None     # (X, Y, Z) MP-PCA's noise SD per voxel (dipy's return_sigma); None for patch2self
    info: dict                   # the method, its settings, the worker's seconds


def parse(spec):
    """A denoiser's name with its setting, if any: "mppca" (radius PATCH_RADIUS), "mppca:r3" (MP-PCA's patch radius),
    "patch2self" (sketch seed SKETCH_SEED), "patch2self:s1" (Patch2Self's sketch seed) -> (method, settings)."""
    method, _, setting = str(spec).partition(":")
    if method not in METHODS:
        raise ValueError(f"denoise: {spec!r}: the method is not one of {METHODS}")
    settings = {"patch_radius": PATCH_RADIUS} if method == "mppca" else {"sketch_seed": SKETCH_SEED}
    key = {"mppca": "r", "patch2self": "s"}[method]
    if setting:
        if not (setting[:1] == key and setting[1:].isdigit()):
            raise ValueError(f"denoise: {spec!r}: {method}'s setting is {key}<integer>")
        settings[next(iter(settings))] = int(setting[1:])
    return method, settings


def denoise(dwi, bval, method="mppca", workers=None) -> Denoised:
    """dwi (X, Y, Z, G), bval (G,): the series denoised in a subprocess (module docstring). method: a name parse()
    takes. workers: MP-PCA's slabs, one process each (default: one per core)."""
    method, settings = parse(method)
    workers = int(workers or os.cpu_count() or 1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "dwi.npy", np.asarray(dwi, np.float32))
        np.save(tmp / "bval.npy", np.asarray(bval, np.float64))
        (tmp / "args.json").write_text(json.dumps({"method": method, "workers": workers, **settings}))
        subprocess.run([sys.executable, "-m", "tractline.denoise", str(tmp)], check=True)
        out = np.load(tmp / "out.npy")
        sigma = np.load(tmp / "sigma.npy") if (tmp / "sigma.npy").exists() else None
        info = json.loads((tmp / "info.json").read_text())
    return Denoised(out, sigma, info)


def summary(raw, d: Denoised, bval, mask=None):
    """What the run's stats record: the method and settings, and over the bright voxels (the raw mean b0 above its
    Otsu threshold, as pipeline.residual_left's) the RMS of what was removed (all volumes, the b0s, the DW volumes),
    its mean (a bias), and MP-PCA's sigma (quartiles; the share it could not estimate). mask: the same over these
    voxels too (a brain mask known on the grid, e.g. the tracking mask)."""
    from .mask import otsu
    raw = np.asarray(raw, np.float32)
    b0 = np.asarray(bval) < 50
    mean = raw[..., b0].mean(-1) if b0.any() else raw.mean(-1)
    out = dict(d.info)
    rms = lambda x: round(float(np.sqrt((x ** 2).mean())), 3)
    for name, where in (("bright voxels", mean > otsu(mean)), ("the mask", None if mask is None else np.asarray(mask, bool))):
        if where is None:
            continue
        removed = (raw[where] - d.dwi[where]).astype(np.float64)
        out[f"{name}: voxels"] = int(where.sum())
        out[f"{name}: removed RMS, all / b0s / DW volumes"] = [rms(removed), rms(removed[:, b0]) if b0.any() else None,
                                                              rms(removed[:, ~b0]) if (~b0).any() else None]
        out[f"{name}: removed mean (a bias)"] = round(float(removed.mean()), 4)
        if d.sigma is not None:
            sg = d.sigma[where]
            out[f"{name}: sigma (quartiles)"] = [round(float(x), 3) for x in np.nanquantile(sg, [0.25, 0.5, 0.75])]
            out[f"{name}: sigma not estimated (no noise edge found: the patch left as it was)"] = round(float(np.isnan(sg).mean()), 4)
    return out


# ------------------------------------------------------------------------------------------------ the subprocess

def _slabs(n, parts, r):
    """Split axis length n into `parts` ranges [a, b); per range the data [a - 2r, b + 2r) its patches read and the
    centers [a - r, b + r) (dipy's loop keeps the centers r from the data's edges), clipped to [0, n)."""
    edges = np.linspace(0, n, max(1, min(parts, n)) + 1).round().astype(int)
    return [(int(a), int(b), max(a - 2 * r, 0), min(b + 2 * r, n)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _mppca_slab(path, axis, a, b, lo, hi, r):
    """One slab through dipy's mppca: the data [lo, hi) along `axis`, centers masked to [a - r, b + r) (dipy's own
    loop range then trims them to r from the slab's edges, which are the volume's own at its ends); the output's
    [a, b) back."""
    import warnings
    from dipy.denoise.localpca import mppca
    # dipy's noise classifier takes the mean of an empty slice where it finds no noise edge (sigma NaN, the patch left
    # as it was); summary() counts those voxels
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
    arr = np.load(path, mmap_mode="r")
    sl = [slice(None)] * 3
    sl[axis] = slice(lo, hi)
    slab = np.ascontiguousarray(arr[tuple(sl)])
    mask = np.zeros(slab.shape[:3], bool)
    m = [slice(None)] * 3
    m[axis] = slice(max(a - r, 0) - lo, min(b + r, arr.shape[axis]) - lo)
    mask[tuple(m)] = True
    den, sigma = mppca(slab, mask=mask, patch_radius=r, return_sigma=True, suppress_warning=True)
    k = [slice(None)] * 3
    k[axis] = slice(a - lo, b - lo)
    return a, b, np.asarray(den[tuple(k)], np.float32), np.asarray(sigma[tuple(k)], np.float32)


def _main(tmp: Path):
    args = json.loads((tmp / "args.json").read_text())
    t0 = time.time()
    method = args["method"]
    if method == "mppca":
        r = args["patch_radius"]
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = "1"                                            # the slabs' processes: one thread each
        arr = np.load(tmp / "dwi.npy", mmap_mode="r")
        axis = int(np.argmax(arr.shape[:3]))
        slabs = _slabs(arr.shape[axis], args["workers"], r)
        out = np.empty(arr.shape, np.float32)
        sigma = np.empty(arr.shape[:3], np.float32)
        with ProcessPoolExecutor(min(args["workers"], len(slabs)), mp_context=mp.get_context("spawn")) as ex:
            for a, b, den, sg in ex.map(_mppca_slab, *zip(*[(str(tmp / "dwi.npy"), axis, a, b, lo, hi, r) for a, b, lo, hi in slabs])):
                sl = [slice(None)] * 3
                sl[axis] = slice(a, b)
                out[tuple(sl)], sigma[tuple(sl)] = den, sg
        np.save(tmp / "sigma.npy", sigma)
        info = {"method": "mppca", "patch_radius": r, "pca_method": "eig", "mask": None, "slabs": len(slabs),
                "slab axis": axis, "volumes": int(arr.shape[-1])}
    else:
        os.environ["TQDM_DISABLE"] = "1"                                   # its progress bar, before tqdm loads
        from dipy.denoise.patch2self import patch2self
        arr = np.load(tmp / "dwi.npy")
        bval = np.load(tmp / "bval.npy")
        settings = dict(patch_radius=(0, 0, 0), model="ols", b0_threshold=50, b0_denoising=True, clip_negative_vals=True,
                        shift_intensity=False, version=3)
        np.random.seed(args["sketch_seed"])
        out = np.asarray(patch2self(arr, bval, tmp_dir=str(tmp), **settings), np.float32)
        info = {"method": "patch2self", **{k: (list(v) if isinstance(v, tuple) else v) for k, v in settings.items()},
                "sketch seed": args["sketch_seed"], "volumes": int(arr.shape[-1])}
    import dipy
    info.update({"dipy": dipy.__version__, "seconds": round(time.time() - t0, 1)})
    np.save(tmp / "out.npy", out)
    (tmp / "info.json").write_text(json.dumps(info))


if __name__ == "__main__":
    _main(Path(sys.argv[1]))
