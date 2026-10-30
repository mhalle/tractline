"""An automatic stopping_fa for track:ptt's stopping "fa" (NOTES 2026-10-29, "Stopping at low FA, set from each scan"):
the best threshold sits just above the FA of cortex (0.15 where cortex's median FA is 0.12-0.13, 0.20 on a noisier
Prisma scan where it is 0.17: ptt_fa_anatomy), so an estimate of cortex's FA from the DWI alone sets it. This script
compares estimators with FastSurfer's cortex on the scans that have labels, and measures their repeatability.

    uv run python bench/ptt_fa_auto.py [--prisma <dwi> <labels.mgz> <t1_to_dwi.npz> <out folder>]
                                                   # -> results/ptt_fa_auto.json (the Prisma rows: <out folder>)
    uv run python bench/ptt_fa_auto.py --scan <dwi> <labels.mgz> <t1_to_dwi.npz> <tag> <out folder>
                                                   # one scan only -> <out folder>/ptt_fa_auto_<tag>.json

Estimators (DTI on the tracked shell with the b0s, over the brain mask):
  mask mode      the mode of FA below 0.3 over the mask (a histogram, 0.005 wide, smoothed by a 0.01 Gaussian);
  tissue mode    the same over the mask's voxels with mean diffusivity below 1.5e-3 mm^2/s (CSF and its partial volume
                 out; gray and white matter in);
  tissue p20     the 20th percentile of those voxels' FA;
  low-shell mode the mode of FA below 0.3 (on the tracked shell) over the voxels whose mean diffusivity on the lowest
                 shell is below 1.0, 1.1 or 1.2e-3 mm^2/s: at b = 2000-3000 the sulci's fluid and cortex are not
                 separable by diffusivity, at the lowest shell they are (class medians measured interactively, not
                 recorded: the Prisma scan's sulcal fluid 1.23 against cortex's 0.74 x 1e-3 at b = 1500);
  relative mode  the same with the diffusivity bound relative to the white matter's on that shell (the median MD of its
                 voxels with FA > 0.4 there), 1.4, 1.5 or 1.6 times it: a bound that follows the lowest shell's b.
Truth (where FastSurfer labels exist): the median FA of the voxels FastSurfer calls cortex.
Scans: PAT16 and cIIIs04 (cst_thalamus_outside.subjects) and, given, the Prisma scan, each with median_otsu and the
no-CSF SynthStrip mask; cIIIs04's four scans (A and its rescans, robust_benchmark's work files: the median_otsu mask on
A's grid) for repeatability, and the same four loaded in their own space with every shell (the low-shell
estimator); ON-Harmony 16974 and the ISMRM 2015 phantom (its T1's gray class: 0.114, step 6) for range.
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from tractline import bids, csd as C, pipeline as P
from tractline.data import DATA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cst_thalamus_outside import nearest_3000, subjects  # noqa: E402
from ensemble_endpoints import Anatomy  # noqa: E402
from ptt_fa_anatomy import MgzAnatomy, Tissue, CLASSES  # noqa: E402

MD_TISSUE = 1.5e-3
MASKS = ("median-otsu", "synthstrip-nocsf")
DWI_16974 = DATA / "ds004712/sub-16974/ses-OXF1PRI001/dwi/sub-16974_ses-OXF1PRI001_dir-AP_dwi.nii.gz"


def mode_below(x, hi=0.3, width=0.005, smooth=0.01):
    x = x[x < hi]
    h, e = np.histogram(x, bins=int(round(hi / width)), range=(0, hi))
    h = gaussian_filter1d(h.astype(float), smooth / width)
    return float((e[:-1] + width / 2)[np.argmax(h)])


def estimators(fa, md):
    tissue = md < MD_TISSUE
    return {"mask mode": round(mode_below(fa), 4), "tissue mode": round(mode_below(fa[tissue]), 4),
            "tissue p20": round(float(np.percentile(fa[tissue], 20)), 4), "tissue share": round(float(tissue.mean()), 3)}


LOW_MD = (1.0e-3, 1.1e-3, 1.2e-3)
REL_MD = (1.4, 1.5, 1.6)                                              # relative mode: times the white matter's MD


def low_shell(bval):
    b = np.asarray(bval, float)
    from tractline.motion_eddy import shell_groups
    groups, nominal = shell_groups(b)
    return min(float(np.median(b[g])) for g, n in zip(groups, nominal) if n > 0)


def dti(dwi, bval, bvec, affine, mask, shell):
    d, b, g = P._shell_volumes(dwi, bval, bvec, shell)
    ev, _ = C.tensor_fit(d[mask].astype(np.float64), b, P._image_axes(g, affine))
    return C.fa(ev), ev.mean(1)


def auto_of(fa, md):
    """pipeline.auto_stopping_fa on one shell (its own MD), or its refusal."""
    try:
        v, cortex = P.auto_stopping_fa(fa, md, fa)
        return {"stopping_fa": v, "cortex FA estimate": cortex}
    except ValueError as e:
        return {"refused": str(e)}


def scan_rows(s, anat, to_t1):
    """Per mask: the estimators, and FastSurfer's cortex median FA at the same voxels."""
    corr = P.correct(s, P.Timer(), "mps")
    shell = nearest_3000(s.bval)
    tissue = Tissue(anat) if anat is not None else None
    out = {"shell": shell}
    for impl in MASKS:
        mask = P.brain_mask(corr.dwi, s.bval, P.Timer(), device="mps", impl=impl, affine=s.affine)
        fa, md = dti(corr.dwi, s.bval, s.bvec, s.affine, mask, shell)
        row = estimators(fa, md)
        lo = low_shell(s.bval)
        _, md_lo = dti(corr.dwi, s.bval, s.bvec, s.affine, mask, lo)
        row["lowest shell"] = lo
        for x in LOW_MD:
            row[f"low-shell mode, MD < {x * 1e3:.1f}e-3"] = round(mode_below(fa[md_lo < x]), 4)
        fa_lo, _ = dti(corr.dwi, s.bval, s.bvec, s.affine, mask, lo)
        wm_md = float(np.median(md_lo[fa_lo > 0.4]))
        row["lowest shell: white matter's MD (FA > 0.4)"] = round(wm_md * 1e3, 3)
        for r in REL_MD:
            row[f"relative mode, MD < {r:g} x white matter's"] = round(mode_below(fa[md_lo < r * wm_md]), 4)
        sh = P.auto_shell(s.bval)                                     # the pipeline's own choice of shell
        _, md_a = dti(corr.dwi, s.bval, s.bvec, s.affine, mask, sh)
        fa_a, _ = dti(corr.dwi, s.bval, s.bvec, s.affine, mask, sh)
        v, cortex = P.auto_stopping_fa(fa, md_a, fa_a)
        row["auto (pipeline.auto_stopping_fa)"] = {"stopping_fa": v, "cortex FA estimate": cortex, "auto shell": sh}
        if tissue is not None:
            ijk = np.argwhere(mask)
            world = ijk @ s.affine[:3, :3].T + s.affine[:3, 3]
            cls = tissue.classes(world @ to_t1[:3, :3].T + to_t1[:3, 3])
            row["FastSurfer cortex median FA"] = round(float(np.median(fa[cls == CLASSES.index("cortex")])), 4)
        out[impl] = row
        print(json.dumps(row), flush=True)
    return out


def main(prisma=None):
    if prisma is not None and len(prisma) != 4:
        sys.exit("usage: ptt_fa_auto.py [--prisma <dwi> <labels.mgz> <t1_to_dwi.npz> <out folder>]")
    res = {"MD tissue": MD_TISSUE}
    for name, s, to_t1 in subjects():
        res[name] = scan_rows(s, Anatomy(name), to_t1)
        print(name, flush=True)
    res["16974"] = scan_rows(bids.load(DWI_16974), None, None)
    import robust_benchmark as RB
    import _benchmark_dipy as BD
    rep = {}
    for scan in RB.CIII_SCANS:
        w = np.load(RB.work_file(scan)[0])
        mask = w["mask"].astype(bool)
        ev, _ = C.tensor_fit(w["dwi"][mask].astype(np.float64), w["bval"], BD.flip(w["affine"]) @ w["bvec"])
        rep[scan] = estimators(C.fa(ev), ev.mean(1))
        rep[scan]["auto (pipeline.auto_stopping_fa, single shell)"] = auto_of(C.fa(ev), ev.mean(1))
        print(scan, json.dumps(rep[scan]), flush=True)
    res["cIIIs04, four scans on A's grid (median_otsu)"] = rep
    from _masivar import load as load_masivar
    subject, (ses, sc), rescans, _, _ = RB.PAIRS[RB.PAIR]
    own = {}
    for scan, (se, k) in {"A": (ses, sc), **rescans}.items():
        own[scan] = scan_rows(load_masivar(subject, se, k), None, None)
        print(scan, flush=True)
    res["cIIIs04, four scans in their own space, every shell"] = own
    w = np.load(RB.work_file("phantom")[0])
    mask = w["mask"].astype(bool)
    ev, _ = C.tensor_fit(w["dwi"][mask].astype(np.float64), w["bval"], BD.flip(w["affine"]) @ w["bvec"])
    res["phantom"] = {**estimators(C.fa(ev), ev.mean(1)), "gray class median FA (step 6)": 0.114,
                      "auto (pipeline.auto_stopping_fa)": auto_of(C.fa(ev), ev.mean(1))}
    (HERE / "results" / "ptt_fa_auto.json").write_text(json.dumps(res, indent=1) + "\n")
    if prisma:
        dwi, labels, fit, out = prisma
        row = scan_rows(bids.load(dwi), MgzAnatomy(labels), np.linalg.inv(np.load(fit)["t1_to_dwi"]))
        (Path(out) / "ptt_fa_auto_prisma.json").write_text(json.dumps(row, indent=1) + "\n")


def one_scan(args):
    """--scan: the estimators and auto against FastSurfer's cortex on one scan (FastSurfer's .mgz label volume)."""
    if len(args) != 5:
        sys.exit("usage: ptt_fa_auto.py --scan <dwi> <labels.mgz> <t1_to_dwi.npz> <tag> <out folder>")
    dwi, labels, fit, tag, out = args
    row = scan_rows(bids.load(dwi), MgzAnatomy(labels), np.linalg.inv(np.load(fit)["t1_to_dwi"]))
    (Path(out) / f"ptt_fa_auto_{tag}.json").write_text(json.dumps(row, indent=1) + "\n")


if __name__ == "__main__":
    if "--scan" in sys.argv:
        one_scan(sys.argv[sys.argv.index("--scan") + 1:])
    else:
        main(sys.argv[sys.argv.index("--prisma") + 1:][:4] if "--prisma" in sys.argv else None)
