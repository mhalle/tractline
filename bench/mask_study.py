"""Does the brain mask cost the brainstem and cerebellar tracts? (NOTES 2026-10-27, "The mask and the brainstem").

    uv run python bench/mask_study.py        # -> results/mask_study.json

Masks ("arms"), each used for seeding and for stopping unless said:
  A   today's: median_otsu of the mean b0 (tractline.mask; median radius 4, 4 passes);
  A2  median_otsu with a lighter filter (radius 2, 1 pass), coverage only: is the filter what loses the brainstem?
  C   seeds in A, stopping only where the b0 falls below a tenth of A's median b0 (no brain mask for stopping);
  C2  A grown, voxel by voxel (6-connected, up to 10 mm), through voxels whose median-filtered mean diffusivity (b ~1000
      shell) is 0.3-1.1 um^2/ms and whose b0 is above noise (a twentieth of A's median): brain-like by diffusion, which
      the coil's intensity falloff does not change;
  D1  SynthStrip on the mean b0 (tractline.skullstrip), border 1 mm (its default);
  D3  the same, border 3 mm (grown 2 mm).
1. Tracking, ON-Harmony sub-16974 OXF1PRI001 (Head_32 coil), uncorrected, the default shell: seeds, streamlines,
   labeled, Other, the UKF's time (Metal), and the streamlines RapidParc gives the tracts the mask could cut
   (CST, CPC, MCP, ICP, the two intra-cerebellar classes, PLIC) and two it should not (AF, CC4).
2. Coverage, ON-Harmony (five subjects, both coils; no labels): white-matter-like voxels each mask leaves out - FA >
   0.35, MD 0.4-1.0 um^2/ms (muscle's MD is ~1.4-1.6, CSF's ~3), b0 above a fifth of A's median, within 20 mm of A -
   in ml. A yardstick that needs no segmentation; skullstrip_check.py has the FastSurfer comparison.
"""
import json, time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

from tractline import bids, pipeline as P, skullstrip as K, ukf as U
from tractline.data import DATA
from tractline.fetch import weights
from tractline.labelers.base import TRACT_NAMES
from tractline.mask import median_otsu, multi_median
from tractline.prep import prepare

HERE = Path(__file__).resolve().parent
ONH = DATA / "ds004712"
WATCH = ["CST", "CPC", "MCP", "ICP", "Intra-CBLM-I-P", "Intra-CBLM-PaT", "PLIC", "AF", "CC4"]


def dti(d, bval, bvec):
    """FA and MD (um^2/ms) by log-linear least squares on the b0s and the b ~1000 shell."""
    sel = (bval < 50) | (np.abs(bval - 1000) < 100)
    S = np.clip(d[..., sel], 1, None).reshape(-1, sel.sum())
    b, g = bval[sel], bvec[:, sel].T
    B = np.c_[np.ones(len(b)), -b[:, None] * np.c_[g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2,
                                                    2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]]]
    x = np.linalg.lstsq(B, np.log(S).T, rcond=None)[0].T
    ev = np.linalg.eigvalsh(x[:, [1, 4, 5, 4, 2, 6, 5, 6, 3]].reshape(-1, 3, 3))
    md = ev.mean(1)
    fa = np.sqrt(1.5 * ((ev - md[:, None]) ** 2).sum(1) / np.maximum((ev ** 2).sum(1), 1e-12))
    return fa.reshape(d.shape[:3]), (md * 1000).reshape(d.shape[:3])


def grown(b0, md, core, zooms, md_lo=0.3, md_hi=1.1, max_mm=10.0):
    mds = ndi.median_filter(md, 3)
    ok = (mds > md_lo) & (mds < md_hi) & (b0 > 0.05 * np.median(b0[core]))
    st = ndi.generate_binary_structure(3, 1)
    m = core.copy()
    for _ in range(int(round(max_mm / min(zooms)))):
        n = m | (ndi.binary_dilation(m, st) & ok)
        if n.sum() == m.sum():
            break
        m = n
    return m


def signal_mask(b0, A):
    sig = ndi.binary_fill_holes(multi_median(b0, 1, 1) > 0.1 * np.median(b0[A]))
    lab, _ = ndi.label(sig)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def masks(b0, md, affine, device):
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))
    A = median_otsu(b0, 4, 4, device=device)[1]
    sdt = K.signed_distance(b0, affine, weights("synthstrip"), device)
    return A, {"A": A, "C2": grown(b0, md, A, zooms), "D1": sdt < 1, "D3": sdt < 3}


def tracking(device="mps"):
    p = ONH / "sub-16974/ses-OXF1PRI001/dwi/sub-16974_ses-OXF1PRI001_dir-AP_dwi.nii.gz"
    s = bids.load(p)
    dwi = np.asarray(s.dwi, np.float32)
    b0 = dwi[..., s.bval < 50].mean(-1)
    _, md = dti(dwi, s.bval, s.bvec)
    A, M = masks(b0, md, s.affine, device)
    arms = {"A": (A, A), "C": (A, signal_mask(b0, A)), **{k: (M[k], M[k]) for k in ("C2", "D1", "D3")}}
    shell = P.default_shell(s.bval)
    t = prepare(dwi, s.affine, s.bval, s.bvec, shell=shell, device=device, mask=A)
    labeler = P.default_labeler(device)
    vox = float(abs(np.linalg.det(s.affine[:3, :3]))) / 1000
    out = {"scan": "ON-Harmony sub-16974 OXF1PRI001, uncorrected", "shell": shell}
    for name, (seed_m, stop_m) in arms.items():
        pts = U.seeds(U.from_arrays(t.dwi, t.header, seed_m))[0]
        D = U.from_arrays(t.dwi, t.header, stop_m)
        t0 = time.perf_counter()
        fibers, _ = U.track(D, backend="metal", batch=P.GPU_BATCH, seed_points=pts.cpu().numpy())
        dt = time.perf_counter() - t0
        L = labeler(fibers, draws=(0,), logp=False)
        tr = np.asarray(L.tract)
        out[name] = {"stopping mask ml": round(float(stop_m.sum() * vox)), "seeds": int(len(pts)),
                     "streamlines": len(fibers), "labeled": int(L.keep.sum()), "Other": int((tr == 42).sum()),
                     "ukf s": round(dt, 1), **{k: int((tr == TRACT_NAMES.index(k)).sum()) for k in WATCH}}
        print(name, json.dumps(out[name]), flush=True)
    return out


def coverage(device="mps"):
    out = {}
    for sub in ("03286", "10975", "14221", "15320", "16974"):
        for ses in ("OXF1PRI001", "OXF2PRI001"):
            p = ONH / f"sub-{sub}/ses-{ses}/dwi/sub-{sub}_ses-{ses}_dir-AP_dwi.nii.gz"
            img = nib.load(p)
            d = np.asarray(img.dataobj, np.float32)
            bval, bvec = np.loadtxt(str(p)[:-7] + ".bval"), np.loadtxt(str(p)[:-7] + ".bvec")
            b0 = d[..., bval < 50].mean(-1)
            fa, md = dti(d, bval, bvec)
            zooms = img.header.get_zooms()[:3]
            vox = float(np.prod(zooms)) / 1000
            A, M = masks(b0, md, img.affine, device)
            near = ndi.distance_transform_edt(~A, sampling=zooms) <= 20
            wm = (fa > 0.35) & (md > 0.4) & (md < 1.0) & (b0 > 0.2 * np.median(b0[A])) & near
            M = {"A": M["A"], "A2": median_otsu(b0, 2, 1, device=device)[1], **{k: M[k] for k in ("C2", "D1", "D3")}}
            out[f"{sub} {ses}"] = {k: {"ml": round(float(m.sum() * vox)),
                                       "WM-like left out, ml": round(float((wm & ~m).sum() * vox), 1)}
                                   for k, m in M.items()}
            print(sub, ses, json.dumps(out[f"{sub} {ses}"]), flush=True)
    return out


def main():
    res = {"tracking": tracking(), "coverage (ON-Harmony)": coverage()}
    (HERE / "results" / "mask_study.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
