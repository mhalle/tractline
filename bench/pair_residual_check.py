"""A residual check before correcting (the rule albula-diffusion's planning.ts applies to its own fit): fit
the field to a supposed reversed pair, correct each series' mean b0 with it, and see how much of the two
series' difference is left; refuse the pair when too much is. Does it tell good pairs from bad ones?
Our estimate (the pipeline's defaults, "mps"); in the brain (median_otsu of the first series' mean b0):
    left = ||corrected mean A - corrected mean B|| / ||mean A - mean B||
(pipeline.residual_left, as the pipeline reports it: the field applied with each series' own polarity, as
labeled, the volumes scaled to a common mean, over the mean b0's bright voxels; motion is not applied, so
it counts as left),
and the same with motion: every b0 unwarped as the estimate's model does it - its own fitted rigid motion,
the field, the Jacobian (susceptibility.unwarp, trilinear, full resolution, the volumes scaled to a common
mean as the estimate scales them) - the series' means compared:
    left_with_motion = ||mean unwarped A - mean unwarped B|| / ||mean A - mean B||

ds001226 (12 patients): the AP/PA pair (good); the AP b0s in two halves labeled opposite (bad: one polarity).
ds005123 (the subjects fetch_ds005123.py fetched): the AP and PA field maps (good: one shim); the diffusion
b0s with the PA field map (reversed, but shimmed differently from the diffusion series: the field is not
the diffusion series'); the diffusion b0s with the same-polarity AP field map labeled opposite (bad).

    uv run bench/pair_residual_check.py [--dataset ds001226|ds005123|both]

Writes results/pair_residual_check.json.
"""
import argparse, json
from pathlib import Path
import numpy as np, nibabel as nib, torch
from tractline import susceptibility as S, pipeline as P
from tractline.data import DATA
from tractline.mask import median_otsu

HERE = Path(__file__).resolve().parent
VEC = {"j": (0, 1, 0), "j-": (0, -1, 0), "i": (1, 0, 0), "i-": (-1, 0, 0)}


def check(A, B, pe_a, pe_b, vox, ro, brain):
    """Fit A (X,Y,Z,n) and B with phase-encoding vectors pe_a, pe_b (one each); the fraction of the mean
    difference left after correction, and the displacement's 99th percentile (mm) for A."""
    b0s = np.concatenate([A, B], -1)
    pev = np.array([pe_a] * A.shape[-1] + [pe_b] * B.shape[-1], float)
    h, motion, _ = S.estimate(b0s, vox, pev, ro, device="mps")
    ax = int(np.argmax(np.abs(pe_a)))
    sa = float(np.sign(pe_a[ax]))
    from types import SimpleNamespace
    left = P.residual_left(SimpleNamespace(b0s=b0s, pe_vectors=pev, pe_axis=ax, pe_sign=sa, readout_s=ro, b0_readout_s=None), h)
    disp = np.abs(S.displacement_mm(h, ro, sa, vox[ax]))[brain]
    # with motion: the estimate's model at full resolution
    dt = torch.float64
    img = torch.as_tensor(np.moveaxis(b0s, -1, 0), dtype=dt)
    img = img * (img.mean() / img.mean(dim=(1, 2, 3), keepdim=True))
    vox_t = torch.as_tensor(np.asarray(vox, float), dtype=dt)
    center = (torch.tensor(b0s.shape[:3], dtype=dt) - 1) / 2 * vox_t
    Rt = [S.rigid(torch.as_tensor(m, dtype=dt), center) for m in motion]
    ht = torch.as_tensor(h, dtype=dt)
    dh = torch.as_tensor(np.gradient(h, axis=ax), dtype=dt)
    scale = torch.as_tensor(pev[:, ax] * ro, dtype=dt)
    u = S.unwarp(img, ht, dh, ax, scale, torch.stack([R for R, _ in Rt]), torch.stack([t_ for _, t_ in Rt]), vox_t).numpy()
    na = A.shape[-1]
    raw = img.numpy()
    e0 = (raw[:na].mean(0) - raw[na:].mean(0))[brain]; e1 = (u[:na].mean(0) - u[na:].mean(0))[brain]
    P.release_memory("mps")
    return {"left": None if left is None else round(left, 3),                # None: not judged (noise only)
            "left_with_motion": round(float(np.linalg.norm(e1) / np.linalg.norm(e0)), 3),
            "motion_max_mm": round(float(np.abs(motion[:, :3]).max()), 2),
            "displacement_99th_mm": round(float(np.quantile(disp, 0.99)), 2)}


def ds001226():
    import sys; sys.path.insert(0, str(HERE))
    from _ds001226 import load
    out = {}
    for sub in "PAT05 PAT07 PAT08 PAT13 PAT14 PAT16 PAT19 PAT20 PAT23 PAT25 PAT26 PAT29".split():
        s = load(sub)
        a = [i for i, v in enumerate(s.pe_vectors) if np.allclose(v, s.pe_vectors[0])]
        b = [i for i in range(s.b0s.shape[-1]) if i not in a]
        A, B = s.b0s[..., a], s.b0s[..., b]
        _, brain = median_otsu(A.mean(-1).astype(np.float32), median_radius=4, numpass=4, device="mps")
        pe = s.pe_vectors[0]; h = len(a) // 2
        out[sub] = {"pair (AP + PA)": check(A, B, pe, s.pe_vectors[b[0]], s.vox, s.readout_s, brain),
                    "bad: AP halves labeled opposite": check(A[..., :h], A[..., h:], pe, -pe, s.vox, s.readout_s, brain)}
        print(sub, json.dumps(out[sub]), flush=True)
    return out


def ds005123():
    D = DATA / "ds005123"
    out = {}
    for sub in sorted(p.name for p in D.glob("sub-*")):
        L = lambda f: nib.load(D / sub / f)
        img = L(f"dwi/{sub}_desc-leadingb0_dwi.nii.gz")
        dwi = np.asarray(img.dataobj, np.float64)
        ap = np.asarray(L(f"fmap/{sub}_acq-dwi_dir-AP_epi.nii.gz").dataobj, np.float64)
        pa = np.asarray(L(f"fmap/{sub}_acq-dwi_dir-PA_epi.nii.gz").dataobj, np.float64)
        side = {k: json.loads((D / sub / f).read_text()) for k, f in (("dwi", f"dwi/{sub}_dwi.json"),
                ("ap", f"fmap/{sub}_acq-dwi_dir-AP_epi.json"), ("pa", f"fmap/{sub}_acq-dwi_dir-PA_epi.json"))}
        pe = {k: np.array(VEC[v["PhaseEncodingDirection"]], float) for k, v in side.items()}
        ro, vox = side["dwi"]["TotalReadoutTime"], np.array(img.header.get_zooms()[:3], float)
        _, brain = median_otsu(dwi.mean(-1).astype(np.float32), median_radius=4, numpass=4, device="mps")
        out[sub] = {"shim_dwi_equals_fieldmaps": side["dwi"].get("ShimSetting") == side["ap"].get("ShimSetting"),
                    "pair (AP + PA field maps)": check(ap, pa, pe["ap"], pe["pa"], vox, ro, brain),
                    "re-shimmed pair (dwi b0s + PA field map)": check(dwi, pa, pe["dwi"], pe["pa"], vox, ro, brain),
                    "bad: dwi b0s + AP field map labeled opposite": check(dwi, ap, pe["dwi"], -pe["ap"], vox, ro, brain)}
        print(sub, json.dumps(out[sub]), flush=True)
    return out


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--dataset", default="both", choices=["ds001226", "ds005123", "both"])
    args = ap_.parse_args()
    res = {}
    if args.dataset in ("ds005123", "both"):
        res["ds005123"] = ds005123()
    if args.dataset in ("ds001226", "both"):
        res["ds001226"] = ds001226()
    (HERE / "results/pair_residual_check.json").write_text(json.dumps(res, indent=1))
    for ds, subs in res.items():
        for c in [k for k, v in next(iter(subs.values())).items() if isinstance(v, dict)]:
            for key in ("left", "left_with_motion"):
                vals = [v[c][key] for v in subs.values()]
                xs = sorted(x for x in vals if x is not None)
                nj = sum(x is None for x in vals)
                print(ds, c, key, (f"{xs[0]:.2f}-{xs[-1]:.2f} median {xs[len(xs) // 2]:.2f}" if xs else "-")
                      + (f"; not judged {nj}" if nj else ""))
