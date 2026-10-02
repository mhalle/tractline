"""Does distortion correction put PAT16's diffusion data where the T1 says the anatomy is? The T1
(MPRAGE, 1 mm, essentially undistorted) is the independent arbiter between the scan as acquired,
FSL topup's correction and ours (_susc.py) - none of which saw it.

    DATA/.venv/bin/python bench/tractography/t1_alignment.py

Per arm, from the mean of the AP DWI's 6 b0s:
  1. rigid start: the scanner's own coordinates (same session), refined rigidly with the cost below at
     FWHM 8 then 4 mm (dipy.align and torch load two OpenMP runtimes in one process; not needed);
  2. joint fit: rigid (6 parameters) + a residual displacement d (mm) along the phase-encoding axis,
     a cubic B-spline (knots 25 -> 20 -> 15 mm) with its mean over the brain held at zero (a uniform
     shift is the rigid fit's), the b0 resampled at x + d(x) e_pe times the Jacobian; the cost
     normalized gradient fields (1 - (n_b0 . n_T1)^2: contrast-free, so T1 against the T2-weighted b0
     works), plus a bending penalty, the same for every arm. FWHM 6 -> 4 -> 0 mm.
d is the distortion a rigid alignment to the T1 cannot remove: what an overlay of tracts on the T1
would be off by.
  - validation: the uncorrected arm's d against topup's own displacement (field x readout x voxel):
    if the T1 recovers topup's map independently, the measure measures distortion;
  - precision: the fit repeated on b0s 1-3 and 4-6 separately, |d_a - d_b| / 2 the error of one;
  - regions: inside the topup arm's brain mask, where topup displaces > 3 mm and elsewhere.

Writes results/t1_alignment.json and results/t1_alignment.png.
"""
import json, time
from pathlib import Path
import numpy as np, nibabel as nib, nrrd
from scipy.ndimage import binary_dilation
import torch
import _susc as S

HERE = Path(__file__).resolve().parent
TD = Path.home() / "tmp/data/tractography/ds001226"
SUB = TD / "sub-PAT16/ses-preop"
DER = TD / "derived"
ARMS = {"uncorrected": SUB / "dwi/sub-PAT16_ses-preop_acq-AP_dwi.nii.gz",
        "topup": DER / "PAT16/topup/dwi_AP_topup.nii.gz",
        "ours": DER / "PAT16/susc/dwi_AP_ours2.nii.gz"}
PE = 1                                                                    # j: AP acquisition, "j-"
dt = torch.float64
torch.set_num_threads(8)

t1img = nib.load(SUB / "anat/sub-PAT16_ses-preop_T1w.nii.gz")
t1 = np.asarray(t1img.dataobj, dtype=np.float64)
bval = np.loadtxt(SUB / "dwi/sub-PAT16_ses-preop_acq-AP_dwi.bval")
b0i = np.flatnonzero(bval < 50)
ref = nib.load(ARMS["uncorrected"])
A, shape = ref.affine, ref.shape[:3]
vox = np.asarray(ref.header.get_zooms()[:3], float)

topup_mask = nrrd.read(str(DER / "PAT16_topup/mask.nrrd"))[0].astype(bool)
fit_mask = binary_dilation(topup_mask, iterations=2)
for a in ("PAT16", "PAT16_ours2"):
    fit_mask |= binary_dilation(nrrd.read(str(DER / a / "mask.nrrd"))[0].astype(bool), iterations=2)
side = json.loads((SUB / "dwi/sub-PAT16_ses-preop_acq-AP_dwi.json").read_text())
h_topup = np.asarray(nib.load(DER / "PAT16/topup/field_hz.nii.gz").dataobj, dtype=np.float64)
d_topup = -1 * side["TotalReadoutTime"] * h_topup * vox[PE]               # mm, j-: the AP scan's displacement
big = topup_mask & (np.abs(d_topup) > 3)

# ------------------------------------------------------------------ the fit

T1s = torch.as_tensor(S.smooth(torch.as_tensor(t1[None]), 1.5, np.ones(3))[0].numpy())   # anti-alias for 2.5 mm
Ainv_t1 = torch.as_tensor(np.linalg.inv(t1img.affine), dtype=dt)
At = torch.as_tensor(A, dtype=dt)
grid = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=dt) for n in shape], indexing="ij"), -1)
world = grid @ At[:3, :3].T + At[:3, 3]                                   # (X, Y, Z, 3) mm
M = torch.as_tensor(fit_mask)
Bm = torch.as_tensor(topup_mask)


def sample(img, vx):
    """Trilinear samples of img (X, Y, Z) at voxel coordinates vx (..., 3), zero outside."""
    sizes = torch.tensor(img.shape, dtype=dt)
    g = (2 * vx / (sizes - 1) - 1)[..., [2, 1, 0]]
    return torch.nn.functional.grid_sample(img[None, None], g[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]


def t1_on_grid(T):
    """The T1 resampled on the b0 grid by T (4x4: b0 world -> T1 world)."""
    w = world @ T[:3, :3].T + T[:3, 3]
    return sample(T1s, w @ Ainv_t1[:3, :3].T + Ainv_t1[:3, 3])


def ngf(a, b, mask):
    def n(img):
        g = torch.stack(torch.gradient(img, spacing=tuple(vox)), -1)
        eta = g.norm(dim=-1)[mask].mean() * 0.1
        return g / torch.sqrt((g ** 2).sum(-1, keepdim=True) + eta ** 2)
    return (1 - ((n(a) * n(b)).sum(-1)) ** 2)[mask].mean()


def fit(b0, T0, lam=1.0):
    """Rigid delta (about the volume center) + residual PE displacement d (mm); returns (d, T, log)."""
    b0 = torch.as_tensor(b0, dtype=dt)
    center = torch.as_tensor(world.reshape(-1, 3).mean(0))
    p = torch.zeros(6, dtype=dt, requires_grad=True)
    c = None
    log = []
    for spacing, fwhm in ((25, 6), (20, 4), (15, 0)):
        Bs = [S.basis(n, spacing / v, "cpu", dt) for n, v in zip(shape, vox)]
        B, dB, d2B = zip(*Bs)
        if c is None:
            c = torch.zeros(*[b.shape[1] for b in B], dtype=dt)
        else:
            c = S.project(S.field(c, prevB), B).detach().contiguous()
        c.requires_grad_(True)
        bs = S.smooth(b0[None], fwhm, vox)[0]
        coef = S.prefilter(bs[None], PE)
        opt = torch.optim.LBFGS([p, c], lr=1, max_iter=60, history_size=20, line_search_fn="strong_wolfe")
        ps = torch.tensor([1, 1, 1, 0.01, 0.01, 0.01], dtype=dt)            # mm, rad

        def model():
            d = S.field(c, B)
            d = d - d[M].mean()
            R, t = S.rigid(p * ps, center)
            T = T0.clone(); T[:3, :3] = T0[:3, :3] @ R; T[:3, 3] = T0[:3, :3] @ t + T0[:3, 3]
            dd = torch.gradient(d, dim=PE)[0]
            warped = S.unwarp_pe_cubic(coef, d / vox[PE], dd / vox[PE], PE, torch.ones(1, dtype=dt))[0]
            ref_ = S.smooth(t1_on_grid(T)[None], fwhm, vox)[0]
            return d, T, ngf(warped, ref_, M), S.bending(c, B, dB, d2B, vox)

        def closure():
            opt.zero_grad()
            _, _, cost, be = model()
            loss = cost + lam * be
            loss.backward()
            return loss
        opt.step(closure)
        prevB = B
        d, T, cost, be = model()
        log.append({"knots_mm": spacing, "fwhm_mm": fwhm, "ngf": round(float(cost), 5), "bending": float(be)})
    return d.detach().numpy(), T.detach(), log


def rigid_start(b0):
    """Rigid only (no displacement), from the identity in scanner coordinates."""
    b0 = torch.as_tensor(b0, dtype=dt)
    center = torch.as_tensor(world.reshape(-1, 3).mean(0))
    ps = torch.tensor([1, 1, 1, 0.01, 0.01, 0.01], dtype=dt)
    p = torch.zeros(6, dtype=dt, requires_grad=True)
    for fwhm in (8, 4):
        bs = S.smooth(b0[None], fwhm, vox)[0]
        opt = torch.optim.LBFGS([p], lr=1, max_iter=60, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            R, t = S.rigid(p * ps, center)
            T = torch.eye(4, dtype=dt); T[:3, :3] = R; T[:3, 3] = t
            loss = ngf(bs, S.smooth(t1_on_grid(T)[None], fwhm, vox)[0], M)
            loss.backward()
            return loss
        opt.step(closure)
    R, t = S.rigid((p * ps).detach(), center)
    T = torch.eye(4, dtype=dt); T[:3, :3] = R; T[:3, 3] = t
    return T


def rigid_only_ngf(b0, T):
    return float(ngf(torch.as_tensor(b0, dtype=dt), t1_on_grid(T), M))


def stats(x, m):
    v = np.abs(x[m])
    return [round(float(np.median(v)), 2), round(float(np.quantile(v, 0.9)), 2), round(float(np.quantile(v, 0.99)), 2)]


res = {"data": "ds001226 PAT16: mean AP b0 of each arm against the T1w (MPRAGE 1 mm)",
       "regions": {"brain_voxels": int(topup_mask.sum()), "topup_displaces_gt_3mm_voxels": int(big.sum())},
       "columns": "|residual displacement| along PE, mm: median / 90th / 99th percentile", "arms": {}}
fields = {}
for name, path in ARMS.items():
    t0 = time.time()
    dwi = nib.load(path)
    b0s = np.asarray(dwi.dataobj, dtype=np.float64)[..., b0i]
    assert np.allclose(dwi.affine, A, atol=1e-3)
    b0 = b0s.mean(-1)
    T0 = rigid_start(b0)
    d, T, log = fit(b0, T0)
    da, _, _ = fit(b0s[..., :3].mean(-1), T0)
    db, _, _ = fit(b0s[..., 3:].mean(-1), T0)
    err = np.abs(da - db) / 2
    fields[name] = d
    res["arms"][name] = {
        "residual_brain": stats(d, topup_mask), "residual_where_topup_gt_3mm": stats(d, big), "residual_elsewhere": stats(d, topup_mask & ~big),
        "half_split_error_brain": stats(err, topup_mask), "half_split_error_where_topup_gt_3mm": stats(err, big),
        "ngf_rigid_only": round(rigid_only_ngf(b0, T0), 5), "ngf_after": log[-1]["ngf"],
        "r_residual_vs_topup_displacement": round(float(np.corrcoef(d[topup_mask], d_topup[topup_mask])[0, 1]), 3),
        "rigid_start_translation_mm": [round(float(v), 2) for v in T0[:3, 3]],
        "rigid_start_rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(T0[:3, :3].numpy()) - 1) / 2, -1, 1)))), 2),
        "seconds": round(time.time() - t0, 1), "levels": log}
    print(name, json.dumps({k: v for k, v in res["arms"][name].items() if k != "levels"}), flush=True)

u = fields["uncorrected"]
res["validation"] = {"topup_displacement_brain": stats(d_topup, topup_mask),
                     "uncorrected_residual_r_vs_topup": res["arms"]["uncorrected"]["r_residual_vs_topup_displacement"],
                     "uncorrected_residual_slope_vs_topup": round(float(np.polyfit(d_topup[topup_mask], u[topup_mask], 1)[0]), 3)}
(HERE / "results/t1_alignment.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(DER / "PAT16/t1_alignment_fields.npz", d_topup=d_topup, mask=topup_mask, **fields)
print(json.dumps(res["validation"]))
