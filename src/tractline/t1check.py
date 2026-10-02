"""The distortion a rigid alignment to the T1 cannot remove (t1_alignment.py's measure, for any scan).

  1. rigid start: the scanner's own coordinates (same session), refined rigidly with the cost below at
     FWHM 8 then 4 mm;
  2. joint fit: rigid (6 parameters) + a residual displacement d (mm) along the phase-encoding axis,
     a cubic B-spline (knots 25 -> 20 -> 15 mm) with its mean over the fit mask held at zero (a
     uniform shift is the rigid fit's), the b0 resampled at x + d(x) e_pe times the Jacobian; the cost
     normalized gradient fields (1 - (n_b0 . n_T1)^2: contrast-free, so T1 against the T2-weighted
     b0 works), plus a bending penalty. FWHM 6 -> 4 -> 0 mm.
d is what an overlay of tracts on the T1 would be off by. CPU, float64 (dipy.align and torch load two
OpenMP runtimes in one process, so no dipy here).
"""
from __future__ import annotations

import numpy as np, nibabel as nib, torch
from scipy.ndimage import affine_transform, distance_transform_edt
from . import susceptibility as S

dt = torch.float64


def stats(x, m):
    """|x| over mask m: median, 90th and 99th percentile."""
    v = np.abs(x[m])
    return [round(float(np.median(v)), 2), round(float(np.quantile(v, 0.9)), 2), round(float(np.quantile(v, 0.99)), 2)]


def tumor_regions(t1img, mask_img, margin_mm=10):
    """The tumor (mask > 0.5, read by its own header: ds001226's arrays are stored left-right flipped
    against the T1's, with headers to match) and a margin_mm shell around it, both on the T1 grid."""
    Mt = np.linalg.inv(mask_img.affine) @ t1img.affine
    tumor = affine_transform(np.asarray(mask_img.dataobj, float), Mt[:3, :3], Mt[:3, 3], order=1) > 0.5
    dist = distance_transform_edt(~tumor, sampling=t1img.header.get_zooms()[:3])
    return tumor, (dist > 0) & (dist <= margin_mm)


class T1Check:
    def __init__(self, t1img, affine, shape, fit_mask, pe_axis, vox):
        """t1img: the T1 (nibabel); affine, shape: the b0 grid; fit_mask: where the cost is taken (the
        brain, dilated: its edge carries the signal); pe_axis: the b0 grid's phase-encoding axis; vox:
        its voxel sizes as the header states them (the affine's column norms differ by ~1e-8, enough
        to move L-BFGS's path by up to 0.05 mm in the result)."""
        t1 = np.asarray(t1img.dataobj, dtype=np.float64)
        self.t1 = t1
        self.T1s = torch.as_tensor(S.smooth(torch.as_tensor(t1[None]), 1.5, np.ones(3))[0].numpy())   # anti-alias for 2.5 mm
        self.Ainv_t1 = torch.as_tensor(np.linalg.inv(t1img.affine), dtype=dt)
        self.shape, self.pe = tuple(shape), pe_axis
        self.vox = np.asarray(vox, float)
        At = torch.as_tensor(affine, dtype=dt)
        grid = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=dt) for n in shape], indexing="ij"), -1)
        self.world = grid @ At[:3, :3].T + At[:3, 3]                       # (X, Y, Z, 3) mm
        self.center = self.world.reshape(-1, 3).mean(0)
        self.M = torch.as_tensor(fit_mask)

    def sample(self, img, vx):
        """Trilinear samples of img (X, Y, Z) at voxel coordinates vx (..., 3), zero outside."""
        sizes = torch.tensor(img.shape, dtype=dt)
        g = (2 * vx / (sizes - 1) - 1)[..., [2, 1, 0]]
        return torch.nn.functional.grid_sample(img[None, None], g[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]

    def on_grid(self, T, img=None):
        """The T1 (or another image on its grid) resampled on the b0 grid by T (4x4: b0 world -> T1 world)."""
        w = self.world @ T[:3, :3].T + T[:3, 3]
        return self.sample(self.T1s if img is None else torch.as_tensor(img, dtype=dt), w @ self.Ainv_t1[:3, :3].T + self.Ainv_t1[:3, 3])

    def ngf(self, a, b):
        def n(img):
            g = torch.stack(torch.gradient(img, spacing=tuple(self.vox)), -1)
            eta = g.norm(dim=-1)[self.M].mean() * 0.1
            return g / torch.sqrt((g ** 2).sum(-1, keepdim=True) + eta ** 2)
        return (1 - ((n(a) * n(b)).sum(-1)) ** 2)[self.M].mean()

    def rigid_start(self, b0):
        """Rigid only (no displacement), from the identity in scanner coordinates."""
        b0 = torch.as_tensor(b0, dtype=dt)
        ps = torch.tensor([1, 1, 1, 0.01, 0.01, 0.01], dtype=dt)
        p = torch.zeros(6, dtype=dt, requires_grad=True)
        for fwhm in (8, 4):
            bs = S.smooth(b0[None], fwhm, self.vox)[0]
            opt = torch.optim.LBFGS([p], lr=1, max_iter=60, history_size=20, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                R, t = S.rigid(p * ps, self.center)
                T = torch.eye(4, dtype=dt); T[:3, :3] = R; T[:3, 3] = t
                loss = self.ngf(bs, S.smooth(self.on_grid(T)[None], fwhm, self.vox)[0])
                loss.backward()
                return loss
            opt.step(closure)
        R, t = S.rigid((p * ps).detach(), self.center)
        T = torch.eye(4, dtype=dt); T[:3, :3] = R; T[:3, 3] = t
        return T

    def fit(self, b0, T0, lam=1.0):
        """Rigid delta (about the volume center) + residual PE displacement d (mm); returns (d, T, log)."""
        b0 = torch.as_tensor(b0, dtype=dt)
        vox, PE, M = self.vox, self.pe, self.M
        p = torch.zeros(6, dtype=dt, requires_grad=True)
        c = prevB = None
        log = []
        for spacing, fwhm in ((25, 6), (20, 4), (15, 0)):
            Bs = [S.basis(n, spacing / v, "cpu", dt) for n, v in zip(self.shape, vox)]
            B, dB, d2B = zip(*Bs)
            if c is None:
                c = torch.zeros(*[b.shape[1] for b in B], dtype=dt)
            else:
                c = S.project(S.field(c, prevB), B).detach().contiguous()
            c.requires_grad_(True)
            bs = S.smooth(b0[None], fwhm, vox)[0]
            coef = S.prefilter(bs[None], PE)
            opt = torch.optim.LBFGS([p, c], lr=1, max_iter=60, history_size=20, line_search_fn="strong_wolfe")
            ps = torch.tensor([1, 1, 1, 0.01, 0.01, 0.01], dtype=dt)        # mm, rad

            def model():
                d = S.field(c, B)
                d = d - d[M].mean()
                R, t = S.rigid(p * ps, self.center)
                T = T0.clone(); T[:3, :3] = T0[:3, :3] @ R; T[:3, 3] = T0[:3, :3] @ t + T0[:3, 3]
                dd = torch.gradient(d, dim=PE)[0]
                warped = S.unwarp_pe_cubic(coef, d / vox[PE], dd / vox[PE], PE, torch.ones(1, dtype=dt))[0]
                ref_ = S.smooth(self.on_grid(T)[None], fwhm, vox)[0]
                return d, T, self.ngf(warped, ref_), S.bending(c, B, dB, d2B, vox)

            def closure():
                opt.zero_grad()
                _, _, cost, be = model()
                loss = cost + lam * be
                loss.backward()
                return loss
            opt.step(closure)
            prevB = B
            with torch.no_grad():
                d, T, cost, be = model()
            log.append({"knots_mm": spacing, "fwhm_mm": fwhm, "ngf": round(float(cost), 5), "bending": float(be)})
        return d.detach().numpy(), T.detach(), log

    def rigid_only_ngf(self, b0, T):
        return float(self.ngf(torch.as_tensor(b0, dtype=dt), self.on_grid(T)))

    @staticmethod
    def rigid_summary(T0):
        return {"rigid_start_translation_mm": [round(float(v), 2) for v in T0[:3, 3]],
                "rigid_start_rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(T0[:3, :3].numpy()) - 1) / 2, -1, 1)))), 2)}
