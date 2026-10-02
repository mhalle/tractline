"""Wild bootstrap of a single-shell DWI: replicates the scanner could have given, at the scan's own noise.

Per voxel, the diffusion-weighted volumes are fitted with real symmetric spherical harmonics (DIPY's
descoteaux basis, order `sh_order`) and the b0 volumes with their mean. A replicate is the fit plus
each residual, scaled by 1/sqrt(1 - leverage) and multiplied by an independent random sign
(Rademacher), per voxel and volume (the wild bootstrap; Whitcher et al. 2008, Jones 2008). Values are
rounded and clipped to the stored type, as the scanner stores them. The residuals hold whatever the
smooth fit does not explain - mostly noise, some model error - so `noise_check` compares them with the
spread of the repeated b0s, which no model touches.
"""
import re
import numpy as np


def read(nhdr: str):
    """The DWI as stored (i, j, k, G), its gradients (G, 3) and the b0 flags (UKF's rule: bmax |g|^2 <= 50)."""
    import nrrd
    raw, h = nrrd.read(nhdr)
    keys = sorted(k for k in h if k.startswith("DWMRI_gradient_"))
    g = np.array([[float(v) for v in h[k].split()] for k in keys])
    bmax = int(re.match(r"\s*(-?\d+)", h["DWMRI_b-value"]).group(1))
    return raw, g, (bmax * (g * g).sum(1)) <= 50


class WildBootstrap:
    def __init__(self, raw, g, is_b0, sh_order=6):
        from dipy.reconst.shm import real_sh_descoteaux
        self.raw, self.dtype = raw, raw.dtype
        self.b0, self.dw = np.flatnonzero(is_b0), np.flatnonzero(~is_b0)
        u = g[self.dw] / np.linalg.norm(g[self.dw], axis=1)[:, None]
        theta, phi = np.arccos(np.clip(u[:, 2], -1, 1)), np.arctan2(u[:, 1], u[:, 0])
        B, *_ = real_sh_descoteaux(sh_order, theta, phi, legacy=False)
        pinv = np.linalg.pinv(B)
        lev = np.einsum("ij,ji->i", B, pinv)                         # diag of the hat matrix B pinv(B)
        S = raw.reshape(-1, raw.shape[-1]).astype(np.float32)
        dw = S[:, self.dw]
        fit_dw = (dw @ pinv.T.astype(np.float32)) @ B.T.astype(np.float32)
        b0 = S[:, self.b0]
        fit_b0 = np.repeat(b0.mean(1, keepdims=True), len(self.b0), 1)
        self.fit = np.empty_like(S)
        self.fit[:, self.dw], self.fit[:, self.b0] = fit_dw, fit_b0
        self.res = np.empty_like(S)
        self.res[:, self.dw] = (dw - fit_dw) / np.sqrt(1 - lev).astype(np.float32)
        self.res[:, self.b0] = (b0 - fit_b0) / np.float32(np.sqrt(1 - 1 / len(self.b0)))
        self.sh_order, self.n_coef = sh_order, B.shape[1]

    def replicate(self, seed: int):
        rng = np.random.default_rng(seed)
        sign = rng.integers(0, 2, self.res.shape, dtype=np.int8) * 2 - 1
        x = np.rint(self.fit + self.res * sign)
        info = np.iinfo(self.dtype) if np.issubdtype(self.dtype, np.integer) else None
        x = np.clip(x, max(info.min, 0), info.max) if info is not None else np.clip(x, 0, None)   # a float input (a corrected DWI): no negative signal either
        return x.astype(self.dtype).reshape(self.raw.shape)

    def noise_check(self, mask) -> dict:
        """In the mask: the residual RMS of the DW fit, against the b0s' own spread (the noise SD, from
        repeats), both after the leverage correction; and the b0 SNR."""
        m = mask.reshape(-1) > 0
        sd_b0 = float(np.sqrt((self.res[m][:, self.b0] ** 2).mean()))
        rms_dw = float(np.sqrt((self.res[m][:, self.dw] ** 2).mean()))
        return {"sh_order": self.sh_order, "sh_coefficients": self.n_coef, "dw_volumes": len(self.dw),
                "b0_volumes": len(self.b0), "b0_noise_sd": round(sd_b0, 2), "dw_residual_rms": round(rms_dw, 2),
                "dw_residual_over_b0_sd": round(rms_dw / sd_b0, 3),
                "b0_snr": round(float(self.fit[m][:, self.b0[0]].mean()) / sd_b0, 1)}
