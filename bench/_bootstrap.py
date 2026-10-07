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


def sh_order_for(directions: int, fraction=0.6) -> int:
    """The largest even SH order whose coefficients are at most `fraction` of a shell's directions (leaving
    the residuals degrees of freedom): 16 -> 2, 30 -> 4, 50 -> 6."""
    L = 0
    while (L + 3) * (L + 4) // 2 <= fraction * directions:
        L += 2
    return L


class ShellWildBootstrap:
    """The wild bootstrap of a multi-shell DWI as acquired (X, Y, Z, G), before any correction: each shell's
    volumes fitted with real symmetric spherical harmonics of the order its directions allow
    (sh_order_for), the b0s with their mean; residuals scaled by 1/sqrt(1 - leverage), signs flipped per
    voxel and volume. `extra_b0s` (X, Y, Z, V), when given - another series' b0s on the same grid (the
    reversed pair) - are replicated the same way about their own mean. A replicate's values are rounded and
    clipped as the input's type stores them (extra b0s: float, >= 0). The b0 residuals carry what moved
    between the b0s (motion, CSF pulsation) as well as noise: on PAT16 a b0 SD of 67 against 6.2 in the
    diffusion-weighted volumes (NOTES 2026-10-02), so the replicates perturb the field estimate by the
    scan's variability, not thermal noise alone."""
    def __init__(self, dwi, bval, bvec, extra_b0s=None, b0_max=50.0, shell_round=100.0, b0_noise="residual", mask=None):
        """b0_noise: "residual" - the b0s' own residuals (motion and pulsation in them); "thermal" - every b0
        residual (the extra b0s' too) scaled by one factor so their RMS in `mask` (required) is the DW
        volumes' residual RMS there: the spatial pattern kept, at the noise level the DW fit leaves."""
        from dipy.reconst.shm import real_sh_descoteaux
        assert b0_noise in ("residual", "thermal") and (b0_noise == "residual" or mask is not None)
        self.shape, self.dtype = dwi.shape, dwi.dtype
        S = np.asarray(dwi, np.float32).reshape(-1, dwi.shape[-1])
        self.fit, self.res = np.empty_like(S), np.empty_like(S)
        bval, bvec = np.asarray(bval, float), np.asarray(bvec, float)
        b0 = np.flatnonzero(bval < b0_max)
        self.shells = {}
        self._mean_fit(S, b0)
        shells = np.round(bval / shell_round) * shell_round
        for b in sorted(set(shells[bval >= b0_max].tolist())):
            idx = np.flatnonzero((shells == b) & (bval >= b0_max))
            u = bvec[:, idx].T / np.linalg.norm(bvec[:, idx].T, axis=1)[:, None]
            L = sh_order_for(len(idx))
            theta, phi = np.arccos(np.clip(u[:, 2], -1, 1)), np.arctan2(u[:, 1], u[:, 0])
            B, *_ = real_sh_descoteaux(L, theta, phi, legacy=False)
            pinv = np.linalg.pinv(B)
            lev = np.einsum("ij,ji->i", B, pinv)
            y = S[:, idx]
            f = (y @ pinv.T.astype(np.float32)) @ B.T.astype(np.float32)
            self.fit[:, idx], self.res[:, idx] = f, (y - f) / np.sqrt(1 - lev).astype(np.float32)
            self.shells[int(b)] = {"directions": len(idx), "sh_order": L, "coefficients": B.shape[1]}
        self.b0 = b0
        self.extra = None
        if extra_b0s is not None:
            E = np.asarray(extra_b0s, np.float32).reshape(-1, extra_b0s.shape[-1])
            m = E.mean(1, keepdims=True)
            self.extra = (np.repeat(m, E.shape[1], 1), (E - m) / np.float32(np.sqrt(1 - 1 / E.shape[1])), extra_b0s.shape)
        self.b0_noise, self.b0_scale = b0_noise, 1.0
        if b0_noise == "thermal":
            mk = np.asarray(mask).reshape(-1) > 0
            dw = np.setdiff1d(np.arange(self.res.shape[1]), self.b0)
            self.b0_scale = float(np.sqrt((self.res[mk][:, dw] ** 2).mean()) / np.sqrt((self.res[mk][:, self.b0] ** 2).mean()))
            self.res[:, self.b0] *= np.float32(self.b0_scale)
            if self.extra is not None:
                self.extra = (self.extra[0], self.extra[1] * np.float32(self.b0_scale), self.extra[2])

    def _mean_fit(self, S, idx):
        y = S[:, idx]
        m = y.mean(1, keepdims=True)
        self.fit[:, idx], self.res[:, idx] = m, (y - m) / np.float32(np.sqrt(1 - 1 / len(idx)))

    def replicate(self, seed: int):
        """(dwi replicate (X, Y, Z, G) as the input's type, extra b0s replicate (X, Y, Z, V) float64 or None)."""
        rng = np.random.default_rng(seed)
        x = np.rint(self.fit + self.res * (rng.integers(0, 2, self.res.shape, dtype=np.int8) * 2 - 1))
        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            x = np.clip(x, max(info.min, 0), info.max)
        else:
            x = np.clip(x, 0, None)
        dwi = x.astype(self.dtype).reshape(self.shape)
        if self.extra is None:
            return dwi, None
        fit, res, shape = self.extra
        e = np.clip(fit + res * (rng.integers(0, 2, res.shape, dtype=np.int8) * 2 - 1), 0, None)
        return dwi, e.astype(np.float64).reshape(shape)

    def noise_check(self, mask) -> dict:
        m = np.asarray(mask).reshape(-1) > 0
        dw = np.setdiff1d(np.arange(self.res.shape[1]), self.b0)
        sd_b0 = float(np.sqrt((self.res[m][:, self.b0] ** 2).mean()))
        rms_dw = float(np.sqrt((self.res[m][:, dw] ** 2).mean()))
        return {"shells": self.shells, "b0_volumes": len(self.b0), "b0_noise": self.b0_noise, "b0_scale": round(self.b0_scale, 4),
                "b0_residual_sd": round(sd_b0, 2),
                "dw_residual_rms": round(rms_dw, 2), "dw_residual_over_b0_sd": round(rms_dw / sd_b0, 3)}
