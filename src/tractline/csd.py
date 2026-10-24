"""Constrained spherical deconvolution of one shell (Tournier et al. 2007), batched over voxels in torch: the model slot's
"csd" (docs/sota-pipeline-design.md, section 7.1). The oracle is DIPY 1.12 (ConstrainedSphericalDeconvModel,
auto_response_ssst) with the step 6 benchmark's settings. Provenance, as the project's policy asks: DIPY's code (csdeconv,
the ssst response, its ROI helpers, the WLS tensor fit) was read to understand it and to match its behavior; this is our
own implementation, batched (one GEMM per iteration over the constraint directions, chunked solves), not a translation.
Choices taken from DIPY's code, beyond its docstring's account and the papers: the threshold from the FOD's mean (the
order-0 term; DIPY's docstring says the max), the full-order fallback when the order-4 FOD has no negative direction,
Cholesky first and 1e-5 I only on failure, the ROI's shrink rule min(c, n - c), and the eigenvalue floor 1e-6 / max|A|.
tests/test_csd.py and bench/csd_check.py check it against DIPY. The order is the caller's (DIPY's benchmark uses 8 from 45
directions, else 6).

    resp = response_ssst(dwi, bval, bvec)                    # the single-fiber response: Response(evals, S0, voxels, ratio)
    model = CSD(bval, bvec, resp, order=8)                   # sphere: the constraint directions (default: hemisphere(181))
    coeffs, info = model.fit(dwi, mask, device="cpu")        # (X, Y, Z, 45) float32 SH coefficients (sh.basis, legacy)

The response (response_ssst): a cube of radius `roi_radius` voxels about the volume's center (shrunk to fit inside the
volume), a weighted-least-squares tensor fit there (weights: the signal an ordinary least-squares fit predicts, the signal
clipped below at 1e-4), the voxels with FA above `fa_threshold`, and the mean of their two largest eigenvalues (the third
set to the second) with the mean of their b0 signals as S0.

The deconvolution: the response's signal along the shell's gradients as a tensor along z, fitted in SH (least squares)
and reduced to its rotational harmonics; the forward matrix X = diag(those, per l) B_dwi; the constraint matrix B_reg on
the constraint sphere, scaled by lambda * n_coeffs * r_0 / (sqrt(n_sphere) sqrt(362)) (Tournier et al. 2012's scaling, as
DIPY applies it). Per voxel, with s the shell's raw signals:
  z = X^T s; f0 solves (X^T X) f0 = z by Cholesky (with 1e-5 I added if that fails);
  threshold = tau * (the order-0 coefficient's amplitude, B_reg[0, 0] f0[0]): the FOD's mean scaled by tau;
  the negative set: directions where the order-4 part of f0 falls below it (else where all of f0 does; if none, f0 is the
  answer, 0 iterations);
  repeat up to 50 times: f solves (X^T X + B_H^T B_H) f = z, B_H the negative set's rows of B_reg; the new negative set
  from B_reg f; stop when it is the same set.
Batched: the set is a 0/1 weight per constraint direction, Q = X^T X + B^T diag(w) B for a chunk of voxels at once (one
GEMM over the directions' outer products), then a batched Cholesky (an LU solve on MPS); a voxel stops changing once
its set repeats. float64 on the CPU and CUDA. MPS has no float64: there the iterations run in float32, but the first step
(f0, the threshold, the first negative set) is always made in float64 on the host. When X^T X is near-singular (condition
number above COND_FLOAT32: fewer directions than coefficients, or nearly so) float32 cannot be trusted and the whole fit
runs on the CPU in float64; fit()'s info says where it ran.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import torch
from . import sh

B0_THRESHOLD = 50.0                 # s/mm^2: below it a volume is a b0 (DIPY's gradient table default)
MIN_SIGNAL = 1e-4                   # the tensor fit's floor before the log (DIPY's MIN_POSITIVE_SIGNAL)
MU = 1e-5                           # added to X^T X when its Cholesky fails
CONVERGENCE = 50                    # iterations at most
COND_FLOAT32 = 1e8                  # X^T X above this condition number: no float32 fit (MPS falls back to the CPU)
CHUNK = 20_000                      # voxels per batched solve: Q is (chunk, n_coeffs, n_coeffs), ~320 MB in float64


@dataclass
class Response:
    evals: np.ndarray                # (3,) mm^2/s: the mean largest eigenvalue, then the mean second one twice
    S0: float                        # the mean b0 signal of the voxels used
    voxels: int                      # how many voxels passed the FA threshold
    ratio: float                     # evals[1] / evals[0]


def hemisphere(n: int = 181, iterations: int = 500) -> np.ndarray:
    """(n, 3) unit vectors on the upper hemisphere, spread evenly as axes: our own constraint sphere. The FOD is even
    (antipodally symmetric), so one hemisphere constrains it fully and each direction counts once. DIPY's small_sphere is
    a hemisphere of 181 too (a data file we do not ship); the lambda scaling below counts the sphere's directions.
    Made by electrostatic repulsion between axes (each point and its antipode charged alike, as repulsion spheres are
    made), from a Fibonacci lattice, with a step that shrinks: deterministic. At 181 the nearest axes are 10-11.5 degrees
    apart (tests/test_csd.py)."""
    return _hemisphere(n, iterations).copy()


@lru_cache(maxsize=8)
def _hemisphere(n, iterations):
    """hemisphere(), made once per (n, iterations)."""
    i = np.arange(n) + 0.5
    z = 1.0 - i / n
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    v = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
    for k in range(iterations):
        force = np.zeros_like(v)
        for s in (1.0, -1.0):                                             # the points and their antipodes
            d = v[:, None, :] - s * v[None, :, :]
            dist = np.linalg.norm(d, axis=-1)
            np.fill_diagonal(dist, np.inf)                                # itself; its own antipode (radial: no force)
            force += (d / dist[..., None] ** 3).sum(1)
        force -= (force * v).sum(1, keepdims=True) * v                     # tangent to the sphere
        step = 0.5 / n * (1.0 - k / iterations) / np.abs(force).max()
        v = v + step * force
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= np.where(v[:, 2:3] < 0, -1.0, 1.0)                            # back onto the upper hemisphere
    return v


def _dwi(bval, bvec):
    """bval (G,), bvec (3, G) as FSL stores it or (G, 3): (bval, gradients (G, 3), the diffusion-weighted volumes)."""
    bval = np.asarray(bval, np.float64)
    g = np.asarray(bvec, np.float64)
    G = len(bval)
    if G == 3:
        raise ValueError("three volumes: the b-vectors' layout, (3, G) or (G, 3), cannot be told apart")
    if g.shape == (3, G):
        g = g.T
    elif g.shape != (G, 3):
        raise ValueError(f"b-vectors of shape {g.shape} for {G} b-values: (3, {G}) or ({G}, 3) expected")
    return bval, g, bval > B0_THRESHOLD


def tensor_fit(signal, bval, bvec):
    """Weighted least-squares DTI of (n, G) signals: (evals (n, 3) descending, clipped at a small positive floor; S0 (n,))."""
    bval, g, _ = _dwi(bval, bvec)
    A = np.stack([g[:, 0] ** 2, 2 * g[:, 0] * g[:, 1], g[:, 1] ** 2, 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2],
                  g[:, 2] ** 2], 1) * -bval[:, None]
    A = np.concatenate([A, np.ones((len(bval), 1))], 1)                   # log S = log S0 - b g^T D g
    y = np.log(np.maximum(np.asarray(signal, np.float64), MIN_SIGNAL))
    ols = y @ np.linalg.pinv(A).T
    w = np.exp(ols @ A.T)                                                 # the predicted signal: the weights
    p = np.einsum("nij,nj->ni", np.linalg.pinv(A[None] * w[..., None]), w * y)
    D = np.stack([p[:, [0, 1, 3]], p[:, [1, 2, 4]], p[:, [3, 4, 5]]], 1)
    ev = np.linalg.eigvalsh(D)[:, ::-1]
    floor = 1e-6 / np.abs(A[:, :6]).max()
    return np.maximum(ev, floor), np.exp(p[:, 6])


def fa(evals):
    ev = np.asarray(evals, np.float64)
    den = (ev * ev).sum(-1)
    num = (ev[..., 0] - ev[..., 1]) ** 2 + (ev[..., 1] - ev[..., 2]) ** 2 + (ev[..., 2] - ev[..., 0]) ** 2
    return np.sqrt(0.5 * num / np.where(den > 0, den, 1.0)) * (den > 0)


def response_ssst(dwi, bval, bvec, roi_radius=10, fa_threshold=0.7) -> Response:
    """The single-fiber response from the high-FA voxels of a cube about the volume's center (module docstring)."""
    dwi = np.asarray(dwi)
    shape = np.array(dwi.shape[:3])
    c = shape // 2
    r = np.full(3, int(roi_radius))
    for i in range(3):                                                    # shrunk to fit inside the volume
        if c[i] - r[i] < 0 or c[i] + r[i] >= shape[i]:
            r[i] = min(int(c[i]), int(shape[i] - c[i]))
    cube = dwi[c[0] - r[0]:c[0] + r[0] + 1, c[1] - r[1]:c[1] + r[1] + 1, c[2] - r[2]:c[2] + r[2] + 1]
    sig = cube.reshape(-1, dwi.shape[-1]).astype(np.float64)
    ev, _ = tensor_fit(sig, bval, bvec)
    keep = fa(ev) > fa_threshold
    if not keep.any():
        raise ValueError(f"response_ssst: no voxel with FA above {fa_threshold} within {roi_radius} voxels of the center")
    b0 = np.asarray(bval) <= B0_THRESHOLD
    lam = ev[keep, :2].mean(0)
    evals = np.array([lam[0], lam[1], lam[1]])
    return Response(evals, float(sig[keep][:, b0].mean()), int(keep.sum()), float(evals[1] / evals[0]))


class CSD:
    def __init__(self, bval, bvec, response: Response, order=8, sphere=None, lambda_=1.0, tau=0.1):
        bval, g, dw = _dwi(bval, bvec)
        self.order, self.tau, self.dw = order, tau, dw
        m, l = sh.indices(order)
        B_dwi = sh.basis(order, g[dw])
        # the response along the shell's gradients, a tensor along z, in SH; its rotational harmonics (m = 0, over Y_l^0(z))
        D = np.diag([response.evals[1], response.evals[2], response.evals[0]])
        S_r = response.S0 * np.exp(-bval[dw] * np.einsum("gi,ij,gj->g", g[dw], D, g[dw]))
        r_sh = np.linalg.lstsq(B_dwi, S_r, rcond=None)[0]
        r_rh = r_sh[m == 0] / sh.basis(order, np.array([[0.0, 0.0, 1.0]]))[0, m == 0]
        self.r_rh = r_rh
        sphere = hemisphere(181) if sphere is None else np.asarray(sphere, np.float64)
        B_reg = sh.basis(order, sphere)
        lam = lambda_ * len(m) * r_rh[0] / (math.sqrt(len(sphere)) * math.sqrt(362.0))
        self.B_reg = B_reg * lam
        self.X = sh.zonal(order, r_rh)[None, :] * B_dwi
        self.P = self.X.T @ self.X
        self.cond = np.linalg.cond(self.P)
        try:                                                                # 1e-5 I only when the Cholesky fails
            self.LPn, self.Pn = np.linalg.cholesky(self.P), self.P
        except np.linalg.LinAlgError:
            self.Pn = self.P + MU * np.eye(len(self.P))
            self.LPn = np.linalg.cholesky(self.Pn)

    def fit(self, dwi, mask=None, device="cpu"):
        """(coeffs (X, Y, Z, n) float32, zero outside the mask; info: iterations (max, mean), voxels not converged)."""
        dwi = np.asarray(dwi)
        mask = np.ones(dwi.shape[:3], bool) if mask is None else np.asarray(mask, bool)
        S = dwi[mask][:, self.dw]
        f, iters, used = self.fit_signals(S, device)
        out = np.zeros(dwi.shape[:3] + (self.P.shape[0],), np.float32)
        out[mask] = f
        return out, {"iterations max": int(iters.max(initial=0)), "iterations mean": float(iters.mean()) if len(iters) else 0.0,
                     "not converged": int((iters > CONVERGENCE).sum()), "voxels": int(mask.sum()), "device": used,
                     "condition number of X^T X": float(self.cond)}

    def fit_signals(self, S, device="cpu"):
        """(n, G_dw) raw signals -> (coefficients (n, K) float64, iterations (n,) int - CONVERGENCE + 1: did not
        converge - and the device it ran on)."""
        dev = torch.device(device)
        if dev.type == "mps" and self.cond > COND_FLOAT32:
            dev = torch.device("cpu")
        dt = torch.float32 if dev.type == "mps" else torch.float64
        B = torch.as_tensor(self.B_reg, dtype=dt, device=dev)
        K = self.P.shape[0]
        P = torch.as_tensor(self.Pn, dtype=dt, device=dev)
        O = (B[:, :, None] * B[:, None, :]).reshape(len(B), K * K)         # each direction's outer product: Q by one GEMM
        low = min(15, K)
        if dev.type == "mps":                                               # MPS: cholesky_solve and solve_triangular are
            solve = lambda Q, z: torch.linalg.solve(Q, z)                   # ~50x slower there than an LU solve (measured)
        else:
            solve = lambda Q, z: torch.cholesky_solve(z[..., None], torch.linalg.cholesky(Q))[..., 0]
        outs, its = [], []
        for a in range(0, len(S), CHUNK):
            # the first step in float64 on the host, whatever the device: f0, the threshold and the first negative set
            s = np.asarray(S[a:a + CHUNK], np.float64)
            z64 = s @ self.X                                                # (n, K) = X^T s per voxel
            f0 = np.linalg.solve(self.LPn.T, np.linalg.solve(self.LPn, z64.T)).T
            thr64 = self.B_reg[0, 0] * f0[:, 0] * self.tau
            neg64 = (f0[:, :low] @ self.B_reg[:, :low].T) < thr64[:, None]
            none = ~neg64.any(1)
            neg64[none] = (f0[none] @ self.B_reg.T) < thr64[none, None]
            z = torch.as_tensor(z64, dtype=dt, device=dev)
            f = torch.as_tensor(f0, dtype=dt, device=dev)
            thr = torch.as_tensor(thr64, dtype=dt, device=dev)
            neg = torch.as_tensor(neg64, device=dev)
            it = torch.zeros(len(s), dtype=torch.int64, device=dev)
            active = neg.any(1)                                             # no negative direction: f0 stands
            for k in range(1, CONVERGENCE + 1):
                idx = torch.nonzero(active).flatten()
                if len(idx) == 0:
                    break
                Q = P + (neg[idx].to(dt) @ O).view(len(idx), K, K)
                fi = solve(Q, z[idx])
                f[idx] = fi
                new = (fi @ B.T) < thr[idx, None]
                same = (new == neg[idx]).all(1)
                neg[idx] = new
                it[idx] = k
                active[idx[same]] = False
            it[active] = CONVERGENCE + 1
            outs.append(f.cpu().double().numpy())
            its.append(it.cpu().numpy())
        return ((np.concatenate(outs) if outs else np.zeros((0, K))), (np.concatenate(its) if its else np.zeros(0, int)),
                dev.type)
