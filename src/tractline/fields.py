"""Fields a tracker reads at arbitrary points, batched on the device (docs/sota-pipeline-design.md, section 7.2): trilinear
sampling, the FOD's PMF along a direction, and the anatomical constraint's decision (ACT). The conventions are DIPY's, so
that the PTT oracle and ours read the same values (its interpolation, SHCoeffPmfGen and ActStoppingCriterion were read
to understand them; the code here is our own and checked against DIPY's in tests/test_fields.py):
  - points in voxel coordinates, voxel centers at integers; a point is inside when -0.5 <= x < n - 0.5 on every axis,
    and there the edge voxel is repeated (the neighbor index clamped); outside, nothing is read;
  - the PMF along a direction is the FOD at the vertex nearest that direction (largest |cos|, the first on a tie), from
    SH coefficients interpolated trilinearly (equivalently: the vertex's amplitudes interpolated, which is what is
    stored here); negative values are not clipped;
  - ACT: include > 0.5 ends a path validly (ENDPOINT, tested first), exclude > 0.5 invalidly (INVALIDPOINT), outside
    the image is OUTSIDEIMAGE, else TRACKPOINT.

    g = Grid(volume, device)                      # (X, Y, Z) or (X, Y, Z, C)
    vals, inside = g(points)                      # (n, C) or (n,); inside (n,) bool (vals 0 where outside)
    pmf = PmfField(coeffs, mask, vertices, device)
    v, inside = pmf.value(points, directions)     # (n,)
    status = Act(include, exclude, device)(points)    # TRACKPOINT / ENDPOINT / INVALIDPOINT / OUTSIDEIMAGE
"""
from __future__ import annotations

import numpy as np
import torch
from . import sh

TRACKPOINT, ENDPOINT, INVALIDPOINT, OUTSIDEIMAGE = 0, 1, 2, 3


def _dtype(device):
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def corners(points, shape):
    """Trilinear corners of (n, 3) voxel-coordinate points on a grid of `shape`: (indices (n, 8, 3) int64, weights (n, 8),
    inside (n,) bool), DIPY's conventions (module docstring). Outside points get index 0 and weight 0."""
    n = torch.as_tensor(shape, device=points.device, dtype=points.dtype)
    inside = ((points >= -0.5) & (points < n - 0.5)).all(1)
    f = torch.floor(points)
    r = points - f
    lo = f.to(torch.int64)
    hi = lo + 1
    lim = torch.as_tensor(shape, device=points.device) - 1
    lo = torch.minimum(torch.clamp(lo, min=0), lim)
    hi = torch.minimum(torch.clamp(hi, min=0), lim)
    idx, w = [], []
    for c in range(8):
        bits = [(c >> a) & 1 for a in range(3)]
        idx.append(torch.stack([hi[:, a] if b else lo[:, a] for a, b in enumerate(bits)], 1))
        w.append(torch.prod(torch.stack([r[:, a] if b else 1 - r[:, a] for a, b in enumerate(bits)], 1), 1))
    idx, w = torch.stack(idx, 1), torch.stack(w, 1)
    idx = torch.where(inside[:, None, None], idx, torch.zeros_like(idx))
    w = torch.where(inside[:, None], w, torch.zeros_like(w))
    return idx, w, inside


class Grid:
    """A volume sampled trilinearly at voxel-coordinate points (module docstring)."""
    def __init__(self, volume, device="cpu"):
        v = np.asarray(volume)
        self.scalar = v.ndim == 3
        self.shape = v.shape[:3]
        self.dt = _dtype(device)
        self.data = torch.as_tensor(v.reshape(-1, 1) if self.scalar else v.reshape(int(np.prod(self.shape)), -1),
                                    dtype=self.dt, device=device)

    def __call__(self, points):
        p = torch.as_tensor(points, dtype=self.dt, device=self.data.device)
        idx, w, inside = corners(p, self.shape)
        flat = (idx[..., 0] * self.shape[1] + idx[..., 1]) * self.shape[2] + idx[..., 2]
        vals = (self.data[flat] * w[..., None]).sum(1)
        return (vals[:, 0] if self.scalar else vals), inside


class PmfField:
    """The FOD's amplitude along directions, at points: B @ coeffs on `vertices` for each voxel of `mask` (None: the
    voxels with any nonzero coefficient; coefficients outside it must be zero, as a masked fit leaves them), read at the vertex nearest each direction and interpolated
    trilinearly. vertices: (V, 3) unit vectors, the PMF's sphere (both hemispheres when the tracker needs antipodes
    distinguished; the FOD is even, so a hemisphere and |cos| suffice for its values)."""
    def __init__(self, coeffs, mask, vertices, device="cpu", legacy=True):
        c = np.asarray(coeffs)
        mask = np.any(c != 0, axis=-1) if mask is None else np.asarray(mask, bool)
        if np.any(c[~mask] != 0):                                          # DIPY reads every voxel's coefficients
            raise ValueError("PmfField: coefficients outside the mask are not zero (pass mask=None to use them all)")
        self.shape = c.shape[:3]
        self.dt = _dtype(device)
        order = sh.order_of(c.shape[-1])
        self.vertices = torch.as_tensor(np.asarray(vertices, np.float64), dtype=self.dt, device=device)
        B = sh.basis(order, np.asarray(vertices, np.float64), legacy=legacy)
        rows = np.zeros(int(np.prod(self.shape)), np.int64)
        rows[np.flatnonzero(mask)] = np.arange(1, mask.sum() + 1)
        table = np.zeros((int(mask.sum()) + 1, len(B)))
        table[1:] = c[mask].astype(np.float64) @ B.T
        self.rows = torch.as_tensor(rows, device=device)
        self.table = torch.as_tensor(table, dtype=self.dt, device=device)

    def nearest(self, directions):
        d = torch.as_tensor(directions, dtype=self.dt, device=self.table.device)
        return torch.argmax(torch.abs(d @ self.vertices.T), 1)

    def value(self, points, directions):
        p = torch.as_tensor(points, dtype=self.dt, device=self.table.device)
        k = self.nearest(directions)
        idx, w, inside = corners(p, self.shape)
        flat = (idx[..., 0] * self.shape[1] + idx[..., 1]) * self.shape[2] + idx[..., 2]
        vals = self.table[self.rows[flat], k[:, None]]                     # (n, 8)
        return (vals * w).sum(1), inside

    def all(self, points):
        """(n, V) the PMF on every vertex at each point (for the seed's largest peak)."""
        p = torch.as_tensor(points, dtype=self.dt, device=self.table.device)
        idx, w, inside = corners(p, self.shape)
        flat = (idx[..., 0] * self.shape[1] + idx[..., 1]) * self.shape[2] + idx[..., 2]
        return (self.table[self.rows[flat]] * w[..., None]).sum(1), inside


class Act:
    """The anatomical constraint's decision at points (module docstring)."""
    def __init__(self, include, exclude, device="cpu"):
        self.include, self.exclude = Grid(include, device), Grid(exclude, device)

    def __call__(self, points):
        inc, inside = self.include(points)
        exc, _ = self.exclude(points)
        status = torch.full(inc.shape, TRACKPOINT, dtype=torch.int64, device=inc.device)
        status = torch.where(exc > 0.5, torch.full_like(status, INVALIDPOINT), status)
        status = torch.where(inc > 0.5, torch.full_like(status, ENDPOINT), status)
        return torch.where(inside, status, torch.full_like(status, OUTSIDEIMAGE))
