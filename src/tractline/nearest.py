"""The vertex nearest a direction, exactly, without scanning every vertex: a cube map whose cells each list the only
vertices that can be nearest (largest |cos|) for some direction in the cell. For the PTT kernel's PMF lookups
(tractline.ptt_metal), where a scan of every vertex (362 on DIPY's default_sphere) per lookup was the cost.

Exactness. For a cell with center c and angular radius r (the largest angle from c to a point of the cell), let v_c be the
vertex nearest c and a = angle(c, v_c), angles taken between axes (|cos|). For any direction u in the cell with nearest
vertex v*: angle(u, v*) <= angle(u, v_c) <= r + a, so angle(c, v*) <= angle(c, u) + angle(u, v*) <= a + 2r. Listing every
vertex within a + 2r of c (plus a small margin for rounding) therefore always includes v*, and every vertex tied with it;
scanning the list in vertex order with a strict comparison returns the first of the tied ones, as a full scan does.

    cm = CubeMap(vertices, cells=16)
    k = cm.nearest(directions)              # numpy, for tests; the kernel reads cm.offsets, cm.counts, cm.candidates
"""
from __future__ import annotations

import numpy as np

MARGIN = np.radians(0.05)          # float32 rounding in the kernel's face and cell arithmetic


def cell_of(d, cells):
    """(face 0-5, i, j) of directions d (n, 3): the face is the axis of the largest |component| and its sign; (u, v) the
    other two components over it, in [-1, 1], cut into cells x cells."""
    d = np.asarray(d, np.float64)
    a = np.abs(d)
    axis = np.argmax(a, 1)                                               # the first on a tie, as the kernel
    m = d[np.arange(len(d)), axis]
    face = axis * 2 + (m < 0)
    o1 = np.choose(axis, [d[:, 1], d[:, 2], d[:, 0]])
    o2 = np.choose(axis, [d[:, 2], d[:, 0], d[:, 1]])
    u, v = o1 / np.abs(m), o2 / np.abs(m)
    i = np.clip(np.floor((u + 1) * 0.5 * cells), 0, cells - 1).astype(np.int64)
    j = np.clip(np.floor((v + 1) * 0.5 * cells), 0, cells - 1).astype(np.int64)
    return face, i, j


def _direction(face, u, v):
    axis, neg = face // 2, face % 2
    s = -1.0 if neg else 1.0
    out = np.empty(np.shape(u) + (3,))
    out[..., axis] = s
    out[..., (axis + 1) % 3] = u
    out[..., (axis + 2) % 3] = v
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


class CubeMap:
    def __init__(self, vertices, cells=16):
        V = np.asarray(vertices, np.float64)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        self.cells = cells
        edges = np.linspace(-1.0, 1.0, cells + 1)
        cand, offsets, counts = [], [], []
        t = np.linspace(0.0, 1.0, 5)                                     # the cell's boundary and center sampled
        for face in range(6):
            for i in range(cells):
                for j in range(cells):
                    u0, u1, v0, v1 = edges[i], edges[i + 1], edges[j], edges[j + 1]
                    c = _direction(face, np.array((u0 + u1) / 2), np.array((v0 + v1) / 2))
                    uu, vv = np.meshgrid(u0 + t * (u1 - u0), v0 + t * (v1 - v0))
                    pts = _direction(face, uu.ravel(), vv.ravel())
                    # the cell is convex on the face; its farthest point from c is a corner: the radius
                    r = float(np.arccos(np.clip(np.abs(pts @ c), -1, 1)).max())
                    ang = np.arccos(np.clip(np.abs(V @ c), -1, 1))
                    a = float(ang.min())
                    ids = np.flatnonzero(ang <= a + 2 * r + MARGIN)
                    offsets.append(len(cand))
                    counts.append(len(ids))
                    cand.extend(ids.tolist())
        self.candidates = np.asarray(cand, np.int32)
        self.offsets = np.asarray(offsets, np.int32)
        self.counts = np.asarray(counts, np.int32)
        self.vertices = V

    def cell_index(self, d):
        face, i, j = cell_of(d, self.cells)
        return (face * self.cells + i) * self.cells + j

    def nearest(self, d):
        d = np.asarray(d, np.float64)
        cell = self.cell_index(d)
        out = np.empty(len(d), np.int64)
        for n, (x, c) in enumerate(zip(d, cell)):
            ids = self.candidates[self.offsets[c]:self.offsets[c] + self.counts[c]]
            dots = np.abs(self.vertices[ids] @ x)
            out[n] = ids[np.argmax(dots)]                                 # the first maximum: ids ascend
        return out
