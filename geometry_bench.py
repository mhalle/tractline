"""Geometry encodings of the HCP tractogram: how many bytes the streamlines need, at what error.

    DATA/.venv/bin/python bench/tractography/geometry_bench.py

Every encoding is of all original vertices (21.6 M), in streamline order. Compressed sizes are
blosc zstd (level 5, or 9 where noted) with byte or bit shuffle. The predictive encodings are
lossless relative to their grid: positions are rounded to `quantum` mm about the bounding-box
center, and the stored residual is
    first vertex of a streamline   absolute
    second vertex                  first-order delta
    every later vertex             p[i] - (2 p[i-1] - p[i-2])     (constant-velocity prediction)
so each streamline decodes on its own with two running sums; decode() is that reader, and every
encoding is round-tripped through it before its size is reported.

Writes results/geometry.json.
"""
import json
from pathlib import Path
import numpy as np
from numcodecs import Blosc
from vtk.util.numpy_support import vtk_to_numpy
from _data import HCP_VTP
from tractcloud.vtk_io import read_polydata

OUT = Path(__file__).resolve().parent / "results"
pd = read_polydata(str(HCP_VTP))
pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
off = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
P = pts[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
N, V = len(off) - 1, len(P)
lens = np.diff(off)
k = np.arange(V) - np.repeat(off[:-1], lens)                 # vertex index within its streamline
step = np.linalg.norm(np.diff(P, axis=0), axis=1)[k[1:] > 0]
center = np.round((P.min(0) + P.max(0)) / 2, 2)
z = lambda a, s=Blosc.SHUFFLE, lvl=5: len(Blosc(cname="zstd", clevel=lvl, shuffle=s).encode(np.ascontiguousarray(a)))
row = lambda name, size, err: {"encoding": name, "mb": round(size / 1e6, 1), "bytes_per_vertex": round(size / V, 2),
                               "max_error_mm": err}

def segment_cumsum(x, off):
    """Cumulative sum restarting at every streamline start (x is (V, 3))."""
    c = np.cumsum(x, axis=0)
    before = np.vstack([np.zeros((1, 3), x.dtype), c[off[:-1] - 1][1:]])     # running total before each start
    return c - np.repeat(before, np.diff(off), axis=0)


def decode(R2, k, off):
    """Residuals -> integer positions: the reader's two running sums, per streamline."""
    D = R2.copy()
    D[k == 0] = 0                                    # the first vertex is absolute, not a step
    D = segment_cumsum(D, off)                       # steps: D[1] stored, later D[i] = D[i-1] + r
    first = np.repeat(R2[k == 0], np.diff(off), axis=0)
    D[k == 0] = 0
    return first + segment_cumsum(D, off)


res = [row("float32, raw", V * 12, 0.0), row("float32, zstd", z(P.astype("<f4")), 0.0)]
f16 = P.astype("<f2")
e16 = float(np.abs(f16.astype(np.float64) - P).max())
res += [row("float16, raw", V * 6, round(e16, 4)), row("float16, zstd", z(f16), round(e16, 4))]
for q in (0.01, 0.05, 0.1):
    Q = np.round((P - center) / q).astype(np.int64)
    err = float(np.abs(Q * q + center - P).max())
    res.append(row(f"int16 @ {q} mm absolute, zstd", z(Q.astype("<i2")), round(err, 4)))
    R = np.empty_like(Q)
    R[0] = Q[0]
    R[1:] = Q[1:] - Q[:-1]
    R2 = R.copy()
    R2[2:] = Q[2:] - 2 * Q[1:-1] + Q[:-2]
    R2[k == 0] = Q[k == 0]
    R2[k == 1] = R[k == 1]
    body = k >= 2
    assert (decode(R2, k, off) == Q).all(), f"round trip failed at {q} mm"
    # first-order deltas, as DSI Studio's .tt.gz stores them (there in int8, split steps beyond),
    # under the same coder, for a like-for-like comparison with the second-order prediction
    m1 = int(np.abs(R[k >= 1]).max())
    dt1 = "<i1" if m1 < 128 else "<i2"
    size1 = z(R[k >= 1].astype(dt1), Blosc.BITSHUFFLE, 9) + z(Q[k == 0].astype("<i2"), Blosc.SHUFFLE, 9)
    res.append({**row(f"{q} mm grid, 1st-order deltas, {dt1[1:]}, bitshuffle zstd-9", size1, round(float(np.abs(Q * q + center - P).max()), 4)),
                "residual_max": m1})
    m = int(np.abs(R2[body]).max())
    dt = "<i1" if m < 128 else "<i2"
    size = z(R2[body].astype(dt), Blosc.BITSHUFFLE, 9) + z(R2[~body].astype("<i2"), Blosc.SHUFFLE, 9)
    res.append({**row(f"{q} mm grid, 2nd-order prediction, {dt[1:]} residuals, bitshuffle zstd-9", size, round(err, 4)),
                "residual_max": m, "residual_p99": float(np.quantile(np.abs(R2[body]), 0.99))})

summary = {"tractogram": "HCP 101006, UKF (TractCloud TestData v1.0.0)", "streamlines": N, "vertices": V,
           "vertices_per_streamline": {"median": float(np.median(lens)), "min": int(lens.min()), "max": int(lens.max())},
           "step_mm": {"median": round(float(np.median(step)), 3), "p01": round(float(np.quantile(step, 0.01)), 3),
                       "p99": round(float(np.quantile(step, 0.99)), 3)},
           "lengths_uint16_zstd_mb": round(z(lens.astype("<u2")) / 1e6, 2),
           "encodings": res}
(OUT / "geometry.json").write_text(json.dumps(summary, indent=1))
for r in res:
    print(f"{r['encoding']:60s} {r['mb']:7.1f} MB  {r['bytes_per_vertex']:5.2f} B/vertex  max err {r['max_error_mm']} mm")
