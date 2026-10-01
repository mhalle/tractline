"""Geometry encodings of the HCP tractogram: how many bytes the streamlines need, at what error.

    DATA/.venv/bin/python bench/tractography/geometry_bench.py

Every encoding is of all original vertices (21.6 M), in streamline order. Compressed sizes are
blosc zstd (level 5, or 9 where noted) with byte or bit shuffle. The predictive encodings are
lossless relative to their grid: positions are rounded to `quantum` mm about the bounding-box
center, and the stored residual is
    first vertex of a streamline   absolute
    second vertex                  first-order delta
    every later vertex             p[i] - (2 p[i-1] - p[i-2])     (constant-velocity prediction)
so each streamline decodes on its own with two running sums. The encoder is _geometry.py's,
shared with trako_compare.py; every encoding is round-tripped through its decoder.

Writes results/geometry.json.
"""
import json
from pathlib import Path
import numpy as np
from numcodecs import Blosc
from vtk.util.numpy_support import vtk_to_numpy
from _data import HCP_VTP
from tractcloud.vtk_io import read_polydata
from _geometry import encode, residuals

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

res = [row("float32, raw", V * 12, 0.0), row("float32, zstd", z(P.astype("<f4")), 0.0)]
f16 = P.astype("<f2")
e16 = float(np.abs(f16.astype(np.float64) - P).max())
res += [row("float16, raw", V * 6, round(e16, 4)), row("float16, zstd", z(f16), round(e16, 4))]
for q in (0.01, 0.05, 0.1):
    Q = np.round((P - center) / q).astype(np.int64)
    err = float(np.abs(Q * q + center - P).max())
    res.append(row(f"int16 @ {q} mm absolute, zstd", z(Q.astype("<i2")), round(err, 4)))
    for order in (1, 2):
        size, _ = encode(P, off, q, center, order=order)      # round-trips through the decoder
        R = residuals(Q, k, order)
        m = int(np.abs(R[k >= order]).max())
        name = "1st-order deltas" if order == 1 else "2nd-order prediction"
        res.append({**row(f"{q} mm grid, {name}, {'i1' if m < 128 else 'i2'} residuals, bitshuffle zstd-9", size, round(err, 4)),
                    "residual_max": m, "residual_p99": float(np.quantile(np.abs(R[k >= order]), 0.99))})

summary = {"tractogram": "HCP 101006, UKF (TractCloud TestData v1.0.0)", "streamlines": N, "vertices": V,
           "vertices_per_streamline": {"median": float(np.median(lens)), "min": int(lens.min()), "max": int(lens.max())},
           "step_mm": {"median": round(float(np.median(step)), 3), "p01": round(float(np.quantile(step, 0.01)), 3),
                       "p99": round(float(np.quantile(step, 0.99)), 3)},
           "lengths_uint16_zstd_mb": round(z(lens.astype("<u2")) / 1e6, 2),
           "encodings": res}
(OUT / "geometry.json").write_text(json.dumps(summary, indent=1))
for r in res:
    print(f"{r['encoding']:60s} {r['mb']:7.1f} MB  {r['bytes_per_vertex']:5.2f} B/vertex  max err {r['max_error_mm']} mm")
