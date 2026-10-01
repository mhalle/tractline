"""Conversion speed between the compact form (predictive geometry + rankfield field) and TRX.

    DATA/.venv/bin/python bench/tractography/convert_bench.py

The whole HCP tractogram (440,621 streamlines, 21.6 M vertices) on the 12-bit Draco-equivalent
grid (0.041 mm), with run 0's depth-6 field. Each direction is timed stage by stage, best of 3,
with the per-vertex loops two ways: numpy (whole-array passes, what a quick Python tool does) and
numba (compiled loops, about what a C or Rust converter would get). Blosc runs with its default
threads (8 on the M2: 4 performance, 4 efficiency cores).

compact -> TRX: decompress geometry and field; reconstruct positions (float32, or float16 as
TRX suggests); offsets from the lengths; TractCloud's tract labels as groups (from ranks[0]);
the field planes as per-streamline arrays (dps, element-major); write a TRX directory to disk.
TRX -> compact: read positions and offsets (memory-mapped); quantize, predict, compress at zstd
levels 1, 5 and 9; the field planes back to plane-major and compressed.

Checks: the TRX written from the compact form re-encodes to identical residuals when its
positions are float32 (the grid survives float32 exactly). From float16 TRX it cannot:
float16's spacing is 0.0625 mm past 64 mm from the origin, coarser than the 0.041 mm grid.

Writes results/convert.json.
"""
import json, shutil, time
from pathlib import Path
import numpy as np
import numba
import vtk
from numcodecs import Blosc
from vtk.util.numpy_support import vtk_to_numpy
from _data import DATA, HCP_VTP
from _geometry import decode, line_index, residuals
from tractcloud.tract_mapping import TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as TRACT

OUT = Path(__file__).resolve().parent / "results"
TMP = DATA / "convert_tmp"
BITS = 12
REPS = 3


def best(fn):
    times, res = [], None
    for _ in range(REPS):
        t0 = time.perf_counter(); res = fn(); times.append(time.perf_counter() - t0)
    return res, round(min(times) * 1000, 1)


@numba.njit(cache=True)
def nb_reconstruct(lens, head, body, origin, q, out):
    b = 0; h = 0; o = 0
    for l in range(lens.shape[0]):
        n = lens[l]
        x = np.int64(head[h]); y = np.int64(head[h + 1]); z = np.int64(head[h + 2]); h += 3
        out[o, 0] = origin[0] + x * q; out[o, 1] = origin[1] + y * q; out[o, 2] = origin[2] + z * q; o += 1
        if n < 2:
            continue
        dx = np.int64(head[h]); dy = np.int64(head[h + 1]); dz = np.int64(head[h + 2]); h += 3
        x += dx; y += dy; z += dz
        out[o, 0] = origin[0] + x * q; out[o, 1] = origin[1] + y * q; out[o, 2] = origin[2] + z * q; o += 1
        for i in range(2, n):
            dx += body[b]; dy += body[b + 1]; dz += body[b + 2]; b += 3
            x += dx; y += dy; z += dz
            out[o, 0] = origin[0] + x * q; out[o, 1] = origin[1] + y * q; out[o, 2] = origin[2] + z * q; o += 1


@numba.njit(cache=True)
def nb_residuals(P, off, origin, q, head, body):
    """positions -> head rows (first vertex, first delta) and body rows (second-order residuals)"""
    h = 0; b = 0
    for l in range(off.shape[0] - 1):
        s, e = off[l], off[l + 1]
        px = py = pz = 0; dx = dy = dz = 0
        for i in range(s, e):
            x = np.int64(np.round((P[i, 0] - origin[0]) / q)); y = np.int64(np.round((P[i, 1] - origin[1]) / q))
            z = np.int64(np.round((P[i, 2] - origin[2]) / q))
            k = i - s
            if k == 0:
                head[h, 0] = x; head[h, 1] = y; head[h, 2] = z; h += 1
            elif k == 1:
                dx = x - px; dy = y - py; dz = z - pz
                head[h, 0] = dx; head[h, 1] = dy; head[h, 2] = dz; h += 1
            else:
                ndx = x - px; ndy = y - py; ndz = z - pz
                body[b, 0] = ndx - dx; body[b, 1] = ndy - dy; body[b, 2] = ndz - dz; b += 1
                dx = ndx; dy = ndy; dz = ndz
            px = x; py = y; pz = z


# ---- inputs: the compact form, in memory ----
r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(HCP_VTP)); r.Update(); pd = r.GetOutput()
pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
off = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
P0 = pts[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
N, V = len(off) - 1, len(P0)
origin = P0.min(0)
q = float((P0.max(0) - origin).max()) / (2 ** BITS - 1)
k = line_index(off)
Q = np.round((P0 - origin) / q).astype(np.int64)
R = residuals(Q, k, 2)
der = np.load(DATA / "hcp_full" / "derived.npz")
ranks, support, tail = der["rf_d6_ranks"], der["rf_d6_support"], der["rf_d6_tail"]
codec = lambda lvl, s: Blosc(cname="zstd", clevel=lvl, shuffle=s)
blobs = {"lens": codec(9, Blosc.SHUFFLE).encode(np.diff(off).astype("<u2")),
         "head": codec(9, Blosc.SHUFFLE).encode(np.ascontiguousarray(R[k < 2].astype("<i2"))),
         "body": codec(9, Blosc.BITSHUFFLE).encode(np.ascontiguousarray(R[k >= 2].astype("<i1"))),
         "ranks": codec(9, Blosc.SHUFFLE).encode(ranks), "support": codec(9, Blosc.SHUFFLE).encode(support),
         "tail": codec(9, Blosc.SHUFFLE).encode(tail)}
compact_mb = round(sum(len(b) for b in blobs.values()) / 1e6, 1)
grid = (Q * q + origin)
nb_reconstruct(np.diff(off).astype(np.uint16)[:2], R[k < 2].astype(np.int16).ravel(), R[k >= 2].astype(np.int8).ravel(),
               origin, q, np.empty((off[2], 3), np.float32))                       # compile outside the timer
nb_residuals(P0[:off[2]].astype(np.float32), off[:3], origin, q, np.empty((4, 3), np.int64), np.empty((off[2], 3), np.int64))

# ---- compact -> TRX ----
d2t = {}
(lens, head, body), d2t["decompress geometry"] = best(lambda: (
    np.frombuffer(Blosc().decode(blobs["lens"]), "<u2"), np.frombuffer(Blosc().decode(blobs["head"]), "<i2"),
    np.frombuffer(Blosc().decode(blobs["body"]), "<i1")))
(rk, sp, tl), d2t["decompress field"] = best(lambda: (
    np.frombuffer(Blosc().decode(blobs["ranks"]), "<u2").reshape(6, N), np.frombuffer(Blosc().decode(blobs["support"]), "u1").reshape(5, N),
    np.frombuffer(Blosc().decode(blobs["tail"]), "<u2")))
offs, d2t["offsets"] = best(lambda: np.r_[0, np.cumsum(lens, dtype=np.uint64)].astype(np.uint64))


def np_reconstruct():
    o = offs.astype(np.int64); kk = line_index(o)
    Rd = np.empty((V, 3), np.int64)
    Rd[kk >= 2] = body.reshape(-1, 3); Rd[kk < 2] = head.reshape(-1, 3)
    return (decode(Rd, kk, o, 2) * q + origin).astype(np.float32)


def nb_reconstruct_f32():
    out = np.empty((V, 3), np.float32)
    nb_reconstruct(lens, head, body, origin, q, out)
    return out


pos32, d2t["reconstruct float32 (numpy)"] = best(np_reconstruct)
pos32b, d2t["reconstruct float32 (numba)"] = best(nb_reconstruct_f32)
assert np.array_equal(pos32, pos32b) and np.array_equal(pos32, grid.astype(np.float32))
pos16, d2t["to float16"] = best(lambda: pos32.astype(np.float16))


def groups():
    tract = TRACT[rk[0].astype(np.int64) - 1]
    order = np.argsort(tract, kind="stable").astype(np.uint32)
    cuts = np.searchsorted(tract[order], np.arange(len(TRACT_NAMES) + 1))
    return {TRACT_NAMES[t]: order[cuts[t]:cuts[t + 1]] for t in range(len(TRACT_NAMES)) if cuts[t + 1] > cuts[t]}


grp, d2t["labels to groups"] = best(groups)
dps, d2t["field to dps (element-major)"] = best(lambda: {"rf_ranks.6.uint16": np.ascontiguousarray(rk.T), "rf_gaps.5.uint8": np.ascontiguousarray(sp.T),
                                                        "rf_tail.uint16": tl})


def write_trx(pos, name):
    d = TMP / name
    shutil.rmtree(d, ignore_errors=True); (d / "groups").mkdir(parents=True); (d / "dps").mkdir()
    (d / "header.json").write_text(json.dumps({"VOXEL_TO_RASMM": np.eye(4).tolist(), "DIMENSIONS": [1, 1, 1],
                                               "NB_STREAMLINES": N, "NB_VERTICES": V}))
    pos.tofile(d / f"positions.3.{pos.dtype.name}")
    offs.tofile(d / "offsets.uint64")
    for g, idx in grp.items(): idx.tofile(d / "groups" / f"{g}.uint32")
    for n, a in dps.items(): a.tofile(d / "dps" / n)
    return d


trx32, d2t["write TRX dir, float32"] = best(lambda: write_trx(pos32, "f32"))
trx16, d2t["write TRX dir, float16"] = best(lambda: write_trx(pos16, "f16"))
sizes = {n: round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1) for n, d in (("float32", trx32), ("float16", trx16))}

# ---- TRX -> compact ----
t2c = {}
(Pm, om), t2c["open TRX (memmap)"] = best(lambda: (np.memmap(trx32 / "positions.3.float32", "<f4", "r").reshape(-1, 3),
                                                   np.fromfile(trx32 / "offsets.uint64", "<u8").astype(np.int64)))


def np_encode():
    kk = line_index(om)
    QQ = np.round((np.asarray(Pm, np.float64) - origin) / q).astype(np.int64)
    RR = residuals(QQ, kk, 2)
    return RR[kk < 2], RR[kk >= 2]


def nb_encode():
    head2 = np.empty((N * 2, 3), np.int64); body2 = np.empty((V - 2 * N, 3), np.int64)
    nb_residuals(np.asarray(Pm), om, origin, q, head2, body2)
    return head2, body2


(h_np, b_np), t2c["quantize + predict (numpy)"] = best(np_encode)
(h_nb, b_nb), t2c["quantize + predict (numba)"] = best(nb_encode)
assert np.array_equal(h_np, h_nb) and np.array_equal(b_np, b_nb)
assert np.array_equal(b_nb, R[k >= 2]) and np.array_equal(h_nb, R[k < 2]), "float32 TRX did not round-trip to the grid"
b8, h16 = np.ascontiguousarray(b_nb.astype("<i1")), np.ascontiguousarray(h_nb.astype("<i2"))
levels = {}
for lvl in (1, 5, 9):
    size, ms = best(lambda: len(codec(lvl, Blosc.BITSHUFFLE).encode(b8)) + len(codec(lvl, Blosc.SHUFFLE).encode(h16))
                    + len(codec(lvl, Blosc.SHUFFLE).encode(np.diff(om).astype("<u2"))))
    levels[f"zstd {lvl}"] = {"ms": ms, "geometry_mb": round(size / 1e6, 2)}
t2c["compress geometry"] = levels
fsize, t2c["field from dps + compress (zstd 9)"] = best(lambda: sum(len(codec(9, Blosc.SHUFFLE).encode(np.ascontiguousarray(a.T if a.ndim == 2 else a)))
                                                                    for a in (dps["rf_ranks.6.uint16"], dps["rf_gaps.5.uint8"], dps["rf_tail.uint16"])))
# float16 TRX cannot return to the grid
P16 = np.memmap(trx16 / "positions.3.float16", "<f2", "r").reshape(-1, 3).astype(np.float64)
lost = int((np.round((P16 - origin) / q).astype(np.int64) != Q).any(1).sum())

summary = {"grid_mm": round(q, 4), "vertices": V, "streamlines": N, "compact_mb": compact_mb, "trx_dir_mb": sizes,
           "blosc_threads": 8, "reps": REPS, "compact_to_trx_ms": d2t, "trx_to_compact_ms": t2c,
           "float16_trx_vertices_off_grid": lost}
fast_d = sum(v for kk_, v in d2t.items() if "numpy" not in kk_ and "write" not in kk_)
summary["compact_to_trx_in_memory_ms_numba"] = round(fast_d - d2t["to float16"], 1)
summary["trx_to_compact_ms_numba_zstd5"] = round(t2c["open TRX (memmap)"] + t2c["quantize + predict (numba)"] + levels["zstd 5"]["ms"]
                                                  + t2c["field from dps + compress (zstd 9)"], 1)
(OUT / "convert.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
shutil.rmtree(TMP, ignore_errors=True)
