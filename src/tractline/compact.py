"""Compact storage for a run's labeled streamlines: their full geometry, predictive-coded, and their cluster
log-probabilities as a rank field. Ported from the tractography incubation (medseg, tractography-incubation
branch: bench/tractography/_geometry.py and _pipeline.encode_payload), where it measured 1.55 B per vertex at
0.05 mm (2nd order; float32 is 12) and 12-15 B per streamline for the field, against TRX's 135 MB for one HCP
tractogram with labels only (41 MB for geometry, field and tables).

Geometry: positions rounded to QUANTUM_MM about an origin (the minimum, floored to the grid). Within each
streamline the first vertex is stored absolute, the second as a first-order step, and every later vertex as the
residual of a constant-velocity prediction, p[i] - (2 p[i-1] - p[i-2]). Each streamline decodes on its own with
two running sums. Residuals go through blosc (zstd 9; bitshuffle for the body, shuffle for the head and lengths),
int8 when they fit, else int16 (head int16, else int32). Lossless relative to the grid: a decoded vertex is within
QUANTUM_MM / 2 of the original on every axis.

Field: rankfield's encode of the (n, C) log-probabilities as an (C, n, 1, 1) field, keep "clip" (streamlines have
no neighbors for "shell"), clip 8, depth FIELD_DEPTH, a tail at temperature 1. Decoded back to dense (n, C)
log-probabilities: each kept class its probability, the tail's mass spread evenly over the classes not kept.
Depth 12 holds the ensemble's mixture without loss of its ranking at tract level (docs/uncertainty-ensemble.md:
depth 6 lost 0.011 AUROC, depth 8 most of it back, 12 all).

Needs numcodecs and rankfield (the `rankfield` dependency group); imported only when a compact record is written
or read. Everything is returned as plain arrays (uint8 blobs and small metadata), for an .npz.
"""
from __future__ import annotations

import json
import numpy as np

QUANTUM_MM = 0.05
FIELD_DEPTH = 12
FIELD = dict(keep="clip", clip=8.0, tail_temperatures=(1.0,))
FORMAT = 2                        # a RunRecord file with compact geometry and field (1: dense log-probabilities, 20-point paths)


def _codec(shuffle: str):
    from numcodecs import Blosc
    return Blosc(cname="zstd", clevel=9, shuffle={"bit": Blosc.BITSHUFFLE, "byte": Blosc.SHUFFLE}[shuffle])


def _pack(a: np.ndarray, shuffle: str) -> np.ndarray:
    return np.frombuffer(_codec(shuffle).encode(np.ascontiguousarray(a)), np.uint8)


def _unpack(blob: np.ndarray, dtype: str, shape) -> np.ndarray:
    return np.frombuffer(_codec("byte").decode(np.asarray(blob, np.uint8).tobytes()), dtype).reshape(shape)


def line_index(off: np.ndarray) -> np.ndarray:
    """Each vertex's index within its streamline, from (N + 1,) offsets."""
    return np.arange(off[-1]) - np.repeat(off[:-1], np.diff(off))


def residuals(Q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Integer positions (V, 3) -> 2nd-order residuals (first vertex absolute, second a step)."""
    R = np.empty_like(Q)
    R[0] = Q[0]
    R[1:] = Q[1:] - Q[:-1]
    R2 = R.copy()
    R2[2:] = Q[2:] - 2 * Q[1:-1] + Q[:-2]
    R2[k == 1] = R[k == 1]
    R2[k == 0] = Q[k == 0]
    return R2


def _segment_cumsum(x: np.ndarray, off: np.ndarray) -> np.ndarray:
    c = np.cumsum(x, axis=0)
    before = np.vstack([np.zeros((1, x.shape[1]), x.dtype), c[off[:-1] - 1][1:]])
    return c - np.repeat(before, np.diff(off), axis=0)


def decode_residuals(R: np.ndarray, k: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Residuals -> integer positions: the reader's two running sums, per streamline."""
    first = np.repeat(R[k == 0], np.diff(off), axis=0)
    D = R.copy()
    D[k == 0] = 0
    D = _segment_cumsum(D, off)                  # the steps: D[1] is stored, later D[i] = D[i-1] + r
    D[k == 0] = 0
    return first + _segment_cumsum(D, off)


def encode_geometry(fibers, quantum: float = QUANTUM_MM) -> dict:
    """Streamlines ((m_i, 3) mm arrays, each of 1 or more points) -> {"geom_*": arrays}. Lossless on the grid."""
    lens = np.array([len(f) for f in fibers], np.int64)
    assert len(lens) and lens.min() >= 1, "every streamline needs a point"
    off = np.r_[0, np.cumsum(lens)]
    P = np.concatenate([np.asarray(f, np.float64).reshape(-1, 3) for f in fibers])
    origin = np.floor(P.min(0) / quantum) * quantum
    Q = np.round((P - origin) / quantum).astype(np.int64)
    k = line_index(off)
    R = residuals(Q, k)
    body, head = R[k >= 2], R[k < 2]
    bdt = "<i1" if len(body) == 0 or np.abs(body).max() < 128 else "<i2"
    hdt = "<i2" if np.abs(head).max() < 32768 else "<i4"
    ldt = "<u2" if lens.max() < 65536 else "<u4"
    return {"geom_meta": np.array(json.dumps({"quantum": quantum, "origin": origin.tolist(), "n": int(len(lens)), "vertices": int(off[-1]),
                                              "body": [bdt, int(len(body))], "head": [hdt, int(len(head))], "lens": ldt, "order": 2})),
            "geom_lens": _pack(lens.astype(ldt), "byte"), "geom_head": _pack(head.astype(hdt), "byte"),
            "geom_body": _pack(body.astype(bdt), "bit")}


def decode_geometry(z) -> list[np.ndarray]:
    """{"geom_*": arrays} (an .npz or dict) -> streamlines as (m_i, 3) float32 mm arrays."""
    meta = json.loads(str(z["geom_meta"]))
    lens = _unpack(z["geom_lens"], meta["lens"], (meta["n"],)).astype(np.int64)
    off = np.r_[0, np.cumsum(lens)]
    k = line_index(off)
    R = np.empty((meta["vertices"], 3), np.int64)
    R[k < 2] = _unpack(z["geom_head"], meta["head"][0], (meta["head"][1], 3))
    R[k >= 2] = _unpack(z["geom_body"], meta["body"][0], (meta["body"][1], 3))
    P = (decode_residuals(R, k, off) * meta["quantum"] + np.asarray(meta["origin"])).astype(np.float32)
    return np.split(P, off[1:-1])


def encode_field(logp, depth: int = FIELD_DEPTH) -> dict:
    """(n, C) log-probabilities (numpy or torch, any float dtype) -> {"field_*": arrays}."""
    import torch
    import rankfield as rf
    lp = torch.as_tensor(np.asarray(logp, np.float32)).T.contiguous()[:, :, None, None]   # (C, n, 1, 1)
    code = rf.encode(lp, depth=depth, **FIELD)
    ranks, support = np.asarray(code.ranks), np.asarray(code.support)
    tail = np.zeros(0, np.uint16) if code.tail is None else np.asarray(code.tail)          # None: exhaustive, nothing dropped
    return {"field_meta": np.array(json.dumps({"meta": code.meta, "classes": int(lp.shape[0]), "n": int(lp.shape[1]),
                                               "ranks": [str(ranks.dtype), list(ranks.shape)], "support": [str(support.dtype), list(support.shape)],
                                               "tail": [str(tail.dtype), list(tail.shape)]}, default=_jsonable)),
            "field_ranks": _pack(ranks, "byte"), "field_support": _pack(support, "byte"), "field_tail": _pack(tail, "byte")}


def _jsonable(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (np.ndarray, tuple)):
        return list(x)
    raise TypeError(type(x))


def _code(z):
    import rankfield as rf
    m = json.loads(str(z["field_meta"]))
    ranks = _unpack(z["field_ranks"], *m["ranks"])
    support = _unpack(z["field_support"], *m["support"])
    tail = _unpack(z["field_tail"], *m["tail"])
    return rf.RankField(ranks=ranks, support=support, tail=tail if tail.size else None, meta=m["meta"]), m


def decode_field(z) -> np.ndarray:
    """{"field_*": arrays} -> dense (n, C) float16 log-probabilities: kept classes their decoded probability, the
    tail's mass spread evenly over the rest."""
    import rankfield as rf
    code, m = _code(z)
    ids, p = rf.probabilities(code, 1.0)                             # (depth, n, 1, 1)
    ids, p = ids.reshape(ids.shape[0], -1), p.reshape(p.shape[0], -1)
    n, C = m["n"], m["classes"]
    kept = ids >= 0
    rest = np.clip(1.0 - p.sum(0), 0.0, 1.0) / np.maximum(C - kept.sum(0), 1)       # the tail, per class not kept
    out = np.repeat(rest[:, None], C, 1).astype(np.float64)
    rows = np.broadcast_to(np.arange(n), ids.shape)
    out[rows[kept], ids[kept]] = p[kept]
    return np.log(np.maximum(out, 1e-30)).astype(np.float16)
