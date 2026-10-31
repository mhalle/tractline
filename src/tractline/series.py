"""Two full diffusion series of opposite phase encoding as one: joined for the motion/eddy step, and, when asked, their
matched volumes averaged after it (the pipeline's reversed_series slot).

    s = bids.load("sub-01/dwi/sub-01_dir-AP_dwi.nii.gz", keep_reversed=True)     # s.reversed: the PA series, in full
    m = series.merge(s)                                                          # one subject: AP then PA, per-volume signs
    pairs = series.match(s.bval, s.bvec, s.reversed.bval, s.reversed.bvec)       # what pair-average would pair

The join puts the reversed series after the diffusion series (volume 0 stays the field's frame) and scales it by one
factor, the ratio of the two series' mean b0 over the whole image: a distortion moves signal and does not change how much
of it there is, and a series' gain would otherwise read as a difference between the two polarities. The idea and the
whole-image mean are the HCP pipeline's (its intensity normalization before topup and eddy; its scripts were read for the
method, no code is taken from them).

The average is the HCP pipeline's combination, a plain mean of the corrected volumes, with one difference: HCP pairs
volumes by their place in the series, without looking at them, so one volume missing in the middle misaligns every pair
after it. Here a pair is two volumes whose b-values agree within b_tol and whose axes agree within angle_tol degrees,
found one-to-one for the whole table (antipodes alike, order free), and what does not pair is refused, or left out or
kept alone, as asked and said. The b0s, having no direction, pair in order; the extra b0s of the longer series stay,
single. The b-vector of a pair: the mean of its two rotated vectors, turned to agree, normalized; its b-value: the mean.

numpy only (scipy for the assignment, as motion_eddy needs it).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

B0 = 50.0                        # b < this: a b0, as everywhere in tractline


def axis_sets(g, deg=5.0) -> list:
    """Unit vectors g (n, 3) gathered by axis: a list of index lists, antipodes alike, each vector with the first of its
    set within `deg` degrees (greedy). A series of matched pairs has half as many sets as volumes."""
    cos = math.cos(math.radians(deg))
    g = np.asarray(g, np.float64)
    sets, heads = [], []
    for i, v in enumerate(g):
        near = np.abs(np.asarray(heads) @ v) if heads else np.zeros(0)
        if len(near) and float(near.max()) >= cos:
            sets[int(near.argmax())].append(i)
        else:
            heads.append(v)
            sets.append([i])
    return sets


def _units(bvec):
    g = np.asarray(bvec, np.float64).T
    return g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-9, None)


@dataclass
class Match:
    """match()'s result. Indices are into each series' own volumes; pairs are in the first series' order."""
    pairs: list                                    # [(i, j)]: diffusion-weighted, the two volumes to average
    b0_pairs: list                                 # [(i, j)]: b0s, in order
    unmatched_a: list = field(default_factory=list)      # diffusion-weighted volumes of the first series with no match
    unmatched_b: list = field(default_factory=list)      # ... of the second
    b0_alone_a: list = field(default_factory=list)       # the longer series' extra b0s
    b0_alone_b: list = field(default_factory=list)
    worst_angle_deg: float = 0.0                         # the largest angle within a pair
    worst_b: float = 0.0                                 # the largest b-value difference within a pair


def match(bval_a, bvec_a, bval_b, bvec_b, b_tol=50.0, angle_tol=5.0) -> Match:
    """The pairs of two gradient tables, one-to-one: the assignment of smallest total angle among the volumes whose
    b-values differ by at most b_tol (s/mm^2) and whose axes by at most angle_tol (degrees; antipodes alike: g and -g
    weigh the same signal). In a common frame: bids.load gives the reversed series' b-vectors in the first series'."""
    from scipy.optimize import linear_sum_assignment
    ba, bb = np.asarray(bval_a, np.float64), np.asarray(bval_b, np.float64)
    ga, gb = _units(bvec_a), _units(bvec_b)
    a0, b0 = np.flatnonzero(ba < B0), np.flatnonzero(bb < B0)
    n0 = min(len(a0), len(b0))
    b0_pairs = [(int(i), int(j)) for i, j in zip(a0[:n0], b0[:n0])]
    a, b = np.flatnonzero(ba >= B0), np.flatnonzero(bb >= B0)
    pairs = []
    worst_a = worst_b = 0.0
    if len(a) and len(b):
        ang = np.degrees(np.arccos(np.abs(ga[a] @ gb[b].T).clip(0, 1)))
        ok = (ang <= angle_tol) & (np.abs(ba[a][:, None] - bb[b][None]) <= b_tol)
        cost = np.where(ok, ang, 1e6)
        for i, j in zip(*linear_sum_assignment(cost)):
            if ok[i, j]:
                pairs.append((int(a[i]), int(b[j])))
                worst_a = max(worst_a, float(ang[i, j]))
                worst_b = max(worst_b, abs(float(ba[a[i]] - bb[b[j]])))
    pairs.sort()
    got_a, got_b = {i for i, _ in pairs}, {j for _, j in pairs}
    return Match(pairs, b0_pairs, [int(i) for i in a if int(i) not in got_a], [int(j) for j in b if int(j) not in got_b],
                 [int(i) for i in a0[n0:]], [int(j) for j in b0[n0:]], round(worst_a, 3), round(worst_b, 3))


def check_match(m: Match, unmatched: str, b_tol: float, angle_tol: float) -> None:
    """pair-average's refusal, before any work: unmatched "refuse" and diffusion-weighted volumes without a partner."""
    if unmatched == "refuse" and (m.unmatched_a or m.unmatched_b):
        raise ValueError(
            f"pair-average: {len(m.pairs)} pairs, but {len(m.unmatched_a)} volumes of the diffusion series and "
            f"{len(m.unmatched_b)} of the reversed one have no partner within {b_tol:g} s/mm^2 and {angle_tol:g} degrees "
            f"(the first of them: diffusion series {m.unmatched_a[:8]}, reversed {m.unmatched_b[:8]}): use concatenate, or the slot's option "
            "unmatched 'keep' (they stay, single) or 'drop' (they go), or wider tolerances if the tables really are the same")


def intensity_scale(dwi_a, bval_a, dwi_b, bval_b) -> float:
    """What the second series is multiplied by to give it the first's mean b0 over the whole image (module docstring)."""
    def mean_b0(d, b):
        vols = [np.asarray(d[..., int(v)], np.float64) for v in np.flatnonzero(np.asarray(b) < B0)]
        return float(np.mean([v[np.isfinite(v)].mean() for v in vols]))
    a, b = mean_b0(dwi_a, bval_a), mean_b0(dwi_b, bval_b)
    if not (a > 0 and b > 0):
        raise ValueError("reversed series: a series' mean b0 is not positive")
    return a / b


def merge(s):
    """A copy of subject s (with s.reversed, bids.ReversedSeries) as one series, the diffusion series first: dwi
    (float32), bval and bvec joined, pe_signs and readouts_s per volume (motion_eddy.estimate reads them), n_first its
    first series' volumes, reversed_scale the factor applied. s.b0s and everything else as they were: the field is
    estimated as for the diffusion series alone."""
    r = s.reversed
    k = intensity_scale(s.dwi, s.bval, r.dwi, r.bval)
    G1, G2 = s.dwi.shape[-1], r.dwi.shape[-1]
    out = copy.copy(s)
    d = np.empty(s.dwi.shape[:3] + (G1 + G2,), np.float32)
    d[..., :G1] = s.dwi
    d[..., G1:] = np.asarray(r.dwi, np.float32) * np.float32(k)
    out.dwi = d
    out.bval = np.r_[np.asarray(s.bval, np.float64), np.asarray(r.bval, np.float64)]
    out.bvec = np.c_[np.asarray(s.bvec, np.float64), np.asarray(r.bvec, np.float64)]
    out.pe_signs = np.r_[np.full(G1, float(s.pe_sign)), np.full(G2, float(r.pe_sign))]
    out.readouts_s = np.r_[np.full(G1, float(s.readout_s)), np.full(G2, float(r.readout_s))]
    out.n_first, out.reversed_scale = G1, k
    out.reversed = None
    return out


def average(dwi, bval, bvec, n_first: int, m: Match, unmatched="refuse", left_out=()) -> tuple:
    """The corrected, merged series (X, Y, Z, G), its b-values and rotated b-vectors (3, G) -> the series with each pair
    averaged: (dwi, bval, bvec, a record). In the first series' order, then what stays of the second's. left_out: merged
    indices left out of the fit: a pair that lost one volume keeps the other, single; a pair that lost both, or a
    single volume left out, goes. unmatched: "keep", the unmatched diffusion-weighted volumes stay, single; else they
    go (check_match has refused them for "refuse")."""
    gone = {int(v) for v in left_out}
    a_alone = list(m.b0_alone_a) + (list(m.unmatched_a) if unmatched == "keep" else [])
    b_alone = list(m.b0_alone_b) + (list(m.unmatched_b) if unmatched == "keep" else [])
    entries = sorted([(i, n_first + j) for i, j in m.pairs + m.b0_pairs] + [(i, None) for i in a_alone]) \
        + [(None, n_first + j) for j in sorted(b_alone)]
    kept, single, dropped = [], 0, 0
    for i, j in entries:
        have = [v for v in (i, j) if v is not None and v not in gone]
        if not have:
            dropped += 1
            continue
        single += len(have) == 1 and i is not None and j is not None
        kept.append(have)
    out = np.empty(dwi.shape[:3] + (len(kept),), np.float32)
    b = np.empty(len(kept))
    g = np.zeros((3, len(kept)))
    bvec = np.asarray(bvec, np.float64)
    bval = np.asarray(bval, np.float64)
    for n, have in enumerate(kept):
        if len(have) == 1:
            out[..., n], b[n], g[:, n] = dwi[..., have[0]], bval[have[0]], bvec[:, have[0]]
        else:
            v, w = have
            out[..., n] = (dwi[..., v] + dwi[..., w]) / 2
            b[n] = (bval[v] + bval[w]) / 2
            if b[n] >= B0:
                u = bvec[:, v] + np.sign(float(bvec[:, v] @ bvec[:, w]) or 1.0) * bvec[:, w]
                g[:, n] = u / max(float(np.linalg.norm(u)), 1e-9)
    rec = {"pairs averaged (diffusion-weighted)": len(m.pairs), "b0 pairs averaged": len(m.b0_pairs),
           "volumes in the result": len(kept), "pairs that lost a volume to the fit (kept single)": int(single),
           "entries lost to the fit (gone)": int(dropped),
           "b0s alone (the longer series' extras)": len(m.b0_alone_a) + len(m.b0_alone_b),
           "unmatched diffusion-weighted volumes": {"diffusion series": list(m.unmatched_a), "reversed": list(m.unmatched_b),
                                                    "what happened": {"keep": "kept, single", "drop": "left out",
                                                                      "refuse": "none"}[unmatched]},
           "largest angle in a pair (deg)": m.worst_angle_deg, "largest b-value difference in a pair": m.worst_b}
    return out, b, g, rec
