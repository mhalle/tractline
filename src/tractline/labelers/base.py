"""What every labeler shares: the result (Labels), the length cut, and streamline lengths as the
labelers measure them (docs/labelers.md)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np, torch

# the 43-class scheme TractCloud and RapidParc share (scheme_43.json): 1,600 clusters -> 42 tracts + Other
_SCHEME = json.loads((Path(__file__).with_name("scheme_43.json")).read_text())
TRACT_NAMES = tuple(_SCHEME["tract_names"])           # 0-41 the tracts, 42 "Other"
LUT = np.asarray(_SCHEME["cluster_to_tract"], np.int64)   # (1600,) cluster -> tract
OTHER = 42
MIN_LENGTH_MM = 40.0             # TractCloud's training data: streamlines of 40 mm or more


@dataclass
class Labels:
    keep: np.ndarray             # (n_fibers,) bool: the streamlines >= 40 mm, the ones labeled
    length_mm: np.ndarray        # (n_kept,)
    tract: np.ndarray            # (n_kept,) 0-41 a tract, 42 Other
    logp: torch.Tensor | None    # (n_kept, 1600) float16 cluster log-probabilities (CPU), when asked for

    def kept(self, fibers):
        return [f for f, k in zip(fibers, self.keep) if k]


def lengths(fibers):
    """(points float64 as a float32 file would hold them, offsets, length mm per streamline). Each
    streamline's length is its own segments' sum: an empty or one-point streamline is 0 mm, a NaN point
    makes only its own streamline NaN (so the 40 mm cut drops it)."""
    lens = np.array([len(f) for f in fibers], dtype=np.int64)
    o = np.r_[0, np.cumsum(lens)]
    if o[-1] == 0:
        return np.zeros((0, 3)), o, np.zeros(len(fibers))
    P = np.concatenate([np.asarray(f, np.float32).reshape(-1, 3) for f in fibers]).astype(np.float64)
    seg = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]           # seg[i]: point i-1 -> i
    seg[o[:-1][lens > 0]] = 0.0                                             # no segment across streamlines
    cum = np.cumsum(np.where(np.isnan(seg), 0.0, seg))
    bad = np.cumsum(np.isnan(seg))
    length = np.where(lens > 0, cum[np.maximum(o[1:] - 1, 0)] - cum[np.minimum(o[:-1], len(cum) - 1)], 0.0)
    nan = np.where(lens > 0, bad[np.maximum(o[1:] - 1, 0)] - bad[np.minimum(o[:-1], len(bad) - 1)], 0) > 0
    return P, o, np.where(nan, np.nan, length)


def empty(keep, length, logp=False):
    """Labels with nothing labeled (no streamline of 40 mm or more)."""
    return Labels(keep=keep, length_mm=length[keep], tract=np.zeros(0, np.int64),
                  logp=torch.zeros(0, 1600, dtype=torch.float16) if logp else None)


class LogMean:
    """The log of the mean of several draws' probabilities, from their log-probabilities, without
    underflow: logsumexp over draws minus log(draws), accumulated draw by draw (float32)."""
    def __init__(self):
        self.acc, self.n = None, 0

    def add(self, logp):
        logp = logp.float()
        self.acc = logp if self.acc is None else torch.logaddexp(self.acc, logp)
        self.n += 1

    def result(self):
        return self.acc - float(np.log(self.n))
