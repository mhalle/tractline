"""What every labeler shares: the result (Labels), the length cut, and streamline lengths as the
labelers measure them (docs/labelers.md)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np, torch

OTHER = 42                       # the 43-class scheme TractCloud and RapidParc share: 0-41 a tract, 42 Other
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
    """(points float64 as a float32 file would hold them, offsets, length mm per streamline)."""
    lens = np.array([len(f) for f in fibers])
    P = np.concatenate(fibers).astype(np.float32).astype(np.float64)
    o = np.r_[0, np.cumsum(lens)]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return P, o, seg[o[1:] - 1] - seg[o[:-1]]
