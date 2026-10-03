"""TractCloud's streamline features, batched: arc-length resampling of every streamline to m points,
on any torch device, in chunks of streamlines padded to a rectangle.

Upstream (`tractcloud.inference.extract_ras_features`) loops over streamline lengths and, per
streamline, takes its cumulative arc length from 0, `searchsorted(cum_len, t, side="right") - 1`
clipped to [0, n-2], and interpolates linearly; a streamline under 1e-12 long repeats its first
point, and a segment under 1e-12 interpolates over a length of 1. This is the same rule for a whole
chunk at once: each row is summed from zero as upstream sums it (one global cumulative sum would
reach tens of millions of mm, and differences of those lose the last bits), and the search is
torch's batched searchsorted. In float64 it matches upstream to the last bit (`resample_check.py`).

The TractCloud labeler uses it (on the CPU, float64, from labelers.base.lengths' flat points and
offsets); RapidParc resamples by index instead (labelers.rapidparc.resample).
"""
import numpy as np
import torch


def resample(P: torch.Tensor, off: torch.Tensor, m: int = 15, chunk: int = 65536) -> torch.Tensor:
    """P (V, 3) vertices of N streamlines, off (N + 1,) offsets -> (N, m, 3) at equal arc length."""
    N = off.numel() - 1
    out = torch.empty((N, m, 3), dtype=P.dtype, device=P.device)
    # numpy's linspace, not torch's: torch computes the interior points from both ends, and the
    # 15 fractions differ in the last bit, which moves every interpolated point
    frac = torch.from_numpy(np.linspace(0, 1, m)).to(dtype=P.dtype, device=P.device)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        o = off[s:e + 1]
        n = o[1:] - o[:-1]                                            # (c,) vertices per streamline
        L = int(n.max())
        col = torch.arange(L, device=P.device)
        valid = col[None, :] < n[:, None]                             # (c, L)
        gi = (o[:-1, None] + col[None, :]).clamp(max=P.shape[0] - 1)
        X = P[gi]                                                     # (c, L, 3), padding repeats real points
        d = X[:, 1:] - X[:, :-1]
        seg = ((d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2]).sqrt()   # upstream's order
        seg = torch.where(valid[:, 1:], seg, torch.zeros_like(seg))
        cum = torch.cat([seg.new_zeros(e - s, 1), seg.cumsum(1)], 1)  # (c, L), each row from 0
        total = cum.gather(1, (n - 1).clamp_min(0)[:, None])[:, 0]
        cum_s = torch.where(valid, cum, torch.full_like(cum, float("inf")))   # padding never found
        t = frac[None, :] * total[:, None]                             # (c, m)
        idx = torch.searchsorted(cum_s, t, right=True) - 1
        idx = torch.minimum(idx.clamp_min(0), (n - 2).clamp_min(0)[:, None])
        lo, hi = cum.gather(1, idx), cum.gather(1, idx + 1)
        sl = hi - lo
        sl = torch.where(sl < 1e-12, torch.ones_like(sl), sl)
        w = ((t - lo) / sl)[..., None]
        a = X.gather(1, idx[..., None].expand(-1, -1, 3))
        b = X.gather(1, (idx + 1)[..., None].expand(-1, -1, 3))
        r = a + w * (b - a)
        degenerate = total < 1e-12
        r[degenerate] = X[degenerate, :1].expand(-1, m, -1)
        r[n < 2] = 0                                                  # upstream skips these: they stay zero
        out[s:e] = r
    return out
