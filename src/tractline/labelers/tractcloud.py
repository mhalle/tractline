"""TractCloud labels for a tractogram: the 40 mm cut, 15-point resampling, the context the released
model was trained with (k 20 local neighbors among 10,000 candidate streamlines, k_global 500 global
streamlines shared by the draw; numpy seeded per draw), the network on the device in float32, cluster
probabilities averaged over draws, then the 43 tract labels (42: Other).

Upstream's packaged inference overrides the context to k_global 80 and a 10 % local subsample. On
PAT16 that made one draw's Other fraction range 55.5-66.2 % (SD 3.5 points; at 500: 50.6-51.6 %, SD
0.3) and two single draws agree on 75-88 % of streamlines (at 500: 93-94 %); it also shifts the labels
(+6-7 points of Other). The global sample drives it: one set conditions every streamline of a draw.
The local setting barely matters (NOTES 2026-10-02, "Why TractCloud's labels move"). upstream=True
restores upstream's inference settings, for comparison.
"""
from __future__ import annotations

import sys, types
from dataclasses import dataclass
import numpy as np, torch
import torch.nn.functional as F
from ..resample import resample
from ..data import DATA, MODEL, MASS_CENTER

sys.path.insert(0, str(DATA / "TractCloud/src")); sys.modules.setdefault("vtk", types.ModuleType("vtk"))
from tractcloud import inference as inf
from tractcloud.tract_mapping import TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as LUT

OTHER = 42
MIN_LENGTH_MM = 40.0


@dataclass
class Labels:
    keep: np.ndarray             # (n_fibers,) bool: the streamlines >= 40 mm, the ones labeled
    length_mm: np.ndarray        # (n_kept,)
    tract: np.ndarray            # (n_kept,) the vote over draws: 0-41 a tract, 42 Other
    logp: torch.Tensor | None    # (n_kept, 1600) float16, the first draw's cluster log-probabilities (CPU)

    def kept(self, fibers):
        return [f for f, k in zip(fibers, self.keep) if k]


def _fold(conv, bn):
    """A 1x1 convolution and its BatchNorm (eval) as one affine map: (W (out, in), bias (out,))."""
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return (conv.weight.reshape(conv.weight.shape[0], -1) * s[:, None]).contiguous(), bn.bias - bn.running_mean * s


class MatmulDGCNN(torch.nn.Module):
    """TractCloud's TractDGCNN, its weights, its arithmetic up to rounding, for the CPU: every 1x1
    convolution with its BatchNorm as one matrix product (PyTorch's CPU 1x1 convolution without oneDNN
    is ~50x slower than BLAS: 1,024 against 21 ms for conv5 on the M2), and each edge convolution
    W [f_j - x_i; x_i] split as Wa f_j + (Wb - Wa) x_i, so the neighbors' term is computed once per
    point and gathered. The max over neighbors is taken before adding the point's term and before the
    LeakyReLU: both are monotone, so the max commutes with them exactly."""
    def __init__(self, m):
        super().__init__()
        self.k, self.kp = m.fiber_level_k + m.fiber_level_k_global, m.k_point_level
        self.edge = []
        for c in (m.conv1, m.conv2, m.conv3, m.conv4):
            W, b = _fold(c[0], c[1])
            n = W.shape[1] // 2
            self.edge.append((W[:, :n].T.contiguous(), (W[:, n:] - W[:, :n]).T.contiguous(), b))
        W5, self.b5 = _fold(m.conv5[0], m.conv5[1]); self.W5 = W5.T.contiguous()
        self.head = m

    def _graph(self, h, Wa, Wd, b):
        """h (B, P, C) points' features -> (B, P, out): an edge convolution over each point's kp nearest."""
        from tractcloud.models import tract_knn
        idx = tract_knn(h.transpose(1, 2), k=self.kp)                    # (B, P, kp), upstream's neighbors
        a = h @ Wa                                                       # (B, P, out): every point's neighbor term, once
        nb = torch.gather(a, 1, idx.reshape(len(h), -1, 1).expand(-1, -1, a.shape[-1])).reshape(*idx.shape, -1)
        return F.leaky_relu(nb.amax(2) + h @ Wd + b, 0.2)

    def forward(self, x, info_point_set):
        """x (B, 3, P), info_point_set (B, 3, P, k): as TractDGCNN.forward; log-softmax (B, classes)."""
        Wa, Wd, b = self.edge[0]
        pts = x.transpose(1, 2)                                          # (B, P, 3)
        ctx = info_point_set.permute(0, 2, 3, 1)                         # (B, P, k, 3)
        h1 = F.leaky_relu((ctx @ Wa).amax(2) + pts @ Wd + b, 0.2)        # (B, P, 64)
        h2 = self._graph(h1, *self.edge[1])
        h3 = self._graph(h2, *self.edge[2])
        h4 = self._graph(h3, *self.edge[3])
        h = F.leaky_relu(torch.cat((h1, h2, h3, h4), 2) @ self.W5 + self.b5, 0.2)   # (B, P, 1024)
        m = self.head
        z = torch.cat((h.amax(1), h.mean(1)), 1)
        z = F.leaky_relu(m.bn6(m.linear1(z)), 0.2)
        z = F.leaky_relu(m.bn7(m.linear2(z)), 0.2)
        return F.log_softmax(m.linear3(z), dim=1)


K_GLOBAL, LOCAL_CANDIDATES = 500, 10_000            # as trained (TrainedModel/cli_args.txt: k_global 500, k_ds_rate 1.0 of 10,000)


class Labeler:
    def __init__(self, device="mps", exact=None, upstream=False):
        """exact: upstream's forward (default on the GPU); otherwise MatmulDGCNN (default on the CPU).
        upstream: upstream's inference context (k_global 80, 10 % local subsample) instead of the trained one."""
        self.device = torch.device(device)
        self.k_global = 80 if upstream else K_GLOBAL
        self.upstream = upstream
        self.model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), self.device,
                                       k_override=20, k_global_override=self.k_global)
        if not (self.device.type != "cpu" if exact is None else exact):
            with torch.no_grad():
                self.model = MatmulDGCNN(self.model.eval())
        self.center = np.load(MASS_CENTER)
        self.lut = LUT.astype(np.int64)

    def __call__(self, fibers, draws=(0,), logp=False) -> Labels:
        """fibers: (n, 3) RAS mm arrays. draws: the context seeds whose cluster probabilities are
        averaged. logp: keep the log of the averaged probabilities (one draw: the network's output)."""
        lens = np.array([len(f) for f in fibers])
        P = np.concatenate(fibers).astype(np.float32).astype(np.float64)    # as a VTK file would hold them
        o = np.r_[0, np.cumsum(lens)]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        length = seg[o[1:] - 1] - seg[o[:-1]]
        keep = length >= MIN_LENGTH_MM
        feat = resample(torch.from_numpy(P), torch.from_numpy(o)).numpy()[keep]
        ds_rate = 0.1 if self.upstream else min(1.0, LOCAL_CANDIDATES / max(len(feat), 1))
        centered = inf.center_tractography(feat, self.center)
        one = len(draws) == 1
        argmax = psum = first = None
        for d in draws:
            np.random.seed(d)
            ds = inf.RealDataDataset(centered, k=20, k_global=self.k_global, k_ds_rate=ds_rate)
            Pf = torch.from_numpy(ds.feat).float().to(self.device).transpose(2, 1).contiguous()
            L = torch.from_numpy(ds.local_feat).float().to(self.device).transpose(2, 1).contiguous()
            G = torch.from_numpy(ds.global_feat).float().to(self.device).transpose(2, 1).contiguous()
            parts = []
            with torch.no_grad():
                for a in range(0, len(ds), 1024):
                    b = min(len(ds), a + 1024)
                    out = self.model(Pf[a:b], torch.cat((L[a:b], G.expand(b - a, -1, -1, -1)), 3)).view(-1, 1600)
                    # one draw: its argmax and (if asked) its log-probabilities as float16; several: probabilities summed
                    parts.append((out.argmax(1).cpu(), out.half().cpu() if logp else None) if one else out.float().exp().cpu())
            if one:
                argmax = torch.cat([c for c, _ in parts])
                first = torch.cat([h for _, h in parts]) if logp else None
            else:
                p = torch.cat(parts)
                psum = p if psum is None else psum.add_(p)
        if not one:
            argmax = psum.argmax(1)
            first = (psum / len(draws)).log().half() if logp else None
        return Labels(keep=keep, length_mm=length[keep], tract=self.lut[argmax.numpy()], logp=first)
