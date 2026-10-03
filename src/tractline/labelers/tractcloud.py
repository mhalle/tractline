"""TractCloud labels for a tractogram (TractCloud's network and weights, upstream's model code; the
context built here): the 40 mm cut, 15-point resampling, the context the released model was trained
with (trained_context: each streamline, then its 19 nearest among 10,000 candidates drawn from the whole
tractogram; 500 global streamlines shared by the draw; numpy seeded per draw), the network on the device
in float32, cluster probabilities averaged over draws, then the 43 tract labels (42: Other).

Upstream's packaged inference overrides the context to k_global 80 and a 10 % local subsample taken
within consecutive ~10,000-streamline chunks (slabs of the brain, in seed order). On TractCloud's own
test split that costs 5.5 points of tract accuracy (86.6 % against 92.0 %); on PAT16 one draw's Other
fraction ranged 55.5-66.2 % (NOTES 2026-10-02). upstream=True restores upstream's context exactly, for
comparison. The pipeline's default labeler is RapidParc (labelers/rapidparc.py); this one is optional.
"""
from __future__ import annotations

import sys, types
import numpy as np, torch
import torch.nn.functional as F
from ..resample import resample
from .base import Labels, MIN_LENGTH_MM, OTHER, LogMean, empty, lengths      # noqa: F401 - shared by the labelers; re-exported here
from ..data import DATA, MODEL, MASS_CENTER

import os
# Importing this module: TractCloud's code goes on sys.path (DATA/TractCloud/src) and TRACTCLOUD_DATA_DIR
# defaults to DATA (its own model-cache lookup). Its modules import vtk for file input/output this
# labeler does not use: when vtk is not already imported, a stub stands in during TractCloud's import
# and is removed after it, so a later `import vtk` gets the real one.
os.environ.setdefault("TRACTCLOUD_DATA_DIR", str(DATA))
sys.path.insert(0, str(DATA / "TractCloud/src"))
_stub = None if "vtk" in sys.modules else sys.modules.setdefault("vtk", types.ModuleType("vtk"))
try:
    from tractcloud import inference as inf
    from tractcloud.tract_mapping import _CLUSTER_TO_TRACT_LUT as LUT
finally:
    if _stub is not None and sys.modules.get("vtk") is _stub:
        del sys.modules["vtk"]



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


K_GLOBAL, LOCAL_CANDIDATES, K = 500, 10_000, 20   # as trained (TrainedModel/cli_args.txt: k 20, k_global 500, k_ds_rate 1.0 of 10,000)


def trained_context(feat, seed, k_global=K_GLOBAL, candidates=LOCAL_CANDIDATES, chunk=4096):
    """The context the released model was trained with, for (n, 15, 3) float32 features: (feat, local
    (n, 15, 3, k), global (1, 15, 3, k_global)) in RealDataDataset's layouts. Global: k_global streamlines
    drawn with replacement (np.random.seed(seed), randint, as upstream). Local: each streamline itself,
    then its k - 1 nearest (upstream's distance) among ONE random set of min(n, candidates) streamlines
    from the whole tractogram - training's: k nearest among all of a 10,000-streamline brain, itself
    first at distance 0. Upstream's RealDataDataset instead subsamples within consecutive index chunks of
    ~10,000 (seed order: slabs of the brain), so at k_ds_rate r each streamline sees ~r x 10,000 candidates
    from its own slab - at 10,000 streamlines or fewer the two are the same."""
    n = len(feat)
    np.random.seed(seed)
    glob = feat[np.random.randint(0, n, k_global)].transpose(1, 2, 0)[None].astype(np.float32)
    cand = np.sort(np.random.choice(n, size=min(n, candidates), replace=False))
    ft = torch.from_numpy(np.ascontiguousarray(feat.transpose(0, 2, 1)))           # (n, 3, 15)
    cf = ft[cand]
    local = np.empty((n, feat.shape[1], 3, K), np.float32)
    cand_t = torch.from_numpy(cand)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = inf._fiber_distance_efficient(ft[a:b], cf)                             # (m, C)
        d[cand_t[None, :] == torch.arange(a, b)[:, None]] = float("inf")            # itself comes first, below
        nn = d.topk(k=K - 1, largest=False, dim=-1)[1]
        nb = torch.cat([ft[a:b, None], cf[nn]], 1)                                  # (m, k, 3, 15)
        local[a:b] = nb.permute(0, 3, 2, 1).numpy()
    return feat.astype(np.float32), local, glob


class Labeler:
    def __init__(self, device="mps", exact=None, upstream=False, batch=256):
        """exact: upstream's forward (default on the GPU); otherwise MatmulDGCNN (default on the CPU).
        upstream: upstream's inference context exactly (k_global 80, its RealDataDataset with a 10 % local
        subsample per chunk) instead of the trained one (trained_context). batch: streamlines per forward
        pass - the same labels at any size (checked at 256 and 1,024); 1,024 peaked at 5.9 GB on the M2's
        GPU, 256 at 2.1 GB, as fast."""
        self.device, self.batch = torch.device(device), batch
        self.k_global = 80 if upstream else K_GLOBAL
        self.upstream = upstream
        self.model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), self.device,
                                       k_override=K, k_global_override=self.k_global)
        if not (self.device.type != "cpu" if exact is None else exact):
            with torch.no_grad():
                self.model = MatmulDGCNN(self.model.eval())
        self.center = np.load(MASS_CENTER)
        self.lut = LUT.astype(np.int64)

    def context(self, feat, seed):
        """(feat, local, global) for one draw of the centered (n, 15, 3) features."""
        if self.upstream:
            np.random.seed(seed)
            ds = inf.RealDataDataset(feat, k=K, k_global=self.k_global, k_ds_rate=0.1)
            return ds.feat, ds.local_feat, ds.global_feat
        return trained_context(feat, seed, self.k_global)

    @torch.no_grad()
    def log_probs(self, feat, seed):
        """(n, 1600) cluster log-probabilities (the network's log-softmax, on the device) of one draw, by batch."""
        Fe, Lo, Gl = self.context(feat, seed)
        Pf = torch.from_numpy(Fe).float().to(self.device).transpose(2, 1).contiguous()
        L = torch.from_numpy(Lo).float().to(self.device).transpose(2, 1).contiguous()
        G = torch.from_numpy(Gl).float().to(self.device).transpose(2, 1).contiguous()
        for a in range(0, len(Fe), self.batch):
            b = min(len(Fe), a + self.batch)
            yield a, self.model(Pf[a:b], torch.cat((L[a:b], G.expand(b - a, -1, -1, -1)), 3)).view(-1, 1600)

    def __call__(self, fibers, draws=(0,), logp=False) -> Labels:
        """fibers: (n, 3) RAS mm arrays. draws: the context seeds (any iterable) whose cluster probabilities
        are averaged. logp: keep the cluster log-probabilities (float16; several draws: the log of their
        mean probability, without underflow)."""
        P, o, length = lengths(fibers)                                     # points as a float32 file would hold them
        keep = length >= MIN_LENGTH_MM
        draws = tuple(draws)
        if not keep.any():
            return empty(keep, length, logp)
        if keep.sum() < (K / 0.1 if self.upstream else K):
            raise ValueError(f"TractCloud needs at least {int(K / 0.1) if self.upstream else K} streamlines of {MIN_LENGTH_MM:g} mm "
                             f"or more for its local context; got {int(keep.sum())}")
        feat = inf.center_tractography(resample(torch.from_numpy(P), torch.from_numpy(o)).numpy()[keep], self.center)
        n = len(feat)
        if len(draws) == 1:
            arg = torch.empty(n, dtype=torch.int64)
            lp = torch.empty(n, 1600, dtype=torch.float16) if logp else None
            for a, out in self.log_probs(feat, draws[0]):
                arg[a:a + len(out)] = out.argmax(1).cpu()
                if logp:
                    lp[a:a + len(out)] = out.half().cpu()
        else:
            mean = LogMean()
            for d in draws:
                full = torch.empty(n, 1600)
                for a, out in self.log_probs(feat, d):
                    full[a:a + len(out)] = out.float().cpu()
                mean.add(full)
            m = mean.result()
            arg, lp = m.argmax(1), (m.half() if logp else None)
        return Labels(keep=keep, length_mm=length[keep], tract=self.lut[arg.numpy()], logp=lp)
