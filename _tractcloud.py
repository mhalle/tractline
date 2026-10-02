"""TractCloud labels for a tractogram: the 40 mm cut, 15-point resampling, upstream's context (k 20,
k_global 80, numpy seeded per draw), the network on the device in float32, a majority vote over draws
of the 43 tract labels (42: Other). One draw is the pipeline; several measure TractCloud's own spread.
"""
from __future__ import annotations

import sys, types
from dataclasses import dataclass
import numpy as np, torch
from _resample import resample
from _data import DATA, MODEL, MASS_CENTER

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


class Labeler:
    def __init__(self, device="mps"):
        self.device = torch.device(device)
        self.model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), self.device,
                                       k_override=20, k_global_override=80)
        self.center = np.load(MASS_CENTER)
        self.lut = LUT.astype(np.int64)

    def __call__(self, fibers, draws=(0,), logp=False) -> Labels:
        """fibers: (n, 3) RAS mm arrays. draws: the context seeds voted over. logp: keep the first
        draw's network output (what the rank field encodes)."""
        lens = np.array([len(f) for f in fibers])
        P = np.concatenate(fibers).astype(np.float32).astype(np.float64)    # as a VTK file would hold them
        o = np.r_[0, np.cumsum(lens)]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        length = seg[o[1:] - 1] - seg[o[:-1]]
        keep = length >= MIN_LENGTH_MM
        feat = resample(torch.from_numpy(P), torch.from_numpy(o)).numpy()[keep]
        tracts, first = [], None
        for d in draws:
            np.random.seed(d)
            ds = inf.RealDataDataset(inf.center_tractography(feat, self.center), k=20, k_global=80, k_ds_rate=0.1)
            Pf = torch.from_numpy(ds.feat).float().to(self.device).transpose(2, 1).contiguous()
            L = torch.from_numpy(ds.local_feat).float().to(self.device).transpose(2, 1).contiguous()
            G = torch.from_numpy(ds.global_feat).float().to(self.device).transpose(2, 1).contiguous()
            cl, lp = [], []
            want = logp and first is None
            with torch.no_grad():
                for a in range(0, len(ds), 1024):
                    b = min(len(ds), a + 1024)
                    out = self.model(Pf[a:b], torch.cat((L[a:b], G.expand(b - a, -1, -1, -1)), 3)).view(-1, 1600)
                    cl.append(out.argmax(1).cpu())
                    if want:
                        lp.append(out.half().cpu())
            if want:
                first = torch.cat(lp)
            tracts.append(self.lut[torch.cat(cl).numpy()])
        vote = np.array([np.bincount(c, minlength=43).argmax() for c in np.stack(tracts).T])
        return Labels(keep=keep, length_mm=length[keep], tract=vote, logp=first)
