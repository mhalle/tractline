"""TractCloud labels for a tractogram, as pat16_topup_compare.py takes them: the 40 mm cut, 15-point
resampling, upstream's context (k 20, k_global 80, seeded per draw), the network on MPS in float32, a
majority vote over draws of the 43 tract labels (42: Other)."""
import sys, types
import numpy as np, torch
from _resample import resample
from _data import DATA as TD, MODEL, MASS_CENTER

sys.path.insert(0, str(TD / "TractCloud/src")); sys.modules.setdefault("vtk", types.ModuleType("vtk"))
from tractcloud import inference as inf
from tractcloud.tract_mapping import TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as LUT


class Labeler:
    def __init__(self, dev=torch.device("mps")):
        self.dev = dev
        self.model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), dev, k_override=20, k_global_override=80)
        self.center = np.load(MASS_CENTER)
        self.lut = LUT.astype(np.int64)

    def __call__(self, fibers, draws=range(5)):
        """(vote (n,), kept fibers, their lengths mm): the streamlines >= 40 mm only."""
        lens = np.array([len(f) for f in fibers])
        P = np.concatenate(fibers).astype(np.float32).astype(np.float64)
        o = np.r_[0, np.cumsum(lens)]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        length = seg[o[1:] - 1] - seg[o[:-1]]
        keep = length >= 40
        feat = resample(torch.from_numpy(P), torch.from_numpy(o)).numpy()[keep]
        out = []
        for s in draws:
            np.random.seed(s)
            ds = inf.RealDataDataset(inf.center_tractography(feat, self.center), k=20, k_global=80, k_ds_rate=0.1)
            Pf = torch.from_numpy(ds.feat).float().to(self.dev).transpose(2, 1).contiguous()
            L = torch.from_numpy(ds.local_feat).float().to(self.dev).transpose(2, 1).contiguous()
            G = torch.from_numpy(ds.global_feat).float().to(self.dev).transpose(2, 1).contiguous()
            cl = []
            with torch.no_grad():
                for a in range(0, len(ds), 1024):
                    b = min(len(ds), a + 1024)
                    cl.append(self.model(Pf[a:b], torch.cat((L[a:b], G.expand(b - a, -1, -1, -1)), 3)).view(-1, 1600).argmax(1).cpu())
            out.append(self.lut[torch.cat(cl).numpy()])
        vote = np.array([np.bincount(c, minlength=43).argmax() for c in np.stack(out).T])
        return vote, [f for f, k in zip(fibers, keep) if k], length[keep]
