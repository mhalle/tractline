"""TractCloud's context, two ways, on the full HCP subject: upstream's (kNN within a 10 % draw of
each ~10 k-streamline file chunk) against albula-diffusion's (kNN within ONE 10 % draw of the whole
brain; rkikinis/albula-diffusion tractcloud/tractcloud.ts `draw`/`localNeighbors`, ac1644e).

    modal run bench/tractography/modal_context_variant.py

The network, weights and features are upstream's; only the context differs. The whole-brain variant
runs with three seeds. Upstream's five seeded runs are the ones derived.npz holds (modal_capture.py).
Compared: tract-label agreement within each scheme (its own run-to-run noise) and between them; the
share labeled Other; and whether run 0's tract margin predicts a cross-scheme change as it predicts an
upstream run-to-run change (AUROC). Writes results/context_variant.json.
"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.7", "numpy>=2", "scikit-learn")
         .env({"PYTHONPATH": "/root/pkg"})
         .add_local_dir(str(DATA / "TractCloud/src/tractcloud"), remote_path="/root/pkg/tractcloud"))
vol = modal.Volume.from_name("tractography-bench")
app = modal.App("tractography-context-variant", image=image)
V = "/vol"


@app.function(gpu="A10G", volumes={V: vol}, timeout=3600, memory=49152, cpu=8)
def run(seeds=(1, 2, 3)) -> dict:
    import sys, types
    import numpy as np, torch
    from sklearn.metrics import roc_auc_score
    sys.modules.setdefault("vtk", types.ModuleType("vtk"))
    from tractcloud import inference as inf
    from tractcloud.tract_mapping import _CLUSTER_TO_TRACT_LUT as LUT
    dev = torch.device("cuda")
    torch.backends.cudnn.enabled = False                                   # upstream's arithmetic, as derived.npz
    feat = np.load(f"{V}/hcp/feat.npy")
    centered = inf.center_tractography(feat, np.load(f"{V}/TrainData_800clu800ol/HCP_mass_center.npy")).astype(np.float32)
    N = len(centered)
    model, _ = inf.load_model(f"{V}/TrainedModel/best_tract_f1_model.pth", f"{V}/TrainedModel/cli_args.txt",
                              dev, k_override=20, k_global_override=80)
    F = torch.from_numpy(centered).to(dev)                                 # (N, 15, 3)
    flat = F.reshape(N, -1)
    P = F.transpose(2, 1).contiguous()                                     # (N, 3, 15)

    def whole_brain_context(seed):
        g = np.random.default_rng(seed)
        ds = np.sort(g.choice(N, int(N * 0.1), replace=False))            # one 10 % sample of the whole brain
        glob = g.integers(0, N, 80)                                        # 80 global streamlines, with repeats
        dsf = flat[torch.from_numpy(ds).to(dev)]
        sq = dsf.square().sum(1)
        nn = torch.empty((N, 20), dtype=torch.long, device=dev)
        for s in range(0, N, 8192):
            e = min(N, s + 8192)
            d = flat[s:e].square().sum(1, keepdim=True) + sq[None] - 2 * flat[s:e] @ dsf.T
            nn[s:e] = torch.from_numpy(ds).to(dev)[d.topk(20, largest=False, dim=1).indices]
        L = F[nn].permute(0, 3, 2, 1)                                          # (N, 3, 15, 20) as upstream's local
        G = F[torch.from_numpy(glob).to(dev)].permute(2, 1, 0)[None]           # (1, 3, 15, 80)
        return L, G

    lut = torch.from_numpy(LUT.astype(np.int64)).to(dev)
    tracts = []
    with torch.no_grad():
        for seed in seeds:
            L, G = whole_brain_context(seed)
            lab = torch.empty(N, dtype=torch.long, device=dev)
            for s in range(0, N, 1024):
                e = min(N, s + 1024)
                info = torch.cat((L[s:e], G.expand(e - s, -1, -1, -1)), dim=3)
                lab[s:e] = model(P[s:e], info).argmax(1)
            tracts.append(lut[lab].cpu().numpy())
            del L, G
    der = np.load(f"{V}/hcp_full/derived.npz")
    up = LUT[der["cluster"].astype(np.int64)]                              # (5, N) upstream tracts
    wb = np.stack(tracts)                                                  # (3, N) whole-brain-context tracts
    agree = lambda a, b: round(float((a == b).mean()), 4)
    pairs = lambda X: [agree(X[i], X[j]) for i in range(len(X)) for j in range(i + 1, len(X))]
    cross = [agree(u, w) for u in up for w in wb]
    other = 42
    y = (wb != up[0]).any(0)                                               # a whole-brain run disagrees with upstream run 0
    m = der["m_mass_tract"][0]
    return {"streamlines": N, "upstream_pairwise_agreement": pairs(up), "whole_brain_pairwise_agreement": pairs(wb),
            "cross_scheme_agreement": cross,
            "mean": {"upstream": round(float(np.mean(pairs(up))), 4), "whole_brain": round(float(np.mean(pairs(wb))), 4),
                     "cross": round(float(np.mean(cross)), 4)},
            "other_fraction": {"upstream": [round(float((u == other).mean()), 4) for u in up],
                               "whole_brain": [round(float((w == other).mean()), 4) for w in wb]},
            "auroc_run0_mass_margin_for_cross_change": round(float(roc_auc_score(y, -m)), 4),
            "cross_changed_fraction": round(float(y.mean()), 4)}


@app.local_entrypoint()
def main():
    r = run.remote()
    print(json.dumps(r, indent=1))
    (HERE / "results" / "context_variant.json").write_text(json.dumps(r, indent=1))
