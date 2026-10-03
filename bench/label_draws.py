"""How TractCloud's labels settle with the number of context draws, and which random part moves them.
PAT16, the M2's field (pipeline.correct), the Metal tractogram and the float64 one (CPU), labeled on the
M2's GPU with upstream's network.

TractCloud's context per draw (upstream RealDataDataset): each streamline's k = 20 nearest neighbors
within a random 10 % subsample (local, its own) and k_global = 80 random streamlines (global, ONE set
shared by every streamline of the draw). So:

  1. which part moves the aggregate: Other fraction over a 5 x 5 grid of local seeds x global seeds
     (the variance along each axis);
  2. the number of draws K = 1, 2, 3, 5, 10: two disjoint ensembles on the same tractogram (draws
     0..K-1 and 10..10+K-1), combined three ways - mean cluster probabilities then argmax (upstream's
     rule on the mean), mean tract mass (cluster probabilities summed per tract) then argmax, and the
     majority vote of per-draw tract labels (the pipeline's Labeler); per-streamline agreement between
     the two ensembles and their Other fractions;
  3. across trackers: Metal against float64, seed-matched, at each K (ensemble 0..K-1 on both).

    python bench/label_draws.py

Writes results/label_draws.json.
"""
import json, time
from pathlib import Path
import numpy as np, torch
from tractline import pipeline as P, ukf as U
from tractline.prep import prepare
from tractline.resample import resample
from tractline.labelers.tractcloud import Labeler, inf, MIN_LENGTH_MM
from _ds001226 import load

HERE = Path(__file__).resolve().parent
OTHER, KS = 42, (1, 2, 3, 5, 10)


class Draws:
    """One tractogram's features, and the network's log-probabilities for a (local seed, global seed)."""
    def __init__(self, lab, fibers, k_global=80):
        self.lab, self.k_global = lab, k_global
        lens = np.array([len(f) for f in fibers])
        Pts = np.concatenate(fibers).astype(np.float32).astype(np.float64)
        o = np.r_[0, np.cumsum(lens)]
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pts, axis=0), axis=1))]
        self.keep = (seg[o[1:] - 1] - seg[o[:-1]]) >= MIN_LENGTH_MM
        self.feat = inf.center_tractography(resample(torch.from_numpy(Pts), torch.from_numpy(o)).numpy()[self.keep], lab.center)
        self._ds = {}

    def dataset(self, seed):
        if seed not in self._ds:
            np.random.seed(seed)
            self._ds[seed] = inf.RealDataDataset(self.feat, k=20, k_global=self.k_global, k_ds_rate=0.1)
            if len(self._ds) > 6:                                          # keep memory bounded (~120 MB a dataset)
                self._ds.pop(next(iter(self._ds)))
        return self._ds[seed]

    def logp(self, local_seed, global_seed=None):
        """(n, 1600) log-probabilities, float32 on the CPU; global from global_seed (default: the same draw)."""
        dl = self.dataset(local_seed); dg = self.dataset(local_seed if global_seed is None else global_seed)
        dev = self.lab.device
        Pf = torch.from_numpy(dl.feat).float().to(dev).transpose(2, 1).contiguous()
        L = torch.from_numpy(dl.local_feat).float().to(dev).transpose(2, 1).contiguous()
        G = torch.from_numpy(dg.global_feat).float().to(dev).transpose(2, 1).contiguous()
        out = []
        with torch.no_grad():
            for a in range(0, len(dl), 1024):
                b = min(len(dl), a + 1024)
                out.append(self.lab.model(Pf[a:b], torch.cat((L[a:b], G.expand(b - a, -1, -1, -1)), 3)).view(-1, 1600).float().cpu())
        return torch.cat(out)


def labels_of(lab, lp_sum, mass_sum, votes):
    """The three combinations of one ensemble: (cluster-mean, mass-mean, vote) tract labels."""
    cl = lab.lut[lp_sum.argmax(1).numpy()]
    ms = mass_sum.argmax(1).numpy()
    vt = np.array([np.bincount(c, minlength=43).argmax() for c in np.stack(votes).T])
    return {"cluster_mean": cl, "mass_mean": ms, "vote": vt}


def ensembles(lab, d, start, onehot):
    """Labels of the ensembles draws start..start+K-1 for every K in KS."""
    p_sum = m_sum = None; votes, out = [], {}
    for i in range(max(KS)):
        lp = d.logp(start + i)
        p = lp.exp()
        p_sum = p if p_sum is None else p_sum + p
        m = p @ onehot
        m_sum = m if m_sum is None else m_sum + m
        votes.append(lab.lut[lp.argmax(1).numpy()])
        if i + 1 in KS:
            out[i + 1] = labels_of(lab, p_sum, m_sum, votes)
    return out


def other(t):
    return round(float((t == OTHER).mean()), 4)


if __name__ == "__main__":
    t0 = time.time()
    s = load("PAT16"); timer = P.Timer()
    corr = P.correct(s, timer, device="mps")
    ti = prepare(corr.dwi, s.affine, s.bval, s.bvec, device="mps")
    D = U.from_arrays(ti.dwi, ti.header, ti.mask)
    trk = {"metal": U.track(D, backend="metal"), "f64": U.track(D, dtype=torch.float64, device="cpu", batch=P.CPU_BATCH, workers=8)}
    lab = Labeler("mps")
    onehot = torch.zeros(1600, 43); onehot[torch.arange(1600), torch.as_tensor(lab.lut.astype(np.int64))] = 1
    res = {"subject": "PAT16", "tractograms": {k: {"fibers": len(f), "labeled": None} for k, (f, _) in trk.items()}}
    draws = {k: Draws(lab, f) for k, (f, _) in trk.items()}
    for k in draws:
        res["tractograms"][k]["labeled"] = int(draws[k].keep.sum())

    # 1. local x global, Metal tractogram
    d = draws["metal"]
    s0 = d.logp(0).exp().sum(1)                                            # the network's output is a log-softmax
    res["probability_sums_min_max"] = [round(float(s0.min()), 4), round(float(s0.max()), 4)]
    grid = np.array([[other(lab.lut[d.logp(a, g).argmax(1).numpy()]) for g in range(5)] for a in range(5)])
    res["grid_other_local_x_global"] = grid.tolist()
    res["grid_sd"] = {"along_global (local fixed)": round(float(grid.std(1, ddof=1).mean()), 4),
                      "along_local (global fixed)": round(float(grid.std(0, ddof=1).mean()), 4)}
    print(json.dumps({k: res[k] for k in ("grid_other_local_x_global", "grid_sd")}), flush=True)

    # 2. draws: two disjoint ensembles per tractogram
    E = {k: {"A": ensembles(lab, draws[k], 0, onehot), "B": ensembles(lab, draws[k], 10, onehot)} for k in draws}
    res["self_consistency"] = {k: {K: {rule: {"agreement_A_B": round(float((E[k]["A"][K][rule] == E[k]["B"][K][rule]).mean()), 4),
                                             "other_A_B": [other(E[k]["A"][K][rule]), other(E[k]["B"][K][rule])]}
                                      for rule in ("cluster_mean", "mass_mean", "vote")} for K in KS} for k in E}

    # 3. Metal against float64, seed-matched, ensemble A
    def rows(f, st, keep):
        r = np.full(len(f), -1); r[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
        return {int(s_): r[i] for i, s_ in enumerate(np.asarray(st["seed_index"])) if r[i] >= 0}
    ra, rb = rows(*trk["metal"], draws["metal"].keep), rows(*trk["f64"], draws["f64"].keep)
    both = sorted(set(ra) & set(rb)); ia = np.array([ra[k] for k in both]); ib = np.array([rb[k] for k in both])
    res["metal_vs_f64"] = {"matched": len(both), "by_K": {K: {rule: round(float((E["metal"]["A"][K][rule][ia] == E["f64"]["A"][K][rule][ib]).mean()), 4)
                                                             for rule in ("cluster_mean", "mass_mean", "vote")} for K in KS}}
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps({k: res[k] for k in ("self_consistency", "metal_vs_f64")}, indent=1), flush=True)
    (HERE / "results/label_draws.json").write_text(json.dumps(res, indent=1))
