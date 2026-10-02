"""TractCloud's labels on four whole-brain HARDI tractographies, all labeled the same way on this Mac
(MPS, which gives the CPU's labels exactly): float64 (the CUDA run, torch_fibers.npz), the Metal kernel
(ukf_metal_check.py --full), float32 torch on CUDA and one bootstrap replicate (modal_ukf_labels.py,
from the Volume). Same seeds, 40 mm cut, 15-point resampling, upstream context.

    python bench/mac_labels.py

Each tractography gets 5-draw majority votes (draws 0-4); float64 gets a second vote (draws 5-9).
Each is compared seed by seed with BOTH float64 votes, so the spread between the two shows how much
of a difference is the votes' own. Writes results/mac_labels.json.
"""
import json, sys, time, types
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
from tractline import ukf as U
from tractline.resample import resample
from tractline.data import DATA, MODEL, MASS_CENTER
sys.path.insert(0, str(DATA / "TractCloud/src")); sys.modules.setdefault("vtk", types.ModuleType("vtk"))
from tractcloud import inference as inf
from tractcloud.tract_mapping import _CLUSTER_TO_TRACT_LUT as LUT

HERE = Path(__file__).resolve().parent
H, V = DATA / "ukf/hardi", DATA / "ukf32/variants"
lut = LUT.astype(np.int64)
dev = torch.device("mps")
model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), dev, k_override=20, k_global_override=80)
center = np.load(MASS_CENTER)

D = U.load(str(H / "dwi.nhdr"), str(H / "mask.nrrd"))
off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
pts, *_ = U.seeds(D, off)

def load(name):
    if name == "f64":                                                 # no seed index stored: match seed points
        z = np.load(H / "torch_fibers.npz"); P, o = z["points"], z["offsets"]
        seed_ras = pts.numpy()[:, ::-1] @ D["i2r"][:3, :3].T + D["i2r"][:3, 3]
        dd, ii = cKDTree(P).query(seed_ras)
        fi = np.searchsorted(o, ii, side="right") - 1
        ok = dd < 1e-6
        assert np.array_equal(fi[ok], np.arange(len(o) - 1)), "every fiber matched once, in seed order"
        return P, o, np.flatnonzero(ok)
    z = np.load({"metal": DATA / "ukf32/metal_full.npz"}.get(name, V / f"{name}.npz"))
    return z["points"], z["offsets"], z["seed_index"]

def labels(P, o, draws):
    P = P.astype(np.float32).astype(np.float64)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    keep = (seg[o[1:] - 1] - seg[o[:-1]]) >= 40
    feat = resample(torch.from_numpy(P), torch.from_numpy(o)).numpy()[keep]
    out = []
    for s in draws:
        np.random.seed(s)
        ds = inf.RealDataDataset(inf.center_tractography(feat, center), k=20, k_global=80, k_ds_rate=0.1)
        Pf = torch.from_numpy(ds.feat).float().to(dev).transpose(2, 1).contiguous()
        L = torch.from_numpy(ds.local_feat).float().to(dev).transpose(2, 1).contiguous()
        G = torch.from_numpy(ds.global_feat).float().to(dev).transpose(2, 1).contiguous()
        cl = []
        with torch.no_grad():
            for a in range(0, len(ds), 1024):
                b = min(len(ds), a + 1024)
                cl.append(model(Pf[a:b], torch.cat((L[a:b], G.expand(b - a, -1, -1, -1)), 3)).view(-1, 1600).argmax(1).cpu())
        out.append(lut[torch.cat(cl).numpy()])
    return np.stack(out), keep

vote = lambda T: np.array([np.bincount(c, minlength=43).argmax() for c in T.T])
t0 = time.time()
lab = {}
for name in ("f64", "metal", "f32", "boot0"):
    P, o, si = load(name)
    T, keep = labels(P, o, range(10) if name == "f64" else range(5))
    lab[name] = {"seed": si[keep], "T": T}
    print(name, T.shape, round(time.time() - t0), "s", flush=True)

ref = lab["f64"]; pos = {int(s): i for i, s in enumerate(ref["seed"])}
vA, vB = vote(ref["T"][:5]), vote(ref["T"][5:])
res = {"device": "mps", "floor_f64_vote_0_4_vs_5_9": round(float((vA == vB).mean()), 4)}
for name in ("metal", "f32", "boot0"):
    x = lab[name]; v = vote(x["T"])
    pr = [(i, pos[int(s)]) for i, s in enumerate(x["seed"]) if int(s) in pos]
    a, b = np.array([p[0] for p in pr]), np.array([p[1] for p in pr])
    res[name] = {"both": len(pr), "vs_f64_vote_0_4": round(float((v[a] == vA[b]).mean()), 4),
                 "vs_f64_vote_5_9": round(float((v[a] == vB[b]).mean()), 4),
                 "other_fraction": round(float((v == 42).mean()), 4)}
res["seconds"] = round(time.time() - t0)
print(json.dumps(res, indent=1))
(HERE / "results" / "mac_labels.json").write_text(json.dumps(res, indent=1))
