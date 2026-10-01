"""The faithful pipeline on a Mac, end to end, timed by stage: DWI -> UKF (Metal kernel) -> 40 mm cut ->
TractCloud (MPS) -> the rankfield field. Stanford HARDI, ORG settings, on this machine.

    DATA/.venv/bin/python bench/tractography/mac_pipeline.py [--labels]

Stages:
  load      the DWI and mask, normalized as the binary does (_ukf_torch.load)
  seeds     the binary's seeds, float64 on the CPU
  ukf       _ukf_torch.track(backend="metal"): float32 on the GPU (ukf_labels.json: below the noise floor)
  features  the 40 mm cut (wm_preprocess_all.py -l 40) and 15-point resampling, float64 on the CPU
            (_resample.py, upstream's rule)
  context   upstream's RealDataDataset (np.random.seed(0)): kNN per file chunk, numpy on the CPU
  network   upstream TractCloud on MPS, float32, then again under float16 autocast (labels compared)
  field     rankfield.encode of the log-probabilities, depth 6, keep="clip", clip 8 (encode.py's), CPU

With --labels: 5 context draws on the Metal tractogram and 5 on the float64 CUDA one (torch_fibers.npz),
majority votes compared seed by seed (matched through seed points) - ukf_labels.json's measure, where
float32 on CUDA gave 0.905 against TractCloud's own floor of 0.926.

Writes results/mac_pipeline.json.
"""
import argparse, json, subprocess, sys, time, types
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
import _ukf_torch as U
from _resample import resample
from _data import DATA, MODEL, MASS_CENTER

ap = argparse.ArgumentParser(); ap.add_argument("--labels", action="store_true"); args = ap.parse_args()
HERE = Path(__file__).resolve().parent
H = DATA / "ukf/hardi"
sys.path.insert(0, str(DATA / "TractCloud/src"))
sys.modules.setdefault("vtk", types.ModuleType("vtk"))
from tractcloud import inference as inf
from tractcloud.tract_mapping import _CLUSTER_TO_TRACT_LUT as LUT
import rankfield as rf

sync = torch.mps.synchronize
T = {}
def stage(name, t0):
    sync(); T[name] = round(time.time() - t0, 2); print(name, T[name], "s", flush=True)

t0 = time.time(); D = U.load(str(H / "dwi.nhdr"), str(H / "mask.nrrd")); stage("load", t0)
off = np.array([-4158, -2201, -2855], float); off = off / np.linalg.norm(off) * 0.5
t0 = time.time(); pts, *_ = U.seeds(D, off); stage("seeds", t0)
t0 = time.time(); fibers, st = U.track(D, off, backend="metal"); stage("ukf", t0)

def features(fibers):
    lens = np.array([len(f) for f in fibers])
    P = np.concatenate(fibers).astype(np.float32).astype(np.float64)        # as a VTK file would hold them
    off_ = np.r_[0, np.cumsum(lens)]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    keep = (seg[off_[1:] - 1] - seg[off_[:-1]]) >= 40
    return resample(torch.from_numpy(P), torch.from_numpy(off_)).numpy()[keep], keep
t0 = time.time(); feat, keep = features(fibers); stage("features", t0)

dev = torch.device("mps")
model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), dev,
                          k_override=20, k_global_override=80)
center = np.load(MASS_CENTER)
lut = LUT.astype(np.int64)

def context(feat, seed):
    np.random.seed(seed)
    return inf.RealDataDataset(inf.center_tractography(feat, center), k=20, k_global=80, k_ds_rate=0.1)

def network(ds, half=False):
    P = torch.from_numpy(ds.feat).float().to(dev).transpose(2, 1).contiguous()
    L = torch.from_numpy(ds.local_feat).float().to(dev).transpose(2, 1).contiguous()
    G = torch.from_numpy(ds.global_feat).float().to(dev).transpose(2, 1).contiguous()
    out = torch.empty((len(ds), 1600), dtype=torch.float16, device=dev)
    with torch.no_grad(), torch.autocast("mps", dtype=torch.float16, enabled=half):
        for s in range(0, len(ds), 1024):
            e = min(len(ds), s + 1024)
            out[s:e] = model(P[s:e], torch.cat((L[s:e], G.expand(e - s, -1, -1, -1)), 3)).view(-1, 1600).half()
    return out

t0 = time.time(); ds = context(feat, 0); stage("context", t0)
t0 = time.time(); lp = network(ds); stage("network_fp32", t0)
t0 = time.time(); lp16 = network(ds, half=True); stage("network_fp16", t0)
c32, c16 = lp.float().argmax(1).cpu().numpy(), lp16.float().argmax(1).cpu().numpy()
t0 = time.time()
code = rf.encode(lp.cpu().T[:, :, None, None], keep="clip", depth=6, clip=8.0, tail_temperatures=(1.0,))
T["field"] = round(time.time() - t0, 2); print("field", T["field"], "s")
field_bytes = sum(np.asarray(a).nbytes for a in (code.ranks, code.support, code.tail))

chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
res = {"machine": chip, "torch": torch.__version__, "rankfield": rf.__version__,
       "seeds": st["seeds"], "fibers": st["fibers"], "fiber_steps": st["fiber_steps"], "streamlines_ge_40mm": int(keep.sum()),
       "seconds": T, "total_s": round(sum(T.values()), 1),
       "ukf_steps_per_s": round(st["fiber_steps"] / T["ukf"]),
       "fp16_vs_fp32_on_mps": {"cluster_labels_differ": int((c32 != c16).sum()),
                               "tract_labels_differ": int((lut[c32] != lut[c16]).sum())},
       "field": {"depth": 6, "bytes": int(field_bytes), "bytes_per_streamline": round(field_bytes / len(ds), 2)}}

if args.labels:
    def vote(Tm):
        return np.array([np.bincount(c, minlength=43).argmax() for c in Tm.T])
    def labels(feat):
        return np.stack([lut[network(context(feat, s)).float().argmax(1).cpu().numpy()] for s in range(5)])
    t0 = time.time()
    seed_ras = pts.numpy()[:, ::-1] @ D["i2r"][:3, :3].T + D["i2r"][:3, 3]
    Z = np.load(H / "torch_fibers.npz"); tp, to = Z["points"], Z["offsets"]
    dd, ii = cKDTree(tp).query(seed_ras)
    fi = np.searchsorted(to, ii, side="right") - 1
    ref_seed = np.array([k for k, d in enumerate(dd) if d < 1e-6]); ref_fib = fi[dd < 1e-6]
    ref_fibers = [tp[to[j]:to[j + 1]] for j in ref_fib]
    feat_ref, keep_ref = features(ref_fibers)
    vm, vr = vote(labels(feat)), vote(labels(feat_ref))
    seeds_m = np.array(st["seed_index"])[keep]; seeds_r = ref_seed[keep_ref]
    pos_r = {int(s): i for i, s in enumerate(seeds_r)}
    pairs = [(i, pos_r[int(s)]) for i, s in enumerate(seeds_m) if int(s) in pos_r]
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    res["labels_metal_vs_f64_cuda"] = {"both": len(pairs), "vote_agreement": round(float((vm[a] == vr[b]).mean()), 4),
                                       "other_fraction_metal": round(float((vm == 42).mean()), 4),
                                       "other_fraction_f64": round(float((vr == 42).mean()), 4),
                                       "seconds": round(time.time() - t0, 1),
                                       "for_scale": "ukf_labels.json: float32 on CUDA 0.905, TractCloud's own floor 0.926, bootstrap 0.78"}
print(json.dumps(res, indent=1))
(HERE / "results" / "mac_pipeline.json").write_text(json.dumps(res, indent=1))
