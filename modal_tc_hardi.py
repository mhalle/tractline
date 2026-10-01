"""TractCloud on two tractograms of the same Stanford HARDI scan: UKF with the ORG settings TractCloud
was trained on (two tensors, seeding 0.1, stop FA 0.08 / signal 0.06, a point every 1.8 mm) and UKF
as albula-diffusion runs it by default (two tensors + free water, UKF's defaults: 0.18 / 0.15 / 0.1,
0.9 mm). Both from the C++ program (ukf_bench.py, and the --freeWater run).

    DATA/.venv/bin/python bench/tractography/modal_tc_hardi.py --extract   # features, locally (VTK)
    modal run bench/tractography/modal_tc_hardi.py

Per tractogram, five seeded upstream contexts, upstream network: the share labeled Other, the
run-0 mass-margin distribution, the tract-change rate across the five runs, and the tract mix.
A caveat that applies to both: HARDI is b=2000 at 2 mm, TractCloud's training data b=3000 at 1.25 mm.
Writes results/tc_hardi.json.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
TRACTS = {"org_2t": "ukf_spv1.vtk", "fw_defaults": "ukf_fw_defaults.vtk"}

if "--extract" in sys.argv:                                            # local: VTK -> TractCloud features
    import numpy as np, vtk
    sys.path.insert(0, str(DATA / "TractCloud/src"))
    from tractcloud.inference import extract_ras_features
    for name, f in TRACTS.items():
        r = vtk.vtkPolyDataReader(); r.SetFileName(str(DATA / "ukf/hardi" / f)); r.Update()
        feat = extract_ras_features(r.GetOutput(), num_points=15)
        np.save(DATA / "ukf/hardi" / f"feat_{name}.npy", feat)
        print(name, feat.shape)
    raise SystemExit(0)

import modal                                                           # not needed (or installed) for --extract

image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.7", "numpy>=2")
         .env({"PYTHONPATH": "/root/pkg"})
         .add_local_dir(str(DATA / "TractCloud/src/tractcloud"), remote_path="/root/pkg/tractcloud"))
vol = modal.Volume.from_name("tractography-bench")
app = modal.App("tractography-tc-hardi", image=image)
V = "/vol"


@app.function(gpu="A10G", volumes={V: vol}, timeout=3600, memory=32768, cpu=8)
def run() -> dict:
    import sys, types
    import numpy as np, torch
    sys.modules.setdefault("vtk", types.ModuleType("vtk"))
    from tractcloud import inference as inf
    from tractcloud.tract_mapping import TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as LUT
    dev = torch.device("cuda")
    torch.backends.cudnn.enabled = False
    model, _ = inf.load_model(f"{V}/TrainedModel/best_tract_f1_model.pth", f"{V}/TrainedModel/cli_args.txt",
                              dev, k_override=20, k_global_override=80)
    center = np.load(f"{V}/TrainData_800clu800ol/HCP_mass_center.npy")
    lut = LUT.astype(np.int64)
    out = {}
    for name in TRACTS:
        feat = np.load(f"{V}/ukf/hardi/feat_{name}.npy")
        centered = inf.center_tractography(feat, center)
        labels, mass0 = [], None
        for seed in range(5):
            np.random.seed(seed)
            ds = inf.RealDataDataset(centered, k=20, k_global=80, k_ds_rate=0.1)
            P = torch.from_numpy(ds.feat).to(dev).transpose(2, 1).contiguous()
            L = torch.from_numpy(ds.local_feat).to(dev).transpose(2, 1).contiguous()
            G = torch.from_numpy(ds.global_feat).to(dev).transpose(2, 1).contiguous()
            lp = torch.empty((len(ds), 1600), device=dev)
            with torch.no_grad():
                for s in range(0, len(ds), 1024):
                    e = min(len(ds), s + 1024)
                    lp[s:e] = model(P[s:e], torch.cat((L[s:e], G.expand(e - s, -1, -1, -1)), 3)).view(-1, 1600)
            cl = lp.argmax(1)
            labels.append(lut[cl.cpu().numpy()])
            if seed == 0:                                              # run-0 mass margin of the labeled tract
                lt = torch.from_numpy(lut).to(dev)
                tract = lt[cl]
                inside = lt[None, :] == tract[:, None]
                p = lp.exp()
                pin = (p * inside).sum(1); pout = (p * ~inside).sum(1)
                mass0 = (pin.log() - pout.clamp_min(1e-30).log()).cpu().numpy()
        T = np.stack(labels)
        changed = (T[1:] != T[0]).any(0)
        counts = np.bincount(T[0], minlength=43)
        q = lambda x: [round(float(v), 3) for v in np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])]
        out[name] = {"streamlines": int(len(feat)), "other_fraction_run0": round(float((T[0] == 42).mean()), 4),
                     "tract_changed_fraction_5runs": round(float(changed.mean()), 4),
                     "mass_margin_run0_quantiles_10_25_50_75_90": q(mass0),
                     "margin_under_0.5_fraction": round(float((mass0 < 0.5).mean()), 4),
                     "named_tracts_with_50_plus": int((counts[:42] >= 50).sum()),
                     "tract_share_run0": {TRACT_NAMES[t]: round(float(c / len(feat)), 4) for t, c in enumerate(counts) if c}}
    a, b = out["org_2t"]["tract_share_run0"], out["fw_defaults"]["tract_share_run0"]
    keys = sorted(set(a) | set(b))
    va, vb = np.array([a.get(k, 0) for k in keys]), np.array([b.get(k, 0) for k in keys])
    out["tract_mix_correlation"] = round(float(np.corrcoef(va, vb)[0, 1]), 4)
    return out


@app.local_entrypoint()
def main():
    r = run.remote()
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "tract_share_run0"}) for k, v in r.items()}, indent=1))
    (HERE / "results" / "tc_hardi.json").write_text(json.dumps(r, indent=1))
