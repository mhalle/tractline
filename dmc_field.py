"""The rank field at connectome scale: DeepMultiConnectome's pretrained two-head model (SlicerDMRI/
DeepMultiConnectome 6c606ec, train_test/trained_model) on a whole tractogram, both heads kept as
fields instead of argmax labels. A format-and-speed test: the labels are not checked against ground
truth (no parcellation here, and the input is not the model's training domain).

    DATA/.venv/bin/python bench/tractography/dmc_field.py [--chunk 32768] [--depth 6]

Input: TractCloud's HCP 101006 UKF tractogram (440,621 streamlines), in HCP's ACPC-aligned space.
The model was trained on iFOD2 tractograms in MNI space, so its outputs here are off its domain;
sizes and times do not depend on that, the class distribution somewhat does.
Model: PointNet without context (k = k_global = 0), 15 points per streamline by arc length
(_resample.py), log-softmax heads of 3,655 classes (Desikan-Killiany, 85 nodes counting "unknown",
85*86/2 pairs) and 13,695 (Destrieux, 165 nodes). The paper states 3,571 and 13,631.
Per chunk of streamlines: the model on MPS (float32, two forward calls as test_realdata.py makes
them), then rankfield.encode of each head on the CPU (keep="clip", clip 8, as encode.py), then blosc.
Checked: each field's top rank equals the head's argmax for every streamline.

Writes results/dmc_field.json.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, vtk
from numcodecs import Blosc
from vtk.util.numpy_support import vtk_to_numpy
import rankfield as rf
from _resample import resample
from _data import DATA, HCP_VTP

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=32768)
ap.add_argument("--depth", type=int, default=6)
args = ap.parse_args()
HERE = Path(__file__).resolve().parent
DMC = DATA / "DeepMultiConnectome"
sys.path.insert(0, str(DMC))
from models.pointnet import PointNetCls

dev = torch.device("mps")
sd = torch.load(DMC / "train_test/trained_model/best_f1_model.pth", map_location="cpu", weights_only=True)
K0, K1 = sd["fc3.weight"].shape[0], sd["fc3_1.weight"].shape[0]
model = PointNetCls(k=0, k_global=0, num_classes_0=K0, num_classes_1=K1)
model.load_state_dict(sd); model.to(dev).eval()

r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(HCP_VTP)); r.Update(); pd = r.GetOutput()
pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
off = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
P = pts[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
t0 = time.perf_counter()
feat = resample(torch.from_numpy(P), torch.from_numpy(off)).float()      # (N, 15, 3)
t_feat = time.perf_counter() - t0
N = feat.shape[0]

T = {"features": t_feat, "inference": 0.0, "encode_0": 0.0, "encode_1": 0.0}
codes = {0: [], 1: []}
mismatch = {0: 0, 1: 0}
z = lambda a: len(Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE).encode(np.ascontiguousarray(np.asarray(a))))
with torch.no_grad():
    for s in range(0, N, args.chunk):
        x = feat[s:s + args.chunk].to(dev).transpose(2, 1).contiguous()
        torch.mps.synchronize(); t0 = time.perf_counter()
        lp = [model(x, None, task_id=h)[0] for h in (0, 1)]
        torch.mps.synchronize(); T["inference"] += time.perf_counter() - t0
        for h in (0, 1):
            L = lp[h].cpu()
            t0 = time.perf_counter()
            code = rf.encode(L.T[:, :, None, None], keep="clip", depth=args.depth, clip=8.0, tail_temperatures=(1.0,))
            T[f"encode_{h}"] += time.perf_counter() - t0
            top = np.asarray(code.ranks).reshape(args.depth, -1)[0].astype(np.int64) - 1
            mismatch[h] += int((top != L.argmax(1).numpy()).sum())
            codes[h].append(code)
        print(f"{min(N, s + args.chunk)} / {N}", {k: round(v, 1) for k, v in T.items()}, flush=True)

res = {"model": "DeepMultiConnectome 6c606ec, pretrained, PointNet k=0", "input": "HCP 101006 UKF (TractCloud TestData), ACPC space",
       "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
       "rankfield": rf.__version__, "rankfield_source": rf.__file__, "streamlines": N, "depth": args.depth, "clip": 8.0,
       "seconds": {k: round(v, 2) for k, v in T.items()}, "heads": {}}
for h, K in ((0, K0), (1, K1)):
    ranks = np.concatenate([np.asarray(c.ranks).reshape(args.depth, -1) for c in codes[h]], axis=1)
    support = np.concatenate([np.asarray(c.support).reshape(args.depth - 1, -1) for c in codes[h]], axis=1)
    tail = np.concatenate([np.asarray(c.tail).reshape(-1) for c in codes[h]])
    raw = ranks.nbytes + support.nbytes + tail.nbytes
    comp = z(ranks) + z(support) + z(tail)
    kept = (ranks != 0).sum(0)
    tl = tail.astype(np.float64) / rf.TAIL_MAX
    res["heads"][f"head_{h}"] = {
        "classes": int(K), "atlas": "aparc+aseg (Desikan-Killiany)" if h == 0 else "aparc.a2009s+aseg (Destrieux)",
        "top_rank_differs_from_argmax": mismatch[h],
        "dense_fp16_mb": round(N * K * 2 / 1e6, 1), "dense_fp16_gb_at_2.93M_streamlines": round(2.93e6 * K * 2 / 1e9, 1),
        "field_raw_mb": round(raw / 1e6, 2), "field_blosc_mb": round(comp / 1e6, 2),
        "bytes_per_streamline_raw": round(raw / N, 2), "bytes_per_streamline_blosc": round(comp / N, 2),
        "kept_planes_hist": {int(k): int(v) for k, v in zip(*np.unique(kept, return_counts=True))},
        "tail_mass_quantiles_50_90_99": [float(f"{v:.3g}") for v in np.quantile(tl, [0.5, 0.9, 0.99])],
        "encode_s": round(T[f"encode_{h}"], 2), "encode_streamlines_per_s": round(N / T[f"encode_{h}"])}
res["inference_streamlines_per_s"] = round(N / T["inference"])
res["at_2.93M_streamlines_estimate_s"] = {k: round(v * 2.93e6 / N, 1) for k, v in T.items()}
print(json.dumps(res, indent=1))
(HERE / "results" / "dmc_field.json").write_text(json.dumps(res, indent=1))
