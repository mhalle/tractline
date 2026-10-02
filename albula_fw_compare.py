"""albula-diffusion's UKF in its DEFAULT mode (two tensors + free water, UKF's default thresholds)
against the C++ program run the same way (`UKFTractography --numTensor 2 --freeWater`, defaults
otherwise: seeding 0.18, stop FA 0.15, stop signal 0.1, a point every 0.9 mm).

    DATA/.venv/bin/python bench/tractography/albula_fw_compare.py [--every 49]

Our torch tracker has no free-water model, so the reference is the C++ output itself. Seeds are the
C++ program's (its voxels + the srand(0) offset, accepted by its rules at seedingThreshold 0.18);
each C++ fiber passes through its seed point exactly, so a port fiber is matched to the C++ fiber
holding its seed point (within 1e-3 mm). C++ fibers under 10 points are absent; so are the port's
here. Writes results/albula_ukf_fw.json.
"""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np, vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy
import _ukf_torch as U

ap = argparse.ArgumentParser(); ap.add_argument("--every", type=int, default=49); args = ap.parse_args()
HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
H, OUT = DATA / "ukf/hardi", DATA / "albula_fw"
OUT.mkdir(exist_ok=True)

D = U.load(str(H / "dwi.nhdr"), str(H / "mask.nrrd"))
off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
pts, *_ = U.seeds(D, off, seeding_threshold=0.18)
sel = np.arange(0, len(pts), args.every)
N = D["N"]; nk, nj, ni = D["dim"]
meta = {"dims": [ni, nj, nk], "voxel": D["voxel"].numpy()[::-1].tolist(), "ijkToRAS": D["i2r"].reshape(-1).tolist(), "G": N,
        "opts": {"freeWater": True}}                                   # every other option at its (= UKF's) default
(OUT / "meta.json").write_text(json.dumps(meta))
D["A"].numpy().astype("<f4").tofile(OUT / "signal.f32")
D["g"][:N].numpy().astype("<f8").tofile(OUT / "g.f64")
D["b"][:N].numpy().astype("<f8").tofile(OUT / "b.f64")
(D["mask"].numpy().view(np.uint8) > 0).astype(np.uint8).tofile(OUT / "mask.u8")
seed_ijk = pts[sel].numpy()[:, ::-1].copy()
seed_ijk.astype("<f8").tofile(OUT / "seeds.f64")

t0 = time.time()
r = subprocess.run(["deno", "run", "-A", "--config", str(HERE / "albula/deno.json"), str(HERE / "albula/ukf_port_run.ts"), str(OUT)],
                   capture_output=True, text=True)
if r.returncode:
    raise SystemExit(r.stderr[-3000:])
port_info = json.loads(r.stdout.strip().splitlines()[-1]); port_info["wall_s"] = round(time.time() - t0, 1)
po = np.fromfile(OUT / "port_offsets.u32", "<u4").astype(np.int64)
pp = np.fromfile(OUT / "port_points.f32", "<f4").reshape(-1, 3).astype(np.float64)
ps = np.fromfile(OUT / "port_seed.i32", "<i4")
port = {int(s): pp[po[i]:po[i + 1]] for i, s in enumerate(ps) if po[i + 1] - po[i] >= 10}

rd = vtk.vtkPolyDataReader(); rd.SetFileName(str(H / "ukf_fw_defaults.vtk")); rd.Update(); pd = rd.GetOutput()
ro = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
rp = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
seed_ras = seed_ijk @ D["i2r"][:3, :3].T + D["i2r"][:3, 3]
dd, ii = cKDTree(rp).query(seed_ras)
ref = {k: rp[ro[f]:ro[f + 1]] for k, (d, f) in enumerate(zip(dd, np.searchsorted(ro, ii, side="right") - 1)) if d < 1e-3}

both = sorted(set(ref) & set(port))
same = [k for k in both if len(ref[k]) == len(port[k])]
d = np.array([min(np.linalg.norm(ref[k] - port[k], axis=1).max(), np.linalg.norm(ref[k] - port[k][::-1], axis=1).max()) for k in same])
q = lambda x: [float(f"{v:.3g}") for v in np.quantile(x, [0.5, 0.9, 0.99, 1.0])] if len(x) else []
res = {"seeds_tested": int(len(sel)), "port": port_info, "reference_fibers_total": int(len(ro) - 1),
       "matched_reference": len(ref), "port_fibers_ge10": len(port), "both": len(both),
       "ref_only": len(set(ref) - set(port)), "port_only": len(set(port) - set(ref)), "same_len": len(same),
       "same_len_max_point_distance_mm_quantiles_50_90_99_100": q(d),
       "same_len_within_1e-3_mm": int((d < 1e-3).sum()), "same_len_within_0.1_mm": int((d < 0.1).sum())}
print(json.dumps(res, indent=1))
(HERE / "results" / "albula_ukf_fw.json").write_text(json.dumps(res, indent=1))
