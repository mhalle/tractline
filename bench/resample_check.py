"""resample.resample against upstream extract_ras_features on the HCP tractogram, and its speed.

    python bench/resample_check.py
"""
import json, time
from pathlib import Path
import numpy as np, torch, vtk
from vtk.util.numpy_support import vtk_to_numpy
from tractline.data import HCP, HCP_VTP
from tractline.resample import resample

r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(HCP_VTP)); r.Update(); pd = r.GetOutput()
pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
off = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
P = pts[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
ref = np.load(HCP / "feat.npy")                                    # upstream, float64 (features.py)
out = {}
Pt, ot = torch.from_numpy(P), torch.from_numpy(off)
resample(Pt[:off[1000]], ot[:1001])                                # warm
t0 = time.perf_counter(); f64 = resample(Pt, ot).numpy(); out["cpu_float64_s"] = round(time.perf_counter() - t0, 2)
out["float64_identical_to_upstream"] = bool(np.array_equal(f64, ref))
out["float64_max_abs_diff"] = float(np.abs(f64 - ref).max())
if torch.backends.mps.is_available():
    Pm, om = Pt.float().to("mps"), ot.to("mps")
    resample(Pm[:off[1000]], om[:1001]); torch.mps.synchronize()
    t0 = time.perf_counter(); f32 = resample(Pm, om); torch.mps.synchronize(); out["mps_float32_s"] = round(time.perf_counter() - t0, 2)
    out["mps_float32_max_abs_diff_mm"] = float(np.abs(f32.cpu().double().numpy() - ref).max())
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parent / "results" / "resample.json").write_text(json.dumps(out, indent=1))
