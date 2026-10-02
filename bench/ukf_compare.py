"""The torch UKF (ukf.track, modal_ukf_track.py) against the Slicer UKF binary, fiber by fiber.

    python bench/ukf_compare.py

Both are in seed order with the same <10-point rule, so fiber k of one is fiber k of the other
when the counts agree. Per fiber: same point count, and the largest distance between corresponding
points (taking either orientation, since the seed tensor's eigenvector sign may differ between
Eigen's SVD and LAPACK's, which reverses the joined order). Writes results/ukf_compare.json.
"""
import json
from pathlib import Path
import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy

H = Path.home() / "tmp/data/tractography/ukf/hardi"
r = vtk.vtkPolyDataReader(); r.SetFileName(str(H / "ukf_spv1.vtk")); r.Update(); pd = r.GetOutput()
ro = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
rp = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
t = np.load(H / "torch_fibers.npz"); tp, to = t["points"], t["offsets"]
nr, nt = len(ro) - 1, len(to) - 1
out = {"reference_fibers": nr, "torch_fibers": nt, "reference_points": int(ro[-1]), "torch_points": int(to[-1])}
n = min(nr, nt)
same_len = 0; dmax = np.full(n, np.inf); reversed_ = 0
for k in range(n):
    a = rp[ro[k]:ro[k + 1]]; b = tp[to[k]:to[k + 1]]
    if len(a) != len(b):
        continue
    same_len += 1
    d1 = np.linalg.norm(a - b, axis=1).max(); d2 = np.linalg.norm(a - b[::-1], axis=1).max()
    dmax[k] = min(d1, d2); reversed_ += d2 < d1
ok = np.isfinite(dmax)
q = lambda x: [float(f"{v:.3g}") for v in np.quantile(x, [0.5, 0.9, 0.99, 0.999, 1.0])]
out.update({"same_point_count": same_len, "same_point_count_fraction": round(same_len / n, 5),
            "max_point_distance_mm_quantiles_50_90_99_99.9_100": q(dmax[ok]),
            "within_1e-6_mm": int((dmax[ok] < 1e-6).sum()), "within_1e-3_mm": int((dmax[ok] < 1e-3).sum()),
            "within_0.1_mm": int((dmax[ok] < 0.1).sum()), "matched_reversed": int(reversed_)})
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parent / "results" / "ukf_compare.json").write_text(json.dumps(out, indent=1))
