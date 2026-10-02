"""_median.py (torch, CPU and GPU) against DIPY and SciPy, voxel for voxel, and timed.

    DATA/.venv/bin/python bench/tractography/median_check.py

  - one pass of the filter against scipy.ndimage.median_filter on random volumes with many ties
    and odd shapes;
  - the mask against dipy.segment.mask.median_otsu on the mean b0 of the 12 cohort patients'
    AP DWIs as acquired, and of PAT16 as corrected by topup and by ours (float32 inputs), with the
    median passes on the CPU and on the GPU (MPS);
  - otsu against dipy.segment.threshold.otsu on the same filtered volumes (identical thresholds).
Writes results/median_check.json.
"""
import json, time
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import median_filter
from dipy.segment.mask import median_otsu as dipy_mo
import torch
import _median as M
from dipy.segment.threshold import otsu as dipy_otsu
from _ds001226 import load, ROOT

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
res = {"filter_vs_scipy": [], "mask_vs_dipy": []}
for shape, levels in (((17, 23, 11), 5), ((40, 31, 29), 1000), ((96, 96, 60), 30)):
    v = rng.integers(0, levels, shape).astype(np.float32) * 0.37
    same = np.array_equal(M.multi_median(v, 4, 1), median_filter(v, size=9, mode="reflect"))
    res["filter_vs_scipy"].append({"shape": shape, "distinct_values": levels, "identical": bool(same)})
vols = {}
for s in "PAT05 PAT07 PAT08 PAT13 PAT14 PAT16 PAT19 PAT20 PAT23 PAT25 PAT26 PAT29".split():
    X = load(s)
    vols[s] = X.dwi[..., X.b0_index].astype(np.float32).mean(-1)
X = load("PAT16")
for name, p in (("PAT16 topup", ROOT / "derived/PAT16/topup/dwi_AP_topup.nii.gz"), ("PAT16 ours", ROOT / "derived/PAT16/susc/dwi_AP_ours_fast.nii.gz")):
    vols[name] = np.asarray(nib.load(p).dataobj, dtype=np.float32)[..., X.b0_index].mean(-1)
devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
for d in devices:
    M.median_otsu(vols["PAT16"], device=d)                                # warm
td, tf = [], {d: [] for d in devices}
for name, b0 in vols.items():
    t0 = time.time(); _, a = dipy_mo(b0, median_radius=4, numpass=4); td.append(time.time() - t0)
    row = {"volume": name, "mask_voxels": int(a.sum())}
    for d in devices:
        t0 = time.time(); _, b = M.median_otsu(b0, device=d); tf[d].append(time.time() - t0)
        row[f"identical_{d}"] = bool(np.array_equal(a, b)); row[f"voxels_differ_{d}"] = int((a != b).sum())
    m = M.multi_median(b0)
    row["otsu_identical"] = bool(dipy_otsu(m) == M.otsu(m))
    row["identical"] = all(row[f"identical_{d}"] for d in devices) and row["otsu_identical"]
    res["mask_vs_dipy"].append(row)
res["seconds_median"] = {"dipy": round(float(np.median(td)), 2), **{f"ours_{d}": round(float(np.median(v)), 2) for d, v in tf.items()}}
res["all_identical"] = all(r["identical"] for k in ("filter_vs_scipy", "mask_vs_dipy") for r in res[k])
print(json.dumps(res, indent=1))
(HERE / "results/median_check.json").write_text(json.dumps(res, indent=1))
