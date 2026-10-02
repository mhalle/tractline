"""_median.py against DIPY and SciPy, voxel for voxel, and timed.

    DATA/.venv/bin/python bench/tractography/median_check.py

  - one pass of the filter against scipy.ndimage.median_filter on random volumes with many ties
    and odd shapes;
  - the mask against dipy.segment.mask.median_otsu on the mean b0 of the 12 cohort patients'
    AP DWIs as acquired, and of PAT16 as corrected by topup and by ours (float32 inputs).
Writes results/median_check.json.
"""
import json, time
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import median_filter
from dipy.segment.mask import median_otsu as dipy_mo
import _median as M
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
M.median_otsu(vols["PAT16"])                                              # numba compile
td, tf = [], []
for name, b0 in vols.items():
    t0 = time.time(); _, a = dipy_mo(b0, median_radius=4, numpass=4); td.append(time.time() - t0)
    t0 = time.time(); _, b = M.median_otsu(b0); tf.append(time.time() - t0)
    res["mask_vs_dipy"].append({"volume": name, "identical": bool(np.array_equal(a, b)), "voxels_differ": int((a != b).sum()), "mask_voxels": int(a.sum())})
res["seconds_median"] = {"dipy": round(float(np.median(td)), 2), "ours": round(float(np.median(tf)), 3)}
res["all_identical"] = all(r["identical"] for k in ("filter_vs_scipy", "mask_vs_dipy") for r in res[k])
print(json.dumps(res, indent=1))
(HERE / "results/median_check.json").write_text(json.dumps(res, indent=1))
