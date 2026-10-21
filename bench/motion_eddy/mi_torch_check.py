"""mi_torch.register checked as mi_dipy.py's self-test, on MPS: the mean b0 and b2000 of ON-Harmony sub-16974 moved by
known rigid transforms (scipy, cubic) and registered back (same contrast), and the b2000 mean against the b0 mean as it
is and moved (cross contrast: consistency, r1 - r0 = -d). With --dipy, mi_dipy's results on the same pairs beside them
(dipy runs in its own process).

    uv run --group dipy bench/motion_eddy/mi_torch_check.py [--dipy]
"""
import json, math, subprocess, sys, tempfile, os, time
from pathlib import Path
import numpy as np, nibabel as nib, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mi_dipy                                                       # moved(): the known shifts (numpy, scipy; no dipy import)
from mi_torch import register
from tractline.data import DATA
from scipy.ndimage import binary_dilation

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
D = str(DATA / "ds004712/sub-16974/ses-OXF1PRI001/dwi/sub-16974_ses-OXF1PRI001_dir-AP_dwi")
img = nib.load(D + ".nii.gz"); data = np.asarray(img.dataobj, np.float32); bval = np.loadtxt(D + ".bval")
vox = [float(v) for v in img.header.get_zooms()[:3]]
b0 = data[..., bval < 50].mean(-1); dw = data[..., np.abs(bval - 2000) < 100].mean(-1)
cen = [(n - 1) / 2 for n in b0.shape]
mask = binary_dilation(b0 > np.quantile(b0, 0.6), iterations=2)
T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
mk = torch.as_tensor(mask, device=dev)


def reg(mov, fixed):
    t0 = time.perf_counter()
    d = register(T(mov), T(fixed), mk, vox)
    if dev.type == "mps":
        torch.mps.synchronize()
    r = d.cpu().tolist()
    return [r[0], r[1], r[2], r[3] * 0.01, r[4] * 0.01, r[5] * 0.01], time.perf_counter() - t0


def reg_dipy(mov, fixed):
    with tempfile.TemporaryDirectory() as td:
        np.savez(os.path.join(td, "in.npz"), mov=np.asarray(mov, np.float64), fixed=np.asarray(fixed, np.float64), mask=mask,
                 vox=np.asarray(vox), cen=np.asarray(cen))
        subprocess.run([sys.executable, str(HERE / "mi_dipy.py"), "register", os.path.join(td, "in.npz"), os.path.join(td, "out.json")],
                       check=True, capture_output=True)
        return json.load(open(os.path.join(td, "out.json")))


fmt = lambda r: [round(v, 3) for v in r[:3]] + [round(math.degrees(v), 3) for v in r[3:]]
tests = [[0.0, 0.7, 0.0, 0.005, 0.0, 0.0], [0.5, -0.3, 0.4, 0.0, 0.004, -0.006]]
rows = []
for d in tests:
    for name, im in (("b0 -> b0", b0), ("b2000 mean -> b2000 mean", dw)):
        r, s = reg(mi_dipy.moved(im, d, vox, cen), im)
        err = [r[i] + d[i] for i in range(6)]
        rows.append({"case": f"same contrast {name}", "shift": d, "error (mm, deg)": fmt(err), "seconds": round(s, 2)})
        print(json.dumps(rows[-1]), flush=True)
r0, s0 = reg(dw, b0)
row = {"case": "cross contrast b2000 mean -> b0 mean, as is", "r0 (mm, deg)": fmt(r0), "seconds": round(s0, 2)}
if "--dipy" in sys.argv:
    row["dipy r0 (mm, deg)"] = fmt(reg_dipy(dw, b0))
rows.append(row); print(json.dumps(row), flush=True)
for d in tests:
    r1, s = reg(mi_dipy.moved(dw, d, vox, cen), b0)
    err = [r1[i] - r0[i] + d[i] for i in range(6)]
    rows.append({"case": "cross contrast, consistency", "shift": d, "error (mm, deg)": fmt(err), "seconds": round(s, 2)})
    print(json.dumps(rows[-1]), flush=True)
