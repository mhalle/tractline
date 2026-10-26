"""The gibbs slot, level 1 on a real scan (docs/sota-pipeline-design.md, section 4): tractline.gibbs against DIPY's
gibbs_removal on cIIIs04's reference scan as acquired (every volume, slices across the third axis), DIPY in a process of
its own with one worker per core. Reported: times; DIPY float32 and our CPU float64, CPU float32 and MPS float32 runs
against DIPY float64 (max |difference|, relative to the scan's 99th percentile, and the voxels differing by more than
0.1 %, 1 % and 5 % of that scale); and how much each changes the scan (the mean |change| relative to the same scale).
Other work on the machine shifts the times.

    uv run --group dipy bench/gibbs_check.py                    # results/gibbs_check.json
"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_runs as CR  # noqa: E402
from tractline import gibbs as G  # noqa: E402

DIPY_PY = """
import sys, time, os, warnings
import numpy as np
warnings.simplefilter("ignore")
from dipy.denoise.gibbs import gibbs_removal
v = np.load(sys.argv[1])
t = time.time()
out = gibbs_removal(v, slice_axis=2, inplace=False, num_processes=os.cpu_count())
np.save(sys.argv[2], out)
print(round(time.time() - t, 2))
"""


def main():
    A = CR.load_checked("cIIIs04", *CR.CARD["cIIIs04"].ref)
    vol = np.asarray(A.dwi, np.float64)
    scale = float(np.quantile(vol, 0.99))
    res = {"scan": "cIIIs04 A as acquired", "shape": list(vol.shape), "scale (99th percentile)": round(scale, 2),
           "cpu_count": os.cpu_count()}
    res["seconds"] = {}

    def against(o):
        dd = np.abs(o.astype(np.float64) - ref) / scale
        return {"max |difference| / scale": float(dd.max()),
                "voxels differing by more than 0.1 % / 1 % / 5 % of scale":
                    [int((dd > 1e-3).sum()), int((dd > 1e-2).sum()), int((dd > 5e-2).sum())],
                "share differing by more than 0.1 % of scale": round(float((dd > 1e-3).mean()), 6),
                "mean |change of the scan| / scale": round(float(np.abs(o.astype(np.float64) - vol).mean() / scale), 6)}
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "v.npy", Path(d) / "o.npy"
        for name, v in (("DIPY float64 (processes: cpu_count)", vol), ("DIPY float32 (processes: cpu_count)",
                                                                       vol.astype(np.float32))):
            np.save(src, v)
            r = subprocess.run([sys.executable, "-I", "-c", DIPY_PY, str(src), str(out)], check=True,
                               capture_output=True, text=True)
            res["seconds"][name] = float(r.stdout.strip().splitlines()[-1])
            if "float64" in name:
                ref = np.load(out)
            else:
                res["DIPY float32"] = against(np.load(out))
    for name, dev, v in (("ours, cpu float64", "cpu", vol), ("ours, cpu float32", "cpu", vol.astype(np.float32)),
                         ("ours, mps float32", "mps", vol.astype(np.float32))):
        G.gibbs_removal(v[..., :2], device=dev)                                  # warm-up
        t0 = time.time()
        o = G.gibbs_removal(v, device=dev)
        res["seconds"][name] = round(time.time() - t0, 2)
        res[name] = against(o)
    res["DIPY: mean |change of the scan| / scale"] = round(float(np.abs(ref - vol).mean() / scale), 6)
    (HERE / "results" / "gibbs_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
