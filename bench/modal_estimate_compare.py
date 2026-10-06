"""The field estimate, committed code against the working copy, in one container per GPU: PAT16, two runs each,
the time per level with its Gauss-Newton and conjugate-gradient iterations, and the displacement differences in
PAT16's brain (deep, and its 3-voxel edge; TRACTLINE_BRAIN_NPZ, a cohort_fields.npz) - old against old and new against
new across processes (the GPU's own floor), new against old. For a regression seen on one GPU (the L40S, 2026-10-06).

    git archive <commit> src/tractline | tar -x -C /tmp/old
    TRACTLINE_OLD_PKG=/tmp/old/src/tractline modal run bench/modal_estimate_compare.py [--gpus L40S,A10]

Writes results/estimate_compare_<gpu>.json.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
PAT = "ds001226/sub-PAT16/ses-preop/dwi"
OLD = os.environ.get("TRACTLINE_OLD_PKG", "")
BRAIN = os.environ.get("TRACTLINE_BRAIN_NPZ", str(DATA / "ds001226/derived/PAT16/cohort_fields.npz"))   # its "brain" mask

RUNNER = r'''
import sys, time, json, numpy as np, torch
from _ds001226 import load
from tractline import susceptibility as S
s = load("PAT16")
out = []
for r in range(2):
    lev, mark = [], [None]
    def prog(L):
        torch.cuda.synchronize(); t = time.perf_counter()
        lev.append({"level": L["level"], "s": round(t - mark[0], 2), "gn": L.get("n_iter"), "cg": L.get("cg_iters"), "evals": L.get("func_evals")}); mark[0] = t
    torch.cuda.synchronize(); mark[0] = t0 = time.perf_counter()
    h, m, _ = S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s, device="cuda", dtype=torch.float32, progress=prog, diagnostics=True)
    torch.cuda.synchronize()
    out.append({"seconds": round(time.perf_counter() - t0, 2), "levels": lev})
np.save(sys.argv[1], S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis]))
print(json.dumps(out))
'''

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.1", "numpy>=2", "scipy", "nibabel")
         .env({"TRACTOGRAPHY_DATA": "/data"})
         .add_local_dir(str(DATA / PAT), remote_path=f"/data/{PAT}")
         .add_local_dir(str(HERE.parent / "src/tractline"), remote_path="/root/new/tractline")
         .add_local_file(str(HERE / "_ds001226.py"), remote_path="/root/bench/_ds001226.py"))
if OLD:
    image = image.add_local_dir(OLD, remote_path="/root/old/tractline")
image = image.add_local_file(BRAIN, remote_path="/root/brain.npz")
app = modal.App("tractline-estimate-compare", image=image)


@app.function(gpu="L40S", timeout=1800, memory=32768, cpu=8)
def compare() -> str:
    import subprocess, sys, numpy as np, torch
    open("/root/run.py", "w").write(RUNNER)
    res = {"gpu": torch.cuda.get_device_name(0)}
    for i, name in enumerate(("old", "new", "old", "new")):
        env = {**os.environ, "PYTHONPATH": f"/root/bench:/root/{name}"}
        r = subprocess.run([sys.executable, "/root/run.py", f"/tmp/d_{name}{i // 2}.npy"], env=env, capture_output=True, text=True)
        res.setdefault(name, []).append(json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else r.stderr[-2000:])
    from scipy.ndimage import binary_erosion
    brain = np.load("/root/brain.npz")["brain"]; deep = binary_erosion(brain, iterations=3); edge = brain & ~deep
    q = lambda d, m: [round(float(np.median(d[m])), 3), round(float(np.quantile(d[m], 0.99)), 3), round(float(d[m].max()), 3)]
    L = lambda k: np.load(f"/tmp/d_{k}.npy")
    res["displacement_diff_mm"] = {}
    for label, a, b in (("old vs old (two processes)", "old0", "old1"), ("new vs new (two processes)", "new0", "new1"), ("new vs old", "new0", "old0")):
        d = np.abs(L(a) - L(b))
        res["displacement_diff_mm"][label] = {"deep": q(d, deep), "edge": q(d, edge), "all_voxels": q(d, np.ones_like(brain, bool))}
    return json.dumps(res)


@app.local_entrypoint()
def main(gpus: str = "L40S,A10"):
    calls = {g: compare.with_options(gpu=g).spawn() for g in gpus.split(",")}
    for g, c in calls.items():
        res = json.loads(c.get())
        (HERE / f"results/estimate_compare_{g.lower()}.json").write_text(json.dumps(res, indent=1))
        print(g, res["gpu"])
        for name in ("old", "new"):
            for proc in res[name]:
                if isinstance(proc, str):
                    print(" ", name, "FAILED", proc[-500:]); continue
                for run in proc:
                    print(f"  {name} {run['seconds']:6.2f} s  ", " ".join(f"L{L['level']}:{L['s']}s/{L['gn']}gn/{L['cg']}cg" for L in run["levels"]))
        for k, v in res["displacement_diff_mm"].items():
            print(f"  {k:28s} deep {v['deep']}  edge {v['edge']}  all voxels {v['all_voxels']}")
