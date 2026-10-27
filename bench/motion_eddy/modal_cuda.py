"""motion_eddy and gibbs on CUDA GPUs (Modal): ON-Harmony 16974 (ds004712, sub-16974 ses-OXF1PRI001), the same call
the pipeline makes (pipeline.correct_motion_eddy: the field, then motion and eddy, then one resample), and the gibbs
slot (pipeline.gibbs_subject).

    modal run bench/motion_eddy/modal_cuda.py [--gpus A10,L4] [--runs 2] [--profile] [--backend auto|torch]
    modal run bench/motion_eddy/modal_cuda.py --tests          # tests/test_motion_eddy.py on the GPU first
    modal run bench/motion_eddy/modal_cuda.py --cli            # `tractline run --motion-eddy --gibbs` end to end, twice
    modal run bench/motion_eddy/modal_cuda.py --cli --warm     # `tractline warm <series>` first, then the runs

Per GPU, one container runs the correction `runs` times (the first pays for torch's warm-up), reporting each stage's
time and the peak device memory; then gibbs on the series, its time and memory, and its float32 result on 8 volumes
against the CPU's. The last run's motion_eddy record (record(): the parameters and QC) comes back for the comparison
with the M2's (bench/motion_eddy/compare_records.py, against a record saved from MPS).
--profile: torch.profiler over one more correction, its top operations by CUDA time.

Writes results/motion_eddy/modal_cuda_<gpu>.json. Costs: an A10 about $1.10/h, an L4 $0.80/h; a few minutes each.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
PKG = HERE.parent.parent / "src/tractline"
SES = "ds004712/sub-16974/ses-OXF1PRI001/dwi"
SERIES = f"/data/{SES}/sub-16974_ses-OXF1PRI001_dir-AP_dwi.nii.gz"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.1", "numpy>=2", "scipy>=1.15", "nibabel", "pytest", "click", "matplotlib")
         .env({"PYTHONPATH": "/root/pkg", "TRACTOGRAPHY_DATA": "/data"})
         .add_local_dir(str(DATA / SES), remote_path=f"/data/{SES}")
         .add_local_dir(str(HERE.parent.parent / "tests"), remote_path="/root/tests")
         .add_local_dir(str(PKG), remote_path="/root/pkg/tractline"))
app = modal.App("tractline-motion-eddy-cuda", image=image)


@app.function(gpu="A10", timeout=3600, memory=32768, cpu=8)
def tests_gpu(select: str = "") -> str:
    """tests/test_motion_eddy.py on the GPU (the CUDA-only tests run; -k select narrows)."""
    import subprocess
    cmd = ["python", "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "/root/tests/test_motion_eddy.py"]
    r = subprocess.run(cmd + (["-k", select] if select else []), capture_output=True, text=True, cwd="/root")
    return r.stdout[-6000:] + r.stderr[-3000:]


@app.function(gpu="A10", timeout=3600, memory=32768, cpu=8)
def cli_gpu(warm: bool = False) -> str:
    """The command end to end on the GPU, twice (the first compiles the Triton kernels and fetches RapidParc's
    weights): its JSON summaries. warm: `tractline warm <series>` first, in its own process (its seconds first)."""
    import subprocess
    import time
    outs = []
    if warm:
        t0 = time.time()
        r = subprocess.run(["python", "-m", "tractline.cli", "warm", SERIES], capture_output=True, text=True)
        outs.append(json.dumps({"warm_s": round(time.time() - t0, 1), "exit": r.returncode, "stderr": r.stderr[-2000:]}))
    for k in range(2):
        r = subprocess.run(["python", "-m", "tractline.cli", "run", SERIES, "--motion-eddy", "--gibbs", "--device",
                            "cuda", "--trx", "/tmp/out.trx", "--qc", "/tmp/qc", "--json", "-", "-q"],
                           capture_output=True, text=True, env={**os.environ, "TRACTLINE_CACHE": "/tmp/cache"})
        outs.append(r.stdout if r.returncode == 0 else f"exit {r.returncode}: {r.stderr[-3000:]}")
    return json.dumps(outs)


@app.function(gpu="A10", timeout=3600, memory=32768, cpu=8)
def run_gpu(runs: int = 2, profile: bool = False, backend: str = "auto") -> str:
    import subprocess
    import time
    import numpy as np
    import torch
    from tractline import bids, pipeline as P, motion_eddy as ME
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    s = bids.load(SERIES)
    out = {"gpu": smi, "torch": torch.__version__, "shape": list(s.dwi.shape), "runs": []}
    for k in range(runs):
        torch.cuda.reset_peak_memory_stats()
        timer = P.Timer()
        t = time.perf_counter()
        corr = P.correct_motion_eddy(s, timer, device="cuda", opts={"backend": backend} if backend != "auto" else None)
        torch.cuda.synchronize()
        out["runs"].append({"backend": corr.motion_eddy["backend"], "seconds": timer.seconds, "wall_s": round(time.perf_counter() - t, 2),
                            "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                            "steps": corr.motion_eddy["qc"].get("steps run")})
        print(out["runs"][-1], flush=True)
    out["record"] = ME._finite(json.loads(json.dumps(corr.motion_eddy, default=lambda o: o.tolist()
                                                     if hasattr(o, "tolist") else str(o))))
    if profile:
        from torch.profiler import profile as prof, ProfilerActivity
        with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            P.correct_motion_eddy(s, P.Timer(), device="cuda", opts={"backend": backend} if backend != "auto" else None)
            torch.cuda.synchronize()
        out["profile"] = p.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        print(out["profile"], flush=True)
    # gibbs on the series, then 8 volumes against the CPU (float32 both)
    from tractline import gibbs as GB
    for k in range(2):
        torch.cuda.reset_peak_memory_stats()
        timer = P.Timer()
        g, counts = P.gibbs_subject(s, timer, {"slice_axis": P.slice_axis_of(s), "n_points": 3}, device="cuda")
        torch.cuda.synchronize()
        out[f"gibbs_run{k}"] = {"seconds": timer.seconds["gibbs"], "counts": counts,
                                "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        print(out[f"gibbs_run{k}"], flush=True)
    sub = np.asarray(s.dwi[..., :8], np.float32)
    a = GB.gibbs_removal(sub, P.slice_axis_of(s), 3, "cuda")
    b = GB.gibbs_removal(sub, P.slice_axis_of(s), 3, "cpu")
    scale = float(np.percentile(np.abs(sub), 99.9))
    d = np.abs(a - b) / scale
    out["gibbs_cuda_vs_cpu"] = {"voxels": int(d.size), "over_0.1pct": int((d > 1e-3).sum()),
                                "over_1pct": int((d > 1e-2).sum()), "max_pct": round(100 * float(d.max()), 3),
                                "mean_change_pct": round(100 * float(np.abs(a - sub).mean() / scale), 3)}
    print(out["gibbs_cuda_vs_cpu"], flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main(gpus: str = "A10", runs: int = 2, profile: bool = False, backend: str = "auto", tests: bool = False,
         select: str = "", cli: bool = False, warm: bool = False):
    """Writes results/motion_eddy/modal_cuda_<gpu>[_<backend>].json; compare_records.py compares its record with an
    MPS run's (record() saved as JSON)."""
    if tests:
        for g in gpus.split(","):
            print(g, tests_gpu.with_options(gpu=g).remote(select))
        return
    if cli:
        for g in gpus.split(","):
            outs = json.loads(cli_gpu.with_options(gpu=g).remote(warm))
            path = HERE.parent / "results" / "motion_eddy" / f"modal_cuda_cli_{g.lower()}{'_warm' if warm else ''}.json"
            path.write_text(json.dumps([json.loads(o) if o.startswith("{") else o for o in outs], indent=1))
            print(g, "->", path)
        return
    calls = {g: run_gpu.with_options(gpu=g).spawn(runs, profile, backend) for g in gpus.split(",")}
    for g, c in calls.items():
        res = json.loads(c.get())
        tag = "" if backend == "auto" else f"_{backend}"
        path = HERE.parent / "results" / "motion_eddy" / f"modal_cuda_{g.lower()}{tag}.json"
        path.write_text(json.dumps(res, indent=1))
        print(g, [r["wall_s"] for r in res["runs"]], "->", path)

