"""The Triton UKF step (ukf_triton.py) on an A10G, tested as the Metal kernel was (ukf_metal_check.py).

    modal run bench/modal_ukf_triton.py --stage smoke    # compile, one step on the fixtures
    modal run bench/modal_ukf_triton.py --stage tune     # one-step throughput by launch config
    modal run bench/modal_ukf_triton.py --stage full     # the whole brain, against float64
    ... --kind block                                                   # the compact kernel (ukf_triton_block.py)
    ... --gpu L40S                                                     # another GPU (results file tagged)

smoke: every fixture of DATA/ukf32/steps.npz (ukf32_compare.py) stepped by the kernel, against float64
       and torch's float32 step (both on the GPU), and against itself (two launches, bit for bit).
tune:  one step of 50,000 half-fibers per launch configuration (fibers per program F, warps), precise
       and fast math; the fastest precise one becomes the default.
full:  the whole brain (98,491 seeds) with the kernel, timed once (--repeat: twice, for determinism),
       the fibers written to the Volume and compared seed by seed with the float64 run
       modal_ukf_labels.py left there (variants/f64.npz) on this machine, not on the GPU.
Costs: GPUs bill per second (an A10G about $1.10/h, an L40S $1.95/h); sweeps compile one kernel per
configuration and are worth running once per GPU type. Timeouts are short on purpose.

Inputs on the tractography-bench Volume (ukf/hardi/). Writes results/ukf_triton_<stage>.json.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src/tractline"                                # the package, shipped as a directory
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.7", "numpy>=2", "pynrrd", "scipy")
         .env({"PYTHONPATH": "/root/pkg:/root", "TRITON_CACHE_DIR": "/vol/triton-cache"})  # compile once per kernel source
         .add_local_dir(str(PKG), remote_path="/root/pkg/tractline")
         .add_local_file(str(HERE / "_fibercmp.py"), remote_path="/root/_fibercmp.py")
         .add_local_file(str(DATA / "ukf32/steps.npz"), remote_path="/root/steps.npz"))
vol = modal.Volume.from_name("tractography-bench")
app = modal.App("tractography-ukf-triton", image=image)
HD = "/vol/ukf/hardi/"


def _setup():
    import numpy as np, torch
    from tractline import ukf as U
    D = U.load(HD + "dwi.nhdr", HD + "mask.nrrd")
    off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
    return D, off


def _one_step(fn, Dd, dtype, Z, **kw):
    import numpy as np, torch
    t = lambda a: torch.as_tensor(a).to(dtype).cuda()
    Q = torch.diag(torch.tensor([0.001] * 3 + [50.0] * 2 + [0.001] * 3 + [50.0] * 2, dtype=dtype)).cuda()
    out = {k: [] for k in ("x", "state", "P", "dir", "stop", "swap", "swap2", "fa", "mean_signal", "inside")}
    steps = np.unique(Z["step"])
    for s in steps:
        r = Z["step"] == s
        x, sa, Pa, m1, stop, info = fn(Dd, t(Z["x"][r]), t(Z["state"][r]), t(Z["P"][r]), t(Z["old"][r]), Q, 0.02, int(s), 834, **kw)
        for k, v in (("x", x), ("state", sa), ("P", Pa), ("dir", m1), ("stop", stop), *info.items()):
            out[k].append(v.cpu().numpy() if v.dtype == torch.bool else v.cpu().double().numpy())
    order = np.argsort(np.concatenate([np.flatnonzero(Z["step"] == s) for s in steps]))
    return {k: np.concatenate(v)[order] for k, v in out.items()}


def _kernel(kind):
    if kind == "block":
        from tractline import ukf_triton_block as T
    else:
        from tractline import ukf_triton as T
    return T


@app.function(gpu="A10G", volumes={"/vol": vol}, timeout=1200, memory=32768)
def smoke(kind: str = "unrolled") -> dict:
    import time
    import numpy as np, torch, triton
    from tractline import ukf as U
    import _fibercmp as C
    T = _kernel(kind)
    D, off = _setup()
    Z = np.load("/root/steps.npz")
    vox = D["voxel"].numpy()
    t0 = time.time()
    Dt = U.at_dtype(D, torch.float32, "cuda")
    tri = _one_step(T.advance, Dt, torch.float32, Z)
    compile_s = time.time() - t0
    tri2 = _one_step(T.advance, Dt, torch.float32, Z)
    ref = _one_step(U.advance, U.at_dtype(D, torch.float64, "cuda"), torch.float64, Z)
    t32 = _one_step(U.advance, Dt, torch.float32, Z)
    return {"kernel": kind, "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "triton": triton.__version__,
            "first_call_s_with_compile": round(compile_s, 1), "rows": int(len(Z["step"])),
            "triton_vs_f64": C.step_diff(tri, ref, vox), "torch32_vs_f64": C.step_diff(t32, ref, vox),
            "triton_vs_torch32": C.step_diff(tri, t32, vox),
            "triton_repeat_identical": bool(all(np.array_equal(tri[k], tri2[k]) for k in tri))}


@app.function(gpu="A10G", volumes={"/vol": vol}, timeout=1800, memory=32768)
def tune(kind: str = "unrolled") -> dict:
    import time
    import numpy as np, torch
    from tractline import ukf as U
    T = _kernel(kind)
    D, off = _setup()
    pts, fwd, inv, e1, fa = U.seeds(D, off)
    Dt = U.at_dtype(D, torch.float32, "cuda")
    B = 50000
    x = pts[:B].float().cuda(); s = fwd[:B].float().cuda(); o = e1[:B].float().cuda()
    P = (0.01 * torch.eye(10)).expand(B, 10, 10).contiguous().cuda()
    Q = torch.diag(torch.tensor([0.001] * 3 + [50.0] * 2 + [0.001] * 3 + [50.0] * 2)).cuda()
    rows = []
    kw = (lambda m: {"math": m}) if kind == "unrolled" else (lambda m: {})
    for math in (("precise", "fast") if kind == "unrolled" else ("precise",)):
        for F in ((1, 2, 4, 8) if kind == "unrolled" else (1, 2, 4)):
            for w in ((1, 2, 4, 8) if kind == "unrolled" else (2, 4, 8)):
                try:
                    t0 = time.time()
                    T.advance(Dt, x, s, P, o, Q, 0.02, 1, 834, F=F, num_warps=w, **kw(math))
                    torch.cuda.synchronize(); comp = time.time() - t0
                    for _ in range(2):
                        T.advance(Dt, x, s, P, o, Q, 0.02, 1, 834, F=F, num_warps=w, **kw(math))
                    torch.cuda.synchronize(); t0 = time.time()
                    for _ in range(5):
                        T.advance(Dt, x, s, P, o, Q, 0.02, 1, 834, F=F, num_warps=w, **kw(math))
                    torch.cuda.synchronize(); dt = (time.time() - t0) / 5
                    rows.append({"math": math, "F": F, "warps": w, "steps_per_s": round(B / dt), "first_call_s": round(comp, 1)})
                except Exception as e:
                    rows.append({"math": math, "F": F, "warps": w, "error": str(e)[:300]})
                print(rows[-1], flush=True)
    return {"gpu": torch.cuda.get_device_name(0), "batch": B, "configs": rows}


@app.function(gpu="A10G", volumes={"/vol": vol}, timeout=900, memory=32768, cpu=4)  # seeds are CPU work
def full(F: int = 0, warps: int = 0, kind: str = "block", repeat: bool = False, tag: str = "") -> dict:
    """Track the whole brain once (twice with `repeat`, for determinism) and write the fibers to the
    Volume (ukf/hardi/variants/triton_<tag>.npz); the comparison with float64 runs locally (compare
    stage), so the GPU is not billed for CPU work."""
    import os, time
    import numpy as np, torch
    from tractline import ukf as U
    T = _kernel(kind)
    if F:
        T.F_DEFAULT, T.WARPS_DEFAULT = F, warps
    D, off = _setup()
    pts, *_ = U.seeds(D, off)
    runs = []
    for _ in range(2 if repeat else 1):
        t0 = time.time()
        f, st = U.track(D, off, seed_points=pts, backend="triton" if kind == "unrolled" else "triton_block")
        torch.cuda.synchronize()
        runs.append((f, st, time.time() - t0))
    f, st, s = runs[-1]
    lens = np.array([len(x) for x in f])
    os.makedirs(HD + "variants", exist_ok=True)
    np.savez(HD + f"variants/triton_{tag or kind}.npz", points=np.concatenate(f).astype(np.float32),
             offsets=np.r_[0, np.cumsum(lens)], seed_index=np.array(st["seed_index"]))
    vol.commit()
    out = {"kernel": kind, "gpu": torch.cuda.get_device_name(0), "F": T.F_DEFAULT, "warps": T.WARPS_DEFAULT,
           "seeds": st["seeds"], "fibers": st["fibers"], "fiber_steps": st["fiber_steps"],
           "seconds_per_run": [round(r[2], 1) for r in runs], "steps_per_s_last_run": round(st["fiber_steps"] / s),
           "fibers_file": f"ukf/hardi/variants/triton_{tag or kind}.npz"}
    if repeat:
        (fa, _, _), (fb, _, _) = runs
        out["repeat_identical"] = bool(len(fa) == len(fb) and all(np.array_equal(a, b) for a, b in zip(fa, fb)))
    return out


def compare_local(tag: str) -> dict:
    """compare_variant.py in the data environment (numpy, the tracker), not the modal CLI's."""
    import subprocess
    r = subprocess.run([str(DATA / ".venv/bin/python"), str(HERE / "compare_variant.py"), f"triton_{tag}"],
                       capture_output=True, text=True, cwd=str(HERE))
    if r.returncode:
        return {"error": r.stderr[-1500:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


@app.local_entrypoint()
def main(stage: str = "smoke", f: int = 0, warps: int = 0, kind: str = "block", gpu: str = "A10G", repeat: bool = False):
    tag = ("" if kind == "unrolled" else f"_{kind}") + ("" if gpu == "A10G" else f"_{gpu.lower()}")
    if stage == "full":
        r = full.with_options(gpu=gpu).remote(f, warps, kind, repeat, tag.lstrip("_"))
        out = HERE / "results" / f"ukf_triton{tag}_{stage}.json"
        out.write_text(json.dumps(r, indent=1))                            # saved before anything can fail
        print(json.dumps(r, indent=1), flush=True)
        r["vs_f64"] = compare_local(tag.lstrip("_"))                     # on this machine, not the GPU's
    else:
        r = {"smoke": smoke, "tune": tune}[stage].with_options(gpu=gpu).remote(kind)
    print(json.dumps(r, indent=1))
    (HERE / "results" / f"ukf_triton{tag}_{stage}.json").write_text(json.dumps(r, indent=1))
