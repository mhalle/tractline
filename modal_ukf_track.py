"""_ukf_torch.track on an A10G in float64: all of Stanford HARDI, against the Slicer UKF binary.

    modal run bench/tractography/modal_ukf_track.py

Inputs on the tractography-bench Volume (ukf/hardi/: DWIConvert's dwi.nhdr/raw and the NRRD mask
that ukf_bench.py fed the binary). The seed offset is macOS libc's rand() after srand(0), the one
the reference binary drew (-4158, -2201, -2855 -> 0.5 voxel). Writes ukf/hardi/torch_fibers.npz
(points, offsets) to the Volume and results/ukf_torch.json.
"""
import json, time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.7", "numpy>=2", "pynrrd")
         .add_local_file(str(HERE / "_ukf_torch.py"), remote_path="/root/_ukf_torch.py"))
vol = modal.Volume.from_name("tractography-bench")
app = modal.App("tractography-ukf-track", image=image)


@app.function(gpu="A10G", volumes={"/vol": vol}, timeout=7200, memory=32768)
def run() -> dict:
    import sys
    import numpy as np, torch
    sys.path.insert(0, "/root")
    import _ukf_torch as U
    H = "/vol/ukf/hardi/"
    t0 = time.time()
    D = U.load(H + "dwi.nhdr", H + "mask.nrrd", device="cuda")
    off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
    torch.cuda.synchronize(); t1 = time.time()
    fibers, stats = U.track(D, off, progress=lambda s0, step, n: print(f"batch {s0} step {step} alive {n}", flush=True))
    torch.cuda.synchronize(); t2 = time.time()
    lens = np.array([len(f) for f in fibers])
    np.savez(H + "torch_fibers.npz", points=np.concatenate(fibers), offsets=np.r_[0, np.cumsum(lens)])
    vol.commit()
    stats.update({"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "dtype": "float64",
                  "load_s": round(t1 - t0, 1), "track_s": round(t2 - t1, 1), "points": int(lens.sum()),
                  "fiber_steps_per_s": int(stats["fiber_steps"] / (t2 - t1))})
    return stats


@app.local_entrypoint()
def main():
    r = run.remote()
    print(json.dumps(r, indent=1))
    (HERE / "results" / "ukf_torch.json").write_text(json.dumps(r, indent=1))
