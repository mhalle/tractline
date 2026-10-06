"""gpu_track_profile.py on CUDA GPUs (Modal): the Triton tracker's steps timed against the half-fibers they carry,
block sizes compared bit for bit; an L4, an A10 and an L40S in parallel. The scans travel in the image as
modal_timing_matrix.py's; the Triton cache is the Volume's (pre-warmed: no compile in the timings).

    modal run bench/modal_gpu_track_profile.py [--gpus L4,A10,L40S] [--scans pat16,hardi,ds005123] [--batches 50000,200000]

Writes results/gpu_track_profile_<gpu>.json.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
PAT = "ds001226/sub-PAT16/ses-preop/dwi"
D5 = "ds005123_full/sub-10317"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.1", "numpy>=2", "scipy", "nibabel")
         .env({"PYTHONPATH": "/root/bench:/root/pkg", "TRACTOGRAPHY_DATA": "/data", "TRITON_CACHE_DIR": "/vol/triton-cache"})
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.nii.gz"), remote_path="/data/ukf/hardi/HARDI150.nii.gz")
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.bval"), remote_path="/data/ukf/hardi/HARDI150.bval")
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.bvec"), remote_path="/data/ukf/hardi/HARDI150.bvec")
         .add_local_dir(str(DATA / PAT), remote_path=f"/data/{PAT}")
         .add_local_dir(str(DATA / D5), remote_path=f"/data/{D5}")
         .add_local_dir(str(HERE.parent / "src/tractline"), remote_path="/root/pkg/tractline")
         .add_local_file(str(HERE / "_ds001226.py"), remote_path="/root/bench/_ds001226.py")
         .add_local_file(str(HERE / "timing_matrix.py"), remote_path="/root/bench/timing_matrix.py")
         .add_local_file(str(HERE / "gpu_track_profile.py"), remote_path="/root/bench/gpu_track_profile.py"))
app = modal.App("tractline-gpu-track-profile", image=image)
vol = modal.Volume.from_name("tractography-bench")


@app.function(gpu="L40S", volumes={"/vol": vol}, timeout=3600, memory=32768, cpu=8)
def run(scans: str, batches: str) -> str:
    from gpu_track_profile import profile
    return json.dumps(profile(tuple(scans.split(",")), "cuda", tuple(int(b) for b in batches.split(","))))


@app.local_entrypoint()
def main(gpus: str = "L4,A10,L40S", scans: str = "pat16,hardi,ds005123", batches: str = "50000,200000"):
    calls = {g: run.with_options(gpu=g).spawn(scans, batches) for g in gpus.split(",")}
    for g, c in calls.items():
        try:
            res = json.loads(c.get())
        except Exception as e:                                             # noqa: BLE001
            print(g, "FAILED", type(e).__name__, e); continue
        (HERE / f"results/gpu_track_profile_{g.lower()}.json").write_text(json.dumps(res, indent=1))
        print(g, "written")
