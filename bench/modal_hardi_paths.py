"""hardi_paths.py on Modal: an L40S (Triton tracker, TractCloud on CUDA) and 32 x86 cores (the CPU
tracker, one process per core). The two run in parallel.

    modal run bench/modal_hardi_paths.py [--what gpu,cpu] [--gpu L40S]

The HARDI scan (91 MB), TractCloud's code and weights travel in the image; the Triton compile cache is
on the tractography-bench Volume. Costs: the L40S about $1.95/h, 32 cores about $1.50/h; a few minutes.
Writes results/hardi_paths_cuda_<gpu>.json and results/hardi_paths_cpu_modal.json.
"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
PKG = HERE.parent / "src/tractline"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=2", "scipy", "nibabel")
         .env({"PYTHONPATH": "/root/bench:/root/pkg", "TRACTOGRAPHY_DATA": "/data", "TRITON_CACHE_DIR": "/vol/triton-cache"})
         .add_local_dir(str(DATA / "TractCloud/src"), remote_path="/data/TractCloud/src")
         .add_local_dir(str(DATA / "TrainedModel"), remote_path="/data/TrainedModel")
         .add_local_file(str(DATA / "TrainData_800clu800ol/HCP_mass_center.npy"), remote_path="/data/TrainData_800clu800ol/HCP_mass_center.npy")
         .add_local_dir(str(DATA / "RapidParc"), remote_path="/data/RapidParc")
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.nii.gz"), remote_path="/data/ukf/hardi/HARDI150.nii.gz")
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.bval"), remote_path="/data/ukf/hardi/HARDI150.bval")
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.bvec"), remote_path="/data/ukf/hardi/HARDI150.bvec")
         .add_local_dir(str(PKG), remote_path="/root/pkg/tractline")
         .add_local_file(str(HERE / "hardi_paths.py"), remote_path="/root/bench/hardi_paths.py"))
app = modal.App("tractline-hardi-paths", image=image)
vol = modal.Volume.from_name("tractography-bench")
CORES = 32


@app.function(gpu="L40S", volumes={"/vol": vol}, timeout=1800, memory=32768, cpu=8)
def on_gpu(runs: int = 2) -> str:
    from hardi_paths import paths
    out = paths("cuda", runs)
    vol.commit()
    return json.dumps(out)


@app.function(cpu=CORES, memory=32768, timeout=1800)
def on_cpu(runs: int = 1) -> str:
    import os, shutil
    if shutil.disk_usage("/dev/shm").total < 2 ** 31:
        os.environ["TRACTOGRAPHY_SHARE"] = "0"                       # each worker gets a copy of the signal instead
    from hardi_paths import paths
    return json.dumps(paths("cpu", runs, workers=CORES))


@app.local_entrypoint()
def main(what: str = "gpu,cpu", gpu: str = "L40S"):
    calls = {}
    if "gpu" in what:
        calls[f"cuda_{gpu.lower()}"] = on_gpu.with_options(gpu=gpu).spawn()
    if "cpu" in what:
        calls["cpu_modal"] = on_cpu.spawn()
    for name, c in calls.items():
        res = json.loads(c.get())
        (HERE / f"results/hardi_paths_{name}.json").write_text(json.dumps(res, indent=1))
        print(name, json.dumps(res, indent=1))
