"""timing_matrix.py on Modal: an L4, an A10 and an L40S (the Triton tracker, the field and RapidParc on
CUDA) and 32 x86 cores (the CPU path: the tracker on 32 worker processes), all in parallel.

    modal run bench/modal_timing_matrix.py [--targets L4,A10,L40S,CPU] [--runs 2] [--only hardi,pat16]
    modal run bench/modal_timing_matrix.py --targets L4,A10,L40S --cache cold --tag nospec      # a cold compile
    modal run bench/modal_timing_matrix.py --targets L4 --cache cold --prewarm --tag prewarm   # warm() first

--cache cold: each GPU target gets a new, empty Triton cache on the Volume (triton-cache-test/<time>-<target>),
so run 0 of each scan pays its cold compile; by default the shared triton-cache. --prewarm: ukf_triton_block.warm()
first (every NPAD compiled, timed; --warm-workers N its processes, default one per NPAD, 1 serial) - run 0 then
should compile nothing. --only none: warm() alone, no scans. --tag: the results' suffix.

The scans travel in the image (HARDI 91 MB, PAT16's two series 46 MB, ds005123 sub-10317 191 MB) with
RapidParc's weights; the Triton compile cache is on the tractography-bench Volume. Costs: an L4 about
$0.80/h, an A10 $1.10/h, an L40S $1.95/h, 32 cores about $1.50/h; under ten minutes each.
Writes results/timing_matrix_<target>.json.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
PKG = HERE.parent / "src/tractline"
PAT = "ds001226/sub-PAT16/ses-preop/dwi"
D5 = "ds005123_full/sub-10317"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.1", "numpy>=2", "scipy", "nibabel")
         .env({"PYTHONPATH": "/root/bench:/root/pkg", "TRACTOGRAPHY_DATA": "/data", "TRITON_CACHE_DIR": "/vol/triton-cache"})
         .add_local_dir(str(DATA / "RapidParc"), remote_path="/data/RapidParc")
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.nii.gz"), remote_path="/data/ukf/hardi/HARDI150.nii.gz")
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.bval"), remote_path="/data/ukf/hardi/HARDI150.bval")
         .add_local_file(str(DATA / "ukf/hardi/HARDI150.bvec"), remote_path="/data/ukf/hardi/HARDI150.bvec")
         .add_local_dir(str(DATA / PAT), remote_path=f"/data/{PAT}")
         .add_local_dir(str(DATA / D5), remote_path=f"/data/{D5}")
         .add_local_dir(str(PKG), remote_path="/root/pkg/tractline")
         .add_local_file(str(HERE / "_ds001226.py"), remote_path="/root/bench/_ds001226.py")
         .add_local_file(str(HERE / "timing_matrix.py"), remote_path="/root/bench/timing_matrix.py"))
app = modal.App("tractline-timing-matrix", image=image)
vol = modal.Volume.from_name("tractography-bench")
CORES = 32


@app.function(gpu="L40S", volumes={"/vol": vol}, timeout=3600, memory=32768, cpu=8)
def on_gpu(runs: int = 2, only: str = "", cache: str = "", prewarm: bool = False, warm_workers: int = 0) -> str:
    import os
    if cache:                                                              # before anything compiles
        os.environ["TRITON_CACHE_DIR"] = f"/vol/{cache}"
        os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    from timing_matrix import matrix, SCANS
    warm = wall = None
    if prewarm:
        import time
        from tractline import ukf_triton_block as K
        t0 = time.time()
        warm = K.warm(workers=warm_workers or None)
        wall = round(time.time() - t0, 1)
        print("warm", warm, "wall", wall, flush=True)
    out = matrix("cuda", runs, only=tuple(only.split(",")) if only else SCANS) if only != "none" else {}
    out["triton_cache"], out["warm_s"], out["warm_wall_s"], out["warm_workers"] = os.environ["TRITON_CACHE_DIR"], warm, wall, warm_workers
    vol.commit()                                                           # the Triton cache, per architecture
    return json.dumps(out)


@app.function(cpu=CORES, memory=32768, timeout=3600)
def on_cpu(runs: int = 2) -> str:
    import os, shutil
    if shutil.disk_usage("/dev/shm").total < 2 ** 31:
        os.environ["TRACTOGRAPHY_SHARE"] = "0"                             # each worker gets a copy of the signal instead
    from timing_matrix import matrix
    return json.dumps(matrix("cpu", runs, workers=CORES))


@app.local_entrypoint()
def main(targets: str = "L4,A10,L40S,CPU", runs: int = 2, only: str = "", cache: str = "", prewarm: bool = False, tag: str = "",
         warm_workers: int = 0):
    import time
    stamp = time.strftime("%Y%m%d-%H%M%S")
    calls = {t: (on_cpu.spawn(runs) if t == "CPU" else
                 on_gpu.with_options(gpu=t).spawn(runs, only, f"triton-cache-test/{stamp}-{t.lower()}" if cache == "cold" else cache,
                                                  prewarm, warm_workers))
             for t in targets.split(",")}
    for t, c in calls.items():
        try:
            res = json.loads(c.get())
        except Exception as e:                                             # noqa: BLE001 - one target's failure keeps the others
            print(t, "FAILED", type(e).__name__, e)
            continue
        (HERE / f"results/timing_matrix_{t.lower()}{'_' + tag if tag else ''}.json").write_text(json.dumps(res, indent=1))
        print(t, "written")
