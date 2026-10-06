"""cpu_profile.py on 32 x86 cores (Modal, cpu=32; the container shows 48 vCPUs, its CPU model not chosen).

    modal run bench/modal_cpu_profile.py [--what tracker,scaling,field] [--workers 8,16,24,32,48] [--threads 8,16,32]
    modal run bench/modal_cpu_profile.py --variants fast,half_bmm,half_bmm_gather      # ukf_cpu_tune.py there

PAT16's two series and the package travel in the image; about 10 minutes, ~$0.25. Writes results/cpu_profile_x86.json.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
PKG = HERE.parent / "src/tractline"
PAT = "ds001226/sub-PAT16/ses-preop/dwi"

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("g++")                                                 # torch.compile's C++ backend (the "compiled" variants)
         .uv_pip_install("torch==2.14.1", "numpy>=2", "scipy", "nibabel")
         .env({"PYTHONPATH": "/root/bench:/root/pkg", "TRACTOGRAPHY_DATA": "/data"})
         .add_local_dir(str(DATA / PAT), remote_path=f"/data/{PAT}")
         .add_local_dir(str(PKG), remote_path="/root/pkg/tractline")
         .add_local_file(str(HERE / "_ds001226.py"), remote_path="/root/bench/_ds001226.py")
         .add_local_file(str(HERE / "cpu_profile.py"), remote_path="/root/bench/cpu_profile.py")
         .add_local_file(str(HERE / "ukf_cpu_check.py"), remote_path="/root/bench/ukf_cpu_check.py")
         .add_local_file(str(HERE / "ukf_cpu_tune.py"), remote_path="/root/bench/ukf_cpu_tune.py"))
app = modal.App("tractline-cpu-profile", image=image)


@app.function(cpu=32, memory=32768, timeout=3600)
def run(what: str, workers: str, threads: str) -> str:
    import shutil
    if shutil.disk_usage("/dev/shm").total < 2 ** 31:
        os.environ["TRACTOGRAPHY_SHARE"] = "0"
    from cpu_profile import profile
    return json.dumps(profile(tuple(what.split(",")), [int(w) for w in workers.split(",")], [int(t) for t in threads.split(",")]))


@app.function(cpu=32, memory=32768, timeout=3600)
def tune(variants: str, check: bool) -> str:
    from ukf_cpu_tune import main
    return json.dumps(main(variants.split(","), check))


@app.local_entrypoint()
def main(what: str = "tracker,scaling,field", workers: str = "8,16,24,32,48", threads: str = "8,16,32", tag: str = "x86",
         variants: str = "", check: bool = True):
    if variants:                                                           # ukf_cpu_tune.py's variants instead
        res = json.loads(tune.remote(variants, check))
        (HERE / f"results/ukf_cpu_tune_{tag}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ("speed", "against_float64")}, indent=1))
        return
    res = json.loads(run.remote(what, workers, threads))
    (HERE / f"results/cpu_profile_{tag}.json").write_text(json.dumps(res, indent=1))
    print("written", tag)
