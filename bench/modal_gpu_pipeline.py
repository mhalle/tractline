"""The pipeline end to end on CUDA GPUs (Modal), PAT16: the field estimate in float32 on the GPU, the
Triton block kernel for the tracker, the default labeler (RapidParc) on the GPU.

    modal run bench/modal_gpu_pipeline.py [--gpus A10,L40S,CPU] [--runs 2] [--tf32 on] [--save-fibers] [--copies 1]

CPU: a Modal CPU container (cpu=32). --tf32 on: TF32 left on inside the pipeline (the experiment; by
default it is off, as the pipeline runs - earlier results' "--tf32 off" files, _c0_/_c1_tf32off, are that
default). --save-fibers: the first run's tractogram and labels to the Volume, tractline/PAT16_<gpu>_tf32<mode>.npz;
fetch one with `modal volume get tractography-bench tractline/<file> $TRACTOGRAPHY_DATA/ds001226/derived/PAT16/`
(modal_rapidparc_check.py reads PAT16_a10_tf32off.npz, saved so by an earlier version).

Per GPU, one container runs the pipeline twice: the first run pays for torch's and Triton's warm-up
(the kernel's compile is cached on the tractography-bench Volume, per GPU architecture), the second is
the steady state. Each run returns its stage times, the fiber and label counts and the tract counts;
the first also the field (float32, for the comparison with the M2's, `gpu_pipeline_compare.py`).
Everything the pipeline needs travels in the image (RapidParc's weights, PAT16's two DWI
series: ~60 MB). Costs: an A10 about $1.10/h, an L40S $1.95/h; a few minutes each.

Writes results/modal_gpu_pipeline_<gpu>.json and DATA/ds001226/derived/PAT16/gpu_field_<gpu>.b64.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
PKG = HERE.parent / "src/tractline"                                # the package, shipped as a directory
PAT = "ds001226/sub-PAT16/ses-preop/dwi"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.14.1", "numpy>=2", "scipy", "nibabel")      # torch's CUDA wheel brings Triton
         .env({"PYTHONPATH": "/root/bench:/root/pkg", "TRACTOGRAPHY_DATA": "/data", "TRITON_CACHE_DIR": "/vol/triton-cache"})
         .add_local_dir(str(DATA / "RapidParc"), remote_path="/data/RapidParc")
         .add_local_dir(str(DATA / PAT), remote_path=f"/data/{PAT}")
         .add_local_dir(str(PKG), remote_path="/root/pkg/tractline")
         .add_local_file(str(HERE / "_ds001226.py"), remote_path="/root/bench/_ds001226.py"))
app = modal.App("tractline-gpu-pipeline", image=image)
vol = modal.Volume.from_name("tractography-bench")


@app.function(gpu="A10", volumes={"/vol": vol}, timeout=1800, memory=32768, cpu=8)
def pipeline(runs: int = 2, tf32: str = "default", save_fibers: bool = False) -> str:
    return body("cuda", runs, tf32, save_fibers)


@app.function(cpu=32, memory=32768, timeout=1800, volumes={"/vol": vol})
def pipeline_cpu(runs: int = 1, tf32: str = "default", save_fibers: bool = False) -> str:
    """The CPU path on 32 x86 cores: the tracker on 32 worker processes."""
    import os, shutil
    if shutil.disk_usage("/dev/shm").total < 2 ** 31:
        os.environ["TRACTOGRAPHY_SHARE"] = "0"
    return body("cpu", runs, tf32, save_fibers)


def body(device, runs, tf32, save_fibers):
    import base64, zlib
    import numpy as np, torch
    from tractline import pipeline as P
    from _ds001226 import load
    s = load("PAT16")
    if tf32 == "on":                                                       # the experiment: TF32 left on inside the pipeline
        P.exact_float32 = lambda device: __import__("contextlib").nullcontext()
    import subprocess
    cuda = device == "cuda"
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,pci.device_id,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() if cuda else None
    model = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    out = {"nvidia_smi": smi, "tf32": tf32, "gpu": torch.cuda.get_device_name(0) if cuda else f"CPU: {model}", "torch": torch.__version__,
           "cpu_count": __import__("os").cpu_count(), "runs": []}
    labeler = P.default_labeler(device)
    for r in range(runs):
        timer = P.Timer(echo=f"run {r}")
        corr, tg, labels = P.run(s, labeler, timer, device=device, workers=None if cuda else 32)
        import hashlib
        out["runs"].append({"field_sha": hashlib.sha256(corr.field_hz.astype(np.float32).tobytes()).hexdigest()[:16],
                            "fibers_sha": hashlib.sha256(np.concatenate(tg.fibers).astype(np.float32).tobytes()).hexdigest()[:16],
                            "seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages()),
                            "fibers": len(tg.fibers), "fiber_steps": int(tg.stats["fiber_steps"]), "labeled": int(labels.keep.sum()),
                            "tract_counts": np.bincount(labels.tract, minlength=43).tolist()})
        if r == 0 and save_fibers:                                         # the tractogram and its labels, to label elsewhere
            import os
            os.makedirs("/vol/tractline", exist_ok=True)
            name = f"/vol/tractline/PAT16_{torch.cuda.get_device_name(0).split()[-1].lower() if cuda else 'cpu'}_tf32{tf32}.npz"
            np.savez(name, points=np.concatenate(tg.fibers).astype(np.float32), offsets=np.r_[0, np.cumsum([len(f) for f in tg.fibers])],
                     keep=labels.keep, tract=labels.tract)
            out["fibers_file"] = name
        if r == 0:
            out["field_b64"] = base64.b64encode(zlib.compress(corr.field_hz.astype(np.float32).tobytes())).decode()
            out["field_shape"] = list(corr.field_hz.shape)
        print(json.dumps({k: v for k, v in out["runs"][-1].items() if k != "tract_counts"}), flush=True)
    vol.commit()                                                           # the Triton cache; any saved fibers
    return json.dumps(out)


@app.local_entrypoint()
def main(gpus: str = "A10,L40S", runs: int = 2, tf32: str = "default", save_fibers: bool = False, copies: int = 1):
    calls = {(f"{g}_c{i}" if copies > 1 else g): (pipeline_cpu.spawn(1, tf32, save_fibers) if g == "CPU" else
                                                  pipeline.with_options(gpu=g).spawn(runs, tf32, save_fibers))
             for g in gpus.split(",") for i in range(copies)}
    tag = "" if tf32 == "default" else f"_tf32{tf32}"
    for g, c in calls.items():
        res = json.loads(c.get())
        field = res.pop("field_b64")
        (DATA / "ds001226/derived/PAT16" / f"gpu_field_{g.lower()}{tag}.b64").write_text(field)
        (HERE / "results" / f"modal_gpu_pipeline_{g.lower()}{tag}.json").write_text(json.dumps(res, indent=1))
        print(g, json.dumps({"gpu": res["gpu"], "nvidia_smi": res["nvidia_smi"], "runs": [{k: v for k, v in r.items() if k != "tract_counts"} for r in res["runs"]]}, indent=1))
