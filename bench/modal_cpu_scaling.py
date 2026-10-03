"""The CPU pipeline on a many-core x86 machine (Modal, 32 physical cores = 64 vCPU, 32 GiB): how the
tracker scales with worker processes, and the pipeline end to end, on PAT16. One run, no GPU.

    modal run bench/modal_cpu_scaling.py [--what field | --what pipeline]

  1. our correction on the CPU (float32, Gauss-Newton: the pipeline's default), once;
  2. the tracker on the corrected scan (float32, fast, batch 1,024) with 8, 16, 32 and 64 workers
     (64: one per vCPU, hyperthreads) - the fibers checked identical across worker counts;
  3. the whole pipeline (pipeline.run, device "cpu") at the fastest worker count.
Everything the pipeline needs travels in the image (TractCloud's code and weights, PAT16's two DWI
series: ~60 MB); no Volume. Costs ~$0.30 (CPU $0.0000131 / core / s, memory $0.00000222 / GiB / s, about
10 minutes); the timeout caps it near $0.60. Writes results/modal_cpu_scaling.json.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
PKG = HERE.parent / "src/tractline"                                # the package, shipped as a directory
BENCH = ("_ds001226.py",)
PAT = "ds001226/sub-PAT16/ses-preop/dwi"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy>=2", "scipy", "nibabel")                       # the default pipeline's dependencies, no more
         .env({"PYTHONPATH": "/root/bench:/root/pkg", "TRACTOGRAPHY_DATA": "/data"})
         .add_local_dir(str(DATA / "TractCloud/src"), remote_path="/data/TractCloud/src")
         .add_local_dir(str(DATA / "TrainedModel"), remote_path="/data/TrainedModel")
         .add_local_file(str(DATA / "TrainData_800clu800ol/HCP_mass_center.npy"), remote_path="/data/TrainData_800clu800ol/HCP_mass_center.npy")
         .add_local_dir(str(DATA / "RapidParc"), remote_path="/data/RapidParc")              # the default labeler's weights
         .add_local_dir(str(DATA / PAT), remote_path=f"/data/{PAT}"))
image = image.add_local_dir(str(PKG), remote_path="/root/pkg/tractline")
for m in BENCH:
    image = image.add_local_file(str(HERE / m), remote_path=f"/root/bench/{m}")
app = modal.App("tractography-cpu-scaling", image=image)
CORES = 32


@app.function(cpu=CORES, memory=32768, timeout=1200)
def scaling():
    import os, shutil, time
    shm = shutil.disk_usage("/dev/shm").total
    if shm < 2 ** 31:
        os.environ["TRACTOGRAPHY_SHARE"] = "0"                       # each worker gets a copy of the signal instead
    import numpy as np, torch
    torch.set_num_threads(CORES)
    from tractline import pipeline as P, ukf as U
    from _ds001226 import load
    from tractline.prep import prepare
    model = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    res = {"cpu_model": model, "os_cpu_count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
           "physical_cores_requested": CORES, "dev_shm_gib": round(shm / 2 ** 30, 2), "shared_memory": os.environ.get("TRACTOGRAPHY_SHARE", "1") == "1",
           "torch": torch.__version__}
    print(res, flush=True)
    s = load("PAT16")
    timer = P.Timer(echo="correct")
    corr = P.correct(s, timer, device="cpu")
    t = prepare(corr.dwi, s.affine, s.bval, s.bvec)
    D = U.from_arrays(t.dwi, t.header, t.mask)
    res["correct_seconds"] = timer.seconds
    sweep, ref = [], None
    for w in (8, 16, 32, 64):
        try:
            t0 = time.time()
            f, st = U.track(D, dtype=torch.float32, device="cpu", fast=True, batch=P.CPU_BATCH, workers=w)
            tt = time.time() - t0
            same = None if ref is None else (len(f) == len(ref) and all(np.array_equal(a, b) for a, b in zip(f, ref)))
            ref = f if ref is None else ref
            sweep.append({"workers": w, "seconds": round(tt, 1), "k_steps_per_s": round(st["fiber_steps"] / tt / 1e3, 1),
                          "fibers": st["fibers"], "fiber_steps": st["fiber_steps"], "identical_to_8_workers": same})
        except Exception as e:                                       # report, don't lose the rest of the run
            sweep.append({"workers": w, "error": repr(e)[:300]})
        print(sweep[-1], flush=True)
    res["tracking_sweep"] = sweep
    ok = [r for r in sweep if "k_steps_per_s" in r]
    best = max(ok, key=lambda r: r["k_steps_per_s"])["workers"] if ok else 1
    timer = P.Timer(echo="pipeline")
    corr, tg, labels = P.run(s, P.default_labeler("cpu"), timer, device="cpu", workers=best)
    res["pipeline"] = {"workers": best, "seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages()),
                       "fibers": tg.stats["fibers"], "labeled": int(labels.keep.sum())}
    return json.dumps(res, default=float)                            # text: the local client has no torch to unpickle with


@app.function(cpu=CORES, memory=32768, timeout=1200)
def field_timing():
    """The field estimate's time on this machine: by thread count on a fresh process, then again after
    a tracking pool has run (the first run's estimate took 101.8 s after the pools, 33.7 s before),
    then the pipeline end to end."""
    import os, time
    import numpy as np, torch
    from tractline import pipeline as P, susceptibility as S, ukf as U
    from _ds001226 import load
    from tractline.prep import prepare
    s = load("PAT16")
    est = lambda: S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s, device="cpu", dtype=torch.float32)
    res = {"os_cpu_count": os.cpu_count(), "estimate_by_threads": {}}
    for th in (8, 16, 32, 48):
        torch.set_num_threads(th)
        t0 = time.time(); h, *_ = est(); res["estimate_by_threads"][th] = round(time.time() - t0, 1)
        print("estimate, threads", th, res["estimate_by_threads"][th], flush=True)
    torch.set_num_threads(32)
    corr = S.apply(s.dwi, h, s.pe_axis, s.pe_sign, s.readout_s)
    t = prepare(corr, s.affine, s.bval, s.bvec); D = U.from_arrays(t.dwi, t.header, t.mask)
    t0 = time.time(); U.track(D, dtype=torch.float32, device="cpu", fast=True, batch=P.CPU_BATCH, workers=32)
    res["tracking_32_workers_s"] = round(time.time() - t0, 1)
    res["threads_after_pool"] = torch.get_num_threads()
    t0 = time.time(); est(); res["estimate_after_pool_32_threads"] = round(time.time() - t0, 1)
    print(res, flush=True)
    timer = P.Timer(echo="pipeline")
    P.run(s, P.default_labeler("cpu"), timer, device="cpu", workers=32)
    res["pipeline"] = {"seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages())}
    return json.dumps(res)


@app.function(cpu=CORES, memory=32768, timeout=900)
def pipeline_only():
    """The pipeline end to end on the CPU at 32 workers, each stage setting its own threads."""
    import torch
    from tractline import pipeline as P
    from _ds001226 import load
    s = load("PAT16")
    timer = P.Timer(echo="pipeline")
    P.run(s, P.default_labeler("cpu"), timer, device="cpu", workers=32)
    return json.dumps({"seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages()),
                       "estimate_threads": P.ESTIMATE_THREADS})


@app.local_entrypoint()
def main(what: str = "scaling"):
    if what == "pipeline":
        res = json.loads(pipeline_only.remote())
        print(json.dumps(res, indent=1))
        (HERE / "results/modal_cpu_pipeline.json").write_text(json.dumps(res, indent=1))
        return
    if what == "field":
        res = json.loads(field_timing.remote())
        print(json.dumps(res, indent=1))
        (HERE / "results/modal_cpu_field_timing.json").write_text(json.dumps(res, indent=1))
        return
    res = json.loads(scaling.remote())
    print(json.dumps(res, indent=1))
    (HERE / "results/modal_cpu_scaling.json").write_text(json.dumps(res, indent=1))
