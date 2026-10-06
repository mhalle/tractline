"""The CPU path end to end on 32 x86 cores, three versions of the tracker in one container (Modal's CPU type
varies between containers, 1.5-1.8x): the committed package before the fast step changed ("old", a git
archive of HEAD passed as --old), the fast step uncompiled ("tuned", TRACTLINE_COMPILE=0) and compiled
("compiled", TRACTLINE_COMPILE=1, Inductor's cache empty at its first run). Each in its own process through
timing_matrix.py (two runs a scan; --only), the compiled one twice: a cold cache, then a warm one. The old
package's path comes from TRACTLINE_OLD_PKG (the image is defined when this file is imported):

    git archive <commit> src/tractline | tar -x -C /tmp/old
    TRACTLINE_OLD_PKG=/tmp/old/src/tractline modal run bench/modal_cpu_compare.py [--only hardi,pat16,ds005123]
        [--what compare,scaling] [--workers 8,16,32]

scaling (a second container, in parallel): the fast step uncompiled and compiled at each worker count on PAT16's
every 2nd seed. Writes results/cpu_compare.json and results/cpu_scaling.json.
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
OLD = os.environ.get("TRACTLINE_OLD_PKG", "")


def image(old):
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("g++")
            .uv_pip_install("torch==2.14.1", "numpy>=2", "scipy", "nibabel")
            .env({"TRACTOGRAPHY_DATA": "/data"})
            .add_local_dir(str(DATA / "RapidParc"), remote_path="/data/RapidParc")
            .add_local_file(str(DATA / "ukf/hardi/HARDI150.nii.gz"), remote_path="/data/ukf/hardi/HARDI150.nii.gz")
            .add_local_file(str(DATA / "ukf/hardi/HARDI150.bval"), remote_path="/data/ukf/hardi/HARDI150.bval")
            .add_local_file(str(DATA / "ukf/hardi/HARDI150.bvec"), remote_path="/data/ukf/hardi/HARDI150.bvec")
            .add_local_dir(str(DATA / PAT), remote_path=f"/data/{PAT}")
            .add_local_dir(str(DATA / D5), remote_path=f"/data/{D5}")
            .add_local_dir(str(PKG), remote_path="/root/new/tractline")
            .add_local_dir(old, remote_path="/root/old/tractline")
            .add_local_file(str(HERE / "_ds001226.py"), remote_path="/root/bench/_ds001226.py")
            .add_local_file(str(HERE / "timing_matrix.py"), remote_path="/root/bench/timing_matrix.py")
            .add_local_file(str(HERE / "cpu_profile.py"), remote_path="/root/bench/cpu_profile.py"))


app = modal.App("tractline-cpu-compare", image=image(OLD) if OLD else None)


@app.function(cpu=32, memory=32768, timeout=7200)
def compare(only: str, versions: str = "old,tuned,compiled_cold,compiled_warm") -> str:
    import shutil, subprocess, sys
    env0 = dict(os.environ)
    if shutil.disk_usage("/dev/shm").total < 2 ** 31:
        env0["TRACTOGRAPHY_SHARE"] = "0"
    out = {}
    for name, pkg, extra in (("old", "/root/old", {}), ("tuned", "/root/new", {"TRACTLINE_COMPILE": "0"}),
                             ("compiled_cold", "/root/new", {"TRACTLINE_COMPILE": "1"}),
                             ("compiled_warm", "/root/new", {"TRACTLINE_COMPILE": "1"})):
        if name not in versions.split(","):
            continue
        env = {**env0, **extra, "PYTHONPATH": f"/root/bench:{pkg}", "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor"}
        path = f"/tmp/{name}.json"
        r = subprocess.run([sys.executable, "/root/bench/timing_matrix.py", "--device", "cpu", "--workers", "32", "--runs", "2",
                            "--only", only, "--out", path], env=env, capture_output=True, text=True)
        print(name, "exit", r.returncode, r.stderr[-1500:] if r.returncode else "", flush=True)
        out[name] = json.load(open(path)) if r.returncode == 0 else {"error": r.stderr[-3000:]}
        for sc, v in out[name].get("scans", {}).items():
            print(name, sc, [(x["seconds"], x.get("scan_to_labels_s"), x.get("compile_s"), x.get("fibers")) for x in v["runs"]], flush=True)
    return json.dumps(out)


@app.function(cpu=32, memory=32768, timeout=7200)
def scaling(workers: str) -> str:
    """The fast step uncompiled and compiled at each worker count, PAT16's every 2nd seed (46 blocks), one
    container: total steps/s, the compile (in the parent) apart."""
    import shutil, sys, time
    sys.path[:0] = ["/root/bench", "/root/new"]
    if shutil.disk_usage("/dev/shm").total < 2 ** 31:
        os.environ["TRACTOGRAPHY_SHARE"] = "0"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_scaling"
    import numpy as np, torch
    from cpu_profile import tracker_input
    from tractline import ukf as U
    _, D = tracker_input()
    sel = np.arange(0, len(U.seeds(D)[0]), 2); sel = sel[:len(sel) // 512 * 512]
    out = []
    for comp in (False, True):
        for w in [int(x) for x in workers.split(",")]:
            t0 = time.perf_counter()
            _, st = U.track(D, select=sel, dtype=torch.float32, fast=True, batch=1024, workers=w, compile=comp)
            wall = time.perf_counter() - t0 - st["compile_s"]
            out.append({"compiled": comp, "workers": w, "track_s": round(wall, 1), "compile_s": st["compile_s"],
                        "steps_per_s": round(st["fiber_steps"] / wall)})
            print(json.dumps(out[-1]), flush=True)
    return json.dumps(out)


@app.local_entrypoint()
def main(only: str = "hardi,pat16,ds005123", what: str = "compare,scaling", workers: str = "8,16,32",
         versions: str = "old,tuned,compiled_cold,compiled_warm", tag: str = ""):
    calls = {}
    if "compare" in what:
        calls["compare"] = compare.spawn(only, versions)
    if "scaling" in what:
        calls["scaling"] = scaling.spawn(workers)
    for k, c in calls.items():
        res = json.loads(c.get())
        (HERE / f"results/cpu_{k}{'_' + tag if tag else ''}.json").write_text(json.dumps(res, indent=1))
        print("written", k)
