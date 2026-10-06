"""Where the CPU path's time goes, on PAT16, on this machine's CPU: the tracker's step taken apart, the tracker
by worker count, and the field estimate by thread count and level.

    uv run bench/cpu_profile.py [--what tracker,scaling,field] [--workers 4,8] [--threads 4,8] [--tag m2]
    modal run bench/modal_cpu_profile.py                     # the same on 32 x86 cores (Modal)

tracker: one process, one thread (a worker's setting), float32 with the fast step, batch 1,024 - the pipeline's
  CPU tracker, uncompiled (results before 2026-10-06's change: the earlier fast step) - on every 24th seed (4 blocks, the scan as acquired). Wall time in the step's parts (interp, the
  filter, H inside it and at the stopping check, the mask, the rest of advance), the step's time against the live
  half-fibers it carries, and torch.profiler's operators by self time over one block.
scaling: the same tracker on every 2nd seed (whole blocks) at each worker count, one thread each: steps/s.
field: susceptibility.estimate as pipeline.correct calls it on the CPU (float32, Gauss-Newton), at each thread
  count: total and per level; torch.profiler's operators over one run at the first count.
Writes results/cpu_profile_<tag>.json.
"""
import argparse, json, os, platform, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def machine():
    import torch
    cpu = "?"
    if Path("/proc/cpuinfo").exists():
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    else:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "torch": torch.__version__, "platform": platform.platform(),
            "torch_config_parallel": torch.__config__.parallel_info().splitlines()[:6]}


def tracker_input():
    from _ds001226 import load
    from tractline import ukf as U
    from tractline.prep import prepare
    s = load("PAT16")
    t = prepare(s.dwi, s.affine, s.bval, s.bvec, shell=2800.0, device="cpu")
    return s, U.from_arrays(t.dwi, t.header, t.mask)


def top_ops(prof, n=25):
    rows = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)
    total = sum(e.self_cpu_time_total for e in rows) or 1
    return [{"op": e.key, "calls": e.count, "self_ms": round(e.self_cpu_time_total / 1e3, 1),
             "share": round(e.self_cpu_time_total / total, 4)} for e in rows[:n]]


def tracker(D, every=24):
    """The step's parts, one process and one thread, by wrapping the module functions advance() looks up."""
    import torch
    from tractline import ukf as U
    torch.set_num_threads(1)
    nseeds = len(U.seeds(D)[0])
    sel = np.arange(0, nseeds, every)[:2048]                               # 4 blocks of 1,024 half-fibers
    acc, calls = defaultdict(float), defaultdict(int)
    in_filter = [False]
    # the fast step's functions (since 2026-10-06) and the exact path's, each under the part's name
    part = {"interp": "interp", "_interp_half": "interp", "ukf_step": "ukf_step", "_ukf_step_fast": "ukf_step",
            "H": "H", "_H_half": "H", "mask_at": "mask_at"}
    orig = {k: getattr(U, k) for k in (*part, "advance")}

    def timed(name, f):
        def w(*a, **k):
            t0 = time.perf_counter()
            r = f(*a, **k)
            key = "H (filter)" if name == "H" and in_filter[0] else ("H (stop check)" if name == "H" else name)
            acc[key] += time.perf_counter() - t0; calls[key] += 1
            return r
        return w

    def step_filter(f):
        def w(*a, **k):
            in_filter[0] = True
            try:
                return timed("ukf_step", f)(*a, **k)
            finally:
                in_filter[0] = False
        return w

    per_step = []                                                          # (live half-fibers, seconds)

    def advance(D, xa, *a, **k):
        t0 = time.perf_counter()
        r = orig["advance"](D, xa, *a, **k)
        per_step.append((len(xa), time.perf_counter() - t0))
        return r

    for k, name in part.items():
        setattr(U, k, step_filter(orig[k]) if name == "ukf_step" else timed(name, orig[k]))
    U.advance = advance
    try:
        t0 = time.perf_counter()
        fibers, st = U.track(D, select=sel, dtype=torch.float32, fast=True, batch=1024)
        wall = time.perf_counter() - t0
    finally:
        for k, v in orig.items():
            setattr(U, k, v)
    steps_total = sum(p[1] for p in per_step)
    parts = {k: round(v, 2) for k, v in acc.items()}
    parts["advance rest"] = round(steps_total - acc["interp"] - acc["ukf_step"] - acc["H (stop check)"] - acc["mask_at"], 2)
    parts["outside the step (loop, compaction, recording, seeds, joining)"] = round(wall - steps_total, 2)
    live = np.array([p[0] for p in per_step]); sec = np.array([p[1] for p in per_step])
    bins = [1, 16, 64, 256, 512, 1025]
    by_live = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (live >= lo) & (live < hi)
        if m.any():
            by_live.append({"live": f"{lo}-{hi - 1}", "steps": int(m.sum()), "seconds": round(float(sec[m].sum()), 2),
                            "ms_per_step": round(float(sec[m].mean() * 1e3), 2),
                            "us_per_half_fiber_step": round(float(sec[m].sum() / live[m].sum() * 1e6), 1)})
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:   # one block, unwrapped
        U.track(D, select=sel[:512], dtype=torch.float32, fast=True, batch=1024)
    ka = prof.key_averages()
    nops = sum(e.count for e in ka)
    nsteps = next((e.count for e in ka if e.key == "aten::nonzero"), 0)    # the loop's one nonzero a step (+1 in seeds)
    return {"seeds": len(sel), "fiber_steps": int(st["fiber_steps"]), "wall_s": round(wall, 2),
            "steps_per_s": round(st["fiber_steps"] / wall), "block_steps": len(per_step), "parts_s": parts,
            "by_live_half_fibers": by_live, "ops_per_block_step": round(nops / max(1, nsteps - 1)),
            "top_ops_one_block": top_ops(prof)}


def scaling(D, workers, every=2):
    import torch
    from tractline import ukf as U
    nseeds = len(U.seeds(D)[0])
    sel = np.arange(0, nseeds, every)
    sel = sel[:len(sel) // 512 * 512]                                      # whole blocks
    out = []
    for w in workers:
        t0 = time.perf_counter()
        _, st = U.track(D, select=sel, dtype=torch.float32, fast=True, batch=1024, workers=w, threads_per_worker=1)
        wall = time.perf_counter() - t0
        out.append({"workers": w, "wall_s": round(wall, 2), "steps_per_s": round(st["fiber_steps"] / wall),
                    "fiber_steps": int(st["fiber_steps"]), "blocks": len(sel) // 512})
        print(json.dumps(out[-1]), flush=True)
    return out


def field(s, threads):
    import torch
    from tractline import susceptibility as S
    out = []
    for i, n in enumerate(threads):
        torch.set_num_threads(n)
        marks = [time.perf_counter()]
        levels = []

        def progress(L):
            marks.append(time.perf_counter())
            levels.append({"level": L.get("level"), "seconds": round(marks[-1] - marks[-2], 2)})

        if i == 0:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s, device="cpu", dtype=torch.float32, progress=progress)
            ops = top_ops(prof)
            marks, levels = [time.perf_counter()], []                      # the timed run: unprofiled
        S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s, device="cpu", dtype=torch.float32, progress=progress)
        out.append({"threads": n, "seconds": round(marks[-1] - marks[0], 2), "levels": levels})
        if i == 0:
            out[-1]["top_ops_profiled_run"] = ops
        print(json.dumps({k: v for k, v in out[-1].items() if k != "top_ops_profiled_run"}), flush=True)
    return out


def profile(what=("tracker", "scaling", "field"), workers=(4, 8), threads=(4, 8)):
    s, D = tracker_input()
    out = {"machine": machine()}
    if "tracker" in what:
        out["tracker"] = tracker(D)
        print(json.dumps({k: v for k, v in out["tracker"].items() if k != "top_ops_one_block"}), flush=True)
    if "scaling" in what:
        out["scaling"] = scaling(D, workers)
    if "field" in what:
        out["field"] = field(s, threads)
    return out


if __name__ == "__main__":                                                  # the scaling runs spawn workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="tracker,scaling,field"); ap.add_argument("--workers", default="4,8")
    ap.add_argument("--threads", default="4,8"); ap.add_argument("--tag", default="m2")
    a = ap.parse_args()
    res = profile(tuple(a.what.split(",")), [int(w) for w in a.workers.split(",")], [int(t) for t in a.threads.split(",")])
    (HERE / f"results/cpu_profile_{a.tag}.json").write_text(json.dumps(res, indent=1))
