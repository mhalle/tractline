"""Where the GPU tracker's time goes, on one device: every step of the Metal (mps) or Triton (cuda) kernel timed
with the device synchronized, against the half-fibers it carried; the rest of the loop (compaction, recording,
seeds, joining) apart; and the same tracking at other block sizes, the fibers compared bit for bit.

    uv run bench/gpu_track_profile.py [--scans hardi,pat16,ds005123] [--batches 50000,200000]   # the M2
    modal run bench/modal_gpu_track_profile.py                                                   # CUDA GPUs

The scans as timing_matrix.py reads them, tracked as acquired (the tracker's cost does not depend on the
correction). "seeds_s": ukf.seeds() alone, on the host - part of every track()'s "loop_s". batch: half-fibers a block (ukf.track's; the pipeline's 50,000): each block steps until its longest
fiber stops. Writes results/gpu_track_profile_<device>.json (or returns it).
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BINS = [1, 256, 1024, 4096, 16384, 10 ** 9]


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def tracker_input(name, device):
    from timing_matrix import read, SHELL
    from tractline.prep import prepare
    from tractline import ukf as U
    s = read(name)
    t = prepare(s.dwi, s.affine, s.bval, s.bvec, shell=SHELL[name], device=device)
    return U.from_arrays(t.dwi, t.header, t.mask), int(t.info.get("directions", 0))       # on the host, as pipeline.track


def profile(scans=("hardi", "pat16", "ds005123"), device="mps", batches=(50_000, 200_000)):
    from tractline import ukf as U
    if device == "mps":
        from tractline import ukf_metal as K
        backend = "metal"
    else:
        from tractline import ukf_triton_block as K
        backend = "triton_block"
    out = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "Apple M2 (mps)", "torch": torch.__version__,
           "scans": {}}
    orig = K.advance
    rec = []

    def timed(D, xa, *a, **k):
        sync(device); t0 = time.perf_counter()
        r = orig(D, xa, *a, **k)
        sync(device); rec.append((len(xa), time.perf_counter() - t0))
        return r

    K.advance = timed
    try:
        for name in scans:
            D, ndir = tracker_input(name, device)
            res = out["scans"][name] = {"directions": ndir, "batches": {}}
            U.track(D, backend=backend, batch=4096, select=np.arange(256))       # warm the kernel (and its compile)
            t0 = time.perf_counter(); U.seeds(D); res["seeds_s"] = round(time.perf_counter() - t0, 2)   # track() does this first
            for batch in batches:
                rec.clear(); sync(device); t0 = time.perf_counter()
                fibers, st = U.track(D, backend=backend, batch=batch)
                sync(device); wall = time.perf_counter() - t0
                live = np.array([r[0] for r in rec]); sec = np.array([r[1] for r in rec])
                by = []
                for lo, hi in zip(BINS[:-1], BINS[1:]):
                    m = (live >= lo) & (live < hi)
                    if m.any():
                        by.append({"live": f"{lo}-{min(hi, 10 ** 6) - 1}", "steps": int(m.sum()), "kernel_s": round(float(sec[m].sum()), 2),
                                   "ms_per_step": round(float(sec[m].mean() * 1e3), 3),
                                   "ns_per_half_fiber_step": round(float(sec[m].sum() / live[m].sum() * 1e9), 1)})
                res["batches"][str(batch)] = {
                    "blocks": -(-int(st["half_fibers"]) // batch), "wall_s": round(wall, 2), "kernel_s": round(float(sec.sum()), 2),
                    "loop_s": round(wall - float(sec.sum()), 2), "block_steps": len(rec), "fiber_steps": int(st["fiber_steps"]),
                    "half_fiber_steps_per_s": round(st["fiber_steps"] / wall), "fibers": len(fibers),
                    "fibers_sha": hashlib.sha256(np.concatenate(fibers).astype(np.float32).tobytes()).hexdigest()[:16], "by_live": by}
                print(json.dumps({"scan": name, "batch": batch, **{k: v for k, v in res["batches"][str(batch)].items() if k != "by_live"}}), flush=True)
            shas = {b["fibers_sha"] for b in res["batches"].values()}
            res["bit_identical_across_batches"] = len(shas) == 1
    finally:
        K.advance = orig
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", default="hardi,pat16,ds005123"); ap.add_argument("--batches", default="50000,200000")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    res = profile(tuple(a.scans.split(",")), a.device, tuple(int(b) for b in a.batches.split(",")))
    (HERE / f"results/gpu_track_profile_{a.device}.json").write_text(json.dumps(res, indent=1))
