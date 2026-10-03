"""Memory across patients in one process, as a batch would run them: the labelers loaded once, then per
patient the pipeline (correction, the Metal tracker) and each labeler (TractCloud at 80 and 500,
RapidParc, hemiaug). After every stage: PyTorch's live MPS memory, the MPS driver's (live plus the
allocator's cache), and the process's physical footprint (what macOS charges it, swapped and
compressed pages included: proc_pid_rusage). Twice: as is, and with cleanup after each patient
(gc.collect, torch.mps.empty_cache).

    uv run bench/memory_batch.py [--cleanup] [--subs PAT05,PAT07,PAT08,PAT13]

Writes results/memory_batch_<as_is|cleanup>.json.
"""
import argparse, ctypes, gc, json, os, time
from pathlib import Path
import torch
from tractline import pipeline as P
from tractline.labelers import tractcloud, rapidparc
from _ds001226 import load

HERE = Path(__file__).resolve().parent


class RusageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
        "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
        "diskio_bytesread", "diskio_byteswritten")]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint_gb():
    info = RusageV2()
    _libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(info))           # RUSAGE_INFO_V2
    return round(info.phys_footprint / 2 ** 30, 2)


def snapshot(stage):
    torch.mps.synchronize()
    return {"stage": stage, "mps_live_gb": round(torch.mps.current_allocated_memory() / 2 ** 30, 2),
            "mps_driver_gb": round(torch.mps.driver_allocated_memory() / 2 ** 30, 2), "footprint_gb": footprint_gb()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--subs", default="PAT05,PAT07,PAT08,PAT13"); args = ap.parse_args()
    rows = [snapshot("start")]
    labelers = {"tractcloud_80": tractcloud.Labeler("mps", upstream=True), "tractcloud_500": tractcloud.Labeler("mps"),
                "rapidparc": rapidparc.Labeler("mps"), "rapidparc_hemiaug": rapidparc.Labeler("mps", model="hemiaug")}
    rows.append(snapshot("labelers loaded"))
    t0 = time.time()
    for sub in args.subs.split(","):
        s = load(sub); timer = P.Timer()
        corr = P.correct(s, timer, device="mps"); rows.append(snapshot(f"{sub} correct"))
        tg = P.track(s, corr.dwi, timer, device="mps"); rows.append(snapshot(f"{sub} track"))
        for name, lab in labelers.items():
            lab(tg.fibers, draws=(0,)); rows.append(snapshot(f"{sub} {name}"))
        del s, corr, tg
        if args.cleanup:
            gc.collect(); torch.mps.empty_cache()
        rows.append(snapshot(f"{sub} end"))
        print(json.dumps(rows[-1]), flush=True)
    res = {"cleanup": args.cleanup, "seconds": round(time.time() - t0, 1), "snapshots": rows}
    (HERE / f"results/memory_batch_{'cleanup' if args.cleanup else 'as_is'}.json").write_text(json.dumps(res, indent=1))
