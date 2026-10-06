"""The pipeline's stage times on three scans, on one device: the cross-platform timing table.

    uv run bench/timing_matrix.py [--device mps|cpu] [--runs 2] [--only hardi,pat16,ds005123,ds005123_forced]
    modal run bench/modal_timing_matrix.py                    # an L4, an A10, an L40S and 32 x86 cores

The scans, each through pipeline.run with the default labeler (RapidParc, one draw):
  hardi      Stanford HARDI (2 mm, 81x106x76, 150 directions at b = 2000): no reversed pair, so no
             correction; the b = 2000 shell tracked
  pat16      ds001226 PAT16 (Siemens Trio, 2.5 mm, 96x96x60, 102 volumes): corrected from its AP/PA b0s;
             the b = 2800 shell (50 directions)
  ds005123   ds005123 sub-10317 (Siemens Prisma, 2 mm, 120x120x69, 145 volumes): its spin-echo field maps
             are refused (acquired under another shim), so it is tracked as acquired; the b = 3250 shell
             (30 directions, the nearest to 3000). Needs DATA/ds005123_full/sub-10317 (the full series,
             191 MB from OpenNeuro, beside the fetched field maps).
  ds005123_forced   the correction alone on ds005123 with the shim check switched off - the field estimate
             and its application timed on a second scanner's grid. TIMING ONLY: this field is not the
             diffusion series' (the refusal is right; NOTES 2026-10-03).

Each scan runs `runs` times in one process: run 0 pays the device's warm-up (and on CUDA the Triton
compile when not cached), the last is the steady state. "read" (the NIfTI into memory, and for the BIDS
scans the partners found and resampled) is timed apart from the pipeline's stages. Writes
results/timing_matrix_<device>.json (or returns the dict, on Modal).
"""
import argparse, hashlib, json, os, platform, subprocess, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
SCANS = ("hardi", "pat16", "ds005123", "ds005123_forced")
SHELL = {"hardi": 2000.0, "pat16": 2800.0, "ds005123": 3250.0}


def read(name):
    from tractline.data import DATA
    if name == "hardi":
        import nibabel as nib
        H = DATA / "ukf/hardi"
        img = nib.load(H / "HARDI150.nii.gz")
        return SimpleNamespace(dwi=np.asarray(img.dataobj), affine=img.affine, vox=np.asarray(img.header.get_zooms()[:3], float),
                               bval=np.loadtxt(H / "HARDI150.bval"), bvec=np.loadtxt(H / "HARDI150.bvec"))
    if name == "pat16":
        from _ds001226 import load
        return load("PAT16")
    from tractline import bids
    path = DATA / "ds005123_full/sub-10317/dwi/sub-10317_dwi.nii.gz"
    if name == "ds005123":
        return bids.load(path)
    sidecar = bids._sidecar                                                 # ds005123_forced: the shim check off
    bids._sidecar = lambda p: {k: v for k, v in sidecar(p).items() if k != "ShimSetting"}
    try:
        return bids.load(path)
    finally:
        bids._sidecar = sidecar


def machine(device):
    import torch
    out = {"device": device, "torch": torch.__version__, "cpu_count": os.cpu_count(), "platform": platform.platform()}
    if device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
        out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip()
    if Path("/proc/cpuinfo").exists():
        out["cpu"] = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    else:
        out["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return out


def matrix(device="mps", runs=2, workers=None, only=SCANS):
    from tractline import pipeline as P
    out = {**machine(device), "workers": workers, "scans": {}}
    labeler = P.default_labeler(device)
    for name in only:
        res = out["scans"][name] = {"runs": []}
        for r in range(runs):
            t0 = time.time()
            s = read(name)
            read_s = round(time.time() - t0, 2)
            timer = P.Timer(echo=f"{device} {name} run {r}")
            if name == "ds005123_forced":                                  # the correction alone
                with P.exact_float32(device):
                    corr = P.correct(s, timer, device)
                P.release_memory(device)
                res["note"] = corr.note
                res["runs"].append({"read_s": read_s, "seconds": timer.seconds, "applied": corr.applied,
                                    "residual_left": corr.residual_left, "b0s": int(s.b0s.shape[-1]) if s.b0s is not None else 0,
                                    "correct_s": timer.total("field_estimate", "field_apply")})
            else:
                corr, tg, labels = P.run(s, labeler, timer, device=device, workers=workers, shell=SHELL[name])
                res["note"] = corr.note
                res["runs"].append({"read_s": read_s, "seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages()),
                                    "corrected": corr.applied, "grid": list(s.dwi.shape), "seeds": int(tg.stats["seeds"]),
                                    "fibers": len(tg.fibers), "fiber_steps": int(tg.stats["fiber_steps"]),
                                    "labeled": int(labels.keep.sum()), "other_fraction": round(float((labels.tract == 42).mean()), 4),
                                    "compile_s": tg.stats.get("compile_s"),
                                    "fibers_sha": hashlib.sha256(np.concatenate(tg.fibers).astype(np.float32).tobytes()).hexdigest()[:16]})
            print(json.dumps({"scan": name, "run": r, **res["runs"][-1]}), flush=True)
    return out


if __name__ == "__main__":                                                  # the CPU path spawns workers, which re-import this script
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps"); ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=None); ap.add_argument("--only", default=",".join(SCANS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = matrix(args.device, args.runs, args.workers, tuple(args.only.split(",")))
    Path(args.out or HERE / f"results/timing_matrix_{args.device}.json").write_text(json.dumps(res, indent=1))
