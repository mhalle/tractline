"""Peak memory of the motion and eddy-current step, in copies of the series (float32), on a real scan and on an
HCP-sized series built from it (NOTES 2026-10-28, "Motion and eddy correction in less memory").

    uv run python bench/motion_eddy/memory_profile.py <tag> [--hcp]   # -> results/motion_eddy/memory_<tag>.json

The series: ON-Harmony sub-16974 OXF1PRI001 (104 x 104 x 72 x 105, 2 mm), corrected with its reversed pair as
tractline run does it (pipeline.correct_motion_eddy: the field, then the motion and eddy step, one resample). With
--hcp, the same scan resampled to 1.25 mm (cubic) and its volumes repeated to 288 (each b-vector rotated by a small
random angle, so no two are the same): an HCP-sized series, 166 x 166 x 115 x 288, for memory only (its fit is not a
measurement of anything).

Reported (GB are 2**30 bytes): the step's time; the peaks of the memory torch has allocated on MPS and of what Metal's
driver holds, sampled every 5 ms in a thread (so a lower bound: a peak shorter than that can be missed); the peak of
the process's footprint (/usr/bin/time -l, a true peak, which counts the CPU side too: on Apple silicon it is the same
memory); each also in copies of the series; and the digests of the fitted parameters (estimate()'s theta_fit, at full
precision) and of the corrected series, so two versions of the code can be compared for exactly the same result.
"""
import hashlib, json, resource, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import torch

from tractline import bids, motion_eddy as ME, pipeline as P
from tractline.data import DATA

HERE = Path(__file__).resolve().parent
DWI = DATA / "ds004712/sub-16974/ses-OXF1PRI001/dwi/sub-16974_ses-OXF1PRI001_dir-AP_dwi.nii.gz"


def hcp_sized(s):
    """s resampled to 1.25 mm and its volumes repeated to 288, b-vectors jittered (memory only)."""
    from scipy.ndimage import zoom
    f = 2.0 / 1.25
    rng = np.random.default_rng(0)
    vols = np.stack([zoom(np.asarray(s.dwi[..., v], np.float32), f, order=3) for v in range(s.dwi.shape[-1])], -1)
    G = 288
    idx = np.resize(np.arange(vols.shape[-1]), G)
    s.dwi = vols[..., idx]
    bv = s.bvec[:, idx] + rng.normal(0, 0.02, (3, G)) * (s.bval[idx] > 50)
    s.bvec = bv / np.clip(np.linalg.norm(bv, axis=0), 1e-9, None) * (s.bval[idx] > 50)
    s.bval = s.bval[idx]
    A = s.affine.copy()
    A[:3, :3] = A[:3, :3] / f
    s.affine = A
    s.b0s = np.stack([zoom(np.asarray(b, np.float32), f, order=3) for b in np.moveaxis(s.b0s, -1, 0)], -1)
    s.vox = tuple(v / f for v in s.vox)
    return s


def main(tag, hcp=False):
    s = bids.load(DWI)
    if hcp:
        s = hcp_sized(s)
    X, Y, Z, G = s.dwi.shape
    copy_gb = X * Y * Z * G * 4 / 2 ** 30
    peak = {"driver": 0, "allocated": 0}
    fits = []
    estimate = ME.estimate

    def keep(*a, **k):
        fits.append(estimate(*a, **k))
        return fits[-1]

    ME.estimate = keep
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["driver"] = max(peak["driver"], torch.mps.driver_allocated_memory())
            peak["allocated"] = max(peak["allocated"], torch.mps.current_allocated_memory())
            time.sleep(0.005)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t = time.perf_counter()
    corr = P.correct_motion_eddy(s, P.Timer(), "mps", {"check_memory": False})
    dt = time.perf_counter() - t
    stop.set()
    th.join()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30                 # bytes on macOS
    res = {"tag": tag, "shape": [X, Y, Z, G], "one copy GB": round(copy_gb, 2), "seconds": round(dt, 1),
           "allocated peak GB": round(peak["allocated"] / 2 ** 30, 2),
           "allocated peak, copies": round(peak["allocated"] / 2 ** 30 / copy_gb, 2),
           "Metal driver peak GB": round(peak["driver"] / 2 ** 30, 2),
           "Metal driver peak, copies": round(peak["driver"] / 2 ** 30 / copy_gb, 2),
           "process peak resident GB": round(rss, 2),
           "theta_fit sha256": hashlib.sha256(np.ascontiguousarray(fits[0].theta_fit).tobytes()).hexdigest(),
           "corrected sha256": hashlib.sha256(np.ascontiguousarray(corr.dwi).tobytes()).hexdigest()}
    print(json.dumps(res))


def with_footprint(args):
    """main() in a child under /usr/bin/time -l (macOS), its record completed with the peak memory footprint, which
    counts the CPU side and Metal's buffers alike."""
    r = subprocess.run(["/usr/bin/time", "-l", sys.executable, __file__, "--child", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-3000:])
    res = json.loads(r.stdout.strip().splitlines()[-1])
    line = next(x for x in r.stderr.splitlines() if "peak memory footprint" in x)
    res["process peak footprint GB"] = round(int(line.split()[0]) / 2 ** 30, 2)
    res["process peak footprint, copies"] = round(res["process peak footprint GB"] / res["one copy GB"], 2)
    out = HERE.parent / "results" / "motion_eddy" / f"memory_{res['tag']}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    if "--child" in sys.argv:
        args = sys.argv[1:]
        main(next(a for a in args if not a.startswith("--")), "--hcp" in args)
    else:
        with_footprint(sys.argv[1:])
