"""The Stanford HARDI brain (2 mm, 150 directions at b = 2000; dipy's stanford_hardi) through the
pipeline's tracking and labeling on one device: the tracker's input (prep.prepare, its own mask),
UKF two-tensor with the ORG settings and the binary's seeds, the default labeler (RapidParc), one draw. No correction: the
scan has no reversed phase-encoding pair. For timing the paths against each other.

    uv run bench/hardi_paths.py [--device mps|cpu] [--runs 2]       # here (the M2)
    modal run bench/modal_hardi_paths.py                                # an L40S and 32 x86 cores

Each run returns its stage times, fibers, fiber steps, labeled streamlines, the Other fraction and a
hash of the fibers; run 0 pays the device's warm-up (and, on CUDA, the Triton kernel's compile when
it is not cached). Writes results/hardi_paths_<device>.json.
"""
import argparse, hashlib, json, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent


def paths(device="mps", runs=2, workers=None, data=None):
    import nibabel as nib, torch
    from tractline import pipeline as P
    from tractline.data import DATA
    H = Path(data) if data else DATA / "ukf/hardi"
    img = nib.load(H / "HARDI150.nii.gz")
    s = SimpleNamespace(affine=img.affine, bval=np.loadtxt(H / "HARDI150.bval"), bvec=np.loadtxt(H / "HARDI150.bvec"))
    dwi = np.asarray(img.dataobj)
    labeler = P.default_labeler(device)
    out = {"device": device, "torch": torch.__version__, "cpu_count": os.cpu_count(), "runs": []}
    if device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
    for r in range(runs):
        timer = P.Timer(echo=f"{device} run {r}")
        with P.exact_float32(device):
            tg = P.track(s, dwi, timer, device=device, workers=workers, shell=2000.0)
            labels = P.label(tg, labeler, timer)
        out["runs"].append({"seconds": timer.seconds, "track_and_label_s": timer.total("prep", "load", "ukf", "label"),
                            "fibers": len(tg.fibers), "fiber_steps": int(tg.stats["fiber_steps"]), "labeled": int(labels.keep.sum()),
                            "other_fraction": round(float((labels.tract == 42).mean()), 4),
                            "fibers_sha": hashlib.sha256(np.concatenate(tg.fibers).astype(np.float32).tobytes()).hexdigest()[:16]})
        print(json.dumps(out["runs"][-1]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    res = paths(args.device, args.runs)
    (HERE / f"results/hardi_paths_{args.device}.json").write_text(json.dumps(res, indent=1))
