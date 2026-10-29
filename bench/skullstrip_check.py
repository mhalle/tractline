"""tractline.skullstrip against a SynthStrip port (synthstrip-torch 0.1.1) and against FastSurfer (NOTES 2026-10-27, "SynthStrip on the b0").

    uv run python bench/skullstrip_check.py --b0s        # the mean b0s, into DATA/skullstrip/
    modal run bench/skullstrip_reference_modal.py                      # the reference's SDT for each
    uv run python bench/skullstrip_check.py              # -> results/skullstrip_check.json

Scans: MASiVar cIIIs04 (session s1Bx1, scan 1, the b1000 series: a Philips field of view that ends at the pons) and
ON-Harmony sub-16974 and sub-03286 on both of their coils (Siemens Prisma, Head_32 and HeadNeck_64; tilted about 24
degrees). Reported:
  - against the reference (synthstrip-torch 0.1.1 with the released weights), per scan: Dice of the masks (SDT < 1
    mm), and |SDT difference| (max, 99th percentile)
    where both are inside the network's +-5 mm band;
  - the network alone, on the reference's own conformed input (cIIIs04_reference_conformed.npz, written by the Modal
    script):
    max |difference| (the architecture; settles nearest upsampling);
  - cIIIs04 against FastSurfer's labels mapped to the DWI (ensemble_endpoints.Anatomy, its T1-to-DWI rigid fit): Dice
    with FastSurfer's brain, coverage of brainstem / cerebellum / cortex, ml outside and missed, for the reference,
    tractline's at its margin (skullstrip.MARGIN) and at no margin, and median_otsu (today's and a lighter filter);
  - time and peak memory per device (a subprocess each; on macOS the peak footprint, which counts Metal's buffers);
  - one full-resolution layer timed both ways (torch's conv3d, and _Conv's three 2D convolutions).
"""
import argparse, json, resource, subprocess, sys, time
from pathlib import Path

import numpy as np
import nibabel as nib

from tractline import skullstrip as K
from tractline.data import DATA
from tractline.fetch import weights
from tractline.mask import median_otsu

HERE = Path(__file__).resolve().parent
DIR = DATA / "skullstrip"
ONH = DATA / "ds004712"
SCANS = {"cIIIs04": None,
         **{f"{s}_{ses}": ONH / f"sub-{s}/ses-{ses}/dwi/sub-{s}_ses-{ses}_dir-AP_dwi.nii.gz"
            for s in ("16974", "03286") for ses in ("OXF1PRI001", "OXF2PRI001")}}


def write_b0s():
    """Each scan's mean b0 as NIfTI (ON-Harmony's in RAS axes, cIIIs04's as stored), the reference tool's input."""
    sys.path.insert(0, str(HERE))
    from _masivar import scan_files
    DIR.mkdir(parents=True, exist_ok=True)
    for tag, p in SCANS.items():
        if p is None:
            p = scan_files("cIIIs04", "s1Bx1", 1, None)["APP"][0][1]
        img = nib.load(p)
        if tag != "cIIIs04":
            img = nib.as_closest_canonical(img)
        bval = np.loadtxt(str(p)[:-7] + ".bval")
        b0 = np.asarray(img.dataobj, np.float32)[..., bval < 50].mean(-1)
        nib.save(nib.Nifti1Image(b0, img.affine), DIR / f"{tag}_b0.nii.gz")
        print(tag, b0.shape)


def timed(tag, device):
    """In this process: one cold run's seconds and the peak resident memory (MB; on macOS it leaves out Metal's
    buffers, which the caller's peak memory footprint counts)."""
    im = nib.load(DIR / f"{tag}_b0.nii.gz")
    t = time.perf_counter()
    K.signed_distance(im.get_fdata().astype(np.float32), im.affine, weights("synthstrip"), device)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 if sys.platform != "darwin" else 1024) / 1024
    print(json.dumps({"seconds": round(time.perf_counter() - t, 1), "peak resident MB": round(peak)}))


def run_timed(tag, device) -> dict:
    """timed() in its own process; on macOS under /usr/bin/time -l, whose "peak memory footprint" includes Metal's."""
    cmd = [sys.executable, __file__, "--timed", tag, device]
    darwin = sys.platform == "darwin"
    r = subprocess.run((["/usr/bin/time", "-l"] if darwin else []) + cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"timed run on {device} failed:\n{r.stderr[-3000:]}")
    out = json.loads(r.stdout.strip().splitlines()[-1])
    if darwin:
        line = next(x for x in r.stderr.splitlines() if "peak memory footprint" in x)
        out["peak memory footprint MB"] = round(int(line.split()[0]) / 2 ** 20)
    return out


def conv_speed(device) -> dict:
    """A 32-slice slab of the 48 -> 16 channel full-resolution layer (256 x 192 x 32): torch's conv3d against _Conv's
    2D path, ms."""
    import torch
    torch.manual_seed(0)
    conv = K._Conv(48, 16, act=False).to(device)
    x = torch.randn(1, 48, 256, 192, 32, device=device)
    sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(device, lambda: None)
    out = {}
    with torch.inference_mode():
        for name, f in (("conv3d", lambda: conv.conv(x)), ("three conv2d", lambda: conv._as_2d(x))):
            f(); sync()
            t = time.perf_counter()
            for _ in range(3):
                f()
            sync()
            out[name] = round((time.perf_counter() - t) / 3 * 1000)
        out["max |difference|"] = float((conv.conv(x) - conv._as_2d(x)).abs().max())
    return out


def dice(a, b):
    return round(float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1)), 4)


def fastsurfer(sdts, b0, affine):
    sys.path.insert(0, str(HERE))
    from ensemble_endpoints import Anatomy, FS
    anat = Anatomy("cIIIs04")
    to_t1 = np.linalg.inv(np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"])
    ijk = np.indices(b0.shape).reshape(3, -1).T
    lab = anat.raw((ijk @ affine[:3, :3].T + affine[:3, 3]) @ to_t1[:3, :3].T + to_t1[:3, 3]).reshape(b0.shape)
    brain = lab > 0
    vox = float(abs(np.linalg.det(affine[:3, :3]))) / 1000
    regions = {"brainstem": np.isin(lab, [16]), "cerebellum": np.isin(lab, [7, 8, 46, 47]), "cortex": lab >= 1000}
    out = {}
    for k, m in sdts.items():
        out[k] = {"dice with FastSurfer's brain": dice(m, brain),
                  **{f"{r} covered": round(float(m[v].mean()), 3) for r, v in regions.items()},
                  "ml outside the brain": round(float((m & ~brain).sum() * vox)),
                  "ml of the brain missed": round(float((brain & ~m).sum() * vox))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0s", action="store_true")
    ap.add_argument("--timed", nargs=2)
    ap.add_argument("--conformed", default=str(DIR / "cIIIs04_reference_conformed.npz"),
                    help="the reference's conformed input (x) and its SDT (sdt), .npz")
    a = ap.parse_args()
    if a.b0s:
        return write_b0s()
    if a.timed:
        return timed(*a.timed)
    w = weights("synthstrip")
    res = {"margin mm": K.MARGIN, "scans": {}}
    for tag in SCANS:
        im = nib.load(DIR / f"{tag}_b0.nii.gz")
        b0 = im.get_fdata().astype(np.float32)
        ref = np.load(DIR / f"{tag}_reference_sdt.npy")
        ours = K.signed_distance(b0, im.affine, w, "mps" if K.torch.backends.mps.is_available() else "cpu")
        band = (np.abs(ref) < 5) & (np.abs(ours) < 5)
        d = np.abs(ours - ref)[band]
        res["scans"][tag] = {"dice with the reference": dice(ours < 1, ref < 1),
                             "|SDT difference| in the band, max / p99 (mm)": [round(float(d.max()), 3),
                                                                              round(float(np.percentile(d, 99)), 3)],
                             "mask ml (ours / reference)": [round(float((ours < 1).sum() * abs(np.linalg.det(im.affine[:3, :3])) / 1000)),
                                                           round(float((ref < 1).sum() * abs(np.linalg.det(im.affine[:3, :3])) / 1000))]}
        if tag == "cIIIs04":
            sdts = {"reference (synthstrip-torch 0.1.1)": ref < 1, f"tractline (margin {K.MARGIN:g} mm)": ours < 1,
                    "tractline (margin 0)": K.signed_distance(b0, im.affine, w, "cpu", margin=0.0) < 1,
                    "median_otsu (radius 4, 4 passes: today's)": median_otsu(b0, 4, 4)[1],
                    "median_otsu (radius 2, 1 pass: a lighter filter)": median_otsu(b0, 2, 1)[1]}
            res["cIIIs04 against FastSurfer"] = fastsurfer(sdts, b0, im.affine)
    if not Path(a.conformed).is_file():
        raise SystemExit(f"{a.conformed} is missing: run bench/skullstrip_reference_modal.py first")
    z = np.load(a.conformed)
    net = K.load(w, "cpu")
    with K.torch.inference_mode():
        y = net(K.torch.as_tensor(z["x"], dtype=K.torch.float32)[None, None])[0, 0].numpy()
    res["network on the reference's conformed input (CPU): max |difference| (mm)"] = float(np.abs(y - z["sdt"]).max())
    devices = ["cpu"] + (["mps"] if K.torch.backends.mps.is_available() else []) + (["cuda"] if K.torch.cuda.is_available() else [])
    res["time and peak memory (16974 OXF1, cold, one process each)"] = {dev: run_timed("16974_OXF1PRI001", dev)
                                                                         for dev in devices}
    res["a 32-slice slab of one full-resolution layer, ms (conv3d vs the 2D path)"] = {dev: conv_speed(dev)
                                                                                      for dev in devices}
    out = HERE / "results" / "skullstrip_check.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
