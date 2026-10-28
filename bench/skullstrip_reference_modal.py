"""A SynthStrip port (synthstrip-torch 0.1.1) on the bench's mean b0s, on a CUDA GPU (Modal): skullstrip_check.py's
reference.

    uv run --group bench python bench/skullstrip_check.py --b0s        # writes the mean b0s (DATA/skullstrip/)
    modal run bench/skullstrip_reference_modal.py                      # the reference's SDT beside each

The reference is mhalle/synthstrip-torch (a torch port of FreeSurfer's mri_synthstrip, FreeSurfer-licensed), run here as
an oracle only, the way bench/ runs FSL: tractline neither imports nor distributes it. It runs on CUDA because on MPS
(torch 2.14) its network returns a constant field at a tilted head's conformed size and on a 16 GB CPU it is killed
for memory, in runs while developing tractline's (NOTES 2026-10-27). For each <tag>_b0.nii.gz it writes
<tag>_reference_sdt.npy: the signed distance resampled (graded, linear) to the b0's grid, in nibabel's array order. For
cIIIs04 it also writes cIIIs04_reference_conformed.npz: the port's own conformed input (x) and its signed distance on
that grid (sdt), both (X, Y, Z), for the network-alone comparison. The weights are the same file tractline pins
(fetch.WEIGHTS["synthstrip"]), from the local cache.
"""
import sys
from pathlib import Path

import modal

if modal.is_local():                                                  # the modal CLI's own environment: tractline's src
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tractline.data import DATA
    from tractline.fetch import WEIGHTS, weights
    DIR, W, SHA = DATA / "skullstrip", weights("synthstrip"), WEIGHTS["synthstrip"][1]
SRC = Path.home() / "Dropbox/development/synthstrip-torch"          # a checkout of mhalle/synthstrip-torch 0.1.1
image = (modal.Image.debian_slim(python_version="3.12").apt_install("build-essential", "git")
         .pip_install("torch==2.7.1", "numpy", "SimpleITK>=2.3", "nibabel", "cython",
                      "surfa @ git+https://github.com/freesurfer/surfa.git@8aa4a5f60708ea4294f9a8d5c4989168c247ac2a")
         .add_local_dir(SRC / "src/synthstrip_torch", "/root/synthstrip_torch", copy=True)
         .add_local_file(W if modal.is_local() else "/dev/null", "/root/synthstrip.1.pt", copy=True)
         .env({"SYNTHSTRIP_TORCH_MODEL": "/root/synthstrip.1.pt", "PYTHONPATH": "/root"}))
app = modal.App("tractline-skullstrip-reference", image=image)


@app.function(gpu="A10", timeout=1800)
def strip(files: dict, sha: str) -> dict:
    import hashlib, io, tempfile
    import numpy as np, SimpleITK as sitk
    from synthstrip_torch import conform, predict_sdt
    assert hashlib.sha256(open("/root/synthstrip.1.pt", "rb").read()).hexdigest() == sha
    out = {}
    for tag, data in files.items():
        with tempfile.TemporaryDirectory() as d:
            f = f"{d}/{tag}.nii.gz"
            open(f, "wb").write(data)
            sdt_zyx, sdt_img = predict_sdt(f, device="cuda")
            if tag == "cIIIs04":
                b = io.BytesIO()
                np.savez_compressed(b, x=np.asarray(conform(f).data, np.float32), sdt=np.asarray(sdt_zyx).transpose(2, 1, 0))
                out["cIIIs04_conformed"] = b.getvalue()
            ref = sitk.ReadImage(f)
            a = sitk.GetArrayFromImage(sitk.Resample(sdt_img, ref, sitk.Transform(), sitk.sitkLinear, 100.0,
                                                     sitk.sitkFloat32)).transpose(2, 1, 0)
            b = io.BytesIO()
            np.save(b, a)
            out[tag] = b.getvalue()
    return out


@app.local_entrypoint()
def main():
    files = {p.name[:-len("_b0.nii.gz")]: p.read_bytes() for p in sorted(DIR.glob("*_b0.nii.gz"))}
    for tag, data in strip.remote(files, SHA).items():
        name = "cIIIs04_reference_conformed.npz" if tag == "cIIIs04_conformed" else f"{tag}_reference_sdt.npy"
        (DIR / name).write_bytes(data)
    print("wrote", len(files), "references into", DIR)
