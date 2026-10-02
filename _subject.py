"""A ds001226 subject's diffusion inputs, as cohort.py and cohort_topup.py read them: the AP DWI, the
reversed-phase-encoding pair, and the b0 stack the field is estimated from (AP b0s, then PA b0s; a
PA slab placed differently - PAT19, PAT20, PAT23, PAT29: rotated 0.8 deg about the slab's center - put
onto the AP grid by the scanner's coordinates, cubic; topup would ignore the headers and leave the
offset to its motion estimate)."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np, nibabel as nib
from scipy.ndimage import affine_transform

TD = Path.home() / "tmp/data/tractography/ds001226"
VEC = {"i": (1, 0, 0), "i-": (-1, 0, 0), "j": (0, 1, 0), "j-": (0, -1, 0), "k": (0, 0, 1), "k-": (0, 0, -1)}


def load(sub):
    SUB = TD / f"sub-{sub}/ses-preop"
    dw = lambda d, ext: SUB / f"dwi/sub-{sub}_ses-preop_acq-{d}_dwi.{ext}"
    side = {d: json.loads(dw(d, "json").read_text()) for d in ("AP", "PA")}
    pe_ap, pe_pa = (np.array(VEC[side[d]["PhaseEncodingDirection"]], float) for d in ("AP", "PA"))
    assert np.allclose(pe_ap, -pe_pa), f"not a reversed pair: {side['AP']['PhaseEncodingDirection']} / {side['PA']['PhaseEncodingDirection']}"
    PE = int(np.argmax(np.abs(pe_ap)))
    apimg, paimg = nib.load(dw("AP", "nii.gz")), nib.load(dw("PA", "nii.gz"))
    A = apimg.affine
    bval, bvec = np.loadtxt(dw("AP", "bval")), np.loadtxt(dw("AP", "bvec"))
    bval_pa = np.loadtxt(dw("PA", "bval"), ndmin=1)
    raw = np.asarray(apimg.dataobj)                                        # int16
    b0i = np.flatnonzero(bval < 50)
    pa_b0 = np.asarray(paimg.dataobj, dtype=np.float64)[..., bval_pa < 50]
    pa_offset = None
    if not np.allclose(A, paimg.affine, atol=1e-3):
        Mpa = np.linalg.inv(paimg.affine) @ A                             # AP voxel -> PA voxel
        pa_b0 = np.stack([affine_transform(pa_b0[..., v], Mpa[:3, :3], Mpa[:3, 3], order=3, mode="constant", cval=0.0)
                          for v in range(pa_b0.shape[-1])], -1).clip(0, None)
        c = (np.array(apimg.shape[:3]) - 1) / 2
        W = paimg.affine @ np.linalg.inv(A)                               # AP world -> PA world as placed
        R = W[:3, :3] / np.linalg.norm(W[:3, :3], axis=0)
        ctr = A[:3, :3] @ c + A[:3, 3]
        pa_offset = {"center_shift_mm": round(float(np.linalg.norm(W[:3, :3] @ ctr + W[:3, 3] - ctr)), 2),
                     "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2)}
    return SimpleNamespace(
        sub=sub, SUB=SUB, side=side, pe_ap=pe_ap, pe_pa=pe_pa, PE=PE, sign=float(pe_ap[PE]), trt=side["AP"]["TotalReadoutTime"],
        apimg=apimg, A=A, vox=np.asarray(apimg.header.get_zooms()[:3], float), bval=bval, bvec=bvec, raw=raw, b0i=b0i,
        b0s=np.concatenate([raw[..., b0i].astype(np.float64), pa_b0], axis=-1),
        pe_rows=np.array([pe_ap] * len(b0i) + [pe_pa] * pa_b0.shape[-1]), pa_offset=pa_offset)
