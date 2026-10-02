"""OpenNeuro ds001226 (BTC_preop, CC0) subjects as the pipeline takes them: the AP DWI, the
reversed-phase-encoding pair, the T1 and the tumor mask.

The b0 stack the field is estimated from is the AP b0s, then the PA b0s. A PA slab placed differently
(PAT19, PAT20, PAT23, PAT29: rotated 0.8 deg about the slab's center) is put onto the AP grid by the
scanner's coordinates first (cubic); FSL topup would ignore the headers and leave the offset to its
motion estimate. PAT03's "PA" series is phase-encoded left-right: not a reversed pair, refused.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import affine_transform
from tractline.data import DATA

ROOT = DATA / "ds001226"
VEC = {"i": (1, 0, 0), "i-": (-1, 0, 0), "j": (0, 1, 0), "j-": (0, -1, 0), "k": (0, 0, 1), "k-": (0, 0, -1)}


@dataclass
class Subject:
    name: str                    # "PAT16"
    dwi: np.ndarray              # (X, Y, Z, G) as stored (int16)
    affine: np.ndarray           # 4x4, voxel -> RAS mm
    vox: np.ndarray              # (3,) mm, as the header states them
    bval: np.ndarray             # (G,)
    bvec: np.ndarray             # (3, G), FSL convention
    pe_axis: int                 # the DWI's phase-encoding axis (0-2)
    pe_sign: float               # +1 / -1: its direction along that axis
    pe_directions: tuple         # the BIDS PhaseEncodingDirection of the AP and the PA series, e.g. ("j-", "j")
    readout_s: float             # total readout time, s (both scans)
    b0s: np.ndarray              # (X, Y, Z, V) float64: the AP b0s, then the PA b0s, on the AP grid
    pe_vectors: np.ndarray       # (V, 3): each b0's phase-encoding vector
    pa_offset: dict | None       # the PA slab's placement against the AP's, when it differs
    dwi_path: Path               # the AP DWI's NIfTI
    t1: Path
    tumor_mask: Path

    @property
    def b0_index(self):
        return np.flatnonzero(self.bval < 50)


def load(name: str) -> Subject:
    d = ROOT / f"sub-{name}/ses-preop"
    dw = lambda acq, ext: d / f"dwi/sub-{name}_ses-preop_acq-{acq}_dwi.{ext}"
    side = {acq: json.loads(dw(acq, "json").read_text()) for acq in ("AP", "PA")}
    pe_ap, pe_pa = (np.array(VEC[side[acq]["PhaseEncodingDirection"]], float) for acq in ("AP", "PA"))
    assert np.allclose(pe_ap, -pe_pa), f"not a reversed pair: {side['AP']['PhaseEncodingDirection']} / {side['PA']['PhaseEncodingDirection']}"
    assert side["AP"]["TotalReadoutTime"] == side["PA"]["TotalReadoutTime"]
    pe_axis = int(np.argmax(np.abs(pe_ap)))
    apimg, paimg = nib.load(dw("AP", "nii.gz")), nib.load(dw("PA", "nii.gz"))
    A = apimg.affine
    bval, bvec = np.loadtxt(dw("AP", "bval")), np.loadtxt(dw("AP", "bvec"))
    dwi = np.asarray(apimg.dataobj)
    pa_b0 = np.asarray(paimg.dataobj, dtype=np.float64)[..., np.loadtxt(dw("PA", "bval"), ndmin=1) < 50]
    pa_offset = None
    if not np.allclose(A, paimg.affine, atol=1e-3):
        Mpa = np.linalg.inv(paimg.affine) @ A                             # AP voxel -> PA voxel
        pa_b0 = np.stack([affine_transform(pa_b0[..., v], Mpa[:3, :3], Mpa[:3, 3], order=3, mode="constant", cval=0.0)
                          for v in range(pa_b0.shape[-1])], -1).clip(0, None)
        W = paimg.affine @ np.linalg.inv(A)                               # AP world -> PA world as placed
        R = W[:3, :3] / np.linalg.norm(W[:3, :3], axis=0)
        ctr = A[:3, :3] @ ((np.array(apimg.shape[:3]) - 1) / 2) + A[:3, 3]
        pa_offset = {"center_shift_mm": round(float(np.linalg.norm(W[:3, :3] @ ctr + W[:3, 3] - ctr)), 2),
                     "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), 2)}
    b0i = np.flatnonzero(bval < 50)
    return Subject(name=name, dwi=dwi, affine=A, vox=np.asarray(apimg.header.get_zooms()[:3], float), bval=bval, bvec=bvec,
                   pe_axis=pe_axis, pe_sign=float(pe_ap[pe_axis]), readout_s=side["AP"]["TotalReadoutTime"],
                   pe_directions=(side["AP"]["PhaseEncodingDirection"], side["PA"]["PhaseEncodingDirection"]),
                   b0s=np.concatenate([dwi[..., b0i].astype(np.float64), pa_b0], axis=-1),
                   pe_vectors=np.array([pe_ap] * len(b0i) + [pe_pa] * pa_b0.shape[-1]), pa_offset=pa_offset,
                   dwi_path=dw("AP", "nii.gz"), t1=d / f"anat/sub-{name}_ses-preop_T1w.nii.gz",
                   tumor_mask=ROOT / f"derivatives/tumor_masks/sub-{name}/anat/sub-{name}_space_T1_label-tumor.nii")
