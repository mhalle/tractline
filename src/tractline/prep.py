"""A DWI (NIfTI array + FSL bval/bvec) into the tracker's input: the DWI, its NRRD header and a mask -
in memory (prepare, for ukf.from_arrays) or as NRRD files (prep, for ukf.load);
pat16_prep.py's rules, for any ds001226 subject.
  - gradients: FSL bvec (image axes; x flipped when the affine's determinant is positive, as dwi.ts
    fromFsl and dcm2niix do) turned into RAS world by the affine's rotation; written with
    space right-anterior-superior and the identity measurement frame.
  - shell: the b0s and one shell (2800, the nearest to the b = 3000 of TractCloud's training
    tractography), as the ORG pipeline tracked one shell.
  - mask: DIPY median_otsu on the mean b0 (median_radius 4, numpass 4), computed by mask.py
    (identical voxel for voxel; torch, CPU or GPU).
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .mask import median_otsu                                          # dipy's, exactly, in torch (median_check.py)


@dataclass
class TrackerInput:
    dwi: np.ndarray              # (X, Y, Z, G') the b0s and the shell, as stored: int16, or float32 after a correction
    header: dict                 # its NRRD header (gradients in RAS, geometry), as pynrrd would read it
    mask: np.ndarray             # (X, Y, Z) bool: median_otsu of the mean b0
    info: dict


def prepare(dwi, affine, bval, bvec, shell=2800.0, device="cpu") -> TrackerInput:
    """dwi (X, Y, Z, G) int or float, affine its voxel -> RAS matrix, bval (G,), bvec (3, G) FSL. In memory:
    ukf.from_arrays(t.dwi, t.header, t.mask) takes it. device: where the mask's median passes
    run (the same mask either way)."""
    keep = (bval < 50) | (np.abs(bval - shell) < 50)
    if not (np.abs(bval - shell) < 50).any() or not (bval < 50).any():
        raise ValueError(f"prepare: no b = {shell:g} shell or no b0 in this scan (b-values: {sorted(set(np.round(bval).astype(int)))})")
    dwi, bval, bvec = dwi[..., keep], bval[keep], bvec[:, keep]
    M = affine[:3, :3]
    spacing = np.linalg.norm(M, axis=0)
    R = M / spacing                                                       # columns: the axes' directions
    g = bvec.copy()
    if np.linalg.det(M) > 0:                                              # FSL's convention
        g[0] *= -1
    g = (R @ g).T                                                         # (G, 3) RAS
    bmax = float(bval.max())
    g = np.where((bval > 50)[:, None], g / np.linalg.norm(g, axis=1, keepdims=True) * np.sqrt(bval / bmax)[:, None], 0.0)
    # integers are stored as int16 (the binary's); outside its range (uint16 scanners) as float32 instead
    floating = not np.issubdtype(dwi.dtype, np.integer) or dwi.max() > 32767 or dwi.min() < -32768
    hdr = {"type": "float" if floating else "short", "dimension": 4, "space": "right-anterior-superior", "sizes": list(dwi.shape),
           "space directions": [M[:, 0].tolist(), M[:, 1].tolist(), M[:, 2].tolist(), [np.nan] * 3],
           "kinds": ["space", "space", "space", "list"], "endian": "little", "encoding": "raw",
           "space origin": affine[:3, 3].tolist(), "measurement frame": np.eye(3).tolist(),
           "modality": "DWMRI", "DWMRI_b-value": f"{bmax:g}"}
    for n, v in enumerate(g):
        hdr[f"DWMRI_gradient_{n:04d}"] = f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}"
    stored = dwi.astype(np.float32 if floating else np.int16)
    b0 = dwi[..., bval < 50].astype(np.float32).mean(-1)
    _, mask = median_otsu(b0, median_radius=4, numpass=4, device=device)
    info = {"shape": list(dwi.shape), "voxel_mm": [round(float(v), 3) for v in spacing], "b0": int((bval < 50).sum()),
            "shell": shell, "directions": int((bval > 50).sum()), "affine_det_positive": bool(np.linalg.det(M) > 0),
            "mask_voxels": int(mask.sum())}
    return TrackerInput(stored, hdr, mask, info)


def prep(dwi, affine, bval, bvec, out: Path, shell=2800.0):
    """prepare(), written to out/{dwi.nhdr, dwi.raw, mask.nrrd} for ukf.load; returns (info, mask)."""
    import nrrd
    out.mkdir(parents=True, exist_ok=True)
    t = prepare(dwi, affine, bval, bvec, shell)
    mask, info, M = t.mask, t.info, affine[:3, :3]
    nrrd.write(str(out / "dwi.nhdr"), t.dwi, t.header, detached_header=True)
    nrrd.write(str(out / "mask.nrrd"), mask.astype(np.uint8),
               {"type": "unsigned char", "dimension": 3, "space": "right-anterior-superior", "sizes": list(mask.shape),
                "space directions": [M[:, 0].tolist(), M[:, 1].tolist(), M[:, 2].tolist()], "kinds": ["space"] * 3,
                "space origin": affine[:3, 3].tolist(), "encoding": "gzip"})
    return info, mask
