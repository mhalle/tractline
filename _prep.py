"""A DWI (NIfTI array + FSL bval/bvec) into the tracker's input: a NRRD DWI and a NRRD mask
(pat16_prep.py's rules, for any ds001226 subject).
  - gradients: FSL bvec (image axes; x flipped when the affine's determinant is positive, as dwi.ts
    fromFsl and dcm2niix do) turned into RAS world by the affine's rotation; written with
    space right-anterior-superior and the identity measurement frame.
  - shell: the b0s and one shell (2800, the nearest to the b = 3000 of TractCloud's training
    tractography), as the ORG pipeline tracked one shell.
  - mask: DIPY median_otsu on the mean b0 (median_radius 4, numpass 4), computed by _median.py
    (identical voxel for voxel, ~100x faster).
"""
from pathlib import Path
import numpy as np, nrrd
from _median import median_otsu                                          # dipy's, exactly (median_check.py), 0.1 s instead of 10


def prep(data, A, bval, bvec, out: Path, shell=2800.0):
    """data (i, j, k, G) int or float, A its affine, bval (G,), bvec (3, G) FSL. Writes out/{dwi.nhdr,
    dwi.raw, mask.nrrd}; returns (info, mask)."""
    out.mkdir(parents=True, exist_ok=True)
    keep = (bval < 50) | (np.abs(bval - shell) < 50)
    data, bval, bvec = data[..., keep], bval[keep], bvec[:, keep]
    M = A[:3, :3]
    spacing = np.linalg.norm(M, axis=0)
    R = M / spacing                                                       # columns: the axes' directions
    g = bvec.copy()
    if np.linalg.det(M) > 0:                                              # FSL's convention
        g[0] *= -1
    g = (R @ g).T                                                         # (G, 3) RAS
    bmax = float(bval.max())
    g = np.where((bval > 50)[:, None], g / np.linalg.norm(g, axis=1, keepdims=True) * np.sqrt(bval / bmax)[:, None], 0.0)
    floating = not np.issubdtype(data.dtype, np.integer)
    hdr = {"type": "float" if floating else "short", "dimension": 4, "space": "right-anterior-superior", "sizes": list(data.shape),
           "space directions": [M[:, 0].tolist(), M[:, 1].tolist(), M[:, 2].tolist(), [np.nan] * 3],
           "kinds": ["space", "space", "space", "list"], "endian": "little", "encoding": "raw",
           "space origin": A[:3, 3].tolist(), "measurement frame": np.eye(3).tolist(),
           "modality": "DWMRI", "DWMRI_b-value": f"{bmax:g}"}
    for n, v in enumerate(g):
        hdr[f"DWMRI_gradient_{n:04d}"] = f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}"
    nrrd.write(str(out / "dwi.nhdr"), data.astype(np.float32 if floating else np.int16), hdr, detached_header=True)
    b0 = data[..., bval < 50].astype(np.float32).mean(-1)
    _, mask = median_otsu(b0, median_radius=4, numpass=4)
    nrrd.write(str(out / "mask.nrrd"), mask.astype(np.uint8),
               {"type": "unsigned char", "dimension": 3, "space": "right-anterior-superior", "sizes": list(mask.shape),
                "space directions": [M[:, 0].tolist(), M[:, 1].tolist(), M[:, 2].tolist()], "kinds": ["space"] * 3,
                "space origin": A[:3, 3].tolist(), "encoding": "gzip"})
    info = {"shape": list(data.shape), "voxel_mm": [round(float(v), 3) for v in spacing], "b0": int((bval < 50).sum()),
            "shell": shell, "directions": int((bval > 50).sum()), "affine_det_positive": bool(np.linalg.det(M) > 0),
            "mask_voxels": int(mask.sum())}
    return info, mask
