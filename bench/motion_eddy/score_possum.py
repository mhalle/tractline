"""Corrected DW-POSSUM simulations (Graham, Drobnjak & Zhang 2016; NITRC diffusionsim, public domain) against their
artifact-free ground truth. The release gives no motion parameters, so each corrected volume is registered to the same
volume of the ground truth (same contrast, no noise; mi_torch, rigid plus the quadratic PE polynomial, so residual eddy
distortion counts), and the residual displacement at the mask's edge is scored after the transform common to all
volumes (the output frame) is removed: overall, within each group (b0s, each shell) and between the groups' means;
voxels. Also the image RMSE against the ground truth in the mask, after the frame is removed.

    python score_possum.py <SimulationFiles dir> <case: LargeMovement+EC | SmallMovement+EC> <snr> name=corrected.nii.gz ...
"""
import json, sys
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_erosion
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mi_torch import register

S, CASE, SNR = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sets = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
gt_img = nib.load(str(S / "GroundTruth/ground_truth.nii.gz"))
GT = np.asarray(gt_img.dataobj, np.float32)
vox = np.array(gt_img.header.get_zooms()[:3], float); cen = (np.array(GT.shape[:3]) - 1) / 2
mask = np.asarray(nib.load(str(S / "GroundTruth/mask_mask.nii.gz")).dataobj) > 0
bval = np.loadtxt(S / "GroundTruth/simulation.bval"); G = len(bval)
edge = mask & ~binary_erosion(mask); xe = np.argwhere(edge).astype(float)
groups = [np.flatnonzero(bval < 50)] + [np.flatnonzero((bval >= 50) & (np.abs(bval - b) < 100)) for b in sorted(set(np.round(bval[bval >= 50] / 100) * 100))]
mk = torch.as_tensor(mask, device=dev)
PE, NMON = int(sys.argv[sys.argv.index("--pe") + 1]) if "--pe" in sys.argv else 1, 10


def rot(w):
    cx, cy, cz = np.cos(w); sx, sy, sz = np.sin(w)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def disp(d):                                                                              # mi_torch's map: mov(phi(x)) ~ fixed(x)
    p = (((xe - cen) * vox) @ rot(np.asarray(d[3:6]) * 0.01).T + np.asarray(d[:3])) / vox + cen
    u = (xe - cen) / cen; a, b, c = u.T
    mono = np.stack([np.ones_like(a), a, b, c, a * a, b * b, c * c, a * b, a * c, b * c], -1)
    p[:, PE] += mono @ np.asarray(d[6:])
    return p - xe


def score(d):
    d = d - d.mean(0, keepdims=True)
    within, means = [], []
    for g in groups:
        m = d[g].mean(0, keepdims=True); means.append(m)
        within.append(np.linalg.norm(d[g] - m, axis=-1).ravel())
    mm = np.concatenate(means); n = np.linalg.norm(d, axis=-1)
    return {"edge_error_rms_p95": [round(float(np.sqrt((n ** 2).mean())), 3), round(float(np.quantile(n, 0.95)), 3)],
            "within_shell_rms": round(float(np.sqrt((np.concatenate(within) ** 2).mean())), 3),
            "between_shell_rms": round(float(np.sqrt((np.linalg.norm(mm - mm.mean(0, keepdims=True), axis=-1) ** 2).mean())), 3),
            "worst_volume_rms": round(float(np.sqrt((n ** 2).mean(1)).max()), 3)}


out = {"case": f"{CASE} SNR {SNR}", "volumes": G, "groups": [len(g) for g in groups]}
sets = {"as acquired": str(S / f"{CASE}/data_SNR_{SNR}.nii.gz"), **sets}
for name, f in sets.items():
    C = np.asarray(nib.load(f).dataobj, np.float32)
    ds = []
    for k in range(G):
        d = register(torch.as_tensor(C[..., k], device=dev), torch.as_tensor(GT[..., k], device=dev), mk, vox, pe_axis=PE, nmon=NMON).cpu().double().numpy()
        ds.append(disp(d))
    out[name] = score(np.stack(ds))
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(Path(__file__).resolve().parent.parent / f"results/motion_eddy/possum_{CASE.split('+')[0]}_snr{SNR}.json", "w"), indent=1)
