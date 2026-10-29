"""The brain masks compared on one scan given by path, distortion-corrected as tractline run does it (NOTES 2026-10-28,
"Ron's claims checked; the masks with and without CSF").

    uv run python bench/mask_compare_scan.py <dwi.nii.gz> <tag> [<out folder>]
                                                    # -> <out folder>/mask_compare_<tag>.json, .png (default results/)

A scan that is not public (one shared for a check) keeps its outputs outside the repository: give the out folder.

For each mask - median_otsu (the default), SynthStrip with CSF (border 1), SynthStrip's no-CSF model at borders 2 and 3
(put in at the pipeline's mask step, as cst_thalamus_outside.py does) - the series is loaded (bids.load: its partner from
the sidecars), corrected, tracked on the shell nearest b = 3000 and labeled; recorded: the pairing, whether the field was
applied, the mask's volume, the streamlines and the labeled, Other, and the tracts the mask can cut (CST, CPC, MCP, ICP,
the intra-cerebellar classes, PLIC) beside two it should not (AF, CC4). The figure: the corrected mean b0's mid-sagittal
slice (and one 12 mm to each side) with each mask outlined.
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tractline import bids, pipeline as P
from tractline.labelers.base import TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cst_thalamus_outside import ARMS, _arm, _sdt, nearest_3000  # noqa: E402

WATCH = ["CST", "CPC", "MCP", "ICP", "Intra-CBLM-I-P", "Intra-CBLM-PaT", "PLIC", "AF", "CC4"]
MASKS = {"median-otsu": ARMS["median-otsu"], "synthstrip (border 1)": ARMS["synthstrip (border 1)"],
         "no-CSF, border 2": ARMS["no-CSF, border 2"],
         "no-CSF, border 3": (None, _arm(lambda d, b, a, dev: _sdt(d, b, a, dev, "synthstrip-nocsf")[0] < 3))}
COLORS = {"median-otsu": "r", "synthstrip (border 1)": "cyan", "no-CSF, border 2": "lime", "no-CSF, border 3": "yellow"}


def main(dwi, tag, out=None):
    out = Path(out) if out else HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    s = bids.load(dwi)
    shell = nearest_3000(s.bval)
    labeler = P.default_labeler("mps")
    side = json.loads(Path(str(dwi)[:-7] + ".json").read_text()) if Path(str(dwi)[:-7] + ".json").exists() else {}
    b = np.asarray(s.bval, float)
    shells = {int(round(v, -2)): int(n) for v, n in zip(*np.unique(np.round(b, -2), return_counts=True))}
    res = {"series": Path(dwi).name, "pairing": s.pairing.message, "shell": shell, "shells (volumes)": shells,
           "voxel mm": [round(float(z), 2) for z in np.sqrt((s.affine[:3, :3] ** 2).sum(0))],
           "acquisition": {k: side.get(k) for k in ("Manufacturer", "ManufacturersModelName", "ReceiveCoilName", "ImageType",
                                                    "MultibandAccelerationFactor", "PartialFourier", "PhaseEncodingDirection",
                                                    "TotalReadoutTime") if k in side}}
    masks, b0 = {}, None
    for key, (mask, replace) in MASKS.items():
        original = P.brain_mask
        if replace is not None:
            P.brain_mask = replace
        try:
            corr, tg, labels = P.run(s, labeler, P.Timer(), device="mps", shell=shell, mask=mask)
        finally:
            P.brain_mask = original
        tr = np.asarray(labels.tract)
        vox = float(abs(np.linalg.det(s.affine[:3, :3]))) / 1000
        res[key] = {"field applied": bool(corr.applied), "mask ml": round(float(tg.mask.sum()) * vox), "streamlines": len(tg.fibers),
                    "labeled": int(np.asarray(labels.keep).sum()), "Other": int((tr == TRACT_NAMES.index("Other")).sum()),
                    **{k: int((tr == TRACT_NAMES.index(k)).sum()) for k in WATCH}}
        print(key, json.dumps(res[key]), flush=True)
        masks[key] = tg.mask
        if b0 is None:
            b0 = corr.dwi[..., np.asarray(s.bval) < 50].mean(-1)
    (out / f"mask_compare_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    # the figure, in the scan's voxel axes: x the first, z the third (ok for an axial, roughly LAS/RAS series)
    vz = np.sqrt((s.affine[:3, :3] ** 2).sum(0))
    xm = b0.shape[0] // 2
    off = int(round(12 / vz[0]))
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    for a, x in zip(ax, (xm - off, xm, xm + off)):
        a.imshow(b0[x].T, origin="lower", cmap="gray", vmax=np.quantile(b0, 0.995), aspect=vz[2] / vz[1])
        for k, m in masks.items():
            a.contour(m[x].T, [0.5], colors=COLORS[k], linewidths=1.1)
        a.set_title(f"x = {x}"); a.axis("off")
    ax[0].legend(handles=[plt.Line2D([], [], color=c, label=k) for k, c in COLORS.items()], fontsize=8, loc="upper left")
    fig.suptitle(f"{tag}: corrected mean b0, each mask outlined")
    fig.tight_layout()
    fig.savefig(out / f"mask_compare_{tag}.png", dpi=90)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: mask_compare_scan.py <dwi.nii.gz> <tag> [<out folder>]   (a private scan: give an out folder "
                 "outside the repository)")
    main(*sys.argv[1:4])
