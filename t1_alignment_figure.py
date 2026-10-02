"""results/t1_alignment.png from t1_alignment.py's fields: topup's displacement of the AP scan (what
the T1 should find in the uncorrected arm), and each arm's residual displacement against the T1,
one axial slice (the one with the most voxels topup displaces > 3 mm) and the mid-sagittal slice.
Neurological orientation (patient's left on the image's right), anterior up (axial) or right (sagittal).

    DATA/.venv/bin/python bench/tractography/t1_alignment_figure.py
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
F = np.load(Path.home() / "tmp/data/tractography/ds001226/derived/PAT16/t1_alignment_fields.npz")
m = F["mask"]
dt = F["d_topup"] - np.median(F["d_topup"][m])                  # as the residuals: uniform shift to the rigid fit
panels = [("topup's displacement\n(the uncorrected scan's distortion)", dt), ("uncorrected:\nresidual against the T1", F["uncorrected"]),
          ("topup:\nresidual against the T1", F["topup"]), ("ours:\nresidual against the T1", F["ours"])]
k = int(np.argmax((m & (np.abs(F["d_topup"]) > 3)).sum((0, 1))))
i = m.shape[0] // 2
fig, ax = plt.subplots(2, 4, figsize=(12, 6.4), constrained_layout=True)
for c, (title, d) in enumerate(panels):
    for r, (sl, mk) in enumerate(((d[:, :, k], m[:, :, k]), (d[i], m[i]))):
        im = ax[r, c].imshow(np.where(mk, sl, np.nan).T, origin="lower", cmap="RdBu_r", vmin=-8, vmax=8, interpolation="nearest")
        ax[r, c].contour(mk.T, levels=[0.5], colors="0.35", linewidths=0.6)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, c].set_title(title, fontsize=10)
ax[0, 0].set_ylabel(f"axial, slice {k}", fontsize=9); ax[1, 0].set_ylabel("sagittal, midline", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.7, label="displacement along phase encoding (mm)")
fig.suptitle("PAT16: distortion left after a rigid alignment of each scan's mean b0 to the T1", fontsize=11)
fig.savefig(HERE / "results/t1_alignment.png", dpi=130)
