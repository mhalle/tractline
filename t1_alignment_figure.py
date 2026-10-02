"""results/t1_alignment.png from t1_alignment.py's fields: topup's displacement of the AP scan (what
the T1 should find in the uncorrected arm), and each arm's residual displacement against the T1,
an axial and a sagittal slice through the tumor's center, the tumor outlined in black (carried onto
each arm's grid by its own rigid fit). The b0 grid's i axis runs to the patient's left: the axial
slice is shown with the patient's right on the image's left (radiological), anterior up; the sagittal
with anterior to the right.

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
tum = {"topup's displacement": F["topup_tumor"], "uncorrected": F["uncorrected_tumor"], "topup": F["topup_tumor"], "ours": F["ours_tumor"]}
i, _, k = np.round(np.argwhere(F["topup_tumor"]).mean(0)).astype(int)
fig, ax = plt.subplots(2, 4, figsize=(12, 6.4), constrained_layout=True)
for c, (title, d) in enumerate(panels):
    tu = list(tum.values())[c]
    for r, (sl, mk, tt) in enumerate(((d[:, :, k], m[:, :, k], tu[:, :, k]), (d[i], m[i], tu[i]))):
        im = ax[r, c].imshow(np.where(mk, sl, np.nan).T, origin="lower", cmap="RdBu_r", vmin=-8, vmax=8, interpolation="nearest")
        ax[r, c].contour(mk.T, levels=[0.5], colors="0.35", linewidths=0.6)
        ax[r, c].contour(tt.T.astype(float), levels=[0.5], colors="k", linewidths=1.2)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    ax[0, c].set_title(title, fontsize=10)
ax[0, 0].set_ylabel(f"axial through the tumor (slice {k})", fontsize=9); ax[1, 0].set_ylabel(f"sagittal through the tumor (slice {i})", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.7, label="displacement along phase encoding (mm)")
fig.suptitle("PAT16: distortion left after a rigid alignment of each scan's mean b0 to the T1 (tumor outlined)", fontsize=11)
fig.savefig(HERE / "results/t1_alignment.png", dpi=130)
