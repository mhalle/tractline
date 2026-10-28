"""Is the corticospinal tract traced at all, before any naming? (Lauren O'Donnell's question 2, 2026-10-28; NOTES
2026-10-28, "Ron's claims checked; the masks with and without CSF; CST traced, whatever its name".)

    uv run --group bench python bench/cst_traced.py        # -> results/cst_traced.json

On the subjects of cst_thalamus_outside.py (PAT16, cIIIs04; corrected, the shell nearest b = 3000), with the default
mask (median_otsu) and SynthStrip's no-CSF mask at border 2: every streamline, named or not, held to the corticospinal
anatomy of ensemble_endpoints.RULES["CST"] (one end in motor cortex, the other in the brainstem, the ventral DC or the
field of view's lower face), with FastSurfer's labels. For those: how many there are, how far down they reach (the
lowest point against the brainstem label's extent: below its middle, and in its lowest third, which is mostly the
medulla), and what RapidParc names them (or under 40 mm: not labeled).
"""
import json, sys
from pathlib import Path

import numpy as np

from tractline import pipeline as P
from tractline.labelers.base import TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_endpoints import Anatomy, consistent, ends_info  # noqa: E402
from cst_thalamus_outside import ARMS, BRAINSTEM, nearest_3000, subjects  # noqa: E402

CST = TRACT_NAMES.index("CST")


def main():
    res = {}
    labeler = P.default_labeler("mps")
    for name, s, to_t1 in subjects():
        anat = Anatomy(name)
        stem = np.argwhere(anat.lab == BRAINSTEM)
        z = (stem @ anat.affine[:3, :3].T + anat.affine[:3, 3])[:, 2]
        zlo, zhi = float(z.min()), float(z.max())
        res[name] = {"brainstem label, z range in the T1 (mm)": [round(zlo, 1), round(zhi, 1)]}
        for key in ("median-otsu", "no-CSF, border 2"):
            mask, replace = ARMS[key]
            original = P.brain_mask
            if replace is not None:
                P.brain_mask = replace
            try:
                corr, tg, labels = P.run(s, labeler, P.Timer(), device="mps", shell=nearest_3000(s.bval), mask=mask)
            finally:
                P.brain_mask = original
            fibers = tg.fibers
            ra, rb, sa, sb, enters = ends_info(fibers, anat, to_t1, s.affine)
            anat_cst, _ = consistent(np.full(len(fibers), CST), ra, rb, sa, sb, enters)
            idx = np.flatnonzero(anat_cst)
            low = np.array([float((fibers[i] @ to_t1[:3, :3].T + to_t1[:3, 3])[:, 2].min()) for i in idx])
            keep = np.asarray(labels.keep)
            name_of = np.full(len(fibers), -1)
            name_of[np.flatnonzero(keep)] = np.asarray(labels.tract)
            names = {}
            for i in idx:
                n = "under 40 mm (not labeled)" if name_of[i] < 0 else TRACT_NAMES[name_of[i]]
                names[n] = names.get(n, 0) + 1
            named_cst = int((name_of == CST).sum())
            r = {"streamlines": len(fibers),
                 "motor cortex to brainstem by FastSurfer's anatomy, whatever the name": int(len(idx)),
                 "of them reaching below the brainstem's middle": int((low < (zlo + zhi) / 2).sum()),
                 "of them reaching its lowest third (mostly medulla)": int((low < zlo + (zhi - zlo) / 3).sum()),
                 "their names (top 8)": dict(sorted(names.items(), key=lambda kv: -kv[1])[:8]),
                 "named CST (all)": named_cst,
                 "named CST that fit the anatomy": int(anat_cst[name_of == CST].sum())}
            res[name][key] = r
            print(name, key, json.dumps(r), flush=True)
    (HERE / "results" / "cst_traced.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
