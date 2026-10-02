"""A whole-brain tractography a Modal run left on the Volume (ukf/hardi/variants/<name>.npz), against the
float64 run there (variants/f64.npz), on this machine. Prints the comparison as JSON (_fibercmp.compare).

    python bench/compare_variant.py triton_block_l40s

modal_ukf_triton.py calls this after a full run: the `modal` CLI runs in its own environment, without
numpy, and comparing is CPU work that should not be billed at GPU rates.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from tractline import ukf as U
import _fibercmp as C

DATA = Path.home() / "tmp/data/tractography"
name = sys.argv[1]
V = DATA / "ukf32/variants"
V.mkdir(parents=True, exist_ok=True)
for n in (name, "f64"):
    if not (V / f"{n}.npz").exists() or n == name:                    # the variant is fetched fresh every time
        subprocess.run(["modal", "volume", "get", "--force", "tractography-bench", f"ukf/hardi/variants/{n}.npz", str(V / f"{n}.npz")],
                       check=True, capture_output=True)
load = lambda z: {int(k): z["points"][z["offsets"][i]:z["offsets"][i + 1]].astype(np.float64) for i, k in enumerate(z["seed_index"])}
mine, ref = load(np.load(V / f"{name}.npz")), load(np.load(V / "f64.npz"))
H = DATA / "ukf/hardi"
D = U.load(str(H / "dwi.nhdr"), str(H / "mask.nrrd"))
print(json.dumps(C.compare(mine, ref, D["i2r"], tuple(int(v) for v in D["dim"][::-1]))))
