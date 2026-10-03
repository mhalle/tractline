"""labelers.rapidparc against RapidParc's own package (v1.0.4, in its own environment: DATA/rapidparc-ref,
torch 2.14.1 as ours), PAT16's Metal tractogram (the M2's field), the streamlines of 40 mm or more as
float32.

Per model (rapidparc, hemiaug) and shuffle seed (0, 42): RapidParc's 1,600-cluster argmax
(return_anatomical_clusters=False) on the CPU, batches of 16 groups, against ours on the CPU (expected
identical) and on the M2's GPU (agreement); the tract labels too.

    python bench/rapidparc_check.py

Writes results/rapidparc_check.json.
"""
import json, subprocess, time
from pathlib import Path
import numpy as np, torch
from tractline import pipeline as P, ukf as U
from tractline.prep import prepare
from tractline.data import DATA
from tractline.labelers.base import lengths, MIN_LENGTH_MM
from tractline.labelers.rapidparc import Labeler, resample
from _ds001226 import load, ROOT

HERE = Path(__file__).resolve().parent
REF = DATA / "rapidparc-ref/bin/python"
INPUT = ROOT / "derived/PAT16/rapidparc_input.npz"
RUNS = [(m, s) for m in ("rapidparc", "hemiaug") for s in (0, 42)]

REF_SCRIPT = """
import json, sys, numpy as np, torch
from RapidParc import RapidParc
z = np.load(sys.argv[1]); off, pts = z["offsets"], z["points"]
fibers = [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)]
out = {}
for m, s in json.loads(sys.argv[2]):
    y = RapidParc(model_name_or_path=m, inputTractogram=fibers, eval_batch_size=16, eval_context_size=2000,
                  return_anatomical_clusters=False, device=torch.device("cpu"), seed=s)
    np.save(sys.argv[3] + f"_{m}_{s}.npy", y.numpy())
    out[f"{m}_{s}"] = int(len(y))
print(json.dumps(out))
"""

if __name__ == "__main__":
    s = load("PAT16"); timer = P.Timer()
    corr = P.correct(s, timer, device="mps")
    ti = prepare(corr.dwi, s.affine, s.bval, s.bvec, device="mps")
    fibers, _ = U.track(U.from_arrays(ti.dwi, ti.header, ti.mask), backend="metal")
    _, _, length = lengths(fibers)
    kept = [np.asarray(f, np.float32) for f, k in zip(fibers, length >= MIN_LENGTH_MM) if k]
    np.savez(INPUT, points=np.concatenate(kept), offsets=np.r_[0, np.cumsum([len(f) for f in kept])])
    t0 = time.time()
    r = subprocess.run([str(REF), "-c", REF_SCRIPT, str(INPUT), json.dumps(RUNS), str(INPUT.with_suffix(""))],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    res = {"subject": "PAT16", "streamlines": len(kept), "reference_seconds": round(time.time() - t0, 1), "runs": {}}
    feat = resample(kept)
    labs = {(m, d): Labeler(d, model=m) for m in ("rapidparc", "hemiaug") for d in ("cpu", "mps")}
    for m, sd in RUNS:
        ref = np.load(f"{INPUT.with_suffix('')}_{m}_{sd}.npy")
        row = {}
        for dev in ("cpu", "mps"):
            lab = labs[(m, dev)]
            t0 = time.time(); ours = lab.logits(feat, sd).argmax(1).numpy(); secs = round(time.time() - t0, 1)
            row[dev] = {"clusters_identical": bool(np.array_equal(ours, ref)), "cluster_agreement": round(float((ours == ref).mean()), 5),
                        "tract_agreement": round(float((lab.lut[ours] == lab.lut[ref]).mean()), 5), "seconds": secs}
        row["other_fraction"] = round(float((labs[(m, "cpu")].lut[ref] == 42).mean()), 4)
        res["runs"][f"{m}, seed {sd}"] = row
        print(m, sd, json.dumps(row), flush=True)
    (HERE / "results/rapidparc_check.json").write_text(json.dumps(res, indent=1))
