"""labelers.rapidparc against RapidParc's own package (v1.0.4), side by side in one Modal container: on
a Modal CPU container (cpu=32; expected bit-identical: same machine, same torch) and on an A10G (CUDA;
TF32 off for both; requested as gpu="A10", Modal placed it on an A10G). Input: PAT16's tractogram from the A10 run (derived/PAT16/PAT16_a10_tf32off.npz, 42,169 fibers),
the streamlines of 40 mm or more as float32.

    modal run bench/modal_rapidparc_check.py [--what cpu,gpu]

Per model (rapidparc, hemiaug) and shuffle seed (0, 42): RapidParc's 1,600-cluster argmax
(return_anatomical_clusters=False, batches of 16 groups) against ours; the Other fraction; times.
RapidParc's package fetches its weights from its GitHub release; ours reads the same files shipped from
DATA/RapidParc. Writes results/rapidparc_check_modal.json.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
PKG = HERE.parent / "src/tractline"
INPUT = "ds001226/derived/PAT16/PAT16_a10_tf32off.npz"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("RapidParc==1.0.4", "torch==2.14.1")
         .env({"PYTHONPATH": "/root/pkg", "TRACTOGRAPHY_DATA": "/data"})
         .add_local_dir(str(DATA / "RapidParc"), remote_path="/data/RapidParc")
         .add_local_file(str(DATA / INPUT), remote_path=f"/data/{INPUT}")
         .add_local_dir(str(PKG), remote_path="/root/pkg/tractline"))
app = modal.App("tractline-rapidparc-check", image=image)


def check(device: str) -> dict:
    import time, numpy as np, torch
    from RapidParc import RapidParc
    from tractline.labelers.base import lengths, MIN_LENGTH_MM
    from tractline.labelers.rapidparc import Labeler, resample
    if device == "cuda":
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    z = np.load(f"/data/{INPUT}"); off, pts = z["offsets"], z["points"]          # each z[...] reads the whole array: once
    fibers = [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    _, _, length = lengths(fibers)
    kept = [np.asarray(f, np.float32) for f, k in zip(fibers, length >= MIN_LENGTH_MM) if k]
    feat = resample(kept)
    out = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "torch": torch.__version__,
           "streamlines": len(kept), "runs": {}}
    for m in ("rapidparc", "hemiaug"):
        lab = Labeler(device, model=m)
        for s in (0, 42):
            t0 = time.time()
            ref = RapidParc(model_name_or_path=m, inputTractogram=kept, eval_batch_size=16, eval_context_size=2000,
                            return_anatomical_clusters=False, device=torch.device(device), seed=s).numpy()
            t_ref = round(time.time() - t0, 1)
            t0 = time.time(); ours = lab.logits(feat, s).argmax(1).numpy(); t_ours = round(time.time() - t0, 1)
            out["runs"][f"{m}, seed {s}"] = {"clusters_identical": bool(np.array_equal(ours, ref)),
                                             "cluster_agreement": round(float((ours == ref).mean()), 5),
                                             "tract_agreement": round(float((lab.lut[ours] == lab.lut[ref]).mean()), 5),
                                             "other_fraction": [round(float((lab.lut[ref] == 42).mean()), 4), round(float((lab.lut[ours] == 42).mean()), 4)],
                                             "seconds_reference_ours": [t_ref, t_ours]}
            print(m, s, json.dumps(out["runs"][f"{m}, seed {s}"]), flush=True)
    return out


@app.function(cpu=32, memory=32768, timeout=1800)
def on_cpu() -> str:
    import torch
    torch.set_num_threads(32)
    return json.dumps(check("cpu"))


@app.function(gpu="A10", memory=32768, timeout=1800)
def on_gpu() -> str:
    return json.dumps(check("cuda"))


@app.local_entrypoint()
def main(what: str = "cpu,gpu"):
    calls = {k: f.spawn() for k, f in (("cpu", on_cpu), ("gpu", on_gpu)) if k in what}
    res = {k: json.loads(c.get()) for k, c in calls.items()}
    (HERE / "results/rapidparc_check_modal.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
