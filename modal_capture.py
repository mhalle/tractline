"""M3 step 1: the full HCP subject (440,621 streamlines), five seeded runs, on Modal GPUs.

    modal run bench/tractography/modal_capture.py                  # capture runs 0-4, then derive
    modal run bench/tractography/modal_capture.py --runs 1,2,3,4   # only these, then derive
    modal run bench/tractography/modal_capture.py --derive-only    # re-derive from the Volume

The same capture as capture.py (upstream RealDataDataset under np.random.seed(SEEDS[r]),
upstream model, the log-softmax kept instead of its argmax), on every streamline instead of
the 100 k targets, with upstream's own CUDA setting (cudnn off, as TractCloudPipeline does).
Each run writes its (N, 1600) fp16 log-softmax to the Volume (1.4 GB each, never downloaded).
derive() then reads the five and writes one small derived.npz to download:

    cluster (R, N) int16                   the argmax of each run
    m_best / m_mass {tract, fold} (R, N)   own-group margins, full field (tract; outliers folded in)
    p_max, cluster_margin (R, N)           the one-byte baselines
    rf_d{6,8}_ranks/_support/_tail         run 0 encoded by rankfield, keep="clip", clip 8
    rf_d{6,8}_meta                         the rankfield meta block (JSON)

and returns the full-subject M1 numbers, so the 100 k subsample can be checked against the
whole brain.

Inputs live on the Modal Volume `tractography-bench` (uploaded once from DATA, nothing
re-downloaded from GitHub):
    modal volume create tractography-bench
    modal volume put tractography-bench DATA/hcp/feat.npy hcp/feat.npy
    modal volume put tractography-bench DATA/TrainedModel TrainedModel
    modal volume put tractography-bench DATA/TrainData_800clu800ol/HCP_mass_center.npy TrainData_800clu800ol/HCP_mass_center.npy
"""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
SEEDS = (0, 1, 2, 3, 4)                          # _data.SEEDS; _data is not imported (it makes dirs)
RANKFIELD = "rankfield[torch] @ git+https://github.com/mhalle/rankfield.git@v0.3.10"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch>=2.7", "numpy>=2", "scikit-learn", RANKFIELD)
    .env({"PYTHONPATH": "/root/pkg"})
    .add_local_dir(str(DATA / "TractCloud/src/tractcloud"), remote_path="/root/pkg/tractcloud")
    .add_local_file(str(HERE / "_groups.py"), remote_path="/root/pkg/_groups.py")
)
vol = modal.Volume.from_name("tractography-bench")
app = modal.App("tractography-capture", image=image)
V = "/vol"


def _upstream():
    """tractcloud.inference imports vtk at module top for extract_ras_features, which the
    workers never call (the features come precomputed). A stub keeps the upstream module
    unchanged without VTK's OpenGL stack in a headless image."""
    import sys, types
    sys.modules.setdefault("vtk", types.ModuleType("vtk"))
    from tractcloud import inference
    return inference


@app.function(gpu="A10G", volumes={V: vol}, timeout=3600, memory=16384)
def capture(r: int, seed: int) -> dict:
    import time
    import numpy as np, torch
    inf = _upstream()
    from torch.utils.data import DataLoader
    dev = torch.device("cuda")
    torch.backends.cudnn.enabled = False                       # as TractCloudPipeline on cuda
    feat = np.load(f"{V}/hcp/feat.npy")
    model, _ = inf.load_model(f"{V}/TrainedModel/best_tract_f1_model.pth", f"{V}/TrainedModel/cli_args.txt",
                              dev, k_override=20, k_global_override=80)
    centered = inf.center_tractography(feat, np.load(f"{V}/TrainData_800clu800ol/HCP_mass_center.npy"))
    t0 = time.time()
    np.random.seed(seed)
    ds = inf.RealDataDataset(centered, k=20, k_global=80, k_ds_rate=0.1)
    t1 = time.time()
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    out = np.empty((len(ds), 1600), np.float16)
    label = np.empty(len(ds), np.int16)
    at = 0
    with torch.no_grad():
        for points, k_local in loader:
            n = points.shape[0]
            k_global_t = torch.from_numpy(ds.global_feat).repeat(n, 1, 1, 1).transpose(2, 1)
            info = torch.cat((k_local.transpose(2, 1), k_global_t), dim=3)
            lp = model(points.transpose(2, 1).to(dev), info.to(dev)).view(-1, 1600)
            label[at:at + n] = lp.argmax(1).cpu().numpy()
            out[at:at + n] = lp.half().cpu().numpy()
            at += n
    t2 = time.time()
    Path(f"{V}/hcp_full").mkdir(exist_ok=True)
    np.save(f"{V}/hcp_full/logp_run{r}.npy", out)
    np.save(f"{V}/hcp_full/label_run{r}.npy", label)
    vol.commit()
    return {"run": r, "seed": seed, "streamlines": len(ds), "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),        # a TorchVersion: unpicklable without torch locally
            "context_s": round(t1 - t0, 1), "inference_s": round(t2 - t1, 1),
            "fp16_argmax_differs": int((out.astype(np.float32).argmax(1) != label).sum())}


@app.function(volumes={V: vol}, cpu=8, memory=49152, timeout=3600)
def derive(runs: int = len(SEEDS)) -> dict:
    import json, time
    import numpy as np, torch
    import rankfield as rf
    from sklearn.metrics import roc_auc_score
    from _groups import GROUPINGS, lse, margins
    vol.reload()
    tract_lut = GROUPINGS["tract (TractCloud's)"][0]
    fold_lut = GROUPINGS["tract, outliers folded in"][0]
    keys = ("cluster", "m_best_tract", "m_mass_tract", "m_best_fold", "m_mass_fold", "p_max", "cluster_margin")
    acc = {k: [] for k in keys}
    enc = {}
    for r in range(runs):
        l16 = np.load(f"{V}/hcp_full/logp_run{r}.npy")
        if r == 0:
            lg = torch.from_numpy(l16).T[:, :, None, None]
            for depth in (6, 8):
                t0 = time.time()
                code = rf.encode(lg, keep="clip", depth=depth, clip=8.0, tail_temperatures=(1.0,))
                enc[f"rf_d{depth}_ranks"] = code.ranks.reshape(depth, -1)
                enc[f"rf_d{depth}_support"] = code.support.reshape(depth - 1, -1)
                enc[f"rf_d{depth}_tail"] = code.tail.reshape(-1)
                enc[f"rf_d{depth}_meta"] = np.array(json.dumps({**code.meta, "encode_s": round(time.time() - t0, 1)}))
            del lg, code
        l = l16.astype(np.float32)
        del l16
        l -= lse(l, 1)[:, None]
        cluster = l.argmax(1)
        top2 = np.partition(l, -2, axis=1)[:, -2:]
        mt, mf = margins(l, tract_lut), margins(l, fold_lut)
        for k, v in (("cluster", cluster.astype(np.int16)), ("m_best_tract", mt["best"]), ("m_mass_tract", mt["mass"]),
                     ("m_best_fold", mf["best"]), ("m_mass_fold", mf["mass"]),
                     ("p_max", np.exp(top2[:, 1])), ("cluster_margin", top2[:, 1] - top2[:, 0])):
            acc[k].append(np.asarray(v, np.float32) if k != "cluster" else v)
        del l, mt, mf, top2
    arr = {k: np.stack(v) for k, v in acc.items()}
    np.savez(f"{V}/hcp_full/derived.npz", **arr, **enc)
    vol.commit()

    cl = arr["cluster"].astype(np.int64)
    auroc = lambda x, t: round(float(roc_auc_score(t, -x)), 4)
    out = {"streamlines": int(cl.shape[1]), "runs": runs}
    for name, lut, best, mass in (("tract", tract_lut, "m_best_tract", "m_mass_tract"),
                                  ("tract, outliers folded in", fold_lut, "m_best_fold", "m_mass_fold")):
        lab = lut[cl]
        y = (lab[1:] != lab[0]).any(0)
        out[name] = {"changed_fraction": round(float(y.mean()), 4),
                     "auroc_run0": {"best-class margin": auroc(arr[best][0], y), "mass margin": auroc(arr[mass][0], y),
                                    "p_max": auroc(arr["p_max"][0], y), "cluster margin": auroc(arr["cluster_margin"][0], y),
                                    "tract mass margin (42, fixed)": auroc(arr["m_mass_tract"][0], y)}}
    out["cluster_changed_fraction"] = round(float((cl[1:] != cl[0]).any(0).mean()), 4)
    out["encode_s"] = {d: json.loads(str(enc[f"rf_d{d}_meta"]))["encode_s"] for d in (6, 8)}
    return out


@app.local_entrypoint()
def main(derive_only: bool = False, runs: str = "0,1,2,3,4"):
    import json
    if not derive_only:
        todo = [int(r) for r in runs.split(",")]
        for res in capture.starmap([(r, SEEDS[r]) for r in todo]):
            print(json.dumps(res), flush=True)
    print(json.dumps(derive.remote(), indent=1))
