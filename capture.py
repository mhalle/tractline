"""M0 step 1: capture TractCloud's full 1600-class log-softmax, R seeded runs, on the targets.

    DATA/.venv/bin/python bench/tractography/capture.py              # runs 0..4
    DATA/.venv/bin/python bench/tractography/capture.py --runs 0 --verify 10240

The context is TractCloud's own, built over the WHOLE brain by upstream RealDataDataset
unchanged, under np.random.seed(SEEDS[r]): the k_global = 80 streamlines drawn from the whole
brain, and each streamline's k = 20 neighbors searched in a 10 % draw from ITS OWN ~10 k chunk
of the file (file order - not from the whole brain). Both draws come from numpy's global
stream, in that order. The model then runs on the target streamlines only, so a target's label
is exactly what a full-brain TractCloud run with that seed gives it.

The one change from upstream is run_inference keeping the log-softmax instead of its argmax.
--verify V runs upstream run_inference on the first V targets of run 0 (a multiple of the batch
size, so the batches are identical) and requires the same labels.

Writes per run, in DATA/hcp/: logp_run{r}.npy ((T, 1600) fp16), label_run{r}.npy (argmax of
the fp32 output, int16), capture_run{r}.json (timings, the fp16-vs-fp32 argmax agreement).
"""
import argparse, json, platform, time
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from _data import HCP, MASS_CENTER, MODEL, SEEDS
from tractcloud.inference import RealDataDataset, center_tractography, load_model, run_inference

K, K_GLOBAL, K_DS_RATE, CLASSES = 20, 80, 0.1, 1600      # TractCloudPipeline's defaults

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=",".join(str(r) for r in range(len(SEEDS))))
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--verify", type=int, default=0, help="check the first V targets of the first run against upstream")
args = ap.parse_args()
runs = [int(r) for r in args.runs.split(",")]
assert args.verify % args.batch == 0, "--verify must be a multiple of --batch so the batches match"

device = torch.device("cpu")
feat = np.load(HCP / "feat.npy")
targets = np.load(HCP / "targets.npy")
model, _ = load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), device,
                      k_override=K, k_global_override=K_GLOBAL)
centered = center_tractography(feat, np.load(MASS_CENTER))


def capture(loader, global_feat):
    """Upstream run_inference, keeping the (T, 1600) log-softmax as float32."""
    out = np.empty((len(loader.dataset), CLASSES), np.float32)
    at = 0
    with torch.no_grad():
        for points, k_local in loader:
            n = points.shape[0]
            points = points.transpose(2, 1)
            k_local = k_local.transpose(2, 1)
            k_global_t = torch.from_numpy(global_feat).repeat(n, 1, 1, 1).transpose(2, 1)
            info = torch.cat((k_local, k_global_t), dim=3)
            out[at:at + n] = model(points.to(device), info.to(device)).view(-1, CLASSES).cpu().numpy()
            at += n
    return out


for i, r in enumerate(runs):
    seed = SEEDS[r]
    t0 = time.time()
    np.random.seed(seed)
    ds = RealDataDataset(centered, k=K, k_global=K_GLOBAL, k_ds_rate=K_DS_RATE)
    t1 = time.time()
    loader = DataLoader(Subset(ds, targets), batch_size=args.batch, shuffle=False)
    logp = capture(loader, ds.global_feat)
    t2 = time.time()
    label = logp.argmax(1)
    logp16 = logp.astype(np.float16)
    label16 = logp16.argmax(1)                       # first maximal index, rankfield's tie rule too
    np.save(HCP / f"logp_run{r}.npy", logp16)
    np.save(HCP / f"label_run{r}.npy", label.astype(np.int16))
    info = {"run": r, "seed": seed, "targets": len(targets), "device": str(device),
            "torch": torch.__version__, "machine": platform.machine(), "threads": torch.get_num_threads(),
            "context_s": round(t1 - t0, 1), "inference_s": round(t2 - t1, 1),
            "fp16_argmax_differs": int((label16 != label).sum()),
            "logp_min": float(logp.min()), "max_logp_mean": float(logp.max(1).mean())}
    if args.verify and i == 0:
        sub = DataLoader(Subset(ds, targets[:args.verify]), batch_size=args.batch, shuffle=False)
        upstream = np.asarray(run_inference(model, sub, ds.global_feat, CLASSES, device))
        info["verify"] = {"n": args.verify, "upstream_differs": int((upstream != label[:args.verify]).sum())}
    (HCP / f"capture_run{r}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    del ds, loader, logp, logp16
