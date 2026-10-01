"""Stage timings for the full HCP subject on one Modal GPU: context, TractCloud inference, and
the field encode where the logits already are (CUDA) and on the container's CPU.

    modal run bench/tractography/modal_timing.py

Same inputs and capture as modal_capture.py (the tractography-bench Volume), but nothing is
written back. rankfield is the WORKING TREE of the rankfield checkout beside this one (the
encode optimization of 2026-10-01, rankfield 3efef7f), mounted over the pinned release, so the
encode timed here is the one about to ship; the pinned v0.3.10 is timed beside it on the CPU.
"""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
RANKFIELD_SRC = Path.home() / "Dropbox/development/rankfield/.claude/worktrees/rankfield-tractography-e892a6/src/rankfield"
RANKFIELD_PIN = "rankfield[torch] @ git+https://github.com/mhalle/rankfield.git@v0.3.10"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch>=2.7", "numpy>=2", RANKFIELD_PIN)
    .env({"PYTHONPATH": "/root/pkg"})
    .add_local_dir(str(DATA / "TractCloud/src/tractcloud"), remote_path="/root/pkg/tractcloud")
    .add_local_dir(str(RANKFIELD_SRC), remote_path="/root/new/rankfield")
)
vol = modal.Volume.from_name("tractography-bench")
app = modal.App("tractography-timing", image=image)
V = "/vol"


@app.function(gpu="A10G", volumes={V: vol}, timeout=3600, memory=32768, cpu=8)
def timing(seed: int = 0) -> dict:
    import importlib, sys, time, types
    import numpy as np, torch
    sys.modules.setdefault("vtk", types.ModuleType("vtk"))          # see modal_capture._upstream
    from tractcloud import inference as inf
    from torch.utils.data import DataLoader
    t = {}
    dev = torch.device("cuda")
    torch.backends.cudnn.enabled = False
    a = time.perf_counter()
    feat = np.load(f"{V}/hcp/feat.npy")
    model, _ = inf.load_model(f"{V}/TrainedModel/best_tract_f1_model.pth", f"{V}/TrainedModel/cli_args.txt",
                              dev, k_override=20, k_global_override=80)
    centered = inf.center_tractography(feat, np.load(f"{V}/TrainData_800clu800ol/HCP_mass_center.npy"))
    t["load model and features"] = time.perf_counter() - a
    a = time.perf_counter()
    np.random.seed(seed)
    ds = inf.RealDataDataset(centered, k=20, k_global=80, k_ds_rate=0.1)
    t["context (kNN + global sample)"] = time.perf_counter() - a
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    out = torch.empty((len(ds), 1600), dtype=torch.float16, device=dev)  # the logits stay on the GPU
    at = 0
    torch.cuda.synchronize(); a = time.perf_counter()
    with torch.no_grad():
        for points, k_local in loader:
            n = points.shape[0]
            k_global_t = torch.from_numpy(ds.global_feat).repeat(n, 1, 1, 1).transpose(2, 1)
            info = torch.cat((k_local.transpose(2, 1), k_global_t), dim=3)
            out[at:at + n] = model(points.transpose(2, 1).to(dev), info.to(dev)).view(-1, 1600).half()
            at += n
    torch.cuda.synchronize()
    t["inference (A10G)"] = time.perf_counter() - a
    del model, ds, loader

    sys.path.insert(0, "/root/new")                                   # the working tree, over the pin
    import rankfield as new
    assert new.__file__.startswith("/root/new"), new.__file__
    lg = out.T[:, :, None, None]
    for name, x, budget in (("field encode, new, CUDA", lg, 8 << 30), ("field encode, new, CPU x8", lg.cpu(), 4 << 30)):
        best = None
        for _ in range(2):
            if x.is_cuda: torch.cuda.synchronize()
            a = time.perf_counter()
            code = new.encode(x, keep="clip", depth=6, clip=8.0, memory_budget=budget)
            if x.is_cuda: torch.cuda.synchronize()
            best = min(best or 1e9, time.perf_counter() - a)
        t[name] = best
        planes = {n: np.asarray(getattr(code, n)) for n in ("ranks", "support", "tail")}
        if "CUDA" in name:
            gpu_planes = planes
    same = all(np.array_equal(gpu_planes[n], planes[n]) for n in ("ranks", "support"))
    for m in [k for k in list(sys.modules) if k == "rankfield" or k.startswith("rankfield.")]:
        del sys.modules[m]
    sys.path.remove("/root/new")
    import rankfield as old
    assert "/root/new" not in old.__file__
    a = time.perf_counter(); old.encode(lg.cpu(), keep="clip", depth=6, clip=8.0, memory_budget=4 << 30)
    t["field encode, v0.3.10, CPU x8"] = time.perf_counter() - a
    return {"streamlines": int(out.shape[0]), "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
            "cpu_threads": torch.get_num_threads(), "seconds": {k: round(v, 2) for k, v in t.items()},
            "cuda_and_cpu_ranks_support_identical": bool(same)}


@app.local_entrypoint()
def main():
    import json
    r = timing.remote()
    print(json.dumps(r, indent=1))
    (HERE / "results" / "timing_gpu.json").write_text(json.dumps(r, indent=1))
