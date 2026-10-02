"""TractCloud inference, optimized step by step, on one A10G: what each change buys, and what it
does to the labels.

    modal run bench/modal_infer_opt.py

The full HCP subject (440,621 streamlines), one context (np.random.seed(0), upstream
RealDataDataset), built once and shared by every variant, so only the inference differs. Each
variant is timed (best of 2, synchronized) and compared with the upstream variant A: cluster and
tract labels that differ, and the largest log-probability difference.

    A  upstream: fp32, cudnn off, batch 1024, the context assembled on the CPU per batch
    B  the context resident on the GPU, assembled there (no per-batch host-to-device copy)
    C  B + cudnn on (upstream turns it off on CUDA)
    D  C + TF32 matmul and convolution
    E  C + autocast fp16
    F  C + autocast bf16
    G  E with batch 4096
    H  G + torch.compile
Also: the kNN context built on the GPU from the same random draws, against the CPU's.
"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch>=2.7", "numpy>=2")
         .env({"PYTHONPATH": "/root/pkg"})
         .add_local_dir(str(DATA / "TractCloud/src/tractcloud"), remote_path="/root/pkg/tractcloud"))
vol = modal.Volume.from_name("tractography-bench")
app = modal.App("tractography-infer-opt", image=image)
V = "/vol"


@app.function(gpu="A10G", volumes={V: vol}, timeout=5400, memory=49152, cpu=8)
def bench() -> dict:
    import contextlib, sys, time, types
    import numpy as np, torch
    sys.modules.setdefault("vtk", types.ModuleType("vtk"))
    from tractcloud import inference as inf
    from torch.utils.data import DataLoader
    dev = torch.device("cuda")
    feat = np.load(f"{V}/hcp/feat.npy")
    centered = inf.center_tractography(feat, np.load(f"{V}/TrainData_800clu800ol/HCP_mass_center.npy"))

    def model_fresh():
        m, _ = inf.load_model(f"{V}/TrainedModel/best_tract_f1_model.pth", f"{V}/TrainedModel/cli_args.txt",
                              dev, k_override=20, k_global_override=80)
        return m

    t0 = time.perf_counter()
    np.random.seed(0)
    ds = inf.RealDataDataset(centered, k=20, k_global=80, k_ds_rate=0.1)
    ctx_cpu_s = time.perf_counter() - t0
    N = len(ds)

    def sync():
        torch.cuda.synchronize()

    def run_upstream(model, batch=1024):
        out = torch.empty((N, 1600), dtype=torch.float32, device=dev)
        loader = DataLoader(ds, batch_size=batch, shuffle=False)
        at = 0
        with torch.no_grad():
            for points, k_local in loader:
                n = points.shape[0]
                g = torch.from_numpy(ds.global_feat).repeat(n, 1, 1, 1).transpose(2, 1)
                info = torch.cat((k_local.transpose(2, 1), g), dim=3)
                out[at:at + n] = model(points.transpose(2, 1).to(dev), info.to(dev)).view(-1, 1600)
                at += n
        return out

    # the context resident on the GPU: points (N, 3, 15), local (N, 3, 15, 20), global (1, 3, 15, 80)
    P_gpu = torch.from_numpy(ds.feat).to(dev).transpose(2, 1).contiguous()
    L_gpu = torch.from_numpy(ds.local_feat).to(dev).transpose(2, 1).contiguous()
    G_gpu = torch.from_numpy(ds.global_feat).to(dev).transpose(2, 1).contiguous()

    def run_resident(model, batch=1024, amp=None):
        out = torch.empty((N, 1600), dtype=torch.float32, device=dev)
        ctx = torch.autocast("cuda", dtype=amp) if amp else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            for s in range(0, N, batch):
                e = min(N, s + batch)
                info = torch.cat((L_gpu[s:e], G_gpu.expand(e - s, -1, -1, -1)), dim=3)
                out[s:e] = model(P_gpu[s:e], info).view(-1, 1600).float()
        return out

    def timed(fn, reps=2):
        best, res = 1e9, None
        for _ in range(reps):
            sync(); a = time.perf_counter(); res = fn(); sync()
            best = min(best, time.perf_counter() - a)
        return res, best

    def settings(cudnn, tf32):
        torch.backends.cudnn.enabled = cudnn
        torch.backends.cudnn.benchmark = cudnn
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

    from tractcloud.tract_mapping import _CLUSTER_TO_TRACT_LUT as LUT
    LUT_t = torch.from_numpy(LUT.astype(np.int64)).to(dev)
    rows, ref = [], None

    def record(name, out, secs):
        nonlocal ref
        lab = out.argmax(1)
        row = {"variant": name, "seconds": round(secs, 2), "streamlines_per_s": int(N / secs)}
        if ref is None:
            ref = (out, lab)
        else:
            row["cluster_labels_differ"] = int((lab != ref[1]).sum())
            row["tract_labels_differ"] = int((LUT_t[lab] != LUT_t[ref[1]]).sum())
            row["max_abs_logp_diff"] = round(float((out - ref[0]).abs().max()), 4)
        rows.append(row)
        print(json.dumps(row), flush=True)

    m = model_fresh()
    settings(False, False); out, s = timed(lambda: run_upstream(m)); record("A upstream", out, s)
    settings(False, False); out, s = timed(lambda: run_resident(m)); record("B context on GPU", out, s)
    settings(True, False); out, s = timed(lambda: run_resident(m)); record("C + cudnn", out, s)
    settings(True, True); out, s = timed(lambda: run_resident(m)); record("D + TF32", out, s)
    settings(True, False); out, s = timed(lambda: run_resident(m, amp=torch.float16)); record("E + fp16 autocast", out, s)
    settings(True, False); out, s = timed(lambda: run_resident(m, amp=torch.bfloat16)); record("F + bf16 autocast", out, s)
    settings(True, False); out, s = timed(lambda: run_resident(m, batch=4096, amp=torch.float16)); record("G fp16, batch 4096", out, s)
    try:
        mc = torch.compile(model_fresh())
        settings(True, False)
        run_resident(mc, batch=4096, amp=torch.float16)                       # compile outside the timer
        out, s = timed(lambda: run_resident(mc, batch=4096, amp=torch.float16)); record("H + torch.compile", out, s)
    except Exception as e:                                                    # noqa: BLE001 - report, do not fail the run
        rows.append({"variant": "H + torch.compile", "error": f"{type(e).__name__}: {e}"[:300]})

    # the kNN context on the GPU, from the same random draws, against the CPU's
    feat32 = torch.from_numpy(ds.feat).to(dev)                                 # (N, 15, 3)
    sync(); a = time.perf_counter()
    rng = np.random.RandomState(0)
    rng.randint(0, N, 80)                                                     # the global draw comes first
    num_iter = max(N // 10000, 1); per = N // num_iter + 1
    same = total = 0
    for i in range(num_iter):
        s0, e0 = i * per, min((i + 1) * per, N)
        cur = feat32[s0:e0].reshape(e0 - s0, -1)
        ds_idx = torch.from_numpy(rng.choice(e0 - s0, size=int((e0 - s0) * 0.1), replace=False)).to(dev)
        sub = cur[ds_idx]
        d = (cur.square().sum(1, keepdim=True) + sub.square().sum(1)[None] - 2 * cur @ sub.T).clamp_min(0).sqrt()
        nn = ds_idx[d.topk(20, largest=False, dim=1).indices]                # (n, 20) chunk-local indices
        knn = feat32[s0:e0][nn]                                               # (n, 20, 15, 3)
        cpu = torch.from_numpy(ds.local_feat[s0:e0]).to(dev)                  # (n, 15, 3, 20)
        same += int(torch.isclose(knn.permute(0, 2, 3, 1), cpu, atol=1e-5).all(dim=(1, 2, 3)).sum())
        total += e0 - s0
    sync()
    knn_gpu = {"seconds": round(time.perf_counter() - a, 2), "streamlines_with_identical_context": same, "of": total}
    return {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "streamlines": N,
            "context_cpu_s": round(ctx_cpu_s, 2), "context_gpu": knn_gpu, "variants": rows}


@app.function(gpu="A10G", volumes={V: vol}, timeout=3600, memory=49152, cpu=8)
def bench_exact() -> dict:
    """Follow-up: torch.compile in exact fp32 (cudnn on, no TF32), and where fp16's flips land:
    the run-0 tract best-class margin of the streamlines fp16 relabels, against all of them."""
    import contextlib, sys, time, types
    import numpy as np, torch
    sys.modules.setdefault("vtk", types.ModuleType("vtk"))
    from tractcloud import inference as inf
    from tractcloud.tract_mapping import _CLUSTER_TO_TRACT_LUT as LUT
    dev = torch.device("cuda")
    feat = np.load(f"{V}/hcp/feat.npy")
    centered = inf.center_tractography(feat, np.load(f"{V}/TrainData_800clu800ol/HCP_mass_center.npy"))
    np.random.seed(0)
    ds = inf.RealDataDataset(centered, k=20, k_global=80, k_ds_rate=0.1)
    N = len(ds)
    P = torch.from_numpy(ds.feat).to(dev).transpose(2, 1).contiguous()
    L = torch.from_numpy(ds.local_feat).to(dev).transpose(2, 1).contiguous()
    G = torch.from_numpy(ds.global_feat).to(dev).transpose(2, 1).contiguous()
    load = lambda: inf.load_model(f"{V}/TrainedModel/best_tract_f1_model.pth", f"{V}/TrainedModel/cli_args.txt",
                                  dev, k_override=20, k_global_override=80)[0]

    def run(model, batch, amp=None):
        out = torch.empty((N, 1600), dtype=torch.float32, device=dev)
        ctx = torch.autocast("cuda", dtype=amp) if amp else contextlib.nullcontext()
        with torch.no_grad(), ctx:
            for s in range(0, N, batch):
                e = min(N, s + batch)
                out[s:e] = model(P[s:e], torch.cat((L[s:e], G.expand(e - s, -1, -1, -1)), dim=3)).view(-1, 1600).float()
        return out

    def timed(fn):
        best, res = 1e9, None
        for _ in range(2):
            torch.cuda.synchronize(); a = time.perf_counter(); res = fn(); torch.cuda.synchronize()
            best = min(best, time.perf_counter() - a)
        return res, round(best, 2)

    torch.backends.cudnn.enabled = False
    ref, t_ref = timed(lambda: run(load(), 1024))                        # upstream arithmetic
    torch.backends.cudnn.enabled = True; torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    rows = {"reference fp32, cudnn off": {"seconds": t_ref}}
    lut = torch.from_numpy(LUT.astype(np.int64)).to(dev)
    tract_ref = lut[ref.argmax(1)]
    # run-0 best-class margin of the labeled tract: winner minus the best class outside its tract
    outside = ref.masked_fill(lut[None, :] == tract_ref[:, None], float("-inf")).max(1).values
    margin = ref.max(1).values - outside
    for name, fn in (("fp32 cudnn, batch 4096", lambda m: run(m, 4096)),
                     ("fp32 cudnn + torch.compile, batch 4096", lambda m: run(m, 4096)),
                     ("fp16 autocast + torch.compile, batch 4096", lambda m: run(m, 4096, torch.float16))):
        m = load()
        if "compile" in name:
            m = torch.compile(m); fn(m)                                    # compile outside the timer
        out, t = timed(lambda: fn(m))
        lab = out.argmax(1)
        moved = lut[lab] != tract_ref
        row = {"seconds": t, "cluster_labels_differ": int((lab != ref.argmax(1)).sum()), "tract_labels_differ": int(moved.sum()),
               "max_abs_logp_diff": round(float((out - ref).abs().max()), 4)}
        if moved.any():
            row["relabeled_margin_median"] = round(float(margin[moved].median()), 3)
            row["relabeled_with_margin_under_0.5"] = round(float((margin[moved] < 0.5).float().mean()), 3)
        rows[name] = row
        print(name, row, flush=True)
    return {"all_margin_median": round(float(margin.median()), 3),
            "all_with_margin_under_0.5": round(float((margin < 0.5).float().mean()), 3), "rows": rows}


@app.local_entrypoint()
def exact():
    r = bench_exact.remote()
    print(json.dumps(r, indent=1))
    (HERE / "results" / "infer_opt_exact.json").write_text(json.dumps(r, indent=1))


@app.local_entrypoint()
def main():
    r = bench.remote()
    print(json.dumps(r, indent=1))
    (HERE / "results" / "infer_opt.json").write_text(json.dumps(r, indent=1))
