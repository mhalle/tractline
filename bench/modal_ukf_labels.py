"""Does float32 tracking change TractCloud's tract labels more than the scan's own noise does, or more
than TractCloud's own random context does? Full Stanford HARDI, ORG settings, on A10Gs.

    modal run bench/modal_ukf_labels.py

Four whole-brain tractographies from the SAME seed points (the 98,491 the binary accepts on the scan
as acquired; ukf.track(seed_points=...)):
  f64     float64, the scan as acquired (the reference: it matches the Slicer binary)
  f32     float32, the scan as acquired
  boot0/1 float64, wild-bootstrap replicates 0 and 1 (_bootstrap.py, as ukf_noise_floor.py makes them)
Each is cut at 40 mm (wm_preprocess_all.py -l 40, as TractCloud's input was), resampled to 15 points
(resample.py, upstream's rule), centered on HCP's mass center and labeled by upstream TractCloud
(k 20, k_global 80, cudnn off) with seeded context draws: 10 for f64, 5 for the others.

Compared, per seed both keep: tract-label agreement of single draws and of 5-draw majority votes,
against the floor of TractCloud's own draws on f64 (draw vs draw; vote of 0-4 vs vote of 5-9); the
share labeled Other; the tract mix (Pearson r of per-tract counts); and agreement split by whether the
fiber itself moved (same point count and within 0.1 mm of f64's, or not).

Tractographies go to the Volume (ukf/hardi/variants/); writes results/ukf_labels.json.
"""
import os
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src/tractline"                                # the package, shipped as a directory
DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.14.1", "numpy>=2", "pynrrd", "dipy")
         .env({"PYTHONPATH": "/root/pkg:/root"})
         .add_local_dir(str(PKG), remote_path="/root/pkg/tractline")
         .add_local_file(str(HERE / "_bootstrap.py"), remote_path="/root/_bootstrap.py")
         .add_local_dir(str(DATA / "TractCloud/src/tractcloud"), remote_path="/root/pkg/tractcloud"))
vol = modal.Volume.from_name("tractography-bench")
app = modal.App("tractography-ukf-labels", image=image)
V, HD = "/vol", "/vol/ukf/hardi/"
VARIANTS = {"f64": ("float64", None), "f32": ("float32", None), "boot0": ("float64", 20261001), "boot1": ("float64", 20261002)}


@app.function(gpu="A10G", volumes={V: vol}, timeout=7200, memory=32768)
def track(name: str) -> dict:
    import os, time
    import numpy as np, torch
    from tractline import ukf as U
    from _bootstrap import WildBootstrap, read
    dtype, boot = VARIANTS[name]
    D = U.load(HD + "dwi.nhdr", HD + "mask.nrrd")                    # seeds from the scan as acquired, on the CPU
    off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
    pts, *_ = U.seeds(D, off)
    if boot is not None:
        raw, g, b0 = read(HD + "dwi.nhdr")
        D = U.load(HD + "dwi.nhdr", HD + "mask.nrrd", data=WildBootstrap(raw, g, b0, sh_order=6).replicate(boot))
    t0 = time.time()
    f, st = U.track(D, off, seed_points=pts, dtype=getattr(torch, dtype), device="cuda")
    torch.cuda.synchronize()
    s = time.time() - t0
    os.makedirs(HD + "variants", exist_ok=True)
    lens = np.array([len(x) for x in f])
    np.savez(HD + f"variants/{name}.npz", points=np.concatenate(f).astype(np.float32), offsets=np.r_[0, np.cumsum(lens)],
             seed_index=np.array(st["seed_index"]))
    vol.commit()
    return {"name": name, "dtype": dtype, "bootstrap_seed": boot, "seeds": st["seeds"], "fibers": st["fibers"],
            "fiber_steps": st["fiber_steps"], "track_s": round(s, 1), "steps_per_s": int(st["fiber_steps"] / s)}


@app.function(gpu="A10G", volumes={V: vol}, timeout=3600, memory=49152, cpu=8)
def label() -> dict:
    import sys, types
    import numpy as np, torch
    sys.modules.setdefault("vtk", types.ModuleType("vtk"))
    from tractcloud import inference as inf
    from tractcloud.tract_mapping import TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as LUT
    from tractline.resample import resample
    vol.reload()
    dev = torch.device("cuda")
    torch.backends.cudnn.enabled = False
    model, _ = inf.load_model(f"{V}/TrainedModel/best_tract_f1_model.pth", f"{V}/TrainedModel/cli_args.txt",
                              dev, k_override=20, k_global_override=80)
    center = np.load(f"{V}/TrainData_800clu800ol/HCP_mass_center.npy")
    lut = LUT.astype(np.int64)

    def labels(feat, draws):
        out = []
        centered = inf.center_tractography(feat, center)
        for seed in draws:
            np.random.seed(seed)
            ds = inf.RealDataDataset(centered, k=20, k_global=80, k_ds_rate=0.1)
            P = torch.from_numpy(ds.feat).to(dev).transpose(2, 1).contiguous()
            L = torch.from_numpy(ds.local_feat).to(dev).transpose(2, 1).contiguous()
            G = torch.from_numpy(ds.global_feat).to(dev).transpose(2, 1).contiguous()
            cl = torch.empty(len(ds), dtype=torch.long, device=dev)
            with torch.no_grad():
                for s in range(0, len(ds), 1024):
                    e = min(len(ds), s + 1024)
                    cl[s:e] = model(P[s:e], torch.cat((L[s:e], G.expand(e - s, -1, -1, -1)), 3)).view(-1, 1600).argmax(1)
            out.append(lut[cl.cpu().numpy()])
        return np.stack(out)

    def vote(T):                                                         # majority over draws; ties to the lowest tract id
        return np.array([np.bincount(col, minlength=43).argmax() for col in T.T])

    tr, lab = {}, {}
    for name in VARIANTS:
        z = np.load(HD + f"variants/{name}.npz")
        Pts, off, si = z["points"].astype(np.float64), z["offsets"], z["seed_index"]
        seglen = np.linalg.norm(np.diff(Pts, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seglen)]
        length = cum[off[1:] - 1] - cum[off[:-1]]
        keep = length >= 40
        feat = resample(torch.from_numpy(Pts), torch.from_numpy(off)).numpy()[keep]
        tr[name] = {"seed_index": si[keep], "points": Pts, "offsets": off, "keep": keep}
        lab[name] = labels(feat, range(10) if name == "f64" else range(5))

    ref = tr["f64"]; rk = {int(s): i for i, s in enumerate(ref["seed_index"])}
    rfib = {int(s): j for j, s in enumerate(np.flatnonzero(ref["keep"]))}  # kept index -> fiber index
    agree = lambda a, b: round(float((a == b).mean()), 4)
    T0 = lab["f64"]
    out = {"floor_tractcloud_draws_on_f64": {
        "single_draw_pairs_mean": round(float(np.mean([agree(T0[i], T0[j]) for i in range(5) for j in range(i + 1, 5)])), 4),
        "vote_0_4_vs_vote_5_9": agree(vote(T0[:5]), vote(T0[5:])),
        "streamlines": int(T0.shape[1]), "other_fraction_draw0": round(float((T0[0] == 42).mean()), 4)}}
    v_ref = vote(T0[:5])
    counts = lambda t: np.bincount(t, minlength=43)
    v_alt = vote(T0[5:])
    out["floor_tractcloud_draws_on_f64"].update({
        "vote_tract_mix_r": round(float(np.corrcoef(counts(v_alt)[:42], counts(v_ref)[:42])[0, 1]), 5),
        "vote_per_tract_count_change_median_abs_rel": round(float(np.median(np.abs(counts(v_alt)[:42] - counts(v_ref)[:42]) / np.maximum(counts(v_ref)[:42], 1))), 4),
        "vote_other_fraction_5_9": round(float((v_alt == 42).mean()), 4)})
    for name in ("f32", "boot0", "boot1"):
        x = tr[name]; T = lab[name]
        both = [(i, rk[int(s)]) for i, s in enumerate(x["seed_index"]) if int(s) in rk]
        ix, ir = np.array([b[0] for b in both]), np.array([b[1] for b in both])
        # did the fiber itself move? same point count and every point within 0.1 mm of f64's
        xs = np.flatnonzero(x["keep"])
        rs = np.flatnonzero(ref["keep"])
        moved = np.ones(len(both), bool)
        for n, (a, b) in enumerate(zip(ix, ir)):
            fa, fb = xs[a], rs[b]
            pa = x["points"][x["offsets"][fa]:x["offsets"][fa + 1]]
            pb = ref["points"][ref["offsets"][fb]:ref["offsets"][fb + 1]]
            if len(pa) == len(pb):
                moved[n] = min(np.linalg.norm(pa - pb, axis=1).max(), np.linalg.norm(pa - pb[::-1], axis=1).max()) >= 0.1
        v = vote(T)
        single = [agree(T[d][ix], T0[d][ir]) for d in range(5)]
        same_v = v[ix] == v_ref[ir]
        out[name] = {"streamlines_ge_40mm": int(T.shape[1]), "both": int(len(both)),
                     "fibers_moved_fraction": round(float(moved.mean()), 4),
                     "single_draw_agreement_with_f64_mean": round(float(np.mean(single)), 4),
                     "vote_agreement_with_f64": round(float(same_v.mean()), 4),
                     "vote_agreement_unmoved_fibers": round(float(same_v[~moved].mean()), 4) if (~moved).any() else None,
                     "vote_agreement_moved_fibers": round(float(same_v[moved].mean()), 4) if moved.any() else None,
                     "other_fraction_vote": round(float((v == 42).mean()), 4),
                     "tract_mix_r_vs_f64": round(float(np.corrcoef(counts(v)[:42], counts(v_ref)[:42])[0, 1]), 5),
                     "per_tract_count_change_median_abs_rel": round(float(np.median(np.abs(counts(v)[:42] - counts(v_ref)[:42]) / np.maximum(counts(v_ref)[:42], 1))), 4)}
    out["f64"] = {"streamlines_ge_40mm": int(T0.shape[1]), "other_fraction_vote": round(float((v_ref == 42).mean()), 4),
                  "named_tracts_50_plus": int((counts(v_ref)[:42] >= 50).sum())}
    out["tract_names"] = list(TRACT_NAMES)
    return out


@app.local_entrypoint()
def main(skip_tracking: bool = False):
    res = {}
    if not skip_tracking:
        res["tracking"] = list(track.map(list(VARIANTS)))
        print(json.dumps(res["tracking"], indent=1))
    res["labels"] = label.remote()
    res["labels"].pop("tract_names", None)
    print(json.dumps(res["labels"], indent=1))
    (HERE / "results" / "ukf_labels.json").write_text(json.dumps(res, indent=1))
