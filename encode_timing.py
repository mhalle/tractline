"""How long the compact form takes to WRITE on this machine: the whole HCP 101006 subject (440,621
streamlines, 21.6 M vertices), geometry and field, as payload.json sized them.

    DATA/.venv/bin/python bench/tractography/encode_timing.py [--repeats 3]

Geometry (_geometry.py's encoding, on TRAKO's grid for 14 and 12 bits: extent / (2^bits - 1)):
  quantize to the grid, second-order residuals, blosc (zstd 9; bitshuffle for the residual body,
  byte shuffle for the head rows and the uint16 lengths). Timed without _geometry.encode's
  round-trip assert; the round trip is checked once, outside the clock.
Field: rankfield.encode of run 0's float16 log-probabilities (hcp_full/logp_run0.npy, from
  modal_capture.py), keep="clip", depth 6, clip 8 (encode.py's), then blosc (zstd 9, shuffle) of
  ranks, support and tail. Run 0's labels are checked against the field's top rank, outside the clock.
Reading the inputs (the VTP, the 1.4 GB log-probabilities) is not timed: a pipeline has them in memory.
Three geometry configurations: payload.json's (int64 arithmetic, zstd 9), the same in int32 (identical
residuals, checked), and int32 with zstd 5 (larger output); the field at zstd 9 and 5. rankfield is
whatever is importable; the optimized encoder (rankfield worktree 3efef7f, unreleased) is 2.7x faster than
0.3.10's: run with PYTHONPATH pointing at that worktree's src.

Writes results/encode_timing.json.
"""
import argparse, json, platform, subprocess, time
from pathlib import Path
import numpy as np, torch, vtk
from numcodecs import Blosc, blosc
from vtk.util.numpy_support import vtk_to_numpy
import rankfield as rf
from _data import DATA, HCP_VTP
from _geometry import line_index, residuals, decode

ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=3); args = ap.parse_args()
HERE = Path(__file__).resolve().parent

r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(HCP_VTP)); r.Update(); pd = r.GetOutput()
pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
off = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
P = pts[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
lo = P.min(0); extent = float((P.max(0) - lo).max())
logp = np.load(DATA / "hcp_full/logp_run0.npy", mmap_mode="r")
logp = np.ascontiguousarray(logp)                                       # in memory, as after inference
N = len(off) - 1
assert logp.shape == (N, 1600), logp.shape


def residuals32(Q, k):
    """residuals(Q, k, 2) in int32: the same values (checked), half the memory traffic."""
    R = np.empty_like(Q); R[0] = Q[0]; np.subtract(Q[1:], Q[:-1], out=R[1:])
    R2 = R.copy(); np.subtract(R[2:], R[1:-1], out=R2[2:])
    m = k == 1; R2[m] = R[m]; m = k == 0; R2[m] = Q[m]
    return R2


def geometry(quantum, int32=False, clevel=9):
    t = {}
    t0 = time.perf_counter()
    if int32:
        k = line_index(off.astype(np.int32))
        Q = np.round((P - lo) / quantum).astype(np.int32)
        R = residuals32(Q, k)
    else:
        k = line_index(off)
        Q = np.round((P - lo) / quantum).astype(np.int64)
        R = residuals(Q, k, 2)
    t["grid_and_residuals"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    body, head = R[k >= 2], R[k < 2]
    bdt = "<i1" if np.abs(body).max() < 128 else "<i2"
    hdt = "<i2" if np.abs(head).max() < 32768 else "<i4"
    b = Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.BITSHUFFLE).encode(np.ascontiguousarray(body.astype(bdt)))
    h = Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.SHUFFLE).encode(np.ascontiguousarray(head.astype(hdt)))
    ln = Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.SHUFFLE).encode(np.diff(off).astype("<u2"))
    t["compress"] = time.perf_counter() - t0
    return t, len(b) + len(h) + len(ln), (Q, R, k)


def field(clevel=9):
    t = {}
    t0 = time.perf_counter()
    code = rf.encode(torch.from_numpy(logp).T[:, :, None, None], keep="clip", depth=6, clip=8.0, tail_temperatures=(1.0,))
    t["rankfield_encode"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    z = lambda a: Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.SHUFFLE).encode(np.ascontiguousarray(np.asarray(a)))
    nbytes = sum(len(z(a)) for a in (code.ranks, code.support, code.tail))
    t["compress"] = time.perf_counter() - t0
    return t, nbytes, code


chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
res = {"machine": chip or platform.processor(), "torch_threads": torch.get_num_threads(), "blosc_threads": blosc.get_nthreads(),
       "rankfield": rf.__version__, "streamlines": N, "vertices": int(off[-1]), "geometry": {}, "field": {}}
GEOM = {"payload (int64, zstd 9)": dict(), "int32 residuals, zstd 9": dict(int32=True), "int32 residuals, zstd 5": dict(int32=True, clevel=5)}
rows = lambda runs: [{s: round(v, 2) for s, v in t.items()} | {"total": round(sum(t.values()), 2)} for t, _, _ in runs]
best = lambda runs: round(min(sum(t.values()) for t, _, _ in runs), 2)
for bits in (14, 12):
    q = extent / (2 ** bits - 1)
    res["geometry"][f"{bits}_bits"] = {"quantum_mm": round(q, 5)}
    ref = None
    for name, kw in GEOM.items():
        runs = [geometry(q, **kw) for _ in range(args.repeats)]
        _, nb, (Q, R, k) = runs[-1]
        assert (decode(R.astype(np.int64), k.astype(np.int64), off, 2) == Q).all(), "geometry round trip"
        ref = R if ref is None else ref
        res["geometry"][f"{bits}_bits"][name] = {"mb": round(nb / 1e6, 2), "residuals_identical": bool((R == ref).all()),
                                                 "best_s": best(runs), "seconds": rows(runs)}
for name, lvl in (("zstd 9", 9), ("zstd 5", 5)):
    runs = [field(lvl) for _ in range(args.repeats)]
    _, nb, code = runs[-1]
    top = np.asarray(code.ranks).reshape(6, N)[0].astype(np.int64) - 1
    res["field"][name] = {"depth": 6, "mb": round(nb / 1e6, 2), "top_rank_equals_argmax": bool((top == logp.argmax(1)).all()),
                          "best_s": best(runs), "seconds": rows(runs)}
res["totals"] = {f"{b}_bits, {g} + field {fz}": {"mb": round(res["geometry"][f"{b}_bits"][g]["mb"] + res["field"][fz]["mb"] + 0.04, 1),
                                                  "best_s": round(res["geometry"][f"{b}_bits"][g]["best_s"] + res["field"][fz]["best_s"], 2)}
                 for b in (14, 12) for g, fz in (("payload (int64, zstd 9)", "zstd 9"), ("int32 residuals, zstd 9", "zstd 9"),
                                                  ("int32 residuals, zstd 5", "zstd 5"))}
res["rankfield_source"] = rf.__file__
print(json.dumps(res, indent=1))
(HERE / "results" / "encode_timing.json").write_text(json.dumps(res, indent=1))
