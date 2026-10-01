"""Inputs for decode_bench.mjs: the same tractogram encoded both ways, as raw buffers.

    TRAKO_PY bench/tractography/decode/decode_prep.py          (TRAKO_PY as in trako_compare.py)
    cd DATA/decode && npm install draco3d numcodecs && node <repo>/bench/tractography/decode/decode_bench.mjs DATA/decode

Writes DATA/decode/ for 14, 12 and 11 bits:
    draco_pos_<bits>.drc            TRAKO's raw Draco position buffer (out of its base64 glTF)
    ours_<bits>_{body,head,lens}.blosc   the predictive encoding on Draco's grid (_geometry.py)
    ours_<bits>.json                quantum, origin, residual dtype, counts
    truth_<bits>.npy                the grid positions as float32: what an exact decode returns
and times both decoders in Python (results/decode_python.json). The Python reconstruction is
_geometry.decode, plain numpy over int64 - several whole-array passes, not a tight loop - so it
is an upper bound on what the predictive decode costs in Python.
"""
import base64, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import pygltflib, TrakoDracoPy, vtk
import trako as TKO
from numcodecs import Blosc
from vtk.util.numpy_support import vtk_to_numpy
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _geometry import decode, line_index, residuals

DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
VTP = DATA / "TestData/HCP/101006_ukf_pp_with_region.vtp"
OUT = DATA / "decode"
OUT.mkdir(exist_ok=True)
r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(VTP)); r.Update(); pd = r.GetOutput()
pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
off = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
P = pts[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
V, lo = len(P), P.min(0)
extent = float((P.max(0) - lo).max())
k = line_index(off)
z = lambda a, s: Blosc(cname="zstd", clevel=9, shuffle=s).encode(np.ascontiguousarray(a))
rows = []
for bits in (14, 12, 11):
    conf = {f: {"position": f == "POSITION", "sequential": True, "quantization_bits": bits, "compression_level": 1,
                "quantization_range": -1, "quantization_origin": None} for f in ("*", "POSITION")}
    with tempfile.TemporaryDirectory() as tmp:
        tko = os.path.join(tmp, "t.tko")
        TKO.Encoder.fromVtp(str(VTP), config=conf, verbose=False, coords_only=True).save(tko)
        g = pygltflib.GLTF2().load(tko)
    acc = g.accessors[g.meshes[0].primitives[0].attributes.POSITION]
    bv = g.bufferViews[acc.bufferView]
    uri = g.buffers[bv.buffer].uri
    drc = base64.b64decode(uri[uri.index(",") + 1:])[bv.byteOffset:bv.byteOffset + bv.byteLength]
    (OUT / f"draco_pos_{bits}.drc").write_bytes(drc)

    q = extent / (2 ** bits - 1)
    Q = np.round((P - lo) / q).astype(np.int64)
    R = residuals(Q, k, 2)
    body, head = R[k >= 2], R[k < 2]
    bdt = "<i1" if np.abs(body).max() < 128 else "<i2"
    bufs = {"body": z(body.astype(bdt), Blosc.BITSHUFFLE), "head": z(head.astype("<i2"), Blosc.SHUFFLE),
            "lens": z(np.diff(off).astype("<u2"), Blosc.SHUFFLE)}
    for n, b in bufs.items():
        (OUT / f"ours_{bits}_{n}.blosc").write_bytes(b)
    (OUT / f"ours_{bits}.json").write_text(json.dumps({"bits": bits, "quantum": q, "origin": lo.tolist(), "body_dtype": bdt,
                                                      "vertices": V, "streamlines": len(off) - 1}))
    truth = (Q * q + lo).astype(np.float32)
    np.save(OUT / f"truth_{bits}.npy", truth)

    td, tp = [], []
    for _ in range(3):
        t0 = time.perf_counter(); pc = TrakoDracoPy.decode_point_cloud_buffer(drc); t1 = time.perf_counter()
        np.asarray(pc.points, dtype=np.float32); t2 = time.perf_counter()
        td.append((t1 - t0, t2 - t1))
        t0 = time.perf_counter()
        lens = np.frombuffer(Blosc().decode(bufs["lens"]), "<u2").astype(np.int64)
        o = np.r_[0, np.cumsum(lens)]; kk = line_index(o)
        Rd = np.empty((V, 3), np.int64)
        Rd[kk >= 2] = np.frombuffer(Blosc().decode(bufs["body"]), bdt).reshape(-1, 3)
        Rd[kk < 2] = np.frombuffer(Blosc().decode(bufs["head"]), "<i2").reshape(-1, 3)
        t1 = time.perf_counter()
        Pd = (decode(Rd, kk, o, 2) * q + lo).astype(np.float32)
        t2 = time.perf_counter()
        tp.append((t1 - t0, t2 - t1))
    assert np.array_equal(Pd, truth)
    rows.append({"bits": bits, "draco_bytes": len(drc), "predictive_bytes": sum(len(b) for b in bufs.values()),
                 "python_draco_decode_s": round(min(a for a, _ in td), 3),
                 "python_draco_points_to_numpy_s": round(min(b for _, b in td), 3),
                 "python_predictive_decompress_s": round(min(a for a, _ in tp), 3),
                 "python_predictive_reconstruct_s": round(min(b for _, b in tp), 3)})
    print(json.dumps(rows[-1]), flush=True)
(HERE.parent / "results" / "decode_python.json").write_text(json.dumps({"vertices": V, "rows": rows}, indent=1))
