"""Inputs for the time-to-display benchmark (render/), on top of decode_prep.py's.

    TRAKO_PY bench/tractography/decode/render_prep.py

Writes to DATA/decode/:
    hcp_<bits>.tko              TRAKO's file as a viewer would fetch it (glTF JSON, Draco buffers
                                embedded base64)
    chunks_<bits>/cNN.bin       the predictive encoding in 50 chunks of consecutive streamlines,
                                each self-contained: u32 lines, u32 lensBytes, u32 headBytes,
                                u32 bodyBytes, u32 bodyItemBytes (1: int8, 2: int16), then the
                                three blosc blobs (lengths, head, body)
    render_<bits>.json          sizes on the wire: .tko raw and gzipped (what HTTP compression
                                would send), the predictive blobs, the chunk files
"""
import gzip, json, os, struct, sys
from pathlib import Path
import numpy as np
import trako as TKO
import vtk
from numcodecs import Blosc
from vtk.util.numpy_support import vtk_to_numpy
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _geometry import line_index, residuals

DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
VTP = DATA / "TestData/HCP/101006_ukf_pp_with_region.vtp"
OUT = DATA / "decode"
CHUNKS = 50
r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(VTP)); r.Update(); pd = r.GetOutput()
pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
off = vtk_to_numpy(pd.GetLines().GetOffsetsArray()).astype(np.int64)
P = pts[vtk_to_numpy(pd.GetLines().GetConnectivityArray())]
lo = P.min(0)
extent = float((P.max(0) - lo).max())
k = line_index(off)
z = lambda a, s: Blosc(cname="zstd", clevel=9, shuffle=s).encode(np.ascontiguousarray(a))

for bits in (14, 12):
    conf = {f: {"position": f == "POSITION", "sequential": True, "quantization_bits": bits, "compression_level": 1,
                "quantization_range": -1, "quantization_origin": None} for f in ("*", "POSITION")}
    tko = OUT / f"hcp_{bits}.tko"
    TKO.Encoder.fromVtp(str(VTP), config=conf, verbose=False, coords_only=True).save(str(tko))
    tko_gz = len(gzip.compress(tko.read_bytes(), 6))

    q = extent / (2 ** bits - 1)
    Q = np.round((P - lo) / q).astype(np.int64)
    R = residuals(Q, k, 2)
    bdt = "<i1" if np.abs(R[k >= 2]).max() < 128 else "<i2"      # int16 at 14 bits (max 146)
    cdir = OUT / f"chunks_{bits}"
    cdir.mkdir(exist_ok=True)
    sizes = []
    for c, lines in enumerate(np.array_split(np.arange(len(off) - 1), CHUNKS)):
        v0, v1 = off[lines[0]], off[lines[-1] + 1]
        kk, RR = k[v0:v1], R[v0:v1]
        lens = z(np.diff(off[lines[0]:lines[-1] + 2]).astype("<u2"), Blosc.SHUFFLE)
        head = z(RR[kk < 2].astype("<i2"), Blosc.SHUFFLE)
        body = z(RR[kk >= 2].astype(bdt), Blosc.BITSHUFFLE)
        blob = struct.pack("<IIIII", len(lines), len(lens), len(head), len(body), np.dtype(bdt).itemsize) + lens + head + body
        (cdir / f"c{c:02d}.bin").write_bytes(blob)
        sizes.append(len(blob))
    single = sum(os.path.getsize(OUT / f"ours_{bits}_{n}.blosc") for n in ("body", "head", "lens"))
    info = {"bits": bits, "quantum": q, "origin": lo.tolist(), "vertices": int(off[-1]), "streamlines": len(off) - 1,
            "chunks": CHUNKS, "body_dtype": bdt, "tko_bytes": tko.stat().st_size, "tko_gzip_bytes": tko_gz,
            "predictive_single_bytes": single, "predictive_chunked_bytes": sum(sizes),
            "chunk_bytes_first": sizes[0]}
    (OUT / f"render_{bits}.json").write_text(json.dumps(info))
    print(json.dumps(info), flush=True)
