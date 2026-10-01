"""TRAKO against the predictive encoding, on the same tractogram, at matched grids.

    TRAKO_PY bench/tractography/trako_compare.py

TRAKO (Haehn et al., MICCAI 2020; github.com/bostongfx/TRAKO, MIT) stores streamlines as glTF
with Draco point-cloud compression: positions quantized to `quantization_bits` over the
bounding box's largest extent, sequential (order-preserving) encoding with Draco's difference
prediction, and the streamline lengths as a second Draco attribute. Defaults: 14 bits,
compression level 1.

Both sides encode the HCP tractogram's coordinates only (`coords_only=True`: its two per-vertex
arrays are left out of both). For each TRAKO setting the predictive encoding runs on the SAME
grid (Draco's quantum: extent / (2^bits - 1)), so the two differ only in prediction and entropy
coding. Errors are measured against the original float32 vertices, in streamline order; file
sizes are the whole .tko on TRAKO's side, and residuals plus lengths on ours.

TRAKO_PY: a Python with trako and TrakoDracoPy installed. In 2026 that took building
TrakoDracoPy from source: its setup.py pins cmake < 3.15 and imports packaging.LegacyVersion
(both removed here), and its shipped Cython output predates Python 3.11 (regenerated with
cython < 3), and it predates NumPy 2 and NumPy 1.24: it ran under numpy 1.26 with its three
uses of the removed aliases np.float / np.int replaced by float / int, which they aliased.
numcodecs is needed too, for the predictive side.

Writes results/trako.json.
"""
import json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import trako as TKO
from vtk.util.numpy_support import vtk_to_numpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _geometry import encode, lengths_bytes

HERE = Path(__file__).resolve().parent
VTP = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography")) / "TestData/HCP/101006_ukf_pp_with_region.vtp"


def vertices(pd):
    """(V, 3) float64 in streamline order, and (N + 1,) offsets."""
    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    lines = pd.GetLines()
    if lines.GetOffsetsArray() is not None and lines.GetNumberOfCells():
        off = vtk_to_numpy(lines.GetOffsetsArray()).astype(np.int64)
        conn = vtk_to_numpy(lines.GetConnectivityArray()).astype(np.int64)
    else:                                                            # legacy cell array
        raw = vtk_to_numpy(lines.GetData()).astype(np.int64)
        lens, conn, i = [], [], 0
        while i < len(raw):
            n = raw[i]; lens.append(n); conn.append(raw[i + 1:i + 1 + n]); i += n + 1
        off = np.r_[0, np.cumsum(lens)]; conn = np.concatenate(conn)
    return pts[conn], off


import vtk
r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(VTP)); r.Update()
P, off = vertices(r.GetOutput())
V, N = len(P), len(off) - 1
lo, hi = P.min(0), P.max(0)
extent = float((hi - lo).max())
rows = []
for bits in (14, 12, 11):
    for cl in (1, 10):
        conf = {"*": {"position": False, "sequential": True, "quantization_bits": bits, "compression_level": cl,
                      "quantization_range": -1, "quantization_origin": None},
                "POSITION": {"position": True, "sequential": True, "quantization_bits": bits, "compression_level": cl,
                             "quantization_range": -1, "quantization_origin": None}}
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "t.tko")
            t0 = time.time()
            TKO.Encoder.fromVtp(str(VTP), config=conf, verbose=False, coords_only=True).save(out)
            t_enc = time.time() - t0
            size = os.path.getsize(out)
            t0 = time.time()
            D, doff = vertices(TKO.Decoder.toVtp(out, verbose=False))
            t_dec = time.time() - t0
        assert (doff == off).all(), "TRAKO changed the streamline structure"
        err = np.abs(D - P)
        q = extent / (2 ** bits - 1)
        t0 = time.time()
        ours, Q = encode(P, off, q, lo, order=2)
        t_ours = time.time() - t0
        ours += lengths_bytes(off)
        oerr = np.abs(Q * q + lo - P)
        rows.append({
            "bits": bits, "grid_mm": round(q, 4), "draco_compression_level": cl,
            "trako": {"bytes": size, "B_per_vertex": round(size / V, 3), "max_axis_err_mm": round(float(err.max()), 4),
                      "mean_euclid_err_mm": round(float(np.linalg.norm(D - P, axis=1).mean()), 4),
                      "encode_s": round(t_enc, 1), "decode_s": round(t_dec, 1)},
            "predictive": {"bytes": int(ours), "B_per_vertex": round(ours / V, 3), "max_axis_err_mm": round(float(oerr.max()), 4),
                           "mean_euclid_err_mm": round(float(np.linalg.norm(Q * q + lo - P, axis=1).mean()), 4),
                           "encode_s": round(t_ours, 1)},
            "trako_over_predictive": round(size / ours, 2)})
        print(json.dumps(rows[-1]), flush=True)

(HERE / "results" / "trako.json").write_text(json.dumps(
    {"tractogram": "HCP 101006, UKF (TractCloud TestData v1.0.0)", "streamlines": N, "vertices": V,
     "extent_mm": round(extent, 2), "coords_only": True, "rows": rows}, indent=1))
