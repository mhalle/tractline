"""UKF tractography timed on this Mac, from the prebuilt Slicer extension, on a public DWI.

    python bench/ukf_bench.py [--seeds-per-voxel 1] [--threads 8]

The one stage of the pipeline not measured elsewhere (pipeline.md §5). Nothing is installed into
/Applications/Slicer.app:
  - UKFTractography for Slicer 5.12.3 (revision 34627), macOS x86_64, UKF commit 2d2b661:
    34627-macosx-amd64-UKFTractography-git2d2b661-2025-06-02.tar.gz, 871,923 bytes, from
    slicer-packages.kitware.com, SHA-512 checked against the server's at download. It is unpacked
    under DATA/ukf/ and the installed Slicer's lib/, Frameworks/ and bin/ are linked beside it,
    which is where the binary's @rpath (@loader_path/../../../../../) looks. It runs under Rosetta.
  - Stanford HARDI (DIPY's fetch_stanford_hardi; stacks.stanford.edu druid:yx282xq2090), MD5s
    matching DIPY's: 81x106x76 at 2 mm, 150 directions at b=2000 plus 10 b0.
  - DWI -> NRRD: Slicer's DWIConvert (FSLToNrrd). Mask: DIPY median_otsu on the mean b0
    (median_radius 4, numpass 4), written as NRRD on the DWI's grid: UKF reads masks with teem's
    nrrd reader, not ITK's.
  - UKF with the ORG-atlas settings (SupWMA, arXiv 2207.08975): two tensors, seedingThreshold 0.1,
    stoppingFA 0.08, stoppingThreshold 0.06, recordLength 1.8 (measured on the HCP test tractogram).

Writes DATA/ukf/hardi/ukf_spv<N>.vtk and .log (UKF prints every parameter, user-set or default),
and results/ukf.json: wall and CPU time, peak memory, streamlines and points, and how many
survive the 40 mm cut that wm_preprocess_all.py -l 40 would apply.
"""
import argparse, json, os, re, resource, subprocess, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seeds-per-voxel", type=int, default=1)
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()

DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
H = DATA / "ukf" / "hardi"
PKG = DATA / "ukf" / "34627-macosx-amd64-UKFTractography-git2d2b661-2025-06-02" / "Slicer.app" / "Contents"
UKF = PKG / "Extensions-34627/UKFTractography/lib/Slicer-5.12/cli-modules/UKFTractography"
OUT = Path(__file__).resolve().parent / "results"

# the mask as NRRD on the DWI's grid
import nibabel as nib
import nrrd
if not (H / "mask.nrrd").exists():
    h = nrrd.read_header(str(H / "dwi.nhdr"))
    dom = [i for i, k in enumerate(h["kinds"]) if k in ("domain", "space")]
    m = np.asarray(nib.load(H / "mask.nii.gz").dataobj).astype(np.uint8)
    assert tuple(m.shape) == tuple(np.asarray(h["sizes"])[dom]), (m.shape, h["sizes"])
    dirs = h["space directions"]
    nrrd.write(str(H / "mask.nrrd"), m, {"type": "unsigned char", "dimension": 3, "space": h["space"],
               "kinds": ["domain"] * 3, "space directions": np.array([np.asarray(dirs[i], float) for i in dom]),
               "space origin": np.asarray(h["space origin"], float), "encoding": "gzip"})

tag = f"spv{args.seeds_per_voxel}"
cmd = [str(UKF), "--dwiFile", "dwi.nhdr", "--maskFile", "mask.nrrd", "--tracts", f"ukf_{tag}.vtk",
       "--numTensor", "2", "--seedingThreshold", "0.1", "--stoppingFA", "0.08", "--stoppingThreshold", "0.06",
       "--recordLength", "1.8", "--seedsPerVoxel", str(args.seeds_per_voxel), "--numThreads", str(args.threads)]
r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
t0 = time.time()
p = subprocess.run(cmd, cwd=H, capture_output=True, text=True)
wall = time.time() - t0
r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
(H / f"ukf_{tag}.log").write_text(p.stdout + p.stderr)
if p.returncode:
    raise SystemExit(f"UKF failed ({p.returncode}):\n{(p.stdout + p.stderr)[-2000:]}")

import vtk
from vtk.util.numpy_support import vtk_to_numpy
rd = vtk.vtkPolyDataReader(); rd.SetFileName(str(H / f"ukf_{tag}.vtk")); rd.Update(); pd = rd.GetOutput()
lines = pd.GetLines()
off = vtk_to_numpy(lines.GetOffsetsArray()).astype(np.int64)
P = vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)[vtk_to_numpy(lines.GetConnectivityArray())]
seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
starts = np.zeros(len(P), bool); starts[off[:-1]] = True
seg[starts[1:]] = 0
length = np.add.reduceat(np.r_[seg, 0], off[:-1])
params = dict(re.findall(r"^[*-] (\w+): (\S+)", p.stdout + p.stderr, re.M))
user_set = re.findall(r"^\* (\w+):", p.stdout + p.stderr, re.M)
res = {"tool": "UKFTractography 2d2b661 (Slicer 5.12.3 extension, macOS x86_64 under Rosetta on an Apple M2)",
       "data": "Stanford HARDI, 81x106x76 at 2 mm, 150 directions b=2000 + 10 b0",
       "mask_voxels": int(np.asarray(nib.load(H / "mask.nii.gz").dataobj).sum()),
       "seeds_per_voxel": args.seeds_per_voxel, "threads": args.threads,
       "wall_s": round(wall, 1), "cpu_s": round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 1),
       "peak_rss_mb": round(r1.ru_maxrss / 1e6, 1),
       "streamlines": int(pd.GetNumberOfLines()), "points": int(pd.GetNumberOfPoints()),
       "streamlines_over_40mm": int((length >= 40).sum()),
       "point_arrays": [pd.GetPointData().GetArrayName(i) for i in range(pd.GetPointData().GetNumberOfArrays())],
       "parameters_reported": params, "user_set": user_set, "command": " ".join(cmd[1:])}
res["streamlines_per_s"] = round(res["streamlines"] / wall, 1)
OUT.mkdir(exist_ok=True)
prev = json.loads((OUT / "ukf.json").read_text()) if (OUT / "ukf.json").exists() else {}
prev[tag] = res
(OUT / "ukf.json").write_text(json.dumps(prev, indent=1))
print(json.dumps(res, indent=1))
