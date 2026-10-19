"""TractSeg's side of the tracker benchmark (bench/robust_benchmark.py; docs/robustness-plan.md, step 6). Runs in the
isolated TractSeg 2.10 environment (DATA/tools/tractseg-2.10, Python 3.11, its own torch), never in tractline's:

    <tractseg-2.10>/bin/python -I bench/_tractseg_run.py PEAKS.npz OUT.npz

PEAKS.npz (bench/_benchmark_dipy.py csd --peaks-mni): (x, y, z, 9) peaks on the MNI template's 1.25 mm grid, LAS voxel
order with MNI's affine signs, each peak in world (RAS) coordinates scaled by its amplitude: what TractSeg's own
`TractSeg -i peaks.nii.gz` gets from MRtrix's sh2peaks after its flip_axis_to_match_MNI_space (a no-op here). The caller
sets TRACTSEG_WEIGHTS_DIR to a directory holding pretrained_weights_tract_segmentation_v3.npz (the Zenodo 3518348
best_weights_ep220.npz) so that nothing is downloaded, and a dead proxy so that any download attempt fails.

The run is TractSeg's default tract segmentation (python_api.run_tractseg: TractSeg_PeakRot4, the 72 TractQuerier+
bundles, three orientations fused, threshold 0.5, bundle-specific postprocessing), on the CPU (TractSeg uses CUDA only).
OUT.npz: masks (72, x, y, z) packed bits, the bundle names, the seconds.
"""
import json, os, sys, time
import numpy as np


def one_openmp():
    """TractSeg's img_utils imports dipy.align.imaffine.AffineMap at module level, for change_spacing_4D only (not on this
    path). Importing it loads DIPY's own libomp beside torch's, and the process aborts (OMP Error #15). A stub stands in
    for that one module, raising if ever called, so only torch's OpenMP runtime is loaded; KMP_DUPLICATE_LIB_OK (unsafe)
    is not used."""
    import types
    m = types.ModuleType("dipy.align.imaffine")

    class AffineMap:
        def __init__(self, *a, **k):
            raise RuntimeError("dipy.align.imaffine is stubbed in this process (bench/_tractseg_run.py one_openmp)")
    m.AffineMap = AffineMap
    sys.modules["dipy.align.imaffine"] = m


def main():
    src, out = sys.argv[1], sys.argv[2]
    one_openmp()
    from tractseg.libs.system_config import SystemConfig as C
    weights = os.path.join(C.WEIGHTS_DIR, "pretrained_weights_tract_segmentation_v3.npz")
    if not os.path.exists(weights):
        raise SystemExit(f"{weights} missing: refusing to let TractSeg download it")
    import torch
    from tractseg.python_api import run_tractseg
    from tractseg.data.dataset_specific_utils import get_bundle_names
    assert not torch.cuda.is_available()
    z = np.load(src)
    peaks = z["peaks"].astype(np.float32)
    t0 = time.time()
    seg = run_tractseg(peaks, output_type="tract_segmentation", nr_cpus=os.cpu_count())
    seconds = round(time.time() - t0, 1)
    names = get_bundle_names("All")[1:]
    assert seg.shape == (*peaks.shape[:3], len(names)), (seg.shape, len(names))
    masks = np.moveaxis(seg > 0.5, -1, 0)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, masks=np.packbits(masks.ravel()), shape=np.array(masks.shape), names=np.array(names),
                        info=np.array(json.dumps({"seconds": seconds, "torch": torch.__version__, "threads": torch.get_num_threads(),
                                                  "weights": os.path.realpath(weights)})))
    os.replace(tmp, out)
    print(json.dumps({"seconds": seconds, "voxels per bundle (median)": int(np.median(masks.reshape(len(names), -1).sum(1)))}), flush=True)


if __name__ == "__main__":
    main()
