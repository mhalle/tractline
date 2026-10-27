"""Two motion_eddy runs' parameters compared (record() as JSON, or a modal_cuda result holding one under "record"):
per volume the largest difference in translation (mm), rotation (degrees) and eddy coefficient (voxels), and the QC's
summary figures side by side.

    python bench/motion_eddy/compare_records.py results/motion_eddy/modal_cuda_a10.json <an MPS run's record.json>
"""
import json
import sys
import numpy as np


def compare(rec_a: dict, rec_b: dict) -> dict:
    """Run a's motion and eddy parameters against run b's: per volume, the largest difference in translation
    (mm), rotation (degrees) and eddy coefficient (voxels), and the QC's summary figures side by side."""
    a, b = np.asarray(rec_a["theta"], float), np.asarray(rec_b["theta"], float)
    dt, dr, de = np.abs(a[:, :3] - b[:, :3]), np.abs(a[:, 3:6] - b[:, 3:6]), np.abs(a[:, 6:] - b[:, 6:])
    keys = [k for k in rec_b["qc"] if ("framewise" in k or "edge" in k) and "per volume" not in k]
    return {"translation_mm_max": round(float(dt.max()), 4), "translation_mm_median": round(float(np.median(dt)), 5),
            "rotation_deg_max": round(float(np.degrees(dr).max()), 4),
            "eddy_vox_max": round(float(de.max()), 4) if de.size else None,
            "qc": {k: [rec_a["qc"].get(k), rec_b["qc"].get(k)] for k in keys}}


if __name__ == "__main__":
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    print(json.dumps(compare(a.get("record", a), b.get("record", b)), indent=1))
