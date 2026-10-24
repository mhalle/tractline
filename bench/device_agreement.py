"""Do the devices agree on a real scan? (docs/sota-pipeline-design.md, decision 6: the config hash leaves the device out, so
a calibration map learned on one device is applied on another.) cIIIs04's reference scan through the default pipeline at
b = 2000 (calibration_runs.pipeline_config) on this machine's CPU, against its stored record made on the M2's GPU (MPS):
the record the cohort III calibration map was learned from.

    uv run bench/device_agreement.py [--device cpu] [--workers N]

Compared, streamlines matched by seed voxel (the record's keys): the seeds shared, each shared streamline's winning cluster
and tract, how far its path is from the stored one (the record holds paths to 0.025 mm), the tracking mask (A_mask.npy),
and the per-tract voxel counts (correlation and the presence at the calibration's density threshold, as Dice). Writes
results/device_agreement_cIIIs04.json.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import calibration_runs as CR
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.labelers.base import LUT

HERE = Path(__file__).resolve().parent
SUBJECT = "cIIIs04"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    R = CR.Records(SUBJECT)
    stored = E.RunRecord.load(R.path("A"), logp=False, fibers=True)
    A = CR.load_checked(SUBJECT, *CR.CARD[SUBJECT].ref)
    cfg = CR.pipeline_config()
    t0 = time.time()
    r = P.execute(A, cfg, P.Timer(), device=args.device, workers=args.workers, labeler=P.default_labeler(args.device))
    seconds = round(time.time() - t0, 1)
    new = E.RunRecord.of(r.tractogram, r.labels, A.affine, A.dwi.shape[:3])

    shared, i_new, i_old = np.intersect1d(new.keys, stored.keys, return_indices=True)
    cl_same = new.cluster[i_new] == stored.cluster[i_old]
    tr_same = LUT[new.cluster[i_new]] == LUT[stored.cluster[i_old]]
    far, longer = [], []
    for a, b in zip((new.fibers[i] for i in i_new), (stored.fibers[i] for i in i_old)):
        n = min(len(a), len(b))
        far.append(float(np.abs(np.asarray(a[:n], np.float64) - np.asarray(b[:n], np.float64)).max()) if n else np.inf)
        longer.append(abs(len(a) - len(b)))
    far, longer = np.array(far), np.array(longer)
    mask_old = np.load(R.path("A_mask", ".npy"))
    mask_new = r.tractogram.mask
    cn, co = new.counts.astype(np.float64), stored.counts.astype(np.float64)
    corr = [float(np.corrcoef(cn[t], co[t])[0, 1]) for t in range(42) if co[t].sum() > 0 and cn[t].sum() > 0]
    out = {
        "subject": SUBJECT, "device": args.device, "workers": args.workers or os.cpu_count(), "seconds": seconds,
        "config sha256": r.config["sha256"], "stored": "MPS (M2), calibration_runs records, A.npz",
        "streamlines (new, stored)": [len(new.keys), len(stored.keys)],
        "seeds shared, share of stored": round(len(shared) / len(stored.keys), 5),
        "shared: same winning cluster": round(float(cl_same.mean()), 5),
        "shared: same tract": round(float(tr_same.mean()), 5),
        "shared: path within 0.05 mm (quantization 0.025)": round(float((far <= 0.05).mean()), 5),
        "shared: path more than 1 mm apart": round(float((far > 1.0).mean()), 5),
        "shared: point count differs by more than 2": round(float((longer > 2).mean()), 5),
        "mask voxels (new, stored), differing": [int(mask_new.sum()), int(mask_old.sum()), int((mask_new ^ mask_old).sum())],
        "per-tract voxel counts correlation (42 tracts): min, median": [round(min(corr), 5), round(float(np.median(corr)), 5)],
        "runtime": r.runtime, "git": CR.git_state(), "device_agreement.py sha256": CR.sha256(Path(__file__).resolve()),
    }
    (HERE / "results" / f"device_agreement_{SUBJECT}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
