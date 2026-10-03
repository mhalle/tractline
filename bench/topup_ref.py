"""The reference susceptibility-distortion correction for PAT16: FSL topup + applytopup, as HCP's
preprocessing used (the conditions TractCloud's training tractography came from). FSL is a test
reference here, not a pipeline dependency (non-commercial license); our own correction is to be
judged against what this writes.

    python bench/topup_ref.py

FSL: the fsl-topup conda package (FSL's channel) in DATA/fsl-env. Inputs, ds001226 PAT16: the AP DWI
(phase encoding j-, 6 b0 among 102 volumes) and the PA pair (j, 2 b0), total readout time from the
BIDS sidecars. topup estimates the off-resonance field from all 8 b0s with FSL's b02b0.cnf;
applytopup corrects the 102 AP volumes with Jacobian modulation (--method=jac: one phase-encoding
direction carries the diffusion weighting).

Writes DATA/ds001226/derived/PAT16/topup/ (b0s.nii.gz, acqparams.txt, topup_* fit, field_hz.nii.gz,
b0_unwarped.nii.gz, dwi_AP_topup.nii.gz) and results/topup_ref.json (commands, versions, times).
"""
import json, os, subprocess, time
from pathlib import Path
import numpy as np, nibabel as nib

HERE = Path(__file__).resolve().parent
TD = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
FSL = TD / "fsl-env"
SRC = TD / "ds001226/sub-PAT16/ses-preop/dwi"
OUT = TD / "ds001226/derived/PAT16/topup"
OUT.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "FSLDIR": str(FSL), "FSLOUTPUTTYPE": "NIFTI_GZ", "PATH": f"{FSL / 'bin'}:{os.environ['PATH']}"}
CNF = FSL / "src/fsl-topup/flirtsch/b02b0.cnf"

ap = nib.load(SRC / "sub-PAT16_ses-preop_acq-AP_dwi.nii.gz")
pa = nib.load(SRC / "sub-PAT16_ses-preop_acq-PA_dwi.nii.gz")
bval = np.loadtxt(SRC / "sub-PAT16_ses-preop_acq-AP_dwi.bval")
side = {d: json.loads((SRC / f"sub-PAT16_ses-preop_acq-{d}_dwi.json").read_text()) for d in ("AP", "PA")}
pe = {"j": "0 1 0", "j-": "0 -1 0", "i": "1 0 0", "i-": "-1 0 0"}
b0_ap = np.flatnonzero(bval < 50)
A, Pv = np.asarray(ap.dataobj), np.asarray(pa.dataobj)
b0s = np.concatenate([A[..., b0_ap], Pv], axis=-1).astype(np.float32)
nib.save(nib.Nifti1Image(b0s, ap.affine, ap.header), OUT / "b0s.nii.gz")
rows = [f"{pe[side['AP']['PhaseEncodingDirection']]} {side['AP']['TotalReadoutTime']}"] * len(b0_ap) + \
       [f"{pe[side['PA']['PhaseEncodingDirection']]} {side['PA']['TotalReadoutTime']}"] * Pv.shape[-1]
(OUT / "acqparams.txt").write_text("\n".join(rows) + "\n")

def run(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=OUT)
    if r.returncode:
        raise SystemExit(f"{cmd[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return round(time.time() - t0, 1)

cmd_topup = [str(FSL / "bin/topup"), "--imain=b0s", "--datain=acqparams.txt", f"--config={CNF}", "--out=topup",
             "--fout=field_hz", "--iout=b0_unwarped"]
t_topup = run(cmd_topup)
cmd_apply = [str(FSL / "bin/applytopup"), f"--imain={SRC / 'sub-PAT16_ses-preop_acq-AP_dwi.nii.gz'}", "--inindex=1",
             "--datain=acqparams.txt", "--topup=topup", "--method=jac", "--out=dwi_AP_topup"]
t_apply = run(cmd_apply)

field = np.asarray(nib.load(OUT / "field_hz.nii.gz").dataobj)
trt = side["AP"]["TotalReadoutTime"]
shift = field * trt                                                     # voxels along the phase-encoding axis
version = subprocess.run([str(FSL / "bin/topup"), "--version"], env=env, capture_output=True, text=True)
res = {"fsl": "fsl-topup (FSL conda channel), " + (version.stdout or version.stderr).strip().splitlines()[0] if (version.stdout or version.stderr) else "fsl-topup",
       "config": "b02b0.cnf", "b0_ap": int(len(b0_ap)), "b0_pa": int(Pv.shape[-1]), "total_readout_time_s": trt,
       "commands": {"topup": " ".join(c.replace(str(FSL), "$FSL") for c in cmd_topup),
                    "applytopup": " ".join(c.replace(str(FSL), "$FSL").replace(str(SRC), "$SRC") for c in cmd_apply)},
       "seconds": {"topup": t_topup, "applytopup": t_apply},
       "field_hz_quantiles_1_50_99": [round(float(v), 1) for v in np.quantile(field, [0.01, 0.5, 0.99])],
       "shift_voxels_abs_99th_and_max": [round(float(np.quantile(np.abs(shift), 0.99)), 2), round(float(np.abs(shift).max()), 2)]}
print(json.dumps(res, indent=1))
(HERE / "results" / "topup_ref.json").write_text(json.dumps(res, indent=1))
