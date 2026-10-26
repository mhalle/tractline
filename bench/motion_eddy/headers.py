"""Header check for docs/gpu-motion-eddy-design.md section 6.3: shape, voxel size, type and sidecar fields of one scan per protocol."""
import glob, os, json, nibabel as nib
from tractline.data import DATA
D = str(DATA)
pats=[f"{D}/ds003416/sub-cIIs01/ses-s1Bx1/dwi/*b2000n96*APP*dwi.nii.gz",
      f"{D}/ds003416/sub-cIIs01/ses-s1Bx1/dwi/*b1000n32*APP*dwi.nii.gz",
      f"{D}/ds003416/sub-cIIs01/ses-s1Ax1/dwi/*b2000n96*dwi.nii.gz",
      f"{D}/ds004712/sub-*/ses-*/dwi/*dir-AP*dwi.nii.gz"]
for p in pats:
    fs=sorted(glob.glob(p))[:1]
    for f in fs:
        h=nib.load(f).header
        sc=f.replace(".nii.gz",".json"); meta={}
        if os.path.exists(sc):
            j=json.load(open(sc)); meta={k:j.get(k) for k in ("PhaseEncodingDirection","TotalReadoutTime","SliceTiming","MultibandAccelerationFactor","RepetitionTime","Manufacturer")}
            if meta.get("SliceTiming"): meta["SliceTiming"]=f"{len(meta['SliceTiming'])} entries"
        print(os.path.basename(f), h.get_data_shape(), [round(float(z),3) for z in h.get_zooms()], h.get_data_dtype(), meta)
