# tractline: UKF tractography with head motion and eddy current correction

This guide introduces tractline, a pipeline that takes a dMRI scan to two-tensor UKF tractography labeled with the
ORG atlas's tracts, with the scan corrected for susceptibility distortion, head motion and eddy currents on the way.
It explains how to run it, what each step does, and how to read the quality control (QC) outputs the motion and eddy
current correction writes alongside the tractography. It assumes familiarity with UKF tractography, the ORG atlas,
and FSL's topup and eddy, and describes tractline in those terms. The code is at https://github.com/mhalle/tractline.

## 1. What tractline is

tractline runs, in one program, a sequence that is usually assembled from several tools: correction of the dMRI scan
for susceptibility distortion, head motion and eddy currents; two-tensor UKF tractography; and the naming of each
streamline as one of the ORG atlas's tracts. Each step is a reimplementation in PyTorch, checked against the tool it
reproduces, and the whole sequence runs on an Apple GPU, a CUDA GPU, or the CPU. On a laptop GPU (Apple M2), a
105-volume scan takes about two minutes from the raw series to labeled streamlines, of which the motion and eddy
current correction takes 20 to 30 seconds. Section 2 shows how to run it; the rest of the guide describes what each
step does and how to read the QC outputs.

The pipeline is research software. It is not approved for clinical use.

## 2. Running it

tractline installs a command, `tractline`. With uv, it runs without a separate install step: `uvx` builds an
environment for it on first use, keeps it, and reuses it afterwards. This command corrects one dMRI series for
susceptibility distortion, head motion and eddy currents, runs UKF tractography, labels the streamlines and writes them
to a TRX file:

```
uvx --from "tractline[cli] @ git+https://github.com/mhalle/tractline@main" \
    tractline run sub-01/dwi/sub-01_dir-AP_dwi.nii.gz --motion-eddy --trx out/sub-01.trx
```

`@main` runs the latest code; a commit hash in its place pins the version, so a set of scans can be processed
with the same code throughout. The first run downloads PyTorch and the trained weights of the labeler (6.6 MB, from
RapidParc's release, checked against a pinned checksum), so it takes longer than later runs. `tractline fetch`
downloads the weights of every labeler ahead of time, for example before working offline. From a checkout of the
repository, `uv run tractline run ...` does the same.

The TRX file holds every streamline the tracker kept, in RAS millimeters, grouped by tract. For each streamline it
also stores the tract label (one of the 42 tracts or "Other"), the most probable atlas cluster, the probability of
the assigned tract and its margin over the next most probable tract, the length, and the seed. Streamlines shorter than 40 mm are kept but left unlabeled, as the
labelers were trained on longer ones. trx-python reads the file and can convert it to other streamline formats.

Two options add further output. `--qc qc/sub-01` writes the QC files of the motion and eddy current correction
(section 5) into that folder. `--write-corrected` also writes the corrected dMRI series there, with its b-values and
rotated b-vectors, for use with other tools. `--json summary.json` saves a summary of the run.

The reversed phase-encoding series used for the susceptibility field is found from the BIDS sidecars, or can be named
with `--partners`. If none is found, the motion and eddy current correction still runs, without the field, and the
printed note says so. The motion and eddy current correction is off unless `--motion-eddy` is given; without it,
only the susceptibility field is corrected. The eddy current model is quadratic by default; `--motion-eddy linear`
selects the linear model, and `--motion-eddy rigid` corrects head motion alone, with no eddy current model.
`--outliers off` keeps the correction but turns off the replacement of outlier slices. `--shell` sets the b-value
tracked (by default the shell with the most volumes). `--labeler tractcloud` labels with TractCloud in place of
RapidParc (section 3); its code and trained model are downloaded from TractCloud's GitHub on first use (about 42 MB,
at the version tractline was tested with, checked against pinned checksums). `tractline run --help` lists every
option.

The command runs on an Apple GPU when there is one, then on a CUDA GPU, then on the CPU (`--device` chooses). On a
laptop GPU a 105-volume scan takes about two minutes. On the CPU the motion and eddy current correction gives the same
result but takes 12 to 13 minutes for a 105-volume scan. On an Apple GPU, a series too large for the available memory
is refused before the run starts, with a message; this applies to HCP-sized acquisitions on a 16 GB machine. Such a
series can be run on the CPU (`--device cpu`), or tried on the GPU anyway with `--no-memory-check`. The
correction has not yet been tested on CUDA.

The same run from Python:

```python
from tractline import bids, pipeline as P, motion_eddy as ME
s = bids.load("sub-01/dwi/sub-01_dir-AP_dwi.nii.gz")
corr, tractogram, labels = P.run(s, None, P.Timer(), shell=2000.0, motion_eddy=True, trx="out/sub-01.trx")
ME.write_qc(corr.motion_eddy, "qc/sub-01")            # the QC files, if wanted
print(corr.note, corr.warnings)
```

## 3. The steps

The input is a dMRI series in BIDS form (NIfTI with its `.bval`, `.bvec` and JSON sidecar; DICOM through dcm2niix).
The series acquired with the opposite phase-encoding direction, used for the susceptibility field, is found from the
sidecars (`B0FieldIdentifier`, `IntendedFor`, or a series in the same folder acquired the other way) and checked: the
same phase-encoding axis, both polarities, the same shim settings, and a stated or safely assumed readout time. A
scan without a usable pair is processed without susceptibility correction, and the output says why.

**Susceptibility field.** The field is estimated from the b0 images of both phase-encoding directions with topup's
model (Andersson et al. 2003): a B-spline field fitted by Gauss-Newton on a coarse-to-fine schedule, with the
motion of the b0 images estimated alongside it. Its result agrees with topup's, and it takes about 15 seconds on an
Apple GPU, against about 11 minutes for topup on the same machine.

**Head motion and eddy currents.** For every volume, the correction estimates the head motion (three translations,
three rotations) and the distortion caused by eddy currents, modeled as a linear or quadratic function of position
along the phase-encoding axis, as in FSL eddy (Andersson and Sotiropoulos 2016). Each volume is registered to a
prediction of it from the other volumes of its shell, so that volumes of different diffusion contrast can be aligned.
The b0 images are predicted from the mean of the other b0 images. Slices that have lost signal are replaced by their
prediction, as with eddy's `--repol` option (Andersson et al. 2016). Section 4 describes this step in more detail.

**One resampling.** Each volume is then resampled once, with its susceptibility displacement, its motion and its eddy
current distortion combined into a single map, and scaled by the map's Jacobian. The b-vectors are rotated with the
head.

**Tractography input.** A brain mask is computed from the mean b0 image (DIPY's `median_otsu`, reproduced exactly).
One shell is tracked, together with the b0 images: `run_pipeline.py` picks the shell with the most volumes unless
`--shell` names another, and from Python the shell is given as `shell=`.

**UKF tractography.** The tracker is the two-tensor unscented Kalman filter of Malcolm et al. (2010), reimplemented
from pnlbwh/ukftractography with the settings used to build the ORG atlas (two tensors, its seeding and stopping
thresholds and record length) and the binary's own seeding, every voxel of the mask. In double precision it follows
the binary's arithmetic. The GPU kernels run in single precision; their streamlines differ from double precision far
less than the streamlines from two acquisitions of the same subject differ from each other.

**Tract labels.** Each streamline of 40 mm or more is assigned to one of the ORG atlas's 800 fiber clusters (Zhang et
al. 2018), which are then mapped to 42 named tracts and "Other", the same mapping TractCloud uses. The default
labeler is RapidParc (Bisten et al. 2026), which on TractCloud's labeled test set reaches 94.5 % tract accuracy;
TractCloud itself (Xue et al. 2023) is available as an alternative, with its released network and weights. tractline
runs it with the context its model was trained with: each streamline with its 19 nearest among 10,000 streamlines
drawn from the whole tractogram, and 500 global streamlines. TractCloud's packaged inference uses a smaller context
(80 global streamlines, local neighbors from consecutive chunks of about 10,000); on TractCloud's labeled test set
the trained context reaches 92.0 % tract accuracy and the packaged one 86.6 %. The output can be written as a TRX
file with each streamline's tract label and cluster probabilities.

## 4. Head motion and eddy current correction

The correction follows the approach of FSL eddy, with its own fit. All volumes are fitted together on the GPU by
Gauss-Newton. Each volume's target is a leave-one-out prediction from the other volumes of its shell (regularized
spherical harmonics of order 6), compared with the data in the volume's own acquired space, so that noisy data are
never interpolated before the fit. The prediction is rebuilt from the current estimates at every step.

Three conventions follow eddy's implementation. The constant part of the eddy current field cannot be told apart from
a translation along the phase-encoding axis; as in eddy, the part of that shift that varies linearly with the
diffusion gradient is assigned to eddy currents, and the rest to head motion. The first b0 image is the reference
position, as it is the frame of the susceptibility field. And the corrected output is masked to the voxels that every
volume covers.

Two safeguards protect the fit from individual bad volumes. Slices that have lost signal, through motion during the
acquisition of a slice or through cardiac pulsation, are detected by comparing each slice with its prediction within
the shell: a slice whose mean residual is more than four robust standard deviations below the shell's, and that has
lost at least 10 % of its predicted signal, is replaced by its prediction during the fit and, by default, in the
output. Volumes that cannot be registered at all, such as an empty volume or one whose signal is far from the rest of
its shell, are excluded from the correction. An excluded volume is kept in the corrected series without motion
correction, but it is not used for tractography.

On real scans where both were run, this correction and eddy agreed to within about 0.1 voxel at the edge of the brain.
On one scan corrected with the same susceptibility field, the corrected images differed from eddy's by 2 % of the
brain's mean signal (median over volumes), and the rotated b-vectors by 0.14 degrees. eddy took 59 minutes on six CPU
threads for that 55-volume scan; this correction takes about 20 seconds on an Apple GPU.

## 5. The QC files

With `--qc`, the motion and eddy current correction writes three files alongside the
tractography.

`motion_eddy_qc.tsv` has one row per volume, in the order of acquisition. Its columns are:

| column | meaning |
|---|---|
| `volume`, `b` | the volume's index and its nominal b-value (0 for b0 images) |
| `tx_mm`, `ty_mm`, `tz_mm` | translations, in mm |
| `rx_deg`, `ry_deg`, `rz_deg` | rotations, in degrees |
| `rigid_edge_rms_mm`, `rigid_edge_max_mm` | how far head motion moved the edge of the brain in this volume (RMS and maximum over the edge) |
| `eddy_edge_rms_mm`, `eddy_edge_max_mm` | the same for the eddy current distortion |
| `framewise_displacement_mm` | the change in head position from the previous volume, as defined by Power et al. (2012) |
| `residual_rms`, `residual_z` | how well the corrected volume matches its prediction from the other volumes, and that value as a robust z-score within its shell |
| `outlier_slices` | the number of slices replaced because of signal loss |
| `left_out` | 1 if the volume was excluded from the correction |
| `eddy_e0_vox` ... | the eddy current coefficients (voxels along the phase-encoding axis) |

`motion_eddy_qc.json` summarizes the scan: the model and reference frame, whether the susceptibility field was used,
the mean and maximum framewise displacement, the median and largest displacement at the brain's edge from motion and
from eddy currents, the outlier slices by volume, the excluded volumes with the reason for each, and the run time.

`motion_eddy_qc.png` plots the same per-volume measures against volume index, with the b0 volumes marked as gray
bands.

The command also prints warnings, which `--json` saves in the summary (`corr.warnings` from Python): excluded volumes
with their reasons, outlier slices, volumes whose residual z-score exceeds 4, and any non-numeric values found in the
data.

## 6. Reading the QC

**Head motion.** The framewise displacement and the rigid displacement at the brain's edge are the main motion
measures. On ON-Harmony subject 16974, a low-motion scan acquired on a Siemens Prisma (105 volumes, 2 mm isotropic,
b = 1000 and 2000 s/mm²), the mean framewise displacement was 0.26 mm (maximum 0.58 mm), and the median rigid
displacement at the brain's edge was 0.24 mm. On ON-Harmony subject 10975, a session with substantial motion, the mean
framewise displacement was 0.52 mm (maximum 1.24 mm), and the median rigid displacement at the brain's edge was
2.2 mm (maximum 4.1 mm). The two measures answer different questions. Framewise displacement measures the change
from one volume to the next, and is low when the head drifts slowly. The displacement at the brain's edge measures
the distance from the reference position, the first b0 image, and shows such a drift.

One property of the method affects the framewise displacement. Because a translation along the phase-encoding axis
and a constant eddy current field shift the image the same way, their split (section 4) leaves a remainder in the
phase-encoding translation that fluctuates by 0.1 to 0.3 mm from one volume to the next, even when the head is still.
On low-motion scans it can account for more than half of the framewise displacement. A pattern of small, alternating
steps in the phase-encoding translation (`ty_mm` for anterior-posterior phase encoding) is this effect, not head
motion.

**Eddy currents.** The RMS eddy current displacement at the brain's edge is usually below 1 mm. On 16974 its median
over volumes was 0.51 mm, with a largest value of 1.0 mm, and it was larger at b = 2000 than at b = 1000 s/mm². The
maximum at any single point can be several times larger, because the quadratic terms grow toward the corners of the
brain; the summary therefore reports the RMS first.

**Outlier slices.** A few outlier slices in a scan are common. On ten scans from four datasets, the number ranged from
0 to 13, and most had fewer than 10. Clusters in the lowest slices across many volumes are usually cardiac pulsation.
A volume in which several slices are flagged, often in pairs separated by the number of slices divided by the
multiband factor, has lost signal through motion during its acquisition. In both cases the replaced slices are filled
in from the other volumes, so the corrected data remain usable, but many outlier slices in a scan point to
substantial motion.

**Excluded volumes.** An excluded volume could not be corrected. The reason is given in the JSON summary and in the
warnings: a mean signal far from the rest of its shell (an empty or corrupted volume), a correction that failed to
converge, or an estimated displacement of more than 30 mm. Excluded volumes are not used for tractography. A scan
with more than a few excluded volumes deserves a manual check.

**Poorly fitting volumes.** A residual z-score above 4 means that the corrected volume differs from its prediction
more than the other volumes of its shell do. It can point to motion within the volume, an artifact the correction
does not model, or, occasionally, a borderline case on a clean scan.

## 7. Known limitations

The motion and eddy current correction is the newest part of the pipeline, and the following limitations are known.
They are documented in more detail in `docs/gpu-motion-eddy-design.md` and in `bench/results/NOTES.md`.

- A bright artifact present at the same place in every volume, such as the signal pile-up left by a strong
  susceptibility field that was not corrected, biases the registration by about half a voxel. Robust weighting would
  address it and has not been implemented.
- On synthetic data in which every voxel of the brain has the same strong anisotropy, the registration drifts by about
  0.3 voxel, an effect of the leave-one-out prediction. Real scans agreed with eddy, but eddy may share the bias.
- The thresholds that decide which slices are outliers and which volumes are excluded were set on a small number of
  scans (ON-Harmony, MASiVar, OpenNeuro ds001226, and the SHORELine simulations).
- Slices acquired together in a multiband acquisition are judged separately, not as a group as in eddy's `--mb`.
- Slice-to-volume motion, and the interaction of motion with the susceptibility field, are not modeled.

## References

Andersson JLR, Skare S, Ashburner J. How to correct susceptibility distortions in spin-echo echo-planar images:
application to diffusion tensor imaging. *NeuroImage*. 2003;20(2):870-888.

Andersson JLR, Sotiropoulos SN. An integrated approach to correction for off-resonance effects and subject movement
in diffusion MR imaging. *NeuroImage*. 2016;125:1063-1078.

Andersson JLR, Graham MS, Zsoldos E, Sotiropoulos SN. Incorporating outlier detection and replacement into a
non-parametric framework for movement and distortion correction of diffusion MR images. *NeuroImage*.
2016;141:556-572.

Bastiani M, Cottaar M, Fitzgibbon SP, et al. Automated quality control for within and between studies diffusion MRI
data using a non-parametric framework for movement and distortion correction. *NeuroImage*. 2019;184:801-812.

Bisten et al. RapidParc: a global-context transformer for parallel, accurate, and lesion-robust tractogram
parcellation. *Imaging Neuroscience*. 2026.

Cetin-Karayumak S, Zhang F, Zurrin R, Billah T, Zekelman L, Makris N, Pieper S, O'Donnell LJ, Rathi Y. Harmonized
diffusion MRI data and white matter measures from the Adolescent Brain Cognitive Development Study. *Scientific
Data*. 2024;11:249.

Malcolm JG, Shenton ME, Rathi Y. Filtered multitensor tractography. *IEEE Transactions on Medical Imaging*.
2010;29(9):1664-1675.

Power JD, Barnes KA, Snyder AZ, Schlaggar BL, Petersen SE. Spurious but systematic correlations in functional
connectivity MRI networks arise from subject motion. *NeuroImage*. 2012;59(3):2142-2154.

Xue T, Chen Y, Zhang C, Golby AJ, Makris N, Rathi Y, Cai W, Zhang F, O'Donnell LJ. TractCloud: registration-free
tractography parcellation with a novel local-global streamline point cloud representation. *MICCAI*. 2023.

Zhang F, Wu Y, Norton I, Rigolo L, Rathi Y, Makris N, O'Donnell LJ. An anatomically curated fiber clustering white
matter atlas for consistent white matter tract parcellation across the lifespan. *NeuroImage*. 2018;179:429-447.
