# Third-party notices

tractline's code and files are its own, licensed under the Apache License 2.0 (`LICENSE`), except the
one file under Redistributed. Several modules implement published methods independently, following the original software's
behavior so they can be checked against it; they are not distributed under the originals' licenses.
This file credits those originals and records their licenses for reference, lists the third-party
material the package redistributes with the notices its licenses require, and names the software and
data used at run time or in the bench but not distributed here.

## Redistributed

### RapidParc's tract scheme (BSD 3-Clause)

`labelers/scheme_43.json` (the 43 class names - 42 tracts and Other - and the 1,600 -> 43 cluster mapping) is taken from
RapidParc's release files (github.com/MedVisBonn/RapidParc; the same scheme as TractCloud's). Its
license asks that this notice accompany it:

```
BSD 3-Clause License

Copyright (c) 2026, Visualization and Medical Image Analysis Group, University of Bonn

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```


## Independent implementations (tractline's code; the originals credited)

- **RapidParc** (BSD 3-Clause, the notice above): `labelers/rapidparc.py` implements its inference
  (von Bornhaupt et al., Imaging Neuroscience 2026), with parameter names matching its released weights so
  they load directly. The weights are not distributed here; the user downloads them from its release.
- **UKF Tractography** (UKF Tractography Contribution and Software License Agreement, BSD style; The
  Brigham and Women's Hospital; github.com/pnlbwh/ukftractography, commit 2d2b661): `ukf.py`, `ukf_metal.py`
  and `ukf_triton*.py` implement its two-tensor unscented Kalman filter tractography as the ORG atlas
  configures it.
- **DIPY** (BSD 3-Clause; dipy developers): `mask.py` implements DIPY's `median_otsu` - the repeated
  median filter, Otsu's threshold (Otsu, IEEE Trans. SMC 1979) with DIPY's histogram conventions, and the
  masking - to match it voxel for voxel. DIPY's Otsu function is itself scikit-image's (BSD 3-Clause; the
  scikit-image team).
- **FSL topup** (FSL license; FMRIB, University of Oxford): `susceptibility.py` implements the
  susceptibility model of Andersson, Skare & Ashburner (NeuroImage 2003) and follows the multi-resolution
  schedule of topup's `b02b0.cnf` (knot spacing, subsampling, smoothing, iterations, regularization). FSL
  itself is used only in `bench/`, as a test reference.
- **TractCloud** (3D Slicer Contribution and Software License Agreement, BSD style;
  github.com/SlicerDMRI/TractCloud; Xue et al., MICCAI 2023): `labelers/tractcloud.py`'s `MatmulDGCNN`
  re-expresses its network's arithmetic for the CPU, and `trained_context` builds the context its released
  model was trained with.

## Used at run time, not distributed

- **TractCloud's code and weights**: the optional TractCloud labeler imports its code and loads its
  weights from the user's `$TRACTOGRAPHY_DATA`, under its own license.
- **RapidParc's weights**: loaded from `$TRACTOGRAPHY_DATA`, under its own license.

## Data and results in `bench/`

- **OpenNeuro ds001226** (BTC_preop; CC0): the 12 patients' scans; not in the repository; results
  derived from them are.
- **Human Connectome Project**: TractCloud's test split and HCP test tractogram (from its release) are
  HCP-derived and used under the HCP data-use terms; they are not in the repository. Results computed
  from them (`accuracy_tractcloud_test.json`, `resample.json`, ...) are; the HCP's terms ask for this
  acknowledgment: Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium
  (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH
  Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell
  Center for Systems Neuroscience at Washington University. The committed results computed from them are
  aggregate numbers only (accuracies, label counts, differences, timings: `accuracy_tractcloud_test.json`,
  `infer_opt*.json`, `resample.json`, under 2 kB each) - no images, streamlines or per-subject values.
- **Stanford HARDI** (via DIPY's `stanford_hardi` fetcher): used in `bench/` for timing; not in the
  repository.
- **OpenNeuro ds005123** (Smith, Sharp, Dachs et al.; CC0; doi:10.18112/openneuro.ds005123.v1.1.3): a slice of
  12 subjects (each diffusion series' two leading b0s and its spin-echo field maps, `fetch_ds005123.py`) for
  the phase-encoding and pairing tests; not in the repository; aggregate results are.
- **dcm2niix's validation sets** (dcm_qa_polar, dcm_qa_ge, dcm_qa_trt; github.com/neurolabusc): GE DICOM for
  the header tests; not in the repository.

## Tools used by the bench, not distributed

- **dcm2niix** (Chris Rorden; BSD 2-Clause, "for research purposes only and is not a clinical tool"): run as
  a program on DICOM; its source, read for `docs/dcm2niix-dmri-map.md`, which cites its rules by line.
- **FSL** (topup, applytopup; FMRIB, University of Oxford; FSL license, non-commercial): the reference the
  correction is checked against.
