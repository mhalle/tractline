# Third-party notices (draft)

tractline's own license is not yet chosen. This file lists what the repository takes from others, how,
and under what terms. It is a draft for review before the repository is made public.

## Redistributed or derived in the package (`src/tractline`)

### RapidParc - BSD 3-Clause
github.com/MedVisBonn/RapidParc (v1.0.4 code, v1.0.0 weights). `labelers/rapidparc.py` writes its
inference anew following its code (its parameter names, so its released safetensors load directly);
`labelers/scheme_43.json` (tract names and the 1,600 -> 43 cluster mapping) is taken from its release
files. Its weights are not in the repository; the user downloads them.

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

### UKF Tractography - UKF Tractography Contribution and Software License Agreement (BSD style)
github.com/pnlbwh/ukftractography, commit 2d2b661 (The Brigham and Women's Hospital). `ukf.py`,
`ukf_metal.py` and `ukf_triton*.py` re-implement its two-tensor UKF as the ORG atlas configures it, from
its source; no file of it is copied. Its license is BSD style with Slicer-derived extensions (attribution
and notices preserved; no endorsement); full text: github.com/pnlbwh/ukftractography/blob/master/LICENSE.txt.
The binary used as the reference in `bench/` is not in the repository.

### DIPY - BSD 3-Clause; scikit-image - BSD 3-Clause
`mask.py` reproduces DIPY's `median_otsu` (Copyright (c) 2008-2026, dipy developers) and its `otsu`
threshold, itself scikit-image's arithmetic (Copyright 2009-2022 the scikit-image team), in torch, to
match them voxel for voxel. No file is copied. Licenses: github.com/dipy/dipy/blob/master/LICENSE,
github.com/scikit-image/scikit-image/blob/main/LICENSE.txt.

### FSL topup - the model and b02b0.cnf's schedule
`susceptibility.py` implements the susceptibility model of Andersson, Skare & Ashburner (2003), as FSL's
topup does, and uses the level schedule from topup's `b02b0.cnf` (knot spacing, subsampling, smoothing,
iterations, regularization). No FSL code is copied or linked. FSL's license is non-commercial; FSL itself
is used only in `bench/` (topup as a test reference), never by the package. *To review: whether the
b02b0.cnf parameter values need an attribution or permission beyond the citation.*

## Used at run time, not redistributed

### TractCloud - 3D Slicer Contribution and Software License Agreement (BSD style)
github.com/SlicerDMRI/TractCloud. The optional TractCloud labeler imports its code and loads its weights
from the user's `$TRACTOGRAPHY_DATA`; neither is in the repository. `labelers/tractcloud.py`'s
`MatmulDGCNN` re-expresses its network's arithmetic for the CPU, reading its weights; its
`trained_context` builds the context its released model was trained with. Full license text in its repository.

## Data and results in `bench/`

- **OpenNeuro ds001226** (BTC_preop; CC0): the 12 patients' scans; not in the repository; results
  derived from them are.
- **Human Connectome Project**: TractCloud's test split and HCP test tractogram (from its release) are
  HCP-derived and used under the HCP data-use terms; they are not in the repository. Results computed
  from them (`accuracy_tractcloud_test.json`, `resample.json`, ...) are; the HCP's terms ask for this
  acknowledgement: Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium
  (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH
  Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell
  Center for Systems Neuroscience at Washington University. *To review before going public: which
  committed results are HCP-derived, and whether any could identify a subject.*
- **Stanford HARDI** (via DIPY's `stanford_hardi` fetcher): used in `bench/` for timing; not in the
  repository.
