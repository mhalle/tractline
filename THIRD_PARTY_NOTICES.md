# Third-party notices

tractline's code and files are its own, licensed under the Apache License 2.0 (`LICENSE`), except the
two files under Redistributed. Several modules implement published methods independently, following the original software's
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


### TractCloud's atlas centroid (3D Slicer license, Part B)

`labelers/tractcloud_data/HCP_mass_center.npy` (308 bytes: 15 points of a mean streamline that the TractCloud
labeler centers its inputs on) is TractCloud's file, unmodified, from its v1.0.0 release
(github.com/SlicerDMRI/TractCloud, `TrainData_800clu800ol.tar.gz`; Xue et al., MICCAI 2023). It was computed by its
authors from Human Connectome Project tractography. TractCloud is distributed under the 3D Slicer Contribution and
Software License Agreement, whose Part B requires the following to accompany any copy:

All or portions of this licensed product (such portions are the "Software") have been obtained under license from
The Brigham and Women's Hospital, Inc. and are subject to the following terms and conditions:

```
PART B. DOWNLOADING AGREEMENT - License from Brigham with Right to
Sublicense ("Software License").

1. As used in this Software License, "you" means the individual
   downloading and/or using, reproducing, modifying, displaying and/or
   distributing the Software and the institution or entity which
   employs or is otherwise affiliated with such individual in
   connection therewith. The Brigham and Women's Hospital,
   Inc. ("Brigham") hereby grants you, with right to sublicense, with
   respect to Brigham's rights in the software, and data, if any,
   which is the subject of this Software License (collectively, the
   "Software"), a royalty-free, non-exclusive license to use,
   reproduce, make derivative works of, display and distribute the
   Software, provided that:

(a) you accept and adhere to all of the terms and conditions of this
Software License;

(b) in connection with any copy of or sublicense of all or any portion
of the Software, all of the terms and conditions in this Software
License shall appear in and shall apply to such copy and such
sublicense, including without limitation all source and executable
forms and on any user documentation, prefaced with the following
words: "All or portions of this licensed product (such portions are
the "Software") have been obtained under license from The Brigham and
Women's Hospital, Inc. and are subject to the following terms and
conditions:"

(c) you preserve and maintain all applicable attributions, copyright
notices and licenses included in or applicable to the Software;

(d) modified versions of the Software must be clearly identified and
marked as such, and must not be misrepresented as being the original
Software; and

(e) you consider making, but are under no obligation to make, the
source code of any of your modifications to the Software freely
available to others on an open source basis.

2. The license granted in this Software License includes without
   limitation the right to (i) incorporate the Software into
   proprietary programs (subject to any restrictions applicable to
   such programs), (ii) add your own copyright statement to your
   modifications of the Software, and (iii) provide additional or
   different license terms and conditions in your sublicenses of
   modifications of the Software; provided that in each case your use,
   reproduction or distribution of such modifications otherwise
   complies with the conditions stated in this Software License.

3. This Software License does not grant any rights with respect to
   third party software, except those rights that Brigham has been
   authorized by a third party to grant to you, and accordingly you
   are solely responsible for (i) obtaining any permissions from third
   parties that you need to use, reproduce, make derivative works of,
   display and distribute the Software, and (ii) informing your
   sublicensees, including without limitation your end-users, of their
   obligations to secure any such required permissions.

4. The Software has been designed for research purposes only and has
   not been reviewed or approved by the Food and Drug Administration
   or by any other agency. YOU ACKNOWLEDGE AND AGREE THAT CLINICAL
   APPLICATIONS ARE NEITHER RECOMMENDED NOR ADVISED. Any
   commercialization of the Software is at the sole risk of the party
   or parties engaged in such commercialization. You further agree to
   use, reproduce, make derivative works of, display and distribute
   the Software in compliance with all applicable governmental laws,
   regulations and orders, including without limitation those relating
   to export and import control.

5. The Software is provided "AS IS" and neither Brigham nor any
   contributor to the software (each a "Contributor") shall have any
   obligation to provide maintenance, support, updates, enhancements
   or modifications thereto. BRIGHAM AND ALL CONTRIBUTORS SPECIFICALLY
   DISCLAIM ALL EXPRESS AND IMPLIED WARRANTIES OF ANY KIND INCLUDING,
   BUT NOT LIMITED TO, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR
   A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
   BRIGHAM OR ANY CONTRIBUTOR BE LIABLE TO ANY PARTY FOR DIRECT,
   INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY ARISING IN ANY WAY
   RELATED TO THE SOFTWARE, EVEN IF BRIGHAM OR ANY CONTRIBUTOR HAS
   BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. TO THE MAXIMUM
   EXTENT NOT PROHIBITED BY LAW OR REGULATION, YOU FURTHER ASSUME ALL
   LIABILITY FOR YOUR USE, REPRODUCTION, MAKING OF DERIVATIVE WORKS,
   DISPLAY, LICENSE OR DISTRIBUTION OF THE SOFTWARE AND AGREE TO
   INDEMNIFY AND HOLD HARMLESS BRIGHAM AND ALL CONTRIBUTORS FROM AND
   AGAINST ANY AND ALL CLAIMS, SUITS, ACTIONS, DEMANDS AND JUDGMENTS
   ARISING THEREFROM.

6. None of the names, logos or trademarks of Brigham or any of
   Brigham's affiliates or any of the Contributors, or any funding
   agency, may be used to endorse or promote products produced in
   whole or in part by operation of the Software or derived from or
   based on the Software without specific prior written permission
   from the applicable party.

7. Any use, reproduction or distribution of the Software which is not
   in accordance with this Software License shall automatically revoke
   all rights granted to you under this Software License and render
   Paragraphs 1 and 2 of this Software License null and void.

8. This Software License does not grant any rights in or to any
   intellectual property owned by Brigham or any Contributor except
   those rights expressly granted hereunder.
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
- **SynthStrip** (Hoopes, Mora, Dalca, Fischl & Hoffmann, NeuroImage 2022; FreeSurfer's `mri_synthstrip`, under the
  FreeSurfer Software License Agreement): `skullstrip.py` is an independent code base for its network and test-time
  conform, written from the paper and the tensor shapes of its released weights; it is not copied or translated from `mri_synthstrip` or
  from ports of it. A port (mhalle/synthstrip-torch) is run in `bench/` only, as a test reference.
- **TractCloud** (3D Slicer Contribution and Software License Agreement, BSD style;
  github.com/SlicerDMRI/TractCloud; Xue et al., MICCAI 2023): `labelers/tractcloud.py`'s `MatmulDGCNN`
  re-expresses its network's arithmetic for the CPU, and `trained_context` builds the context its released
  model was trained with.

## Used at run time, not distributed

- **TractCloud's code and weights**: the optional TractCloud labeler imports its code and loads its
  weights from the user's `$TRACTOGRAPHY_DATA` or tractline's cache (downloaded from its GitHub on first use), under
  its own license; its atlas centroid ships with tractline (Redistributed, above).
- **RapidParc's weights**: loaded from `$TRACTOGRAPHY_DATA`, under its own license.
- **SynthStrip's weights** (`synthstrip.1.pt` and `synthstrip.nocsf.1.pt`, for `--mask synthstrip` and
  `--mask synthstrip-nocsf`): downloaded by the user from its authors'
  page (surfer.nmr.mgh.harvard.edu/docs/synthstrip), which offers them under the MIT license or CC BY 4.0;
  tractline uses them under MIT and makes no copies of them (the license's notice travels with copies).

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
