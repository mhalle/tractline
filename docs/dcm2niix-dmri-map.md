# dcm2niix's diffusion knowledge: a map for duckn's `dwmri`

**For:** the duckn convention's authors (dwmri 2.0, `convention-2.0`), from tractline, 2026-10-03.
**Purpose:** dcm2niix is the most complete public record of what scanner vendors put in diffusion DICOM,
and of the rules that turn it into b-values, directions, phase encoding, readout and slice timing. duckn's
`dwmri` should be able to hold every fact those rules produce - and say how each was obtained - even when a
reader takes its input from dcm2niix's BIDS sidecar rather than from DICOM. This is a map of where those
facts are, to read the source against.

**Source:** github.com/rordenlab/dcm2niix, commit 8282dbf (2026-07-24), BSD-2 (Chris Rorden; "for research
purposes only and is not a clinical tool"). The core files are kept at
`$TRACTOGRAPHY_DATA/dicom/dcm2niix-8282dbf/` (`nii_dicom.cpp` 9,774 lines: reading; `nii_dicom_batch.cpp`
15,489: series assembly and the sidecar). Line numbers below are at that commit. Validation data, as
dcm2niix's own comments cite: the `dcm_qa*` repositories (github.com/neurolabusc); three GE ones are cloned
at `$TRACTOGRAPHY_DATA/dicom/` (dcm_qa_polar, dcm_qa_ge, dcm_qa_trt).

## What to check dwmri 2.0 against

For each item: does dwmri (with `acquisition`, provenance 1.1, and tractline's proposal in
`docs/duckn-proposal.md`) have a place for the fact, and for how it was obtained?

### Directions and b-values

| Vendor | Source (nii_dicom.cpp unless noted) | Frame and rule |
|---|---|---|
| standard | 0018,9087 / 0018,9089; enhanced per-frame 5600-5808 | patient (LPS) |
| Siemens | CSA image header (0029,1010) `B_value`, `DiffusionGradientDirection` (1619-1672); else 0019,100C / 0019,100E (6847, 6874) | patient; a negative B_value clamped to 0; Syngo 2004 b0 marker (-1.0001 x3) forces b = 0 (9113) |
| GE | 0043,1039 mod 1e9 (4663); 0019,10BB/BC/BD (5170) | **image frame** (freq, phase, slice); rotated by `geCorrectBvecs` (batch 309-425): plane-dependent flip, Y negated, ROW phase swaps x/y; a shorter vector under the series' maximum b is a lower shell, b scaled by |g|^2 rounded to 5 (batch 371-395) |
| Philips | 2001,1003; 2005,10B0/B1/B2 (7911, 8102) | patient (RL/AP/FH); classic b > 0 with a zero vector is a derived (trace) volume (9655) |
| Canon/Toshiba | parsed from ImageComments "b=...(x,y,z)" (9506-9536) | **image frame** unless 0018,9089 is present; axis swizzle behind a compile switch (9524) |
| UIH | 0065,1009 / 0065,1037 (7352, 7387) | |
| Bruker | 0018,9602 only (8026) | b-matrix recovery commented out (4769-4817) |

Also: trace / ADC / FA volumes (ImageType, 0018,9075 ISOTROPIC) and how they are dropped (batch
3958-3996); the b = 0 threshold (50 Siemens, 6 others: batch 3789); vectors written as `.mvec` when they
come from a b-matrix (batch 4220). dwmri 2.0 §4 (`null` directions, `directionality`) covers trace volumes;
check the rest.

### Phase-encoding polarity

| Vendor | Source | Note |
|---|---|---|
| Siemens | CSA `PhaseEncodingDirectionPositive` (1670); XA: 0021,111C, 1 = positive (7275) | |
| GE | 0018,9034 RectilinearPhaseEncodeReordering, LINEAR = flipped (6705, RX27+); else the 0043,102A binary blob: EPI bit, polarity bit, ky direction, offsets shifting at version 25.002 (8419-8509); `epi_pepolar` alternates per volume, the reversed split off as series + 1000 (9588-9607) | |
| UIH | 0065,1058, **1 = flipped** (opposite to Siemens) (7377) | |
| Philips, Canon | none | the sidecar then has `PhaseEncodingAxis` only (batch 3513-3521) |

The sidecar's `j` / `j-` sign also depends on dcm2niix's own row flip (`isFlipY`, batch 3537-3542). tractline
measured that the **absolute** polarity does not change a topup-style correction (12 patients, 0.0 mm); the
relative polarity of the series in a pair does.

### Readout time and echo spacing (batch)

- Siemens: EES = 1 / (BandwidthPerPixelPhaseEncode x ReconMatrixPE) (3221); BWPPPE from CSA or 0019,1028 /
  XA 0021,1153.
- GE: from 0043,102C echo spacing and ASSET 0043,1083, partial-Fourier rounding; the variables are named
  "NotPhysical" (3244-3262).
- Philips: estimated from WaterFatShift 2001,1022 and EPI factor 2001,1013 with an empirical constant
  (3214-3243).
- TotalReadoutTime = EES x (ReconMatrixPE - 1); UIH: AcquisitionDuration (3472-3475).

Each is stated, derived by a formula, or estimated - tractline's proposal §3.3 (`acquisition.sources`).

### Shim

Siemens: protocol text (ASCCONV, CSA series header) `sGRADSPEC.asGPAData[0].lOffsetX/Y/Z` (or
`sGRADSPEC.lOffsetX/Y/Z`) and `sGRADSPEC.alShimCurrent[0..4]` - eight values (batch 1005-1030). GE:
0043,1002-1004, the linear three (nii_dicom.cpp 8394). On OpenNeuro ds005123 the diffusion series and its
field maps differ in the second-order Siemens terms - a re-shim that invalidates the pairing (tractline
NOTES 2026-10-03). Proposal §3.1.

### Slice timing and multiband (batch 8299-8776, 9551-9954)

Siemens: mosaic `MosaicRefAcqTimes`; 2D from AcquisitionTime; XA from the second volume; a "desperate"
ucMode rescue. GE: synthesized from the protocol block (0025,101B, gzip XML) and the software version
(rules switch at 27.0 R03; multiband below version 26 unsupported); diffusion with gradient cycling gets
none. Multiband inferred from repeated times, overriding the header's (8495-8524). dwmri keeps
`slice_timing` and `slice_dimension` (2.0 §7); check whether "inferred, not stated" needs saying (§3.3).

### Series assembly (batch 11333-12354, 13910-14347)

Mosaic unpacking (nii_dicom.cpp 3332); enhanced multi-frame ordering by DimensionIndexValues (9266-9420),
with Philips and Canon heuristics ("Guessing temporal order", 9330; Canon's TemporalPositionIndex
"INCORRECT", 9301); Siemens VB12/13 `_ep_b` diffusion stacked across series numbers (batch 14051-14058);
splitting echoes, phase, real/imaginary (13708-13909).

### Marked uncertain in the source

GE non-axial and non-HFS (batch 329, 340), GE row phase encoding "untested" (364-365, 404-405), Bruker
"experimental" (481-491), slice-timing midnight crossing "untested" (8371), "VALIDATE SLICETIMING AND
BVECS" (9925-9928), the GE phase-encoding inference behind a debug define (nii_dicom.cpp 8423), "untested
method to detect slice timing for GE" (7440), "Ugly kludge" for Philips classic DTI (9437).

## What a pass over the source could produce for duckn

1. Fields dwmri 2.0 lacks, with the vendor facts that need them (candidates: shim, per-volume polarity,
   value sources - already proposed; slice-timing provenance; partial Fourier and in-plane acceleration
   as stated facts; derived-volume kinds beyond trace).
2. Converter guidance: per vendor, which element gives which `dwmri` field, in which frame, by which rule
   and version - the content of dwmri 2.0 §5 made complete, with dcm2niix's line numbers as the record.
3. A test plan against the `dcm_qa*` sets: a duckn converter's fields against dcm2niix's sidecars.
