"""The ISMRM 2015 Tractography Challenge phantom (Zenodo 10.5281/zenodo.572345, CC BY 4.0; Maier-Hein et al. 2017) as
the pipeline takes it: the version with a reversed phase-encoding b0, so that the field correction runs as on a real
scan (docs/robustness-plan.md, step 4; docs/validation-datasets.md, "ISMRM 2015 Tractography Challenge").

Under DATA/phantoms/ismrm2015 (each archive extracted into its own directory, as downloaded):
  challenge_data/extracted/ISMRM_2015_Tracto_challenge_data_with_reversed_phase/   the challenge data:
      Diffusion_WITH_REVERSEPHASE.nii.gz (.bvals, .bvecs)  90 x 108 x 90 at 2 mm, 34 volumes: the b0 (volume 0), the
                                     reversed b0 (volume 1, per the archive's README), 32 directions at b = 1000
      fmap_Hz.nii.gz                 "the field map corresponding to Diffusion.nii.gz, in Hertz" (README), on the DWI's grid
      T1.nii.gz                      the T1-like image, 1 mm, the same world space
  scoring_renauld2023/extracted/     the 2023 ROI scoring (Renauld et al. 2023): bundles/, ROI/, the two config files, t1.nii.gz

The series as the pipeline takes it: the DWI is volume 0 and the 32 directions (33 volumes); the field's b0s are volume
0, then volume 1, phase-encoded the other way along the image's y axis (README: "Phase direction: Y"). Three choices
the archive does not state, decided from the data (bench/robust_phantom.py check; NOTES, step 4):
  - the gradients' x sign (GRADIENT_X): the .bvecs are read as FSL's convention as prep.prepare reads every scan
    (image axes, x flipped when the affine's determinant is positive, as here), then x flipped again or not;
  - the phase-encoding sign (PE_SIGN) along j: the one whose field has fmap_Hz's sign. The correction itself does not
    depend on it (the field and the sign flip together); the comparison with fmap_Hz does;
  - the total readout time (READOUT_S): the README gives "Dwell time = 1ms" and no acceleration or partial Fourier.
    In Fiberfox the dwell time is the time to read one line of k-space (its tLine), so the phase-encoding lines (108,
    the image's y size) span 108 lines x 1 ms; FSL's total readout time, the time from the first to the last line's
    center, is (108 - 1) x 1 ms = 0.107 s. TE = 108 ms, the same 108 lines x 1 ms, is consistent with it. Uncertain:
    Fiberfox's own convention could be 108 ms (a 1 % difference) or differ by more; fmap_Hz, if it is the simulated
    field, settles it (bench/robust_phantom.py check fits the scale of the estimated field against it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np, nibabel as nib
from tractline.data import DATA

ROOT = DATA / "phantoms" / "ismrm2015"
CHALLENGE = ROOT / "challenge_data" / "extracted" / "ISMRM_2015_Tracto_challenge_data_with_reversed_phase"
SCORING = ROOT / "scoring_renauld2023" / "extracted"
REFERENCE = SCORING / "t1.nii.gz"            # the scorer's reference space (1 mm, LPS voxel order)
FMAP_HZ = CHALLENGE / "fmap_Hz.nii.gz"
SHELL = 1000.0                               # the one shell (b 999.998 and 1000 as stated)
PE_LINES = 108                               # the phase-encoding lines: the image's y size
DWELL_S = 1e-3                               # README: "Dwell time = 1ms" (Fiberfox's tLine: one k-space line)
READOUT_S = (PE_LINES - 1) * DWELL_S         # 0.107 s, FSL's total readout time (module docstring)
PE_SIGN = 1.0                                # volume 0 phase-encoded along +j: the sign under which the estimated field has
                                             # fmap_Hz's sign (r 0.965; -1 gives -0.965). The correction is the same either way
GRADIENT_X = -1.0                            # the .bvecs' x flipped against FSL's convention: the tensor then follows the
                                             # ground-truth bundles (median 6.6 deg, against 38.8 deg as FSL reads them)


@dataclass
class Subject:
    name: str                    # "ismrm2015"
    dwi: np.ndarray              # (X, Y, Z, 33) int16: volume 0 (b0) and the 32 directions, as stored
    affine: np.ndarray           # 4x4, voxel -> RAS mm
    vox: np.ndarray              # (3,) mm
    bval: np.ndarray             # (33,)
    bvec: np.ndarray             # (3, 33), FSL convention (x as GRADIENT_X makes it)
    pe_axis: int                 # 1 (the image's y)
    pe_sign: float               # the DWI's sign along it (PE_SIGN)
    readout_s: float             # READOUT_S
    b0s: np.ndarray              # (X, Y, Z, 2) float64: volume 0, then the reversed b0 (volume 1)
    pe_vectors: np.ndarray       # (2, 3)
    notes: list = field(default_factory=list)

    @property
    def b0_index(self):
        return np.flatnonzero(self.bval < 50)


def load(gradient_x=GRADIENT_X, pe_sign=PE_SIGN, readout_s=READOUT_S) -> Subject:
    """The phantom as a pipeline subject; the keywords are for the checks that decided the module's constants."""
    stem = CHALLENGE / "Diffusion_WITH_REVERSEPHASE"
    img = nib.load(stem.with_suffix(".nii.gz"))
    data = np.asarray(img.dataobj)
    bval, bvec = np.loadtxt(stem.with_suffix(".bvals")), np.loadtxt(stem.with_suffix(".bvecs"))
    assert data.shape[-1] == len(bval) == bvec.shape[1] == 34, (data.shape, bval.shape, bvec.shape)
    assert bval[0] < 50 and bval[1] < 50 and (np.abs(bval[2:] - SHELL) < 1).all(), bval
    assert data.shape[1] == PE_LINES, data.shape
    keep = np.r_[0, 2:34]
    dwi, bval, bvec = data[..., keep], bval[keep], bvec[:, keep].copy()
    bvec[0] *= gradient_x
    pe = np.array([0.0, pe_sign, 0.0])
    b0s = np.stack([data[..., 0], data[..., 1]], -1).astype(np.float64)
    return Subject(name="ismrm2015", dwi=dwi, affine=img.affine, vox=np.asarray(img.header.get_zooms()[:3], float),
                   bval=bval, bvec=bvec, pe_axis=1, pe_sign=float(pe_sign), readout_s=float(readout_s), b0s=b0s,
                   pe_vectors=np.stack([pe, -pe]),
                   notes=[f"gradient x {gradient_x:+g} against FSL's convention", f"pe_sign {pe_sign:+g} along j",
                          f"readout {readout_s} s"])


def fmap_hz() -> np.ndarray:
    """fmap_Hz on the DWI's grid (the archive's README: the same affine as the diffusion images; checked here)."""
    img = nib.load(FMAP_HZ)
    ref = nib.load(CHALLENGE / "Diffusion_WITH_REVERSEPHASE.nii.gz")
    assert np.allclose(img.affine, ref.affine) and img.shape == ref.shape[:3]
    return np.asarray(img.dataobj, np.float64)
