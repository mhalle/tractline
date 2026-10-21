"""Same model, another site: does one Siemens Prisma scan's 17-run ensemble envelope, recalibrated on that Prisma, hold
its stated coverage on the same person's scan from a second Prisma at another site? (docs/ismrm2027-abstract.md; the
MASiVar cross-vendor test is bench/calibration_crosssite.py, whose machinery this reuses.)

    uv run --group dipy --group rankfield bench/calibration_onharmony.py check [--subject ...] [--coherence]
    uv run --group dipy --group rankfield bench/calibration_onharmony.py plan
    uv run --group dipy --group rankfield bench/calibration_onharmony.py run --scanner OXF1 [--subject 13192 ...] [--runs 16] [--tags ...] [--smoke]
    uv run --group dipy --group rankfield bench/calibration_onharmony.py collect --scanner OXF1 [--subject ...] [--runs 16] [--smoke] [--force]
    uv run --group dipy --group rankfield bench/calibration_onharmony.py evaluate [--runs 16] [--margin] [--dry-run --out <elsewhere> [--smoke]]

The data (bench/fetch_ds004712.py): OpenNeuro ds004712, ON-Harmony v2.0.1 (CC0; Warrington et al., Sci Data 12:609,
2025, doi:10.1038/s41597-025-04822-2), the two Oxford Siemens Prismas: OXF1PRI (WIN-FMRIB, John Radcliffe Hospital,
Head_32 coil, serial 66050) and OXF2PRI (WIN-OHBA, Warneford Hospital, HeadNeck_64 coil, serial 66093), about 2 km
apart, one UK Biobank-style protocol on both (the same sequence, cmrr_mbep2d_diff, software syngo MR E11). All 20
subjects have a first session on each; four have six sessions on one of them (the README): 13192 and 16974 on
OXF1PRI, 14229 and 16766 on OXF2PRI (RESCAN). What a session holds (`check` prints it, from the sidecars):
  - dir-AP (PhaseEncodingDirection j-): 104 x 104 x 72 at 2.0192 x 2.0192 x 2.0 mm (a 210 mm field of view), 105
    volumes: 5 b0s (stated b = 5, at volumes 0, 21, 42, 63 and 84, each with a nonzero b-vector: zeroed here), 50 at
    b 990-1005 and 50 at b 1985-2010 interleaved; uint16, slope 1; TE 92 ms, TR 3.6 s, multiband 3, partial Fourier
    0.75, no in-plane acceleration, TotalReadoutTime 0.0690115 s; series "diff_AP_MPopt_MB3_50b1000_50b2000_8b0_lowflip";
  - dir-PA (j): the same grid, TE, TR, readout time and shim; phase A subjects 3 volumes, all b0 (stated b = 0, zero
    b-vectors, int32, "diff 3dirs b=0 pepolar"); phase B subjects 6: 3 b0s (b = 5) then 3 b2000 along about the three
    axes ("diff_PA_MPopt_MB3_3b0_lowflip"). Only the PA b0s are used, for the field.

The protocol (PROTOCOLS, PRISMA_UKB_OXF1 and PRISMA_UKB_OXF2: one protocol, a descriptor per scanner, each a strict
`check` as calibration_cohortII.PROTOCOLS): the b2000 shell alone with the 5 b0s (55 volumes), tracked at b = 2000,
the field from the 5 forward and 3 reversed b0s. Items: calibration_transfer.check's (volumes, per shell, matrix,
voxel, phase encoding, the reversed b0s and their brightness against the forward b0s), the site and scanner (BIDS
session label), and from the scan as acquired and the sidecars: the forward volumes and shells as acquired, the b0
positions, the stated b range of the 2000 shell, the reversed series' volume count and voxel, and per series
Manufacturer, model, field strength, coil, serial number, sequence, software, TE, TR, multiband factor, partial
Fourier, in-plane acceleration and TotalReadoutTime, and the shim identical between the two series. Each refuses the
other scanner's scans (serial number, coil, session label), every MASiVar scan (cohort III, cIs1, the cohort II
protocols, site 3), and every MASiVar descriptor (and the cohort III map's match) refuses these (`check`, evaluate's
"refusals").
The tracked shell: b2000 alone, as cohort II's B2000 and S3_2000 (b2000n96 alone), so the comparison with them changes
the scanner and not the shell. Joining the b1000 shell would not change the tracking: the tracker takes the b0s and
the 2000 shell only (prep.prepare), and the field is estimated from the b0s; it would add 50 volumes to the
correction, the bootstrap and the memory, and change the replicates' random draws, not their statistics.
The loader (load): tractline.bids.load on the AP series finds the PA series (same folder, same axis, the other way)
and checks it (axis, shim, echo time, voxel sizes, coverage, readout times stated): its pairing must be that PA
series. The b0s are then checked as calibration_crosssite.load_s3 checks Siemens': each B0_OVER_DW times the series'
DW volumes' brain median, no DW volume b0-like (B0_LIKE of the b0s), the DW b-vectors unit length; the PA b0s by
their stated b and B0_OVER_DW over their own series' DW volumes (phase B) or B0_LIKE of the forward b0s (phase A, no
DW volumes). The readout time is the sidecars' TotalReadoutTime (both series equal), not the nominal value MASiVar
needs. The phase encoding is the sidecars' (AP j-, PA j).
The bootstrap (calibration_runs.Runner.replicates: _bootstrap.ShellWildBootstrap, smooth signs, FWHM 4 voxels, b0
noise "added", A's mask) needs nothing changed, but differs from B2000's in what it has: 5 forward b0s (B2000: 1, which
borrows a DW residual pattern) so the b0s have residuals of their own, scaled to the DW residual RMS; 3 reversed b0s
about their own mean (B2000: 1, borrowed); the 50-direction shell fitted at SH order 6 (28 coefficients, mean
leverage 0.56) against 96 directions at order 8 (45, 0.47). Multiband 3 on 72 slices excites slices 24 apart together:
their noise can be correlated across those slices (and amplified by the g-factor, more on the 32-channel coil), which
the smooth signs (each slice its own) do not reproduce; the smooth signs keep the in-plane correlation only. Recorded
per A in runs.json ("bootstrap_smooth", noise_check) for the two coils. 13192's smoke run (OXF1PRI001): b0_scale 0.21,
that is the 5 b0s' own residuals are about 5 times the DW residual RMS (motion, pulsation and drift between b0s 21
volumes apart, not noise), scaled down to it as "added" intends.
Registration QC: on 13192's smoke run the cross-site rescan's best b0 NCC is 0.76 (flagged, below NCC_QC 0.8; starts a
and b agree within 0.06 deg and 0.04 mm) while its FA NCC (0.63) and FA correlation with A (0.83) are as high as the
same-scanner rescan's (0.57, 0.84; b0 NCC 0.94): the two coils' receive profiles change the b0 contrast. The flag is
kept and reported (calibration_cohortII's rule), not used to drop a pair.

Per subject and A's scanner (PAIRS; Records in DATA/ensemble/ds004712-<subject>/card/<subject>-PRISMA_<scanner>/,
--smoke: ...-smoke/): A = the scanner's first session, its 17-run ensemble (A and 16 replicates, the B2000 settings:
smooth signs, FWHM 4, b0 noise "added", A's mask, draw 0, today's default pipeline at b = 2000); the rescans (tags =
their session labels):
  - cross site (TERM_X): the other Prisma's first session;
  - same scanner (TERM_SAME): the scanner's sessions 2-6, for the subjects with six on it (RESCAN).
Every rescan is corrected on its own grid, registered by mode 3 (calibration_cohortII.register_multi, bench/
_register_fa.py: (a) mode 1's pipeline on the mean b0s, (b) FA onto FA, (c) the other direction's transform inverted
where that folder has one; each refined by rigid MI on the corrected mean b0s, the best b0 NCC kept, flagged below
NCC_QC), resampled onto A's grid with its b-vectors turned and tracked there: <tag>reg3_onA. Before tracking, the
b-vector check of calibration_crosssite.bvec_check (the principal direction in A's corpus callosum body and CST below
it; agreement with A's streamline tangents as stated and with each axis flipped) and FA's correlation with A in A's
mask; A's own check once A is tracked ("A_bvec_check"). The OXF1 folders (A = OXF1PRI001) are the forward direction:
all 20 subjects, their cross-site rescan OXF2PRI001, and for 13192 and 16974 the same-scanner OXF1PRI002-006. The OXF2
folders (A = OXF2PRI001) are needed for 14229 and 16766 (their same-scanner OXF2PRI002-006, and their reverse
cross-site OXF1PRI001 with it); for the other 18 subjects they are the optional reverse direction.

collect: calibration_cohortII.collect's statements for every rescan of a folder (its tags, each with its term), in the
folder as statements_R<runs + 1>_reg3.npz.

evaluate (no pipeline runs; results/calibration_onharmony.json, provenance as the other scripts'):
  (a) as stated: envelope coverage at stated levels 0.5-0.95 (calibration_card.coverage_curve; a subject's mean over
      its rescans of a term), streamline and point ECE: same-scanner rescans (per scanner and pooled) and cross site
      (forward; reverse where run);
  (b) a cross-site map, leave one subject out (fitted, calibration_card.fit_term, on the other subjects' cross-site
      pairs, applied to the held-out subject's: coverage at the map's level per target, the widest outline where it
      never reaches it);
  (c) the key test: a same-scanner map (fitted on the within-scanner rescans: OXF1's two subjects, OXF2's two, or all
      four) applied to the cross-site pairs, the target person always left out; and the reverse, the cross-site map
      (without the target person) applied to the within-scanner pairs; beside each target's own-term LOSO map;
  (d) the cross-site map's learning curve: training sets of k = 1, 2, 4, 8 (and 19) subjects, up to LC.MAX_DRAWS sets
      per held-out subject (calibration_learning_curve's fit and scores);
  (e) the widest outline's coverage per pair; with --margin, scratchpad/crosssite/margin.py's steps past it (widest +
      a half step: 6-neighbors with density >= RHO / 2 in a run; + 1 voxel, 6- and 26-neighbors) from the records;
  (f) "fig6 (COHORT2_SCHEMA shape)": protocol "PRISMA OXF1->OXF2" (the cross-site pairs, as stated and LOSO map),
      "PRISMA same scanner" (the within-scanner pairs), transfer "PRISMA OXF1->OXF2" (the same-scanner map, without the
      target person, on the cross-site pairs); ismrm2027_figures.py is not changed.
  refusals: as above.

Imports calibration_cohortII, calibration_crosssite, calibration_runs, calibration_card, calibration_transfer and
calibration_learning_curve (constants, imports and definitions on import; nothing runs) and never writes them.
"""
import argparse, json, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_cohortII as CII   # noqa: E402
import calibration_crosssite as CS   # noqa: E402
import calibration_runs as CR   # noqa: E402
import calibration_card as CC   # noqa: E402
import calibration_transfer as CT   # noqa: E402
import calibration_learning_curve as LC   # noqa: E402
import _masivar as M   # noqa: E402
from ensemble_retest import onto_reference, mean_b0  # noqa: E402

DS = "ds004712"
ROOT = DATA / DS
SUBJECTS = ("03286", "03997", "10975", "12813", "13192", "13305", "14221", "14229", "14230", "14482",
            "15320", "16745", "16766", "16793", "16794", "16841", "16842", "16974", "16975", "16981")
SCANNERS = {"OXF1": "OXF1PRI", "OXF2": "OXF2PRI"}
OTHER = {"OXF1": "OXF2", "OXF2": "OXF1"}
RESCAN = {"13192": "OXF1", "16974": "OXF1", "14229": "OXF2", "16766": "OXF2"}   # six sessions on one Prisma (README)
TERM_SAME = "same scanner (other session)"
TERM_X = "cross site (the other Prisma)"
SHELL = 2000.0
REG = 3                                   # every rescan: registration mode 3
RUNS = CR.RUNS
LEVELS = CII.SCHEMA_LEVELS
TABLE = (0.8, 0.9, 0.95)
LC_K = (1, 2, 4, 8, 19)
OUT = HERE / "results/calibration_onharmony.json"
B0_OVER_DW, B0_LIKE = CII.B0_OVER_DW, M.B0_LIKE
FIG6_X, FIG6_SAME = "PRISMA OXF1->OXF2", "PRISMA same scanner"
SECONDS = {"A": 110, "rescan": 200, "replicate": 110,   # per run on the M2, 13192's smoke runs (2026-10-19, idle machine): A 85 s
           "collect per record": 11}                  # and its b-vector check about 25 s; a rescan its correction, registration
                                                      # and resampling (55-86 s), its b-vector and FA checks (41-45 s) and its _onA
                                                      # run (83-86 s); replicate r001 109 s (cohort II's idle B2000 replicates
                                                      # varied 63-128 s); collect about 11 s per record (A, replicates, rescans)


def _proto(sc, serial, coil, station, institution):
    lab = SCANNERS[sc]
    side = {"Manufacturer": "Siemens", "ManufacturersModelName": "Prisma", "MagneticFieldStrength": 3, "ReceiveCoilName": coil,
            "DeviceSerialNumber": serial, "PulseSequenceDetails": "%CustomerSeq%\\cmrr_mbep2d_diff", "SoftwareVersions": "syngo MR E11",
            "EchoTime": 0.092, "RepetitionTime": 3.6, "MultibandAccelerationFactor": 3, "PartialFourier": 0.75,
            "ParallelReductionFactorInPlane": None, "TotalReadoutTime": 0.0690115}
    return {"scanner": lab, "site": f"{institution} ({station})", "coil": coil,
            "series": "dir-AP: 5 b0s (b = 5), 50 b1000, 50 b2000 interleaved; the b0s and b2000 kept (55 volumes)",
            "reversed series": "dir-PA in the same session: its b0s only (3; phase B's 3 b2000 volumes are not used)",
            "tracked shell": SHELL, "resolution": "2.0192 x 2.0192 x 2.0 mm (104 x 104 x 72), both series",
            "note": f"UK Biobank-style protocol on {lab} ({coil}), b2000 x50 alone, tracked at 2000",
            "check": {"site and scanner (BIDS session label)": lab, "volumes": 55, "volumes per shell": {"0": 5, "2000": 50},
                      "shell_round": 100, "matrix": [104, 104, 72], "voxel_mm": [2.0192, 2.0192, 2.0], "voxel_tolerance_mm": 0.01,
                      "phase_encoding": ["j-", "j"], "reversed_b0s": 3,
                      "reversed_b0_over_forward": [0.8, 1.1],     # seen 0.832-1.026 (180 b0s, 60 sessions; 14229 OXF2PRI005 0.83)
                      "reversed series voxel_mm": [2.0192, 2.0192, 2.0],
                      "forward volumes as acquired": 105, "forward shells as acquired": {"0": 5, "1000": 50, "2000": 50},
                      "forward b0s at (volume index, as acquired)": [0, 21, 42, 63, 84],
                      "stated b of the 2000 shell (min, max)": [1975, 2025],
                      "reversed volumes as acquired (one of)": [3, 6],
                      "shim identical (forward, reversed)": True,
                      "sidecar (forward and reversed)": side}}


PROTOCOLS = {"PRISMA_UKB_OXF1": _proto("OXF1", "66050", "Head_32", "AWP66050", "The John Radcliffe"),
             "PRISMA_UKB_OXF2": _proto("OXF2", "66093", "HeadNeck_64", "AWP66093", "Warneford Hospital")}
KEY = {sc: f"PRISMA_UKB_{sc}" for sc in SCANNERS}


def scanner_of_session(ses):
    return next(sc for sc, lab in SCANNERS.items() if ses.startswith(lab))


# ----------------------------------------------------------------------------------------------- the scans

def files(subject, session):
    d = ROOT / f"sub-{subject}/ses-{session}/dwi"
    return {pe: d / f"sub-{subject}_ses-{session}_dir-{pe}_dwi.nii.gz" for pe in ("AP", "PA")}


def sessions(subject, sc):
    return sorted(p.name[4:] for p in (ROOT / f"sub-{subject}").glob(f"ses-{SCANNERS[sc]}00?") if files(subject, p.name[4:])["AP"].exists())


def _side(p):
    return json.loads(p.with_name(p.name[:-7] + ".json").read_text())


def load(subject, session) -> M.Subject:
    """A session's AP series, b0s and b2000 kept, with the PA b0s for the field (module docstring)."""
    import nibabel as nib
    from tractline import bids
    f = files(subject, session)
    s = bids.load(f["AP"])
    pr = s.pairing
    assert pr.ok and [Path(p).resolve() for p in pr.series] == [f["PA"].resolve()], \
        f"{subject} {session}: bids pairing {pr.how} {[Path(p).name for p in pr.series]}: {pr.message}"
    sa, sp = _side(f["AP"]), _side(f["PA"])
    bval_all, bvec_all = np.asarray(s.bval, float), np.asarray(s.bvec, float)
    dwi_all = np.asarray(s.dwi)
    b0 = np.flatnonzero(bval_all < 50)
    dw = np.flatnonzero(bval_all >= 50)
    keep = np.flatnonzero((bval_all < 50) | (np.abs(bval_all - SHELL) < 50))       # prep.prepare's selection
    norms = np.linalg.norm(bvec_all[:, dw], axis=0)
    assert np.allclose(norms, 1, atol=0.01), f"{f['AP'].name}: DW b-vectors not unit length ({norms.min():.3f}-{norms.max():.3f})"
    ref = dwi_all[..., b0].astype(np.float64).mean(-1)
    b0_med = M.brain_median(dwi_all[..., b0].astype(np.float64), ref)
    dw_med = M.brain_median(dwi_all[..., dw].astype(np.float64), ref)
    over = b0_med / np.median(dw_med)
    assert (over >= B0_OVER_DW).all(), f"{f['AP'].name}: stated b0s {over.round(3).tolist()} x the DW volumes: not all b0s"
    dw_like = dw_med / np.median(b0_med)
    assert dw_like.max() < B0_LIKE, f"{f['AP'].name}: a DW volume at {dw_like.max():.2f} of the b0s (b0-like)"
    shell = np.abs(bval_all - SHELL) < 50
    img = nib.load(f["AP"])
    series = [{"file": f["AP"].name, "volumes as acquired": int(len(bval_all)), "volumes kept": int(len(keep)), "b0s": int(len(b0)),
               "b0s at (as acquired)": b0.tolist(), "b0s stated b": sorted(set(bval_all[b0].tolist())),
               "b0 b-vectors as stated": "zero" if np.abs(bvec_all[:, b0]).sum() == 0 else "nonzero (zeroed here)",
               "b0s over dw": over.round(3).tolist(), "dw over b0s (max)": round(float(dw_like.max()), 3),
               "shells as acquired": {str(int(k)): int(v) for k, v in zip(*np.unique(np.round(bval_all[bval_all >= 50] / 1000) * 1000, return_counts=True))} | {"0": int(len(b0))},
               "stated b of the shell (min, max)": [int(bval_all[shell].min()), int(bval_all[shell].max())],
               "dtype": str(img.get_data_dtype()), "sidecar": {k: sa.get(k) for k in PROTOCOLS["PRISMA_UKB_OXF1"]["check"]["sidecar (forward and reversed)"]},
               "SeriesDescription": sa.get("SeriesDescription"), "ShimSetting": sa.get("ShimSetting")}]
    dwi = dwi_all[..., keep].astype(np.float32)
    bval = np.where(bval_all[keep] < 50, 0.0, bval_all[keep])
    bvec = bvec_all[:, keep].copy()
    bvec[:, bval < 50] = 0.0                                   # Siemens writes a direction for its b = 5 volumes
    b0_level = np.median(b0_med)
    pimg = nib.load(f["PA"])
    rraw = np.asarray(pimg.dataobj, np.float64)
    rraw = rraw[..., None] if rraw.ndim == 3 else rraw
    rb = np.loadtxt(f["PA"].with_name(f["PA"].name[:-7] + ".bval"), ndmin=1)
    rvol, how = CII.onto_coarser(rraw, pimg, img.affine, img.shape[:3], np.asarray(img.header.get_zooms()[:3], float))
    ratio = M.brain_median(rvol, ref) / b0_level
    rdw = np.flatnonzero(rb >= 50)
    own = ratio / np.median(ratio[rdw]) if len(rdw) else None
    take = np.flatnonzero((rb < 50) & ((own >= B0_OVER_DW) if own is not None else (ratio >= B0_LIKE)))
    assert len(take) == (rb < 50).sum(), f"{f['PA'].name}: stated b0s {np.flatnonzero(rb < 50).tolist()}, of which {take.tolist()} are b0s"
    series.append({"file": f"APA (the reversed series): {f['PA'].name}", "volumes": int(len(rb)), "stated_b": rb.tolist(),
                   "over_app_b0s": np.round(ratio, 3).tolist(), "over_own_dw": None if own is None else np.round(own, 3).tolist(),
                   "taken_as_b0s": take.tolist(), "voxel_mm": [round(float(v), 5) for v in pimg.header.get_zooms()[:3]],
                   "onto the APP grid": how, "dtype": str(pimg.get_data_dtype()),
                   "sidecar": {k: sp.get(k) for k in PROTOCOLS["PRISMA_UKB_OXF1"]["check"]["sidecar (forward and reversed)"]},
                   "SeriesDescription": sp.get("SeriesDescription"), "ShimSetting": sp.get("ShimSetting")})
    ros = np.asarray(s.b0_readout_s, float)
    assert np.allclose(ros, s.readout_s), f"{subject} {session}: readout times differ between the series {sorted(set(ros.tolist()))}"
    pe_ap, pe_pa = (np.array(M.VEC[x["PhaseEncodingDirection"]], float) for x in (sa, sp))
    assert np.allclose(pe_ap, -pe_pa), f"not a reversed pair: {sa['PhaseEncodingDirection']} / {sp['PhaseEncodingDirection']}"
    pe_axis = int(np.argmax(np.abs(pe_ap)))
    fb0 = np.flatnonzero(bval < 50)
    b0s = np.concatenate([dwi[..., fb0].astype(np.float64), rvol[..., take]], -1)
    assert np.allclose(b0s, s.b0s, atol=1e-3), f"{subject} {session}: the b0s differ from tractline.bids's"
    notes = [f"tractline.bids pairing: {pr.message}", f"readout time {s.readout_s} s (TotalReadoutTime, both series)",
             "phase encoding from the sidecars", "Siemens: b0s stated b = 5 with a direction (zeroed); per-direction b 1985-2010",
             f"kept: the {len(fb0)} b0s and the b2000 shell ({int(shell.sum())} volumes) of {len(bval_all)}"]
    return M.Subject(name=f"{subject}_{session}_PRISMA_UKB", dwi=dwi, affine=img.affine, vox=np.asarray(img.header.get_zooms()[:3], float),
                     bval=bval, bvec=bvec, pe_axis=pe_axis, pe_sign=float(pe_ap[pe_axis]),
                     pe_directions=(sa["PhaseEncodingDirection"], sp["PhaseEncodingDirection"]), readout_s=float(s.readout_s), b0s=b0s,
                     pe_vectors=np.array([pe_ap] * len(fb0) + [pe_pa] * len(take)), series=series, notes=notes)


def _same(a, b):
    if isinstance(b, float) and a is not None:
        return isinstance(a, (int, float)) and abs(float(a) - b) <= 1e-6 * max(1.0, abs(b))
    return a == b


def check(s, key_or_check):
    """A PRISMA_UKB descriptor against a loaded scan (this module's, or any MASiVar loader's): {item: (found, ok)}.
    calibration_transfer.check's items, then the session label, the reversed series' voxel, the scan as acquired and
    the sidecars' items (None where a loader does not record them: refused)."""
    pr = PROTOCOLS[key_or_check]["check"] if isinstance(key_or_check, str) else key_or_check
    out = CT.check(s, pr)
    app = next(x for x in s.series if "APA" not in x["file"])
    apa = next(x for x in s.series if "APA" in x["file"])
    parts = s.name.split("_")
    found = {"site and scanner (BIDS session label)": parts[1][:7] if len(parts) > 1 else None,
             "reversed series voxel_mm": apa.get("voxel_mm"),
             "forward volumes as acquired": app.get("volumes as acquired"),
             "forward shells as acquired": app.get("shells as acquired"),
             "forward b0s at (volume index, as acquired)": app.get("b0s at (as acquired)"),
             "stated b of the 2000 shell (min, max)": app.get("stated b of the shell (min, max)"),
             "reversed volumes as acquired (one of)": apa.get("volumes"),
             "shim identical (forward, reversed)": (app.get("ShimSetting") == apa.get("ShimSetting")) if app.get("ShimSetting") is not None else None,
             "sidecar (forward and reversed)": [app.get("sidecar"), apa.get("sidecar")]}
    for k, v in found.items():
        want = pr[k]
        if k == "reversed series voxel_mm":
            ok = v is not None and bool(np.allclose(v, want, atol=pr["voxel_tolerance_mm"]))
        elif k == "stated b of the 2000 shell (min, max)":
            ok = v is not None and want[0] <= v[0] and v[1] <= want[1]
        elif k == "reversed volumes as acquired (one of)":
            ok = v in want
        elif k == "forward shells as acquired":
            ok = v is not None and {a: b for a, b in v.items()} == want
        elif k == "sidecar (forward and reversed)":
            bad = {}
            for side_name, side in zip(("forward", "reversed"), v):
                for item, w in want.items():
                    got = None if side is None else side.get(item)
                    if not _same(got, w):
                        bad[f"{side_name} {item}"] = got
            v, ok = (bad or "as expected"), not bad
        else:
            ok = v == want
        out[k] = (v, bool(ok))
    return out


def load_checked(subject, session, key=None):
    key = key or KEY[scanner_of_session(session)]
    s = load(subject, session)
    bad = {k: v for k, (v, ok) in check(s, key).items() if not ok}
    if bad:
        raise ValueError(f"{s.name}: not protocol {key}: {bad}")
    return s


# ----------------------------------------------------------------------------------------------- b-vectors without tracking

def coherence(s, fa_min=0.4):
    """A fiber-coherence check of the b-vectors that needs no tracking (in the spirit of Schilling et al. 2019): a
    tensor fit (calibration_crosssite.tensor_fit, the tracker's gradients) as stated and with each native axis flipped;
    per voxel with FA >= fa_min, |cos| between its principal direction and that of the voxel one step along it (both
    ways). The b-vectors as stated should give the most coherent field. Also the midline (|x| <= 4 mm from the mask's
    median x) above the mask's center: the share of high-FA voxels whose direction is mostly left-right (corpus callosum)."""
    out = {}
    for name, flip in (("as stated", None), ("x flipped", 0), ("y flipped", 1), ("z flipped", 2)):
        nb = s.bvec.copy()
        if flip is not None:
            nb[flip] *= -1
        idx, e1, fa, mask = CS.tensor_fit(s.dwi, s.affine, s.bval, nb, SHELL)
        shape = mask.shape
        pos = np.full(int(np.prod(shape)), -1)
        pos[idx] = np.arange(len(idx))
        ijk = np.array(np.unravel_index(idx, shape)).T
        inv = np.linalg.inv(s.affine[:3, :3])
        d = e1 @ inv.T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        sel = np.flatnonzero(fa >= fa_min)
        cos = []
        for sign in (1, -1):
            nb_ijk = np.rint(ijk[sel] + sign * d[sel]).astype(int)
            inside = np.all((nb_ijk >= 0) & (nb_ijk < shape), 1)
            j = np.full(len(sel), -1)
            j[inside] = pos[np.ravel_multi_index(nb_ijk[inside].T, shape)]
            ok = j >= 0
            ok[ok] = fa[j[ok]] >= fa_min
            cos.append(np.abs((e1[sel][ok] * e1[j[ok]]).sum(1)))
        cos = np.concatenate(cos)
        world = ijk @ s.affine[:3, :3].T + s.affine[:3, 3]
        xm, zm = np.median(world[:, 0]), np.median(world[:, 2])
        mid = sel[(np.abs(world[sel, 0] - xm) <= 4) & (world[sel, 2] > zm)]
        out[name] = {"voxels (FA >= %g)" % fa_min: int(len(sel)), "mean |cos| with the next voxel along it": round(float(cos.mean()), 4),
                     "midline above center: voxels, share mostly left-right": [int(len(mid)), round(float(np.mean(np.abs(e1[mid]).argmax(1) == 0)), 3) if len(mid) else None]}
    m = {k: v["mean |cos| with the next voxel along it"] for k, v in out.items()}
    out["as stated most coherent"] = bool(m["as stated"] > max(v for k, v in m.items() if k != "as stated"))
    return out


# ----------------------------------------------------------------------------------------------- pairs and records

def pair(subject, sc):
    """A = the scanner's first session; the rescans: the other Prisma's first session (cross site) and, for RESCAN's
    subjects on this scanner, sessions 2-6 (same scanner). Tags are the session labels."""
    own = sessions(subject, sc)
    ref = f"{SCANNERS[sc]}001"
    assert ref in own, f"{subject}: no {ref}"
    other = {f"{SCANNERS[OTHER[sc]]}001": (f"{SCANNERS[OTHER[sc]]}001", 1)}
    if RESCAN.get(subject) == sc:
        other.update({ses: (ses, 1) for ses in own if ses != ref})
    return CR.Pair(f"{subject}-PRISMA_{sc}", (ref, 1), {}, other,
                   notes=(PROTOCOLS[KEY[sc]]["note"], "rescans: the other Prisma's first session (cross site)" +
                          ("; this scanner's sessions 2-6 (same scanner)" if RESCAN.get(subject) == sc else "")))


def term_of(sc, tag):
    return TERM_SAME if tag.startswith(SCANNERS[sc]) else TERM_X


class Records(CR.Records):
    def __init__(self, subject, sc, smoke=False):
        self.subject, self.sc, self.pair = subject, sc, pair(subject, sc)
        self.dir = DATA / "ensemble" / f"{DS}-{subject}" / "card" / (self.pair.name + ("-smoke" if smoke else ""))
        self.reuse = None


def module_sha256():
    return CR.sha256(Path(__file__).resolve())


def pipeline_settings():
    s = CR.pipeline_settings()
    s["track"] = {**s["track"], "shell": SHELL}
    s["volumes kept"] = "the b0s and the b2000 shell of the AP series"
    return s


class Runner(CII.Runner):
    """calibration_cohortII.Runner's one() (tracked at b = 2000) and calibration_runs.Runner's replicates, with this
    module's loader, descriptors, records and mode 3 rescans."""
    def __init__(self, subject, sc, smoke=False):
        self.key, self.shell, self.sc = KEY[sc], SHELL, sc
        self.R = Records(subject, sc, smoke)
        self.subject, self.pair = subject, self.R.pair
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": CR.git_state(), "calibration_onharmony.py sha256": module_sha256(),
                                        "calibration_cohortII.py sha256": CII.module_sha256(), "pipeline": pipeline_settings(),
                                        "protocol": {"key": self.key, **PROTOCOLS[self.key]}, "smoke": smoke, "pair": {
            "reference": self.pair.ref, "rescans": self.pair.other, "terms": {t: term_of(sc, t) for t in self.pair.other},
            "notes": list(self.pair.notes)}, "registration": CII.REGISTRATION[REG]})
        self.labeler = None

    def reference(self):
        A = load_checked(self.subject, self.pair.ref[0], self.key)
        parts = {"A": self.R.have("A"), "A_meanb0": self.R.have("A_meanb0", ".npy"), "A_mask": self.R.have("A_mask", ".npy")}
        if not any(parts.values()):
            self.log["A_scan"] = {"name": A.name, "series": A.series, "notes": A.notes, "shape": list(A.dwi.shape)}

            def companions(corr, tg):
                for name, arr in (("A_meanb0", mean_b0(corr.dwi, A.bval)), ("A_mask", tg.mask)):
                    p = self.R.new(name, ".npy")
                    tmp = p.with_name(p.stem + ".partial.npy")
                    np.save(tmp, arr)
                    tmp.replace(p)
            self.one(A, "A", before_record=companions)
        elif not all(parts.values()):
            raise RuntimeError(f"{self.subject}: A's records are incomplete {parts}: remove the companions of a missing A.npz by hand")
        if "A_bvec_check" not in self.log:
            t0 = time.time()
            rec = E.RunRecord.load(self.R.path("A"), logp=False, fibers=True)
            self.log["A_bvec_check"] = {**CS.bvec_check(A.dwi, A.affine, A.bval, A.bvec, self.shell, rec), "seconds": round(time.time() - t0, 1)}
            self.save_log()
            print(self.subject, self.sc, "A b-vector check:", json.dumps(CS.brief(self.log["A_bvec_check"])), flush=True)
        return A, np.load(self.R.path("A_meanb0", ".npy")), np.load(self.R.path("A_mask", ".npy"))

    def other_direction(self, tag):
        """The other direction's transform for a cross-site rescan: the folder with A = this rescan's session, its entry
        for this A (dipy's start, static (A) world -> moving (rescan) world, is that entry's to_A); else (None, None)."""
        if term_of(self.sc, tag) != TERM_X:
            return None, None
        d = Records(self.subject, OTHER[self.sc], smoke=self.R.dir.name.endswith("-smoke")).dir
        if not (d / "runs.json").exists():
            return None, None
        e = json.loads((d / "runs.json").read_text()).get(f"{CII.rescan_tag(self.pair.ref[0], REG)}_scan")
        return (np.array(e["to_A"]), f"{d.name} {CII.rescan_tag(self.pair.ref[0], REG)}_scan") if e else (None, None)

    def rescan3(self, A, ref_b0, mask, tag):
        name = CII.rescan_tag(tag, REG)
        if self.R.have(f"{name}_onA"):
            return
        t0 = time.time()
        ses = self.pair.other[tag][0]
        B = load_checked(self.subject, ses)
        timer = P.Timer()
        with P.exact_float32(CR.DEVICE):
            cb = P.correct(B, timer, CR.DEVICE)
        P.release_memory(CR.DEVICE)
        other, other_from = self.other_direction(tag)
        r = CII.register_multi(B, mean_b0(cb.dwi, B.bval), A, ref_b0, other)
        e = CII.reg3_entry(r, None, tag, CII.mask_world(mask, A.affine), other_from)
        e["decision"] = "mode 3 (no mode 2 here): tracked"
        T = np.array(e["to_A"])
        dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, T, A.affine, A.dwi.shape[:3])
        secs = round(time.time() - t0, 1)
        t1 = time.time()
        rec = E.RunRecord.load(self.R.path("A"), logp=False, fibers=True)
        chk = CS.bvec_check(dwi, A.affine, B.bval, bvec, self.shell, rec, flips_in=bvec @ np.linalg.pinv(B.bvec))
        fa_r = CS.fa_correlation(A.dwi, A.bval, A.bvec, dwi, B.bval, bvec, A.affine, self.shell, mask)
        e.update({"name": B.name, "term": term_of(self.sc, tag), "session": ses, "series": B.series, "notes": B.notes, "shape": list(B.dwi.shape),
                  "residual_left": cb.residual_left, "field_hz_p99_abs": round(float(np.quantile(np.abs(cb.field_hz), 0.99)), 2),
                  "correction warnings": cb.warnings, "correct_seconds": timer.seconds, "correct_register_resample_s": secs,
                  "FA correlation with A, in A's mask (A as acquired; the rescan corrected, on A's grid)": fa_r,
                  "b-vector check (the rescan on A's grid, flips in its native axes)": chk, "checks_seconds": round(time.time() - t1, 1)})
        self.log[f"{name}_scan"] = e
        self.save_log()
        print(self.subject, self.sc, name, e["term"], "best", e["best"], e["best b0 ncc"], "QC flag", e["QC"]["flag"],
              json.dumps({k: e[k] for k in ("rotation_deg", "translation_mm", "residual_left")}), "FA r", fa_r,
              "bvec as stated best:", chk["as stated agrees best"], flush=True)
        self.one(replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), f"{name}_onA", corrected=dwi)


def run(args):
    for sub in args.subject:
        t0 = time.time()
        rn = Runner(sub, args.scanner, args.smoke)
        A, ref_b0, mask = rn.reference()
        tags = [t for t in rn.pair.other if not args.tags or t in args.tags]
        for tag in sorted(tags, key=lambda t: (term_of(args.scanner, t) != TERM_X, t)):     # cross site first
            rn.rescan3(A, ref_b0, mask, tag)
        rn.replicates(A, mask, args.runs)
        print(f"{sub} PRISMA_{args.scanner} done {time.time() - t0:.0f}s, {CR.free_gb():.1f} GB free", flush=True)


# ----------------------------------------------------------------------------------------------- the statements

def statements_name(R):
    return f"statements_R{R}_reg{REG}.npz"


def collect(args):
    """calibration_cohortII.collect for every rescan of a folder (its tags and terms)."""
    import nibabel as nib
    from ensemble_grid_check import Grid
    for sub in args.subject:
        t0 = time.time()
        R = Records(sub, args.scanner, args.smoke)
        out = R.dir / statements_name(args.runs + 1)
        if out.exists() and not args.force:
            raise FileExistsError(f"{out} exists (--force to rebuild it)")
        log = json.loads((R.dir / "runs.json").read_text())
        paths = R.ensemble("", args.runs)
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{sub} {args.scanner}: {len(missing)} of A's {len(paths)} runs missing, the first {missing[0]}")
        tags = [CII.rescan_tag(t, REG) for t in R.pair.other if R.have(f"{CII.rescan_tag(t, REG)}_onA")]
        img = nib.load(files(sub, R.pair.ref[0])["AP"])
        A0 = E.RunRecord.load(paths[0], logp=False)
        grid = Grid("dwi", img.affine, A0.shape)
        mask = np.load(R.path("A_mask", ".npy"))
        cent = CC.centers(mask, A0.counts, A0.shape)
        Mx = CC.sphere_matrix(cent, A0.shape, grid.h)
        keys = A0.keys
        print(f"{sub} {args.scanner}: R = {len(paths)}, rescans {tags}, {len(cent)} points", flush=True)
        A = CC.Scan(paths, keys, grid, Mx, t0)
        Rn = A.R
        tA = A.T[0]
        s_ok = (A.T >= 0).all(0)
        s_mode, s_votes = CC.mode_votes(A.T[:, s_ok], range(E.N_TRACTS))
        s_Avotes = (A.T[:, s_ok] == tA[s_ok]).sum(0).astype(np.int16)
        p_mode, p_votes = CC.mode_votes(A.P, range(-1, CC.NAMED))
        cnt, _ = CC.vote_hist(A.votes, A.own, Rn)
        arrays = {"keys": keys[s_ok], "s_A": tA[s_ok], "s_mode": s_mode, "s_votes": s_votes, "s_Avotes": s_Avotes,
                  "centers": cent, "p_mode": p_mode, "p_votes": p_votes, "e_cnt": cnt, "e_Asize": A.own.sum(1).astype(np.int64)}
        for tag in tags:
            k, t, pres, main = CC.run_stats(R.path(f"{tag}_onA"), grid, Mx)
            arrays[f"s_tract_{tag}"] = CC.on_keys(keys, k, t)[s_ok]
            arrays[f"p_main_{tag}"] = main
            arrays[f"e_hit_{tag}"] = CC.vote_hist(A.votes, pres, Rn)[1]
            arrays[f"e_Ahit_{tag}"] = (A.own & pres).sum(1).astype(np.int64)
            print(f"  {tag}_onA {time.time() - t0:.0f}s", flush=True)
        meta = {"subject": sub, "protocol": KEY[args.scanner], "pair": R.dir.name, "R": Rn, "runs": [str(p.relative_to(DATA)) for p in paths],
                "rescans": {t: {"term": log[f"{t}_scan"]["term"], "scan": log[f"{t}_scan"]["name"], "session": log[f"{t}_scan"]["session"],
                                "rotation_deg": log[f"{t}_scan"]["rotation_deg"], "translation_mm": log[f"{t}_scan"]["translation_mm"],
                                "ncc_after": log[f"{t}_scan"]["ncc_after"], "registration mode": REG, "record": f"{t}_onA",
                                "QC flag": log[f"{t}_scan"]["QC"]["flag"]} for t in tags},
                "registration mode": REG, "pooled": {}, "points": int(len(cent)), "seeds labeled in all R runs": int(s_ok.sum()),
                "records git at creation": log["created"]["git"], "collect git": CR.git_state(),
                "calibration_onharmony.py sha256": module_sha256(), "seconds": round(time.time() - t0, 1)}
        tmp = out.with_name(out.stem + ".partial.npz")
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.replace(out)
        print(json.dumps({k: v for k, v in meta.items() if k != "runs"}, indent=1), flush=True)


# ----------------------------------------------------------------------------------------------- plan and check

def plan(args):
    total = 0
    for sc in ("OXF1", "OXF2"):
        secs = 0
        print(f"\nA on {sc} ({SCANNERS[sc]}001):")
        for sub in SUBJECTS:
            p = pair(sub, sc)
            R = Records(sub, sc)
            need = {"A": int(not R.have("A")), "rescan": sum(not R.have(f"{CII.rescan_tag(t, REG)}_onA") for t in p.other),
                    "replicate": sum(not R.have(f"{CR.FLIP}/r{r:03d}") for r in range(1, args.runs + 1))}
            phase = "needed" if sc == "OXF1" or RESCAN.get(sub) == "OXF2" else "optional (reverse)"
            s = sum(SECONDS[k] * v for k, v in need.items())
            s += SECONDS["collect per record"] * (1 + args.runs + len(p.other)) if not (R.dir / statements_name(args.runs + 1)).exists() else 0
            print(f"  {sub}: rescans {list(p.other)}; to run {need} ({s / 3600:.1f} h); {phase}")
            secs += s
        print(f"  {sc}: about {secs / 3600:.1f} h (SECONDS {SECONDS})")
        total += secs
    print(f"\n{CR.free_gb():.1f} GB free")


def refusals(scans):
    """Each PRISMA_UKB descriptor against MASiVar scans (cohort III, cIs1, each cohort II protocol's first loadable A,
    site 3) and ON-Harmony's given `scans`; every MASiVar descriptor (cohort III, cIs1, cohort II's, S3_2000) and the
    cohort III map's match (protocol items) against `scans`."""
    from _masivar import load as load_iii
    from tractline.labelers.rapidparc import HASHES
    card = json.loads(CII.CARD_PATH.read_text())
    masivar = {"cohort III (cIIIs03 A)": load_iii(CT.CONTROL[0], *CT.CONTROL[1], CR.SHELLS), "cIs1 A": load_iii("cIs1", *CT.PAIR.ref, CT.SHELLS)}
    for key in CII.PROTOCOLS:
        for sub, pr in CII.pairs(key).items():
            try:
                masivar[f"cohort II {key}"] = CII.load(sub, pr.ref[0], key)
                break
            except (FileNotFoundError, AssertionError):
                continue
    try:
        masivar["site 3 (S3_2000)"] = CS.load_s3(CS.SUBJECTS[0])
    except FileNotFoundError:
        pass
    theirs = {f"{name}: {s.name}": {"accepted by": [k for k in PROTOCOLS if not [i for i, (_, ok) in check(s, k).items() if not ok]]}
              for name, s in masivar.items()}
    own = {}
    for s in scans:
        row = {k: not [i for i, (_, ok) in check(s, k).items() if not ok] for k in PROTOCOLS}
        row.update({f"cohort II {k}": not [i for i, (_, ok) in CII.check(s, k).items() if not ok] for k in CII.PROTOCOLS})
        row.update({"cohort III": not [i for i, (_, ok) in CT.check(s, CR.PROTOCOL).items() if not ok],
                    "cIs1": not [i for i, (_, ok) in CT.check(s, CT.PROTOCOL_CIS1).items() if not ok],
                    "S3_2000": not [i for i, (_, ok) in CS.check(s).items() if not ok],
                    "cohort III map's match (protocol items only)": CT.match(card, s, {}, card["validity"]["match"]["R"], HASHES["rapidparc"],
                                                                              [CII.TERM])["items"]["protocol"]["ok"]})
        own[s.name] = {"accepted by": [k for k, ok in row.items() if ok], "refused by": [k for k, ok in row.items() if not ok]}
    want = lambda s: KEY[scanner_of_session(s.split("_")[1])]  # noqa: E731
    return {"MASiVar scans against the PRISMA_UKB descriptors": theirs, "ON-Harmony scans against every descriptor": own,
            "every MASiVar scan refused by both PRISMA_UKB descriptors": all(not v["accepted by"] for v in theirs.values()),
            "each ON-Harmony scan accepted by its own scanner's descriptor only": all(v["accepted by"] == [want(n)] for n, v in own.items())}


def check_cmd(args):
    """Load every session of the subjects (default all), check it against both descriptors, print the b0s, the PA
    content, polarity and readout time; --coherence: the tracking-free b-vector check on each first session; and the
    refusals table on two scans (13192's first session on each Prisma)."""
    rows, ratios = {}, []
    for sub in args.subject:
        for sc in SCANNERS:
            for ses in sessions(sub, sc):
                t0 = time.time()
                s = load(sub, ses)
                app, apa = s.series
                verdict = {k: [i for i, (_, ok) in check(s, k).items() if not ok] for k in PROTOCOLS}
                ratios += [apa["over_app_b0s"][i] for i in apa["taken_as_b0s"]]
                row = {"b0s at": app["b0s at (as acquired)"], "b0s over dw": app["b0s over dw"], "dw over b0s (max)": app["dw over b0s (max)"],
                       "shell b": app["stated b of the shell (min, max)"], "PA": [apa["volumes"], apa["stated_b"], apa["taken_as_b0s"], apa["over_app_b0s"], apa["over_own_dw"]],
                       "PE": list(s.pe_directions), "readout": s.readout_s, "kept": list(s.dwi.shape),
                       "refused by own descriptor": verdict[KEY[sc]], "accepted by the other's": not verdict[KEY[OTHER[sc]]]}
                if args.coherence and ses.endswith("001"):
                    row["coherence"] = coherence(s)
                rows[s.name] = row
                print(s.name, json.dumps({k: v for k, v in row.items() if k != "coherence"}),
                      ("coherence " + json.dumps({k: (v if isinstance(v, bool) else [v["mean |cos| with the next voxel along it"],
                                                                                     v["midline above center: voxels, share mostly left-right"]])
                                                  for k, v in row["coherence"].items()})) if "coherence" in row else "", f"{time.time() - t0:.0f}s", flush=True)
    print(f"reversed b0 over forward: {min(ratios):.3f}-{max(ratios):.3f} ({len(ratios)} b0s)")
    ok_own = all(not r["refused by own descriptor"] for r in rows.values())
    print("every session accepted by its own descriptor:", ok_own, "; none by the other's:", not any(r["accepted by the other's"] for r in rows.values()))
    ref = [load("13192", f"{SCANNERS[sc]}001") for sc in SCANNERS]
    rf = refusals(ref)
    print(json.dumps(rf, indent=1, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps({"sessions": rows, "reversed b0 over forward (min, max)": [min(ratios), max(ratios)],
                                              "refusals": rf}, indent=1, default=str))
        print(f"wrote {args.out}")


# ----------------------------------------------------------------------------------------------- evaluation

def statements(sc, runs, smoke=False, dry=False):
    """{subject: calibration_card.Subject} for the folders of A's scanner sc with statements at R = runs + 1 (dry: any R)."""
    out = {}
    for sub in SUBJECTS:
        d = Records(sub, sc, smoke).dir
        p = d / statements_name(runs + 1)
        if not p.exists() and dry:
            found = sorted(d.glob(f"statements_R*_reg{REG}.npz"))
            p = found[-1] if found else p
        if p.exists():
            out[sub] = CC.Subject(p)
            out[sub].path = p
    return out


def at(curve, q):
    return CII.at(curve, q)


def stated(s, term):
    return CS.stated(s, term)


def pick(xs, qs=TABLE):
    return CS.pick(xs, qs)


def mean_curve(rows):
    return CS.mean_curve(rows)


def table_mean(per, k):
    names = [n for n in per if per[n].get(k)]
    return {f"{q:g}": round(float(np.mean([per[n][k][f"{q:g}"] for n in names])), 4) for q in TABLE} if names else None


def per_rescan(s):
    """Per rescan: term, registration, QC flag, coverage at stated 0.8 and 0.9 (calibration_card.per_rescan)."""
    rows = CC.per_rescan(s)
    for t, r in rows.items():
        r["QC flag"] = s.meta["rescans"][t].get("QC flag")
    return rows


def widest(s, term):
    sh = CC.term_shapes(s, term)
    return None if sh is None else {k: round(float(sh["widest"][k]), 4) for k in ("coverage", "precision", "size")}


def as_stated_block(S, term):
    per = {sub: {**stated(s, term), "widest outline": widest(s, term), "rescans": s.tags(term)} for sub, s in S.items() if s.tags(term)}
    cov = {sub: r["coverage at stated level"] for sub, r in per.items()}
    return {"per subject": per, "subjects": list(per), "mean coverage at stated level (levels)": mean_curve(list(cov.values())),
            "mean coverage (0.8, 0.9, 0.95)": {f"{q:g}": round(float(np.mean([pick(c)[f'{q:g}'] for c in cov.values()])), 4) for q in TABLE} if cov else None,
            "mean ECE: streamline / point": [round(float(np.mean([r[m] for r in per.values()])), 4) for m in ("streamline ECE", "point ECE")] if per else None,
            "mean widest coverage": round(float(np.mean([r["widest outline"]["coverage"] for r in per.values()])), 4) if per else None}


def loso_block(S, term, pat):
    have = {n: s for n, s in S.items() if s.tags(term)}
    per = {n: CS.map_on([x for m, x in have.items() if m != n], term, s, term, pat) for n, s in have.items()} if len(have) >= 2 else {}
    return {"per subject": per, "subjects": list(per),
            "mean coverage at target (0.8, 0.9, 0.95)": {f"{q:g}": round(float(np.mean([pick(r["coverage at target"])[f'{q:g}'] for r in per.values()])), 4) for q in TABLE} if per else None,
            "mean ECE (map): streamline / point": [round(float(np.mean([r[m] for r in per.values()])), 4) for m in ("streamline ECE (map)", "point ECE (map)")] if per else None}


def transfer_block(train, train_term, test, test_term, pat, note):
    """A map fitted on train (the target person always left out) applied to test's rescans of test_term."""
    per, fitted = {}, {}
    for n, s in test.items():
        if not s.tags(test_term):
            continue
        tr = [x for m, x in train.items() if m != n and x.tags(train_term)]
        if tr:
            per[n] = CS.map_on(tr, train_term, s, test_term, pat)
            fitted[n] = [x.name for x in tr]
    return {"note": note, "map term": train_term, "applied to term": test_term, "map fitted on, per target": fitted, "per subject": per,
            "mean coverage at target (0.8, 0.9, 0.95)": {f"{q:g}": round(float(np.mean([pick(r["coverage at target"])[f'{q:g}'] for r in per.values()])), 4) for q in TABLE} if per else None,
            "mean ECE (map): streamline / point": [round(float(np.mean([r[m] for r in per.values()])), 4) for m in ("streamline ECE (map)", "point ECE (map)")] if per else None}


def learning_curve(S, term, pat, ks=LC_K):
    """(d): per held-out subject, maps fitted on k of the others (every combination up to LC.MAX_DRAWS, else that many
    at random, seed LC.SEED), scored on the held-out subject's rescans (calibration_learning_curve.score)."""
    have = {n: s for n, s in S.items() if s.tags(term)}
    data = {n: LC.Data(s, term) for n, s in have.items()}
    wide = {n: d.widest() for n, d in data.items()}
    names = list(data)
    rng = np.random.default_rng(LC.SEED)
    out = {"subjects": names, "as stated (k = 0)": LC.summarize([LC.score("as stated", data[n], wide[n], pat) for n in names]) if names else None, "k": {}}
    for k in ks:
        if k > len(names) - 1:
            continue
        rows = []
        for n in names:
            others = [m for m in names if m != n]
            for c in LC.draws(others, k, rng):
                rows.append(LC.score(LC.fit([data[others[j]] for j in c]), data[n], wide[n], pat))
        out["k"][str(k)] = {**LC.summarize(rows), "training sets per held-out subject": len(LC.draws([0] * (len(names) - 1), k, np.random.default_rng(0)))}
    return out


def margin_rows(sub, sc, smoke=False, runs=RUNS):
    """(e) --margin: scratchpad/crosssite/margin.py's steps past the widest outline, for every rescan of a folder."""
    import nibabel as nib
    from scipy.ndimage import binary_dilation, generate_binary_structure
    from ensemble_grid_check import Grid, sample, RHO
    from tractline.labelers.base import LUT
    Lut = np.asarray(LUT)
    R = Records(sub, sc, smoke)
    paths = R.ensemble("", runs)
    img = nib.load(files(sub, R.pair.ref[0])["AP"])
    A0 = E.RunRecord.load(paths[0], logp=False)
    grid = Grid("dwi", img.affine, A0.shape)
    mask = np.load(R.path("A_mask", ".npy")).ravel()

    def presence(path):
        rec = E.RunRecord.load(path, logp=False, fibers=True)
        Pt, Ot = sample(rec.fibers)
        pt = Lut[rec.cluster][Ot]
        n = int(np.prod(grid.shape))
        full, half = np.zeros((CC.NAMED, n), bool), np.zeros((CC.NAMED, n), bool)
        for k in range(CC.NAMED):
            s = pt == k
            if s.any():
                d = grid.density(grid.counts(Pt[s], Ot[s])).ravel()
                full[k], half[k] = d >= RHO, d >= RHO / 2
        return full, half

    def dil(U, conn):
        st = generate_binary_structure(3, conn)
        return np.stack([binary_dilation(u.reshape(A0.shape), st).ravel() & mask if u.any() else u for u in U])

    def scores(Ee, T):
        both, size, t = (Ee & T).sum(), Ee.sum(), T.sum()
        return {"coverage": round(float(both / t), 4), "precision": round(float(both / max(size, 1)), 4), "size": round(float(size / t), 4)}
    runs = [presence(p) for p in paths]
    U = np.stack([r[0] for r in runs]).any(0)
    Uh = np.stack([r[1] for r in runs]).any(0)
    d6, d26 = dil(U, 1), dil(U, 3)
    steps = {"widest": U, "widest + half step (6-neighbors with density >= RHO/2 in a run)": U | (d6 & Uh),
             "widest + 1 voxel (6-neighbors, 2 mm)": U | d6, "widest + 1 voxel (26-neighbors, <= 3.5 mm)": U | d26}
    log = json.loads((R.dir / "runs.json").read_text())
    out = {}
    for t in R.pair.other:
        tag = CII.rescan_tag(t, REG)
        if not R.have(f"{tag}_onA"):
            continue
        T = presence(R.path(f"{tag}_onA"))[0]
        out[tag] = {"term": log[f"{tag}_scan"]["term"], **{k: scores(s, T) for k, s in steps.items()}}
    return out


def fig6_block(X, Wsame, loso_x, loso_w, carry):
    pr = PROTOCOLS[KEY["OXF1"]]
    xs = {n: stated(s, TERM_X)["coverage at stated level"] for n, s in X.items() if s.tags(TERM_X)}
    ws = {n: stated(s, TERM_SAME)["coverage at stated level"] for n, s in Wsame.items()}
    row = lambda subjects, a, m: {"scanner": "OXF1PRI", "voxel_mm": pr["check"]["voxel_mm"], "tracked shell": int(SHELL),  # noqa: E731
                                  "directions": pr["check"]["volumes per shell"]["2000"], "subjects": subjects,
                                  "as stated": {"per subject": a, "mean": mean_curve(list(a.values()))},
                                  "map (leave one subject out)": None if not m else {
                                      "per subject": {n: r["coverage at target"] for n, r in m.items()},
                                      "mean": mean_curve([r["coverage at target"] for r in m.values()]),
                                      "reached": {n: r["reached"] for n, r in m.items()}}}
    prot = {FIG6_X: {**row(list(xs), xs, loso_x["per subject"]),
                     "rescan": "the same person's first session on the other Prisma (OXF2PRI, WIN-OHBA, 64-channel coil), the same protocol"},
            FIG6_SAME: {**row(list(ws), ws, loso_w["per subject"]), "scanner": "OXF1PRI and OXF2PRI (each subject's own)",
                        "rescan": "sessions 2-6 on the same Prisma (13192, 16974 on OXF1PRI; 14229, 16766 on OXF2PRI), mean per subject"}}
    c = carry["per subject"]
    trans = {FIG6_X: {"map from": FIG6_SAME, "applied to": FIG6_X, "map fitted on": list(Wsame), "target person excluded": True,
                      "map fitted on, per target subject": carry["map fitted on, per target"],
                      "confounded": "the map's pairs are on both Prismas (A on OXF1PRI for two subjects, OXF2PRI for two); the targets are OXF1 -> OXF2",
                      "per subject": {n: r["coverage at target"] for n, r in c.items()}, "mean": mean_curve([r["coverage at target"] for r in c.values()]),
                      "reached": {n: r["reached"] for n, r in c.items()}}}
    return {"schema": CII.SCHEMA, "mock": False, "levels": LEVELS.tolist(), "protocols": prot, "transfer (fixed protocol, other scanner)": trans,
            "note": "COHORT2_SCHEMA's shape; fig6_protocol_gap draws the cohort II protocol names and the site 3 row only, so these "
                    "rows need their names added there to be drawn (ismrm2027_figures.py is not changed by this script)"}


def scan_days():
    """{subject: {session label: day}} from the dataset's scan_delays.csv (days, each subject's first scan of the study
    at 0): the first sessions by scanner column, sessions 2-6 by its repeat columns on its "Repeat scanner code"."""
    import csv
    p = ROOT / "scan_delays.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sub = (row.get("") or "").strip()
            if sub not in SUBJECTS:
                continue
            day = lambda v: int(float(v)) if v not in (None, "") else None  # noqa: E731
            d = {f"{lab}001": day(row.get(lab)) for lab in SCANNERS.values()}
            rep = (row.get("Repeat scanner code") or "").strip()
            if rep in SCANNERS.values():
                d.update({f"{rep}00{k}": day(row.get(f"repeat_00{k}")) for k in range(2, 7)})
            out[sub] = d
    return out


def pair_gaps(S_by):
    """Days between A and each rescan (scan_delays.csv), per folder."""
    days = scan_days()
    out = {}
    for sc, S in S_by.items():
        for sub, s in S.items():
            a = days.get(sub, {}).get(f"{SCANNERS[sc]}001")
            out[f"{sub} {sc}"] = {t: [r["term"], None if a is None or days[sub].get(r.get("session", t[:10])) is None
                                      else abs(days[sub][r.get("session", t[:10])] - a)] for t, r in s.meta["rescans"].items()}
    return out


def provenance(S_by, smoke=False):
    out = {}
    for sc, S in S_by.items():
        for sub, s in S.items():
            R = Records(sub, sc, smoke)
            log = json.loads((R.dir / "runs.json").read_text())
            out[f"{sub} {sc}"] = {"records": str(R.dir.relative_to(DATA)), "records git at creation": log["created"]["git"],
                                  "calibration_onharmony.py sha256 at creation": log["created"].get("calibration_onharmony.py sha256"),
                                  "statements": s.path.name, "statements sha256": CR.sha256(s.path), "runs.json sha256": CR.sha256(R.dir / "runs.json"),
                                  "A.npz sha256": CR.sha256(R.path("A")), "R": s.R,
                                  "rescans (term, rotation deg, translation mm, NCC, QC flag)": {
                                      t: [r["term"], r["rotation_deg"], r["translation_mm"], r["ncc_after"], r.get("QC flag")] for t, r in s.meta["rescans"].items()},
                                  "A's input files sha256": {p.name: CR.sha256(p) for p in files(sub, R.pair.ref[0]).values()}}
    return out


def evaluate(args):
    out_path = Path(args.out)
    if args.dry_run and out_path.resolve() == OUT.resolve():
        sys.exit("a dry run writes elsewhere: pass --out")
    pat = CC.pat16_table()
    S = {sc: statements(sc, args.runs, args.smoke, args.dry_run) for sc in SCANNERS}
    X = {n: s for n, s in S["OXF1"].items() if s.tags(TERM_X)}                       # forward cross site
    Xr = {n: s for n, s in S["OXF2"].items() if s.tags(TERM_X)}                      # reverse
    W = {n: S[sc][n] for n, sc in RESCAN.items() if n in S[sc] and S[sc][n].tags(TERM_SAME)}
    W1 = {n: s for n, s in W.items() if RESCAN[n] == "OXF1"}
    W2 = {n: s for n, s in W.items() if RESCAN[n] == "OXF2"}
    if not args.dry_run:
        miss = [n for n in SUBJECTS if n not in X] + [f"{n} (same scanner)" for n in RESCAN if n not in W]
        if miss:
            sys.exit(f"{statements_name(args.runs + 1)} missing for {miss} (a partial evaluation only as --dry-run)")
    Rs = {s.R for d in S.values() for s in d.values()}
    assert len(Rs) <= 1, f"statements at different R: {sorted(Rs)}"
    res = {"what": "ON-Harmony (ds004712 v2.0.1): one Siemens Prisma scan's ensemble envelope against the same person's scan on a "
                   "second Prisma at another Oxford site (same model and protocol), as stated and with recalibration maps; beside "
                   "same-scanner rescans",
           "dry run": bool(args.dry_run), "smoke records": bool(args.smoke), "R": Rs.pop() if Rs else None, "terms": [TERM_X, TERM_SAME],
           "levels": LEVELS.tolist(), "registration": CII.REGISTRATION[REG],
           "coverage": "calibration_card.coverage_curve: the share of the rescan's density presence inside the envelope, the 42 named "
                       "tracts pooled, weighted by the rescan's voxels; a subject's mean over its rescans of a term",
           "protocol descriptors": PROTOCOLS,
           "subjects": {"cross site, forward (A on OXF1PRI)": list(X), "cross site, reverse (A on OXF2PRI)": list(Xr),
                        "same scanner": {n: [RESCAN[n], W[n].tags(TERM_SAME)] for n in W}}}
    # (a) as stated
    res["(a) as stated"] = {"cross site, forward (OXF1 -> OXF2)": as_stated_block(X, TERM_X),
                            "cross site, reverse (OXF2 -> OXF1)": as_stated_block(Xr, TERM_X),
                            "same scanner, both Prismas": as_stated_block(W, TERM_SAME),
                            "same scanner, OXF1PRI": as_stated_block(W1, TERM_SAME), "same scanner, OXF2PRI": as_stated_block(W2, TERM_SAME),
                            "per rescan (same-scanner subjects)": {n: per_rescan(s) for n, s in W.items()},
                            "the same subjects' cross site (forward), for the four same-scanner subjects": as_stated_block({n: X[n] for n in W if n in X}, TERM_X)}
    # (b) cross-site map, leave one subject out
    loso_x = loso_block(X, TERM_X, pat)
    loso_w = loso_block(W, TERM_SAME, pat)
    res["(b) cross-site map, leave one subject out"] = {"forward": loso_x, "reverse": loso_block(Xr, TERM_X, pat),
                                                         "same scanner (both Prismas), for reference": loso_w}
    # (c) does a same-model, same-scanner map transfer across sites? and back
    carry = transfer_block(W, TERM_SAME, X, TERM_X, pat, "the same-scanner map (both Prismas' within-scanner pairs), target person left out, on the cross-site pairs")
    res["(c) transfer"] = {
        "same-scanner map (both Prismas) -> cross site": carry,
        "same-scanner map (OXF1PRI pairs) -> cross site": transfer_block(W1, TERM_SAME, X, TERM_X, pat, "fitted on 13192 and 16974 (A on OXF1PRI), target left out"),
        "same-scanner map (OXF2PRI pairs) -> cross site": transfer_block(W2, TERM_SAME, X, TERM_X, pat, "fitted on 14229 and 16766 (A on OXF2PRI), target left out"),
        "cross-site map -> same scanner": transfer_block(X, TERM_X, W, TERM_SAME, pat, "the forward cross-site map (19 subjects: the target person left out) on the within-scanner pairs"),
        "cross-site map -> cross site, reverse direction": transfer_block(X, TERM_X, Xr, TERM_X, pat, "the forward cross-site map (target left out) on the reverse pairs"),
        "summary (mean coverage at 0.8, 0.9, 0.95)": None}
    tsub = list(carry["per subject"])
    tab = {"cross site as stated": {f"{q:g}": round(float(np.mean([pick(stated(X[n], TERM_X)["coverage at stated level"])[f'{q:g}'] for n in tsub])), 4) for q in TABLE} if tsub else None,
           "cross site, cross-site LOSO map": {f"{q:g}": round(float(np.mean([pick(loso_x["per subject"][n]["coverage at target"])[f'{q:g}'] for n in tsub if n in loso_x["per subject"]])), 4) for q in TABLE} if tsub and loso_x["per subject"] else None,
           "cross site, same-scanner map": carry["mean coverage at target (0.8, 0.9, 0.95)"],
           "same scanner as stated": res["(a) as stated"]["same scanner, both Prismas"]["mean coverage (0.8, 0.9, 0.95)"],
           "same scanner, same-scanner LOSO map": loso_w["mean coverage at target (0.8, 0.9, 0.95)"],
           "same scanner, cross-site map": res["(c) transfer"]["cross-site map -> same scanner"]["mean coverage at target (0.8, 0.9, 0.95)"]}
    res["(c) transfer"]["summary (mean coverage at 0.8, 0.9, 0.95)"] = {"cross-site targets (all with a same-scanner map)": tsub, **tab}
    # (d) learning curve of the cross-site map
    res["(d) cross-site map learning curve"] = learning_curve(X, TERM_X, pat)
    # (e) widest outline, and the margin past it
    res["(e) widest outline coverage"] = {"cross site, forward": {n: widest(s, TERM_X) for n, s in X.items()},
                                          "same scanner": {n: widest(s, TERM_SAME) for n, s in W.items()}}
    if args.margin:
        rows = {}
        for sc, d in S.items():
            for n in d:
                if (sc == "OXF1" and n in X) or n in W:
                    rows[f"{n} {sc}"] = margin_rows(n, sc, args.smoke, args.runs)
                    print("margin", n, sc, json.dumps({t: {k: v["coverage"] for k, v in r.items() if k != "term"} for t, r in rows[f"{n} {sc}"].items()}), flush=True)
        keys = ("widest", "widest + half step (6-neighbors with density >= RHO/2 in a run)", "widest + 1 voxel (6-neighbors, 2 mm)", "widest + 1 voxel (26-neighbors, <= 3.5 mm)")
        summ = {}
        for name, term, sc in (("cross site, forward", TERM_X, "OXF1"), ("same scanner", TERM_SAME, None)):
            rr = [r for f, x in rows.items() for r in x.values() if r["term"] == term and (sc is None or f.endswith(sc))]
            summ[name] ={k: {m: round(float(np.mean([r[k][m] for r in rr])), 4) for m in ("coverage", "precision", "size")} for k in keys} if rr else None
        res["(e) margin past the widest outline"] = {"per folder and rescan": rows, "mean over rescans": summ,
                                                     "note": "scratchpad/crosssite/margin.py's steps on this grid (2.0192 x 2.0192 x 2 mm); every rescan weighted equally"}
    gaps = pair_gaps(S)
    res["days between A and the rescan (scan_delays.csv)"] = {
        "per folder": gaps, "note": "the dataset's scan_delays.csv (days from each subject's first scan of the study)",
        "cross site, forward (min, median, max)": (lambda x: [min(x), float(np.median(x)), max(x)] if x else None)(
            [g[1] for f, r in gaps.items() if f.endswith("OXF1") for g in r.values() if g[0] == TERM_X and g[1] is not None]),
        "same scanner (min, median, max)": (lambda x: [min(x), float(np.median(x)), max(x)] if x else None)(
            [g[1] for r in gaps.values() for g in r.values() if g[0] == TERM_SAME and g[1] is not None])}
    # (f) fig6
    res["fig6 (COHORT2_SCHEMA shape)"] = fig6_block(X, W, loso_x, loso_w, carry)
    scans = [load(SUBJECTS[4], f"{SCANNERS[sc]}001") for sc in SCANNERS]
    res["refusals"] = refusals(scans)
    res["provenance"] = {"git": CR.git_state(), "calibration_onharmony.py sha256": module_sha256(), "calibration_cohortII.py sha256": CII.module_sha256(),
                         "calibration_crosssite.py sha256": CS.module_sha256(), "fetch_ds004712.py sha256": CR.sha256(HERE / "fetch_ds004712.py"),
                         "fetched.json sha256": CR.sha256(ROOT / "fetched.json") if (ROOT / "fetched.json").exists() else None,
                         "status": [x for x in subprocess.run(["git", "status", "--porcelain", "--", "bench/calibration_onharmony.py", "bench/fetch_ds004712.py"],
                                                              cwd=HERE.parent, capture_output=True, text=True).stdout.splitlines() if x],
                         "created (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "records": provenance(S, args.smoke),
                         "downloads": "bench/fetch_ds004712.py, every size equal to OpenNeuro's S3 listing (ds004712/fetched.json)"}
    res["caveats"] = [
        "20 cross-site pairs, one each (the forward direction); 4 same-scanner subjects, 5 rescans each (2 per Prisma): the same-scanner map "
        "is fitted on 3 (or 1) subjects once the target person is left out",
        "the two Prismas differ in coil (32 against 64 channels), site, shim and gradient hardware (serial numbers), not in software or protocol; "
        "time is confounded the other way: the cross-site rescan is mostly the same day (scan_delays.csv: 0 days for 16 of 20 subjects, "
        "at most 5), the same-scanner rescans 0-205 days after A (13192 32-38, 16974 30-37, 14229 119-205, 16766 0-5); see "
        "\"days between A and the rescan\"",
        "multiband 3: the bootstrap's smooth signs keep in-plane noise correlation, not the correlation between slices excited together",
        "the voxel is 2.0192 x 2.0192 x 2.0 mm: a denser grid than cohort II's 2.5 mm, so about twice the seeds and streamlines per run "
        "(every mask voxel is a seed) and a point lattice of about 8 mm",
        "the PA b0s (3) and the 5 forward b0s give the field; the readout time is stated (0.069 s) on both scanners"]
    out_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    short = {"(a)": {k: v.get("mean coverage (0.8, 0.9, 0.95)") for k, v in res["(a) as stated"].items() if isinstance(v, dict) and "mean coverage (0.8, 0.9, 0.95)" in v},
             "(b)": loso_x["mean coverage at target (0.8, 0.9, 0.95)"], "(c)": tab,
             "(d)": {k: {m: v[m]["mean"] for m in v if m.startswith("coverage")} for k, v in res["(d) cross-site map learning curve"]["k"].items()},
             "refusals strict": [res["refusals"]["every MASiVar scan refused by both PRISMA_UKB descriptors"],
                                 res["refusals"]["each ON-Harmony scan accepted by its own scanner's descriptor only"]]}
    print(json.dumps(short, indent=1, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "plan", "run", "collect", "evaluate"))
    ap.add_argument("--scanner", choices=sorted(SCANNERS), default="OXF1", help="run/collect: A's Prisma")
    ap.add_argument("--subject", nargs="+", default=list(SUBJECTS), choices=SUBJECTS)
    ap.add_argument("--runs", type=int, default=RUNS, help="A's replicates (R = runs + 1)")
    ap.add_argument("--tags", nargs="+", default=None, help="run: only these rescans (session labels)")
    ap.add_argument("--smoke", action="store_true", help="run/collect/evaluate: the <pair>-smoke record folders")
    ap.add_argument("--force", action="store_true", help="collect: rebuild an existing statements file (derived, not a record)")
    ap.add_argument("--coherence", action="store_true", help="check: the tracking-free b-vector check on each first session")
    ap.add_argument("--margin", action="store_true", help="evaluate: the steps past the widest outline, from the records (slow)")
    ap.add_argument("--dry-run", action="store_true", help="evaluate: whatever statements exist, at any R; needs --out elsewhere")
    ap.add_argument("--out", default=None, help="evaluate: default results/calibration_onharmony.json; check: a JSON copy")
    args = ap.parse_args()
    if args.what == "evaluate":
        args.out = args.out or str(OUT)
    {"check": check_cmd, "plan": plan, "run": run, "collect": collect, "evaluate": evaluate}[args.what](args)
