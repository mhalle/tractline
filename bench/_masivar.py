"""OpenNeuro ds003416 (MASiVar, CC0; Philips Achieva 3T) scans as the pipeline takes them, for the bootstrap's
test-retest check (bench/fetch_ds003416.py fetches the slice used; NOTES 2026-10-08, "The bootstrap against real
rescans").

A MASiVar scan is several series: one per shell, phase-encoded posterior-to-anterior ("APP", BIDS j), each
with interleaved b0s (stated b = 0.1), and one short reversed series ("APA", j-). load() joins a scan's APP
series, shells ascending, into one DWI and takes the reversed series' b0s for the field: the APP b0s, then the
APA b0s, as _ds001226 orders AP and PA. Two Philips habits are handled from the data, not the names:
  - each series ends with a volume stated b = 0, gradient 0. It would be dropped if darker than the series' own
    b0s (B0_LIKE; a derived isotropic image, a b > 0 trace); in every scan read so far it is a real b0 (0.99-1.0
    of the others), kept.
  - the APA series is named b1000 n3 (bvals 1000 1000 1000 0). Its volumes above B0_LIKE of the APP b0s are taken
    as b0s, the others dropped: so far always the last (0.85-0.95; the three b1000s 0.34-0.45). One reversed b0.
The sidecars state PhaseEncodingDirection only, so the readout time is nominal (NOMINAL_READOUT_S), the same for
every scan: the field in Hz is then on a nominal scale, the displacement and the corrected DWI are not (bids.py: a
nominal readout is safe within one protocol). A series on another grid than the first is put onto it by the
scanner's coordinates (cubic). The stored integers carry a scale per series (Philips: ~130-1300, a stored maximum
of ~600-1800); every series is put on the first APP series' stored units, as float32, so the stored values are
kept and only the ratios between series applied.
"""
from __future__ import annotations

import json, re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import affine_transform
from tractline.data import DATA

ROOT = DATA / "ds003416"
NOMINAL_READOUT_S = 0.05
VEC = {"j": (0, 1, 0), "j-": (0, -1, 0), "i": (1, 0, 0), "i-": (-1, 0, 0), "k": (0, 0, 1), "k-": (0, 0, -1)}
NAME = re.compile(r"acq-b(\d+)n(\d+)r(\w+?)pe(APP|APA)_run-(\d)(\d\d)_dwi\.nii\.gz$")
B0_LIKE = 0.6                     # a volume above this fraction of the b0s' brain median counts as a b0 (b0s: 0.85-1.0;
                                  # b1000: 0.35-0.45)


@dataclass
class Subject:
    name: str                    # "cIIIs01_s1Bx1_scan2"
    dwi: np.ndarray              # (X, Y, Z, G) the APP series joined, shells ascending
    affine: np.ndarray
    vox: np.ndarray
    bval: np.ndarray             # (G,) the stated b = 0.1 as 0
    bvec: np.ndarray             # (3, G)
    pe_axis: int
    pe_sign: float
    pe_directions: tuple
    readout_s: float             # nominal
    b0s: np.ndarray              # (X, Y, Z, V) float64: the APP b0s, then the APA b0s
    pe_vectors: np.ndarray       # (V, 3)
    series: list = field(default_factory=list)     # what was read, and what was dropped from each
    notes: list = field(default_factory=list)

    @property
    def b0_index(self):
        return np.flatnonzero(self.bval < 50)


def scan_files(subject: str, session: str, scan: int, shells=None):
    """{"APP": [(b, path)...] ascending, "APA": path} for one scan."""
    d = ROOT / f"sub-{subject}/ses-{session}/dwi"
    app, apa = [], []
    for p in sorted(d.glob(f"sub-{subject}_ses-{session}_acq-*_dwi.nii.gz")):
        m = NAME.search(p.name)
        if m and int(m.group(5)) == scan:
            b = int(m.group(1))
            if m.group(4) == "APA":
                apa.append(p)
            elif shells is None or b in shells:
                app.append((b, p))
    assert len(apa) == 1, f"{subject} {session} scan {scan}: {len(apa)} reversed series"
    assert app, f"{subject} {session} scan {scan}: no APP series"
    return {"APP": sorted(app), "APA": apa[0]}


def _side(p: Path, ext):
    return p.with_name(p.name[:-7] + ext)


def _read(p: Path):
    """(image, stored values, bval, bvec, the file's scale). nibabel keeps the file's scl_slope on the proxy (the
    loaded header says NaN)."""
    img = nib.load(p)
    assert not img.dataobj.inter, f"{p.name}: an intercept"
    return img, img.dataobj.get_unscaled(), np.loadtxt(_side(p, ".bval"), ndmin=1), np.loadtxt(_side(p, ".bvec"), ndmin=2), float(img.dataobj.slope)


def _onto(vol, src_affine, dst_affine, dst_shape):
    if np.allclose(src_affine, dst_affine, atol=1e-3) and vol.shape[:3] == tuple(dst_shape):
        return vol
    M = np.linalg.inv(src_affine) @ dst_affine
    v = vol.reshape(*vol.shape[:3], -1).astype(np.float64)
    out = np.stack([affine_transform(v[..., k], M[:3, :3], M[:3, 3], output_shape=tuple(dst_shape), order=3, mode="constant", cval=0.0)
                    for k in range(v.shape[-1])], -1).clip(0, None)
    return out.reshape(*dst_shape, *vol.shape[3:])


def brain_median(vols, ref_b0):
    """Per volume, the median over the voxels where the reference b0 mean is above a third of its 99th percentile."""
    m = ref_b0 > np.quantile(ref_b0, 0.99) / 3
    return np.array([np.median(vols[..., k][m]) for k in range(vols.shape[-1])])


def load(subject: str, session: str, scan: int, shells=None) -> Subject:
    files = scan_files(subject, session, scan, shells)
    first = None
    dws, bvals, bvecs, series, notes = [], [], [], [], []
    pe = unit = None
    for b, p in files["APP"]:
        img, raw, bval, bvec, slope = _read(p)
        side = json.loads(_side(p, ".json").read_text())
        pe = pe or side["PhaseEncodingDirection"]
        assert side["PhaseEncodingDirection"] == pe
        if first is None:
            first, unit = img, slope
        vol = _onto(raw.astype(np.float32) * np.float32(slope / unit), img.affine, first.affine, first.shape[:3]).astype(np.float32)
        b0 = bval < 50
        last = len(bval) - 1
        drop = []
        if b0[last] and np.abs(bvec[:, last]).sum() == 0 and b0[:last].sum() >= 2:
            ref = vol[..., np.flatnonzero(b0[:last])].astype(np.float64).mean(-1)
            ratio = brain_median(vol[..., [last]].astype(np.float64), ref)[0] / np.median(brain_median(vol[..., np.flatnonzero(b0[:last])].astype(np.float64), ref))
            if ratio < B0_LIKE:
                drop = [last]
            series.append({"file": p.name, "volumes": len(bval), "scale": round(slope / unit, 4), "b0s": int(b0[:last].sum()) + (0 if drop else 1),
                           "last_volume_over_b0s": round(float(ratio), 3), "last_volume": "dropped (not a b0)" if drop else "kept (a b0)"})
        else:
            series.append({"file": p.name, "volumes": len(bval), "scale": round(slope / unit, 4), "b0s": int(b0.sum())})
        keep = np.setdiff1d(np.arange(len(bval)), drop)
        dws.append(vol[..., keep]); bvals.append(np.where(bval[keep] < 50, 0.0, bval[keep])); bvecs.append(bvec[:, keep])
    dwi = np.concatenate(dws, -1)
    bval, bvec = np.concatenate(bvals), np.concatenate(bvecs, 1)
    A = first.affine
    b0i = np.flatnonzero(bval < 50)
    ref = dwi[..., b0i].astype(np.float64).mean(-1)
    b0_level = np.median(brain_median(dwi[..., b0i].astype(np.float64), ref))
    # the reversed series: its volumes about as bright as the b0s
    p = files["APA"]
    img, raw, rb, rv, slope = _read(p)
    side = json.loads(_side(p, ".json").read_text())
    rvol = _onto(raw.astype(np.float64) * (slope / unit), img.affine, A, first.shape[:3])
    ratio = brain_median(rvol, ref) / b0_level
    take = np.flatnonzero(ratio >= B0_LIKE)
    series.append({"file": p.name, "volumes": len(rb), "scale": round(slope / unit, 4), "stated_b": rb.tolist(), "over_app_b0s": np.round(ratio, 3).tolist(),
                   "taken_as_b0s": take.tolist()})
    assert len(take), f"{p.name}: no volume near the brightness of the APP b0s ({np.round(ratio, 3).tolist()}): no reversed b0"
    pe_app, pe_apa = (np.array(VEC[d], float) for d in (pe, side["PhaseEncodingDirection"]))
    assert np.allclose(pe_app, -pe_apa), f"not a reversed pair: {pe} / {side['PhaseEncodingDirection']}"
    pe_axis = int(np.argmax(np.abs(pe_app)))
    notes.append(f"readout time nominal ({NOMINAL_READOUT_S} s): the sidecars state none")
    if not np.allclose(img.affine, A, atol=1e-3):
        notes.append("the reversed series was put onto the APP grid by the scanner's coordinates")
    return Subject(name=f"{subject}_{session}_scan{scan}", dwi=dwi, affine=A, vox=np.asarray(first.header.get_zooms()[:3], float),
                   bval=bval, bvec=bvec, pe_axis=pe_axis, pe_sign=float(pe_app[pe_axis]), pe_directions=(pe, side["PhaseEncodingDirection"]),
                   readout_s=NOMINAL_READOUT_S, b0s=np.concatenate([dwi[..., b0i].astype(np.float64), rvol[..., take]], -1),
                   pe_vectors=np.array([pe_app] * len(b0i) + [pe_apa] * len(take)), series=series, notes=notes)
