"""A BIDS diffusion series as pipeline.run's subject, with the series that correct it found and checked.

    from tractline import bids, pipeline as P
    s = bids.load("sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz")
    print(s.pairing.message)                     # what will correct it, or why nothing will
    corr, tg, labels = P.run(s, None, P.Timer())   # corrects when s.pairing.ok; otherwise tracks as acquired

Needs nibabel and scipy (`tractline[bids]`). DICOM reaches it through dcm2niix (`dcm2niix -b y -z y`): its
NIfTI, .bval/.bvec and JSON sidecars are this module's input.

The series that measure the field (`pairing.series`): the diffusion series' own b0s (always first: volume
0 is the estimate's fixed reference, so the field is in the diffusion series' frame), then, in this order
of evidence:
  1. every series whose sidecar's B0FieldIdentifier is one the diffusion sidecar's B0FieldSource names;
  2. else every fmap/*_epi series whose IntendedFor names this diffusion series;
  3. else every other diffusion series in the same folder phase-encoded along the same axis, opposite way
     (ds001226's acq-AP / acq-PA);
or `load(..., partners=[paths])` to say it. A partner's b0s (all its volumes, for an EPI field map) are put
onto the diffusion series' grid by the scanner's coordinates (cubic) when the two grids differ.

Checked before the pair is used - refused when a check fails, the reason in pairing.message (tractline's
NOTES 2026-10-03: a re-shimmed pair gave a confident, wrong 10 mm field; absolute polarity does not
matter, relative polarity does; a nominal readout time is safe within one protocol only):
  - every partner phase-encodes along the diffusion series' axis (in the world, within 10 degrees), and
    the series together cover both polarities;
  - ShimSetting identical to the diffusion series' (when both sidecars state it; unknown otherwise, said);
  - TotalReadoutTime stated; when a partner's is missing it takes the diffusion series' only if the
    protocols match (echo time within 1 ms, voxel sizes within 0.01 mm), said as assumed;
  - echo time within 1 ms and voxel sizes within 0.01 mm (a different contrast or resolution is refused).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

AXES = {"i": 0, "j": 1, "k": 2}


def _vec(code: str) -> np.ndarray:
    v = np.zeros(3)
    v[AXES[code[0]]] = -1.0 if code.endswith("-") else 1.0
    return v


def _sidecar(nii: Path) -> dict:
    p = Path(str(nii).replace(".nii.gz", ".json").replace(".nii", ".json"))
    return json.loads(p.read_text()) if p.exists() else {}


def _stem(nii: Path) -> str:
    return nii.name.replace(".nii.gz", "").replace(".nii", "")


@dataclass
class Check:
    name: str
    status: str                  # "ok", "failed", "unknown" (not stated: proceeds, said)
    detail: str = ""


@dataclass
class Pairing:
    ok: bool                     # the field can be estimated and applied
    how: str                     # "B0FieldIdentifier", "IntendedFor", "same folder", "given", or "none"
    series: list                 # partner NIfTI paths
    checks: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)    # field -> "stated" / "assumed"
    message: str = ""


@dataclass
class Subject:
    """pipeline.run's subject (pipeline.py's docstring), from BIDS."""
    name: str
    dwi: np.ndarray              # (X, Y, Z, G) as stored
    affine: np.ndarray
    vox: np.ndarray
    bval: np.ndarray
    bvec: np.ndarray             # (3, G), FSL
    pe_axis: int
    pe_sign: float
    readout_s: float             # the diffusion series'
    pairing: Pairing
    b0s: np.ndarray | None       # (X, Y, Z, V) float64, the diffusion b0s first; None when not pairing.ok
    pe_vectors: np.ndarray | None
    b0_readout_s: np.ndarray | None   # (V,), each b0's series' readout time
    dwi_path: Path
    sidecar: dict


def _candidates(dwi: Path, side: dict) -> tuple[str, list[Path]]:
    folder = dwi.parent
    session = folder.parent
    niis = sorted(p for p in session.glob("*/*.nii*") if p != dwi)
    src = side.get("B0FieldSource")
    if src:
        want = {src} if isinstance(src, str) else set(src)
        hits = []
        for p in niis:
            ident = _sidecar(p).get("B0FieldIdentifier")
            have = {ident} if isinstance(ident, str) else set(ident or [])
            if want & have:
                hits.append(p)
        if hits:
            return "B0FieldIdentifier", hits
    hits = []
    for p in niis:
        if p.parent.name != "fmap" or not _stem(p).endswith("_epi"):
            continue
        intended = _sidecar(p).get("IntendedFor", [])
        intended = [intended] if isinstance(intended, str) else intended
        if any(str(i).split(":")[-1].endswith(dwi.name) for i in intended):
            hits.append(p)
    if hits:
        return "IntendedFor", hits
    pe = side.get("PhaseEncodingDirection")
    hits = [p for p in sorted(folder.glob("*_dwi.nii*")) if p != dwi and pe
            and _sidecar(p).get("PhaseEncodingDirection", "")[:1] == pe[:1]
            and _sidecar(p).get("PhaseEncodingDirection") != pe]
    return ("same folder", hits) if hits else ("none", [])


def load(dwi_path, partners=None, name=None) -> Subject:
    """The diffusion series at dwi_path (a BIDS *_dwi.nii[.gz] with .bval, .bvec and .json beside it) and
    the series that correct it (found as the module docstring says, or `partners`, NIfTI paths)."""
    import nibabel as nib
    from scipy.ndimage import affine_transform
    dwi_path = Path(dwi_path)
    stem = str(dwi_path).replace(".nii.gz", "").replace(".nii", "")
    side = _sidecar(dwi_path)
    img = nib.load(dwi_path)
    A = img.affine
    dwi = np.asarray(img.dataobj)
    vox = np.asarray(img.header.get_zooms()[:3], float)
    bval, bvec = np.loadtxt(stem + ".bval", ndmin=1), np.loadtxt(stem + ".bvec", ndmin=2)
    if "PhaseEncodingDirection" not in side:
        raise ValueError(f"{dwi_path}: the sidecar states no PhaseEncodingDirection (dcm2niix writes it when the "
                         "DICOM records the polarity; PhaseEncodingAxis alone is not enough to correct)")
    pe0 = _vec(side["PhaseEncodingDirection"])
    pe_axis = int(np.argmax(np.abs(pe0)))
    readout = side.get("TotalReadoutTime")
    how, cands = ("given", [Path(p) for p in partners]) if partners is not None else _candidates(dwi_path, side)
    checks, sources = [], {"diffusion TotalReadoutTime": "stated" if readout is not None else "missing"}
    b0i = np.flatnonzero(bval < 50)
    vols, pevs, ros = [dwi[..., b0i].astype(np.float64)], [pe0] * len(b0i), [readout] * len(b0i)
    world = lambda M, v: (M[:3, :3] @ v) / np.linalg.norm(M[:3, :3] @ v)
    w0 = world(A, pe0)
    failed = []
    for p in cands:
        ps, pimg = _sidecar(p), nib.load(p)
        label = p.name
        if "PhaseEncodingDirection" not in ps:
            failed.append(f"{label}: no PhaseEncodingDirection"); continue
        pv = _vec(ps["PhaseEncodingDirection"])
        w = world(pimg.affine, pv)
        cos = float(w @ w0)
        if abs(cos) < np.cos(np.radians(10)):
            checks.append(Check(f"{label}: phase-encoding axis", "failed", f"{np.degrees(np.arccos(min(1, abs(cos)))):.0f} deg from the diffusion series'"))
            failed.append(f"{label}: phase-encoded along another axis"); continue
        checks.append(Check(f"{label}: phase-encoding axis", "ok", "same axis, " + ("opposite" if cos < 0 else "same") + " polarity"))
        s0, s1 = side.get("ShimSetting"), ps.get("ShimSetting")
        if s0 is None or s1 is None:
            checks.append(Check(f"{label}: shim", "unknown", "not stated in both sidecars"))
        elif list(s0) != list(s1):
            checks.append(Check(f"{label}: shim", "failed", f"ShimSetting {s1} differs from the diffusion series' {s0}"))
            failed.append(f"{label}: acquired under another shim (the field it measures is not the diffusion series')"); continue
        else:
            checks.append(Check(f"{label}: shim", "ok", "identical ShimSetting"))
        te0, te1 = side.get("EchoTime"), ps.get("EchoTime")
        pvox = np.asarray(pimg.header.get_zooms()[:3], float)
        same_vox = bool(np.allclose(np.sort(pvox), np.sort(vox), atol=0.01))
        same_te = te0 is not None and te1 is not None and abs(te0 - te1) <= 0.001
        if not same_vox or (te0 is not None and te1 is not None and not same_te):
            checks.append(Check(f"{label}: protocol", "failed", f"echo time {te1} vs {te0} s, voxels {pvox.round(3).tolist()} vs {vox.round(3).tolist()} mm"))
            failed.append(f"{label}: a different protocol (echo time or voxel size)"); continue
        checks.append(Check(f"{label}: protocol", "ok" if same_te else "unknown",
                            "echo time and voxel sizes match" if same_te else "voxel sizes match; echo time not stated in both"))
        ro = ps.get("TotalReadoutTime")
        if ro is None:
            if readout is None or not same_te:
                failed.append(f"{label}: no TotalReadoutTime, and none can be assumed"); continue
            ro = readout; sources[f"{label} TotalReadoutTime"] = "assumed (the diffusion series', protocols matching)"
        else:
            sources[f"{label} TotalReadoutTime"] = "stated"
        data = np.asarray(pimg.dataobj, dtype=np.float64)
        data = data[..., None] if data.ndim == 3 else data
        if _stem(p).endswith("_dwi"):
            pb = np.loadtxt(str(p).replace(".nii.gz", ".bval").replace(".nii", ".bval"), ndmin=1)
            data = data[..., pb < 50]
        if not (np.allclose(A, pimg.affine, atol=1e-3) and data.shape[:3] == dwi.shape[:3]):
            M = np.linalg.inv(pimg.affine) @ A                               # diffusion voxel -> partner voxel
            data = np.stack([affine_transform(data[..., v], M[:3, :3], M[:3, 3], output_shape=dwi.shape[:3], order=3,
                                              mode="constant", cval=0.0) for v in range(data.shape[-1])], -1).clip(0, None)
            pv = np.sign(cos) * pe0                                           # its direction on the diffusion grid
        vols.append(data); pevs += [pv] * data.shape[-1]; ros += [ro] * data.shape[-1]
    pe_vectors = np.array(pevs)
    both = bool(len(cands)) and len({float(v[pe_axis]) for v in pe_vectors}) == 2
    if readout is None and not failed:
        failed.append("the diffusion series states no TotalReadoutTime")
    if not cands:
        others = [f"{p.name} ({_sidecar(p).get('PhaseEncodingDirection', '?')})" for p in sorted(dwi_path.parent.glob("*_dwi.nii*"))
                  if p != dwi_path]
        msg, ok = ("not corrected: no series to measure the field with (no B0FieldIdentifier/IntendedFor partner, and no "
                   f"diffusion series in the folder phase-encoded the other way along {'ijk'[pe_axis]}"
                   + (f"; the folder's others: {', '.join(others)}" if others else "") + ")"), False
    elif failed:
        msg, ok = "not corrected: " + "; ".join(failed), False
    elif not both:
        msg, ok = "not corrected: the series all phase-encode one way - a field needs both polarities", False
    else:
        unknown = [c.name for c in checks if c.status == "unknown"]
        msg, ok = (f"corrected with {', '.join(p.name for p in cands)} ({how})"
                   + (f"; not verified: {', '.join(unknown)}" if unknown else "")), True
    pairing = Pairing(ok, how, cands, checks, sources, msg)
    return Subject(name=name or _stem(dwi_path), dwi=dwi, affine=A, vox=vox, bval=bval, bvec=bvec, pe_axis=pe_axis,
                   pe_sign=float(pe0[pe_axis]), readout_s=readout if readout is not None else float("nan"), pairing=pairing,
                   b0s=np.concatenate(vols, -1) if ok else None, pe_vectors=pe_vectors if ok else None,
                   b0_readout_s=np.array(ros, float) if ok else None, dwi_path=dwi_path, sidecar=side)
