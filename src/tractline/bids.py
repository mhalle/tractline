"""A BIDS diffusion series as pipeline.run's subject, with the series that correct it found and checked.

    from tractline import bids, pipeline as P
    s = bids.load("sub-01/ses-1/dwi/sub-01_ses-1_dwi.nii.gz")
    print(s.pairing.message)                     # what will correct it, or why nothing will
    corr, tg, labels = P.run(s, None, P.Timer())   # corrects when s.pairing.ok; otherwise tracks as acquired

Needs nibabel and scipy (`tractline[bids]`). DICOM reaches it through dcm2niix's output, organized as BIDS
(dcm2bids or heudiconv do that; dcm2niix alone writes flat names - then name the partners with `partners=`).
Inside a BIDS dataset (a folder above with dataset_description.json), sidecar fields and .bval/.bvec follow
BIDS's inheritance principle: files nearer the root, whose entities are a subset of the series' and whose
suffix is the same, are merged under the series' own. Outside one, only the files beside it of its own name.

The series that measure the field (`pairing.series`): the diffusion series' own b0s (always first: volume
0 is the estimate's fixed reference, so the field is in the diffusion series' frame), then, in this order
of evidence:
  1. the EPI series (_epi, _dwi, _sbref, _bold) whose B0FieldIdentifier is one the diffusion sidecar's
     B0FieldSource names (GRE field maps - phasediff, magnitude, fieldmap - are refused: not supported);
  2. else every fmap/*_epi series whose IntendedFor names this diffusion series;
  3. else every other diffusion series in the same folder phase-encoded along the same world axis, the
     other way (ds001226's acq-AP / acq-PA);
or `load(..., partners=[paths])` to say it. Candidates of the diffusion series' own run (or with no run)
are preferred. A partner on another grid is put onto the diffusion series' by the scanner's coordinates
(cubic).

Checked for each partner (tractline's NOTES 2026-10-03: on OpenNeuro ds005123 a pair acquired under
different shims gave a confident, wrong field - 9-22 mm at the 99th percentile where the field maps' own pair
gave 5-7 mm; absolute polarity does not matter, relative polarity does; a nominal readout time is safe
within one protocol only):
  - its phase-encoding axis is the diffusion series' in the world (within 10 degrees);
  - ShimSetting identical (when both sidecars state it; "not verified" otherwise, said);
  - echo time within 1 ms and voxel sizes within 0.01 mm along each world axis - strict: a spin-echo field
    map of another echo time or resolution, which topup users often pair, is refused here;
  - it covers the diffusion series' brain (99 % of its bright voxels) once on its grid;
  - it has b0 volumes (an EPI field map: all its volumes);
  - a readout time: TotalReadoutTime, else EstimatedTotalReadoutTime (said as estimated), else the
    diffusion series' only when the protocols match (echo time, voxel sizes, the matrix along the
    phase-encoding axis, ParallelReductionFactorInPlane and PartialFourier where stated; said as assumed).
A partner that fails is dropped when the others still cover both polarities (said), else the pairing is
refused with the reasons.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

AXES = {"i": 0, "j": 1, "k": 2}
EPI = {"epi", "dwi", "sbref", "bold"}
GRE = {"phasediff", "phase1", "phase2", "magnitude", "magnitude1", "magnitude2", "fieldmap"}


def _stem(p: Path) -> str:
    return p.name[:-7] if p.name.endswith(".nii.gz") else p.name[:-4] if p.name.endswith(".nii") else p.stem


def _sib(p: Path, ext: str) -> Path:
    return p.with_name(_stem(p) + ext)


def _entities(stem: str) -> tuple[dict, str]:
    parts = stem.split("_")
    return dict(x.split("-", 1) for x in parts[:-1] if "-" in x), parts[-1]


def _root(p: Path) -> Path | None:
    """The BIDS dataset p is in (the nearest folder above it with dataset_description.json), or None."""
    for d in p.parents:
        if (d / "dataset_description.json").exists():
            return d
    return None


def _inherited(p: Path, ext: str) -> list[Path]:
    """The files of this extension that apply to p, least specific first: inside a BIDS dataset, by BIDS's
    inheritance principle; outside one (dcm2niix's flat output), only the file beside it of its own name."""
    root = _root(p)
    if root is None:
        f = _sib(p, ext)
        return [f] if f.exists() else []
    ents, suffix = _entities(_stem(p))
    dirs = [root] + [d for d in reversed(p.parents) if root in d.parents]
    out = []
    for d in dirs:
        found = []
        for f in d.glob(f"*{ext}"):
            fe, fs = _entities(f.name[: -len(ext)])
            if fs == suffix and all(ents.get(k) == v for k, v in fe.items()):
                found.append((len(fe), f))
        out += [f for _, f in sorted(found)]
    return out


def _sidecar(p: Path) -> dict:
    merged = {}
    for f in _inherited(p, ".json"):
        try:
            merged.update(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    return merged


def _table(p: Path, ext: str):
    files = _inherited(p, ext)
    return np.loadtxt(files[-1], ndmin=1 if ext == ".bval" else 2) if files else None


def _pe(code) -> np.ndarray | None:
    """A BIDS PhaseEncodingDirection ("j", "i-", ...) as a voxel-axis vector; None when not one."""
    if not isinstance(code, str) or code.rstrip("-") not in AXES or len(code) > 2:
        return None
    v = np.zeros(3)
    v[AXES[code[0]]] = -1.0 if code.endswith("-") else 1.0
    return v


def _world(M, v):
    w = M[:3, :3] @ v
    return w / np.linalg.norm(w)


def _world_voxels(M, zooms):
    """The voxel size along each world axis (the image axis nearest to it)."""
    D = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
    return np.array([zooms[int(np.argmax(np.abs(D[w])))] for w in range(3)], float)


@dataclass
class Check:
    name: str
    status: str                  # "ok", "failed", "unknown" (not stated: proceeds, said)
    detail: str = ""


@dataclass
class Pairing:
    ok: bool                     # the field can be estimated and applied
    how: str                     # "B0FieldIdentifier", "IntendedFor", "same folder", "given", or "none"
    series: list                 # the partner NIfTI paths used
    checks: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)    # field -> "stated" / "estimated" / "assumed (...)"
    message: str = ""
    dropped: list = field(default_factory=list)    # partners found but not used, with the reason


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
    readout_s: float             # the diffusion series' (NaN when it states none: then pairing is not ok)
    pairing: Pairing
    b0s: np.ndarray | None       # (X, Y, Z, V) float64, the diffusion b0s first; None when not pairing.ok
    pe_vectors: np.ndarray | None
    b0_readout_s: np.ndarray | None   # (V,), each b0's series' readout time
    dwi_path: Path
    sidecar: dict


def _readout(side) -> tuple[float | None, str]:
    if side.get("TotalReadoutTime") is not None:
        return float(side["TotalReadoutTime"]), "stated"
    if side.get("EstimatedTotalReadoutTime") is not None:
        return float(side["EstimatedTotalReadoutTime"]), "estimated (EstimatedTotalReadoutTime)"
    return None, "missing"


def _names(entry, dwi: Path) -> bool:
    """Whether an IntendedFor entry names this diffusion series (any path form, extension or none)."""
    tail = str(entry).split(":")[-1].strip("/")
    name = Path(tail).name
    name = name[:-7] if name.endswith(".nii.gz") else name[:-4] if name.endswith(".nii") else name
    parent = Path(tail).parent.name
    return name == _stem(dwi) and parent in ("", dwi.parent.name)


def _candidates(dwi: Path, side: dict, w0: np.ndarray) -> tuple[str, list[Path], list[str]]:
    import nibabel as nib
    folder = dwi.parent
    session = folder.parent
    niis = sorted(p for p in session.glob("*/*.nii*") if p != dwi)
    notes, refused = [], []
    src = side.get("B0FieldSource")
    if src:
        want = {src} if isinstance(src, str) else set(src)
        hits = []
        for p in niis:
            ident = _sidecar(p).get("B0FieldIdentifier")
            have = {ident} if isinstance(ident, str) else set(ident or [])
            if want & have:
                suffix = _entities(_stem(p))[1]
                if suffix in GRE:
                    refused.append(f"{p.name}: a GRE field map ({suffix}) - not supported")
                elif suffix in EPI:
                    hits.append(p)
        if hits or refused:
            return "B0FieldIdentifier", hits, refused
        notes.append(f"B0FieldSource {sorted(want)} names no series here")
    def intended(p):
        x = _sidecar(p).get("IntendedFor", [])
        return any(_names(i, dwi) for i in ([x] if isinstance(x, str) else x))
    hits = [p for p in niis if p.parent.name == "fmap" and _entities(_stem(p))[1] == "epi" and intended(p)]
    if hits:
        return "IntendedFor", hits, notes
    hits = []
    for p in sorted(folder.glob("*_dwi.nii*")):
        v = _pe(_sidecar(p).get("PhaseEncodingDirection")) if p != dwi else None
        if v is not None and float(_world(nib.load(p).affine, v) @ w0) < -np.cos(np.radians(10)):
            hits.append(p)
    return ("same folder", hits, notes) if hits else ("none", [], notes)


def load(dwi_path, partners=None, name=None) -> Subject:
    """The diffusion series at dwi_path (a BIDS *_dwi.nii[.gz] with its .bval, .bvec and .json, inherited or
    beside it) and the series that correct it (found as the module docstring says, or `partners`, NIfTI
    paths)."""
    try:
        import nibabel as nib
        from scipy.ndimage import affine_transform
    except ImportError as e:
        raise ImportError("tractline.bids needs nibabel and scipy: install tractline[bids]") from e
    from .mask import otsu
    dwi_path = Path(dwi_path)
    side = _sidecar(dwi_path)
    img = nib.load(dwi_path)
    A = img.affine
    shape = img.shape[:3]
    bval, bvec = _table(dwi_path, ".bval"), _table(dwi_path, ".bvec")
    if bval is None or bvec is None:
        raise ValueError(f"{dwi_path}: no .bval/.bvec (beside it or inherited)")
    G = img.shape[3] if len(img.shape) > 3 else 1
    if len(bval) != G or bvec.shape != (3, G):
        raise ValueError(f"{dwi_path}: {G} volumes, {len(bval)} b-values, bvec {bvec.shape}")
    b0i = np.flatnonzero(bval < 50)
    if not len(b0i):
        raise ValueError(f"{dwi_path}: no b0 volume (b < 50) - the tracker and the correction need them")
    pe0 = _pe(side.get("PhaseEncodingDirection"))
    if pe0 is None:
        raise ValueError(f"{dwi_path}: no usable PhaseEncodingDirection in its sidecar ({side.get('PhaseEncodingDirection')!r}: "
                         "i, j or k with an optional -; PhaseEncodingAxis alone is not enough to correct)")
    dwi = np.asarray(img.dataobj)
    vox = np.asarray(img.header.get_zooms()[:3], float)
    pe_axis = int(np.argmax(np.abs(pe0)))
    w0 = _world(A, pe0)
    readout, ro_src = _readout(side)
    if partners is not None:
        how, cands, notes = "given", [Path(p) for p in partners], []
    else:
        how, cands, notes = _candidates(dwi_path, side, w0)
    run = _entities(_stem(dwi_path))[0].get("run")
    if run is not None and len(cands) > 1:
        same = [p for p in cands if _entities(_stem(p))[0].get("run") in (None, run)]
        if same and len(same) < len(cands):
            notes.append(f"other runs left out: {', '.join(p.name for p in cands if p not in same)}")
            cands = same
    mean_b0 = dwi[..., b0i].astype(np.float64).mean(-1) if dwi.ndim == 4 else dwi.astype(np.float64)
    brain = mean_b0 > otsu(mean_b0)
    checks, sources = [], {"diffusion TotalReadoutTime": ro_src}
    used, dropped = [], [r for r in notes if ": a GRE field map" in r]
    vols, pevs, ros = [dwi[..., b0i].astype(np.float64)], [pe0] * len(b0i), [readout] * len(b0i)
    vox_w = _world_voxels(A, vox)

    def drop(p, why):
        dropped.append(f"{p.name}: {why}")

    for p in cands:
        ps, label = _sidecar(p), p.name
        try:
            pimg = nib.load(p)
        except Exception as e:                                               # noqa: BLE001
            drop(p, f"unreadable ({type(e).__name__})"); continue
        pv = _pe(ps.get("PhaseEncodingDirection"))
        if pv is None:
            drop(p, f"no usable PhaseEncodingDirection ({ps.get('PhaseEncodingDirection')!r})"); continue
        cos = float(_world(pimg.affine, pv) @ w0)
        if abs(cos) < np.cos(np.radians(10)):
            checks.append(Check(f"{label}: phase-encoding axis", "failed", f"{np.degrees(np.arccos(min(1, abs(cos)))):.0f} deg from the diffusion series'"))
            drop(p, "phase-encoded along another axis"); continue
        checks.append(Check(f"{label}: phase-encoding axis", "ok", "same axis, " + ("opposite" if cos < 0 else "same") + " polarity"))
        s0, s1 = side.get("ShimSetting"), ps.get("ShimSetting")
        s0, s1 = [None if s is None else [float(x) for x in np.atleast_1d(s)] for s in (s0, s1)]
        if s0 is None or s1 is None:
            checks.append(Check(f"{label}: shim", "unknown", "not stated in both sidecars"))
        elif s0 != s1:
            checks.append(Check(f"{label}: shim", "failed", f"ShimSetting {s1} differs from the diffusion series' {s0}"))
            drop(p, "acquired under another shim (the field it measures is not the diffusion series')"); continue
        else:
            checks.append(Check(f"{label}: shim", "ok", "identical ShimSetting"))
        te0, te1 = side.get("EchoTime"), ps.get("EchoTime")
        pzoom = np.asarray(pimg.header.get_zooms()[:3], float)
        same_vox = bool(np.allclose(_world_voxels(pimg.affine, pzoom), vox_w, atol=0.01))
        te_known = te0 is not None and te1 is not None
        same_te = te_known and abs(float(te0) - float(te1)) <= 0.001
        if not same_vox or (te_known and not same_te):
            checks.append(Check(f"{label}: protocol", "failed", f"echo time {te1} vs {te0} s, voxels {_world_voxels(pimg.affine, pzoom).round(3).tolist()} vs {vox_w.round(3).tolist()} mm (per world axis)"))
            drop(p, "a different protocol (echo time or voxel size)"); continue
        checks.append(Check(f"{label}: protocol", "ok" if same_te else "unknown",
                            "echo time and voxel sizes match" if same_te else "voxel sizes match; echo time not stated in both"))
        ro, src_ro = _readout(ps)
        if ro is None:
            pax = int(np.argmax(np.abs(pv)))
            same_matrix = pimg.shape[pax] == shape[pe_axis]
            same_accel = all(side.get(k) == ps.get(k) for k in ("ParallelReductionFactorInPlane", "PartialFourier")
                             if side.get(k) is not None and ps.get(k) is not None)
            if readout is None or not (same_te and same_matrix and same_accel):
                drop(p, "no TotalReadoutTime, and none can be assumed (the protocols are not shown to match)"); continue
            ro, src_ro = readout, "assumed (the diffusion series': echo time, voxels, matrix along phase encoding match)"
        sources[f"{label} TotalReadoutTime"] = src_ro
        if _entities(_stem(p))[1] == "dwi":
            pb = _table(p, ".bval")
            nvol = pimg.shape[3] if len(pimg.shape) > 3 else 1
            if pb is None or len(pb) != nvol:
                drop(p, "its .bval is missing or does not match its volumes"); continue
            idx = np.flatnonzero(pb < 50)
        else:
            idx = np.arange(pimg.shape[3] if len(pimg.shape) > 3 else 1)
        if not len(idx):
            drop(p, "no b0 volume (b < 50)"); continue
        data = np.stack([np.asarray(pimg.dataobj[..., int(v)] if len(pimg.shape) > 3 else pimg.dataobj, np.float64)
                         for v in idx], -1)
        if not (np.allclose(A, pimg.affine, atol=1e-3) and data.shape[:3] == shape):
            M = np.linalg.inv(pimg.affine) @ A                               # diffusion voxel -> partner voxel
            cover = affine_transform(np.ones(pimg.shape[:3]), M[:3, :3], M[:3, 3], output_shape=shape, order=0,
                                     mode="constant", cval=0.0) > 0.5
            frac = float(cover[brain].mean())
            if frac < 0.99:
                drop(p, f"covers {frac:.0%} of the diffusion series' brain on its grid (99 % needed)"); continue
            data = np.stack([affine_transform(data[..., v], M[:3, :3], M[:3, 3], output_shape=shape, order=3,
                                              mode="constant", cval=0.0) for v in range(data.shape[-1])], -1).clip(0, None)
            pv = (1.0 if cos > 0 else -1.0) * pe0                            # its direction on the diffusion grid
        vols.append(data); pevs += [pv] * data.shape[-1]; ros += [ro] * data.shape[-1]; used.append(p)
    pe_vectors = np.array(pevs)
    both = len({float(np.sign(v[pe_axis])) for v in pe_vectors}) == 2
    extra = (f"; not used: {'; '.join(dropped)}" if dropped else "") + (f"; {'; '.join(n for n in notes if n not in dropped)}"
                                                                        if [n for n in notes if n not in dropped] else "")
    if readout is None:
        msg, ok = "not corrected: the diffusion series states no TotalReadoutTime" + extra, False
    elif not cands:
        others = [f"{p.name} ({_sidecar(p).get('PhaseEncodingDirection', '?')})" for p in sorted(dwi_path.parent.glob("*_dwi.nii*"))
                  if p != dwi_path]
        msg, ok = ("not corrected: " + ("no partner series given" if how == "given" else
                   "no series to measure the field with (no B0FieldIdentifier/IntendedFor partner, and no diffusion series "
                   f"in the folder phase-encoded the other way along its axis{'; the folder' + chr(39) + 's others: ' + ', '.join(others) if others else ''})")
                   + extra), False
    elif not both:
        msg, ok = ("not corrected: " + ("; ".join(dropped) if dropped and not used else
                   "the series used all phase-encode one way - a field needs both polarities") + (extra if used else "")), False
    else:
        unknown = [c.name for c in checks if c.status == "unknown" and any(c.name.startswith(u.name) for u in used)]
        msg, ok = (f"corrected with {', '.join(p.name for p in used)} ({how})"
                   + (f"; not verified: {', '.join(unknown)}" if unknown else "") + extra), True
    pairing = Pairing(ok, how, used, checks, sources, msg, dropped)
    return Subject(name=name or _stem(dwi_path), dwi=dwi, affine=A, vox=vox, bval=bval, bvec=bvec, pe_axis=pe_axis,
                   pe_sign=float(pe0[pe_axis]), readout_s=readout if readout is not None else float("nan"), pairing=pairing,
                   b0s=np.concatenate(vols, -1) if ok else None, pe_vectors=pe_vectors if ok else None,
                   b0_readout_s=np.array(ros, float) if ok else None, dwi_path=dwi_path, sidecar=side)
