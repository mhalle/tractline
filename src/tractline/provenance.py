"""What produced an output: a run's record, written beside its TRX as <name>.tractline.json.

Two identities, for two questions:

  pipeline_id    "what algorithm was asked for": the config's sha256 (PipelineConfig.hash()) - its slots, their
                 implementations, versions and options. The device is left out on purpose
                 (docs/sota-pipeline-design.md, decision 6), so a calibration map keyed on it carries across devices.
  execution_id   "what computation produced these numbers": a sha256 over the pipeline_id and what else changes them -
                 the device type and model (the GPU's name, or the CPU's), the operating system's version and build
                 (an update can change Metal's compiler), on CUDA the runtime and driver versions, the labeler's
                 device, the backend each step ran (Metal, Triton or torch), the CPU tracker's compiled step, the
                 versions of tractline (and, in a checkout, its git commit with a sha256 of any uncommitted change),
                 torch, numpy, scipy, Python, Triton and (with the denoise slot) dipy, and the sha256 of the files the
                 labeler actually loaded. Runs on the same input with the same execution_id
                 are expected to give the same streamlines (the TRX's bytes also depend on its positions' type and
                 labeled_only, recorded with the output); ones with the same pipeline_id but another execution_id run the same algorithm on other
                 arithmetic (a CPU and an Apple GPU agree on 93.8 % of shared seeds' tracts, not bit for bit:
                 docs/sota-pipeline-design.md, decision 6).

The record also holds: each step's implementation status (validated, experimental, wrapper) and the tool it is checked
against; the runtime (versions, device, workers, backends); the inputs (path, size and sha256 of the series, its
.bval, .bvec and sidecar, and of the series that measured the field); the pairing decision; the correction's note and
warnings; the stage times; the output (path, size, sha256, streamline counts); and, in a checkout, the sha256 of
uv.lock (the environment `uv sync --locked` reproduces).

Files are identified by their sha256. Their paths are recorded relative to the BIDS dataset they are in (with the
dataset folder's name), or as the file's name outside one, so the record can be shared without the machine's folder
names (a user name, a project, an accession-like folder). absolute_paths=True (the command's --record-absolute-paths)
records full paths instead, for local bookkeeping.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tractline.provenance/2"      # /2 (0.6.1): shareable paths, weights as loaded, the driver


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(trx) -> Path:
    """out/sub-01.trx (a zip) or out/sub-01 (a directory) -> out/sub-01.tractline.json."""
    p = Path(trx)
    return p.with_name((p.stem if p.suffix == ".trx" else p.name) + ".tractline.json")


def _version(dist: str):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=1)
def _cpu_model() -> str:
    """The processor's model ("Apple M2", "AMD EPYC 7R13 ..."), where the platform says; else its architecture."""
    import subprocess
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            if out:
                return out
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


@functools.lru_cache(maxsize=1)
def _os_build() -> str | None:
    """The operating system's build: macOS's (sw_vers; a point update can change Metal's compiler), else the kernel's."""
    import subprocess
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return platform.version() or None


@functools.lru_cache(maxsize=1)
def _cuda_driver() -> str | None:
    """The CUDA driver's version, asked of the driver library itself (cuDriverGetVersion: "12.4" for 12040): no
    subprocess, no timeout, the same answer every time on a machine. None where the library cannot be loaded."""
    import ctypes
    import ctypes.util
    for name in ("libcuda.so.1", "libcuda.so", ctypes.util.find_library("cuda") or "", "nvcuda.dll"):
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        v = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(v)) == 0 and v.value > 0:
            return f"{v.value // 1000}.{(v.value % 1000) // 10}"
    return None


def device_info(device) -> dict:
    """The device and its software: a CUDA GPU's name, compute capability, CUDA runtime and driver; on an Apple GPU or
    the CPU, the chip; on all, the operating system's version and build."""
    import torch
    d = torch.device(device)
    out = {"type": d.type}
    if d.type == "cuda":
        idx = d.index if d.index is not None else torch.cuda.current_device()
        out["name"] = torch.cuda.get_device_name(idx)
        out["compute capability"] = ".".join(map(str, torch.cuda.get_device_capability(idx)))
        out["cuda runtime"] = torch.version.cuda
        out["driver"] = _cuda_driver()
    else:
        out["name"] = _cpu_model()                       # an Apple GPU is the chip's: M1, M2 and M4 differ
    out["os"] = f"{platform.system()} {platform.mac_ver()[0] or platform.release()}"
    out["os build"] = _os_build()
    return out


@functools.lru_cache(maxsize=1)
def _git() -> dict | None:
    """In a checkout: the commit and whether files differ from it (a dev build shares its version number with the
    release it follows, so the commit tells them apart). Once per process, as the chip's model: a bench running the
    pipeline many times does not run git each time."""
    import subprocess
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if head.returncode != 0:
        return None
    out = {"commit": head.stdout.strip(), "modified": bool(dirty.stdout.strip())}
    try:                                                 # what the changes are: two dirty runs of other code differ
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--binary"], capture_output=True, timeout=20)
        new = subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z", "src"],
                             capture_output=True, timeout=10)
        h = hashlib.sha256(diff.stdout)
        for name in sorted(n for n in new.stdout.decode().split("\0") if n):
            h.update(name.encode() + b"\0")
            h.update((root / name).read_bytes())
        if out["modified"] or new.stdout.strip(b"\0"):
            out["modified"] = True
            out["changes sha256"] = h.hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def weights(labeler, absolute=False) -> list:
    """The files the labeler actually loaded, each with its sha256 (computed, not taken from the pins) and whether it
    is the pinned release's ("pinned": false flags a local copy that differs, as a modified TractCloud checkout in
    $TRACTOGRAPHY_DATA would); a caller's labeler: its class."""
    if labeler is None:
        return []
    from . import fetch
    mod = type(labeler).__module__
    out = []
    if mod == "tractline.labelers.rapidparc":
        name = getattr(labeler, "model_name", "rapidparc")
        files = [(f"RapidParc {name}", fetch.weights(name), fetch.WEIGHTS[name][1])]
    elif mod == "tractline.labelers.tractcloud":
        import sys
        tc = sys.modules[mod]
        src = Path(tc._SOURCE) / "tractcloud"
        files = [(f"TractCloud {k}", src / k, sha) for k, (sha, _) in fetch.TRACTCLOUD_SOURCE.items()]
        pins = {Path(rel).name: sha for _, _, wanted in fetch.TRACTCLOUD_ARCHIVES.values() for sha, rel in wanted.values()}
        files += [(f"TractCloud {f.name}", f, pins[f.name]) for f in (Path(tc.MODEL) / "best_tract_f1_model.pth",
                                                                    Path(tc.MODEL) / "cli_args.txt", Path(tc.MASS_CENTER))]
    else:
        return [{"name": f"{mod}.{type(labeler).__qualname__}", "sha256": None}]
    for label, path, pin in files:
        rec = _file(path, absolute) or {"path": str(path), "sha256": None}
        out.append({"name": label, **rec, "pinned": rec["sha256"] == pin})
    return out


def _where(path, absolute=False) -> dict:
    """Where a file is, as the record says it: relative to its BIDS dataset (and the dataset folder's name), or its
    name alone outside one; absolute=True: the full path."""
    p = Path(os.path.abspath(path))                     # not resolved: a DataLad/annex link keeps its dataset path
    if absolute:
        return {"path": str(p)}
    from .bids import _root
    root = _root(p)
    if root is not None:
        return {"dataset": root.name, "path": p.relative_to(root).as_posix()}
    return {"path": p.name}


def _file(path, absolute=False) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    return {**_where(p, absolute), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def inputs(s, absolute=False) -> list:
    """The files the subject came from: the diffusion series with its .bval, .bvec and sidecar, then the series that
    measured the field; empty for a subject built from arrays."""
    dwi = getattr(s, "dwi_path", None)
    if dwi is None:
        return []
    from .bids import _inherited
    out = []
    for i, nii in enumerate([Path(dwi)] + [Path(p) for p in getattr(getattr(s, "pairing", None), "series", []) or []]):
        role = "diffusion series" if i == 0 else "field series"
        for f in [nii] + [g for ext in (".bval", ".bvec", ".json") for g in _inherited(nii, ext)]:
            rec = _file(f, absolute)                     # the series, then its tables and sidecars, inherited ones too
            if rec is not None:
                out.append({"role": role, **rec})
    return out


def statuses(config: dict) -> dict:
    """Each slot that ran: its implementation, version, status and the tool it is checked against."""
    from .config import implementation
    out = {}
    for slot, e in config["slots"].items():
        impl = implementation(slot, e["impl"])
        out[slot] = {"impl": e["impl"], "version": e["version"], "status": impl.status, "oracle": impl.oracle}
    return out


def experimental(config: dict) -> list:
    """The steps a run used that are marked experimental, as 'slot: impl (checked against oracle)'."""
    return [f"{slot}: {v['impl']}" + (f"; checked against {v['oracle']}" if v["oracle"] else "")
            for slot, v in statuses(config).items() if v["status"] == "experimental"]


def execution_id(pipeline_id: str, runtime: dict, device: dict, weights_: list) -> str:
    """sha256 over the pipeline_id and what changes a run's numbers beyond it (the module docstring)."""
    keep = {k: runtime.get(k) for k in ("tractline", "tractline commit", "torch", "numpy", "scipy", "dipy", "python",
                                        "triton", "torch_compile", "labeler device", "ukf backend",
                                        "motion_eddy backend", "ptt backend")}
    body = {"pipeline_id": pipeline_id, "device": {k: device.get(k) for k in ("type", "name", "compute capability",
                                                                             "cuda runtime", "driver", "os", "os build")},
            "runtime": keep, "weights": sorted(w.get("sha256") or w["name"] for w in weights_)}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _lock(absolute=False) -> dict | None:
    """In a checkout: uv.lock's sha256 (the environment `uv sync --locked` reproduces)."""
    root = Path(__file__).resolve().parents[2]
    lock, proj = root / "uv.lock", root / "pyproject.toml"
    if lock.is_file() and proj.is_file() and 'name = "tractline"' in proj.read_text():
        return {**({"path": str(lock)} if absolute else {}), "sha256": sha256_file(lock)}
    return None


def _digest(path, absolute=False) -> dict:
    """A written TRX: a zip's size and sha256; a directory's total size and a sha256 over its files' paths and hashes."""
    p = Path(path)
    if p.is_file():
        return _file(p, absolute)
    files = sorted(f for f in p.rglob("*") if f.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(p).as_posix()}\0{sha256_file(f)}\n".encode())
    return {**_where(p, absolute), "bytes": sum(f.stat().st_size for f in files), "sha256": h.hexdigest(),
            "sha256 of": "the directory's files (path and sha256 of each, in order)"}


def record(run, s, device, labeler, timer=None, trx=None, trx_options=None, absolute_paths=False) -> dict:
    """The run's record (the module docstring). run: pipeline.Run; s: its subject; trx: the written TRX, if any, and
    trx_options what it was written with (positions, labeled_only); absolute_paths: full paths in place of the
    shareable ones."""
    config, runtime = run.config, dict(run.runtime)
    if runtime.get("device", "").startswith("cuda"):
        runtime["triton"] = _version("triton")
    git = _git()
    if git is not None:
        runtime["tractline commit"] = git["commit"] + (f"+modified:{git.get('changes sha256', '?')}"
                                                      if git["modified"] else "")
    runtime["scipy"] = _version("scipy")
    if config["slots"].get("denoise"):
        runtime["dipy"] = _version("dipy")
    if labeler is not None and getattr(labeler, "device", None) is not None:
        import torch
        ld = torch.device(labeler.device)
        if ld.type == "cuda" and ld.index is None:                         # "cuda" and "cuda:0": one device, one id
            ld = torch.device("cuda", torch.cuda.current_device())
        runtime["labeler device"] = str(ld)
    dev = device_info(device)
    w = weights(labeler, absolute_paths)
    pid = config.get("sha256")
    if pid is None:                                      # a caller's labeler object: the config is not hashable
        pid = "unhashed:" + hashlib.sha256(json.dumps({k: v for k, v in config.items() if k != "sha256"},
                                                      sort_keys=True, default=str).encode()).hexdigest()
    corr = run.correction
    pairing = getattr(s, "pairing", None)
    rec = {"schema": SCHEMA, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "pipeline_id": pid, "execution_id": execution_id(pid, runtime, dev, w),
           "pipeline": config, "steps": statuses(config), "experimental": experimental(config),
           "runtime": runtime, "device": dev, "weights": w,
           "packages": {d: _version(d) for d in ("scipy", "nibabel", "dipy", "scikit-learn", "click")
                        if _version(d) is not None},
           "environment": {"platform": platform.platform(), "uv.lock": _lock(absolute_paths), "git": git,
                           "TRACTLINE_COMPILE": os.environ.get("TRACTLINE_COMPILE")},
           "inputs": inputs(s, absolute_paths), "paths": "absolute" if absolute_paths else "shareable",
           "pairing": None if pairing is None else {"ok": pairing.ok, "how": pairing.how, "message": pairing.message,
                                                    "series": [_where(p, absolute_paths) for p in pairing.series]},
           "correction": {"applied": corr.applied, "note": corr.note, "warnings": list(corr.warnings or []),
                          "residual_left": corr.residual_left},
           "seconds": dict(timer.seconds) if timer is not None else None,
           "output": None}
    if trx is not None:
        opts = {"positions": "float32", "labeled_only": False, **(trx_options or {})}
        labels = run.labels
        labeled = None if labels is None else int(labels.keep.sum())
        rec["output"] = {**_digest(trx, absolute_paths), **opts,
                         "streamlines": labeled if opts["labeled_only"] else len(run.tractogram.fibers),
                         "labeled": labeled}
    from .motion_eddy import _finite
    out = _finite(json.loads(json.dumps(rec, default=str)))
    return out if absolute_paths else _scrub(out, s, trx)


def _scrub(rec, s, trx):
    """A shareable record: the folders above the inputs' datasets (or the inputs and the TRX themselves, outside one)
    and the home folder taken out of every string, so a path quoted in a message (a sidecar's error, a note) does not
    carry them."""
    from .bids import _root
    prefixes = set()
    for f in [getattr(s, "dwi_path", None), *(getattr(getattr(s, "pairing", None), "series", None) or []), trx]:
        if f is None:
            continue
        p = Path(os.path.abspath(f))
        root = _root(p)
        prefixes.add(str(root.parent if root is not None else p.parent) + os.sep)
    home = str(Path.home())

    def clean(v):
        if isinstance(v, str):
            for pre in sorted(prefixes, key=len, reverse=True):
                v = v.replace(pre, "")
            return v.replace(home + os.sep, "~" + os.sep)
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return clean(rec)


def write(path, rec: dict) -> Path:
    """The record as JSON at path, written whole (a temporary file beside it, renamed into place)."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)                    # as an ordinary file would be (mkstemp makes it 0600)
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f, indent=1, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
