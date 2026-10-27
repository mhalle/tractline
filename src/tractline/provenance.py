"""What produced an output: a run's record, written beside its TRX as <name>.tractline.json.

Two identities, for two questions:

  pipeline_id    "what algorithm was asked for": the config's sha256 (PipelineConfig.hash()) - its slots, their
                 implementations, versions and options. The device is left out on purpose
                 (docs/sota-pipeline-design.md, decision 6), so a calibration map keyed on it carries across devices.
  execution_id   "what computation produced these numbers": a sha256 over the pipeline_id and what else changes them -
                 the device type and model (the GPU's name, or the CPU's), the labeler's device, the backend each
                 step ran (Metal, Triton or torch), the CPU tracker's compiled step, the versions of tractline (and,
                 in a checkout, its git commit), torch, numpy, scipy, Python, Triton and (with the denoise slot)
                 dipy, and the sha256 of the labeler's weights. Runs on the same input with the same execution_id
                 are expected to give the same streamlines (the TRX's bytes also depend on its positions' type and
                 labeled_only, recorded with the output); ones with the same pipeline_id but another execution_id run the same algorithm on other
                 arithmetic (a CPU and an Apple GPU agree on 93.8 % of shared seeds' tracts, not bit for bit:
                 docs/sota-pipeline-design.md, decision 6).

The record also holds: each step's implementation status (validated, experimental, wrapper) and the tool it is checked
against; the runtime (versions, device, workers, backends); the inputs (path, size and sha256 of the series, its
.bval, .bvec and sidecar, and of the series that measured the field); the pairing decision; the correction's note and
warnings; the stage times; the output (path, size, sha256, streamline counts); and, in a checkout, the sha256 of
uv.lock (the environment `uv sync --locked` reproduces). Paths are absolute, as found on the machine that ran: they
trace the files, and also show its folder names to anyone the record is shared with.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tractline.provenance/1"


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


def device_info(device) -> dict:
    """The device type and model: a CUDA GPU's name and compute capability; on an Apple GPU or the CPU, the chip."""
    import torch
    d = torch.device(device)
    out = {"type": d.type}
    if d.type == "cuda":
        out["name"] = torch.cuda.get_device_name(d)
        out["compute capability"] = ".".join(map(str, torch.cuda.get_device_capability(d)))
    else:
        out["name"] = _cpu_model()                       # an Apple GPU is the chip's: M1, M2 and M4 differ
    out["os"] = f"{platform.system()} {platform.mac_ver()[0] or platform.release()}"
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


def weights(labeler) -> list:
    """The labeler's weights as pinned files with their sha256 (tractline.fetch's pins), or its class when unknown."""
    if labeler is None:
        return []
    from . import fetch
    mod = type(labeler).__module__
    if mod.endswith("labelers.rapidparc"):
        name = getattr(labeler, "model_name", "rapidparc")
        url, sha, _ = fetch.WEIGHTS[name]
        return [{"name": f"RapidParc {name}", "url": url, "sha256": sha}]
    if mod.endswith("labelers.tractcloud"):
        out = [{"name": f"TractCloud {k}", "commit": fetch.TRACTCLOUD_COMMIT, "sha256": sha}
               for k, (sha, _) in fetch.TRACTCLOUD_SOURCE.items()]
        for url, _, wanted in fetch.TRACTCLOUD_ARCHIVES.values():
            out += [{"name": f"TractCloud {m}", "url": url, "sha256": sha} for m, (sha, _) in wanted.items()]
        return out
    return [{"name": f"{mod}.{type(labeler).__qualname__}", "sha256": None}]


def _file(path) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    return {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def inputs(s) -> list:
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
            rec = _file(f)                               # the series, then its tables and sidecars, inherited ones too
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
    body = {"pipeline_id": pipeline_id, "device": {k: device.get(k) for k in ("type", "name", "compute capability")},
            "runtime": keep, "weights": sorted(w.get("sha256") or w["name"] for w in weights_)}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _lock() -> dict | None:
    """In a checkout: uv.lock's sha256 (the environment `uv sync --locked` reproduces)."""
    root = Path(__file__).resolve().parents[2]
    lock, proj = root / "uv.lock", root / "pyproject.toml"
    if lock.is_file() and proj.is_file() and 'name = "tractline"' in proj.read_text():
        return {"path": str(lock), "sha256": sha256_file(lock)}
    return None


def _digest(path) -> dict:
    """A written TRX: a zip's size and sha256; a directory's total size and a sha256 over its files' paths and hashes."""
    p = Path(path)
    if p.is_file():
        return _file(p)
    files = sorted(f for f in p.rglob("*") if f.is_file())
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(p).as_posix()}\0{sha256_file(f)}\n".encode())
    return {"path": str(p.resolve()), "bytes": sum(f.stat().st_size for f in files), "sha256": h.hexdigest(),
            "sha256 of": "the directory's files (path and sha256 of each, in order)"}


def record(run, s, device, labeler, timer=None, trx=None, trx_options=None) -> dict:
    """The run's record (the module docstring). run: pipeline.Run; s: its subject; trx: the written TRX, if any, and
    trx_options what it was written with (positions, labeled_only)."""
    config, runtime = run.config, dict(run.runtime)
    if runtime.get("device", "").startswith("cuda"):
        runtime["triton"] = _version("triton")
    git = _git()
    if git is not None:
        runtime["tractline commit"] = git["commit"] + (f"+modified:{git.get('changes sha256', '?')[:16]}"
                                                      if git["modified"] else "")
    runtime["scipy"] = _version("scipy")
    if config["slots"].get("denoise"):
        runtime["dipy"] = _version("dipy")
    if labeler is not None and getattr(labeler, "device", None) is not None:
        runtime["labeler device"] = str(labeler.device)
    dev = device_info(device)
    w = weights(labeler)
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
           "environment": {"platform": platform.platform(), "uv.lock": _lock(), "git": git,
                           "TRACTLINE_COMPILE": os.environ.get("TRACTLINE_COMPILE")},
           "inputs": inputs(s),
           "pairing": None if pairing is None else {"ok": pairing.ok, "how": pairing.how, "message": pairing.message,
                                                    "series": [str(p) for p in pairing.series]},
           "correction": {"applied": corr.applied, "note": corr.note, "warnings": list(corr.warnings or []),
                          "residual_left": corr.residual_left},
           "seconds": dict(timer.seconds) if timer is not None else None,
           "output": None}
    if trx is not None:
        opts = {"positions": "float32", "labeled_only": False, **(trx_options or {})}
        labels = run.labels
        labeled = None if labels is None else int(labels.keep.sum())
        rec["output"] = {**_digest(trx), **opts,
                         "streamlines": labeled if opts["labeled_only"] else len(run.tractogram.fibers),
                         "labeled": labeled}
    from .motion_eddy import _finite
    return _finite(json.loads(json.dumps(rec, default=str)))


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
