"""Weights downloaded on first use, verified, and kept. Stdlib only.

tractline's default labeler needs RapidParc's trained weights (6.6 MB per model, rapidparc and hemiaug, from its
authors' GitHub release). A checkout keeps them in $TRACTOGRAPHY_DATA (tractline.data); an installed tractline
(`uvx`, `pip`) has no such folder, so the first run with a model downloads its file into a user cache and every later
run reads it from there. Nothing is
redistributed by tractline: the URLs are the authors' release assets, and each file is checked against the sha256
tractline pins (the same values as labelers.rapidparc.HASHES; a test holds them equal) before it is kept.

The rules follow haversack's fetchlib (one door for every request; read for method): the request names itself (a
"tractline/<version>" User-Agent - some hosts refuse Python's default), no credentials are ever sent, a download is
capped at a size it declares, and a file appears in the cache only whole and verified (written to a temporary name
beside it, hashed while streaming, renamed into place).

    path = weights("rapidparc")         # $TRACTOGRAPHY_DATA/RapidParc/rapidparc.safetensors if there, else the cache
    src, model, centroid = tractcloud() # TractCloud's code and trained model the same way; the centroid: the packaged copy

TractCloud (Xue et al. 2023, the optional labeler) comes the same way, from its authors' GitHub: the four modules of its
package that labelers.tractcloud imports, at a pinned commit, each checked against its sha256; its trained model from
its v1.0.0 release (TrainedModel.tar.gz, 8.4 MB). The atlas centroid its inputs are centered on (HCP_mass_center.npy,
308 bytes: 15 points of a mean streamline, from inside the release's 166 MB TrainData_800clu800ol.tar.gz) ships with
tractline instead (labelers/tractcloud_data/, byte for byte, its sha256 checked on use), under 3D Slicer's license,
whose Part B THIRD_PARTY_NOTICES.md carries. Its code is under the same license, as published there.

Where the cache is: $TRACTLINE_CACHE if set; else ~/Library/Caches/tractline (macOS), %LOCALAPPDATA%\\tractline
(Windows), $XDG_CACHE_HOME/tractline or ~/.cache/tractline (elsewhere). TRACTLINE_OFFLINE=1 refuses to download.
"""
from __future__ import annotations

import gzip
import hashlib
import http.client
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zlib
from pathlib import Path

#: name -> (url, sha256, size in bytes): the files tractline may fetch, each its authors' release asset
WEIGHTS = {
    "rapidparc": ("https://github.com/MedVisBonn/RapidParc/releases/download/v1.0.0/rapidparc.safetensors",
                  "6a158dad6a0b6124946da102b6037a6f7f8a16fb24e7caec15eaf95213e7b793", 6574128),
    "hemiaug": ("https://github.com/MedVisBonn/RapidParc/releases/download/v1.0.0/hemiaug.safetensors",
                "b66dfce6135aac56c3f835264e5ae9630b2497859425ad98cdc6d842c6f94435", 6574128),
    # SynthStrip's released model (Hoopes et al. 2022), offered by its authors under MIT or CC BY 4.0; taken under MIT
    "synthstrip": ("https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
                   "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33", 30851709),
}
#: name -> (folder, file name): where each weights file sits under $TRACTOGRAPHY_DATA (a checkout) or the cache
WEIGHT_FILES = {"rapidparc": ("RapidParc", "rapidparc.safetensors"), "hemiaug": ("RapidParc", "hemiaug.safetensors"),
                "synthstrip": ("SynthStrip", "synthstrip.1.pt")}

#: TractCloud: the commit whose package modules labelers.tractcloud runs (name -> (sha256, size)), and the release
#: archives with the members it needs (archive -> (url, size, {member: (sha256, path under the cache's TractCloud/)}))
TRACTCLOUD_COMMIT = "94de6275d07d7b5eeb50e4cd4d6e85fde29d5fb0"
TRACTCLOUD_SOURCE = {
    "__init__.py": ("393c956eae7783a1d8e4eb66b3c06de18fe61917114dc164a1207f13d444a085", 86),
    "inference.py": ("524b45cb54c3967c03722dcdefa6d5f1e1d64582e461f3a552ca066a27131e05", 9560),
    "models.py": ("05f69f3f37b2cf50155e293da338ea03c3e6dc0154175a56cbda96829f2f60d3", 9212),
    "tract_mapping.py": ("5479054a5cc29dcd3e68ae99372e4298832cc4e6c295e12d4ffd7147d0bc7b43", 8947),
}
_TC = "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0"
TRACTCLOUD_ARCHIVES = {
    "model": (f"{_TC}/TrainedModel.tar.gz", 8409790, {
        "TrainedModel/best_tract_f1_model.pth": ("93b01cf532ff4e1739754866bb9d7b71c5f5dffc9e8e6d4cf95ac8c6a0431d6a",
                                                 "TrainedModel/best_tract_f1_model.pth"),
        "TrainedModel/cli_args.txt": ("fd521ca95673fb440a84c14078e287f42214975be60b7ff523e6b0df90480fda",
                                      "TrainedModel/cli_args.txt")}),
}

#: TractCloud's atlas centroid, shipped with tractline (labelers/tractcloud_data/HCP_mass_center.npy; its release's
#: TrainData_800clu800ol.tar.gz holds the original): (sha256, size)
TRACTCLOUD_CENTROID = ("62cbbae3dcf1c8d3792af3453c338bbce6fceed4705810a10e23b75b8d4b0fa9", 308)


class FetchError(RuntimeError):
    """A file could not be fetched or did not verify."""


def cache_dir() -> Path:
    if os.environ.get("TRACTLINE_CACHE"):
        return Path(os.environ["TRACTLINE_CACHE"])
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "tractline"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "tractline"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tractline"     # an empty value: unset


def user_agent() -> str:
    from . import __version__
    return f"tractline/{__version__} (+https://github.com/mhalle/tractline)"


def urlopen(url: str, timeout: float = 60):
    """The one place a request is made (tests replace it)."""
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": user_agent()}), timeout=timeout)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Capped:
    """A response read through: refused past `size` bytes; progress(done, size) as bytes arrive."""
    def __init__(self, r, url: str, size: int, progress=None):
        self.r, self.url, self.size, self.progress, self.done = r, url, size, progress, 0

    def read(self, n=-1):
        b = self.r.read(n)
        self.done += len(b)
        if self.done > self.size:
            raise FetchError(f"{self.url}: more than the expected {self.size} bytes")
        if self.progress and b:
            self.progress(self.done, self.size)
        return b


_NETWORK = (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError)


def _offline(what: str, url: str):
    if os.environ.get("TRACTLINE_OFFLINE") == "1":
        raise FetchError(f"{what} (or does not verify) and TRACTLINE_OFFLINE=1: not downloading {url}")


def fetch(url: str, sha256: str, dest: Path, size: int, progress=None) -> Path:
    """dest, downloaded from url if it is not already there and whole: streamed to a temporary file beside it, hashed
    on the way, refused past `size` bytes or on a wrong digest, then renamed into place. progress(done, size) is
    called as bytes arrive."""
    dest = Path(dest)
    if dest.exists() and sha256_of(dest) == sha256:
        return dest
    _offline(f"{dest.name} is not in {dest.parent}", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    h = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out, urlopen(url) as r:
            body = _Capped(r, url, size, progress)
            for chunk in iter(lambda: body.read(1 << 20), b""):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != sha256:
            raise FetchError(f"{url}: sha256 {h.hexdigest()}, expected {sha256}")
        os.replace(tmp, dest)
    except _NETWORK as e:
        Path(tmp).unlink(missing_ok=True)
        raise FetchError(f"could not download {url}: {e}") from e
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest


def members(url: str, size: int, wanted: dict, root: Path, progress=None) -> dict:
    """Named files from a .tar.gz release asset, streamed: each regular member whose name (less a leading "./") is a
    key of wanted -> (sha256, path under root) is hashed as it is written, refused on a wrong digest, and renamed into
    place whole; the download stops once all are in, and other members are read past, never written. Files already
    in place and verified are not fetched again. Returns {member: path}."""
    root = Path(root)
    out = {k: root / rel for k, (_, rel) in wanted.items()}
    todo = {k: v for k, v in wanted.items() if not (out[k].exists() and sha256_of(out[k]) == v[0])}
    if not todo:
        return out
    _offline(f"{', '.join(sorted(todo))} not in {root}", url)
    tmps, body = [], None
    try:
        with urlopen(url) as r, tarfile.open(fileobj=(body := _Capped(r, url, size, progress)), mode="r|gz") as tar:
            for m in tar:
                name = m.name[2:] if m.name.startswith("./") else m.name
                if name not in todo or not m.isfile():
                    continue
                sha, _ = todo.pop(name)
                dest = out[name]
                dest.parent.mkdir(parents=True, exist_ok=True)
                h, src = hashlib.sha256(), tar.extractfile(m)
                fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
                tmps.append(tmp)
                with os.fdopen(fd, "wb") as f:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        h.update(chunk)
                        f.write(chunk)
                if h.hexdigest() != sha:
                    raise FetchError(f"{url}: {name}: sha256 {h.hexdigest()}, expected {sha}")
                os.replace(tmp, dest)
                tmps.remove(tmp)
                if not todo:
                    break
        if todo:                                  # urllib ends a cut-off body quietly: say which it was
            why = "not in the archive" if body.done >= size else f"the download ended early ({body.done} of {size} bytes)"
            raise FetchError(f"{url}: {', '.join(sorted(todo))}: {why}")
    except (*_NETWORK, tarfile.TarError, gzip.BadGzipFile, zlib.error, EOFError) as e:
        raise FetchError(f"could not download {url}: {e}") from e
    finally:
        for t in tmps:
            Path(t).unlink(missing_ok=True)
    return out


def tractcloud(progress_for=None) -> tuple:
    """(source, model, centroid) for labelers.tractcloud: the folder to put on sys.path (holding the package
    `tractcloud`), the trained model's folder (best_tract_f1_model.pth, cli_args.txt) and the atlas centroid
    (HCP_mass_center.npy). The source and the model are the checkout's ($TRACTOGRAPHY_DATA/TractCloud/src,
    TrainedModel) when present, else the cache's, fetched there on first use (about 8.4 MB); the centroid is the copy
    shipped with tractline (centroid()). progress_for(what) -> a progress(done, size) callback, for the release
    archive."""
    from .data import DATA
    root = cache_dir() / "TractCloud"
    src = DATA / "TractCloud" / "src"
    if not (src / "tractcloud").is_dir():
        src = root / TRACTCLOUD_COMMIT[:12] / "src"
        base = f"https://raw.githubusercontent.com/SlicerDMRI/TractCloud/{TRACTCLOUD_COMMIT}/src/tractcloud"
        for name, (sha, size) in TRACTCLOUD_SOURCE.items():
            fetch(f"{base}/{name}", sha, src / "tractcloud" / name, size)
    model = DATA / "TrainedModel"
    if not ((model / "best_tract_f1_model.pth").exists() and (model / "cli_args.txt").exists()):
        url, size, wanted = TRACTCLOUD_ARCHIVES["model"]
        members(url, size, wanted, root, progress_for and progress_for("TractCloud's model"))
        model = root / "TrainedModel"
    return src, model, centroid()


def centroid() -> Path:
    """TractCloud's atlas centroid as shipped with tractline (labelers/tractcloud_data/HCP_mass_center.npy), checked
    against its pinned sha256 (TRACTCLOUD_CENTROID)."""
    from importlib.resources import files
    p = Path(str(files("tractline.labelers") / "tractcloud_data" / "HCP_mass_center.npy"))
    sha, size = TRACTCLOUD_CENTROID
    if not p.is_file() or p.stat().st_size != size or sha256_of(p) != sha:
        raise FetchError(f"{p}: not TractCloud's centroid (sha256 {sha[:12]}..., {size} bytes) - the installation is "
                         "damaged; reinstall tractline")
    return p


def weights(name: str, progress=None) -> Path:
    """The path of a weights file: the checkout's ($TRACTOGRAPHY_DATA/<folder>/<file>, WEIGHT_FILES) when present,
    else the cache's, downloaded there on first use."""
    from .data import DATA
    if name not in WEIGHTS:
        raise KeyError(f"no weights named {name!r}: {sorted(WEIGHTS)}")
    folder, file = WEIGHT_FILES[name]
    local = DATA / folder / file
    if local.exists():
        return local
    url, sha, size = WEIGHTS[name]
    return fetch(url, sha, cache_dir() / folder / file, size, progress)
