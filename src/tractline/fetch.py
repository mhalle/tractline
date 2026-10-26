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

Where the cache is: $TRACTLINE_CACHE if set; else ~/Library/Caches/tractline (macOS), %LOCALAPPDATA%\\tractline
(Windows), $XDG_CACHE_HOME/tractline or ~/.cache/tractline (elsewhere). TRACTLINE_OFFLINE=1 refuses to download.
"""
from __future__ import annotations

import hashlib
import http.client
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

#: name -> (url, sha256, size in bytes): the files tractline may fetch, each its authors' release asset
WEIGHTS = {
    "rapidparc": ("https://github.com/MedVisBonn/RapidParc/releases/download/v1.0.0/rapidparc.safetensors",
                  "6a158dad6a0b6124946da102b6037a6f7f8a16fb24e7caec15eaf95213e7b793", 6574128),
    "hemiaug": ("https://github.com/MedVisBonn/RapidParc/releases/download/v1.0.0/hemiaug.safetensors",
                "b66dfce6135aac56c3f835264e5ae9630b2497859425ad98cdc6d842c6f94435", 6574128),
}


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


def fetch(url: str, sha256: str, dest: Path, size: int, progress=None) -> Path:
    """dest, downloaded from url if it is not already there and whole: streamed to a temporary file beside it, hashed
    on the way, refused past `size` bytes or on a wrong digest, then renamed into place. progress(done, size) is
    called as bytes arrive."""
    dest = Path(dest)
    if dest.exists() and sha256_of(dest) == sha256:
        return dest
    if os.environ.get("TRACTLINE_OFFLINE") == "1":
        raise FetchError(f"{dest.name} is not in {dest.parent} (or does not verify) and TRACTLINE_OFFLINE=1: not "
                         f"downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    h, done = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, "wb") as out, urlopen(url) as r:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                done += len(chunk)
                if done > size:
                    raise FetchError(f"{url}: more than the expected {size} bytes")
                h.update(chunk)
                out.write(chunk)
                if progress:
                    progress(done, size)
        if h.hexdigest() != sha256:
            raise FetchError(f"{url}: sha256 {h.hexdigest()}, expected {sha256}")
        os.replace(tmp, dest)
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError) as e:
        Path(tmp).unlink(missing_ok=True)
        raise FetchError(f"could not download {url}: {e}") from e
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest


def weights(name: str, progress=None) -> Path:
    """The path of a weights file: the checkout's ($TRACTOGRAPHY_DATA/RapidParc/<name>.safetensors) when present, else
    the cache's, downloaded there on first use."""
    from .data import DATA
    if name not in WEIGHTS:
        raise KeyError(f"no weights named {name!r}: {sorted(WEIGHTS)}")
    local = DATA / "RapidParc" / f"{name}.safetensors"
    if local.exists():
        return local
    url, sha, size = WEIGHTS[name]
    return fetch(url, sha, cache_dir() / "RapidParc" / f"{name}.safetensors", size, progress)
