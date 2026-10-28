"""The tractline command (tractline.cli) and the weights fetch (tractline.fetch): the fetch downloads once into the
cache, verifies, refuses a wrong digest or an oversized body, and leaves nothing half-written; `tractline run` takes a
small BIDS series (the test phantom, with a reversed b0 pair beside it) to a TRX file, QC files and the corrected
series; usage errors are refused before any work."""
import io
import json
from pathlib import Path
import numpy as np
import pytest
from click.testing import CliRunner
from tractline import cli, fetch as FE


@pytest.fixture(autouse=True)
def _online(monkeypatch):
    """The fetch tests serve their own bytes: TRACTLINE_OFFLINE from the caller's environment does not apply."""
    monkeypatch.delenv("TRACTLINE_OFFLINE", raising=False)


class _Body(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(monkeypatch, payload, calls):
    def fake(url, timeout=60):
        calls.append(url)
        return _Body(payload)
    monkeypatch.setattr(FE, "urlopen", fake)


def test_fetch_once_verified(tmp_path, monkeypatch):
    import hashlib
    payload = b"weights" * 1000
    sha = hashlib.sha256(payload).hexdigest()
    calls = []
    _serve(monkeypatch, payload, calls)
    dest = tmp_path / "w.bin"
    assert FE.fetch("https://example.org/w.bin", sha, dest, len(payload)).read_bytes() == payload
    FE.fetch("https://example.org/w.bin", sha, dest, len(payload))
    assert len(calls) == 1                                                  # kept: not fetched again
    with pytest.raises(FE.FetchError, match="sha256"):
        FE.fetch("https://example.org/w.bin", "0" * 64, tmp_path / "x.bin", len(payload))
    with pytest.raises(FE.FetchError, match="more than"):
        FE.fetch("https://example.org/w.bin", sha, tmp_path / "y.bin", 10)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.bin"]          # nothing half-written left behind
    monkeypatch.setenv("TRACTLINE_OFFLINE", "1")
    with pytest.raises(FE.FetchError, match="OFFLINE"):
        FE.fetch("https://example.org/w.bin", sha, tmp_path / "z.bin", len(payload))


def test_weights_prefer_the_checkout(tmp_path, monkeypatch):
    """$TRACTOGRAPHY_DATA's copy when there; else the cache's, downloaded there."""
    import tractline.data as D
    monkeypatch.setattr(D, "DATA", tmp_path / "data")
    monkeypatch.setenv("TRACTLINE_CACHE", str(tmp_path / "cache"))
    payload = b"rapid" * 100
    import hashlib
    monkeypatch.setitem(FE.WEIGHTS, "rapidparc", ("https://example.org/r", hashlib.sha256(payload).hexdigest(), 500))
    calls = []
    _serve(monkeypatch, payload, calls)
    p = FE.weights("rapidparc")
    assert p == tmp_path / "cache" / "RapidParc" / "rapidparc.safetensors" and len(calls) == 1
    local = tmp_path / "data" / "RapidParc" / "rapidparc.safetensors"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"x")
    assert FE.weights("rapidparc") == local and len(calls) == 1


def test_usage_errors(tmp_path):
    f = tmp_path / "x.nii.gz"
    f.write_bytes(b"")
    r = CliRunner().invoke(cli.main, ["run", str(f), "--qc", str(tmp_path / "qc")])
    assert r.exit_code != 0 and "need --motion-eddy" in r.output
    r = CliRunner().invoke(cli.main, ["run", str(f), "--motion-eddy", "--write-corrected"])
    assert r.exit_code != 0 and "--qc" in r.output


def _bids(tmp_path):
    """sub-01/dwi: the phantom's series (AP: j-) and its reversed b0 pair (PA: j), as BIDS."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import nibabel as nib
    from phantom import subject
    s = subject()
    d = tmp_path / "sub-01" / "dwi"
    d.mkdir(parents=True)
    side = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05, "EchoTime": 0.08}
    ap = d / "sub-01_dir-AP_dwi"
    nib.save(nib.Nifti1Image(s.dwi, s.affine), str(ap) + ".nii.gz")
    np.savetxt(str(ap) + ".bval", s.bval[None], fmt="%g")
    np.savetxt(str(ap) + ".bvec", s.bvec, fmt="%.6f")
    (d / "sub-01_dir-AP_dwi.json").write_text(json.dumps(side))
    pa = d / "sub-01_dir-PA_dwi"
    nib.save(nib.Nifti1Image(s.b0s[..., 2:].astype(np.float32), s.affine), str(pa) + ".nii.gz")
    np.savetxt(str(pa) + ".bval", np.zeros((1, 2)), fmt="%g")
    np.savetxt(str(pa) + ".bvec", np.zeros((3, 2)), fmt="%g")
    (d / "sub-01_dir-PA_dwi.json").write_text(json.dumps({**side, "PhaseEncodingDirection": "j"}))
    return str(ap) + ".nii.gz", s


def test_run_end_to_end(tmp_path):
    from tractline.data import DATA
    if not (DATA / "RapidParc" / "rapidparc.safetensors").exists():
        pytest.skip("RapidParc's weights are not in $TRACTOGRAPHY_DATA (the test does not download)")
    dwi, s = _bids(tmp_path)
    out, qc = tmp_path / "new" / "out.trx", tmp_path / "qc"                # the TRX's folder does not exist yet
    r = CliRunner().invoke(cli.main, ["run", dwi, "--device", "cpu", "--workers", "1", "--motion-eddy", "rigid",
                                      "--trx", str(out), "--qc", str(qc), "--write-corrected", "--json", "-", "-q"])
    assert r.exit_code == 0, r.output
    summary = json.loads(r.stdout)
    assert summary["corrected"] and summary["streamlines"] > 0 and summary["shell"] == 2800.0
    assert len(summary["execution_id"]) == 64 and summary["pipeline_id"]
    assert any(x.startswith("motion_eddy") for x in summary["experimental"])
    side = json.loads(Path(summary["provenance"]).read_text())
    assert side["execution_id"] == summary["execution_id"] and side["output"]["path"] == out.name     # shareable
    assert [f["role"] for f in side["inputs"]].count("diffusion series") == 4
    assert out.exists() and (qc / "motion_eddy_qc.tsv").exists() and (qc / "motion_eddy_qc.json").exists()
    import nibabel as nib
    assert nib.load(str(qc / "corrected.nii.gz")).shape == s.dwi.shape
    assert np.loadtxt(qc / "corrected.bvec").shape == s.bvec.shape


def test_fetch_network_error_is_a_fetch_error(tmp_path, monkeypatch):
    import urllib.error

    def down(url, timeout=60):
        raise urllib.error.URLError("no route to host")
    monkeypatch.setattr(FE, "urlopen", down)
    with pytest.raises(FE.FetchError, match="could not download"):
        FE.fetch("https://example.org/w.bin", "0" * 64, tmp_path / "w.bin", 10)
    assert list(tmp_path.iterdir()) == []                                   # no temporary file left


def test_fetch_hashes_match_the_labeler():
    from tractline.labelers import rapidparc
    assert set(rapidparc.HASHES) <= set(FE.WEIGHTS)
    for name, sha in rapidparc.HASHES.items():
        assert FE.WEIGHTS[name][1] == sha


def test_auto_shell():
    """The median of the most populous shell, grouped to the nearest 100: a shell stored as 1050 stays 1050."""
    assert cli.auto_shell([0, 1050, 1050, 1045, 2000, 0]) == 1050
    assert cli.auto_shell([0, 995, 1000, 1005, 3000, 3000]) == 1000
    assert cli.auto_shell([5, 2800, 2800]) == 2800
    assert cli.auto_shell([0] + [1145, 1150, 1155] * 10 + [2000] * 20) == 1150     # jitter about a rounding boundary
    import click
    with pytest.raises(click.ClickException):
        cli.auto_shell([0, 5, 0])


def test_bad_input_is_a_message(tmp_path):
    """A file that is not a NIfTI: exit 1 and one line, not a traceback."""
    f = tmp_path / "x.nii.gz"
    f.write_bytes(b"not an image")
    r = CliRunner().invoke(cli.main, ["run", str(f), "--labeler", "rapidparc", "--device", "cpu"])
    assert r.exit_code == 1 and "Error:" in r.output and "Traceback" not in r.output, r.output
    assert isinstance(r.exception, SystemExit)                              # click's exit, not an uncaught error


def test_json_to_stdout_with_progress(tmp_path):
    """--json - without -q: the progress goes to standard error, standard output is the JSON alone."""
    from tractline.data import DATA
    if not (DATA / "RapidParc" / "rapidparc.safetensors").exists():
        pytest.skip("RapidParc's weights are not in $TRACTOGRAPHY_DATA (the test does not download)")
    dwi, s = _bids(tmp_path)
    r = CliRunner().invoke(cli.main, ["run", dwi, "--device", "cpu", "--workers", "1", "--json", "-"])
    assert r.exit_code == 0, r.output
    summary = json.loads(r.stdout)
    assert summary["streamlines"] > 0 and "streamlines" in r.stderr         # the progress went to standard error


def _targz(files):
    """A .tar.gz in memory: {name: bytes}, in order."""
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_members_streamed(tmp_path, monkeypatch):
    """Only the named members are written, each verified; the rest is read past; a wrong digest, a missing member or
    an oversized archive is refused with nothing half-written."""
    import hashlib
    sha = lambda b: hashlib.sha256(b).hexdigest()                          # noqa: E731
    a, b = b"model" * 100, b"args"
    blob = _targz({"./M/skip.bin": b"x" * 5000, "./M/a.bin": a, "M/b.txt": b, "./M/after.bin": b"y" * 10})
    calls = []
    _serve(monkeypatch, blob, calls)
    want = {"M/a.bin": (sha(a), "M/a.bin"), "M/b.txt": (sha(b), "M/b.txt")}
    out = FE.members("https://example.org/m.tgz", len(blob), want, tmp_path)
    assert out["M/a.bin"].read_bytes() == a and out["M/b.txt"].read_bytes() == b
    assert sorted(p.name for p in (tmp_path / "M").iterdir()) == ["a.bin", "b.txt"]
    FE.members("https://example.org/m.tgz", len(blob), want, tmp_path)
    assert len(calls) == 1                                                  # kept: not fetched again
    with pytest.raises(FE.FetchError, match="sha256"):
        FE.members("https://example.org/m.tgz", len(blob), {"M/a.bin": ("0" * 64, "N/a.bin")}, tmp_path)
    with pytest.raises(FE.FetchError, match="not in the archive"):
        FE.members("https://example.org/m.tgz", len(blob), {"M/zz": ("0" * 64, "N/zz")}, tmp_path)
    with pytest.raises(FE.FetchError, match="more than"):
        FE.members("https://example.org/m.tgz", 100, {"M/a.bin": (sha(a), "N/a.bin")}, tmp_path)
    with pytest.raises(FE.FetchError, match="ended early"):                # a body cut short, quietly by urllib
        FE.members("https://example.org/m.tgz", len(blob) + 1000, {"M/zz": ("0" * 64, "N/zz")}, tmp_path)
    assert not (tmp_path / "N").exists() or list((tmp_path / "N").iterdir()) == []


def test_tractcloud_checkout_else_cache(tmp_path, monkeypatch):
    """TractCloud's code and model: fetched into the cache when the data folder lacks them (the source files at the
    pinned commit, the members from the release archive), the checkout's when present; the centroid is always the copy
    shipped with tractline, never downloaded."""
    import hashlib
    import tractline.data as D
    sha = lambda b: hashlib.sha256(b).hexdigest()                          # noqa: E731
    monkeypatch.setattr(D, "DATA", tmp_path / "data")
    monkeypatch.setenv("TRACTLINE_CACHE", str(tmp_path / "cache"))
    src = {"__init__.py": b"", "inference.py": b"# inference"}
    pth, args = b"weights", b"args"
    tgz_model = _targz({"./TrainedModel/cli_args.txt": args, "./TrainedModel/best_tract_f1_model.pth": pth})
    monkeypatch.setattr(FE, "TRACTCLOUD_SOURCE", {k: (sha(v), len(v)) for k, v in src.items()})
    monkeypatch.setattr(FE, "TRACTCLOUD_ARCHIVES", {
        "model": ("https://example.org/model.tgz", len(tgz_model), {
            "TrainedModel/best_tract_f1_model.pth": (sha(pth), "TrainedModel/best_tract_f1_model.pth"),
            "TrainedModel/cli_args.txt": (sha(args), "TrainedModel/cli_args.txt")})})
    calls = []

    def fake(url, timeout=60):
        calls.append(url)
        if url.endswith("model.tgz"):
            return _Body(tgz_model)
        return _Body(src[url.rsplit("/", 1)[1]])
    monkeypatch.setattr(FE, "urlopen", fake)
    s, m, c = FE.tractcloud()
    root = tmp_path / "cache" / "TractCloud"
    assert (s / "tractcloud" / "inference.py").read_bytes() == b"# inference" and FE.TRACTCLOUD_COMMIT in calls[0]
    assert m == root / "TrainedModel" and (m / "best_tract_f1_model.pth").read_bytes() == pth
    assert c == FE.centroid() and FE.sha256_of(c) == FE.TRACTCLOUD_CENTROID[0] and len(calls) == 3   # no archive read
    FE.tractcloud()
    assert len(calls) == 3                                                  # all kept
    for p in ("TractCloud/src/tractcloud", "TrainedModel"):
        (tmp_path / "data" / p).mkdir(parents=True)
    for p in ("TrainedModel/best_tract_f1_model.pth", "TrainedModel/cli_args.txt"):
        (tmp_path / "data" / p).write_bytes(b"x")
    s, m, c = FE.tractcloud()
    assert s == tmp_path / "data" / "TractCloud" / "src" and m == tmp_path / "data" / "TrainedModel"
    assert c == FE.centroid() and len(calls) == 3


def test_shipped_centroid_is_tractclouds(monkeypatch, tmp_path):
    """The centroid shipped with tractline is the release's file, byte for byte (its pinned digest, and the checkout's
    copy when present), and a damaged copy is refused."""
    from tractline.data import DATA
    import numpy as np
    c = FE.centroid()
    assert np.load(c).shape == (15, 3)
    orig = DATA / "TrainData_800clu800ol" / "HCP_mass_center.npy"
    if orig.exists():
        assert FE.sha256_of(orig) == FE.sha256_of(c)
    monkeypatch.setattr(FE, "TRACTCLOUD_CENTROID", ("0" * 64, 308))
    with pytest.raises(FE.FetchError, match="damaged"):
        FE.centroid()


def test_tractcloud_pins_match_the_checkout():
    """The pinned digests are the checkout's files (TractCloud at the pinned commit, its v1.0.0 release)."""
    from tractline.data import DATA
    src = DATA / "TractCloud" / "src" / "tractcloud"
    if not (src / "inference.py").exists() or not (DATA / "TrainedModel" / "best_tract_f1_model.pth").exists():
        pytest.skip("no TractCloud checkout in $TRACTOGRAPHY_DATA")
    for name, (sha, size) in FE.TRACTCLOUD_SOURCE.items():
        assert FE.sha256_of(src / name) == sha and (src / name).stat().st_size == size, name
    for _, _, wanted in FE.TRACTCLOUD_ARCHIVES.values():
        for sha, rel in wanted.values():
            assert FE.sha256_of(DATA / rel) == sha, rel
    assert set(FE.TRACTCLOUD_ARCHIVES) == {"model"}                         # the centroid ships with tractline


def test_run_tractcloud(tmp_path):
    """`tractline run --labeler tractcloud` on the phantom (the checkout's TractCloud; the test does not download)."""
    from tractline.data import DATA
    if not all(p.exists() for p in (DATA / "TractCloud" / "src" / "tractcloud", DATA / "TrainedModel" / "cli_args.txt",
                                    DATA / "TrainedModel" / "best_tract_f1_model.pth")):
        pytest.skip("no complete TractCloud checkout in $TRACTOGRAPHY_DATA")
    dwi, s = _bids(tmp_path)
    r = CliRunner().invoke(cli.main, ["run", dwi, "--device", "cpu", "--workers", "1", "--labeler", "tractcloud",
                                      "--json", "-", "-q"])
    assert r.exit_code == 0, r.output
    summary = json.loads(r.stdout)
    assert summary["streamlines"] > 0 and 0 <= summary["labeled"] <= summary["streamlines"]


def test_run_gibbs(tmp_path):
    """`tractline run --gibbs --motion-eddy rigid`: Gibbs removal first, timed and counted in the summary."""
    from tractline.data import DATA
    if not (DATA / "RapidParc" / "rapidparc.safetensors").exists():
        pytest.skip("RapidParc's weights are not in $TRACTOGRAPHY_DATA (the test does not download)")
    dwi, s = _bids(tmp_path)
    r = CliRunner().invoke(cli.main, ["run", dwi, "--device", "cpu", "--workers", "1", "--gibbs", "--motion-eddy",
                                      "rigid", "--json", "-", "-q"])
    assert r.exit_code == 0, r.output
    summary = json.loads(r.stdout)
    assert summary["seconds"]["gibbs"] > 0 and summary["gibbs"]["b0s from the DWI"] >= 1
    assert summary["scan_to_labels_s"] >= summary["seconds"]["gibbs"] and summary["streamlines"] > 0


def test_warm_without_cuda(monkeypatch):
    """`tractline warm` where there is no CUDA GPU: says there is nothing to compile ahead, and succeeds."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = CliRunner().invoke(cli.main, ["warm"])
    assert r.exit_code == 0 and "nothing to compile" in r.output


def test_docs():
    """`tractline docs`: the commands' help, the README and the QC guide, as one Markdown text or one at a time."""
    r = CliRunner().invoke(cli.main, ["docs"])
    assert r.exit_code == 0, r.output
    for part in ("## tractline run --help", "# tractline\n", "Head motion and eddy current correction"):
        assert part in r.stdout, part
    readme = CliRunner().invoke(cli.main, ["docs", "readme"]).stdout
    assert readme.startswith("# tractline") and "--help\n\n```" not in readme
    assert "## 5. The QC files" in CliRunner().invoke(cli.main, ["docs", "guide"]).stdout
