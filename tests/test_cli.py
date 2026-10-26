"""The tractline command (tractline.cli) and the weights fetch (tractline.fetch): the fetch downloads once into the
cache, verifies, refuses a wrong digest or an oversized body, and leaves nothing half-written; `tractline run` takes a
small BIDS series (the test phantom, with a reversed b0 pair beside it) to a TRX file, QC files and the corrected
series; usage errors are refused before any work."""
import io
import json
import numpy as np
import pytest
from click.testing import CliRunner
from tractline import cli, fetch as FE


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
    for name, (_, sha, _) in FE.WEIGHTS.items():
        assert rapidparc.HASHES[name] == sha


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
