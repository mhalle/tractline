"""The run record (tractline.provenance): written beside every TRX, with the pipeline_id (the config's hash) and an
execution_id that changes with what changes the numbers (device, backends, versions, weights) but not with the run's
time; the steps' statuses; the inputs' and the output's sha256. Also: run() returns a Run that still unpacks and
indexes as the old tuple, chooses the device and the shell as the command does, and a TRX is written whole."""
import json
from pathlib import Path

import pytest
from phantom import subject
from tractline import pipeline as P, provenance as PV
from tractline.data import DATA

WEIGHTS = (DATA / "RapidParc" / "rapidparc.safetensors").exists()
need_weights = pytest.mark.skipif(not WEIGHTS, reason="RapidParc's weights are not in $TRACTOGRAPHY_DATA")


@pytest.fixture(scope="module")
def phantom_runs(tmp_path_factory):
    """Two runs of the phantom on the CPU with the default labeler, each writing a TRX into a new folder."""
    if not WEIGHTS:
        pytest.skip("RapidParc's weights are not in $TRACTOGRAPHY_DATA")
    d = tmp_path_factory.mktemp("prov")
    s = subject()
    return s, [(P.run(s, None, P.Timer(), trx=d / f"new{k}" / "out.trx", device="cpu", workers=1), d / f"new{k}" / "out.trx")
               for k in range(2)]


def test_sidecar_written(phantom_runs):
    s, [(r, trx), _] = phantom_runs
    side = PV.sidecar_path(trx)
    assert side == trx.parent / "out.tractline.json" and side.exists()
    rec = json.loads(side.read_text())
    assert rec == r.provenance and rec["schema"] == PV.SCHEMA
    assert rec["pipeline_id"] == r.config["sha256"] and len(rec["execution_id"]) == 64
    assert rec["output"]["sha256"] == PV.sha256_file(trx) and rec["output"]["streamlines"] == len(r.tractogram.fibers)
    assert rec["steps"]["track"] == {"impl": "ukf", "version": rec["steps"]["track"]["version"], "status": "validated",
                                     "oracle": "Slicer UKFTractography"}
    assert rec["runtime"]["ukf backend"] == "torch (fast step)" and rec["device"]["type"] == "cpu"
    assert rec["weights"][0]["sha256"] == "6a158dad6a0b6124946da102b6037a6f7f8a16fb24e7caec15eaf95213e7b793"
    assert rec["inputs"] == [] and rec["experimental"] == []          # a subject from arrays; the default pipeline
    assert not [p for p in trx.parent.iterdir() if p.suffix == ".part" or p.name.startswith("tmp")]


def test_ids_stable_and_specific(phantom_runs):
    """The same computation gives the same execution_id at another time; another device or backend another one."""
    _, [(a, _), (b, _)] = phantom_runs
    assert a.provenance["execution_id"] == b.provenance["execution_id"]
    assert a.provenance["created"] <= b.provenance["created"]
    pid, rt, dev, w = a.provenance["pipeline_id"], a.provenance["runtime"], a.provenance["device"], a.provenance["weights"]
    base = PV.execution_id(pid, rt, dev, w)
    assert base == a.provenance["execution_id"]
    assert PV.execution_id(pid, rt, {**dev, "type": "mps"}, w) != base
    assert PV.execution_id(pid, {**rt, "torch": "0.0"}, dev, w) != base
    assert PV.execution_id(pid, rt, dev, [{"name": "x", "sha256": "0" * 64}]) != base
    assert PV.execution_id("other", rt, dev, w) != base
    assert PV.execution_id(pid, {**rt, "workers": 99}, dev, w) == base           # the CPU's workers: the same fibers


def test_run_is_a_run_and_a_tuple(phantom_runs):
    _, [(r, _), _] = phantom_runs
    corr, tg, labels = r
    assert isinstance(r, P.Run) and len(r) == 3 and r[1] is tg and r[0] is corr and r[2] is labels


def test_inputs_hashed(tmp_path):
    """A BIDS subject's files: the series, its .bval, .bvec and sidecar, and the series that measured its field."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cli import _bids
    from tractline import bids
    dwi, _ = _bids(tmp_path)
    s = bids.load(dwi)
    files = PV.inputs(s)
    roles = [f["role"] for f in files]
    assert roles.count("diffusion series") == 4 and roles.count("field series") >= 2
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in files)
    assert files[0]["sha256"] == PV.sha256_file(dwi)


def test_sidecar_path():
    assert PV.sidecar_path("out/sub-01.trx") == Path("out/sub-01.tractline.json")
    assert PV.sidecar_path("out/sub-01") == Path("out/sub-01.tractline.json")


def test_experimental_listed():
    cfg = P.config_for(motion_eddy=True, gibbs=True).to_dict()
    steps = PV.experimental(cfg)
    assert any(x.startswith("motion_eddy: tractline") and "FSL eddy" in x for x in steps)
    assert any(x.startswith("gibbs: subvoxel-shift") for x in steps) and len(steps) == 2


def test_defaults_resolved():
    assert P.resolve_device("auto") in ("mps", "cuda", "cpu") and P.resolve_device("cpu") == "cpu"
    assert P.default_shell([0, 1000, 1000, 2000, 2000, 2000]) == 2000.0
    with pytest.raises(ValueError, match="no diffusion-weighted"):
        P.default_shell([0, 0])


def test_run_with_config_refuses_keywords():
    with pytest.raises(ValueError, match="in the config"):
        P.run(subject(), None, P.Timer(), config=P.config_for(), shell=1000.0)


@need_weights
def test_run_default_shell_is_the_scans(tmp_path):
    """No shell given: the shell with the most volumes, recorded in the config."""
    s = subject()
    r = P.run(s, None, P.Timer(), device="cpu", workers=1)
    assert r.config["slots"]["model"]["options"]["shell"] == P.default_shell(s.bval)
    assert r.provenance["output"] is None and r.provenance["seconds"]


def test_malformed_sidecar_is_an_error(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cli import _bids
    from tractline import bids
    dwi, _ = _bids(tmp_path)
    side = Path(str(dwi).replace(".nii.gz", ".json"))
    side.write_text('{"PhaseEncodingDirection": "j-", ')
    with pytest.raises(ValueError, match="not valid JSON") as e:
        bids.load(dwi)
    assert side.name in str(e.value)
    side.write_text("[1, 2]")
    with pytest.raises(ValueError, match="not a JSON object"):
        bids.load(dwi)


def test_more_identity(phantom_runs):
    """The labeler's device, the chip, and the commit in a checkout are recorded; the labeler's device is in the id."""
    _, [(a, trx), _] = phantom_runs
    rec = a.provenance
    assert rec["runtime"]["labeler device"] == "cpu" and rec["device"]["name"] and rec["device"]["os"]
    assert PV.execution_id(rec["pipeline_id"], {**rec["runtime"], "labeler device": "mps"}, rec["device"],
                           rec["weights"]) != rec["execution_id"]
    if rec["environment"]["git"] is not None:                     # a checkout
        assert rec["runtime"]["tractline commit"].startswith(rec["environment"]["git"]["commit"])
    assert rec["output"]["positions"] == "float32" and rec["output"]["labeled_only"] is False
    import os
    import stat
    mode = stat.S_IMODE(os.stat(PV.sidecar_path(trx)).st_mode)
    assert mode & 0o044 == 0o044 or os.umask(0o022) & 0o044            # readable by others, as an ordinary file


@need_weights
def test_labeled_only_and_directory_output(tmp_path):
    """labeled_only: the record counts what the TRX holds; a TRX directory gets a digest over its files."""
    s = subject()
    r = P.run(s, None, P.Timer(), trx=tmp_path / "out" / "lab.trx", device="cpu", workers=1, labeled_only=True)
    out = r.provenance["output"]
    assert out["labeled_only"] is True and out["streamlines"] == out["labeled"] == int(r.labels.keep.sum())
    r = P.run(s, None, P.Timer(), trx=tmp_path / "out" / "asdir", device="cpu", workers=1)
    out = r.provenance["output"]
    assert len(out["sha256"]) == 64 and out["bytes"] > 0 and (tmp_path / "out" / "asdir.tractline.json").exists()


def test_bad_sidecars_elsewhere(tmp_path):
    """A malformed sidecar of another series (a T1w) does not stop the load; a partner with one is dropped with the
    reason; a UTF-8 byte-order mark is accepted."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cli import _bids
    from tractline import bids
    dwi, _ = _bids(tmp_path)
    anat = tmp_path / "sub-01" / "anat"
    anat.mkdir()
    (anat / "sub-01_T1w.json").write_text("{not json")
    assert bids.load(dwi).pairing.ok
    pa = Path(str(dwi).replace("dir-AP", "dir-PA").replace(".nii.gz", ".json"))
    good = pa.read_text()
    pa.write_text("{broken")
    s = bids.load(dwi)
    assert not s.pairing.ok and "not valid JSON" in s.pairing.message
    pa.write_bytes(b"\xef\xbb\xbf" + good.encode())
    assert bids.load(dwi).pairing.ok


@need_weights
def test_record_failure_never_costs_the_run(tmp_path, monkeypatch):
    """A failure while building the record: the run still returns, with a warning, and the sidecar says what failed."""
    def broken(*a, **k):
        raise RuntimeError("simulated")
    monkeypatch.setattr(PV, "record", broken)
    with pytest.warns(UserWarning, match="incomplete"):
        r = P.run(subject(), None, P.Timer(), trx=tmp_path / "o.trx", device="cpu", workers=1)
    side = json.loads(PV.sidecar_path(tmp_path / "o.trx").read_text())
    assert "simulated" in side["error"] and side["pipeline_id"] == r.config["sha256"] and len(r.tractogram.fibers)
