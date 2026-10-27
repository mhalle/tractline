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
    assert rec["weights"][0]["pinned"] and rec["paths"] == "shareable" and "/" not in rec["output"]["path"]
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
    assert PV.execution_id(pid, rt, {**dev, "os build": "other"}, w) != base               # an OS update
    assert PV.execution_id(pid, rt, {**dev, "driver": "999.1"}, w) != base
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


def test_paths_shareable_unless_asked(tmp_path):
    """Inside a BIDS dataset: the dataset folder's name and the path in it; outside: the file's name; absolute=True:
    the full path."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cli import _bids
    from tractline import bids
    dwi, _ = _bids(tmp_path / "study")
    s = bids.load(dwi)
    assert [f["path"] for f in PV.inputs(s)][0] == Path(dwi).name and "dataset" not in PV.inputs(s)[0]
    (tmp_path / "study" / "dataset_description.json").write_text('{"Name": "x", "BIDSVersion": "1.9.0"}')
    s = bids.load(dwi)
    first = PV.inputs(s)[0]
    assert first["dataset"] == "study" and first["path"] == "sub-01/dwi/" + Path(dwi).name
    assert PV.inputs(s, absolute=True)[0]["path"] == str(Path(dwi).resolve())


def test_tractcloud_files_as_loaded(tmp_path, monkeypatch):
    """TractCloud's record hashes the files it loaded: a local copy that differs from the pins is flagged, and moves the
    execution_id."""
    import sys
    import types
    from tractline import fetch as FE
    src = tmp_path / "src" / "tractcloud"
    src.mkdir(parents=True)
    for k in FE.TRACTCLOUD_SOURCE:
        (src / k).write_text("# " + k)
    model = tmp_path / "TrainedModel"
    model.mkdir()
    (model / "best_tract_f1_model.pth").write_bytes(b"weights")
    (model / "cli_args.txt").write_text("args")
    centroid = tmp_path / "HCP_mass_center.npy"
    centroid.write_bytes(b"centroid")
    mod = types.ModuleType("tractline.labelers.tractcloud")              # the module's paths, as the real one has
    mod._SOURCE, mod.MODEL, mod.MASS_CENTER = tmp_path / "src", model, centroid
    monkeypatch.setitem(sys.modules, "tractline.labelers.tractcloud", mod)
    Labeler = type("Labeler", (), {"__module__": "tractline.labelers.tractcloud"})
    w = PV.weights(Labeler())
    assert len(w) == 7 and not any(x["pinned"] for x in w)
    assert w[-1]["sha256"] == PV.sha256_file(centroid) and w[-1]["path"] == "HCP_mass_center.npy"
    dev = {"type": "cpu", "name": "x"}
    a = PV.execution_id("p", {}, dev, w)
    centroid.write_bytes(b"changed")
    assert PV.execution_id("p", {}, dev, PV.weights(Labeler())) != a


def test_shareable_record_has_no_folders(tmp_path, monkeypatch):
    """A path quoted in a message (another series' bad sidecar) does not carry the folders above the dataset into a
    shareable record; absolute_paths keeps them. A symlinked input (DataLad, git-annex) keeps its dataset path."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_cli import _bids
    from tractline import bids
    from types import SimpleNamespace
    study = tmp_path / "private-project-123" / "study"
    dwi, _ = _bids(study)
    (study / "dataset_description.json").write_text('{"Name": "x", "BIDSVersion": "1.9.0"}')
    fmap = study / "sub-01" / "fmap"
    fmap.mkdir()
    (fmap / "sub-01_dir-PA_epi.json").write_text("{bad")
    (fmap / "sub-01_dir-PA_epi.nii.gz").write_bytes(Path(dwi).read_bytes())
    s = bids.load(dwi)
    run = SimpleNamespace(config={"sha256": "p", "slots": {}}, runtime={}, correction=SimpleNamespace(
        applied=True, note=s.pairing.message, warnings=[], residual_left=None), labels=None, tractogram=None)
    monkeypatch.setattr(PV, "statuses", lambda c: {})
    rec = PV.record(run, s, "cpu", None)
    text = json.dumps(rec)
    assert "private-project-123" not in text and str(tmp_path) not in text
    assert "private-project-123" in json.dumps(PV.record(run, s, "cpu", None, absolute_paths=True))
    link = tmp_path / "annex" / "object.nii.gz"
    link.parent.mkdir()
    Path(dwi).rename(link)
    Path(dwi).symlink_to(link)
    assert PV._where(dwi)["path"] == "sub-01/dwi/" + Path(dwi).name


def test_lookalike_labeler_is_not_ours():
    """A caller's labeler in a module whose name merely ends like tractline's: recorded by its class, no download."""
    Labeler = type("Labeler", (), {"__module__": "mylab.labelers.rapidparc"})
    assert PV.weights(Labeler()) == [{"name": "mylab.labelers.rapidparc.Labeler", "sha256": None}]


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA")
def test_cuda_device_record():
    """On CUDA: the runtime and the driver (from the driver library itself) are recorded, the same on every call, and
    "cuda" and "cuda:0" describe one device."""
    import re
    a, b = PV.device_info("cuda"), PV.device_info("cuda:0")
    assert a == b and re.fullmatch(r"\d+\.\d+", a["driver"] or "") and a["cuda runtime"]
