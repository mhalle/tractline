"""The pipeline's configuration (tractline.config): checked at construction, hashed stably, round-tripped, and run()'s
keywords translated into it; the mask slot's mask is prepare's own."""
import inspect, subprocess, sys
import numpy as np
import pytest
from phantom import subject
from tractline import config as C, pipeline as P, ukf as U
from tractline.config import PipelineConfig
from tractline.prep import prepare

# ukf-rapidparc@1 is released: its contents, and so its hash, never change. A new number (@2) is the way to change it.
DEFAULT_SHA256 = "28d01a1d3947a3f2f2de450b9b472c61e228129a156fad14be37b252f5574678"


def test_default_preset_is_pinned():
    cfg = PipelineConfig.preset("default")
    assert cfg.origin == "ukf-rapidparc@1" and cfg == PipelineConfig.preset("ukf-rapidparc@1")
    assert cfg.hash() == DEFAULT_SHA256
    assert P.config_for() == cfg                                             # run()'s defaults are the preset


def test_ukf_defaults_are_the_trackers():
    sig = inspect.signature(U.track).parameters
    opts = C.implementation("track", "ukf").options
    assert set(opts) == P.TRACKER_SETTINGS
    assert {k: o.default for k, o in opts.items()} == {k: sig[k].default for k in opts}


def test_checked_at_construction():
    cfg = PipelineConfig.preset("default")
    with pytest.raises(ValueError, match="not among the slots"):
        cfg.with_(bias="n4")
    with pytest.raises(ValueError, match="is not one of"):
        cfg.with_(track="ifod2")
    with pytest.raises(ValueError, match="not among its options"):
        cfg.with_(track={"Rss": 0.1})
    with pytest.raises(ValueError, match="not a finite number"):
        cfg.with_(track={"Rs": "0.1"})
    with pytest.raises(ValueError, match="below"):
        cfg.with_(track={"Rs": -1.0})
    with pytest.raises(ValueError, match="is not one of"):
        cfg.with_(denoise={"impl": "mppca", "order": "later"})
    with pytest.raises(ValueError, match="requires the mask slot"):
        cfg.with_(mask=None)
    with pytest.raises(ValueError, match="slot is off"):
        cfg.with_(denoise={"patch_radius": 3})                               # options for a slot that is off
    with pytest.raises(ValueError, match="not among the tracker's settings"):
        P.config_for(tracker={"Rss": 0.1})


def test_with_and_hash():
    cfg = PipelineConfig.preset("default")
    assert cfg.with_(model={"shell": 2800}) is cfg                           # no change: the same config, the same origin
    a, b = cfg.with_(model={"shell": 2000}), cfg.with_(model={"shell": 2000.0})
    assert a == b and a.hash() == b.hash() != cfg.hash()                     # an int for a float option: the same hash
    assert a.origin == "ukf-rapidparc@1 + changes"
    assert cfg.with_(track={"Rs": 0.02}).hash() == cfg.hash()                # the default, stated: the same hash
    assert cfg.with_(track={"Rs": 0.1}).opts("track")["Qm"] == 0.001         # options merge into the implementation's
    off = cfg.with_(field=None)
    assert not off.is_on("field") and "field" not in off.canonical()["slots"] and off.hash() != cfg.hash()
    assert cfg.with_(bundles={"impl": "caller-object", "class": "x"}).hash() is None


def test_round_trip():
    cfg = P.config_for(tracker={"Rs": 0.1}, denoise="mppca:r3", denoise_order="corrected", shell=2000)
    d = cfg.to_dict()
    assert PipelineConfig.from_dict(d) == cfg and d["sha256"] == cfg.hash()
    assert d["slots"]["denoise"] == {"impl": "mppca", "version": "1", "options": {"order": "corrected", "patch_radius": 3}}
    with pytest.raises(ValueError, match="hash does not match"):
        PipelineConfig.from_dict({**d, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="was version"):
        PipelineConfig.from_dict({**d, "slots": {**d["slots"], "track": {**d["slots"]["track"], "version": "0"}}})


def test_config_needs_neither_numpy_nor_torch():
    code = "import sys, tractline.config; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mask_slot_is_prepares_mask():
    s = subject()
    dwi = np.asarray(s.dwi, np.float32)
    own = prepare(dwi, s.affine, s.bval, s.bvec, shell=2800.0).mask
    assert np.array_equal(P.brain_mask(dwi, s.bval, P.Timer(), device="cpu"), own)


def stand_in(fibers, draws=(0,), logp=False):
    from tractline.labelers.base import Labels, MIN_LENGTH_MM, lengths
    _, _, length = lengths(fibers)
    keep = length >= MIN_LENGTH_MM
    return Labels(keep=keep, length_mm=length[keep], tract=np.zeros(int(keep.sum()), np.int64), logp=None)


def test_execute_records_and_field_off():
    s = subject()
    cfg = P.config_for(stand_in).with_(field=None)
    timer = P.Timer()
    r = P.execute(s, cfg, timer, device="cpu", workers=1, labeler=stand_in)
    corr, tg, labels = r
    assert not corr.applied and "field slot is off" in corr.note and corr.field_hz is None
    assert corr.provenance == {"config": r.config, "runtime": r.runtime}
    assert r.config["slots"]["bundles"]["impl"] == "caller-object" and r.config["sha256"] is None
    assert "field" not in r.config["slots"] and r.runtime["device"] == "cpu"
    assert len(tg.fibers) > 0 and timer.total(*P.pipeline_stages()) > 0
    with pytest.raises(ValueError, match="not the bundles slot's"):
        P.execute(s, P.config_for(), P.Timer(), device="cpu", workers=1, labeler=stand_in)
    with pytest.raises(ValueError, match="not as keywords too"):
        P.run(s, stand_in, P.Timer(), device="cpu", config=cfg, shell=2000.0)


def test_review_fixes():
    cfg = PipelineConfig.preset("default")
    # a Stage built by hand with an int for a float option: normalized, so it equals and hashes as the preset
    stages = tuple((s, C.Stage("ukf-two-tensor", (("shell", 2800),)) if s == "model" else st) for s, st in cfg.stages)
    hand = PipelineConfig(stages)
    assert hand == cfg and hand.hash() == cfg.hash() and PipelineConfig.from_dict(hand.to_dict()) == cfg
    # the origin is neither compared nor kept as "+ changes" once the stages are a preset's again
    back = cfg.with_(denoise="mppca").with_(denoise=None)
    assert back == cfg and back.origin == "ukf-rapidparc@1"
    # a record's origin is not trusted; a version is required
    d = cfg.with_(track={"Rs": 0.1}).to_dict()
    assert PipelineConfig.from_dict({**d, "origin": "ukf-rapidparc@1"}).origin == "(from a record)"
    assert PipelineConfig.from_dict(cfg.to_dict()).origin == "ukf-rapidparc@1"
    no_version = {**d, "slots": {**d["slots"], "track": {k: v for k, v in d["slots"]["track"].items() if k != "version"}}}
    no_version.pop("sha256")
    with pytest.raises(ValueError, match="needs impl and version"):
        PipelineConfig.from_dict(no_version)
    # presets are read-only
    with pytest.raises(TypeError):
        C.PRESETS["ukf-rapidparc@1"]["bundles"] = "tractcloud"
    # numpy's numbers pass, as plain ones
    assert P.config_for(tracker={"Rs": np.float32(0.25)}, shell=np.int64(2000)).hash() == \
        P.config_for(tracker={"Rs": float(np.float32(0.25))}, shell=2000.0).hash()
    # TractCloud's forward is always named: no "auto"
    assert C.implementation("bundles", "tractcloud").options["exact"].choices == ("yes", "no")
