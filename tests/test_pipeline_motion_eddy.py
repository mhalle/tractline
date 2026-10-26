"""The motion_eddy slot in the pipeline (pipeline.correct_motion_eddy through execute): it runs between the field
estimate and the single resample, hands the rotated b-vectors to the tracker, records its parameters and QC, runs
without the field too, and leaves the default config alone. The phantom checks plumbing only: its uniform box with a
sharp band does not determine a fit (tests/test_motion_eddy.py, test_correct_one_call); accuracy is
test_motion_eddy.py's."""
import json
import numpy as np
import pytest
from phantom import subject
from tractline import motion_eddy as ME
from tractline import pipeline as P
from tractline.config import PipelineConfig


def stand_in(fibers, draws=(0,), logp=False):
    from tractline.labelers.base import Labels, MIN_LENGTH_MM, lengths
    _, _, length = lengths(fibers)
    keep = length >= MIN_LENGTH_MM
    return Labels(keep=keep, length_mm=length[keep], tract=np.zeros(int(keep.sum()), np.int64), logp=None)


@pytest.fixture(scope="module")
def runs():
    out = {}
    for name, field in (("with field", True), ("field off", False)):
        s = subject()
        cfg = P.config_for(stand_in, motion_eddy={"model": "rigid", "steps": 3})
        if not field:
            cfg = cfg.with_(field=None)
        timer = P.Timer()
        out[name] = (s, cfg, timer, P.execute(s, cfg, timer, device="cpu", workers=1, labeler=stand_in))
    return out


def test_default_untouched():
    cfg = PipelineConfig.preset("default")
    assert cfg.hash() == "28d01a1d3947a3f2f2de450b9b472c61e228129a156fad14be37b252f5574678"
    assert not cfg.is_on("motion_eddy") and "motion_eddy" not in P.pipeline_stages("", cfg)
    on = P.config_for(motion_eddy=True)
    assert on.hash() != cfg.hash() and on.opts("motion_eddy")["model"] == "quadratic"
    assert P.pipeline_stages("", on)[:3] == ["field_estimate", "motion_eddy", "field_apply"]
    with pytest.raises(ValueError):
        P.config_for(motion_eddy="cubic")


def test_slot_runs_with_the_field(runs):
    s, cfg, timer, r = runs["with field"]
    c = r.correction
    assert c.field_hz is not None and c.motion_eddy["field used"] and c.motion_eddy["reference"] == "first-b0"
    assert c.bvec.shape == s.bvec.shape and c.dwi.shape == s.dwi.shape and c.dwi.dtype == np.float32
    assert {"field_estimate", "motion_eddy", "field_apply"} <= set(timer.seconds)
    assert "one resample" in c.note
    assert r.config["slots"]["motion_eddy"]["impl"] == "tractline" and len(r.tractogram.fibers) > 0
    assert len(c.motion_eddy["theta"]) == s.dwi.shape[-1] and len(c.motion_eddy["theta"][0]) == 6   # rigid


def test_slot_runs_without_the_field(runs):
    s, cfg, timer, r = runs["field off"]
    c = r.correction
    assert c.field_hz is None and not c.motion_eddy["field used"] and c.motion_eddy["reference"] == "b0-mean"
    assert "field slot is off" in c.note and c.bvec is not None


def test_tracker_gets_the_rotated_bvecs(runs, monkeypatch):
    """prepare receives Correction.bvec, not the b-vectors as acquired."""
    s, cfg, _, _ = runs["with field"]
    seen = {}
    real = P.prepare

    def spy(dwi, affine, bval, bvec, **kw):
        seen["bvec"] = np.array(bvec)
        return real(dwi, affine, bval, bvec, **kw)
    monkeypatch.setattr(P, "prepare", spy)
    r = P.execute(subject(), cfg, P.Timer(), device="cpu", workers=1, labeler=stand_in)
    assert np.array_equal(seen["bvec"], r.correction.bvec) and not np.array_equal(seen["bvec"], s.bvec)


def test_qc_files(runs, tmp_path):
    s, _, _, r = runs["with field"]
    summary = ME.write_qc(r.correction.motion_eddy, tmp_path)
    lines = (tmp_path / "motion_eddy_qc.tsv").read_text().splitlines()
    assert lines[0].split("\t")[:len(ME.QC_COLUMNS)] == list(ME.QC_COLUMNS) and len(lines) == 1 + s.dwi.shape[-1]
    assert json.loads((tmp_path / "motion_eddy_qc.json").read_text()) == json.loads(json.dumps(summary))
    assert summary["volumes"] == s.dwi.shape[-1] and summary["field used"]


def test_more_plumbing(tmp_path, monkeypatch):
    """The quadratic model's QC columns; no pair with the field slot on; run()'s keyword and its clash with config=;
    the default run's stages exactly STAGES; an odd SH order refused before any work; PTT gets the rotated b-vectors."""
    s = subject()
    cfg = P.config_for(stand_in, motion_eddy={"model": "quadratic", "steps": 2})
    r = P.execute(s, cfg, P.Timer(), device="cpu", workers=1, labeler=stand_in)
    ME.write_qc(r.correction.motion_eddy, tmp_path)
    head = (tmp_path / "motion_eddy_qc.tsv").read_text().splitlines()[0].split("\t")
    assert head[len(ME.QC_COLUMNS):] == [f"eddy_e{k}_vox" for k in range(10)]
    assert r.runtime["motion_eddy backend"] == "torch" and r.correction.applied
    s2 = subject()
    s2.b0s = None                                                           # no pair, the field slot on
    c2 = P.execute(s2, cfg, P.Timer(), device="cpu", workers=1, labeler=stand_in).correction
    assert not c2.applied and c2.field_hz is None and c2.note.startswith("not corrected: no b0s from both")
    timer = P.Timer()
    corr, _, _ = P.run(subject(), stand_in, timer, device="cpu", workers=1, motion_eddy={"model": "rigid", "steps": 2})
    assert corr.motion_eddy["model"] == "rigid" and "motion_eddy" in timer.seconds
    with pytest.raises(ValueError, match="motion_eddy"):
        P.run(subject(), stand_in, P.Timer(), device="cpu", workers=1, config=cfg, motion_eddy="rigid")
    timer = P.Timer()
    P.run(subject(), stand_in, timer, device="cpu", workers=1, config=P.config_for(stand_in), motion_eddy=False)
    assert set(timer.seconds) == set(P.STAGES)
    with pytest.raises(ValueError):
        P.config_for(motion_eddy={"sh_order": 5})
    seen = {}
    real = P._shell_volumes

    def spy(dwi, bval, bvec, shell):
        seen["bvec"] = np.array(bvec)
        return real(dwi, bval, bvec, shell)
    monkeypatch.setattr(P, "_shell_volumes", spy)
    pcfg = P.config_for(stand_in, motion_eddy={"model": "rigid", "steps": 2}).with_(
        model="csd", track={"impl": "ptt", "stopping": "mask", "seeding": "fa", "seed_count": 300, "min_len": 10.0})
    rp = P.execute(subject(), pcfg, P.Timer(), device="cpu", workers=1, labeler=stand_in)
    keep = (np.asarray(subject().bval) < 50) | (np.abs(np.asarray(subject().bval) - pcfg.opts("model")["shell"]) < 50)
    assert np.array_equal(seen["bvec"], rp.correction.bvec[:, keep])
