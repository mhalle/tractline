"""tractline.bids on synthetic BIDS sessions written from the phantom: each way of finding the series that
correct a diffusion series, each check that refuses them, and the pipeline with and without a pair."""
import json
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("scipy")
from phantom import subject                                                  # noqa: E402
from tractline import bids, pipeline as P                                   # noqa: E402

SIDE = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05, "EchoTime": 0.08, "ShimSetting": [1, 2, 3]}


def write(path, data, affine, side, bval=None, bvec=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, np.float32), affine), path)
    path.with_name(path.name.replace(".nii.gz", ".json")).write_text(json.dumps(side))
    if bval is not None:
        np.savetxt(path.with_name(path.name.replace(".nii.gz", ".bval")), np.asarray(bval)[None])
        np.savetxt(path.with_name(path.name.replace(".nii.gz", ".bvec")), bvec)


@pytest.fixture
def session(tmp_path):
    """A writer: the diffusion series (j-) always; partners as asked."""
    s = subject()
    ses = tmp_path / "sub-01"
    dwi = ses / "dwi" / "sub-01_acq-AP_dwi.nii.gz"

    def make(dwi_side=None, partners=()):
        write(dwi, s.dwi, s.affine, {**SIDE, **(dwi_side or {})}, s.bval, s.bvec)
        for rel, side, data, affine in partners:
            p = ses / rel
            extra = {}
            if p.name.endswith("_dwi.nii.gz"):
                extra = dict(bval=[0.0] * data.shape[-1], bvec=np.zeros((3, data.shape[-1])))
            write(p, data, s.affine if affine is None else affine, {**SIDE, **side}, **extra)
        return dwi
    make.s = s
    return make


def pa(s):
    return s.b0s[..., 2:]                                                    # the phantom's reversed b0s


def test_same_folder_pair(session):
    dwi = session(partners=[("dwi/sub-01_acq-PA_dwi.nii.gz", {"PhaseEncodingDirection": "j"}, pa(session.s), None)])
    b = bids.load(dwi)
    assert b.pairing.ok and b.pairing.how == "same folder", b.pairing.message
    assert b.b0s.shape[-1] == 4 and [v[1] for v in b.pe_vectors] == [-1, -1, 1, 1]
    assert all(c.status == "ok" for c in b.pairing.checks)


def test_intended_for_and_b0field_precedence(session):
    fm = {"IntendedFor": ["bids::sub-01/dwi/sub-01_acq-AP_dwi.nii.gz"]}
    parts = [("fmap/sub-01_dir-AP_epi.nii.gz", {**fm}, pa(session.s), None),
             ("fmap/sub-01_dir-PA_epi.nii.gz", {**fm, "PhaseEncodingDirection": "j"}, pa(session.s), None)]
    b = bids.load(session(partners=parts))
    assert b.pairing.ok and b.pairing.how == "IntendedFor" and len(b.pairing.series) == 2
    assert b.b0s.shape[-1] == 6                                              # its own 2, then AP's 2 and PA's 2
    parts.append(("fmap/sub-01_acq-other_dir-PA_epi.nii.gz", {"PhaseEncodingDirection": "j", "B0FieldIdentifier": "f1"}, pa(session.s), None))
    b = bids.load(session(dwi_side={"B0FieldSource": "f1"}, partners=parts))
    assert b.pairing.how == "B0FieldIdentifier" and [p.name for p in b.pairing.series] == ["sub-01_acq-other_dir-PA_epi.nii.gz"]


@pytest.mark.parametrize("side, words", [
    ({"PhaseEncodingDirection": "j", "ShimSetting": [1, 2, 4]}, "another shim"),
    ({"PhaseEncodingDirection": "j", "EchoTime": 0.1}, "different protocol"),
    ({"PhaseEncodingDirection": "i"}, "another axis"),
    ({"PhaseEncodingDirection": "j-"}, "one way"),
])
def test_refused(session, side, words):
    dwi = session(partners=[("fmap/sub-01_dir-X_epi.nii.gz", {**side, "IntendedFor": "sub-01/dwi/sub-01_acq-AP_dwi.nii.gz"}, pa(session.s), None)])
    b = bids.load(dwi)
    assert not b.pairing.ok and words in b.pairing.message and b.b0s is None, b.pairing.message


def test_missing_readout_assumed_when_protocols_match(session):
    side = {"PhaseEncodingDirection": "j", "IntendedFor": "sub-01/dwi/sub-01_acq-AP_dwi.nii.gz", "TotalReadoutTime": None}
    dwi = session(partners=[("fmap/sub-01_dir-PA_epi.nii.gz", side, pa(session.s), None)])
    p = dwi.parent.parent / "fmap" / "sub-01_dir-PA_epi.json"
    d = json.loads(p.read_text()); d.pop("TotalReadoutTime"); p.write_text(json.dumps(d))
    b = bids.load(dwi)
    assert b.pairing.ok and "assumed" in b.pairing.sources["sub-01_dir-PA_epi.nii.gz TotalReadoutTime"]
    assert np.all(b.b0_readout_s == 0.05)


def test_unknown_shim_said(session):
    side = {"PhaseEncodingDirection": "j", "ShimSetting": None}
    dwi = session(partners=[("dwi/sub-01_acq-PA_dwi.nii.gz", side, pa(session.s), None)])
    p = dwi.parent / "sub-01_acq-PA_dwi.json"
    d = json.loads(p.read_text()); d.pop("ShimSetting"); p.write_text(json.dumps(d))
    b = bids.load(dwi)
    assert b.pairing.ok and "not verified" in b.pairing.message and "shim" in b.pairing.message


def test_partner_on_another_grid_resampled(session):
    A = session.s.affine.copy(); A[:3, 3] += [1.0, 0.0, 2.0]                     # placed half a voxel and a voxel off
    dwi = session(partners=[("dwi/sub-01_acq-PA_dwi.nii.gz", {"PhaseEncodingDirection": "j"}, pa(session.s), A)])
    b = bids.load(dwi)
    assert b.pairing.ok and b.b0s.shape[:3] == session.s.dwi.shape[:3] and np.isfinite(b.b0s).all()
    assert [v[1] for v in b.pe_vectors] == [-1, -1, 1, 1]


def test_no_phase_encoding_refused(session):
    dwi = session(dwi_side={"PhaseEncodingDirection": None})
    p = dwi.with_name(dwi.name.replace(".nii.gz", ".json"))
    d = json.loads(p.read_text()); d.pop("PhaseEncodingDirection"); p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="PhaseEncodingDirection"):
        bids.load(dwi)


def test_pipeline_corrects_only_with_a_pair(session):
    from test_pipeline_cpu import stand_in
    b = bids.load(session())                                                 # no partner
    assert not b.pairing.ok and "no series" in b.pairing.message
    corr, tg, _ = P.run(b, stand_in, P.Timer(), device="cpu", workers=1)
    assert not corr.applied and corr.field_hz is None and corr.note == b.pairing.message and len(tg.fibers) > 500
    b = bids.load(session(partners=[("dwi/sub-01_acq-PA_dwi.nii.gz", {"PhaseEncodingDirection": "j"}, pa(session.s), None)]))
    timer = P.Timer()
    corr = P.correct(b, timer, device="cpu")
    assert corr.applied and corr.residual_left is not None and np.isfinite(corr.residual_left)
    assert np.quantile(np.abs(corr.field_hz[session.s.brain]), 0.99) < 2.0     # no distortion in the phantom
    assert set(timer.seconds) == {"field_estimate", "field_apply"}
