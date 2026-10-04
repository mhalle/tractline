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
    assert not corr.applied and corr.field_hz is None and corr.note == b.pairing.message and len(tg.fibers) > 700
    b = bids.load(session(partners=[("dwi/sub-01_acq-PA_dwi.nii.gz", {"PhaseEncodingDirection": "j"}, pa(session.s), None)]))
    timer = P.Timer()
    corr = P.correct(b, timer, device="cpu")
    # the phantom's pair has no distortion: the two series differ by noise, so the residual is not judged
    assert corr.applied and corr.residual_left is None and "not judged" in corr.note and not corr.warnings
    assert np.quantile(np.abs(corr.field_hz[session.s.brain]), 0.99) < 2.0     # no distortion in the phantom
    assert set(timer.seconds) == {"field_estimate", "field_apply"}


def test_partial_coverage_refused(session):
    """A partner covering half the diffusion grid gave a 484 Hz field on the undistorted phantom: refused."""
    A = session.s.affine.copy(); A[:3, 3] += A[:3, 2] * 10                    # its slab starts 10 slices up
    dwi = session(partners=[("dwi/sub-01_acq-PA_dwi.nii.gz", {"PhaseEncodingDirection": "j"}, pa(session.s)[:, :, :10], A)])
    b = bids.load(dwi)
    assert not b.pairing.ok and "covers" in b.pairing.message, b.pairing.message


def test_gre_field_map_refused(session):
    dwi = session(dwi_side={"B0FieldSource": "f1"},
                  partners=[("fmap/sub-01_phasediff.nii.gz", {"B0FieldIdentifier": "f1", "PhaseEncodingDirection": "j"}, pa(session.s)[..., :1], None)])
    b = bids.load(dwi)
    assert not b.pairing.ok and "GRE field map" in b.pairing.message, b.pairing.message


def test_no_b0_refused(session):
    dwi = session()
    np.savetxt(dwi.with_name("sub-01_acq-AP_dwi.bval"), np.full((1, session.s.dwi.shape[-1]), 1000.0))
    with pytest.raises(ValueError, match="no b0"):
        bids.load(dwi)


def test_partner_without_b0_dropped_others_used(session):
    s = session.s
    parts = [("dwi/sub-01_acq-PA_dwi.nii.gz", {"PhaseEncodingDirection": "j"}, pa(s), None),
             ("dwi/sub-01_acq-PB_dwi.nii.gz", {"PhaseEncodingDirection": "j"}, pa(s), None)]
    dwi = session(partners=parts)
    np.savetxt(dwi.with_name("sub-01_acq-PB_dwi.bval"), np.full((1, 2), 1000.0))
    b = bids.load(dwi)
    assert b.pairing.ok and [p.name for p in b.pairing.series] == ["sub-01_acq-PA_dwi.nii.gz"]
    assert any("PB" in d and "no b0" in d for d in b.pairing.dropped) and "not used" in b.pairing.message


def test_failing_partner_dropped_when_others_suffice(session):
    s = session.s
    fm = "sub-01/dwi/sub-01_acq-AP_dwi.nii.gz"
    parts = [("fmap/sub-01_dir-PA_epi.nii.gz", {"PhaseEncodingDirection": "j", "IntendedFor": fm}, pa(s), None),
             ("fmap/sub-01_acq-x_dir-PA_epi.nii.gz", {"PhaseEncodingDirection": "j", "IntendedFor": fm, "ShimSetting": [9, 9, 9]}, pa(s), None)]
    b = bids.load(session(partners=parts))
    assert b.pairing.ok and len(b.pairing.series) == 1 and "another shim" in b.pairing.message


def test_inheritance(tmp_path, session):
    """PhaseEncodingDirection and the .bval/.bvec from the dataset root (BIDS's inheritance principle)."""
    dwi = session(partners=[("dwi/sub-01_acq-PA_dwi.nii.gz", {"PhaseEncodingDirection": "j"}, pa(session.s), None)])
    root = dwi.parent.parent.parent                                          # the dataset: above sub-01
    (root / "dataset_description.json").write_text(json.dumps({"Name": "t", "BIDSVersion": "1.9.0"}))
    p = dwi.with_name("sub-01_acq-AP_dwi.json")
    d = json.loads(p.read_text()); d.pop("PhaseEncodingDirection"); p.write_text(json.dumps(d))
    (root / "dwi.json").write_text(json.dumps({"PhaseEncodingDirection": "j-"}))
    for ext in (".bval", ".bvec"):
        src = dwi.with_name("sub-01_acq-AP_dwi" + ext)
        (root / ("dwi" + ext)).write_text(src.read_text()); src.unlink()
    b = bids.load(dwi)
    assert b.pe_sign == -1 and len(b.bval) == session.s.dwi.shape[-1] and b.pairing.ok, b.pairing.message


def test_swapped_voxels_refused(tmp_path):
    """Voxel sizes compared per world axis: diffusion (2, 2.5, 2) mm against a partner's (2.5, 2, 2) - equal
    when sorted, but 2.5 against 2 mm along phase encoding (y): refused."""
    s = subject()
    ses = tmp_path / "sub-01"
    dwi = ses / "dwi" / "sub-01_acq-AP_dwi.nii.gz"
    write(dwi, s.dwi, np.diag([-2.0, 2.5, 2.0, 1.0]), SIDE, s.bval, s.bvec)
    write(ses / "dwi" / "sub-01_acq-PA_dwi.nii.gz", pa(s), np.diag([-2.5, 2.0, 2.0, 1.0]), {**SIDE, "PhaseEncodingDirection": "j"},
          bval=[0.0, 0.0], bvec=np.zeros((3, 2)))
    b = bids.load(dwi)
    assert not b.pairing.ok and "different protocol" in b.pairing.message, b.pairing.message


def test_readout_not_assumed_when_matrix_differs(session):
    s = session.s
    side = {"PhaseEncodingDirection": "j", "IntendedFor": "sub-01/dwi/sub-01_acq-AP_dwi.nii.gz"}
    dwi = session(partners=[("fmap/sub-01_dir-PA_epi.nii.gz", side, pa(s)[:, ::2], None)])
    p = dwi.parent.parent / "fmap" / "sub-01_dir-PA_epi.json"
    d = json.loads(p.read_text()); d.pop("TotalReadoutTime"); p.write_text(json.dumps(d))
    b = bids.load(dwi)
    assert not b.pairing.ok and "none can be assumed" in b.pairing.message, b.pairing.message


def test_estimated_readout_used_and_said(session):
    side = {"PhaseEncodingDirection": "j", "IntendedFor": "sub-01/dwi/sub-01_acq-AP_dwi.nii.gz", "EstimatedTotalReadoutTime": 0.05}
    dwi = session(partners=[("fmap/sub-01_dir-PA_epi.nii.gz", side, pa(session.s), None)])
    p = dwi.parent.parent / "fmap" / "sub-01_dir-PA_epi.json"
    d = json.loads(p.read_text()); d.pop("TotalReadoutTime"); p.write_text(json.dumps(d))
    b = bids.load(dwi)
    assert b.pairing.ok and b.pairing.sources["sub-01_dir-PA_epi.nii.gz TotalReadoutTime"].startswith("estimated")


@pytest.mark.parametrize("code", ["y-", None, 3])
def test_unusable_phase_encoding_refused_cleanly(session, code):
    dwi = session()
    p = dwi.with_name("sub-01_acq-AP_dwi.json")
    d = json.loads(p.read_text()); d["PhaseEncodingDirection"] = code; p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="PhaseEncodingDirection"):
        bids.load(dwi)


def test_path_with_nii_in_a_folder_name(tmp_path):
    s = subject()
    ses = tmp_path / "proj.nii_out" / "sub-01"
    dwi = ses / "dwi" / "sub-01_acq-AP_dwi.nii.gz"
    write(dwi, s.dwi, s.affine, SIDE, s.bval, s.bvec)
    write(ses / "dwi" / "sub-01_acq-PA_dwi.nii.gz", pa(s), s.affine, {**SIDE, "PhaseEncodingDirection": "j"},
          bval=[0.0, 0.0], bvec=np.zeros((3, 2)))
    assert bids.load(dwi).pairing.ok


def test_intended_for_without_extension_or_other_form(session):
    side = {"PhaseEncodingDirection": "j", "IntendedFor": ["bids::sub-01/dwi/sub-01_acq-AP_dwi.nii"]}
    b = bids.load(session(partners=[("fmap/sub-01_dir-PA_epi.nii.gz", side, pa(session.s), None)]))
    assert b.pairing.ok and b.pairing.how == "IntendedFor"


def test_missing_extra_named(tmp_path):
    import subprocess, sys
    code = ("import sys, importlib.abc, importlib.machinery\n"
            "class B(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, n, p, t=None):\n"
            "        if n.split('.')[0] == 'nibabel': raise ImportError('blocked')\n"
            "sys.meta_path.insert(0, B())\n"
            "from tractline import bids\n"
            f"bids.load({str(tmp_path / 'x_dwi.nii.gz')!r})\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "tractline[bids]" in r.stderr, r.stderr[-800:]


def test_flat_dcm2niix_output_with_partners(tmp_path):
    """dcm2niix's flat output (no BIDS tree, no dataset_description.json): only same-named sidecars, the
    partner named."""
    s = subject()
    dwi = tmp_path / "DWI_AP_5.nii.gz"
    write(dwi, s.dwi, s.affine, SIDE, s.bval, s.bvec)
    write(tmp_path / "DWI_PA_6.nii.gz", pa(s), s.affine, {**SIDE, "PhaseEncodingDirection": "j"}, bval=[0.0, 0.0], bvec=np.zeros((3, 2)))
    (tmp_path / "OTHER_5.json").write_text(json.dumps({"PhaseEncodingDirection": "i"}))     # same last token: not inherited
    b = bids.load(dwi, partners=[tmp_path / "DWI_PA_6.nii.gz"])
    assert b.pairing.ok and b.pairing.how == "given" and b.pe_sign == -1, b.pairing.message
