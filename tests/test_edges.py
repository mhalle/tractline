"""Inputs the pipeline's modules refuse, and what they do at the edges."""
import subprocess, sys, textwrap
import numpy as np, torch
import pytest
from phantom import subject
from tractline import prep, susceptibility as S, trx, mask
from tractline.labelers.base import Labels


def test_prepare_names_a_missing_shell():
    s = subject()
    with pytest.raises(ValueError, match="no b = 1000 shell"):
        prep.prepare(s.dwi, s.affine, s.bval, s.bvec, shell=1000.0)


def test_prepare_keeps_uint16_unwrapped():
    s = subject()
    t = prep.prepare((s.dwi * 40).astype(np.uint16), s.affine, s.bval, s.bvec)      # values past int16's range
    assert t.header["type"] == "float" and np.asarray(t.dwi).max() > 32767


def test_estimate_refuses_one_direction_and_nonfinite():
    s = subject()
    with pytest.raises(ValueError):
        S.estimate(s.b0s[..., :2], s.vox, s.pe_vectors[:2], s.readout_s, device="cpu")
    b0s = s.b0s.copy(); b0s[3, 3, 3, 0] = np.nan
    with pytest.raises(ValueError):
        S.estimate(b0s, s.vox, s.pe_vectors, s.readout_s, device="cpu")


def test_estimate_identical_b0s_give_a_finite_zero_field():
    s = subject()
    b0s = np.repeat(s.b0s[..., :1], 4, -1)
    h, _, _ = S.estimate(b0s, s.vox, s.pe_vectors, s.readout_s, device="cpu", dtype=torch.float32)
    assert np.isfinite(h).all() and np.abs(h).max() < 1e-3


def test_otsu_matches_dipy():
    dipy = pytest.importorskip("dipy.segment.threshold")                   # the dipy group
    rng = np.random.default_rng(0)
    for k in range(200):
        x = np.concatenate([rng.normal(0, 1, 400), rng.normal(rng.uniform(1, 6), rng.uniform(0.3, 2), rng.integers(10, 900))])
        assert dipy.otsu(x) == mask.otsu(x)


def test_trx_write_refuses(tmp_path):
    keep = np.ones(1, bool)
    labels = Labels(keep=keep, length_mm=np.array([50.0]), tract=np.array([1]), logp=None)
    with pytest.raises(ValueError, match="logp=True"):
        trx.write(tmp_path / "x.trx", None, None, labels)
    other = tmp_path / "notes"; other.mkdir(); (other / "keep.txt").write_text("mine")
    labels.logp = torch.zeros(1, 1600, dtype=torch.float16)
    s = subject()
    from types import SimpleNamespace
    tg = SimpleNamespace(fibers=[np.zeros((3, 3), np.float32)], stats={"seed_index": np.array([0])}, mask=s.brain)
    with pytest.raises(FileExistsError):
        trx.write(other, s, tg, labels)
    assert (other / "keep.txt").read_text() == "mine"


@pytest.mark.parametrize("guard, ok", [
    ('if __name__ == "__main__":\n    check()', True),
    ('if "__main__" == __name__:\n    check()', True),
    ('if __name__ in ("__main__",):\n    check()', True),
    ('# if __name__ == "__main__":\ncheck()', False),
    ('check()', False),
])
def test_main_guard_check(tmp_path, guard, ok):
    script = tmp_path / "script.py"
    script.write_text("from tractline.ukf import require_main_guard as check\n" + textwrap.dedent(guard) + "\n")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stderr[-1500:]
