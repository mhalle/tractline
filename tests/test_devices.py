"""The devices run the same pipeline (docs/sota-pipeline-design.md, decision 6: the config hash leaves the device out).
These tolerances catch a device that runs a different pipeline. They do not show that a calibration map carries across
devices. On a real scan the devices differ per streamline by about a fifth of the ensemble's spread (cIIIs04, CPU
against MPS: 93.8 % same tract, 31 % of paths more than 1 mm apart; bench/device_agreement.py). At the ensemble level
the calibrated coverage agrees within 0.003 (bench/device_ensemble.py). Skipped where there is no GPU.

On the phantom (2026-10-24, M2), CPU against MPS:
- the field in the brain differs by 0.075 Hz at its 99th percentile (0.16 at most): 0.004 voxel of displacement;
- 14 of 4,530 mask voxels differ;
- the seeds are the same 786; the density correlation is 0.999998; one streamline ends more than 1 mm from its
  counterpart.

The phantom is weak evidence:
- it is one straight band with no distortion: |field| is under 1 Hz at the 99th percentile;
- its ~790 streamlines are all labeled one tract;
- on an NVIDIA A10 (2026-10-27, its first run) the field differs by 0.74 Hz at the 99th percentile, 0.037 voxel
  (CUDA runs every level of the estimate on the GPU; MPS runs the coarse ones on the CPU, as the CPU run does). The
  field is compared as displacement, the quantity that moves the image: under 0.1 voxel.
Streamlines are matched by seed voxel (ensemble.seed_keys), so a differing mask voxel that moves a few seeds does not turn
the per-streamline checks off."""
import numpy as np
import pytest
import torch
from phantom import subject
from tractline import pipeline as P
from tractline.ensemble import seed_keys

GPUS = [d for d, ok in (("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())) if ok]


def density(tg, affine, shape):
    """Streamlines per voxel (each counted once in a voxel it visits), on the scan's grid."""
    inv = np.linalg.inv(affine)
    d = np.zeros(shape)
    for f in tg.fibers:
        ijk = np.unique(np.round(f @ inv[:3, :3].T + inv[:3, 3]).astype(int), axis=0)
        ijk = ijk[np.all((ijk >= 0) & (ijk < shape), axis=1)]
        np.add.at(d, tuple(ijk.T), 1)
    return d


@pytest.fixture(scope="module")
def cpu_run():
    from tractline.labelers.rapidparc import Labeler
    s = subject()
    return s, P.execute(s, P.config_for(), P.Timer(), device="cpu", workers=2, labeler=Labeler("cpu"))


@pytest.mark.skipif(not GPUS, reason="no GPU")
@pytest.mark.parametrize("device", GPUS)
def test_gpu_agrees_with_cpu(cpu_run, device):
    from tractline.labelers.rapidparc import Labeler
    s, a = cpu_run
    b = P.execute(s, P.config_for(), P.Timer(), device=device, labeler=Labeler(device))
    assert a.config["sha256"] == b.config["sha256"]                          # one key for both: what this test licenses
    assert a.provenance["pipeline_id"] == b.provenance["pipeline_id"]       # the same algorithm ...
    assert a.provenance["execution_id"] != b.provenance["execution_id"]     # ... on other arithmetic
    df = np.abs(a.correction.field_hz - b.correction.field_hz)[s.brain]
    assert np.quantile(df, 0.99) * s.readout_s < 0.1                          # voxels of displacement along PE
    ma, mb = a.tractogram.mask, b.tractogram.mask
    assert (ma ^ mb).sum() < 0.01 * ma.sum()
    shape = s.dwi.shape[:3]
    r = np.corrcoef(density(a.tractogram, s.affine, shape).ravel(), density(b.tractogram, s.affine, shape).ravel())[0, 1]
    assert r > 0.999
    # streamlines matched by seed voxel: most seeds shared, and the shared ones the same paths and labels
    ka, kb = seed_keys(a.tractogram.stats["seed_voxel"]), seed_keys(b.tractogram.stats["seed_voxel"])
    assert len(np.unique(ka)) == len(ka) and len(np.unique(kb)) == len(kb)   # one streamline per seed voxel
    shared, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    assert len(shared) >= 0.98 * max(len(ka), len(kb))
    fa, fb = a.tractogram.fibers, b.tractogram.fibers
    length = np.array([abs(len(fa[i]) - len(fb[j])) for i, j in zip(ia, ib)])
    apart = np.array([np.abs(fa[i][:min(len(fa[i]), len(fb[j]))] - fb[j][:min(len(fa[i]), len(fb[j]))]).max() > 1.0
                      for i, j in zip(ia, ib)])
    assert np.mean(apart | (length > 2)) < 0.02                              # a path apart, or one that stops elsewhere
    la, lb = a.labels, b.labels
    ta = np.full(len(fa), -1)
    ta[np.flatnonzero(la.keep)] = la.tract
    tb = np.full(len(fb), -1)
    tb[np.flatnonzero(lb.keep)] = lb.tract
    assert np.mean(ta[ia] == tb[ib]) > 0.98                                   # kept or not, and the tract when kept
