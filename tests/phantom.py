"""A small synthetic subject for the tests: a block of "brain" (isotropic diffusion) crossed by a band of
fibers along the y axis, scanned at b = 2800 with two b0s, and a reversed-phase-encoding pair of b0s with
no distortion. Everything pipeline.run reads, in its conventions (pipeline.py's docstring)."""
from types import SimpleNamespace
import numpy as np

SHAPE, VOX, B, NDIR = (32, 44, 20), 2.0, 2800.0, 30


def directions(n):
    """n unit vectors spread over the sphere (a Fibonacci spiral)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def subject(seed=0, noise=8.0):
    rng = np.random.default_rng(seed)
    X, Y, Z = SHAPE
    x, y, z = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")
    brain = (np.abs(x - X / 2 + 0.5) < X / 2 - 4) & (np.abs(y - Y / 2 + 0.5) < Y / 2 - 3) & (np.abs(z - Z / 2 + 0.5) < Z / 2 - 4)
    band = brain & (np.abs(x - X / 2 + 0.5) < 2) & (np.abs(z - Z / 2 + 0.5) < 2)
    g = directions(NDIR)
    l1, l2, iso = 1.7e-3, 0.3e-3, 0.8e-3                                  # mm^2/s
    adc_band = l2 + (l1 - l2) * g[:, 1] ** 2                              # the band's tensor, its axis along y
    s0 = 1000.0 * brain + 30.0
    dw = np.where(band[..., None], np.exp(-B * adc_band), np.exp(-B * iso)) * s0[..., None]
    dwi = np.concatenate([s0[..., None], s0[..., None], dw], -1)
    dwi = np.abs(dwi + rng.normal(0, noise, dwi.shape)).astype(np.float32)
    bval = np.r_[0.0, 0.0, np.full(NDIR, B)]
    bvec = np.concatenate([np.zeros((2, 3)), g]).T                        # (3, G), FSL: image axes
    pa = np.abs(s0[..., None] + rng.normal(0, noise, SHAPE + (2,)))
    b0s = np.concatenate([dwi[..., :2].astype(np.float64), pa], -1)
    affine = np.diag([-VOX, VOX, VOX, 1.0]); affine[:3, 3] = [X * VOX / 2, -Y * VOX / 2, -Z * VOX / 2]
    return SimpleNamespace(dwi=dwi, affine=affine, vox=np.full(3, VOX), bval=bval, bvec=bvec, b0s=b0s,
                           pe_vectors=np.array([[0, -1, 0], [0, -1, 0], [0, 1, 0], [0, 1, 0]], float),
                           pe_axis=1, pe_sign=-1.0, readout_s=0.05, brain=brain, band=band)
