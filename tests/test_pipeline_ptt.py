"""The model slot's csd and the track slot's ptt through pipeline.execute, on the synthetic phantom (one band of fibers):
the configs check their inputs, the run tracks along the band, records its provenance and stage times, and repeats
exactly (CPU: the torch prototype; MPS: the Metal kernel)."""
import numpy as np
import pytest
import torch
from phantom import subject
from tractline import pipeline as P
from tractline.config import PipelineConfig


def stand_in(fibers, draws=(0,), logp=False):
    from tractline.labelers.base import Labels, MIN_LENGTH_MM, lengths
    _, _, length = lengths(fibers)
    keep = length >= MIN_LENGTH_MM
    return Labels(keep=keep, length_mm=length[keep], tract=np.zeros(int(keep.sum()), np.int64), logp=None)


def cfg(**track):
    return P.config_for(stand_in).with_(model="csd", track={"impl": "ptt", "stopping": "mask", "seeding": "fa",
                                                            "seed_count": 300, "min_len": 10.0, **track})


def test_config_and_inputs_checked():
    with pytest.raises(ValueError, match="requires model"):
        PipelineConfig.preset("default").with_(track="ptt")                  # ptt needs csd
    s = subject()
    c = P.config_for(stand_in).with_(model="csd", track="ptt")               # stopping act, seeding wm: tissue needed
    with pytest.raises(ValueError, match="needs tissue"):
        P.execute(s, c, P.Timer(), device="cpu", labeler=stand_in)
    with pytest.raises(ValueError, match="tissue of shape"):
        P.execute(s, c, P.Timer(), device="cpu", labeler=stand_in, tissue=np.zeros((2, 2, 2, 4)))


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_csd_ptt_tracks_the_band(device):
    s = subject()
    c = cfg()
    timer = P.Timer()
    r = P.execute(s, c, timer, device=device, labeler=stand_in)
    tg = r.tractogram
    assert len(tg.fibers) > 0.5 * 300 and r.labels.keep.sum() > 0
    assert tg.stats["backend"] == ("metal" if device == "mps" else "torch")
    assert r.config["slots"]["track"]["impl"] == "ptt" and r.runtime["ptt backend"] == tg.stats["backend"]
    assert set(P.pipeline_stages(cfg=c)) <= set(timer.seconds) and timer.total(*P.pipeline_stages(cfg=c)) > 0
    # the band runs along one axis: the segments mostly follow the dominant direction
    seg = np.concatenate([np.diff(f, axis=0) for f in tg.fibers])
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    axis = np.argmax(np.abs(seg).mean(0))
    assert np.abs(seg[:, axis]).mean() > 0.9
    again = P.execute(s, c, P.Timer(), device=device, labeler=stand_in)
    assert len(again.tractogram.fibers) == len(tg.fibers)
    assert all(np.array_equal(a, b) for a, b in zip(again.tractogram.fibers, tg.fibers))
    assert np.array_equal(again.tractogram.stats["seed_index"], tg.stats["seed_index"])


def test_refusals_before_work():
    s = subject()
    with pytest.raises(ValueError, match="grouping must be 'index'"):
        P.execute(s, cfg().with_(bundles={"impl": "rapidparc", "grouping": "hash-sorted"}), P.Timer(), device="cpu")
    with pytest.raises(ValueError, match="does not use it"):
        P.execute(s, cfg(), P.Timer(), device="cpu", labeler=stand_in, tissue=np.zeros(s.dwi.shape[:3] + (4,)))


def phantom_tissue(s):
    """WM in the band; GM caps across the brain at the band's two ends (along y); CSF in slabs beside the band."""
    X, Y, Z = s.dwi.shape[:3]
    x, y, z = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")
    brain, band = s.brain, s.band if hasattr(s, "band") else None
    if band is None:
        band = brain & (np.abs(x - X / 2 + 0.5) < 2) & (np.abs(z - Z / 2 + 0.5) < 2)
    ys = np.flatnonzero(brain.any((0, 2)))
    caps = brain & ((y <= ys[0] + 2) | (y >= ys[-1] - 2))
    csf = brain & ~caps & (np.abs(x - X / 2 + 0.5) >= 4) & (np.abs(x - X / 2 + 0.5) < 6)
    pve = np.zeros((X, Y, Z, 4))
    pve[..., 0] = band & ~caps
    pve[..., 1] = caps
    pve[..., 3] = csf
    rest = brain & ~band & ~caps & ~csf
    pve[..., 0] += 0.4 * rest                                                   # the rest of the brain: mixed, below 0.5
    pve[..., 1] += 0.4 * rest
    return pve, caps


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_act_and_wm_seeding(device):
    """With tissue maps: every kept point but a streamline's seed is a TRACKPOINT under the same ACT (the deciding point
    is not kept), so none is in CSF; nearly every streamline ends within a step of a GM cap; the run records the maps'
    hash."""
    from tractline import fields as F
    s = subject()
    pve, caps = phantom_tissue(s)
    c = P.config_for(stand_in).with_(model="csd", track={"impl": "ptt", "seed_count": 300, "min_len": 10.0})
    r = P.execute(s, c, P.Timer(), device=device, labeler=stand_in, tissue=pve)
    fibers = r.tractogram.fibers
    assert len(fibers) > 50 and r.runtime["tissue"]["shape"] == list(pve.shape)
    inc = pve[..., 1].copy()
    inc[pve.sum(-1) < 1e-6] = 1.0
    for ax in range(3):
        for idx in (0, -1):
            sl = [slice(None)] * 3
            sl[ax] = idx
            inc[tuple(sl)] = 1.0
    act = F.Act(inc, pve[..., 3])
    inv = np.linalg.inv(s.affine)
    pts = np.concatenate(fibers)
    vox = pts @ inv[:3, :3].T + inv[:3, 3]
    st = act(vox).numpy()
    per = np.split(st != F.TRACKPOINT, np.cumsum([len(f) for f in fibers])[:-1])
    # the seed is not tested (as DIPY): uniform in a WM voxel next to a GM cap, it may sit where the interpolated
    # include exceeds 0.5 (measured: 1 of 28,915 points on MPS); every other kept point is a TRACKPOINT
    assert max(int(x.sum()) for x in per) <= 1 and (st != F.TRACKPOINT).mean() < 0.001
    ends = np.concatenate([[f[0], f[-1]] for f in fibers]) @ inv[:3, :3].T + inv[:3, 3]
    cap_y = np.flatnonzero(caps.any((0, 2)))
    near = np.min(np.abs(ends[:, 1:2] - cap_y[None, :]), 1) <= 1.0           # within a voxel (a 0.5 mm step) of a cap
    assert near.mean() > 0.9
