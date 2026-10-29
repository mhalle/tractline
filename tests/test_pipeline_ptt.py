"""The model slot's csd and the track slot's ptt through pipeline.execute, on the synthetic phantom (one band of fibers):
the configs check their inputs, the run tracks along the band, records its provenance and stage times, and repeats
exactly (CPU: the torch prototype; MPS: the Metal kernel)."""
import numpy as np
import pytest
import torch
from phantom import subject
from tractline import fields as F, pipeline as P, ptt
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
    d = P.config_for(stand_in).with_(model="csd", track="ptt").opts("track")    # the defaults: from the DWI alone
    assert (d["stopping"], d["stopping_fa"], d["seeding"]) == ("fa", "auto", "fa")
    c = P.config_for(stand_in).with_(model="csd", track={"impl": "ptt", "stopping": "act", "seeding": "wm"})
    with pytest.raises(ValueError, match="needs tissue"):
        P.execute(s, c, P.Timer(), device="cpu", labeler=stand_in)
    with pytest.raises(ValueError, match="tissue of shape"):
        P.execute(s, c, P.Timer(), device="cpu", labeler=stand_in, tissue=np.zeros((2, 2, 2, 4)))
    with pytest.warns(UserWarning, match="seeds by FA"):                       # act alone: seeding stays fa
        with pytest.raises(ValueError, match="needs tissue"):
            P.execute(s, P.config_for(stand_in).with_(model="csd", track={"impl": "ptt", "stopping": "act"}),
                      P.Timer(), device="cpu", labeler=stand_in)
    with pytest.raises(ValueError, match="resolve auto first"):
        P.ptt_stopping({"stopping": "fa", "stopping_fa": "auto"}, np.ones((4, 4, 4), bool), fa_map=np.zeros((4, 4, 4)))


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


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_mask_stopping_keeps_open_ends(device):
    """stopping "mask" (version 2) keeps the paths that end inside the mask for want of a direction: more streamlines
    than the same run keeping only paths that leave the mask at both ends, and a superset of their seeds."""
    s = subject()
    kept = P.execute(s, cfg(), P.Timer(), device=device, labeler=stand_in).tractogram
    real = ptt.Tracker._valid
    try:
        ptt.Tracker._valid = lambda self, st: (st == F.ENDPOINT) | (st == F.OUTSIDEIMAGE)
        strict = P.execute(s, cfg(), P.Timer(), device=device, labeler=stand_in).tractogram
    finally:
        ptt.Tracker._valid = real
    assert len(kept.fibers) > len(strict.fibers)
    assert set(strict.stats["seed_index"]) <= set(kept.stats["seed_index"])


def test_stopping_maps():
    """ptt_stopping: mask includes outside the mask only; fa also where FA < stopping_fa inside it; neither excludes."""
    mask = np.zeros((6, 6, 6), bool)
    mask[1:5, 1:5, 1:5] = True
    fa = np.where(mask, 0.5, 0.0)
    fa[2, 2, 2] = 0.05
    o = {"stopping": "mask", "stopping_fa": 0.1}
    inc, exc = P.ptt_stopping(o, mask, fa_map=fa)
    assert np.array_equal(inc > 0.5, ~mask) and not exc.any()
    inc, exc = P.ptt_stopping({**o, "stopping": "fa"}, mask, fa_map=fa)
    want = ~mask
    want[2, 2, 2] = True
    assert np.array_equal(inc > 0.5, want) and not exc.any()
    # a mask cut by the field of view (as at the pons): fa makes its faces valid ends, mask does not
    cut = mask.copy()
    cut[1:5, 1:5, 0] = True
    fa_cut = np.where(cut, 0.5, 0.0)
    assert (P.ptt_stopping({**o, "stopping": "fa"}, cut, fa_map=fa_cut)[0][1:5, 1:5, 0] > 0.5).all()
    assert not (P.ptt_stopping(o, cut, fa_map=fa_cut)[0][1:5, 1:5, 0] > 0.5).any()


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_fa_stopping(device):
    """stopping "fa" tracks the band, and its FA map matters: more streamlines than at stopping_fa 0 (no low-FA end:
    version 1's mask rule). On the Metal kernel (random numbers keyed by the seed: a path is the same up to where it
    stops) its kept seeds are among stopping "mask"'s on this band; not a guarantee (a forward half that stops at its
    seed starts the backward half differently, and mask's longer path can pass max_len)."""
    s = subject()
    fa = P.execute(s, cfg(stopping="fa", stopping_fa=0.15), P.Timer(), device=device, labeler=stand_in).tractogram
    none = P.execute(s, cfg(stopping="fa", stopping_fa=0.0), P.Timer(), device=device, labeler=stand_in).tractogram
    assert len(fa.fibers) > len(none.fibers)
    if device == "mps":
        mask = P.execute(s, cfg(), P.Timer(), device=device, labeler=stand_in).tractogram
        assert set(fa.stats["seed_index"]) <= set(mask.stats["seed_index"])


def test_fa_above_fa_seed_warns():
    s = subject()
    with pytest.warns(UserWarning, match="above fa_seed"):
        P.execute(s, cfg(stopping="fa", stopping_fa=0.3, seed_count=20), P.Timer(), device="cpu", labeler=stand_in)


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
    c = P.config_for(stand_in).with_(model="csd", track={"impl": "ptt", "stopping": "act", "seeding": "wm",
                                                         "seed_count": 300, "min_len": 10.0})
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


def test_auto_stopping_fa():
    """auto_stopping_fa finds cortex's FA among gray and white matter, not the sulci's fluid (lower FA, higher MD), and
    sets 1.15 times it; too little white matter or tissue is refused, not guessed."""
    g = np.random.default_rng(0)
    n = 20_000
    fa = np.r_[g.normal(0.13, 0.02, n), g.normal(0.45, 0.08, n), g.normal(0.09, 0.02, 2 * n)].clip(0.01, 0.95)
    md_low = np.r_[g.normal(0.8e-3, 5e-5, n), g.normal(0.7e-3, 5e-5, n), g.normal(2.0e-3, 2e-4, 2 * n)]
    fa_low = fa + 0.02                                                           # the low shell's FA: another array
    value, cortex = P.auto_stopping_fa(fa, md_low, fa_low)
    assert abs(cortex - 0.13) <= 0.006 and value == round(1.15 * cortex, 4)    # the tracked shell's FA, not fa_low's
    assert P.auto_stopping_fa(fa, np.full_like(md_low, 0.8e-3), fa_low)[1] < 0.11   # no MD contrast: the fluid pulls it
    with pytest.raises(ValueError, match="too little white matter"):
        P.auto_stopping_fa(fa[:n], md_low[:n], fa_low[:n])
    with pytest.raises(ValueError, match="too little tissue"):
        P.auto_stopping_fa(fa[n:n + 1500], md_low[n:n + 1500], fa_low[n:n + 1500])   # white matter alone
    with pytest.raises(ValueError, match="expected of a brain"):                # the low hump not a cortex's
        P.auto_stopping_fa(np.r_[fa[n:2 * n], g.normal(0.27, 0.02, n)], md_low[:2 * n], np.r_[fa[n:2 * n], fa[n:2 * n]])
    from tractline import config
    impl = config.implementation("track", "ptt")
    assert impl.resolve({"stopping": "fa", "stopping_fa": "auto"})["stopping_fa"] == "auto"
    with pytest.raises(ValueError, match="not a finite number"):
        impl.resolve({"stopping_fa": "automatic"})


def test_auto_shell():
    """auto_shell: the lowest shell with enough directions and b, a warning above 1600, a refusal with none."""
    b = np.r_[np.zeros(4), np.full(3, 200.0), np.full(30, 1000.0), np.full(30, 2000.0)]
    assert P.auto_shell(b) == 1000.0
    with pytest.warns(UserWarning, match="barely differ"):
        assert P.auto_shell(np.r_[np.zeros(4), np.full(60, 3000.0)]) == 3000.0
    with pytest.raises(ValueError, match="needs a shell"):
        P.auto_shell(np.r_[np.zeros(4), np.full(6, 1000.0)])


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_auto_stopping_fa_runs(device, monkeypatch):
    """stopping_fa auto through execute. The band is not a brain, so auto refuses it; with the plausible range opened
    it sets itself, recorded in the stats and the runtime, and tracks."""
    s = subject()
    with pytest.raises(ValueError, match="give stopping_fa a number"):
        P.execute(s, cfg(stopping="fa", stopping_fa="auto"), P.Timer(), device=device, labeler=stand_in)
    monkeypatch.setitem(P.AUTO_FA, "plausible", (0.0, 1.0))
    r = P.execute(s, cfg(stopping="fa", stopping_fa="auto"), P.Timer(), device=device, labeler=stand_in)
    v = r.tractogram.stats.get("stopping_fa (auto)")
    rec = r.runtime["ptt stopping_fa (auto)"]
    assert v is not None and rec["stopping_fa (auto)"] == v == round(1.15 * rec["cortex FA estimate"], 4)
    assert rec["auto shell"] == P.auto_shell(s.bval)
