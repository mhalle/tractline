"""Which hard decision turns a small input change into a different path (docs/robustness-plan.md, step 2: diagnose).

    uv run --group dipy --group rankfield bench/robust_ukf_diagnose.py run --arm epsilon|fieldonly [--runs 1 2 ..] [--scale 1]
    uv run --group dipy --group rankfield bench/robust_ukf_diagnose.py analyze [--arms epsilon fieldonly]

The tracker is deterministic given its input, yet a 1e-6 relative nudge of the corrected DWI (ensemble_stages' epsilon
arm) moves 3.4 % of streamlines by more than 5 mm, and a field re-estimated from noisy b0s (fieldonly, about 0.1 mm of
displacement) 34 % (NOTES 2026-10-17). Here a sample of cIIIs04 A's seeds is tracked again on Metal, as the records
were, with ukf.track's observe hook (off by default; the fibers are the same with it) recording every step of both
half-fibers: the new position, the stop decision and which conditions fired (outside the mask, mean signal below
stopping_threshold 0.06, FA below stopping_fa 0.08, the step limit), both tensor switches (sw: the second tensor is
closer to the previous direction; sw2: within 20 degrees and the smaller FA <= 0.2), FA, the mean signal and the
filter's state after the swaps (both tensors: the second's direction and FA, the angle between them).

run: per arm and run r, three tractographies of the sampled seeds (select, in each input's own seed list):
  A     A's corrected DWI (A's field, stages/A_field_hz.npy), its own mask: the reference;
  P     the perturbed input with its own mask, as the pipeline would track it:
          epsilon    A's corrected DWI x (1 + EPS u), u uniform in [-1, 1], seeded by 10,000 + r (ensemble_stages' arm,
                     rebuilt exactly);
          fieldonly  A's DWI as acquired corrected with replicate r's field, stages/fieldonly/r???_field_hz.npy. Those
                     are stored as float16 (0.125 Hz steps at 158 Hz), so P is not the stored run's input: its paths
                     are compared with the stored record's (the reproduction check);
  Pfix  the perturbed input with A's mask held fixed (the seeds and the inside test A's): the mask's part.
The sample is stratified on the stored records (A.npz against stages/<arm>/r???.npz, MDF over A's seeds): STRATA
seeds per run (times --scale) from paths more than 5 mm apart, (0.1, 5] mm, at most 0.1 mm, and A's seeds the run's
record lacks; seeded by r. Records in DATA/ensemble/cIIIs04/robustness/ukf/<arm>/r???.npz (never overwritten; a run
whose file exists is skipped), the steps of every half-fiber and the fibers.

analyze: per pair (A against P, A against Pfix), MDF of the fibers (20 points); a pair diverges when MDF > 5 mm, or
when one side's fiber is under 10 points (dropped by the tracker) and the other's is not. For each diverging pair, the
half-fiber whose path moved more (mean distance of the two halves at 20 points, from the seed) is classified by its
first event. The halves separate at the first step where their positions are more than SEP_MM apart, or where one
stops and the other goes on. The switch flags alone cannot say whether a switch mattered: both tensors start as the
seed tensor, so the flags differ within the first steps while the tensors are still one (79 % of pairs that stay
together), and once one side has swapped two equal tensors the states' tensor order differs for good. So the
directions decide: a turn is a step where the two sides' step directions move JUMP_DEG further apart than at the
step before. The first event at or before the separation:
  switch    a turn where one side's step lies along the other side's second tensor (within JUMP_DEG): the sides
            followed different tensors; labeled by which flags differ at that step;
  turn      a turn with no such match: the tensor estimate itself jumped;
  stop      no turn; one side stops and the other does not, by its conditions: "outside the mask, the other input's
            mask includes the point" is the mask itself; "outside the mask" otherwise (both masks agree there: a
            position difference across a mask voxel's edge);
  seed      the initial directions differ by more than SEED_DEG, a turn at the first step, or a separation within
            SEED_STEPS steps of the seed;
  drift     none of these: no >= 5-degree event before the positions part. What the rules leave, not a mechanism:
            a swap between tensors less than 5 degrees apart is invisible to them (the switch flags beyond
            FLAG_FROM_MM are reported beside it).
The growth profile of each classified half: the first step the halves are LEVELS_MM apart, and the first step the
voxel the tracker reads (round(position): the interpolation's 3 x 3 x 3 neighborhood and the mask) differs.
Reported, weighted back to each run's population by stratum unless a key says unweighted: the classes, where the event
is (arc length from the seed, the steps from it to the separation, how far apart the halves were just before it),
whether it is at a crossing (A's second tensor FA > CROSS_FA and the tensors more than CROSS_DEG apart: two tensors
the tracker itself treats as distinct), against the steps before it; the base rate of the same events in pairs that
stay together; the margins of the threshold stops; the mask's share (A's mask held fixed); and why A's seeds the run's
record lacks are lost. Writes results/robust_ukf_diagnose.json.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, torch
from tractline import pipeline as P
from tractline import ensemble as E
from tractline import susceptibility as S
from tractline import ukf as U
from tractline.data import DATA
from tractline.prep import prepare
from tractline.labelers.base import lengths, MIN_LENGTH_MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS                          # noqa: E402
from ensemble_stages import PAIR, EPS, where as stages     # noqa: E402

DEVICE = "mps"
OUT = DATA / "ensemble" / "cIIIs04" / "robustness" / "ukf"
MIN_FREE_GB = 20.0
ARMS = ("epsilon", "fieldonly")
#: seeds per run and stratum (the stored records' MDF of the run's path to A's), times --scale
STRATA = (("> 5 mm", 150), ("(0.1, 5] mm", 50), ("<= 0.1 mm", 50), ("unmatched", 25))
DIVERGED_MM = 5.0
SEP_MM = 0.5                     # the halves have separated where their positions at one step are this far apart
FLAG_FROM_MM = 3.0               # the switch flags compared beyond this arc length: before it the two tensors are still the
                                 # seed tensor, and a swap of the two is harmless
JUMP_DEG = 5.0                   # the two sides' step directions moving this much further apart in one step: a turn, not drift
LEVELS_MM = (1e-4, 1e-3, 1e-2, 0.1, SEP_MM)   # the growth profile: the first step the halves are this far apart
SEED_DEG, SEED_STEPS = 1.0, 3
CROSS_FA, CROSS_DEG = 0.2, 20.0  # the tracker's own constants for two distinct tensors (sw2)
STEP_MM, MAX_STEPS = 0.3, 834    # ukf.track's defaults: step_length, ceil(max_half_length / step_length)
STOP_FA, STOP_SIG = 0.08, 0.06
# flags per step (observe): bits
STOP, SW, SW2, INSIDE, SIGLOW, FALOW = 1, 2, 4, 8, 16, 32
QUANTUM = 0.05                   # the compact records' geometry (mm)


def free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


# ----------------------------------------------------------------------------------------------- tracking

def observed(D, keys):
    """The seeds with these keys (those in D's seed list), tracked on Metal with every step observed. Returns a dict:
    keys tracked (in D's seed order), their seed points (k, j, i), initial directions and seed tensors' FA, the fibers (RAS mm, empty when
    the tracker drops one under 10 points), and per step of every half-fiber (2 j forward, 2 j + 1 backward, as
    ukf.track pairs them): half, step, position (k, j, i), flags, fa, mean signal and the filter's state after the
    step's swaps (float32; the step's own direction is the positions' difference)."""
    pts, _, _, e1, fa0 = U.seeds(D)
    allkeys = E.seed_keys(np.rint(pts.cpu().numpy() - U.SRAND0_OFFSET).astype(np.int64))
    pos = {int(k): i for i, k in enumerate(allkeys)}
    sel = np.array(sorted(pos[int(k)] for k in keys if int(k) in pos), np.int64)
    rows = []

    def observe(step, index, x, state, direction, stop, info):
        fl = (stop.int() * STOP | info["swap"].int() * SW | info["swap2"].int() * SW2 | info["inside"].int() * INSIDE
              | (info["mean_signal"] < STOP_SIG).int() * SIGLOW | (info["fa"] < STOP_FA).int() * FALOW)
        v = torch.cat([x, info["fa"][:, None], info["mean_signal"][:, None], fl.float()[:, None], state], 1)
        rows.append((step, index.cpu().numpy(), v.cpu().numpy()))

    fibers, st = U.track(D, backend="metal", batch=P.GPU_BATCH, select=sel, observe=observe)
    half = np.concatenate([i for _, i, _ in rows]).astype(np.int32)
    step = np.concatenate([np.full(len(i), s, np.int16) for s, i, _ in rows])
    V = np.concatenate([v for _, _, v in rows])
    o = np.lexsort((step, half))
    fib = dict(zip(st["seed_index"], fibers))
    F = [np.asarray(fib.get(int(k), np.zeros((0, 3))), np.float32) for k in sel]
    return {"keys": allkeys[sel], "seed": pts[sel].cpu().numpy().astype(np.float32), "e1": e1[sel].cpu().numpy().astype(np.float32),
            "fa0": fa0[sel].cpu().numpy().astype(np.float32),
            "half": half[o], "step": step[o], "pos": V[o, :3], "fa": V[o, 3], "ms": V[o, 4], "flags": V[o, 5].astype(np.uint8),
            "state": V[o, 6:16],
            "flen": np.array([len(f) for f in F], np.int32), "fib": np.concatenate(F + [np.zeros((0, 3), np.float32)])}


def sample(recA, rec, r, scale):
    """A's seed keys to track for run r, their strata (index into STRATA), each stratum's population, and the stored
    MDF of each sampled seed (NaN: unmatched)."""
    SA = E.StreamlineEnsemble(recA)
    ri, qi = SA.match(rec)
    m = np.full(len(recA.keys), np.nan)
    m[ri] = E.mdf(recA.paths[ri], rec.paths[qi])
    groups = [m > DIVERGED_MM, (m > 0.1) & (m <= DIVERGED_MM), m <= 0.1, np.isnan(m)]
    rng = np.random.default_rng(r)
    keys, strata, mdf = [], [], []
    for s, ((_, n), g) in enumerate(zip(STRATA, groups)):
        idx = np.flatnonzero(g)
        pick = np.sort(rng.choice(idx, min(len(idx), int(round(n * scale))), replace=False))
        keys.append(recA.keys[pick]); strata.append(np.full(len(pick), s, np.int8)); mdf.append(m[pick])
    return np.concatenate(keys), np.concatenate(strata), np.array([int(g.sum()) for g in groups]), np.concatenate(mdf)


def fiber_dev(fibers_by_key, rec, exact):
    """The re-tracked fibers against a stored record, over the keys both have: exact (the same input) - the largest
    vertex deviation (the records hold 0.05 mm, so at most 0.025 when the paths are the same) and point-count
    mismatches; else the MDF quantiles and how many cross DIVERGED_MM."""
    pos = {int(k): i for i, k in enumerate(rec.keys)}
    pairs = [(f, rec.fibers[pos[k]]) for k, f in fibers_by_key.items() if k in pos and len(f)]
    if exact:
        n_ok = [len(a) == len(b) for a, b in pairs]
        dev = [float(np.abs(a - b).max()) for (a, b), ok in zip(pairs, n_ok) if ok]
        return {"compared": len(pairs), "point counts differ": int(len(pairs) - sum(n_ok)),
                "max vertex deviation mm": round(max(dev), 4) if dev else None}
    m = E.mdf(E.resample([a for a, _ in pairs]), E.resample([b for _, b in pairs]))
    return {"compared": len(pairs), "MDF to the stored run: quartiles mm": [round(float(q), 3) for q in np.quantile(m, [0.25, 0.5, 0.75])],
            "share above 5 mm": round(float((m > DIVERGED_MM).mean()), 4)}


def run(args):
    from _masivar import load
    subject, (ses, scan), _, shells, shell = PAIRS[PAIR]
    d, sd = stages()
    out = OUT / args.arm
    out.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    A = load(subject, ses, scan, shells)
    apply = lambda dwi, h: S.apply(dwi, h, A.pe_axis, A.pe_sign, A.readout_s)
    dA = apply(A.dwi, np.load(sd / "A_field_hz.npy").astype(np.float64))
    tA = prepare(dA, A.affine, A.bval, A.bvec, shell=shell, device=DEVICE)
    DA = U.from_arrays(tA.dwi, tA.header, tA.mask)
    mA_path = OUT / "A_mask.npy"
    if mA_path.exists():
        assert np.array_equal(np.load(mA_path), tA.mask), f"{mA_path}: not this A's mask"
    else:
        np.save(mA_path, tA.mask)
    recA = E.RunRecord.load(d / "A.npz", logp=False, fibers=True)
    for r in args.runs:
        name = out / f"r{r:03d}.npz"
        if name.exists():
            print(args.arm, name.name, "exists, skipped", flush=True)
            continue
        if free_gb(OUT) < MIN_FREE_GB:
            raise SystemExit(f"stopping: {free_gb(OUT):.1f} GB free, below {MIN_FREE_GB} GB")
        t0, sec = time.time(), {}
        rec = E.RunRecord.load(sd / args.arm / f"r{r:03d}.npz", logp=False, fibers=True)
        keys, strata, population, mdf_stored = sample(recA, rec, r, args.scale)
        t = time.time()
        if args.arm == "epsilon":
            u = np.random.default_rng(10_000 + r).uniform(-1, 1, dA.shape).astype(np.float32)
            dP = (dA * (1 + np.float32(EPS) * u)).astype(np.float32)
            del u
        else:
            dP = apply(A.dwi, np.load(sd / "fieldonly" / f"r{r:03d}_field_hz.npy").astype(np.float64))
        tP = prepare(dP, A.affine, A.bval, A.bvec, shell=shell, device=DEVICE)
        del dP
        sec["input"] = round(time.time() - t, 1)
        tracks = {}
        for tag, D in (("A", DA), ("P", U.from_arrays(tP.dwi, tP.header, tP.mask)), ("Pfix", U.from_arrays(tP.dwi, tP.header, tA.mask))):
            t = time.time()
            tracks[tag] = observed(D, keys)
            sec[tag] = round(time.time() - t, 1)
            del D
            P.release_memory(DEVICE)
        byk = lambda T: {int(k): T["fib"][o - n:o] for k, n, o in zip(T["keys"], T["flen"], np.cumsum(T["flen"]))}
        check = {"A against A.npz": fiber_dev(byk(tracks["A"]), recA, True),
                 f"P against the stored {args.arm} r{r:03d}": fiber_dev(byk(tracks["P"]), rec, args.arm == "epsilon")}
        payload = dict(keys=keys, strata=strata, population=population, mdf_stored=mdf_stored, mask_P=np.packbits(tP.mask),
                       voxel=DA["voxel"].cpu().numpy(), i2r=DA["i2r"])
        for tag, T in tracks.items():
            payload.update({f"{tag}_{k}": v for k, v in T.items()})
        tmp = name.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        if name.exists():                                                   # another process wrote it meanwhile
            tmp.unlink()
            raise SystemExit(f"{name} appeared while this run was tracking; not overwritten")
        os.replace(tmp, name)
        log[f"{args.arm}/r{r:03d}"] = {"seeds sampled": int(len(keys)), "by stratum": np.bincount(strata, minlength=len(STRATA)).tolist(),
                                       "population by stratum": population.tolist(), "tracked (A / P / Pfix)": [int(len(tracks[k]["keys"])) for k in tracks],
                                       "mask voxels: A / P / differing": [int(tA.mask.sum()), int(tP.mask.sum()), int((tA.mask != tP.mask).sum())],
                                       "steps observed (A / P / Pfix)": [int(len(tracks[k]["step"])) for k in tracks],
                                       "reproduction": check, "seconds": sec, "total_s": round(time.time() - t0, 1),
                                       "MB": round(name.stat().st_size / 1e6, 2)}
        log_path.write_text(json.dumps(log, indent=1))
        print(args.arm, name.name, json.dumps(log[f"{args.arm}/r{r:03d}"]), flush=True)
        del tracks, tP
        P.release_memory(DEVICE)


# ----------------------------------------------------------------------------------------------- analysis

class Track:
    """One tractography of a run file (A, P or Pfix): per seed key its row, per half-fiber its steps."""
    def __init__(self, z, tag):
        g = lambda k: z[f"{tag}_{k}"]
        self.keys, self.seed, self.e1, self.fa0 = g("keys"), g("seed"), g("e1"), g("fa0")
        self.row = {int(k): i for i, k in enumerate(self.keys)}
        self.half, self.step, self.pos, self.flags = g("half"), g("step"), g("pos"), g("flags")
        self.fa, self.ms, self.state = g("fa"), g("ms"), g("state")
        # the second tensor: its direction (unit, (i, j, k) mm as the step's m1), FA, and angle to the step's direction
        m2 = self.state[:, 5:8] / np.linalg.norm(self.state[:, 5:8], axis=1, keepdims=True)
        self.m2 = m2
        self.fa2 = U.l2fa(*(torch.from_numpy(np.maximum(self.state[:, c], U.LMIN)) for c in (8, 9, 9))).numpy()
        self.ang = np.full(len(self.step), np.nan, np.float32)               # filled per half-fiber (side)
        self.off = np.searchsorted(self.half, np.arange(2 * len(self.keys) + 1))
        flen = g("flen")
        ends = np.cumsum(flen)
        fib = g("fib")
        self.fibers = [fib[e - n:e] for n, e in zip(flen, ends)]

    def steps(self, j, h):
        """Half-fiber h (0 forward, 1 backward) of seed row j: a slice of the step arrays, steps 1..n (n stops)."""
        a, b = self.off[2 * j + h], self.off[2 * j + h + 1]
        assert np.array_equal(self.step[a:b], np.arange(1, b - a + 1)), "steps not consecutive"
        assert self.flags[b - 1] & STOP and not (self.flags[a:b - 1] & STOP).any(), "a half-fiber stops once, at its last step"
        return slice(a, b)


def stop_ok(T):
    """Every observed stop decision is the OR of the recorded conditions (the kernel's own rule)."""
    f = T.flags
    cond = ((f & INSIDE) == 0) | ((f & SIGLOW) != 0) | ((f & FALOW) != 0) | (T.step > MAX_STEPS)
    return int((cond != ((f & STOP) != 0)).sum())


def conditions(f, step):
    c = [n for n, on in (("outside the mask", not f & INSIDE), ("signal < 0.06", f & SIGLOW), ("FA < 0.08", f & FALOW),
                         ("step limit", step > MAX_STEPS)) if on]
    return " + ".join(c)


def mask_at(mask, p):
    """The mask (i, j, k) at a position (k, j, i), as the tracker reads it (C round, outside the volume: outside)."""
    r = (np.sign(p) * np.floor(np.abs(p) + 0.5)).astype(np.int64)[::-1]
    return bool((r >= 0).all() and (r < mask.shape).all() and mask[tuple(r)])


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def deg(a, b, axial=False):
    c = (a * b).sum(-1)
    return np.degrees(np.arccos(np.clip(np.abs(c) if axial else c, -1, 1)))


def side(T, j, h, vox):
    """A half-fiber's steps 1..n: positions (mm, scaled index axes), each step's direction (the positions'
    difference, the seed first: the tensor it followed), the second tensor's direction, the voxel the next step reads."""
    sl = T.steps(j, h)
    p = np.concatenate([T.seed[j][None], T.pos[sl]])
    pm = p * vox
    r = (np.sign(p) * np.floor(np.abs(p) + 0.5)).astype(np.int64)
    d = unit(np.diff(pm, axis=0))
    m2 = T.m2[sl][:, ::-1]                                                  # (i, j, k) -> (k, j, i), as the positions
    T.ang[sl] = deg(d, m2, True)
    return sl, pm[1:], d, m2, r[:-1]


def first_event(A, ja, ha, B, jb, hb, vox, mask_a, mask_b, seed_deg):
    """The event at which two half-fibers part (module docstring). Returns a dict, or None when they never do."""
    sa, pa, da, ma, ra = side(A, ja, ha, vox)
    sb, pb, db, mb, rb = side(B, jb, hb, vox)
    na, nb = len(pa), len(pb)
    K = min(na, nb)
    dist = np.linalg.norm(pa[:K] - pb[:K], axis=1)
    turn = deg(da[:K], db[:K])
    far = np.flatnonzero(dist > SEP_MM)
    k_pos = int(far[0]) if len(far) else None                                # 0-based step index
    k_stop = K - 1 if na != nb else None
    cands = [k for k in (k_pos, k_stop) if k is not None]
    if not cands:
        return None
    k_sep = min(cands)
    jumps = np.flatnonzero(np.diff(np.r_[0.0, turn[:k_sep + 1]]) > JUMP_DEG)
    k_jump = int(jumps[0]) if len(jumps) else None
    vx = np.flatnonzero((ra[:k_sep + 1] != rb[:k_sep + 1]).any(1))
    first = lambda c: int(np.flatnonzero(c)[0]) + 1 if c.any() else None
    ev = {"separation step": k_sep + 1, "separated by": "stop" if k_sep == k_stop else "position",
          "growth: first step apart by " + " / ".join(f"{v:g}" for v in LEVELS_MM) + " mm": [first(dist[:k_sep + 1] > v) for v in LEVELS_MM],
          "growth: first step the voxel read differs": int(vx[0]) + 1 if len(vx) else None}
    if k_jump is not None and k_jump > 0:
        e = k_jump
        # each side followed one of its two tensors; crossed: one side's step lies along the other's second tensor
        crossed = min(deg(da[e], mb[e], True), deg(ma[e], db[e], True)) < JUMP_DEG
        f = (int(A.flags[sa][e]) ^ int(B.flags[sb][e])) & (SW | SW2)
        ev["class"] = ("switch: " + {0: "flags equal (the states' tensor order differs)", SW: "sw", SW2: "sw2", SW | SW2: "sw and sw2"}[f]
                       if crossed else "turn: the tensor estimate itself (no switch)")
        ev["turn deg"] = float(turn[e])
        if k_stop is not None and e == k_stop:
            ev["at the stopping step"] = True
    elif k_sep == k_stop:
        e = k_stop
        T, s, m_other, other, so = (A, sa, mask_b, B, sb) if na < nb else (B, sb, mask_a, A, sa)
        f = int(T.flags[s][e])
        c = conditions(f, e + 1)
        if c == "outside the mask":
            c = "the mask (the other input's includes the point)" if mask_at(m_other, T.pos[s][e]) else "outside the mask (both masks agree)"
        ev["class"] = "stop: " + c
        ev["stopped side"] = "A" if T is A else "perturbed"
        # the other side's value at the same step, against the threshold that stopped this one
        if f & FALOW:
            ev["FA: stopped / other"] = [float(T.fa[s][e]), float(other.fa[so][e])]
        if f & SIGLOW:
            ev["signal: stopped / other"] = [float(T.ms[s][e]), float(other.ms[so][e])]
    else:
        e = k_sep if k_jump is None else 0
        ev["class"] = "seed" if (k_jump == 0 or seed_deg > SEED_DEG or k_sep < SEED_STEPS) else "drift"
    ev["voxel read differs before the event"] = bool(len(vx) and vx[0] <= e)
    ev.update({"event step": e + 1, "arc length from the seed mm": round((e + 1) * STEP_MM, 2), "steps to separation": k_sep - e,
               "distance just before mm": float(dist[e - 1]) if e > 0 else 0.0,
               "A at the event: fa, fa2, angle": [float(A.fa[sa][e]), float(A.fa2[sa][e]), float(A.ang[sa][e])],
               "A before the event: crossing share": float(crossing(A.fa2[sa][:e], A.ang[sa][:e]).mean()) if e > 0 else None})
    return ev


def crossing(fa2, ang):
    return (fa2 > CROSS_FA) & (ang > CROSS_DEG)


def pair(A, B, k, vox, mask_a, mask_b):
    """A seed's two tractographies compared: MDF, divergence, and (diverging) the first event of the half that moved more;
    and the base rates: whether a turn (the sides' directions JUMP_DEG further apart in one step) happens anywhere while
    the halves are within SEP_MM, and whether they stop at different steps."""
    ja, jb = A.row[k], B.row[k]
    fa_, fb_ = A.fibers[ja], B.fibers[jb]
    if len(fa_) and len(fb_):
        m = float(E.mdf(E.resample([fa_]), E.resample([fb_]))[0])
        div = m > DIVERGED_MM
    else:
        m, div = None, bool(len(fa_)) != bool(len(fb_))
    dot = float(A.e1[ja] @ B.e1[jb])
    flip = dot < 0                                                         # the seed's e1 changed sign: the halves swap
    seed_deg = float(np.degrees(np.arccos(min(abs(dot), 1.0))))
    out = {"mdf": m, "diverged": div, "e1 flipped": flip, "seed angle deg": seed_deg, "seed FA": float(A.fa0[ja]),
           "seed FA, other": float(B.fa0[jb])}
    evs, moved, turned, stop_diff, flags = [], [], False, False, []
    for h in (0, 1):
        hb = 1 - h if flip else h
        _, pa, da, _, _ = side(A, ja, h, vox)
        _, pb, db, _, _ = side(B, jb, hb, vox)
        ra, rb = E.resample([np.concatenate([A.seed[ja][None] * vox, pa[:-1]])]), E.resample([np.concatenate([B.seed[jb][None] * vox, pb[:-1]])])
        moved.append(float(np.linalg.norm(ra - rb, axis=2).mean()))
        evs.append(first_event(A, ja, h, B, jb, hb, vox, mask_a, mask_b, seed_deg))
        K = min(len(pa), len(pb))
        near = np.linalg.norm(pa[:K] - pb[:K], axis=1) <= SEP_MM
        turned |= bool(((np.diff(np.r_[0.0, deg(da[:K], db[:K])]) > JUMP_DEG) & near).any())
        stop_diff |= len(pa) != len(pb)
        # the switch flags differing beyond FLAG_FROM_MM, up to the separation: swaps the 5-degree rule cannot see
        sa, sb = A.steps(ja, h), B.steps(jb, hb)
        ks = evs[-1]["separation step"] - 1 if evs[-1] else K - 1
        e = np.arange(K)
        fd = np.flatnonzero(((((A.flags[sa][:K] ^ B.flags[sb][:K]) & (SW | SW2)) != 0) & ((e + 1) * STEP_MM > FLAG_FROM_MM) & (e <= ks)))
        flags.append({"differ": bool(len(fd)), "angle deg": A.ang[sa][fd].tolist(), "last to separation": int(ks - fd[-1]) if len(fd) else None})
    h = int(np.argmax(moved)) if (evs[0] is None) == (evs[1] is None) else int(evs[0] is None)   # a half with an event
    out.update({"half moved mm": moved, "event": evs[h], "other half's event": evs[1 - h],
                "a turn while together": turned, "stop step differs": stop_diff,
                "flags after 3 mm: the classified half": flags[h],
                "flags after 3 mm: either half": any(f["differ"] for f in flags)})
    return out


def q(x, qs=(0.25, 0.5, 0.75), w=None):
    """Quantiles, weighted when w is given (each value its pair's weight; None values dropped)."""
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    keep = [i for i, v in enumerate(x) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not keep:
        return None
    x, w = np.asarray([x[i] for i in keep], float), w[keep]
    o = np.argsort(x)
    c = (np.cumsum(w[o]) - 0.5 * w[o]) / w.sum()
    return [round(float(np.interp(qq, c, x[o])), 3) for qq in qs]


def wmean(x, w):
    """The weighted mean of x (booleans or values; None dropped), or None."""
    keep = [i for i, v in enumerate(x) if v is not None]
    if not keep:
        return None
    w = np.asarray(w, float)[keep]
    return round(float((np.asarray([x[i] for i in keep], float) * w).sum() / w.sum()), 4)


def summarize(rows, label):
    """Diverging pairs: classes, where, crossings; the base rates in pairs that stay together. Every quantile and share
    is weighted to the population (each pair its stratum's weight) unless its key says unweighted."""
    div = [r for r in rows if r["diverged"] and r["event"] is not None]
    stay = [r for r in rows if not r["diverged"] and r["mdf"] is not None]
    out = {"pairs": len(rows), "diverged": sum(r["diverged"] for r in rows), "diverged with an event": len(div),
           "diverged, no event in either half": sum(r["diverged"] and r["event"] is None for r in rows)}
    cls = sorted({r["event"]["class"] for r in div})
    w = np.array([r["weight"] for r in div])
    ws = np.array([r["weight"] for r in stay])
    out["classes (unweighted)"] = {c: round(sum(r["event"]["class"] == c for r in div) / max(len(div), 1), 4) for c in cls}
    out["classes (weighted to the population)"] = {c: round(float(w[[r["event"]["class"] == c for r in div]].sum() / max(w.sum(), 1e-12)), 4) for c in cls}
    out["classes by family (weighted)"] = {f: round(float(w[[r["event"]["class"].startswith(f) for r in div]].sum() / max(w.sum(), 1e-12)), 4)
                                           for f in ("switch", "turn", "stop", "seed", "drift")}
    E_ = [r["event"] for r in div]
    of = lambda c: [i for i, e in enumerate(E_) if e["class"] == c]
    out["event: arc length from the seed mm (quartiles)"] = q([e["arc length from the seed mm"] for e in E_], w=w)
    out["event: steps to separation (quartiles)"] = q([e["steps to separation"] for e in E_], w=w)
    out["event: distance just before mm (quartiles)"] = q([e["distance just before mm"] for e in E_], (0.5, 0.9, 0.99), w=w)
    out["by class: arc length mm (quartiles), steps to separation (median)"] = {
        c: [q([E_[i]["arc length from the seed mm"] for i in of(c)], w=w[of(c)]), q([E_[i]["steps to separation"] for i in of(c)], (0.5,), w=w[of(c)])]
        for c in cls}
    at = [bool(crossing(np.float32(e["A at the event: fa, fa2, angle"][1]), np.float32(e["A at the event: fa, fa2, angle"][2]))) for e in E_]
    out["crossing share: at the event / the steps before it (mean over pairs)"] = [wmean(at, w), wmean([e["A before the event: crossing share"] for e in E_], w)]
    out["crossing share at the event, by class"] = {c: wmean([at[i] for i in of(c)], w[of(c)]) for c in cls}
    out["A at the event: fa / fa2 / angle (medians)"] = [q([e["A at the event: fa, fa2, angle"][k] for e in E_], (0.5,), w=w)[0] for k in range(3)] if E_ else None
    for key in ("FA: stopped / other", "signal: stopped / other"):
        idx = [i for i, e in enumerate(E_) if key in e]
        if idx:
            out[f"threshold stops, {key.split(':')[0]}: the other side's value (quartiles)"] = q([E_[i][key][1] for i in idx], w=w[idx])
    out["voxel read differs before the event, by class"] = {c: wmean([E_[i]["voxel read differs before the event"] for i in of(c)], w[of(c)]) for c in cls}
    gk = next((k for k in (E_[0] if E_ else {}) if k.startswith("growth: first step apart")), None)
    if gk:
        G = np.array([[np.nan if v is None else v for v in e[gk]] for e in E_], float) * STEP_MM
        out["growth: arc length mm where the halves are first " + gk.split("by ")[1] + " apart (quartiles)"] = [q(list(c), w=w) for c in G.T]
        vx = np.array([np.nan if e["growth: first step the voxel read differs"] is None else e["growth: first step the voxel read differs"] for e in E_]) * STEP_MM
        out["growth: arc length mm where the voxel read first differs (quartiles)"] = q(list(vx), w=w)
        out["growth: the voxel read differs before the halves are 1e-3 mm apart"] = wmean(list(vx < G[:, 1]), w)
    idx = [i for i, e in enumerate(E_) if "turn deg" in e]
    out["turns: deg (quartiles)"] = q([E_[i]["turn deg"] for i in idx], w=w[idx])
    out["the other half separated too"] = wmean([r["other half's event"] is not None for r in div], w)
    out["seed FA (A's seed tensor; both tensors start as it): quartiles, diverged / stayed together"] = [
        q([r["seed FA"] for r in div], w=w), q([r["seed FA"] for r in stay], w=ws)]
    out["e1 flipped, pairs (unweighted counts): diverged / stayed"] = [sum(r["e1 flipped"] for r in div), sum(r["e1 flipped"] for r in stay)]
    out["stayed together (MDF <= 5 mm): pairs, a turn while together, stop step differs"] = [
        len(stay), wmean([r["a turn while together"] for r in stay], ws), wmean([r["stop step differs"] for r in stay], ws)]
    out["diverged: a turn while together, stop step differs"] = [wmean([r["a turn while together"] for r in div], w),
                                                                 wmean([r["stop step differs"] for r in div], w)]
    # the switch flags beyond FLAG_FROM_MM: drift pairs (the classified half, up to its separation) against pairs that
    # stay together (either half): swaps between tensors too close for the 5-degree rule
    dr = of("drift")
    fl = [div[i]["flags after 3 mm: the classified half"] for i in dr]
    angles = [(a, w[i]) for i, f in zip(dr, fl) for a in f["angle deg"]]
    out[f"switch flags differing beyond {FLAG_FROM_MM:g} mm"] = {
        "drift pairs: weighted / unweighted": [wmean([f["differ"] for f in fl], w[dr]), round(float(np.mean([f["differ"] for f in fl])), 4) if fl else None],
        "stayed together: weighted / unweighted": [wmean([r["flags after 3 mm: either half"] for r in stay], ws),
                                                    round(float(np.mean([r["flags after 3 mm: either half"] for r in stay])), 4) if stay else None],
        "drift pairs: angle between A's tensors at those steps, deg (quartiles over steps, weighted by pair)":
            q([a for a, _ in angles], w=[x for _, x in angles]),
        "drift pairs: steps from the last such difference to the separation (quartiles)":
            q([f["last to separation"] for f in fl], w=w[dr])}
    print(label, json.dumps({k: out[k] for k in ("pairs", "diverged", "classes (weighted to the population)")}), flush=True)
    return out


# ----------------------------------------------------------------------------------------------- drift

POS_WINDOW = (1e-4, 0.3)         # mm: the halves' distance is read here, above the float32 position quantum (~1e-5 mm)
                                 # and before they part
STATE_WINDOW = (1e-6, 1e-2)      # the state difference (radians, relative eigenvalues): above float32's rounding
GROWTH_BINS = {"A's FA": ("fa", (0, 0.15, 0.2, 0.3, 0.45, 1.01)),
               "angle between A's tensors (deg)": ("ang", (0, 2, 10, 20, 45, 90.01)),
               "A's second tensor's FA": ("fa2", (0, 0.15, 0.2, 0.3, 0.45, 1.01)),
               "A's mean signal (quintiles of the steps)": ("ms", None),
               "arc length from the seed (mm)": ("arc", (0, 5, 10, 20, 40, 80, 251))}
SEED_FA_BINS = (0.1, 0.15, 0.2, 0.3, 0.45, 1.01)
EARLY_MM = 40.0                  # the conditions along the path: its first EARLY_MM, where differences grow


def chord(a, b, axial=False):
    """The angle (rad) between unit vectors, from their chord: exact near 0, where arccos is not."""
    c = np.linalg.norm(a - b, axis=-1)
    if axial:
        c = np.minimum(c, np.linalg.norm(a + b, axis=-1))
    return 2 * np.arcsin(np.clip(c / 2, 0, 1))


def state_diff(Sa, Sb):
    """Per step, how far two filter states are apart: the largest of the tensors' direction differences (rad) and
    eigenvalue differences (relative), over the better of the two pairings of the tensors (a swap of two equal tensors
    reorders a state for good)."""
    Sa, Sb = Sa.astype(np.float64), Sb.astype(np.float64)
    u = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    rel = lambda x, y: np.abs(x - y) / np.abs(x)

    def one(p, q):
        a, b = 5 * p, 5 * q
        return np.maximum.reduce([chord(u(Sa[:, a:a + 3]), u(Sb[:, b:b + 3]), True), rel(Sa[:, a + 3], Sb[:, b + 3]),
                                  rel(Sa[:, a + 4], Sb[:, b + 4])])
    return np.minimum(np.maximum(one(0, 0), one(1, 1)), np.maximum(one(0, 1), one(1, 0)))


def growth(A, ja, ha, B, jb, hb, vox):
    """Per step of two half-fibers while both go on (from the second): the growth of their position distance and of
    their state difference over the step, decades per mm (NaN outside POS_WINDOW / STATE_WINDOW), and A's conditions."""
    sa, pa, _, _, _ = side(A, ja, ha, vox)
    sb, pb, _, _, _ = side(B, jb, hb, vox)
    K = min(len(pa), len(pb))
    d = np.linalg.norm(pa[:K] - pb[:K], axis=1)
    sd = state_diff(A.state[sa][:K], B.state[sb][:K])
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = np.where((d[:-1] >= POS_WINDOW[0]) & (d[:-1] <= POS_WINDOW[1]), np.log10(d[1:] / d[:-1]) / STEP_MM, np.nan)
        rs = np.where((sd[:-1] >= STATE_WINDOW[0]) & (sd[:-1] <= STATE_WINDOW[1]), np.log10(sd[1:] / sd[:-1]) / STEP_MM, np.nan)
    rp[~np.isfinite(rp)], rs[~np.isfinite(rs)] = np.nan, np.nan
    return {"pos": rp, "state": rs, "fa": A.fa[sa][1:K], "ang": A.ang[sa][1:K], "fa2": A.fa2[sa][1:K], "ms": A.ms[sa][1:K],
            "arc": np.arange(2, K + 1) * STEP_MM}


def drift_summary(G, rows):
    """G: per half-fiber step arrays (growth) with each pair's weight and divergence; rows: the pairs (pair())."""
    cat = {k: np.concatenate([g[k] for g in G]) for k in G[0] if k not in ("w", "div")}
    n = [len(g["pos"]) for g in G]
    w = np.repeat([g["w"] for g in G], n)
    dv = np.repeat([g["div"] for g in G], n)
    wm = lambda x, m: round(float((w[m] * x[m]).sum() / w[m].sum()), 3) if m.any() else None
    okp, oks = ~np.isnan(cat["pos"]), ~np.isnan(cat["state"])
    out = {"growth, decades per mm (weighted mean over steps): position distance / state difference":
           [wm(cat["pos"], okp), wm(cat["state"], oks)],
           "the same, diverged pairs / pairs that stay together: state difference": [wm(cat["state"], oks & dv), wm(cat["state"], oks & ~dv)]}
    for name, (key, edges) in GROWTH_BINS.items():
        x = cat[key]
        if edges is None:
            edges = np.quantile(x[oks], np.linspace(0, 1, 6))
            edges[-1] += 1e-6
        tab = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = (x >= lo) & (x < hi)
            tab[f"[{lo:.3g}, {hi:.3g})"] = {"steps (state window)": int((b & oks).sum()),
                                            "position": wm(cat["pos"], b & okp), "state": wm(cat["state"], b & oks),
                                            "state: diverged / stayed": [wm(cat["state"], b & oks & dv), wm(cat["state"], b & oks & ~dv)]}
        out[f"growth by {name}"] = tab
    # the conditions along the first EARLY_MM of the path, diverged pairs against pairs that stay together
    early = cat["arc"] <= EARLY_MM
    cond = {"near-isotropic (A's FA < 0.2)": cat["fa"] < 0.2, "one tensor (A's tensors within 2 deg)": cat["ang"] < 2,
            "crossing (second tensor's FA > 0.2, tensors > 20 deg apart)": crossing(cat["fa2"], cat["ang"])}
    out[f"share of steps in the first {EARLY_MM:g} mm: diverged / stayed"] = {
        c: [wm(m.astype(float), early & dv), wm(m.astype(float), early & ~dv)] for c, m in cond.items()}
    # divergence by the seed's FA (both tensors start as the seed tensor)
    fa0 = np.array([r["seed FA"] for r in rows]); d = np.array([r["diverged"] for r in rows]); wr = np.array([r["weight"] for r in rows])
    out["by the seed's FA: share of the population / diverged (weighted)"] = {
        f"[{lo:g}, {hi:g})": [round(float(wr[(fa0 >= lo) & (fa0 < hi)].sum() / wr.sum()), 4),
                              round(float((wr * d)[(fa0 >= lo) & (fa0 < hi)].sum() / max(wr[(fa0 >= lo) & (fa0 < hi)].sum(), 1e-12)), 4)]
        for lo, hi in zip(SEED_FA_BINS[:-1], SEED_FA_BINS[1:])}
    lo = fa0 < 0.2
    out["seeds with FA < 0.2: share of the population / of the diverging (weighted)"] = [
        round(float(wr[lo].sum() / wr.sum()), 4), round(float((wr * d)[lo].sum() / (wr * d).sum()), 4)]
    # what differs at the seed: the class "seed" against drift and the pairs that stay together
    groups = {"seed class": [r for r in rows if r["diverged"] and r["event"] and r["event"]["class"] == "seed"],
              "drift class": [r for r in rows if r["diverged"] and r["event"] and r["event"]["class"] == "drift"],
              "stayed together": [r for r in rows if not r["diverged"] and r["mdf"] is not None]}
    W = lambda rs: [r["weight"] for r in rs]
    out["at the seed"] = {g: {"pairs": len(rs),
                              "initial direction: deg (quartiles)": q([r["seed angle deg"] for r in rs], w=W(rs)),
                              "initial direction more than 1 / 10 / 45 deg apart": [wmean([r["seed angle deg"] > t for r in rs], W(rs))
                                                                                    for t in (1, 10, 45)] if rs else None,
                              "seed FA: A (quartiles)": q([r["seed FA"] for r in rs], w=W(rs)),
                              "seed FA: |A - perturbed| (quartiles)": q([abs(r["seed FA"] - r["seed FA, other"]) for r in rs], w=W(rs)),
                              "e1 changed sign (the halves swap)": wmean([r["e1 flipped"] for r in rs], W(rs))}
                          for g, rs in groups.items()}
    return out


def analyze(args):
    t0 = time.time()
    res = {"pair": PAIR, "eps": EPS, "strata": [s for s, _ in STRATA], "diverged mm": DIVERGED_MM, "separation mm": SEP_MM,
           "seed: deg, steps": [SEED_DEG, SEED_STEPS], "crossing: fa2 >, angle >": [CROSS_FA, CROSS_DEG], "arms": {}}
    mask_A = np.load(OUT / "A_mask.npy")
    log = json.loads((OUT / "runs.json").read_text())
    for arm in args.arms:
        files = sorted((OUT / arm).glob("r???.npz"))
        if not files:
            continue
        rows = {"P": [], "Pfix": []}
        G, unmatched = [], {"sampled": 0, "in P's seed list": 0, "P's streamline under 40 mm or dropped": 0}
        lost, stop_bad, both = {}, 0, {"P only": 0, "Pfix only": 0, "both": 0}
        for path in files:
            z = np.load(path)
            vox = z["voxel"]
            mask_P = np.unpackbits(z["mask_P"])[:mask_A.size].reshape(mask_A.shape).astype(bool)
            T = {tag: Track(z, tag) for tag in ("A", "P", "Pfix")}
            stop_bad += sum(stop_ok(t) for t in T.values())
            strata, keys, pop = z["strata"], z["keys"], z["population"]
            n_s = np.bincount(strata, minlength=len(STRATA))
            div = {}
            for k, s in zip(keys.tolist(), strata.tolist()):
                wgt = pop[s] / n_s[s]
                if k not in T["A"].row:
                    lost["not in A's seed list"] = lost.get("not in A's seed list", 0) + 1
                    continue
                if s == 3:                                                  # A's seeds the stored run lacks: why
                    unmatched["sampled"] += 1
                    if k in T["P"].row:
                        unmatched["in P's seed list"] += 1
                        f = T["P"].fibers[T["P"].row[k]]
                        unmatched["P's streamline under 40 mm or dropped"] += int(not len(f) or lengths([f])[2][0] < MIN_LENGTH_MM)
                for tag, mask_b in (("P", mask_P), ("Pfix", mask_A)):
                    if k not in T[tag].row:
                        if tag == "P":
                            vk = (k >> 2 * E.KEY_BITS, (k >> E.KEY_BITS) & 4095, k & 4095)          # (k, j, i)
                            why = "the seed's voxel left P's mask" if not mask_P[vk[::-1]] else "rejected by the seeding thresholds"
                            lost[f"{STRATA[s][0]}: {why}"] = lost.get(f"{STRATA[s][0]}: {why}", 0) + 1
                        continue
                    r = pair(T["A"], T[tag], k, vox, mask_A, mask_b)
                    r.update({"stratum": s, "weight": float(wgt), "run": path.stem})
                    if tag == "P":
                        ja, jb = T["A"].row[k], T["P"].row[k]
                        for h in (0, 1):
                            g = growth(T["A"], ja, h, T["P"], jb, 1 - h if r["e1 flipped"] else h, vox)
                            g.update({"w": float(wgt), "div": bool(r["diverged"])})
                            G.append(g)
                    rows[tag].append(r)
                    div[(k, tag)] = r["diverged"]
            for k in keys.tolist():
                a, b = div.get((k, "P")), div.get((k, "Pfix"))
                if a or b:
                    both["both" if a and b else "P only" if a else "Pfix only"] += 1
        ra = {"runs": [p.stem for p in files], "stop decisions not the OR of the recorded conditions": stop_bad,
              "lost seeds (not in P's seed list)": lost, "the 'unmatched' stratum": unmatched,
              "mask voxels differing from A's, per run": [log[f"{arm}/{p.stem}"]["mask voxels: A / P / differing"][2] for p in files],
              "A's seeds the stored run lacks, per run (mean)": round(float(np.mean([log[f"{arm}/{p.stem}"]["population by stratum"][3] for p in files])), 1),
              "diverged with its own mask (P) / with A's mask (Pfix)": both,
              "reproduction": {p.stem: log.get(f"{arm}/{p.stem}", {}).get("reproduction") for p in files}}
        for tag in ("P", "Pfix"):
            ra[tag] = summarize(rows[tag], f"{arm} {tag}")
            by = {}
            for s, (name, _) in enumerate(STRATA):
                rs = [r for r in rows[tag] if r["stratum"] == s]
                by[name] = {"pairs": len(rs), "diverged": sum(r["diverged"] for r in rs)}
            ra[tag]["by stratum (sampled on the stored records)"] = by
        ra["drift"] = drift_summary(G, rows["P"])
        res["arms"][arm] = ra
    (HERE / "results/robust_ukf_diagnose.json").write_text(json.dumps(res, indent=1))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--arm", default="epsilon", choices=ARMS)
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 17)))
    ap.add_argument("--scale", type=float, default=1.0, help="run: STRATA's counts times this")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    args = ap.parse_args()
    {"run": run, "analyze": analyze}[args.what](args)
