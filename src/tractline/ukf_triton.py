"""UKF tractography's step as one Triton kernel: a drop-in for ukf.advance on CUDA, in float32.
The CUDA counterpart of ukf_metal.py, tested the same way (modal_ukf_triton.py).

Layout: a program takes F half-fibers. Everything per fiber - the 10-state, the covariance's lower
triangle, the Cholesky factors, the sigma points - is a named [F] vector; everything per gradient is an
[F, NPAD] tile (NPAD = N rounded up to a power of two, the gradients along the second axis, padded
columns masked to zero). Triton has no lists of tensors inside a kernel, so the 10x10 algebra is
written out by a generator (`source`) and imported from a file (Triton reads a kernel's source).

The rules are ukf's, which are the binary's. This kernel's own choices:
  - N measurements, the two sums over rows doubled (the binary's 2N rows are each gradient and its
    negation, with equal u'Du), as ukf_metal does.
  - the cross-covariance as sum_s W_s Xt_s H_s - (sum_s W_s Xt_s) zh, which is the binary's
    sum_s W_s Xt_s (H_s - zh) rearranged, so one pass over the sigma points computes H once.
  - Ht = Yk Pxz never formed: I = 2/Rs Yk (Pxz Pxz') Yk and i = 2/Rs Yk (Pxz term), the same algebra.
  - both 10x10 inverses (Pm and Yk + I, symmetric positive definite) through Cholesky: L^-T L^-1.
  - precise arithmetic by default: IEEE-rounded divide and sqrt, libdevice exp and acos, and no
    fused multiply-add (enable_fp_fusion=False); math="fast" swaps in tl.exp, tl.sqrt and "/".
Sums run in a fixed order, so a launch is deterministic on a device.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile

import numpy as np
import torch

NS = 10


def _lo(name: str, i: int, j: int) -> str:
    """A symmetric matrix's entry by its lower-triangle name."""
    return f"{name}{max(i, j)}_{min(i, j)}"


def source(math: str = "precise") -> str:
    """The kernel's Python source."""
    if math == "precise":
        DIV = lambda a, b: f"tl.div_rn({a}, {b})"
        SQRT = lambda a: f"tl.sqrt_rn({a})"
        EXP = lambda a: f"libdevice.exp({a})"
    else:
        DIV = lambda a, b: f"(({a}) / ({b}))"
        SQRT = lambda a: f"tl.sqrt({a})"
        EXP = lambda a: f"tl.exp({a})"
    out = []
    e = lambda s, ind=1: out.append("    " * ind + s)

    def chol_inverse(A: str, R: str, tag: str):
        """R = A^-1 (lower triangle names), A symmetric positive definite, via Cholesky."""
        for j in range(NS):
            d = _lo(A, j, j)
            for k in range(j):
                d = f"({d} - {tag}L{j}_{k} * {tag}L{j}_{k})"
            e(f"{tag}L{j}_{j} = {SQRT(d)}")
            for i in range(j + 1, NS):
                v = _lo(A, i, j)
                for k in range(j):
                    v = f"({v} - {tag}L{i}_{k} * {tag}L{j}_{k})"
                e(f"{tag}L{i}_{j} = {DIV(v, f'{tag}L{j}_{j}')}")
        # M = L^-1, lower: M_ii = 1 / L_ii, M_ij = -(sum_{k=j}^{i-1} L_ik M_kj) / L_ii
        for i in range(NS):
            e(f"{tag}M{i}_{i} = {DIV('1.0', f'{tag}L{i}_{i}')}")
            for j in range(i):
                acc = " + ".join(f"{tag}L{i}_{k} * {tag}M{k}_{j}" for k in range(j, i))
                e(f"{tag}M{i}_{j} = {DIV(f'-({acc})', f'{tag}L{i}_{i}')}")
        # A^-1 = M' M: (i, j) = sum_{k >= max(i, j)} M_ki M_kj
        for i in range(NS):
            for j in range(i + 1):
                acc = " + ".join(f"{tag}M{k}_{i} * {tag}M{k}_{j}" for k in range(i, NS))
                e(f"{R}{i}_{j} = {acc}")

    def tensors(q: str, tag: str):
        """The two tensors of state q0..q9 as D entries times 1e-6 (filter_Simple2T::H), named {tag}{t}_{i}{j}."""
        for t in range(2):
            o = 5 * t
            e(f"_n = {SQRT(f'{q}{o} * {q}{o} + {q}{o + 1} * {q}{o + 1} + {q}{o + 2} * {q}{o + 2}')}")
            e(f"_m0 = {DIV(f'{q}{o}', '_n')}; _m1 = {DIV(f'{q}{o + 1}', '_n')}; _m2 = {DIV(f'{q}{o + 2}', '_n')}")
            e("_neg = _m0 < 0.0")
            e("_m0 = tl.where(_neg, -_m0, _m0); _m1 = tl.where(_neg, -_m1, _m1); _m2 = tl.where(_neg, -_m2, _m2)")
            e(f"_l1 = tl.maximum({q}{o + 3}, 100.0); _l2 = tl.maximum({q}{o + 4}, 100.0)")
            e(f"_r11 = {DIV('_m1 * _m1', '1.0 + _m0')} - 1.0; _r12 = {DIV('_m1 * _m2', '1.0 + _m0')}; "
              f"_r22 = {DIV('_m2 * _m2', '1.0 + _m0')} - 1.0")
            R = [["_m0", "_m1", "_m2"], ["_m1", "_r11", "_r12"], ["_m2", "_r12", "_r22"]]
            L = ["_l1", "_l2", "_l2"]
            for i in range(3):
                for j in range(i, 3):
                    acc = "0.0"
                    for k in range(3):
                        acc = f"({acc} + {R[i][k]} * {L[k]} * {R[j][k]})"
                    e(f"{tag}{t}_{i}{j} = {acc} * 1e-6")

    def h_tile(tag: str, dst: str):
        """dst = H at every gradient [F, NPAD], from tensors {tag}{t}_{ij}, masked to the real gradients."""
        e(f"{dst} = tl.zeros([F, NPAD], dtype=tl.float32)")
        for t in range(2):
            d = lambda i, j: f"{tag}{t}_{min(i, j)}{max(i, j)}[:, None]"
            rows = [f"({d(r, 0)} * gx + {d(r, 1)} * gy + {d(r, 2)} * gz)" for r in range(3)]
            e(f"_q = 0.0 + gx * {rows[0]}")
            e(f"_q = _q + gy * {rows[1]}")
            e(f"_q = _q + gz * {rows[2]}")
            e(f"{dst} = {dst} + {EXP('-bb * _q')} * 0.5")
        e(f"{dst} = tl.where(nm2, {dst}, 0.0)")

    def l2fa(a, b):
        return DIV(f"tl.abs({a} - {b})", SQRT(f"{a} * {a} + 2.0 * {b} * {b}"))

    def cround(x):
        return f"tl.where({x} < 0.0, -tl.floor(-{x} + 0.5), tl.floor({x} + 0.5)).to(tl.int32)"

    out.append("import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n\n")
    out.append("@triton.jit")
    out.append("def ukf_step(A, MASK, G, BV, X, ST, P, OLD, DIR, FLAGS, FAO, MSIG, B, N, nk, nj, ni, step, max_steps,")
    out.append("             vox0, vox1, vox2, sigma, step_length, stop_fa, stop_thr, rr, Qm, Ql, c, W0, Wi,")
    out.append("             F: tl.constexpr, NPAD: tl.constexpr):")
    e("f = tl.program_id(0) * F + tl.arange(0, F)")
    e("fm = f < B")
    e("n = tl.arange(0, NPAD)")
    e("nm = n < N")
    e("nm2 = fm[:, None] & nm[None, :]")
    for k in range(3):
        e(f"x{k} = tl.load(X + f * 3 + {k}, mask=fm, other=0.0)")
        e(f"o{k} = tl.load(OLD + f * 3 + {k}, mask=fm, other=1.0)")
    for k in range(NS):
        e(f"s{k} = tl.load(ST + f * 10 + {k}, mask=fm, other=1.0)")
    for i in range(NS):
        for j in range(i + 1):
            e(f"p{i}_{j} = tl.load(P + f * 100 + {i * NS + j}, mask=fm, other={'1.0' if i == j else '0.0'})")
    e("gx = tl.load(G + n * 3 + 0, mask=nm, other=0.0)[None, :]")
    e("gy = tl.load(G + n * 3 + 1, mask=nm, other=0.0)[None, :]")
    e("gz = tl.load(G + n * 3 + 2, mask=nm, other=0.0)[None, :]")
    e("bb = tl.load(BV + n, mask=nm, other=0.0)[None, :]")

    # ---- interp (Interp3Signal), in the binary's x, y, z loop order
    e("r0 = " + cround("x0")); e("r1 = " + cround("x1")); e("r2 = " + cround("x2"))
    e("z = tl.zeros([F, NPAD], dtype=tl.float32)")
    e("wsum = tl.zeros([F], dtype=tl.float32) + 1e-16")
    for xx in (-1, 0, 1):
        for yy in (-1, 0, 1):
            for zz in (-1, 0, 1):
                e(f"_X = r0 + {xx}; _Y = r1 + {yy}; _Z = r2 + {zz}")
                e("_ok = (_X >= 0) & (_X < nk) & (_Y >= 0) & (_Y < nj) & (_Z >= 0) & (_Z < ni)")
                e("_dx = (_X.to(tl.float32) - x0) * vox0; _dy = (_Y.to(tl.float32) - x1) * vox1; _dz = (_Z.to(tl.float32) - x2) * vox2")
                e(f"_w = {EXP(DIV('-(_dx * _dx + _dy * _dy + _dz * _dz)', 'sigma'))}")
                e("_w = tl.where(_ok, _w, 0.0)")
                e("_v = ((tl.minimum(tl.maximum(_X, 0), nk - 1) * nj + tl.minimum(tl.maximum(_Y, 0), nj - 1)) * ni"
                  " + tl.minimum(tl.maximum(_Z, 0), ni - 1)).to(tl.int64) * N")
                e("_a = tl.load(A + _v[:, None] + n[None, :], mask=nm2, other=0.0)")
                e("z = z + _w[:, None] * _a")
                e("wsum = wsum + _w")
    e(f"z = {DIV('z', 'wsum[:, None]')}")

    # ---- Cholesky of P; sigma points X_s = F(x +/- c L[:, j])
    for j in range(NS):
        d = f"p{j}_{j}"
        for k in range(j):
            d = f"({d} - cL{j}_{k} * cL{j}_{k})"
        e(f"cL{j}_{j} = {SQRT(d)}")
        for i in range(j + 1, NS):
            v = f"p{i}_{j}"
            for k in range(j):
                v = f"({v} - cL{i}_{k} * cL{j}_{k})"
            e(f"cL{i}_{j} = {DIV(v, f'cL{j}_{j}')}")

    def sigma_point():
        """q0..q9 for the loop's si (runtime), F applied; W the weight."""
        e("_col = si - 1 - 10 * (si > 10).to(tl.int32)", 2)
        e("_sg = tl.where(si == 0, 0.0, tl.where(si <= 10, 1.0, -1.0))", 2)
        e("W = tl.where(si == 0, W0, Wi)", 2)
        for k in range(NS):
            sel = "0.0"
            for j in range(k, -1, -1):
                sel = f"tl.where(_col == {j}, cL{k}_{j}, {sel})"
            e(f"q{k} = s{k} + _sg * (c * {sel})", 2)
        for o in (0, 5):
            e(f"_r = {DIV('1.0', SQRT(f'q{o} * q{o} + q{o + 1} * q{o + 1} + q{o + 2} * q{o + 2}'))}", 2)
            e(f"q{o} = q{o} * _r; q{o + 1} = q{o + 1} * _r; q{o + 2} = q{o + 2} * _r", 2)
            e(f"q{o + 3} = tl.maximum(q{o + 3}, 100.0); q{o + 4} = tl.maximum(q{o + 4}, 100.0)", 2)

    for k in range(NS):
        e(f"xh{k} = tl.zeros([F], dtype=tl.float32)")
    e("for si in range(21):")
    sigma_point()
    for k in range(NS):
        e(f"xh{k} = xh{k} + W * q{k}", 2)
    # pass 2: Pm, sum W Xt, zh and the cross-covariance, H once per sigma point
    for i in range(NS):
        for j in range(i + 1):
            e(f"pm{i}_{j} = tl.zeros([F], dtype=tl.float32)")
        e(f"sxt{i} = tl.zeros([F], dtype=tl.float32)")
        e(f"pxz{i} = tl.zeros([F, NPAD], dtype=tl.float32)")
    e("zh = tl.zeros([F, NPAD], dtype=tl.float32)")
    e("for si in range(21):")
    sigma_point()
    for k in range(NS):
        e(f"t{k} = q{k} - xh{k}", 2)
    for i in range(NS):
        for j in range(i + 1):
            e(f"pm{i}_{j} = pm{i}_{j} + t{i} * W * t{j}", 2)
        e(f"sxt{i} = sxt{i} + W * t{i}", 2)
    ind = len(out)
    tensors("q", "D")
    h_tile("D", "h")
    for k in range(ind, len(out)):                                    # those lines belong in the loop
        out[k] = "    " + out[k]
    e("zh = zh + W * h", 2)
    for i in range(NS):
        e(f"pxz{i} = pxz{i} + (t{i} * W)[:, None] * h", 2)
    for i in range(NS):
        e(f"pxz{i} = pxz{i} - sxt{i}[:, None] * zh")
        e(f"pm{i}_{i} = pm{i}_{i} + {'Qm' if i % 5 < 3 else 'Ql'}")

    # ---- the information-form update
    chol_inverse("pm", "yk", "a")                                     # Yk = Pm^-1
    for i in range(NS):
        e(f"yh{i} = " + " + ".join(f"{_lo('yk', i, j)} * xh{j}" for j in range(NS)))
    e("term = (z - zh) + " + " + ".join(f"pxz{i} * yh{i}[:, None]" for i in range(NS)))
    for i in range(NS):
        for j in range(i + 1):
            e(f"S{i}_{j} = tl.sum(pxz{i} * pxz{j}, axis=1)")
        e(f"v{i} = tl.sum(pxz{i} * term, axis=1)")
    for i in range(NS):                                               # T = Yk S
        for j in range(NS):
            e(f"T{i}_{j} = " + " + ".join(f"{_lo('yk', i, k)} * {_lo('S', k, j)}" for k in range(NS)))
    for i in range(NS):                                               # M = Yk + 2/Rs T Yk
        for j in range(i + 1):
            acc = " + ".join(f"T{i}_{k} * {_lo('yk', k, j)}" for k in range(NS))
            e(f"mm{i}_{j} = yk{i}_{j} + 2.0 * (rr * ({acc}))")
        e(f"iv{i} = 2.0 * (rr * (" + " + ".join(f"{_lo('yk', i, j)} * v{j}" for j in range(NS)) + "))")
    chol_inverse("mm", "pn", "b")                                     # Pn = (Yk + I)^-1
    for i in range(NS):
        e(f"s{i} = " + " + ".join(f"{_lo('pn', i, j)} * (iv{j} + yh{j})" for j in range(NS)))

    # ---- State2Tensor2T, the two swaps, FA, the Euler step (tractography.cc)
    for o, m in ((0, "m1"), (5, "m2")):
        e(f"_n = {SQRT(f's{o} * s{o} + s{o + 1} * s{o + 1} + s{o + 2} * s{o + 2}')}")
        e(f"{m}x = {DIV(f's{o}', '_n')}; {m}y = {DIV(f's{o + 1}', '_n')}; {m}z = {DIV(f's{o + 2}', '_n')}")
    e("L1a = tl.maximum(s3, 100.0); L1b = tl.maximum(s4, 100.0); L2a = tl.maximum(s8, 100.0); L2b = tl.maximum(s9, 100.0)")
    for m in ("m1", "m2"):
        e(f"_fl = ({m}x * o0 + {m}y * o1 + {m}z * o2) < 0.0")
        e(f"{m}x = tl.where(_fl, -{m}x, {m}x); {m}y = tl.where(_fl, -{m}y, {m}y); {m}z = tl.where(_fl, -{m}z, {m}z)")
    e(f"fat1 = {l2fa('L1a', 'L1b')}; fat2 = {l2fa('L2a', 'L2b')}")
    e("angle = libdevice.acos(m1x * m2x + m1y * m2y + m1z * m2z) * 57.29577951308232")
    e("sw = (m1x * o0 + m1y * o1 + m1z * o2) < (m2x * o0 + m2y * o1 + m2z * o2)")

    def swap(cond):
        for a in "xyz":
            e(f"_t = m1{a}; m1{a} = tl.where({cond}, m2{a}, m1{a}); m2{a} = tl.where({cond}, _t, m2{a})")
        e(f"_t = L1a; L1a = tl.where({cond}, L2a, L1a); L2a = tl.where({cond}, _t, L2a)")
        e(f"_t = L1b; L1b = tl.where({cond}, L2b, L1b); L2b = tl.where({cond}, _t, L2b)")
        for i in range(5):
            e(f"_t = s{i}; s{i} = tl.where({cond}, s{i + 5}, s{i}); s{i + 5} = tl.where({cond}, _t, s{i + 5})")
        for i in range(NS):                                           # P'[i][j] = P[(i+5)%10][(j+5)%10]
            for j in range(i + 1):
                e(f"_q{i}_{j} = tl.where({cond}, {_lo('pn', (i + 5) % NS, (j + 5) % NS)}, pn{i}_{j})")
        for i in range(NS):
            for j in range(i + 1):
                e(f"pn{i}_{j} = _q{i}_{j}")
    swap("sw")
    e("_t = fat1; fat1 = tl.where(sw, fat2, fat1); fat2 = tl.where(sw, _t, fat2)")
    e("sw2 = (angle <= 20.0) & (tl.minimum(fat1, fat2) <= 0.2) & ~(fat1 > 0.2)")
    swap("sw2")
    e(f"fa = tl.where(L1a < L1b, 0.0, {l2fa('L1a', 'L1b')})")
    e(f"x0 = x0 + {DIV('m1z', 'vox0')} * step_length")
    e(f"x1 = x1 + {DIV('m1y', 'vox1')} * step_length")
    e(f"x2 = x2 + {DIV('m1x', 'vox2')} * step_length")

    # ---- the checks at the new position
    tensors("s", "E")
    h_tile("E", "hs")
    e(f"mean_sig = {DIV('tl.sum(hs, axis=1)', 'N.to(tl.float32)')}")
    e("q0i = " + cround("x0")); e("q1i = " + cround("x1")); e("q2i = " + cround("x2"))
    e("inb = (q0i >= 0) & (q0i < nk) & (q1i >= 0) & (q1i < nj) & (q2i >= 0) & (q2i < ni)")
    e("_mi = ((tl.minimum(tl.maximum(q0i, 0), nk - 1) * nj + tl.minimum(tl.maximum(q1i, 0), nj - 1)) * ni"
      " + tl.minimum(tl.maximum(q2i, 0), ni - 1)).to(tl.int64)")
    e("inside = inb & (tl.load(MASK + _mi, mask=fm, other=0) > 0)")
    e("stop = (~inside) | (mean_sig < stop_thr) | (fa < stop_fa) | (step > max_steps)")

    for k in range(3):
        e(f"tl.store(X + f * 3 + {k}, x{k}, mask=fm)")
    for k, a in enumerate("xyz"):
        e(f"tl.store(DIR + f * 3 + {k}, m1{a}, mask=fm)")
    for k in range(NS):
        e(f"tl.store(ST + f * 10 + {k}, s{k}, mask=fm)")
    for i in range(NS):
        for j in range(NS):
            e(f"tl.store(P + f * 100 + {i * NS + j}, {_lo('pn', i, j)}, mask=fm)")
    e("tl.store(FLAGS + f, stop.to(tl.int32) | (sw.to(tl.int32) << 1) | (sw2.to(tl.int32) << 2) | (inside.to(tl.int32) << 3), mask=fm)")
    e("tl.store(FAO + f, fa, mask=fm)")
    e("tl.store(MSIG + f, mean_sig, mask=fm)")
    return "\n".join(out) + "\n"


_MODS: dict = {}


def kernel(math: str = "precise"):
    """The generated kernel, imported from a file (Triton reads a kernel's source from its module)."""
    if math not in _MODS:
        src = source(math)
        h = hashlib.sha256(src.encode()).hexdigest()[:16]
        d = os.path.join(tempfile.gettempdir(), "ukf_triton")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"ukf_step_{math}_{h}.py")
        if not os.path.exists(path):
            with open(path + ".tmp", "w") as fh:
                fh.write(src)
            os.replace(path + ".tmp", path)
        spec = importlib.util.spec_from_file_location(f"ukf_step_{math}_{h}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[math] = mod.ukf_step
    return _MODS[math]


#: launch defaults (tuned on an A10G by modal_ukf_triton.py)
F_DEFAULT, WARPS_DEFAULT = 4, 4


@torch.no_grad()
def advance(D: dict, xa, sa, Pa, oa, Q, Rs, step, max_steps, step_length=0.3, stopping_fa=0.08,
            stopping_threshold=0.06, F=None, num_warps=None, math="precise"):
    """ukf.advance's contract on CUDA, float32: returns (x, state, P, direction, stop, info).
    D is ukf.at_dtype(D, torch.float32, 'cuda')."""
    F = F or F_DEFAULT
    num_warps = num_warps or WARPS_DEFAULT
    dev = xa.device
    N = int(D["N"])
    B = xa.shape[0]
    nk, nj, ni = (int(v) for v in D["dim"])
    x = xa.contiguous().clone(); s = sa.contiguous().clone(); P = Pa.reshape(B, -1).contiguous().clone()
    o = oa.contiguous()
    d = torch.empty((B, 3), dtype=torch.float32, device=dev)
    flags = torch.empty(B, dtype=torch.int32, device=dev)
    fa = torch.empty(B, dtype=torch.float32, device=dev); ms = torch.empty(B, dtype=torch.float32, device=dev)
    q = torch.diagonal(Q).cpu()
    vox = D["voxel"].cpu()
    f32 = np.float32
    c = float(np.sqrt(f32(10) + f32(0.01), dtype=f32))
    W0 = float(f32(0.01) / (f32(10) + f32(0.01))); Wi = float(f32(0.5) / (f32(10) + f32(0.01)))
    rr = float(f32(1.0) / f32(Rs))
    g = D["g"][:N].contiguous(); b = D["b"][:N].contiguous()
    NPAD = 1 << max(0, (N - 1).bit_length())
    grid = ((B + F - 1) // F,)
    kernel(math)[grid](D["A"], D["mask"], g, b, x, s, P, o, d, flags, fa, ms, B, N, nk, nj, ni, int(step), int(max_steps),
                       float(vox[0]), float(vox[1]), float(vox[2]), float(vox.min()), float(step_length),
                       float(stopping_fa), float(stopping_threshold), rr, float(q[0]), float(q[3]), c, W0, Wi,
                       F=F, NPAD=NPAD, num_warps=num_warps, enable_fp_fusion=False)
    stop = (flags & 1).bool()
    info = {"swap": (flags & 2).bool(), "swap2": (flags & 4).bool(), "fa": fa, "mean_signal": ms, "inside": (flags & 8).bool()}
    return x, s, P.reshape(B, 10, 10), d, stop, info
