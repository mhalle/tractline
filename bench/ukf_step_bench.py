"""One UKF filter step, batched in torch, timed: the kernel a GPU re-implementation of UKF rests on.

    python bench/ukf_step_bench.py

The 2T simple model as pnlbwh/ukftractography 2d2b661 implements it (from its source; ukf.py
implements the same): state n = 10 ([m1, l1, l2, m2, l1, l2]), 21 sigma points with kappa = 0.01 and
equal mean/covariance weights, process model = renormalize m and clamp lambda >= 100, measurement
y_j = 1/2 exp(-b_j u'D1u) + 1/2 exp(-b_j u'D2u) with u'Du = l2 + (l1 - l2)(u.m)^2 (lambda in
1e-6 mm^2/s), and the unscented INFORMATION-filter update (Reddy & Rathi 2016) - no measurement-
sized inverse, only 10x10. The antipodal duplicate rows are dropped (N rows with Rs/2, exactly
equivalent). This times the step only - no interpolation, no tracking bookkeeping - on random but
valid states, so it bounds what a batched tracker could do per step.
"""
import json, time
from pathlib import Path
import numpy as np
import torch

KAPPA, RS, QM, QL, LMIN = 0.01, 0.02, 0.001, 50.0, 100.0
n = 10
W = torch.tensor([KAPPA / (n + KAPPA)] + [1 / (2 * (n + KAPPA))] * (2 * n))
SCALE = (n + KAPPA) ** 0.5
Q = torch.diag(torch.tensor([QM] * 3 + [QL] * 2 + [QM] * 3 + [QL] * 2))


def F(X):
    m1, m2 = X[..., 0:3], X[..., 5:8]
    X = X.clone()
    X[..., 0:3] = m1 / m1.norm(dim=-1, keepdim=True)
    X[..., 5:8] = m2 / m2.norm(dim=-1, keepdim=True)
    X[..., [3, 4, 8, 9]] = X[..., [3, 4, 8, 9]].clamp_min(LMIN)
    return X


def H(X, g, b):
    out = 0
    for o in (0, 5):
        m = X[..., o:o + 3]
        m = m / m.norm(dim=-1, keepdim=True)
        l1, l2 = X[..., o + 3].clamp_min(LMIN), X[..., o + 4].clamp_min(LMIN)
        um = torch.einsum("bsk,nk->bsn", m, g)                         # (B, 21, N)
        out = out + 0.5 * torch.exp(-b * 1e-6 * (l2[..., None] + (l1 - l2)[..., None] * um * um))
    return out


def step(x, P, z, g, b, Wt, Qt):
    B = x.shape[0]
    L, _ = torch.linalg.cholesky_ex(P)
    dX = SCALE * L.transpose(1, 2)                                      # rows: columns of L
    X = torch.cat([x[:, None], x[:, None] + dX, x[:, None] - dX], 1)  # (B, 21, n)
    X = F(X)
    xh = torch.einsum("s,bsk->bk", Wt, X)
    Xt = X - xh[:, None]
    Pm = torch.einsum("bsi,s,bsj->bij", Xt, Wt, Xt) + Qt
    Yk = torch.linalg.inv(Pm)
    yh = torch.einsum("bij,bj->bi", Yk, xh)
    Z = H(X, g, b)
    zh = torch.einsum("s,bsn->bn", Wt, Z)
    Zt = Z - zh[:, None]
    Pxz = torch.einsum("bsi,s,bsn->bin", Xt, Wt, Zt)                   # (B, n, N)
    Ht = Yk @ Pxz
    inv_r = 2.0 / RS                                                    # N rows standing for 2N
    Iinfo = inv_r * Ht @ Ht.transpose(1, 2)
    i = inv_r * torch.einsum("bin,bn->bi", Ht, (z - zh) + torch.einsum("bin,bi->bn", Pxz, yh))
    Pp = torch.linalg.inv(Yk + Iinfo)
    return torch.einsum("bij,bj->bi", Pp, i + yh), Pp


def bench(device, dtype, B, N=150, reps=5):
    gen = torch.Generator().manual_seed(0)
    g = torch.nn.functional.normalize(torch.randn(N, 3, generator=gen), dim=1)
    b = torch.full((N,), 2000.0)
    m1 = torch.nn.functional.normalize(torch.randn(B, 3, generator=gen), dim=1)
    m2 = torch.nn.functional.normalize(torch.randn(B, 3, generator=gen), dim=1)
    lam = torch.stack([torch.full((B,), 1700.0), torch.full((B,), 300.0)], 1)
    x = torch.cat([m1, lam, m2, lam], 1)
    P = 0.01 * torch.eye(n).expand(B, n, n).clone()
    z = H(x[:, None], g, b)[:, 0] * (1 + 0.02 * torch.randn(B, N, generator=gen))
    t = [a.to(device=device, dtype=dtype) for a in (x, P, z, g, b, W, Q)]
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)
    x1, P1 = step(*t); sync()                                           # warm
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); step(*t); sync(); best = min(best, time.perf_counter() - t0)
    ok = bool(torch.isfinite(x1).all())
    return {"device": device, "dtype": str(dtype).replace("torch.", ""), "batch": B, "gradients": N,
            "step_ms": round(best * 1000, 2), "fiber_steps_per_s": int(B / best), "finite": ok}


if __name__ == "__main__":
    rows = []
    for dev, dt, B in (("cpu", torch.float64, 20000), ("cpu", torch.float32, 20000),
                       ("mps", torch.float32, 20000), ("mps", torch.float32, 100000)):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        r = bench(dev, dt, B); rows.append(r); print(json.dumps(r), flush=True)
    out = Path(__file__).resolve().parent / "results" / "ukf_step.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev["local"] = rows
    out.write_text(json.dumps(prev, indent=1))
