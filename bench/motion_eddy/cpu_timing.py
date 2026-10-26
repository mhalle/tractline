"""CPU-only first timing for docs/gpu-motion-eddy-design.md section 3 (superseded by bench_pieces.py on MPS)."""
import time, torch, torch.nn.functional as F
torch.set_num_threads(8)
assert not torch.backends.mps.is_built() or True
for (X,Y,Z,V) in [(96,96,50,100),(104,104,72,105)]:
    img = torch.rand(V,1,X,Y,Z)
    g = torch.stack(torch.meshgrid(*[torch.linspace(-1,1,n) for n in (X,Y,Z)], indexing="ij"),-1)[...,[2,1,0]]
    grid = (g[None] + 0.01*torch.randn(V,1,1,1,3)).contiguous()
    for rep in range(2):
        t0=time.time()
        q = grid.clone().requires_grad_(True)
        S = F.grid_sample(img, q, mode="bilinear", padding_mode="zeros", align_corners=True)
        t1=time.time()
        G, = torch.autograd.grad(S.sum(), q)
        t2=time.time()
    # J'J for P=16 from per-voxel g (3) and 10 monomials, chunked over volumes
    N=X*Y*Z; m = torch.rand(N,10); P=16
    t3=time.time()
    for v0 in range(0,V,16):
        Gv = G[v0:v0+16].reshape(-1,N,3)
        B=Gv.shape[0]; J = torch.cat([(Gv[...,:,None]*m[None,:,None,:4]).reshape(B,N,12), Gv[...,1:2]*m[None,:,4:8]],-1)
        JtJ = torch.einsum("vnp,vnq->vpq", J, J)
    t4=time.time()
    print(f"{X}x{Y}x{Z}x{V}: sample {t1-t0:.2f}s, position grad {t2-t1:.2f}s, J'J (P={P}) {t4-t3:.2f}s", flush=True)
