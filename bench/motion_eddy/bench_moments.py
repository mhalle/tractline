import time, torch
dev = torch.device("mps"); dt = torch.float32
def sync(): torch.mps.synchronize()
def timeit(f, reps=5):
    f(); sync(); ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); f(); sync(); ts.append(time.perf_counter()-t0)
    return round(sorted(ts)[reps//2]*1000,1)
for G, Nm in [(100, 138240), (105, 233625)]:
    m = torch.rand(Nm, 10, device=dev, dtype=dt); iu = torch.triu_indices(10, 10, device=dev)
    mm = (m[:, iu[0]] * m[:, iu[1]]).contiguous()
    Gt = torch.randn(G, 3, Nm, device=dev, dtype=dt); r = torch.randn(G, Nm, device=dev, dtype=dt)
    pairs = torch.tensor([[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]], device=dev)
    def v2d():
        gg = (Gt[:, pairs[:, 0]] * Gt[:, pairs[:, 1]]).reshape(G * 6, Nm)
        H = (gg @ mm).reshape(G, 6, 55)
        b = ((Gt * r[:, None]).reshape(G * 3, Nm) @ m).reshape(G, 3, 10)
        return H, b
    def vchunk(k=8192):          # split K: partial products summed (MPS matmul with huge K)
        gg = (Gt[:, pairs[:, 0]] * Gt[:, pairs[:, 1]]).reshape(G * 6, Nm)
        H = sum(gg[:, i:i+k] @ mm[i:i+k] for i in range(0, Nm, k))
        return H
    print(G, Nm, "2D matmul", timeit(v2d), "ms; K-chunked", timeit(vchunk), "ms; products only",
          timeit(lambda: (Gt[:, pairs[:, 0]] * Gt[:, pairs[:, 1]]).reshape(G * 6, Nm)), "ms", flush=True)
