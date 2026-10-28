"""SynthStrip's brain extraction (Hoopes et al. 2022), an independent code base: its network run with its authors'
released weights, on any image and any torch device, with no dependency beyond torch and numpy.

Written from the paper and the weight file's tensor shapes; not copied or translated from FreeSurfer's mri_synthstrip
or from ports of it. A port (mhalle/synthstrip-torch 0.1.1, run on Modal: bench/skullstrip_reference_modal.py) served
only as a reference for outputs. The sources:
  - the paper (NeuroImage 260:119474, section 3.3): a 3D U-Net of seven resolution levels, each two 3x3x3
    convolutions with leaky ReLU (0.2), max-pooling down, skip connections by concatenation, one final linear
    convolution to a signed distance transform (SDT) in mm; at test time the input conformed to 1 mm isotropic
    voxels by trilinear interpolation and its intensities scaled to [0, 1];
  - the weight file's tensor names and shapes: features 16, 32, 64, 64, 64, 64 down; 64 x 5 and 32 up; 16, 16 and 1
    at full resolution - so six poolings, and inputs padded to a multiple of 64;
  - the reference's outputs, to settle what the paper leaves open (bench/skullstrip_check.py): parameter-free nearest
    upsampling (on the reference's own conformed input the network here gives its output to 0.007 mm); the conformed
    axes being the input's own, permuted and flipped to the nearest LIA (an oblique scan stays oblique); intensities
    over the 99th percentile clipped; and the model's sign, negative inside the brain.
The conformed grid's extent is ours: the input's nonzero box plus MARGIN mm of empty context on every side, rounded
up to a multiple of 64. Context matters where a field of view ends at the brain: with none, on a scan cut at the pons,
the brainstem and cerebellum lose 7-9 points of coverage against the reference's; with 16 mm the masks agree with the reference's at Dice
0.988-0.992 (skullstrip_check.py).

The weights (synthstrip.1.pt, 30.9 MB, fetched on first use: tractline.fetch) are its authors', offered under the MIT
license or CC BY 4.0 at the user's choice (https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/); tractline takes them
under MIT and does not redistribute them. Cite Hoopes A, Mora JS, Dalca AV, Fischl B, Hoffmann M. SynthStrip:
skull-stripping for any brain image. NeuroImage 2022;260:119474.

    sdt = signed_distance(b0, affine, weights("synthstrip"), device="mps")   # (X, Y, Z) mm, negative inside
    mask = sdt < 1.0                                      # SynthStrip's default border; 3.0 grows it by 2 mm

Memory and speed. Run whole, the network's full-resolution levels hold up to 48 channels over the conformed grid (256 x
192 x 256 for a tilted head: 2.4 GB a tensor), torch's 3D convolution on the CPU unfolds its input 27-fold, and MPS
returns zeros once memory runs short. Here no full-resolution tensor wider than one channel is held whole
(Net.forward), every convolution runs in slabs (SLAB), and off CUDA a 3D convolution is three 2D ones (_Conv; faster
than torch's 3D convolution there, skullstrip_check.py times one layer). skullstrip_check.py records the time and peak
memory on that grid per device.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ENC = (16, 32, 64, 64, 64, 64)            # encoder features, finest first (the weight file's shapes)
DEC = (64, 64, 64, 64, 64, 32)            # decoder features, coarsest first
TAIL = (16, 16)                           # full-resolution convolutions after the last skip
LIA = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]).T   # columns: the world directions of the axes L, I, A
OUTSIDE = 100.0                           # the SDT given to points outside the conformed grid: far outside the brain
MARGIN = 16.0                             # mm of empty context added on every side of the input's box
#: elements of a convolution's input (unfolded, on CUDA) held at once, per device. Chosen on a tilted head's 256 x 192 x
#: 256 grid, from runs while developing it (not recorded): on MPS larger slabs were faster but Metal kept a buffer of
#: every slab size it had seen, raising the peak footprint; on the CPU smaller ones were slower. CUDA unfolds (27 x),
#: so 2^27
SLAB = {"mps": 2 ** 23, "cpu": 2 ** 26, "cuda": 2 ** 27}


class _Conv(nn.Module):
    """One 3x3x3 convolution, optionally followed by leaky ReLU (0.2); the attribute path (`conv`) is the file's."""
    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.conv, self.act = nn.Conv3d(cin, cout, 3, padding=1), act

    def forward(self, x):
        """In slabs along the last axis, each with a one-slice halo (the same arithmetic per voxel, in pieces), so no
        call holds more than about SLAB elements. On CUDA, cuDNN's 3D convolution; elsewhere the convolution as three
        2D ones (_as_2d), faster than torch's 3D path there (bench/skullstrip_check.py times one layer: 15x on MPS,
        5-10x on the CPU), which also unfolds its input to 27 x its size."""
        cuda = x.device.type == "cuda"
        run = self.conv if cuda else self._as_2d
        Z = x.shape[-1]
        s = max(1, _slab(x.device) // _per_slice(x.shape[1], x.shape[2] * x.shape[3], cuda))
        if s >= Z:
            y = run(x)
        else:
            y = x.new_empty((x.shape[0], self.conv.out_channels, *x.shape[2:]))
            for z0 in range(0, Z, s):
                z1 = min(Z, z0 + s)
                a, b = max(0, z0 - 1), min(Z, z1 + 1)
                y[..., z0:z1] = run(x[..., a:b])[..., z0 - a:z0 - a + z1 - z0]
        return F.leaky_relu(y, 0.2) if self.act else y

    def _as_2d(self, x):
        """The 3x3x3 convolution (zero padding 1) as the sum of three 3x3 ones over (X, Y), one per slice offset along
        Z, with Z as the batch: the same products, summed in another order (agrees to ~2e-5 on unit-scale inputs)."""
        w, b = self.conv.weight, self.conv.bias
        Z = x.shape[-1]
        out = []
        for n in range(x.shape[0]):
            xb = x[n].permute(3, 0, 1, 2).contiguous()                    # (Z, C, X, Y): the slices as the batch
            y = F.conv2d(xb, w[..., 1], b, padding=1)                     # each slice's own taps
            if Z > 1:                                                     # its neighbors'; past the ends, zero
                y[1:] += F.conv2d(xb[:-1], w[..., 0], None, padding=1)
                y[:-1] += F.conv2d(xb[1:], w[..., 2], None, padding=1)
            out.append(y.permute(1, 2, 3, 0))
        return torch.stack(out)


def _slab(device) -> int:
    return SLAB.get(torch.device(device).type, SLAB["cpu"])


def _release(device):
    """Return the stage's freed blocks to Metal: MPS's allocator otherwise keeps every slab size it has seen."""
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()


def _per_slice(channels, plane, cuda):
    """Elements one slice of a convolution's input costs: unfolded on CUDA's path (27 x), as is on the 2D path."""
    return (27 if cuda else 1) * channels * plane


class Net(nn.Module):
    """The U-Net: encoder levels (two convolutions, keep the skip, max-pool), decoder levels (two convolutions,
    nearest upsampling, concatenate the skip), then two convolutions and the linear one to the SDT."""
    def __init__(self):
        super().__init__()
        self.encoder, c = nn.ModuleList(), 1
        for f in ENC:
            self.encoder.append(nn.Sequential(_Conv(c, f), _Conv(f, f)))
            c = f
        self.decoder = nn.ModuleList()
        for i, f in enumerate(DEC):
            cin = c if i == 0 else c + ENC[-i]
            self.decoder.append(nn.Sequential(_Conv(cin, f), _Conv(f, f)))
            c = f
        self.remaining = nn.ModuleList([_Conv(c + ENC[0], TAIL[0]), _Conv(TAIL[0], TAIL[1]), _Conv(TAIL[1], 1, act=False)])

    def forward(self, x):
        """The U-Net with no full-resolution tensor wider than one channel held whole, and no half-resolution
        concatenation: the first encoder level runs in slabs and is pooled as it goes (its output is recomputed, slab by
        slab, where the last decoder level needs it), and the two finest decoder levels run in slabs. Each slab carries
        a halo of one slice per convolution it runs, so every voxel is computed from what it sees in the whole volume;
        at the volume's own ends the slab's ends are the volume's, zero-padded the same way. The tensors held whole are
        the input, the coarser levels' outputs and skips, and the half-resolution decoder's output; the peak measured
        is in bench/results/skullstrip_check.json. On CUDA, cuDNN's TF32 is turned off for the call (torch allows it
        by default on Ampere and later, ~1e-3 relative): the same float32 arithmetic as on the CPU and MPS."""
        if x.is_cuda:
            with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
                return self._forward(x)
        return self._forward(x)

    def _forward(self, x):
        h = self._level0_pooled(x)
        _release(x.device)
        skips = [None]                                                    # level 0: recomputed in _full_resolution
        for level in self.encoder[1:]:
            h = level(h)
            skips.append(h)
            h = F.max_pool3d(h, 2)
        for i, level in enumerate(self.decoder[:-2]):
            h = torch.cat([F.interpolate(level(h), scale_factor=2, mode="nearest"), skips[-1 - i]], 1)
            skips[-1 - i] = None                                          # each skip is used once: free it
        h = self._up_cat(self.decoder[-2](h), skips[1], self.decoder[-1], halo=2)
        skips[1] = None
        _release(x.device)
        y = self._full_resolution(h, x)
        _release(x.device)
        return y

    @staticmethod
    def _step(channels, plane, device):
        """Slices per slab: a convolution's input under SLAB elements (_per_slice), at least 8, even (pooling pairs
        slices)."""
        return max(8, _slab(device) // _per_slice(channels, plane, device.type == "cuda")) // 2 * 2

    @staticmethod
    def _up(low, a, b):
        """Nearest upsampling (x2 on every axis) of `low`, only slices a..b-1 of the result: slice z reads low's z // 2."""
        idx = torch.arange(a, b, device=low.device) // 2
        return low.index_select(-1, idx).repeat_interleave(2, -2).repeat_interleave(2, -3)

    def _level0_pooled(self, x):
        X, Y, Z = x.shape[2:]
        out = x.new_empty((x.shape[0], ENC[0], X // 2, Y // 2, Z // 2))
        s = self._step(ENC[0], X * Y, x.device)
        for z0 in range(0, Z, s):
            z1 = min(Z, z0 + s)
            a, b = max(0, z0 - 2), min(Z, z1 + 2)
            out[..., z0 // 2:z1 // 2] = F.max_pool3d(self.encoder[0](x[..., a:b])[..., z0 - a:z0 - a + z1 - z0], 2)
        return out

    def _up_cat(self, low, skip, block, halo):
        """block(cat(upsample(low), skip)) in slabs along the last axis."""
        Z = skip.shape[-1]
        out = None
        s = self._step(low.shape[1] + skip.shape[1], skip.shape[2] * skip.shape[3], skip.device)
        for z0 in range(0, Z, s):
            z1 = min(Z, z0 + s)
            a, b = max(0, z0 - halo), min(Z, z1 + halo)
            y = block(torch.cat([self._up(low, a, b), skip[..., a:b]], 1))[..., z0 - a:z0 - a + z1 - z0]
            if out is None:
                out = y.new_empty((*y.shape[:-1], Z))
            out[..., z0:z1] = y
        return out

    def _full_resolution(self, low, x):
        """The last upsampling, the first encoder level's output (recomputed from the input, two convolutions, so a
        two-slice halo), their concatenation and the three full-resolution convolutions (a three-slice halo)."""
        X, Y, Z = x.shape[2:]
        out = x.new_empty((x.shape[0], 1, X, Y, Z))
        s = self._step(low.shape[1] + ENC[0], X * Y, x.device)
        for z0 in range(0, Z, s):
            z1 = min(Z, z0 + s)
            a, b = max(0, z0 - 3), min(Z, z1 + 3)
            a2, b2 = max(0, a - 2), min(Z, b + 2)
            skip = self.encoder[0](x[..., a2:b2])[..., a - a2:a - a2 + b - a]
            y = torch.cat([self._up(low, a, b), skip], 1)
            for level in self.remaining:
                y = level(y)
            out[..., z0:z1] = y[..., z0 - a:z0 - a + z1 - z0]
        return out


_NETS: dict = {}


def load(path, device="cpu") -> Net:
    """The network with the weights at `path` (the released synthstrip.1.pt), cached per (path, device). The file is a
    training checkpoint; only its model_state_dict is read, with torch's weights-only loader."""
    key = (str(path), str(device))
    if key not in _NETS:
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
        net = Net()
        net.load_state_dict(ck["model_state_dict"], strict=True)
        _NETS[key] = net.to(device).eval()
    return _NETS[key]


def _lia_axes(affine) -> tuple[np.ndarray, np.ndarray]:
    """The input's voxel axes as unit world directions, permuted and flipped to the nearest of L, I, A: (D, perm)
    with D (3, 3) the conformed grid's directions (columns) and perm[k] the input axis behind conformed axis k."""
    R = np.asarray(affine, float)[:3, :3]
    U = R / np.linalg.norm(R, axis=0)
    D, perm, free = np.zeros((3, 3)), [0, 0, 0], [0, 1, 2]
    dots = np.abs(LIA.T @ U)                                              # (target, input axis)
    for _ in range(3):                                                    # greedy: the best-aligned pair first
        t, j = np.unravel_index(np.argmax(np.where(np.isin(np.arange(3), free)[None, :], dots, -1)), dots.shape)
        s = np.sign(LIA[:, t] @ U[:, j]) or 1.0
        D[:, t], perm[t] = s * U[:, j], j
        dots[t, :] = -1
        free.remove(j)
    return D, np.asarray(perm)


def conform(vol, affine, device="cpu", interp="linear", margin=MARGIN):
    """The network's input: `vol` (X, Y, Z) resampled (trilinear) to 1 mm voxels on axes D (_lia_axes) over the box,
    in that frame, holding the input's nonzero voxels (its whole field of view when nothing is zero), each side
    rounded up to a multiple of 64 voxels and centered on the box; zero outside the input. Intensities: minus the
    minimum, over the 99th percentile clipped, scaled to [0, 1]. Returns (x (1, 1, n0, n1, n2) float32 on device,
    the conformed grid's vox2world (4, 4))."""
    v = np.nan_to_num(np.asarray(vol, np.float32), nan=0.0, posinf=0.0, neginf=0.0)   # a NaN would poison the scaling
    A = np.asarray(affine, float)
    D, _ = _lia_axes(A)
    ijk = np.argwhere(v > 0) if (v > 0).any() else np.argwhere(np.ones(v.shape, bool))
    lo_v, hi_v = ijk.min(0), ijk.max(0)
    corners = np.array([[a, b, c] for a in (lo_v[0], hi_v[0]) for b in (lo_v[1], hi_v[1]) for c in (lo_v[2], hi_v[2])], float)
    p = (corners @ A[:3, :3].T + A[:3, 3]) @ D                            # the box's corners in the conformed frame (mm)
    lo, hi = p.min(0) - margin, p.max(0) + margin
    shape = (np.ceil((hi - lo + 1) / 64) * 64).astype(int)
    start = (lo + hi) / 2 - (shape - 1) / 2                               # centered: frame coordinate of voxel 0
    C = np.eye(4)
    C[:3, :3], C[:3, 3] = D, D @ start                                    # conformed vox2world (1 mm, axes D)
    x = _sample(torch.as_tensor(v, device=device), np.linalg.inv(A) @ C, shape, fill=0.0, mode=interp)
    x = x - x.min()
    hi99 = float(np.percentile(x.cpu().numpy(), 99))                      # exact (torch.quantile stops at 2^24)
    x = (x / max(hi99, 1e-12)).clamp(0, 1)
    return x[None, None], C


def _sample(v: torch.Tensor, M: np.ndarray, shape, fill: float, mode="linear") -> torch.Tensor:
    """v (X, Y, Z) sampled (trilinear, or nearest) at the points M @ (i, j, k, 1) for every voxel of `shape` (M: target
    voxel -> source voxel), a few target slices at a time; points outside v's voxel centers take `fill`."""
    dev = v.device
    n = torch.as_tensor(v.shape, dtype=torch.float32, device=dev)
    Mt = torch.as_tensor(M, dtype=torch.float32, device=dev)
    out = torch.empty(tuple(int(s) for s in shape), dtype=torch.float32, device=dev)
    step = max(1, 2 ** 22 // (int(shape[1]) * int(shape[2])))
    jk = torch.stack(torch.meshgrid(*[torch.arange(int(s), dtype=torch.float32, device=dev) for s in shape[1:]],
                                    indexing="ij"), -1)                   # (n1, n2, 2)
    for i0 in range(0, int(shape[0]), step):
        i1 = min(int(shape[0]), i0 + step)
        ii = torch.arange(i0, i1, dtype=torch.float32, device=dev)[:, None, None, None].expand(-1, *jk.shape[:2], 1)
        g = torch.cat([ii, jk.expand(i1 - i0, *jk.shape)], -1)
        src = g @ Mt[:3, :3].T + Mt[:3, 3]
        inside = ((src >= 0) & (src <= n - 1)).all(-1)
        grid = (2 * src / (n - 1).clamp_min(1) - 1).flip(-1)              # grid_sample's (W, H, D) = (z, y, x) order
        o = F.grid_sample(v[None, None].float(), grid[None], mode="bilinear" if mode == "linear" else "nearest",
                          align_corners=True)[0, 0]
        out[i0:i1] = torch.where(inside, o, torch.full_like(o, fill))
    return out


def signed_distance(vol, affine, weights, device="cpu", interp="linear", margin=None) -> np.ndarray:
    """SynthStrip's SDT (mm; negative inside the brain, banded at about +-5) on `vol`'s own grid (X, Y, Z): the image
    conformed, the network run, its output resampled back trilinearly (graded, so a threshold gives a sub-voxel
    edge). `weights`: the path of synthstrip.1.pt (tractline.fetch.weights("synthstrip")). Refuses a constant field
    (a backend fault, seen on MPS at large grids) rather than returning it."""
    net = load(weights, device)
    x, C = conform(vol, affine, device, interp, MARGIN if margin is None else margin)
    with torch.inference_mode():
        y = net(x)[0, 0]
    if not torch.isfinite(y).all():
        raise RuntimeError(f"skullstrip: the network's output is not finite on {device} (grid {tuple(x.shape[2:])})")
    if float(y.max() - y.min()) < 1e-6:
        mps = torch.device(device).type == "mps"
        raise RuntimeError(f"skullstrip: the network returned a constant field on {device} (grid {tuple(x.shape[2:])}):"
                           " a backend fault, not a brain"
                           + (" (MPS returns zeros when its memory runs short) - try device='cpu'" if mps else ""))
    M = np.linalg.inv(C) @ np.asarray(affine, float)                      # input voxel -> conformed voxel
    return _sample(y, M, np.asarray(vol).shape, fill=OUTSIDE).cpu().numpy()
