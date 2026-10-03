"""DIPY's median_otsu, exactly, in torch alone (no numba, no dipy): the same four passes of
scipy.ndimage.median_filter (a 9^3 box, mode "reflect": d c b a | a b c d | d c b a) and the same Otsu
threshold, on the CPU or the GPU.

  - each pass: the volume padded by reflection (index arithmetic, scipy's "reflect"), every voxel's
    9^3 window gathered as a view (unfold), its median taken in chunks of slabs - the 365th of 729
    values, one of the window's own, so the result is exact whatever the device;
  - otsu: Otsu's threshold written here, with DIPY's conventions (bin values, the threshold's edge),
    numpy, 256 bins.
Checked voxel for voxel against dipy.segment.mask.median_otsu (median_check.py). ~3 s on the M2,
CPU or GPU; numba's sliding-window histogram did it in 0.2 s but cost a 137 MB dependency that pins
numpy (NOTES 2026-10-02).
"""
import numpy as np, torch


def _reflect(n, r, device):
    i = np.arange(-r, n + r)
    i = np.where(i < 0, -i - 1, i)
    return torch.as_tensor(np.where(i > n - 1, 2 * n - 1 - i, i), device=device)


def median3(v: torch.Tensor, radius=4, chunk=8) -> torch.Tensor:
    """scipy.ndimage.median_filter(v, size=2 radius + 1, mode="reflect") for a 3D tensor, exactly."""
    w = 2 * radius + 1
    P = v
    for ax, n in enumerate(v.shape):
        P = P.index_select(ax, _reflect(n, radius, v.device))
    U = P.unfold(0, w, 1).unfold(1, w, 1).unfold(2, w, 1)                # (X, Y, Z, w, w, w), a view
    out = torch.empty_like(v)
    X, Y, Z = v.shape
    for a in range(0, X, chunk):
        b = min(X, a + chunk)
        out[a:b] = U[a:b].reshape(b - a, Y, Z, -1).median(dim=-1).values
    return out


def otsu(image, nbins=256):
    """Otsu's threshold (Otsu 1979): split the histogram into a lower and an upper class where the
    between-class variance, n0 n1 (mu0 - mu1)^2 up to a constant, is largest. The conventions are
    DIPY's median_otsu's, so the mask matches it: nbins equal bins over the image's range, each bin
    standing for its upper edge, and the threshold the lower edge of the lower class's last bin."""
    counts, edges = np.histogram(image, bins=nbins)
    n = counts.astype(np.float64)
    mass = n * edges[1:]                                                  # count x value, per bin
    # splits k = 0 .. nbins - 2: the lower class is bins 0..k, the upper bins k+1..nbins-1
    n0, s0 = np.cumsum(n)[:-1], np.cumsum(mass)[:-1]
    n1, s1 = np.cumsum(n[::-1])[::-1][1:], np.cumsum(mass[::-1])[::-1][1:]
    between = n0 * n1 * (s0 / n0 - s1 / n1) ** 2                          # every class holds a voxel: the range's
    return edges[np.argmax(between)]                                      # ends are the first and last bins'


def multi_median(vol, radius=4, passes=4, device="cpu"):
    """scipy.ndimage.median_filter(vol, size=2 radius + 1, mode="reflect"), `passes` times, exactly."""
    v = torch.as_tensor(np.ascontiguousarray(vol), device=device)
    for _ in range(passes):
        v = median3(v, radius)
    return v.cpu().numpy()


def median_otsu(b0, median_radius=4, numpass=4, device="cpu"):
    """(masked b0, mask) as dipy.segment.mask.median_otsu(b0, median_radius=, numpass=) for a 3D volume.
    device: where the median passes run ("cpu", "mps", "cuda"); the result is the same."""
    m = multi_median(b0, median_radius, numpass, device)
    mask = m > otsu(m)
    return b0 * mask, mask
