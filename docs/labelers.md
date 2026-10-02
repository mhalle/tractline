# Labelers: a swappable labeling stage (design note, 2026-10-02)

The pipeline's last stage names the streamlines. Today that is TractCloud, run from upstream's
package with our resampling and CPU network around it (`tractline.labelers.tractcloud`). This note plans the stage as
an interface with several implementations: our own TractCloud (a rewrite, the default), upstream's
TractCloud (the reference), and later DeepMultiConnectome. It is written to move with the code into
its own repo; nothing here is built yet.

## Why

- **Dependencies.** Upstream TractCloud (SlicerDMRI/TractCloud 94de627) is pip-installable but requires
  `vtk`, which the pipeline never uses; today we put the data directory's copy on `sys.path` and stub
  `vtk` out. Our own implementation removes both; upstream becomes an optional extra.
- **Verification.** The tracker and the correction were checked against their originals (the Slicer
  UKF binary, FSL topup) by ad hoc scripts. Behind one interface, "ours against upstream, identical
  labels per draw on the cohort" is a standing test, and the same pattern can later hold the tracker's
  and the correction's references.
- **Other labelers.** Different labelers answer different questions (named tracts, connectome
  edges), and the outputs downstream (TRX, the cohort's measures, later the compact exporter) should not
  be written for one of them.

## Where the code is coupled today

- `trx.py` imports `TRACT_NAMES` and the cluster-to-tract `LUT` from `tractcloud` at module level
  and assumes 1,600 clusters and 43 classes.
- `bench/cohort.py` assumes Other is class 42 of 43.
- `Labels` (`keep`, `length_mm`, `tract`, `logp`) carries tract numbers but not the scheme they belong
  to. `pipeline.run` already takes the labeler as an argument.

## The interface

```python
@dataclass(frozen=True)
class Scheme:
    """What a label set's numbers mean."""
    name: str                          # "tract", "dk_pair", "destrieux_pair"
    kind: str                          # "tract": named classes; "pair": unordered pairs of nodes
    names: tuple[str, ...]             # tract names, or node names (kind "pair")
    other: int | None                  # the class meaning Other / unknown, if any
    fine: int | None = None            # fine classes under the labels (TractCloud: 1,600 clusters)
    fine_to_label: np.ndarray | None = None   # (fine,) -> label class
    # kind "pair": label class <-> (node i, node j), i <= j, by the labeler's own ordering (n(n+1)/2 classes)

@dataclass
class LabelSet:
    scheme: Scheme
    label: np.ndarray                  # (n_kept,) class per labeled streamline (a vote, if draws > 1)
    logp: torch.Tensor | None          # (n_kept, fine or classes) log-probabilities, when asked for

@dataclass
class Labels:
    keep: np.ndarray                   # (n_fibers,) bool: the streamlines labeled (the labeler's own cut)
    length_mm: np.ndarray              # (n_kept,)
    sets: dict[str, LabelSet]          # one per head: TractCloud {"tract"}, DeepMultiConnectome {"dk_pair", "destrieux_pair"}

class Labeler(Protocol):
    name: str                          # "tractcloud", "tractcloud-upstream", "deepmulticonnectome"
    schemes: dict[str, Scheme]
    space: str                         # the input it expects: "native" (TractCloud centers it itself) or "mni"
    stochastic: bool                   # TractCloud: random context draws; DeepMultiConnectome: none
    def __call__(self, fibers, *, to_space=None, draws=(0,), logp=False) -> Labels: ...
```

- **Draws** only for stochastic labelers; a deterministic labeler takes `draws=(0,)` and ignores it.
- **Space**: the labeler declares it; the pipeline supplies `to_space` (a transform of streamline
  points), so registration stays a pipeline stage, not a labeler's business.
- **Consumers read the scheme from the result.** TRX: a `tract` set becomes groups (one per name) and
  per-streamline arrays (`<set>`, `<set>_probability`, and `cluster` when `fine` is set); a `pair` set
  becomes per-streamline arrays only (13,695 groups would be noise), its node names in the header, and a
  connectome matrix (pair counts per scheme) as its own output. The cohort's tract measures apply to
  `tract` sets; the compact exporter (the rank field, in medseg for now) encodes any set's `logp`
  directly, not through the TRX.

## Implementations

### 1. Our TractCloud (`tractcloud`, the default)

Upstream's method and weights, our code:

| piece | upstream | ours |
|---|---|---|
| 15-point arc-length resampling | `extract_ras_features` | `resample.py`, already bit-identical in float64 |
| the network | `TractDGCNN`, `load_model` | `MatmulDGCNN` (1x1 convolutions with their batch norms folded into matrix products), weights loaded from upstream's state dict; to serve the GPU too |
| context per streamline: 20 neighbors among a random 10 % subsample, 80 random global streamlines | `RealDataDataset`, `_compute_local_features`, `tract_knn` (~60 lines) | to write: the same numpy draws in the same order, so labels match per draw; on the device |
| 42 tract names, 1,600 -> 43 map | `tract_mapping.py` | copied as data, credited |
| 40 mm cut, centering by HCP's mass center, vote over draws | (our `Labeler`) | kept |

Weights: upstream's `best_tract_f1_model.pth` and `cli_args.txt`, plus `HCP_mass_center.npy`,
fetched from a pinned location (to settle with the repo). License: upstream's (Slicer, permissive);
credit in the repo.

**Acceptance:** on the 12-patient cohort, per draw, labels identical to `tractcloud-upstream` in
float64 (the standard `resample_check.py` and `MatmulDGCNN` already meet); in float32, the agreement
reported (rounding can move near-ties); then switch the default and rerun the cohort.

### 2. Upstream TractCloud (`tractcloud-upstream`, the reference)

A thin adapter over the pip package (optional extra: brings `vtk`). Built first, so the interface
lands with no change in results: the cohort must reproduce exactly through it.

### 3. DeepMultiConnectome (`deepmulticonnectome`, planned, not now)

What we know from the format work (medseg's `dmc_field.py`, NOTES 2026-10-01 "The field at connectome
scale"): SlicerDMRI/DeepMultiConnectome 6c606ec, `models.pointnet.PointNetCls` with two log-softmax
heads; no context (k = k_global = 0), so deterministic; 15 points per streamline (our resampling);
heads of 3,655 classes (Desikan-Killiany, 85 nodes counting "unknown") and 13,695 (Destrieux, 165
nodes), n(n+1)/2 pairs (the paper says 3,571 and 13,631: it does not count "unknown"); rank fields of
both heads cost what TractCloud's does per streamline; inference 26,900 streamlines/s on the M2's GPU.

Its needs, beyond the interface:
- **Registration to MNI**: trained in MNI space; the pipeline has no template registration (the T1
  check aligns the scan to the T1, not the T1 to a template). A T1 -> MNI affine (or nonlinear) stage,
  composed with the existing scan -> T1 alignment, supplies `to_space`.
- **The pair ordering and length cut**: to read from upstream's code (`test_realdata.py`, its data
  preparation) before writing the scheme.
- **Reference**: upstream's `PointNetCls` on identical input, labels identical.
- **Accuracy is its own experiment**: trained on iFOD2 tractograms, not UKF; ground truth needs
  FreeSurfer parcellations (`aparc+aseg`, `aparc.a2009s+aseg`), which ds001226 does not have and which
  tumor brains would make doubtful. HCP has them (under its data-use agreement; ask before downloading).

## Order of work

1. The interface, with `tractcloud-upstream` behind it; consumers (`trx`, `bench/cohort.py`, later the exporter)
   read schemes from `Labels`. The cohort reproduces exactly.
2. Our `tractcloud`: context construction, the network on both devices, weights loading. Accepted
   against upstream per draw; becomes the default; cohort rerun.
3. Later: DeepMultiConnectome - registration to MNI, the adapter, then the accuracy question.
4. Perhaps: the same reference-behind-an-interface pattern for the tracker and the correction.
