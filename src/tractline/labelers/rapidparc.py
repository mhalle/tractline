"""RapidParc labels for a tractogram (Bisten, von Bornhaupt, ..., Schultz: "RapidParc: A Global-Context
Transformer for Parallel, Accurate, and Lesion-Robust Tractogram Parcellation", Imaging Neuroscience 2026;
github.com/MedVisBonn/RapidParc, BSD-3), its inference written here from its code (v1.0.4), the trained
weights its release v1.0.0 publishes (DATA/RapidParc, sha256-checked: model "rapidparc", or "hemiaug",
trained with one-sided augmentation for lesions and surgery).

The same 43-class scheme as TractCloud's (1,600 clusters -> 42 tracts + Other: RapidParc's mapping and
tract names are TractCloud's, checked identical). Per draw d:
  - the streamlines of 40 mm or more (tractline's cut, TractCloud's training data; RapidParc itself
    labels every streamline), as float32;
  - 15 points per streamline, evenly spaced by index (RapidParc's rule, not arc length - UKF's points
    are evenly spaced);
  - the whole set scaled to [-1, 1] per axis, shuffled (torch.Generator seeded with d), cut into groups
    of 2,000 (the last padded with the first streamlines of the shuffle); in the network each group is
    scaled to [-1, 1] again - its context and its frame are its group's, as in training;
  - the embedding (the 45 coordinates and a linear layer's 83, LeakyReLU), PyTorch's transformer encoder
    (8 layers, d_model 128, 1 head, feed-forward 256), a two-layer classifier to 1,600 logits.
Several draws average their cluster probabilities.

The grouping above ("index") is RapidParc's, and the default: a streamline's company depends on every other
streamline in the set, so a run whose seed set differs by a few streamlines labels every streamline in new
company (NOTES 2026-10-17). Two opt-in groupings take each streamline's seed key (ensemble.seed_keys; unique
among the labeled streamlines) and a 64-bit hash of (key, draw) instead of the shuffle. The hash (splitmix64)
is the same in any process or platform - as RapidParc's shuffle is for the same streamlines; what the keyed
groupings add is a company that does not depend on which other seeds exist. Labels still depend on the
network's float rounding (MPS against CPU, for instance):
  - "hash-sorted": the streamlines in hash order, cut every 2,000 and padded as above. Removing a streamline
    shifts each later group by one member, not a reshuffle;
  - "hash-bucket": group floor(u G), u the hash in [0, 1), G = ceil(n / 2,000). A group's members are its
    seeds whatever other seeds exist (as long as G is), in groups of about n / G (1,000-2,000) +- binomial, each run
    at its own size, one group a forward pass (the network is a transformer encoder over the group with no
    positional encoding: any group size fits; it was trained at 2,000).
Both skip the whole set's [-1, 1] scaling: each group's own scaling makes it a no-op up to float rounding,
and without it a group's input does not depend on the other groups' extent.
Dependencies: torch and numpy only (the weights'
safetensors file is read here; RapidParc's package pins pandas, scikit-learn, matplotlib and more).
Checked against RapidParc's package: bench/rapidparc_check.py.
"""
from __future__ import annotations

import json, struct
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Labels, MIN_LENGTH_MM, LUT, LogMean, empty, lengths
from ..data import DATA

WEIGHTS = DATA / "RapidParc"
MODELS = ("rapidparc", "hemiaug")
CONTEXT, POINTS = 2000, 15
GROUPINGS = ("index", "hash-sorted", "hash-bucket")      # index: RapidParc's shuffle (the default); the others need seed keys
SETTINGS = dict(num_layers=8, d_model=128, nhead=1, dim_feedforward=256, dropout=0.1, dim_class_hidden=256, dim_out=1600)


def read_safetensors(path) -> dict:
    """A .safetensors file's tensors (an 8-byte header length, a JSON header, raw little-endian arrays)."""
    dtypes = {"F32": np.float32, "F16": np.float16, "I64": np.int64}
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n]); head.pop("__metadata__", None)
    out = {}
    for k, v in head.items():
        a, b = v["data_offsets"]
        out[k] = torch.from_numpy(np.frombuffer(raw[8 + n + a:8 + n + b], dtype=dtypes[v["dtype"]]).reshape(v["shape"]).copy())
    return out


def normalize(x):
    """(bs, n, points, 3) streamlines scaled to [-1, 1] per axis over each batch item (RapidParc's
    normalize_to_identity_cube). An axis with no extent (a lone straight or planar streamline: RapidParc's
    0/0, NaN) divides by 1 instead; with every extent positive, as in any real tractogram, it is RapidParc's."""
    bs, _, _, dim = x.size()
    mins = torch.amin(x, dim=(1, 2)).view(bs, 1, 1, dim)
    maxs = torch.amax(x, dim=(1, 2)).view(bs, 1, 1, dim)
    extent = maxs - mins
    return 2 * (x - mins) / torch.where(extent > 0, extent, torch.ones_like(extent)) - 1.0


class Embedding(nn.Module):
    """RapidParc's EmbeddingAugFlipBatchGPU at inference (its flip augmentation is training's only)."""
    def __init__(self, points=POINTS, d_model=128):
        super().__init__()
        self.linear = nn.Linear(points * 3, d_model - points * 3)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        bs, n, _, _ = x.size()
        x = x.reshape(bs, n, -1)
        return torch.cat([x, self.activation(self.linear(x))], dim=2)


class Model(nn.Module):
    """RapidParc's TransformerModel, its parameter names (the weights load as they are)."""
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, dim_class_hidden, dim_out):
        super().__init__()
        self.embedding_layer = Embedding(POINTS, d_model)
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                                  dropout=dropout, batch_first=True)
        self.embeddingTransformer = nn.TransformerEncoder(encoder_layer=self.transformerEncoderLayer, num_layers=num_layers,
                                                          enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, dim_class_hidden), nn.ReLU(),
                                        nn.Linear(dim_class_hidden, dim_out))

    def forward(self, x):
        return self.classifier(self.embeddingTransformer(self.embedding_layer(normalize(x))))


def splitmix64(x) -> np.ndarray:
    """splitmix64's finalizer, elementwise on uint64 (wrapping arithmetic): a fixed 64-bit mix, no hash seed."""
    x = np.asarray(x, np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def key_hash(keys, draw) -> np.ndarray:
    """(n,) uint64 hash of each (seed key, draw): the same pair, the same hash, in any run."""
    return splitmix64(np.asarray(keys, np.int64).astype(np.uint64) ^ splitmix64(np.uint64(int(draw))))


def keyed_groups(keys, draw, grouping) -> list:
    """The context groups of a keyed grouping, as index arrays into keys (each in hash order, ties by key):
    "hash-sorted" CONTEXT-wide cuts of the hash order, the last padded from its start as RapidParc pads
    (pad rows are indices < n repeated); "hash-bucket" G = ceil(n / CONTEXT) buckets by the hash's top 53
    bits, the empty ones left out."""
    keys = np.asarray(keys, np.int64)
    n = len(keys)
    h = key_hash(keys, draw)
    order = np.lexsort((keys, h))
    if grouping == "hash-sorted":
        m = -(-n // CONTEXT) * CONTEXT
        return list(np.resize(order, m).reshape(-1, CONTEXT))             # np.resize repeats the order cyclically
    if grouping != "hash-bucket":
        raise ValueError(f"grouping {grouping!r}: one of {GROUPINGS}")
    G = -(-n // CONTEXT)
    b = ((h[order] >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 * G).astype(np.int64)
    return [g for g in np.split(order, np.searchsorted(b, np.arange(1, G))) if len(g)]


def resample(fibers, points=POINTS):
    """(n, points, 3) float32: each streamline's points at np.round(linspace(0, len - 1, points))."""
    return torch.from_numpy(np.stack([f[np.round(np.linspace(0, len(f) - 1, points)).astype(int)] for f in fibers]))


HASHES = {"rapidparc": "6a158dad6a0b6124946da102b6037a6f7f8a16fb24e7caec15eaf95213e7b793",      # RapidParc's package's
          "hemiaug": "b66dfce6135aac56c3f835264e5ae9630b2497859425ad98cdc6d842c6f94435"}        # pypi_package_helper.py


def _verified(path, model):
    import hashlib
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != HASHES[model]:
        raise ValueError(f"{path}: sha256 {digest} is not RapidParc v1.0.0's {model} weights ({HASHES[model]})")
    return path


class Labeler:
    def __init__(self, device="mps", model="rapidparc", batch=16):
        """model: "rapidparc" or "hemiaug" (DATA/RapidParc/<model>.safetensors, else tractline.fetch's cache,
        downloaded from RapidParc's release on first use; sha256-checked either way). batch:
        groups of 2,000 streamlines per forward pass (attention memory: ~16 MB a group a layer)."""
        if model not in MODELS:
            raise ValueError(f"RapidParc model {model!r}: one of {sorted(MODELS)}")
        self.device, self.model_name, self.batch = torch.device(device), model, batch
        self.model = Model(**SETTINGS)
        from ..fetch import weights                                       # the checkout's, or the cache's (downloaded once)
        self.model.load_state_dict(read_safetensors(_verified(weights(model), model)))
        self.model = self.model.eval().to(self.device)
        self.lut = LUT                                                     # RapidParc's mapping file, identical (base.py)

    @torch.inference_mode()
    def batches(self, feat, seed, keys=None, grouping="index"):
        """One draw, batch by batch: (original indices (m,), logits (m, 1600) float32 on the device). For
        grouping "index", the shuffle, the groups and the padding are RapidParc's; the padding repeats the
        shuffled streamlines as often as needed (RapidParc's takes the first CONTEXT - n % CONTEXT once - the
        same rows whenever n >= 1,000 - and fails below); padded rows are dropped here. The keyed groupings
        (keys: (n,) int64 seed keys): keyed_groups."""
        if grouping != "index":
            yield from self._keyed(feat, seed, keys, grouping)
            return
        n = len(feat)
        data = normalize(feat.unsqueeze(0)).squeeze(0)
        perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(int(seed)))
        data = data[perm]
        if n % CONTEXT:
            reps = -(-(CONTEXT - n % CONTEXT) // n) + 1
            data = torch.cat([data] * reps, dim=0)[: n + CONTEXT - n % CONTEXT]
        data = data.reshape(-1, CONTEXT, POINTS, 3)
        for a in range(0, len(data), self.batch):
            out = self.model(data[a:a + self.batch].to(self.device)).reshape(-1, 1600)
            r0 = a * CONTEXT
            real = max(0, min(n - r0, len(out)))                           # rows beyond n are padding
            yield perm[r0:r0 + real], out[:real]

    def _keyed(self, feat, seed, keys, grouping):
        """batches() for the keyed groupings: hash-sorted's equal groups `batch` at a time (pad rows dropped),
        hash-bucket's one group a forward pass (a group's logits then depend on its members alone)."""
        if grouping not in GROUPINGS:
            raise ValueError(f"grouping {grouping!r}: one of {GROUPINGS}")
        if keys is None or len(keys) != len(feat):
            raise ValueError(f"grouping {grouping!r} needs one seed key per streamline")
        n = len(feat)
        if len(np.unique(keys)) != n:
            raise ValueError(f"grouping {grouping!r}: the seed keys of the labeled streamlines must be unique "
                             f"({n - len(np.unique(keys))} repeated)")
        groups = keyed_groups(keys, seed, grouping)
        step = self.batch if grouping == "hash-sorted" else 1
        for a in range(0, len(groups), step):
            idx = np.concatenate(groups[a:a + step])
            out = self.model(feat[torch.from_numpy(idx)].reshape(-1, len(groups[a]), POINTS, 3).to(self.device)).reshape(-1, 1600)
            r0 = a * CONTEXT if grouping == "hash-sorted" else 0
            real = max(0, min(n - r0, len(out))) if grouping == "hash-sorted" else len(out)
            yield torch.from_numpy(idx[:real]), out[:real]

    def logits(self, feat, seed, keys=None, grouping="index"):
        """(n, 1600) logits (float32, CPU) for one draw - for checks against RapidParc's package."""
        out = torch.empty(len(feat), 1600)
        for idx, lg in self.batches(feat, seed, keys, grouping):
            out[idx] = lg.float().cpu()
        return out

    def __call__(self, fibers, draws=(0,), logp=False, keys=None, grouping="index") -> Labels:
        """fibers: (n, 3) RAS mm arrays. draws: the shuffle seeds (any iterable) whose cluster probabilities
        are averaged. logp: keep the cluster log-probabilities (float16; several draws: the log of their mean
        probability, without underflow). grouping: "index" (RapidParc's, the default), "hash-sorted" or
        "hash-bucket" (module docstring), the last two with keys: one seed key per fiber (ensemble.seed_keys;
        unique among the labeled ones)."""
        if grouping not in GROUPINGS:
            raise ValueError(f"grouping {grouping!r}: one of {GROUPINGS}")
        if grouping != "index" and (keys is None or len(keys) != len(fibers)):
            raise ValueError(f"grouping {grouping!r} needs keys, one per fiber")
        _, _, length = lengths(fibers)
        keep = length >= MIN_LENGTH_MM
        draws = tuple(draws)
        if not keep.any():
            return empty(keep, length, logp)
        feat = resample([np.asarray(f, np.float32) for f, k in zip(fibers, keep) if k])
        n = len(feat)
        kk = None if grouping == "index" else np.asarray(keys, np.int64)[keep]
        batches = lambda d: self.batches(feat, d, kk, grouping)
        if len(draws) == 1:
            arg = torch.empty(n, dtype=torch.int64)
            lp = torch.empty(n, 1600, dtype=torch.float16) if logp else None
            for idx, lg in batches(draws[0]):
                arg[idx] = lg.argmax(1).cpu()
                if logp:
                    lp[idx] = F.log_softmax(lg.float(), 1).half().cpu()
        else:
            mean = LogMean()
            for d in draws:
                full = torch.empty(n, 1600)
                for idx, lg in batches(d):
                    full[idx] = F.log_softmax(lg.float(), 1).cpu()
                mean.add(full)
            m = mean.result()
            arg, lp = m.argmax(1), (m.half() if logp else None)
        return Labels(keep=keep, length_mm=length[keep], tract=self.lut[arg.numpy()], logp=lp)
