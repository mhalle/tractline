"""RapidParc labels for a tractogram (von Bornhaupt, Bisten, ..., Schultz: "RapidParc: A Global-Context
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
Several draws average their cluster probabilities. Dependencies: torch and numpy only (the weights'
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
    normalize_to_identity_cube)."""
    bs, _, _, dim = x.size()
    mins = torch.amin(x, dim=(1, 2)).view(bs, 1, 1, dim)
    maxs = torch.amax(x, dim=(1, 2)).view(bs, 1, 1, dim)
    return 2 * (x - mins) / (maxs - mins) - 1.0


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
        """model: "rapidparc" or "hemiaug" (DATA/RapidParc/<model>.safetensors, sha256-checked). batch:
        groups of 2,000 streamlines per forward pass (attention memory: ~16 MB a group a layer)."""
        assert model in MODELS, model
        self.device, self.model_name, self.batch = torch.device(device), model, batch
        self.model = Model(**SETTINGS)
        self.model.load_state_dict(read_safetensors(_verified(WEIGHTS / f"{model}.safetensors", model)))
        self.model = self.model.eval().to(self.device)
        self.lut = LUT                                                     # RapidParc's mapping file, identical (base.py)

    @torch.inference_mode()
    def batches(self, feat, seed):
        """One draw, batch by batch: (original indices (m,), logits (m, 1600) float32 on the device). The
        shuffle, the groups and the padding are RapidParc's; the padding repeats the shuffled streamlines
        as often as needed (RapidParc's takes the first CONTEXT - n % CONTEXT once - the same rows whenever
        n >= 1,000 - and fails below); padded rows are dropped here."""
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

    def logits(self, feat, seed):
        """(n, 1600) logits (float32, CPU) for one draw - for checks against RapidParc's package."""
        out = torch.empty(len(feat), 1600)
        for idx, lg in self.batches(feat, seed):
            out[idx] = lg.float().cpu()
        return out

    def __call__(self, fibers, draws=(0,), logp=False) -> Labels:
        """fibers: (n, 3) RAS mm arrays. draws: the shuffle seeds (any iterable) whose cluster probabilities
        are averaged. logp: keep the cluster log-probabilities (float16; several draws: the log of their mean
        probability, without underflow)."""
        _, _, length = lengths(fibers)
        keep = length >= MIN_LENGTH_MM
        draws = tuple(draws)
        if not keep.any():
            return empty(keep, length)
        feat = resample([np.asarray(f, np.float32) for f, k in zip(fibers, keep) if k])
        n = len(feat)
        if len(draws) == 1:
            arg = torch.empty(n, dtype=torch.int64)
            lp = torch.empty(n, 1600, dtype=torch.float16) if logp else None
            for idx, lg in self.batches(feat, draws[0]):
                arg[idx] = lg.argmax(1).cpu()
                if logp:
                    lp[idx] = F.log_softmax(lg.float(), 1).half().cpu()
        else:
            mean = LogMean()
            for d in draws:
                full = torch.empty(n, 1600)
                for idx, lg in self.batches(feat, d):
                    full[idx] = F.log_softmax(lg.float(), 1).cpu()
                mean.add(full)
            m = mean.result()
            arg, lp = m.argmax(1), (m.half() if logp else None)
        return Labels(keep=keep, length_mm=length[keep], tract=self.lut[arg.numpy()], logp=lp)
