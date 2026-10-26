"""The pipeline as a configuration: its slots, the implementations each slot can take, presets, and the config hash
(docs/sota-pipeline-design.md, phase A). Pure Python: no numpy, no torch, so a calibration map's tooling can read and
check a config without loading the pipeline.

    cfg = PipelineConfig.preset("default")                    # today's pipeline: ukf-rapidparc@1
    cfg = cfg.with_(model={"shell": 2000.0}, track={"Rs": 0.1}, denoise="mppca")
    cfg.hash()                                                # sha256 of what changes the outputs (None: not hashable)
    cfg.to_dict(); PipelineConfig.from_dict(d)                # what a run records; read back, checked again

Slots run in SLOTS' order (pipeline.execute). A slot is off (None) or one implementation with its options. Every
option is checked when the config is built, before any work: unknown slots, implementations, option keys and values
raise ValueError, as do implementations whose required slots are off or set otherwise.

The hash covers each slot that is on: its implementation, the implementation's version and every option, defaults
filled in, as canonical JSON. Slots that are off are left out, so a slot added later does not change the hash of a
config that leaves it off. Runtime choices (device, workers, batch sizes, the TRX file) are not part of a config;
pipeline.execute records them beside it. The device is left out on purpose (docs/sota-pipeline-design.md, decision
6), so a calibration map learned on one device is meant to apply on another. Metal, Triton and the CPU's fast step are
different kernels, and on a real scan they are not close per streamline. On cIIIs04, the CPU against MPS: 93.8 % of the
shared seeds get the same tract, 31 % of the paths end more than 1 mm apart (bench/results/device_agreement_cIIIs04.json).
That is about a fifth of the ensemble's own spread: a bootstrap replicate against A gives 68 % and 97 %. At the
ensemble level, where a map acts, the difference goes away: cIIIs04's 17-run ensemble on the CPU, with the cohort III
map, covers its rescans within 0.003 of the MPS ensemble at every target (bench/device_ensemble.py; one subject, CUDA not
run). tests/test_devices.py checks only that the devices run the same pipeline.
An implementation that cannot be described by its options (a labeler object
the caller made) makes the config unhashable: hash() returns None, and a calibration map refuses it.

Presets are named by what they contain, with a version that never changes once released: any change, even to an
option's default or an implementation's version, is a new number. "default" is an alias, moved only by the owner.
"""
from __future__ import annotations

import hashlib, json, math, numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA = "tractline.config/1"

#: the slots, in the order they run. The design's later slots (bias, anatomy, register_t1, tissue, filter,
#: tractometry) are added here as their implementations arrive; a slot is off unless a config sets it.
SLOTS = ("denoise", "gibbs", "field", "motion_eddy", "mask", "model", "track", "bundles")

#: an implementation's status: "wrapper" runs the established tool itself; "experimental" is ours, not yet through both
#: validation levels; "validated" is ours and through them (docs/sota-pipeline-design.md, section 5)
STATUSES = ("wrapper", "experimental", "validated")


@dataclass(frozen=True)
class Option:
    """One setting of an implementation: its default, its type, and (optionally) the values or range allowed."""
    default: Any
    kind: type                                 # bool, int, float or str
    choices: tuple | None = None
    low: float | None = None                   # inclusive bounds, for int and float
    high: float | None = None
    doc: str = ""

    def check(self, where: str, value):
        """value, normalized (an int given for a float option becomes a float, so the hash does not depend on it)."""
        if self.kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{where}: {value!r} is not a bool")
        elif self.kind is int:                             # numpy's integers too (numbers.Integral), never a bool
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{where}: {value!r} is not an int")
            value = int(value)
        elif self.kind is float:                           # numpy's floats too (numbers.Real), never a bool
            if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError(f"{where}: {value!r} is not a finite number")
            value = float(value)
        elif self.kind is str:
            if not isinstance(value, str):
                raise ValueError(f"{where}: {value!r} is not a string")
        else:
            raise TypeError(f"{where}: option kind {self.kind!r} is not supported")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{where}: {value!r} is not one of {self.choices}")
        if self.low is not None and value < self.low:
            raise ValueError(f"{where}: {value!r} is below {self.low}")
        if self.high is not None and value > self.high:
            raise ValueError(f"{where}: {value!r} is above {self.high}")
        return value


@dataclass(frozen=True)
class Implementation:
    """What a slot can run: its name, version (bumped whenever its outputs change: part of the hash), options, the
    other slots it needs (slot -> the implementations allowed there; an empty tuple: any, as long as it is on), and
    what is known about it. The code that runs it is pipeline.execute's, keyed by (slot, name), so this module stays
    free of numpy and torch."""
    slot: str
    name: str
    version: str
    options: Mapping[str, Option] = field(default_factory=dict)
    requires: Mapping[str, tuple] = field(default_factory=dict)
    devices: tuple = ("mps", "cuda", "cpu")
    stochastic: bool = False                   # draws random numbers: the run context gives it a seed
    scope: str = "replicate"                   # "replicate": rerun per ensemble replicate; "scan": once per scan
    oracle: str | None = None                  # the established tool it is validated against
    status: str = "experimental"
    hashable: bool = True                      # False: its options do not determine its outputs (a caller's object)
    doc: str = ""

    def resolve(self, options: Mapping | None) -> dict:
        """The options checked, with defaults filled in."""
        options = dict(options or {})
        unknown = set(options) - set(self.options)
        if unknown:
            raise ValueError(f"{self.slot}:{self.name}: {sorted(unknown)} not among its options {sorted(self.options)}")
        return {k: o.check(f"{self.slot}:{self.name} option {k}", options.get(k, o.default))
                for k, o in sorted(self.options.items())}


_REGISTRY: dict[tuple[str, str], Implementation] = {}


def register(impl: Implementation) -> Implementation:
    if impl.slot not in SLOTS:
        raise ValueError(f"register: slot {impl.slot!r} is not one of {SLOTS}")
    if impl.status not in STATUSES or impl.scope not in ("replicate", "scan"):
        raise ValueError(f"register: {impl.slot}:{impl.name}: status {impl.status!r} or scope {impl.scope!r}")
    for k, o in impl.options.items():                                  # each default must pass its own check
        o.check(f"register: {impl.slot}:{impl.name} option {k}'s default", o.default)
    key = (impl.slot, impl.name)
    if key in _REGISTRY and _REGISTRY[key] != impl:
        raise ValueError(f"register: {impl.slot}:{impl.name} is already registered, differently")
    _REGISTRY[key] = impl
    return impl


def implementation(slot: str, name: str) -> Implementation:
    if slot not in SLOTS:
        raise ValueError(f"slot {slot!r} is not one of {SLOTS}")
    try:
        return _REGISTRY[(slot, name)]
    except KeyError:
        raise ValueError(f"{slot}: {name!r} is not one of {implementations(slot)}") from None


def implementations(slot: str) -> tuple:
    return tuple(sorted(n for s, n in _REGISTRY if s == slot))


# ------------------------------------------------------------------------------------------------- today's stages

_DENOISE_ORDER = Option("acquired", str, ("acquired", "corrected"),
                        doc="acquired: before the field (the default); corrected: after it, for comparison")
register(Implementation(
    "denoise", "mppca", "1", {"patch_radius": Option(2, int, low=1, high=10, doc="MP-PCA's patch radius"),
                              "order": _DENOISE_ORDER},
    devices=("cpu",), oracle="dipy.denoise.localpca.mppca", status="wrapper",
    doc="dipy's MP-PCA in a process of its own (tractline.denoise)"))
register(Implementation(
    "denoise", "patch2self", "1", {"sketch_seed": Option(0, int, low=0, doc="the count sketch's seed"),
                                   "order": _DENOISE_ORDER},
    devices=("cpu",), oracle="dipy.denoise.patch2self", status="wrapper",
    doc="dipy's Patch2Self (version 3) in a process of its own (tractline.denoise)"))
register(Implementation(
    "gibbs", "subvoxel-shift", "1",
    {"slice_axis": Option(2, int, low=0, high=2, doc="the axis across which the scan's 2D slices were acquired"),
     "n_points": Option(3, int, low=1, high=10, doc="the neighbors whose oscillation the shift minimizes")},
    oracle="DIPY gibbs_removal (Kellner et al. 2016)", status="experimental",
    doc="Gibbs ringing removal by local subvoxel shifts (tractline.gibbs) on the scan as acquired (and the b0s that feed "
        "the field on their own grid), after denoising and before the field. DIPY's method and choices: in float64 its "
        "output to rounding but for shifts tied to rounding; the pipeline runs float32, where near-ties fall either "
        "way, as in DIPY's own float32 (bench/gibbs_check.py). Assumes a full, symmetric k-space: not for partial "
        "Fourier acquisitions without care"))
register(Implementation(
    "field", "tractline", "1", {}, oracle="FSL topup", status="validated",
    doc="topup's model by Gauss-Newton (susceptibility.estimate), applied with the Jacobian (susceptibility.apply); runs "
        "when the subject has b0s from both polarities and a pairing that passed (pipeline.can_correct), else the DWI "
        "as acquired, with the reason"))
register(Implementation(
    "motion_eddy", "tractline", "1",
    {"model": Option("quadratic", str, choices=("rigid", "linear", "quadratic"),
                     doc="the eddy-current polynomial along PE: none, linear (4 terms) or quadratic (10; eddy's --flm)"),
     "steps": Option(40, int, low=1, high=500, doc="Gauss-Newton steps (the cap when tol > 0)"),
     "tol": Option(0.0, float, low=0.0, doc="stop when no volume's step moves an edge point more than this (voxels)"),
     "sh_order": Option(6, int, choices=(0, 2, 4, 6, 8, 10, 12), doc="the per-shell predictor's SH order (even)"),
     "sh_lambda": Option(0.06, float, low=0.0, doc="its Laplace-Beltrami weight"),
     "b0_ec": Option(False, bool, doc="the b0s fit the PE polynomial too (motion-by-susceptibility)"),
     "anchor": Option("none", str, choices=("none", "mi"), doc="each DW shell's corrected mean aligned to the mean b0"),
     "mask_output": Option(True, bool, doc="zero voxels outside some volume's field of view (eddy's output mask)")},
    oracle="FSL eddy (Andersson & Sotiropoulos 2016)", status="experimental",
    doc="head motion and eddy currents per volume (tractline.motion_eddy): rigid + a polynomial along PE, each volume "
        "registered to a leave-one-out prediction from the others; the field (when the field slot corrects) composed "
        "into the same map, so each volume is resampled once; the b-vectors rotated with the head. Runs without a "
        "reversed pair too (motion and eddy only, as eddy without topup)"))
register(Implementation(
    "mask", "median-otsu", "1", {}, oracle="dipy.segment.mask.median_otsu", status="validated",
    doc="median_otsu of the mean b0 (median radius 4, 4 passes), dipy's exactly (tractline.mask)"))
register(Implementation(
    "model", "ukf-two-tensor", "1", {"shell": Option(2800.0, float, low=1.0, doc="the b-value tracked, with the b0s")},
    status="validated",
    doc="the tracker's input: the b0s and one shell, gradients in RAS (prep.prepare); the two-tensor model itself is "
        "the UKF's, fitted as it tracks"))
register(Implementation(
    "track", "ukf", "1",
    {"seeding_threshold": Option(0.1, float, low=0.0), "stopping_fa": Option(0.08, float, low=0.0),
     "stopping_threshold": Option(0.06, float, low=0.0), "step_length": Option(0.3, float, low=1e-3),
     "record_length": Option(1.8, float, low=1e-3), "max_half_length": Option(250.0, float, low=1.0),
     "Qm": Option(0.001, float, low=0.0), "Ql": Option(50.0, float, low=0.0), "Rs": Option(0.02, float, low=0.0),
     "tie_angle": Option(0.0, float, low=0.0), "tie_sw2": Option(False, bool)},
    requires={"model": ("ukf-two-tensor",), "mask": ()}, oracle="Slicer UKFTractography", status="validated",
    doc="UKF two-tensor tractography with the ORG settings and the binary's seeds (ukf.track; Metal on mps, Triton on "
        "cuda, the fast float32 step on the CPU)"))
register(Implementation(
    "model", "csd", "1",
    {"shell": Option(2800.0, float, low=1.0, doc="the b-value fitted, with the b0s"),
     "order": Option("auto", str, ("auto", "6", "8"), doc="the SH order; auto: 8 from 45 directions, else 6 (as DIPY's "
                                                        "benchmark chose)"),
     "lambda": Option(1.0, float, low=0.0, doc="the constraint's weight (Tournier 2012's scaling)"),
     "tau": Option(0.1, float, low=0.0, high=1.0, doc="the constraint's threshold, a share of the FOD's mean"),
     "fa_threshold": Option(0.7, float, low=0.0, high=1.0, doc="the response's voxels: FA above this"),
     "roi_radius": Option(10, int, low=1, doc="the response's voxels: within this many of the volume's center")},
    requires={"mask": ()}, oracle="DIPY ConstrainedSphericalDeconvModel", status="experimental",
    doc="single-shell CSD (tractline.csd): the response from the data, Tournier 2007's constrained fit on our own "
        "181-axis constraint hemisphere; equal to DIPY's on DIPY's sphere (bench/csd_check.py)"))
register(Implementation(
    "track", "ptt", "1",
    {"step_size": Option(0.5, float, low=0.01, doc="mm"),
     "max_angle": Option(20.0, float, low=1.0, high=90.0, doc="degrees per step, real units (at 2 mm voxels, DIPY 1.12's "
                                                              "stated 10; NOTES 2026-10-25)"),
     "probe_length": Option(1.5, float, low=0.1, doc="mm"),
     "probe_quality": Option(7, int, low=3, high=20, doc="points along the probe"),
     "pmf_threshold": Option(0.1, float, low=0.0),
     "min_len": Option(20.0, float, low=0.0, doc="mm"),
     "max_len": Option(250.0, float, low=1.0, doc="mm"),
     "stopping": Option("act", str, ("act", "mask"), doc="act: the anatomical constraint from tissue maps the caller gives "
                                                        "execute (tissue=; GM and background end validly, CSF invalidly); "
                                                        "mask: leaving the brain mask ends validly"),
     "seeding": Option("wm", str, ("wm", "fa"), doc="wm: uniform in the voxels with WM >= 0.5 (tissue maps); fa: uniform "
                                                    "in the mask's voxels with FA >= fa_seed"),
     "fa_seed": Option(0.2, float, low=0.0, high=1.0),
     "seed_count": Option(100_000, int, low=1),
     "seed_rng": Option(1, int, low=0, doc="the seed points' generator"),
     "track_rng": Option(1, int, low=0, doc="the tracking's random numbers")},
    requires={"model": ("csd",), "mask": ()}, oracle="DIPY ptt_tracking (with its units: ptt's validation switch)",
    status="experimental",
    doc="parallel transport tractography with ACT (tractline.ptt; the Metal kernel on mps, the torch prototype "
        "elsewhere), real units; the PMF on our own 362 directions (hemisphere(181) and its antipodes)"))
register(Implementation(
    "bundles", "rapidparc", "1",
    {"model": Option("rapidparc", str, ("rapidparc", "hemiaug"), doc="the released weights' name (rapidparc.MODELS)"),
     "draw": Option(0, int, low=0, doc="the shuffle seed of the one draw labeled"),
     "grouping": Option("index", str, ("index", "hash-sorted", "hash-bucket"))},
    oracle="RapidParc", status="validated",
    doc="RapidParc, one draw (labelers.rapidparc); grouping other than index needs seed keys (ensemble.seed_keys)"))
register(Implementation(
    "bundles", "tractcloud", "1",
    {"upstream": Option(False, bool, doc="upstream's inference context (k_global 80) instead of the trained one"),
     "exact": Option("yes", str, ("yes", "no"), doc="yes: upstream's forward; no: MatmulDGCNN (the labeler's own default "
                     "on the CPU). No 'auto': the hash must name the forward that ran, whatever the device"),
     "draw": Option(0, int, low=0)},
    oracle="TractCloud", status="validated", doc="TractCloud (labelers.tractcloud)"))
register(Implementation(
    "bundles", "caller-object", "1", {"class": Option("", str, doc="the object's class, for the record")},
    hashable=False, status="experimental",
    doc="a labeler object the caller made (run()'s labeler=): run as given; its settings are unknown here"))


#: released presets: read-only, never edited - a change is a new number
PRESETS = MappingProxyType({
    "ukf-rapidparc@1": MappingProxyType({"field": "tractline", "mask": "median-otsu", "model": "ukf-two-tensor",
                                         "track": "ukf", "bundles": "rapidparc"}),
})
ALIASES = MappingProxyType({"default": "ukf-rapidparc@1"})


# --------------------------------------------------------------------------------------------------- the config

def _origin(stages, base: str) -> str:
    """The origin to give these stages: the name of a preset they equal (base's first), else base + " + changes"."""
    for name in ([base] if base in PRESETS else []) + sorted(PRESETS):
        if PipelineConfig.preset(name).stages == stages:
            return name
    return f"{base} + changes" if base else ""


@dataclass(frozen=True)
class Stage:
    impl: str
    options: tuple                             # ((key, value), ...) sorted, defaults filled in

    def opts(self) -> dict:
        return dict(self.options)


def _entry(slot, value, current: Stage | None) -> Stage | None:
    """A slot's setting as given -> a checked Stage. value: None (off); a name (that implementation, its defaults); a
    dict with "impl" (that implementation with these options) or without (these options merged into the current
    implementation's)."""
    if value is None:
        return None
    if isinstance(value, Stage):
        value = {"impl": value.impl, **value.opts()}
    if isinstance(value, str):
        name, options = value, {}
    elif isinstance(value, Mapping):
        options = dict(value)
        name = options.pop("impl", None)
        if name is None:
            if current is None:
                raise ValueError(f"{slot}: options {sorted(options)} given, but the slot is off: name an implementation "
                                 "({'impl': ..., ...})")
            name, options = current.impl, {**current.opts(), **options}
    else:
        raise ValueError(f"{slot}: {value!r} is not None, an implementation's name or a dict")
    impl = implementation(slot, name)
    return Stage(name, tuple(impl.resolve(options).items()))


@dataclass(frozen=True)
class PipelineConfig:
    """One entry per slot (SLOTS), each None (off) or a Stage. Build it with preset(), from_dict() or with_(); the
    constructor checks every entry and every requirement."""
    stages: tuple                              # ((slot, Stage | None), ...) in SLOTS' order
    origin: str = field(default="", compare=False)   # the preset it came from, "+ changes" when changed: for people;
                                                      # neither hashed nor compared

    def __post_init__(self):
        slots = tuple(s for s, _ in self.stages)
        if slots != SLOTS:
            raise ValueError(f"PipelineConfig: slots {slots} are not {SLOTS}")
        # every Stage normalized as its implementation resolves it (checked, defaults filled in, an int for a float
        # option made a float, sorted), however it was built, so equal settings always compare and hash equal
        object.__setattr__(self, "stages", tuple(
            (slot, None if st is None else Stage(st.impl, tuple(implementation(slot, st.impl).resolve(st.opts()).items())))
            for slot, st in self.stages))
        for slot, st in self.stages:
            if st is None:
                continue
            for need, allowed in implementation(slot, st.impl).requires.items():
                other = self[need]
                if other is None:
                    raise ValueError(f"{slot}:{st.impl} requires the {need} slot, which is off")
                if allowed and other.impl not in allowed:
                    raise ValueError(f"{slot}:{st.impl} requires {need} to be one of {allowed}, not {other.impl!r}")
        dn = self["denoise"]
        if self["gibbs"] is not None and dn is not None and dn.opts().get("order") == "corrected":
            # Gibbs removal smooths the noise and correlates it between neighbors, against both denoisers' assumptions:
            # denoise first (MRtrix's order), or not at all
            raise ValueError("gibbs runs on the series as acquired, so denoising must come before it: denoise order "
                             "'acquired', not 'corrected'")

    def __getitem__(self, slot) -> Stage | None:
        for s, st in self.stages:
            if s == slot:
                return st
        raise KeyError(slot)

    @classmethod
    def build(cls, entries: Mapping, origin="") -> "PipelineConfig":
        unknown = set(entries) - set(SLOTS)
        if unknown:
            raise ValueError(f"PipelineConfig: {sorted(unknown)} not among the slots {SLOTS}")
        return cls(tuple((s, _entry(s, entries.get(s), None)) for s in SLOTS), origin)

    @classmethod
    def preset(cls, name: str) -> "PipelineConfig":
        full = ALIASES.get(name, name)
        if full not in PRESETS:
            raise ValueError(f"preset {name!r} is not one of {sorted(PRESETS) + sorted(ALIASES)}")
        return cls.build(PRESETS[full], origin=full)

    def with_(self, **changes) -> "PipelineConfig":
        """A copy with these slots changed: slot=None (off), slot="name", slot={"impl": "name", **options} or
        slot={**options} (merged into the slot's current implementation)."""
        unknown = set(changes) - set(SLOTS)
        if unknown:
            raise ValueError(f"with_: {sorted(unknown)} not among the slots {SLOTS}")
        stages = tuple((s, _entry(s, changes[s], st) if s in changes else st) for s, st in self.stages)
        if stages == self.stages:
            return self
        return PipelineConfig(stages, _origin(stages, self.origin.removesuffix(" + changes")))

    def is_on(self, slot) -> bool:
        return self[slot] is not None

    def opts(self, slot) -> dict:
        st = self[slot]
        return {} if st is None else st.opts()

    def canonical(self) -> dict:
        """What changes the outputs: the slots that are on, each {impl, version, options}."""
        return {"schema": SCHEMA, "slots": {s: {"impl": st.impl, "version": implementation(s, st.impl).version,
                                                "options": st.opts()} for s, st in self.stages if st is not None}}

    def hash(self) -> str | None:
        """sha256 of canonical() as sorted, compact JSON; None when a slot's implementation is not hashable."""
        if any(st is not None and not implementation(s, st.impl).hashable for s, st in self.stages):
            return None
        text = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()

    def to_dict(self) -> dict:
        """canonical() with the origin and the hash: what a run records."""
        return {**self.canonical(), "origin": self.origin, "sha256": self.hash()}

    @classmethod
    def from_dict(cls, d: Mapping) -> "PipelineConfig":
        """Back from to_dict (or canonical). Checked again; a recorded version that differs from the registered one
        raises (the implementation has changed since the record was made), as does a recorded hash that differs."""
        if d.get("schema") != SCHEMA:
            raise ValueError(f"from_dict: schema {d.get('schema')!r} is not {SCHEMA!r}")
        slots = d.get("slots", {})
        for s, e in slots.items():
            if not isinstance(e, Mapping) or "impl" not in e or "version" not in e:
                raise ValueError(f"from_dict: the {s} slot's entry needs impl and version: {e!r}")
            v = implementation(s, e["impl"]).version
            if e["version"] != v:
                raise ValueError(f"from_dict: {s}:{e['impl']} was version {e['version']}, now {v}")
        cfg = cls.build({s: {"impl": e["impl"], **e.get("options", {})} for s, e in slots.items()})
        if "sha256" in d and d["sha256"] != cfg.hash():
            raise ValueError("from_dict: the recorded hash does not match the config")
        # the origin is not taken on trust: a preset's name only when the stages are that preset's
        return PipelineConfig(cfg.stages, _origin(cfg.stages, "") or "(from a record)")

    def describe(self) -> str:
        """One line per slot, for logs."""
        lines = [f"config {self.origin or '(custom)'}  sha256 {self.hash() or 'none (not hashable)'}"]
        for s, st in self.stages:
            lines.append(f"  {s:8s} " + ("off" if st is None else
                                          st.impl + "".join(f" {k}={v}" for k, v in st.options)))
        return "\n".join(lines)
