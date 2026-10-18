"""The runs behind the calibration card (docs/robustness-plan.md, step 5; bench/calibration_card.py fits the card).

    uv run --group dipy --group rankfield bench/calibration_runs.py plan
    uv run --group dipy --group rankfield bench/calibration_runs.py run --subject cIIIs03 [--runs 16] [--tags within]
    uv run --group dipy --group rankfield bench/calibration_runs.py rescans --subject cIIIs02 --tags session4 session5

Data: MASiVar cohort III on scanner B (OpenNeuro ds003416; bench/fetch_ds003416.py, bench/_masivar.py), one protocol:
b1000 x40 + b2000 x56 (each shell its own series, 16 interleaved b0s in all), 2.1 x 2.1 x 2.2 mm, a reversed "APA"
series (one b0). Only scans that match PROTOCOL are used (load_checked: volumes per shell, matrix, voxel size, phase
encoding, one reversed b0 at the expected brightness). CARD below is the one place the eight subjects are defined: per subject the reference scan A, its
same-session rescan and its other-session rescans (MASiVar's scan numbers: run-<scan><acquisition>).

run, per subject (resumable; what exists is skipped):
  A              the reference scan through today's default pipeline (P.correct, P.track at the 2000 shell, the
                 default labeler, draw 0);
  <tag>_onA      each rescan corrected on its own grid, its corrected mean b0 registered rigidly to A's
                 (ensemble_retest.register: dipy, mutual information, in a process of its own), its corrected DWI
                 resampled onto A's grid with the b-vectors turned (ensemble_retest.onto_reference) and tracked there,
                 as ensemble_retest.py's <tag>_onA: its seeds are A's voxels, so seed keys match;
  smooth/r001..  `runs` replicates of A (_bootstrap.ShellWildBootstrap: smooth signs, FWHM 4, b0s "added", A's mask),
                 each corrected, tracked and labeled as A, as ensemble_retest.py run --flip smooth makes them. A and 16
                 replicates are the 17-run ensemble.
rescans (optional; the pooled reference): `runs` replicates of a rescan as acquired (its own mask), each corrected on
  its own grid, resampled onto A's and tracked there: smooth/<tag>_r001.., as ensemble_retest.py rescans.
plan: the pair table, what is on disk, and the runs a full queue still needs (--check: every scan against PROTOCOL).
verify: the reference scan rerun at HEAD and compared with its stored A.npz (keys, clusters, voxel counts, vertices to
  the compact quantization); for a reused record, the evidence that the default path has not changed.

Records: compact (format 2), in DATA/ensemble/<subject>/card/<pair>/ with runs.json beside them (the pipeline's git
commit and settings, each run's log, each rescan's registration). Where CARD names an earlier record folder (`reuse`),
its records are read and never written: a run present there is not repeated. Nothing is ever overwritten (a record's
path is refused if it exists in either folder; records are written under a temporary name and renamed when
complete), and every run first checks the free disk (stops below MIN_FREE_GB).

The choices, from the data on disk (bench/fetch_ds003416.py):
  - cIIIs01's earlier 2.1 mm records (retest/cIIIs01-2.1, 2026-10-08) are format 1: no streamlines, so no density
    presence, and 8 smooth replicates. Its scans are rerun here. Its 2.1 mm scans on disk are s1Bx1 scan 1 and
    s1Bx3 scans 1 and 2, so A is s1Bx3 scan 1 (as that pair's), the same-session rescan s1Bx3 scan 2, and s1Bx1 scan 1
    the other session (s1Bx1 scan 3 and s1Bx4-7 exist in the dataset and are not fetched);
  - cIIIs02 s1Bx5 scan 1's b2000 series is 2.2 mm iso, so that session's rescan is its scan 2;
  - cIIIs04 reuses retest/cIIIs04-2.1 (A, smooth/r001..r016, within_onA, session3_onA, session5_onA, and the three
    rescans' own 16 replicates, all compact, today's default pipeline), and adds sessions 2 and 4.
"""
import argparse, json, os, shutil, subprocess, sys, time
from dataclasses import dataclass, field, replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import register, onto_reference, mean_b0  # noqa: E402

DEVICE = "mps"
SHELLS, SHELL = None, 2000.0    # every shell joined; the tracker's shell (as ensemble_retest's cIIIs04-2.1)
FLIP, FWHM = "smooth", 4.0      # the bootstrap's signs: smoothed in-plane, FWHM 4 voxels (ensemble_retest --flip smooth)
RUNS = 16                       # replicates: A and 16 make the 17-run ensemble
MIN_FREE_GB = 20.0
PROTOCOL = {                    # what load_checked checks: a scan is this protocol only if every item holds
    "volumes": 112,
    "volumes per shell": {"0": 16, "1000": 40, "2000": 56},   # b rounded to SHELL_ROUND; b0s are the forward series'
    "shell_round": 100,
    "matrix": [112, 112, 54],
    "voxel_mm": [2.1429, 2.1429, 2.2], "voxel_tolerance_mm": 0.01,
    "phase_encoding": ["j", "j-"],                       # the forward (APP) series, the reversed (APA) one
    "reversed_b0s": 1,
    "reversed_b0_over_forward": [0.85, 1.05],            # the APA b0's brain median over the APP b0s' (seen: 0.89-0.96)
}
BENCH_MODULES = ("calibration_card.py", "calibration_runs.py", "ensemble_retest.py", "_register.py", "_masivar.py",
                 "_bootstrap.py", "ensemble_grid_check.py", "ensemble_endpoints.py", "ensemble_pooled.py", "ensemble_vorobev.py",
                 "ensemble_pilot.py")                    # the bench code the runs, collect and fit import
SECONDS = {"A": 75, "rescan": 100, "replicate": 77}    # per run on the M2 (cIIIs03's smoke runs, 2026-10-18; a rescan is
                                                       # its correction and registration, 49 s, and its _onA run, 50 s)


@dataclass(frozen=True)
class Pair:
    name: str                   # the record folder: DATA/ensemble/<subject>/card/<name>
    ref: tuple                  # A: (session, scan)
    same: dict                  # same-session rescans {tag: (session, scan)}
    other: dict                 # other-session rescans {tag: (session, scan)}
    reuse: str = None           # an earlier record folder under DATA/ensemble/<subject>/, read, never written
    notes: tuple = field(default=())

    @property
    def rescans(self):
        return {**self.same, **self.other}


CARD = {
    "cIIIs01": Pair("cIIIs01-2.1", ("s1Bx3", 1), {"within": ("s1Bx3", 2)}, {"session1": ("s1Bx1", 1)},
                    notes=("A is s1Bx3 scan 1: the 2.1 mm scans on disk are s1Bx1 scan 1 and s1Bx3 scans 1-2",
                           "rerun: retest/cIIIs01-2.1 is format 1 (no streamlines) with 8 replicates")),
    "cIIIs02": Pair("cIIIs02-2.1", ("s1Bx1", 1), {"within": ("s1Bx1", 2)}, {"session4": ("s1Bx4", 1), "session5": ("s1Bx5", 2)},
                    notes=("session 5 uses scan 2: scan 1's b2000 series is 2.2 mm iso",)),
    "cIIIs03": Pair("cIIIs03-2.1", ("s1Bx1", 1), {"within": ("s1Bx1", 2)}, {}),
    "cIIIs04": Pair("cIIIs04-2.1", ("s1Bx1", 1), {"within": ("s1Bx1", 2)},
                    {"session2": ("s1Bx2", 1), "session3": ("s1Bx3", 1), "session4": ("s1Bx4", 1), "session5": ("s1Bx5", 1)},
                    reuse="retest/cIIIs04-2.1",
                    notes=("A, smooth/r001..r016, within, session3, session5 (and their replicates) from retest/cIIIs04-2.1",)),
    "cIIIs05": Pair("cIIIs05-2.1", ("s1Bx1", 1), {"within": ("s1Bx1", 2)}, {"session2": ("s1Bx2", 1)}),
    "cIIIs06": Pair("cIIIs06-2.1", ("s1Bx1", 1), {"within": ("s1Bx1", 2)}, {}),
    "cIIIs07": Pair("cIIIs07-2.1", ("s1Bx1", 1), {"within": ("s1Bx1", 2)}, {}),
    "cIIIs08": Pair("cIIIs08-2.1", ("s1Bx1", 1), {"within": ("s1Bx1", 2)}, {"session2": ("s1Bx2", 1)}),
}


class Records:
    """A subject's record folders: the card's own (written) and the earlier one CARD names (read only)."""
    def __init__(self, subject):
        self.subject, self.pair = subject, CARD[subject]
        self.dir = DATA / "ensemble" / subject / "card" / self.pair.name
        self.reuse = DATA / "ensemble" / subject / self.pair.reuse if self.pair.reuse else None

    def path(self, name, ext=".npz"):
        """Where record `name` is: the card's folder, else the reused one; a new record's path if neither has it."""
        own = self.dir / f"{name}{ext}"
        if not own.exists() and self.reuse is not None and (self.reuse / f"{name}{ext}").exists():
            return self.reuse / f"{name}{ext}"
        return own

    def have(self, name, ext=".npz"):
        return self.path(name, ext).exists()

    def new(self, name, ext=".npz"):
        """The path for a new record: refused if it exists here or in the reused folder."""
        p = self.dir / f"{name}{ext}"
        if self.have(name, ext):
            raise FileExistsError(f"{self.path(name, ext)} exists: records are never overwritten")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def ensemble(self, tag="", runs=RUNS):
        """The record paths of a scan's 17-run ensemble on A's grid: A (or <tag>_onA) and its replicates."""
        first = "A" if not tag else f"{tag}_onA"
        return [self.path(first)] + [self.path(f"{FLIP}/{tag + '_' if tag else ''}r{r:03d}") for r in range(1, runs + 1)]


def free_gb():
    return shutil.disk_usage(DATA).free / 1e9


def guard_disk():
    if free_gb() < MIN_FREE_GB:
        sys.exit(f"stopped: {free_gb():.1f} GB free, below {MIN_FREE_GB:g} GB")


def git_state():
    """HEAD, and the files that matter (src/ and BENCH_MODULES) that differ from it or are untracked: none, if clean."""
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.rstrip()  # noqa: E731
    dirty = run("status", "--porcelain", "--untracked-files=all", "--", str(HERE.parent / "src"), *(str(HERE / m) for m in BENCH_MODULES))
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty (src/ and the card's bench modules)": [line.strip() for line in dirty.splitlines() if line.strip()]}


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def module_hashes():
    return {m: sha256(HERE / m) for m in BENCH_MODULES}


def pipeline_settings():
    """Today's default pipeline, as these runs call it."""
    return {"correct": "P.correct (default)", "track": {"shell": SHELL, "tracker": "UKF defaults (no tracker= settings)"},
            "denoise": None, "labeler": "P.default_labeler (RapidParc), draw 0, grouping 'index' (the default)",
            "bootstrap": {"class": "_bootstrap.ShellWildBootstrap", "flip": FLIP, "fwhm_voxels": FWHM, "b0_noise": "added",
                          "mask": "A's tracking mask (A_mask.npy)", "seed": "replicate r uses seed r"},
            "record": "tractline.ensemble.RunRecord, compact (format 2)", "device": DEVICE}


def protocol_check(s):
    """What PROTOCOL checks, as found in a loaded scan (_masivar.Subject): {item: (found, ok)}."""
    rnd = PROTOCOL["shell_round"]
    b = (np.round(np.asarray(s.bval) / rnd) * rnd).astype(int)
    per = {str(k): int(n) for k, n in zip(*np.unique(b, return_counts=True))}
    apa = next(x for x in s.series if "APA" in x["file"])
    rev = [apa["over_app_b0s"][i] for i in apa["taken_as_b0s"]]
    lo, hi = PROTOCOL["reversed_b0_over_forward"]
    found = {"volumes": s.dwi.shape[-1], "volumes per shell": per, "matrix": list(s.dwi.shape[:3]),
             "voxel_mm": np.round(s.vox, 4).tolist(), "phase_encoding": list(s.pe_directions), "reversed_b0s": len(rev),
             "reversed_b0_over_forward": rev}
    ok = {"volumes": found["volumes"] == PROTOCOL["volumes"], "volumes per shell": per == PROTOCOL["volumes per shell"],
          "matrix": found["matrix"] == PROTOCOL["matrix"],
          "voxel_mm": bool(np.allclose(s.vox, PROTOCOL["voxel_mm"], atol=PROTOCOL["voxel_tolerance_mm"])),
          "phase_encoding": found["phase_encoding"] == PROTOCOL["phase_encoding"],
          "reversed_b0s": len(rev) == PROTOCOL["reversed_b0s"], "reversed_b0_over_forward": all(lo <= r <= hi for r in rev)}
    return {k: (found[k], ok[k]) for k in found}


def load_checked(subject, session, scan):
    """A scan of the card's protocol (every PROTOCOL item), or an error naming what differs."""
    from _masivar import load
    s = load(subject, session, scan, SHELLS)
    bad = {k: v for k, (v, ok) in protocol_check(s).items() if not ok}
    if bad:
        raise ValueError(f"{s.name}: not the card's protocol: {bad} (expected {({k: PROTOCOL[k] for k in bad})})")
    return s


class Runner:
    def __init__(self, subject):
        self.R = Records(subject)
        self.subject, self.pair = subject, self.R.pair
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": git_state(), "pipeline": pipeline_settings(), "pair": {
            "reference": self.pair.ref, "same session": self.pair.same, "other sessions": self.pair.other,
            "reuse": self.pair.reuse, "notes": list(self.pair.notes)}})
        self.labeler = None

    def scan_log(self, tag):
        """A rescan's registration and series, from this folder's runs.json or the reused folder's."""
        if f"{tag}_scan" in self.log:
            return self.log[f"{tag}_scan"]
        return json.loads((self.R.reuse / "runs.json").read_text())[f"{tag}_scan"]

    def save_log(self):
        tmp = self.log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.log, indent=1))
        os.replace(tmp, self.log_path)

    def one(self, s, name, corrected=None, extra=None, before_record=None):
        """Correct (unless given the corrected DWI), track, label (draw 0), record on s's grid (A's) under `name`.
        before_record(corr, tg), if given, runs before the record is written (its companions first)."""
        guard_disk()
        path = self.R.new(name)
        if self.labeler is None:
            self.labeler = P.default_labeler(DEVICE)
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(DEVICE):
            corr = P.correct(s, timer, DEVICE) if corrected is None else P.Correction(corrected, None, None, None, applied=True, note="given")
            tg = P.track(s, corr.dwi, timer, device=DEVICE, shell=SHELL)
            with timer("label"):
                labels = self.labeler(tg.fibers, draws=(0,), logp=True)
        if before_record is not None:
            before_record(corr, tg)
        tmp = path.with_name(path.stem + ".partial.npz")
        E.RunRecord.of(tg, labels, s.affine, s.dwi.shape[:3]).save(tmp, compact=True)
        os.replace(tmp, path)
        self.log[name] = {"fibers": len(tg.fibers), "labeled": int(labels.keep.sum()), "residual_left": corr.residual_left,
                          "field_hz_p99_abs": round(float(np.quantile(np.abs(corr.field_hz), 0.99)), 2) if corr.field_hz is not None else None,
                          "seconds": timer.seconds, "total_s": round(time.time() - t0, 1), **(extra or {})}
        self.save_log()
        print(self.subject, name, json.dumps(self.log[name]), flush=True)
        P.release_memory(DEVICE)
        return corr, tg

    def reference(self):
        """A's run and its two companions (A_meanb0.npy, A_mask.npy): all three or none. The companions are written
        first and A.npz last, so an interrupted run leaves no A.npz without them; a partial set is refused."""
        ses, scan = self.pair.ref
        A = load_checked(self.subject, ses, scan)
        parts = {"A": self.R.have("A"), "A_meanb0": self.R.have("A_meanb0", ".npy"), "A_mask": self.R.have("A_mask", ".npy")}
        if not any(parts.values()):
            self.log["A_scan"] = {"name": A.name, "series": A.series, "notes": A.notes, "shape": list(A.dwi.shape)}

            def companions(corr, tg):
                for name, arr in (("A_meanb0", mean_b0(corr.dwi, A.bval)), ("A_mask", tg.mask)):
                    p = self.R.new(name, ".npy")
                    tmp = p.with_name(p.stem + ".partial.npy")
                    np.save(tmp, arr)
                    os.replace(tmp, p)
            self.one(A, "A", before_record=companions)
        elif not all(parts.values()):
            raise RuntimeError(f"{self.subject}: A's records are incomplete {parts}: remove the companions of a missing A.npz by hand")
        return A, np.load(self.R.path("A_meanb0", ".npy")), np.load(self.R.path("A_mask", ".npy"))

    def rescan(self, A, ref_b0, tag):
        """Rescan `tag` corrected on its grid, registered to A, resampled onto A's grid and tracked: <tag>_onA."""
        if self.R.have(f"{tag}_onA"):
            return
        ses, scan = self.pair.rescans[tag]
        B = load_checked(self.subject, ses, scan)
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(DEVICE):
            cb = P.correct(B, timer, DEVICE)
        P.release_memory(DEVICE)
        to_ref, ncc = register(mean_b0(cb.dwi, B.bval), B.affine, ref_b0, A.affine)
        rot = np.degrees(np.arccos(np.clip((np.trace(to_ref[:3, :3]) - 1) / 2, -1, 1)))
        dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, to_ref, A.affine, A.dwi.shape[:3])
        self.log[f"{tag}_scan"] = {"name": B.name, "term": "same session" if tag in self.pair.same else "other session",
                                   "series": B.series, "shape": list(B.dwi.shape), "residual_left": cb.residual_left,
                                   "field_hz_p99_abs": round(float(np.quantile(np.abs(cb.field_hz), 0.99)), 2),
                                   "correct_seconds": timer.seconds, "correct_register_resample_s": round(time.time() - t0, 1),
                                   "to_A": np.round(to_ref, 5).tolist(), "rotation_deg": round(float(rot), 2),
                                   "translation_mm": round(float(np.linalg.norm(to_ref[:3, 3])), 2), "ncc_after": round(ncc, 4)}
        self.save_log()
        self.one(replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), f"{tag}_onA", corrected=dwi)

    def replicates(self, A, mask, runs):
        from _bootstrap import ShellWildBootstrap
        need = [r for r in range(1, runs + 1) if not self.R.have(f"{FLIP}/r{r:03d}")]
        if not need:
            return
        wb = ShellWildBootstrap(A.dwi, A.bval, A.bvec, extra_b0s=A.b0s[..., len(A.b0_index):], b0_noise="added", mask=mask,
                                flip=FLIP, fwhm=FWHM)
        self.log.setdefault(f"bootstrap_{FLIP}", wb.noise_check(mask))
        for r in need:
            dwi, apa = wb.replicate(r)
            self.one(replace(A, dwi=dwi, b0s=np.concatenate([dwi[..., A.b0_index].astype(np.float64), apa], -1)), f"{FLIP}/r{r:03d}")

    def rescan_replicates(self, A, tag, runs):
        """A rescan's own ensemble on A's grid (ensemble_retest.rescans): replicates of the rescan as acquired, its own
        mask, each corrected on its grid, resampled onto A's by the registration in runs.json and tracked there."""
        from _bootstrap import ShellWildBootstrap
        from tractline.prep import prepare
        need = [r for r in range(1, runs + 1) if not self.R.have(f"{FLIP}/{tag}_r{r:03d}")]
        if not need:
            return
        ses, scan = self.pair.rescans[tag]
        B = load_checked(self.subject, ses, scan)
        to_ref = np.array(self.scan_log(tag)["to_A"])
        with P.exact_float32(DEVICE):
            mask = prepare(P.correct(B, P.Timer(), DEVICE).dwi, B.affine, B.bval, B.bvec, shell=SHELL, device=DEVICE).mask
        P.release_memory(DEVICE)
        wb = ShellWildBootstrap(B.dwi, B.bval, B.bvec, extra_b0s=B.b0s[..., len(B.b0_index):], b0_noise="added", mask=mask,
                                flip=FLIP, fwhm=FWHM)
        self.log.setdefault(f"bootstrap_{FLIP}_{tag}", wb.noise_check(mask))
        for r in need:
            guard_disk()
            dwi, apa = wb.replicate(r)
            s = replace(B, dwi=dwi, b0s=np.concatenate([dwi[..., B.b0_index].astype(np.float64), apa], -1))
            timer = P.Timer()
            with P.exact_float32(DEVICE):
                corr = P.correct(s, timer, DEVICE)
            P.release_memory(DEVICE)
            on_a, bvec = onto_reference(corr.dwi, B.bvec, B.affine, to_ref, A.affine, A.dwi.shape[:3])
            self.one(replace(A, dwi=on_a, bval=B.bval, bvec=bvec, b0s=None), f"{FLIP}/{tag}_r{r:03d}", corrected=on_a,
                     extra={"rescan residual_left": corr.residual_left,
                            "rescan field_hz_p99_abs": round(float(np.quantile(np.abs(corr.field_hz), 0.99)), 2),
                            "rescan correct_seconds": timer.seconds})


def run(args):
    for subject in args.subject:
        t0 = time.time()
        rn = Runner(subject)
        A, ref_b0, mask = rn.reference()
        for tag in (args.tags or list(rn.pair.rescans)):
            rn.rescan(A, ref_b0, tag)
        rn.replicates(A, mask, args.runs)
        print(f"{subject} done {time.time() - t0:.0f}s, {free_gb():.1f} GB free", flush=True)


def rescans(args):
    for subject in args.subject:
        rn = Runner(subject)
        A, ref_b0, _ = rn.reference()
        for tag in args.tags:
            rn.rescan(A, ref_b0, tag)
            rn.rescan_replicates(A, tag, args.runs)


def verify(args):
    """A rerun at HEAD against a stored A: the reference scan through today's pipeline again (nothing written to the
    records), compared with A.npz: seed keys, clusters, streamline counts per voxel, and every streamline vertex
    within the compact record's quantization (half of compact.QUANTUM_MM per axis). Writes
    results/calibration_verify_<subject>.json."""
    from tractline import compact as C
    for subject in args.subject:
        R = Records(subject)
        stored_path = R.path("A")
        stored = E.RunRecord.load(stored_path, logp=False, fibers=True)
        A = load_checked(subject, *CARD[subject].ref)
        labeler = P.default_labeler(DEVICE)
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(DEVICE):
            corr = P.correct(A, timer, DEVICE)
            tg = P.track(A, corr.dwi, timer, device=DEVICE, shell=SHELL)
            labels = labeler(tg.fibers, draws=(0,), logp=True)
        new = E.RunRecord.of(tg, labels, A.affine, A.dwi.shape[:3])
        same_keys = bool(np.array_equal(new.keys, stored.keys))
        out = {"subject": subject, "stored": str(stored_path.relative_to(DATA)), "git": git_state(), "seconds": round(time.time() - t0, 1),
               "streamlines (new, stored)": [len(new.keys), len(stored.keys)], "seed keys equal": same_keys}
        if same_keys:
            lens = [len(a) == len(b) for a, b in zip(new.fibers, stored.fibers)]
            dev = max(float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()) for a, b, ok in zip(new.fibers, stored.fibers, lens) if ok) if any(lens) else None
            out.update({"clusters equal": bool(np.array_equal(new.cluster, stored.cluster)),
                        "voxel counts equal": bool(np.array_equal(new.counts, stored.counts)),
                        "streamline lengths equal": bool(all(lens)),
                        "max vertex difference mm": dev, "quantization bound mm": C.QUANTUM_MM / 2,
                        "mask equal": bool(np.array_equal(tg.mask, np.load(R.path("A_mask", ".npy"))))})
            out["same as stored"] = bool(out["clusters equal"] and out["voxel counts equal"] and out["streamline lengths equal"]
                                         and dev is not None and dev <= C.QUANTUM_MM / 2 + 1e-4 and out["mask equal"])
        else:
            out["same as stored"] = False
        (HERE / f"results/calibration_verify_{subject}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1), flush=True)
        P.release_memory(DEVICE)


def plan(args):
    """The pair table, and the runs a full queue still needs (A, the rescans' _onA runs, A's replicates)."""
    total, secs = 0, 0
    for subject, pair in CARD.items():
        R = Records(subject)
        need = {"A": int(not R.have("A")),
                "rescan": sum(not R.have(f"{t}_onA") for t in pair.rescans),
                "replicate": sum(not R.have(f"{FLIP}/r{r:03d}") for r in range(1, args.runs + 1))}
        n = sum(need.values())
        total += n
        secs += sum(SECONDS[k] * v for k, v in need.items())
        print(f"{subject}  A {pair.ref[0]} scan {pair.ref[1]}  same: {pair.same}  other: {pair.other}  "
              f"to run: {need} ({n})  in {R.dir.relative_to(DATA)}" + (f" + {pair.reuse} (read)" if pair.reuse else ""))
        if args.check:
            for ses, scan in [pair.ref] + list(pair.rescans.values()):
                from _masivar import load
                s = load(subject, ses, scan, SHELLS)
                bad = [k for k, (_, ok) in protocol_check(s).items() if not ok]
                print(f"    {s.name}: " + ("the protocol" if not bad else f"NOT the protocol: {bad}"), flush=True)
    print(f"{total} runs, about {secs / 3600:.1f} h on the M2; {free_gb():.1f} GB free")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("plan", "run", "rescans", "verify"))
    ap.add_argument("--subject", nargs="+", default=list(CARD), choices=sorted(CARD))
    ap.add_argument("--runs", type=int, default=RUNS, help="replicates (A's; or each rescan's, for rescans)")
    ap.add_argument("--tags", nargs="+", help="run: only these rescans (default: all of the subject's); rescans: required")
    ap.add_argument("--check", action="store_true", help="plan: also load every scan and check it against PROTOCOL")
    args = ap.parse_args()
    if args.what == "rescans" and not args.tags:
        ap.error("rescans needs --tags")
    {"plan": plan, "run": run, "rescans": rescans, "verify": verify}[args.what](args)
