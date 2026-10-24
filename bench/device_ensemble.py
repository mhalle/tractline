"""Does a calibration map carry across devices? (docs/sota-pipeline-design.md, decision 6; NOTES 2026-10-24.) One run of
cIIIs04 on the CPU differs from its MPS run by about a fifth of a bootstrap replicate's difference (device_agreement.py).
This asks the question that matters: cIIIs04's whole 17-run ensemble (A and smooth/r001..r016, as calibration_runs makes
it) run on another device, then judged against the same rescans (the MPS records: the events to predict stay fixed) with
the cohort III map fitted on the other seven subjects. If the device were only another draw from the ensemble's own
distribution, the CPU ensemble's coverage - uncalibrated and calibrated - would match the MPS ensemble's within the
spread between subjects.

    uv run bench/device_ensemble.py run [--device cpu]          # 17 runs into <card folder>-cpu (84 min on the M2's CPU, 8 workers)
    uv run bench/device_ensemble.py collect [--device cpu]      # calibration_card's statements for that ensemble
    uv run bench/device_ensemble.py compare [--device cpu]      # results/device_ensemble_cIIIs04.json

The ensemble's records go into their own folder (the card's folder name + "-cpu"); A's mask and mean b0 are the CPU run's
own, as a user on the CPU would get them. The rescans (<tag>_onA) are read from the MPS folders. The disk guard is
lowered to MIN_FREE_GB here (the runs write about 130 MB; calibration_runs' 20 GB is for its long queues).
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_runs as CR  # noqa: E402
import calibration_card as CC  # noqa: E402

SUBJECT = "cIIIs04"
MIN_FREE_GB = 5.0


class DeviceRecords(CR.Records):
    """The ensemble's records (A, its companions, smooth/r*) in this device's own folder; everything else (the rescans)
    from the MPS records."""
    def __init__(self, subject, device):
        super().__init__(subject)
        self.mps = CR.Records(subject)
        self.dir = self.mps.dir.with_name(f"{self.mps.dir.name}-{device}")
        self.reuse = None

    @staticmethod
    def own(name):
        return name in ("A", "A_meanb0", "A_mask") or name.startswith(f"{CR.FLIP}/r")

    def path(self, name, ext=".npz"):
        mine = self.dir / f"{name}{ext}"
        return mine if self.own(name) or mine.exists() else self.mps.path(name, ext)


class Runner(CR.Runner):
    def __init__(self, subject, device):
        self.device = device
        self.R = DeviceRecords(subject, device)
        self.subject, self.pair = subject, self.R.pair
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": CR.git_state(), "device_ensemble.py sha256": CR.sha256(Path(__file__).resolve()),
                                        "pipeline": CR.pipeline_settings(), "device": device,
                                        "rescans": "read from the MPS records (calibration_runs' folders)",
                                        "disk guard": f"{MIN_FREE_GB:g} GB (lowered here)"})
        self.labeler = None


def run(args):
    CR.MIN_FREE_GB = MIN_FREE_GB
    rn = Runner(SUBJECT, args.device)
    A, _, mask = rn.reference()
    rn.replicates(A, mask, args.runs)
    rn.save_log()


def provenance(device):
    """calibration_card.provenance for these records, which come from more than one folder: each group's records by the
    folder they are in (this device's own, the MPS card folder, the MPS reused folder)."""
    def prov(subject, tags, runs=CR.RUNS):
        R = DeviceRecords(subject, device)
        groups = {"A and its replicates": R.ensemble("", runs), **{f"{t}_onA": [R.path(f"{t}_onA")] for t in tags}}
        out = {}
        for name, paths in groups.items():
            folders = {}
            for q in paths:
                k = str(q.parent.relative_to(CR.DATA))
                folders[k] = folders.get(k, 0) + 1
            out[name] = {"records per folder": folders}
        return out
    return prov


def collect(args):
    CC.Records = lambda subject: DeviceRecords(subject, args.device)
    CC.provenance = provenance(args.device)
    CC.collect(argparse.Namespace(subject=SUBJECT, runs=args.runs, force=args.force))


def compare(args):
    n = args.runs + 1
    others = [CC.Subject(CR.Records(s).dir / f"statements_R{n}.npz") for s in CR.CARD if s != SUBJECT]
    tests = {"mps": CC.Subject(CR.Records(SUBJECT).dir / f"statements_R{n}.npz"),
             args.device: CC.Subject(DeviceRecords(SUBJECT, args.device).dir / f"statements_R{n}.npz")}
    pat = CC.pat16_table()
    out = {"subject": SUBJECT, "map": f"cohort III, fitted on {[s.name for s in others]} (MPS)", "R": n, "terms": {}}
    for term in CC.TERMS:
        ev = {d: CC.evaluate(others, t, term, pat) for d, t in tests.items()}
        if any(v is None for v in ev.values()):
            continue
        out["terms"][term] = {
            "rescans": ev["mps"]["rescans"],
            "envelope coverage at target": {q: {d: {"uncalibrated": list(e["envelope"][q].values())[0], "calibrated": e["envelope"][q]["card: coverage"]}
                                                for d, e in ev.items()} for q in ev["mps"]["envelope"]},
            "streamline ECE (uncalibrated, calibrated)": {d: [e["streamline"]["as stated"], e["streamline"]["card"]] for d, e in ev.items()},
            "point ECE (uncalibrated, calibrated)": {d: [e["point"]["as stated"], e["point"]["card"]] for d, e in ev.items()},
            "shapes (coverage, size)": {d: e["shapes: predicted (card) / observed coverage, size"] for d, e in ev.items()},
        }
    # the scale to judge by: the other subjects' calibrated coverage at each target, held out as cIIIs04 is
    spread = {}
    for term in CC.TERMS:
        rows = []
        for s in [*others, tests["mps"]]:
            e = CC.evaluate([x for x in [*others, tests["mps"]] if x is not s], s, term, pat)
            if e is not None:
                rows.append({q: e["envelope"][q]["card: coverage"] for q in e["envelope"]})
        if rows:
            spread[term] = {q: [round(min(r[q] for r in rows), 4), round(max(r[q] for r in rows), 4)] for q in rows[0]}
    out["held-out calibrated coverage across the eight MPS subjects, min and max (the scale)"] = spread
    p = HERE / "results" / f"device_ensemble_{SUBJECT}.json"
    p.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "collect", "compare"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs", type=int, default=CR.RUNS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"run": run, "collect": collect, "compare": compare}[args.what](args)
