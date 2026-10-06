"""The cross-platform timing table from timing_matrix.py's results: per scan, each platform's stage times
(second run: the steady state) and its first run's total, as Markdown.

    uv run bench/timing_table.py [--tag 1006b] [--first] [--before ""]

--tag: the results' suffix (timing_matrix_<platform>_<tag>.json); --first: the first run's total too; --before: another
suffix ("" for the unsuffixed files) whose totals are shown beside. Prints the tables.
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLATFORMS = [("M2 GPU (MPS)", "mps"), ("M2 CPU, 8 workers", "m2cpu"), ("L4", "l4"), ("A10", "a10"), ("L40S", "l40s"),
             ("x86 CPU, 32 workers", "cpu")]
SCANS = [("Stanford HARDI (no correction; 150 dirs, b 2000)", "hardi"), ("ds001226 PAT16 (corrected; 50 dirs, b 2800)", "pat16"),
         ("ds005123 sub-10317 (field maps refused; 30 dirs, b 3250)", "ds005123")]
STAGES = ["field_estimate", "field_apply", "prep", "load", "ukf", "label"]


def load(platform, tag):
    f = HERE / "results" / f"timing_matrix_{platform}{'_' + tag if tag else ''}.json"
    return json.loads(f.read_text()) if f.exists() else None


LABELS = {"field_estimate": "field estimate", "field_apply": "field apply", "prep": "prep (shell, gradients, mask)",
          "load": "load (tracker input)", "ukf": "tracking (UKF)", "label": "labeling (RapidParc)"}


def table(tag, before=None, first=False):
    """Markdown: per scan, a row per stage and a column per platform (each platform's second run); with first, the
    first run's total too; with before (another tag), that run's total."""
    out = []
    R = {p: load(p, tag) for _, p in PLATFORMS}
    B = {p: load(p, before) for _, p in PLATFORMS} if before is not None else {}
    cols = [(n, p) for n, p in PLATFORMS if R[p]]
    head = lambda: ["| step, seconds | " + " | ".join(n for n, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
    for title, sc in SCANS:
        corrected = any(R[p]["scans"].get(sc, {}).get("runs", [{}])[-1].get("corrected") for _, p in cols)
        out += [f"**{title}**", ""] + head()
        rows = STAGES + ["total"] + (["first run"] if first else []) + (["before"] if before is not None else [])
        for st in rows:
            cells = []
            for _, p in cols:
                runs = R[p]["scans"].get(sc, {}).get("runs", [])
                if not runs:
                    cells.append("-"); continue
                if st == "total":
                    cells.append(f"**{runs[-1]['scan_to_labels_s']:.1f}**")
                elif st == "first run":
                    cells.append(f"{runs[0]['scan_to_labels_s']:.1f}")
                elif st == "before":
                    b = (B.get(p) or {}).get("scans", {}).get(sc, {}).get("runs", [])
                    cells.append(f"{b[-1]['scan_to_labels_s']:.1f}" if b else "-")
                elif st.startswith("field") and not corrected:
                    cells.append("n/a")
                else:
                    cells.append(f"{runs[-1]['seconds'].get(st, 0.0):.2f}")
            out.append(f"| {LABELS.get(st, st)} | " + " | ".join(cells) + " |")
        out.append("")
    fc = [(n, (R[p]["scans"].get("ds005123_forced", {}).get("runs") or [None])[-1]) for n, p in cols]
    if any(r for _, r in fc):
        out += ["**ds005123: the correction alone, forced past the shim check (timing only: this field is not the series')**", ""]
        out += head()
        for st in ("field_estimate", "field_apply"):
            out.append(f"| {LABELS[st]} | " + " | ".join(f"{r['seconds'][st]:.2f}" if r else "-" for _, r in fc) + " |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="1006"); ap.add_argument("--before", default=None)
    ap.add_argument("--first", action="store_true")
    a = ap.parse_args()
    print(table(a.tag, a.before, a.first))
