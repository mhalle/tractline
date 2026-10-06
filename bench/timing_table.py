"""The cross-platform timing table from timing_matrix.py's results: per scan, each platform's stage times
(second run: the steady state) and its first run's total, as Markdown.

    uv run bench/timing_table.py [--tag 1006] [--before ""]

--tag: the results' suffix (timing_matrix_<platform>_<tag>.json); --before: another suffix ("" for the
unsuffixed files) whose totals are shown beside. Prints the tables.
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


def table(tag, before):
    out = []
    R = {p: load(p, tag) for _, p in PLATFORMS}
    B = {p: load(p, before) for _, p in PLATFORMS} if before is not None else {}
    for title, sc in SCANS:
        out += [f"**{title}**", "", "| stage, s | " + " | ".join(n for n, p in PLATFORMS if R[p]) + " |",
                "|---|" + "---|" * sum(1 for _, p in PLATFORMS if R[p])]
        for st in STAGES + ["total", "first run", "before (second run)"]:
            cells = []
            for _, p in PLATFORMS:
                if not R[p]:
                    continue
                runs = R[p]["scans"].get(sc, {}).get("runs", [])
                if not runs:
                    cells.append("-"); continue
                if st == "total":
                    cells.append(f"**{runs[-1]['scan_to_labels_s']}**")
                elif st == "first run":
                    cells.append(str(runs[0]["scan_to_labels_s"]))
                elif st == "before (second run)":
                    b = (B.get(p) or {}).get("scans", {}).get(sc, {}).get("runs", [])
                    cells.append(str(b[-1]["scan_to_labels_s"]) if b else "-")
                else:
                    v = runs[-1]["seconds"].get(st, 0.0)
                    cells.append("-" if v == 0 and st.startswith("field") else str(v))
            if st == "before (second run)" and before is None:
                continue
            out.append(f"| {st.replace('_', ' ')} | " + " | ".join(cells) + " |")
        out.append("")
    fc = []
    for n, p in PLATFORMS:
        r = (R[p] or {}).get("scans", {}).get("ds005123_forced", {}).get("runs", [])
        if r:
            fc.append(f"{n} {r[-1]['seconds']['field_estimate']} + {r[-1]['seconds']['field_apply']} s")
    if fc:
        out.append("ds005123's correction forced past the shim check (timing only; estimate + apply): " + "; ".join(fc) + ".")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="1006"); ap.add_argument("--before", default=None)
    a = ap.parse_args()
    print(table(a.tag, a.before))
