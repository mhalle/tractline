"""Tabulates results/readout_polarity/*.json (readout_polarity.py): per patient, the displacement's 99th
percentile in the brain for the pair and for the control, the readout-scaled and flipped runs against the
base, and the first level's cost ratio.

    uv run bench/readout_polarity_summary.py

Writes results/readout_polarity/summary.json and prints a table.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "results/readout_polarity"

if __name__ == "__main__":
    rows = []
    for p in sorted(D.glob("PAT*.json")):
        r = json.loads(p.read_text())["runs"]
        rows.append({"subject": p.stem,
                     "pair_99th_mm": r["base"]["displacement_abs_mm_median_99th"][1],
                     "control_99th_mm": r["control (AP halves as opposite)"]["displacement_abs_mm_median_99th"][1],
                     "pair_level1_cost": r["base"]["level1_cost_after_over_before"],
                     "control_level1_cost": r["control (AP halves as opposite)"]["level1_cost_after_over_before"],
                     "readout_x0.8_vs_base_99th_mm": r["readout x0.8"]["vs_base_abs_mm_median_99th"][1],
                     "readout_x1.25_vs_base_99th_mm": r["readout x1.25"]["vs_base_abs_mm_median_99th"][1],
                     "flipped_vs_base_99th_mm": r["flipped"]["vs_base_abs_mm_median_99th"][1]})
    keys = list(rows[0])
    print(" | ".join(keys))
    for row in rows:
        print(" | ".join(str(row[k]) for k in keys))
    ratio = [row["pair_99th_mm"] / row["control_99th_mm"] for row in rows]
    summary = {"patients": len(rows), "rows": rows,
               "pair_over_control_99th_min_median": [round(min(ratio), 2), round(sorted(ratio)[len(ratio) // 2], 2)],
               "readout_vs_base_99th_mm_max": max(max(r["readout_x0.8_vs_base_99th_mm"], r["readout_x1.25_vs_base_99th_mm"]) for r in rows),
               "flipped_vs_base_99th_mm_max": max(r["flipped_vs_base_99th_mm"] for r in rows)}
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    (D / "summary.json").write_text(json.dumps(summary, indent=1))
