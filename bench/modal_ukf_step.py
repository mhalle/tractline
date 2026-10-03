"""ukf_step_bench.step on an A10G: float32 and float64, batches of 50k and 200k fibers.

    modal run bench/modal_ukf_step.py
"""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.14.1", "numpy>=2")
         .add_local_file(str(HERE / "ukf_step_bench.py"), remote_path="/root/ukf_step_bench.py"))
app = modal.App("tractography-ukf-step", image=image)


@app.function(gpu="A10G", timeout=1800)
def run() -> list:
    import sys, torch
    sys.path.insert(0, "/root")
    import ukf_step_bench as u
    rows = []
    for dt, B in ((torch.float32, 50000), (torch.float32, 200000), (torch.float64, 50000)):
        r = u.bench("cuda", dt, B); r["gpu"] = torch.cuda.get_device_name(0); rows.append(r); print(json.dumps(r), flush=True)
    return rows


@app.local_entrypoint()
def main():
    rows = run.remote()
    out = HERE / "results" / "ukf_step.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev["a10g"] = rows
    out.write_text(json.dumps(prev, indent=1))
    print(json.dumps(rows, indent=1))
