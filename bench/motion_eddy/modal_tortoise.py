"""TORTOISE (NIH, TORTOISEV4; GPL-3.0, eurotomania/tortoise) on Modal, as a third reference beside eddy and the
prototype for the real scan's motion and eddy correction (docs/gpu-motion-eddy-design.md section 2.4). TORTOISE
registers every volume to the b0 by MI (DIFFPREP), so it has no shell gauge.

    modal run bench/motion_eddy/modal_tortoise.py::help                                 # TORTOISEProcess's options
    modal volume put tractography-bench <dir>/tortoise_in motion_eddy/real16974/in      # dwi.nii.gz, bvals, bvecs, dwi.json
    modal run bench/motion_eddy/modal_tortoise.py::run --both --args "..."              # out_gpu (CUDA, A10) and out_cpu (32 CPUs)
    modal volume get tractography-bench motion_eddy/real16974/out_gpu <dir>/tortoise_out_gpu

Inputs and outputs go through the tractography-bench Volume (the data is ON-Harmony, CC0).
"""
import modal

image = modal.Image.from_registry("eurotomania/tortoise:latest", add_python="3.11")
app = modal.App("tractline-tortoise", image=image)
vol = modal.Volume.from_name("tractography-bench")
ROOT = "/vol/motion_eddy/real16974"


def _sh(cmd, timeout=None):
    import subprocess
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout[-20000:], p.stderr[-20000:]


@app.function(timeout=600)
def help_text():
    out = {}
    for c in ("which TORTOISEProcess TORTOISEProcess_cuda DIFFPREP; ls /usr/local/bin | head -50",
              "TORTOISEProcess 2>&1 | tail -n +280", "ls /usr/src/TORTOISE_installation_folder/TORTOISEV4/settings/mecc_settings; cat /usr/src/TORTOISE_installation_folder/TORTOISEV4/settings/mecc_settings/quadratic*"):
        out[c] = _sh(c, timeout=120)
    return out


@app.function(volumes={"/vol": vol}, timeout=6 * 3600, cpu=32, memory=65536)
def run_cpu(args: str, out: str = "out_cpu"):
    vol.reload()
    code, so, se = _sh(f"cd {ROOT} && rm -rf {out} && mkdir -p {out} && cp -r in/* {out}/ && cd {out} && (date +%s; TORTOISEProcess {args}; echo exit $?; date +%s) > tortoise.log 2>&1; echo $?")
    vol.commit()
    return code, so, se


@app.function(volumes={"/vol": vol}, timeout=6 * 3600, gpu="A10", cpu=8, memory=65536)
def run_gpu(args: str, out: str = "out_gpu"):
    vol.reload()
    code, so, se = _sh(f"cd {ROOT} && rm -rf {out} && mkdir -p {out} && cp -r in/* {out}/ && cd {out} && nvidia-smi -L > gpu.txt 2>&1; (date +%s; TORTOISEProcess_cuda {args}; echo exit $?; date +%s) > tortoise.log 2>&1; echo $?")
    vol.commit()
    return code, so, se


@app.local_entrypoint()
def help():
    for k, (code, so, se) in help_text.remote().items():
        print(f"$ {k}\n[{code}]\n{so}\n{se}\n")


@app.local_entrypoint()
def run(args: str = "", gpu: bool = False, both: bool = False):
    if both:
        hs = [run_gpu.spawn(args), run_cpu.spawn(args)]
        for h in hs:
            print(h.get())
    else:
        print((run_gpu if gpu else run_cpu).remote(args))
