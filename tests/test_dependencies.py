"""The package needs numpy and torch only: in a fresh interpreter with the optional packages made
unimportable, every pipeline module imports and the phantom runs scan to labels on the CPU (one worker,
in process, so the block covers the tracker too)."""
import subprocess, sys
from pathlib import Path

OPTIONAL = ["scipy", "nibabel", "nrrd", "dipy", "vtk", "sklearn", "matplotlib", "trx", "tractcloud", "numba"]   # not triton: torch's Linux wheel brings it

SCRIPT = f"""
import importlib.abc, importlib.machinery, sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
class Refuse(importlib.abc.Loader):
    def create_module(self, spec): raise ImportError(spec.name + " is blocked")
    def exec_module(self, module): pass
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):          # probing answers (torch probes); importing fails
        if name.split(".")[0] in {OPTIONAL!r}:
            return importlib.machinery.ModuleSpec(name, Refuse())
sys.meta_path.insert(0, Block())
import numpy as np
from tractline import pipeline as P, susceptibility, prep, mask, ukf, trx, data
from tractline.labelers import base, rapidparc
from test_pipeline_cpu import stand_in
from phantom import subject
corr, tg, labels = P.run(subject(), stand_in, P.Timer(), device="cpu", workers=1)
assert len(tg.fibers) > 500, len(tg.fibers)
loaded = sorted(m for m in sys.modules if m.split(".")[0] in {OPTIONAL!r})
assert not loaded, loaded
print("ok", len(tg.fibers))
"""


def test_runs_with_numpy_and_torch_only():
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip().startswith("ok")


def test_t1check_names_its_extra():
    r = subprocess.run([sys.executable, "-c", SCRIPT.split("import numpy as np")[0] + "import tractline.t1check"],
                       capture_output=True, text=True)
    assert "tractline[t1check]" in r.stderr
