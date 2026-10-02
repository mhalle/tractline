"""The default pipeline needs only numpy, scipy, torch, nibabel and TractCloud: run it with the other
packages the bench uses made unimportable, on the GPU path and the CPU path.

    DATA/.venv/bin/python bench/tractography/dependency_check.py [--sub PAT16]

Blocked: numba, dipy, nrrd (pynrrd), rankfield, numcodecs, sklearn. Any import of them anywhere on the
pipeline's path fails the run. Writes results/dependency_check.json.
"""
import importlib.abc, json, sys
from pathlib import Path

BLOCKED = ("numba", "dipy", "nrrd", "rankfield", "numcodecs", "sklearn")


class _Refuse(importlib.abc.Loader):
    def create_module(self, spec):
        raise ImportError(f"{spec.name} is blocked: the default pipeline must not need it")

    def exec_module(self, module):
        pass


class _Block(importlib.abc.MetaPathFinder):
    """Importing a blocked package fails; asking whether it exists (torch._dynamo probes optional
    packages with find_spec) still answers - probing is not using."""
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            import importlib.machinery
            return importlib.machinery.ModuleSpec(name, _Refuse())
        return None


if __name__ == "__main__":
    sys.meta_path.insert(0, _Block())
    import argparse
    import torch
    import _pipeline as P
    from _ds001226 import load
    from _tractcloud import Labeler
    ap = argparse.ArgumentParser(); ap.add_argument("--sub", default="PAT16"); args = ap.parse_args()
    s = load(args.sub)
    res = {"blocked": BLOCKED, "runs": {}}
    for device in (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]:
        timer = P.Timer(echo=f"{args.sub} {device}")
        import tempfile
        with tempfile.TemporaryDirectory() as d:                     # the TRX writer is on the path too
            corr, tg, labels, payload = P.run(s, Labeler(device), timer, device=device, trx=Path(d) / "check.trx")
        res["runs"][device] = {"seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages()),
                               "fibers": tg.stats["fibers"], "labeled": int(labels.keep.sum())}
    res["loaded_blocked_modules"] = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    print(json.dumps(res, indent=1))
    (Path(__file__).resolve().parent / "results/dependency_check.json").write_text(json.dumps(res, indent=1))
