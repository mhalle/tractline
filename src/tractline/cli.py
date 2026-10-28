"""The tractline command: a dMRI series to labeled UKF tractography, from the shell.

    tractline run sub-01/dwi/sub-01_dir-AP_dwi.nii.gz --motion-eddy --trx out/sub-01.trx --qc qc/sub-01
    tractline fetch                      # the labelers' weights into the cache ahead of a run (tractline.fetch)
    tractline warm                       # on CUDA: compile the GPU kernels ahead of the first run
    tractline docs                       # the commands' help, the README and the QC guide as one Markdown text

Installed with the `cli` extra (nibabel and scipy to read BIDS, matplotlib for the QC plot), or run without installing:

    uvx --from "tractline[cli] @ git+https://github.com/mhalle/tractline@<tag>" tractline run ...

`run` is pipeline.run behind options: the series read by tractline.bids (the series that correct its distortion
found from the sidecars, or named with --partners), corrected, tracked (UKF two-tensor, the ORG atlas's settings),
labeled (RapidParc by default) and written as TRX; with --motion-eddy, head motion and eddy currents corrected in the
same resample, and with --qc their QC files written (motion_eddy.write_qc, motion_eddy.plot_qc).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

MODELS = ("quadratic", "linear", "rigid")


def _device(name: str) -> str:
    from .pipeline import resolve_device
    return resolve_device(name)


def _version() -> str:
    from . import __version__
    return __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(_version(), prog_name="tractline")
def main():
    """dMRI to labeled UKF tractography: susceptibility, head motion and eddy current correction, two-tensor UKF
    tractography with the ORG atlas's settings, and tract labels."""


@main.command()
@click.argument("dwi", type=click.Path(exists=True, dir_okay=False))
@click.option("--partners", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="A series that measures the field (repeatable); by default found from the BIDS sidecars.")
@click.option("--motion-eddy", type=click.Choice(MODELS), is_flag=False, flag_value="quadratic", default=None,
              help="Correct head motion and eddy currents; the eddy current model (quadratic without a value; "
                   "rigid: head motion alone). Off unless given. Name the series before it, or give the model.")
@click.option("--gibbs", is_flag=True,
              help="Remove Gibbs ringing first, from the series as acquired (DIPY's subvoxel-shift method; the slice "
                   "axis from the sidecar's SliceEncodingDirection, the third axis if it has none). Off unless given.")
@click.option("--mask", "mask", type=click.Choice(["median-otsu", "synthstrip"]), default="median-otsu",
              show_default=True,
              help="The brain mask: median_otsu of the mean b0, or SynthStrip's (experimental; its released weights, "
                   "31 MB, fetched on first use). The tracker seeds in the mask and stops at its edge.")
@click.option("--outliers", type=click.Choice(["replace", "flag", "off"]), default=None,
              help="With --motion-eddy: outlier slices replaced by their prediction (the default), flagged only, or "
                   "not sought.")
@click.option("--trx", type=click.Path(), default=None,
              help="Write the labeled streamlines here: a .trx file, or a directory for any other name.")
@click.option("--float16", is_flag=True, help="TRX positions in float16 (up to 0.03 mm off).")
@click.option("--labeled-only", is_flag=True, help="TRX of the labeled streamlines (40 mm or more) only.")
@click.option("--qc", type=click.Path(file_okay=False), default=None,
              help="With --motion-eddy: write its QC files (TSV, JSON, plot) into this folder.")
@click.option("--write-corrected", is_flag=True,
              help="With --qc: also the corrected series, its b-values and rotated b-vectors, into that folder.")
@click.option("--shell", type=float, default=None,
              help="The b-value tracked (default: the shell with the most volumes). The ORG atlas the labelers follow "
                   "was built on b = 3000: on a multi-shell scan, 3000 or the nearest shell matches it.")
@click.option("--labeler", type=click.Choice(["rapidparc", "hemiaug", "tractcloud"]), default="rapidparc",
              show_default=True, help="RapidParc, its model trained for lesions and surgery, or TractCloud (its code "
                                      "and model fetched from its GitHub on first use, about 8.4 MB).")
@click.option("--device", type=click.Choice(["auto", "mps", "cuda", "cpu"]), default="auto", show_default=True,
              help="Where to run: auto picks the Apple GPU, then CUDA, then the CPU.")
@click.option("--workers", type=int, default=None,
              help="On the CPU: tracking processes (default: one per core; ignored on a GPU).")
@click.option("--no-memory-check", is_flag=True,
              help="With --motion-eddy on a GPU: try a series larger than its memory allows anyway.")
@click.option("--record-absolute-paths", is_flag=True,
              help="Record full file paths in the provenance file (<trx>.tractline.json), not paths relative to the "
                   "dataset: for local bookkeeping, not for sharing.")
@click.option("--json", "json_path", type=click.Path(dir_okay=False), default=None,
              help="Also write the run's summary as JSON here ('-' for standard output).")
@click.option("-q", "--quiet", is_flag=True, help="Print nothing but errors (and --json -).")
def run(dwi, partners, motion_eddy, gibbs, mask, outliers, trx, float16, labeled_only, qc, write_corrected, shell,
        labeler, device, workers, no_memory_check, record_absolute_paths, json_path, quiet):
    """Correct, track and label one dMRI series (NIfTI, BIDS). Errors end in one line; TRACTLINE_DEBUG=1 shows the
    traceback."""
    import os
    from .fetch import FetchError
    try:
        from nibabel.filebasedimages import ImageFileError
    except ImportError:                                                   # reported below, as the extra to install
        ImageFileError = ValueError                                       # noqa: N806
    try:
        _run(dwi, partners, motion_eddy, gibbs, mask, outliers, trx, float16, labeled_only, qc, write_corrected,
             shell, labeler, device, workers, no_memory_check, record_absolute_paths, json_path, quiet)
    except (ValueError, OSError, FetchError, ImageFileError) as e:        # the input, the disk, the network: one line
        if os.environ.get("TRACTLINE_DEBUG") == "1":
            raise
        hint = " (here: --device cpu, or --no-memory-check)" if "check_memory" in str(e) else ""
        raise click.ClickException(str(e) + hint) from e


def auto_shell(bval) -> float:
    """The b-value to track when none is given (pipeline.default_shell)."""
    from .pipeline import default_shell
    try:
        return default_shell(bval)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _run(dwi, partners, motion_eddy, gibbs, mask, outliers, trx, float16, labeled_only, qc, write_corrected, shell,
         labeler, device, workers, no_memory_check, record_absolute_paths, json_path, quiet):
    if (qc or write_corrected or outliers) and not motion_eddy:
        raise click.UsageError("--qc, --write-corrected and --outliers need --motion-eddy")
    if write_corrected and not qc:
        raise click.UsageError("--write-corrected writes into the --qc folder")
    if no_memory_check and not motion_eddy:
        raise click.UsageError("--no-memory-check applies to --motion-eddy")
    json_dir = [Path(json_path).parent] if json_path and json_path != "-" else []
    for folder in ([Path(trx).parent] if trx else []) + ([Path(qc)] if qc else []) + json_dir:
        folder.mkdir(parents=True, exist_ok=True)                    # before the run, not after two minutes of it
    try:
        import nibabel  # noqa: F401
        from . import bids
    except ImportError as e:
        raise click.ClickException(f"reading the series needs nibabel and scipy ({e}): install tractline[cli]")
    from . import pipeline as P
    say = (lambda *a: None) if quiet else (lambda m: click.echo(m, err=True))     # noqa: E731

    dev = _device(device)
    s = bids.load(dwi, partners=list(partners) or None)
    say(f"{s.name}: {s.dwi.shape[-1]} volumes; {s.pairing.message}")
    if shell is None:
        shell = auto_shell(s.bval)
    if labeler == "tractcloud":
        from .fetch import tractcloud as fetch_tractcloud
        fetch_tractcloud(progress_for=None if quiet else _progress)       # with progress, before the import needs it
        try:
            from .labelers import tractcloud
        except ImportError as e:                                          # a partial checkout in $TRACTOGRAPHY_DATA
            raise click.ClickException(f"TractCloud's code could not be loaded ({e}); a checkout's TractCloud/src "
                                       f"is used when present") from e
        lab = tractcloud.Labeler(dev)
    else:
        from .labelers import rapidparc
        from .fetch import weights
        weights(labeler, progress=None if quiet else _progress(f"RapidParc's {labeler} weights"))
        lab = rapidparc.Labeler(dev, model=labeler)
    me = None
    if motion_eddy:
        me = {"model": motion_eddy, **({"outliers": outliers} if outliers else {}),
              **({"check_memory": False} if no_memory_check else {})}
    gb = {"slice_axis": P.slice_axis_of(s)} if gibbs else None
    mk = None if mask == "median-otsu" else mask
    if mk:
        from .fetch import weights as fetch_weights
        fetch_weights("synthstrip", progress=None if quiet else _progress("SynthStrip's weights"))
    timer = P.Timer(echo=None if quiet else "  ", stream=sys.stderr)       # standard output stays for --json -
    from . import provenance as PV
    cfg = P.config_for(shell=shell, motion_eddy=me, gibbs=gb, mask=mk)
    say(f"running on {dev}: shell b = {shell:g}" + (", Gibbs ringing removed" if gb else "")
        + (f", head motion and eddy currents ({motion_eddy})" if me else "") + (f", {mk} mask" if mk else ""))
    for step in PV.experimental(cfg.to_dict()):
        say(f"experimental: {step}")
    r = P.run(s, lab, timer, trx=trx, device=dev, workers=workers, shell=shell, motion_eddy=me,
              gibbs=gb, mask=mk, positions="float16" if float16 else "float32", labeled_only=labeled_only,
              absolute_paths=record_absolute_paths)
    corr, tg, labels = r
    first = timer.seconds.get("gibbs", 0)                                 # before the field: not among pipeline_stages
    summary = {"series": str(dwi), "corrected": corr.applied, "note": corr.note, "residual_left": corr.residual_left,
               "warnings": corr.warnings, "shell": shell, "device": dev, "gibbs": corr.gibbs,
               "seconds": timer.seconds,
               "scan_to_labels_s": round(timer.total(*P.pipeline_stages("", cfg)) + first, 1),
               "streamlines": len(tg.fibers), "labeled": int(labels.keep.sum()), "trx": str(trx) if trx else None,
               "pipeline_id": r.provenance["pipeline_id"], "execution_id": r.provenance["execution_id"],
               "experimental": r.provenance["experimental"],
               "provenance": str(PV.sidecar_path(trx)) if trx else None}
    if qc:
        from . import motion_eddy as ME
        summary["motion_eddy"] = ME.write_qc(corr.motion_eddy, qc)
        if not ME.plot_qc(corr.motion_eddy, Path(qc) / "motion_eddy_qc.png", s.name):
            say("(no QC plot: matplotlib is not installed)")
        if write_corrected:
            _write_corrected(s, corr, Path(qc))
    for w in corr.warnings:
        say(f"warning: {w}")
    say(f"{summary['streamlines']} streamlines, {summary['labeled']} labeled; {summary['scan_to_labels_s']} s"
        + (f"; TRX {trx}" if trx else "") + (f"; QC in {qc}" if qc else ""))
    if trx:
        say(f"provenance: {PV.sidecar_path(trx)} (pipeline {r.provenance['pipeline_id'][:12]}, "
            f"execution {r.provenance['execution_id'][:12]})")
    from .motion_eddy import _finite
    text = json.dumps(_finite(json.loads(json.dumps(summary, default=str))), indent=1, allow_nan=False)
    if json_path == "-":
        click.echo(text)
    elif json_path:
        Path(json_path).write_text(text)


def _write_corrected(s, corr, folder: Path):
    """The corrected series (float32, the source's NIfTI header), its b-values and rotated b-vectors."""
    import nibabel as nib
    import numpy as np
    src = getattr(s, "dwi_path", None)
    hdr = nib.load(str(src)).header.copy() if src is not None else None
    img = nib.Nifti1Image(corr.dwi, s.affine, hdr)
    img.set_data_dtype(np.float32)
    nib.save(img, str(folder / "corrected.nii.gz"))
    np.savetxt(folder / "corrected.bval", np.asarray(s.bval)[None], fmt="%g")
    np.savetxt(folder / "corrected.bvec", corr.bvec, fmt="%.6f")


def _progress(what: str):
    state = {"last": -1}

    def show(done, size):
        pct = int(100 * done / size) if size else 100
        if pct // 10 != state["last"] // 10:
            click.echo(f"downloading {what}: {pct} %", err=True)
            state["last"] = pct
    return show


@main.command()
@click.argument("dwi", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--partners", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="With DWI: the series that measures its field, as for run.")
@click.option("--motion-eddy", "model", type=click.Choice(MODELS), default="quadratic", show_default=True,
              help="The eddy current model to compile for.")
@click.option("--all", "everything", is_flag=True,
              help="Every phase-encoding axis, model and field case (18 combinations; about 20 CPU-minutes).")
@click.option("--workers", type=int, default=None,
              help="Compiling processes (default: one per kernel set, up to the CPUs this process may use).")
def warm(dwi, partners, model, everything, workers):
    """Compile the CUDA kernels ahead of the first run, into Triton's cache (TRITON_CACHE_DIR, by default
    ~/.triton/cache), so no run pays for compiling: the tracker's (every gradient count) and the head motion and eddy
    current correction's. With DWI: for that series' phase-encoding axis and whether it has a series that measures its
    field (a study of one protocol needs only this). Without: AP/PA phase encoding with and without a field. On an Apple
    GPU or the CPU there is nothing to compile ahead."""
    import time
    import torch
    if everything and dwi:
        raise click.UsageError("--all compiles every combination; give a series or --all, not both")
    if partners and not dwi:
        raise click.UsageError("--partners goes with a series")
    if not torch.cuda.is_available():
        click.echo("no CUDA GPU: nothing to compile ahead (Apple GPUs and the CPU compile nothing that is cached)",
                   err=True)
        return
    from . import motion_eddy_triton as T
    if not T.available():
        raise click.ClickException("Triton is not installed, or this GPU is older than it supports (compute capability "
                                   "7.0): on Linux, PyPI's torch brings Triton")
    from . import ukf_triton_block as K
    if workers is None:
        workers = T.cpus()
    try:
        from nibabel.filebasedimages import ImageFileError
    except ImportError:
        ImageFileError = ValueError                                       # noqa: N806
    try:
        if dwi:
            from . import bids, pipeline as P
            s = bids.load(dwi, partners=list(partners) or None)
            which = T.combos((s.pe_axis,), (model,), (P.can_correct(s),))
            what = f"{s.name}: PE axis {s.pe_axis}, {model}, {'with' if P.can_correct(s) else 'without'} a field"
        elif everything:
            which = T.combos((0, 1, 2), tuple(T.MODELS), (True, False))
            what = "every combination"
        else:
            which = T.combos((1,), (model,), (True, False))
            what = f"AP/PA phase encoding, {model}, with and without a field"
    except ImportError as e:
        raise click.ClickException(f"reading a series needs nibabel and scipy ({e}): install tractline[cli]") from e
    except (ValueError, OSError, ImageFileError) as e:
        raise click.ClickException(str(e)) from e
    from concurrent.futures.process import BrokenProcessPool
    t0 = time.perf_counter()
    try:
        click.echo(f"the tracker: {len(K.NPADS)} gradient-count sizes", err=True)
        K.warm(workers=min(workers, len(K.NPADS)))
        click.echo(f"head motion and eddy currents: {what} ({len(which)} kernel set{'s' if len(which) > 1 else ''})",
                   err=True)
        T.warm(which, workers=min(workers, len(which)))
    except BrokenProcessPool as e:
        raise click.ClickException("a compiling process died (memory, or a GPU in exclusive-process mode): try "
                                   "--workers 1") from e
    click.echo(f"compiled into Triton's cache in {time.perf_counter() - t0:.0f} s", err=True)


@main.command()
@click.option("--model", type=click.Choice(["rapidparc", "hemiaug", "tractcloud", "synthstrip", "all"]), default="all",
              show_default=True)
def fetch(model):
    """Download the weights into the cache now (they are otherwise fetched on first use): RapidParc's two models (6.6
    MB each), TractCloud's code and model (about 8.4 MB) and SynthStrip's model (31 MB, for --mask synthstrip), each
    sha256-checked."""
    from . import fetch as FE
    for name in (["rapidparc", "hemiaug", "tractcloud", "synthstrip"] if model == "all" else [model]):
        try:
            if name == "tractcloud":
                src, mod, centroid = FE.tractcloud(progress_for=_progress)
                click.echo(f"tractcloud: {src}, {mod}, {centroid}")
            else:
                click.echo(f"{name}: {FE.weights(name, progress=_progress(name))}")
        except (FE.FetchError, OSError) as e:                               # the network, or the cache's folder
            raise click.ClickException(str(e)) from e


DOCS = {"readme": "README.md", "guide": "motion-eddy-qc-guide.md"}


def _doc(name: str) -> str:
    """A document's Markdown: the copy inside the installed package (tractline/_docs, put there by the wheel build), else
    the repository's (a checkout)."""
    from importlib.resources import files
    packaged = files("tractline") / "_docs" / DOCS[name]
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    local = root / (DOCS[name] if name == "readme" else f"docs/{DOCS[name]}")
    if local.is_file():
        return local.read_text(encoding="utf-8")
    raise click.ClickException(f"{DOCS[name]} is not in this installation")


def _helps() -> str:
    """Every command's --help, as the terminal shows it."""
    out = []
    for name, cmd in [("", main)] + sorted(main.commands.items()):
        prog = f"tractline {name}".strip()
        with cmd.make_context(prog, [], resilient_parsing=True) as ctx:
            out.append(f"## {prog} --help\n\n```\n{cmd.get_help(ctx)}\n```")
    return "\n\n".join(out)


@main.command()
@click.argument("what", type=click.Choice(["all", "readme", "guide", "help"]), default="all")
def docs(what):
    """Print tractline's documentation as one Markdown text, for reading or for an LLM: its version, every command's
    help, the README and the QC guide (all: the default), or one of them. Pipe it on, e.g. `tractline docs | pbcopy`."""
    parts = []
    if what in ("all", "help"):
        parts.append(f"# tractline {_version()}: the commands\n\n{_helps()}")
    if what in ("all", "readme"):
        parts.append(_doc("readme"))
    if what in ("all", "guide"):
        parts.append(_doc("guide"))
    click.echo("\n\n---\n\n".join(parts))


if __name__ == "__main__":
    sys.exit(main())
