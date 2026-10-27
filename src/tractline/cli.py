"""The tractline command: a dMRI series to labeled UKF tractography, from the shell.

    tractline run sub-01/dwi/sub-01_dir-AP_dwi.nii.gz --motion-eddy --trx out/sub-01.trx --qc qc/sub-01
    tractline fetch                      # RapidParc's weights into the cache ahead of a run (tractline.fetch)

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
    if name != "auto":
        return name
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
@click.option("--shell", type=float, default=None, help="The b-value tracked (default: the shell with the most volumes).")
@click.option("--labeler", type=click.Choice(["rapidparc", "hemiaug", "tractcloud"]), default="rapidparc",
              show_default=True, help="RapidParc, its model trained for lesions and surgery, or TractCloud (its code "
                                      "and model fetched from its GitHub on first use, about 42 MB).")
@click.option("--device", type=click.Choice(["auto", "mps", "cuda", "cpu"]), default="auto", show_default=True,
              help="Where to run: auto picks the Apple GPU, then CUDA, then the CPU.")
@click.option("--workers", type=int, default=None,
              help="On the CPU: tracking processes (default: one per core; ignored on a GPU).")
@click.option("--no-memory-check", is_flag=True,
              help="With --motion-eddy on an Apple GPU: try a series larger than the recommended memory anyway.")
@click.option("--json", "json_path", type=click.Path(dir_okay=False), default=None,
              help="Also write the run's summary as JSON here ('-' for standard output).")
@click.option("-q", "--quiet", is_flag=True, help="Print nothing but errors (and --json -).")
def run(dwi, partners, motion_eddy, outliers, trx, float16, labeled_only, qc, write_corrected, shell, labeler, device,
        workers, no_memory_check, json_path, quiet):
    """Correct, track and label one dMRI series (NIfTI, BIDS). Errors end in one line; TRACTLINE_DEBUG=1 shows the
    traceback."""
    import os
    from .fetch import FetchError
    try:
        from nibabel.filebasedimages import ImageFileError
    except ImportError:                                                   # reported below, as the extra to install
        ImageFileError = ValueError                                       # noqa: N806
    try:
        _run(dwi, partners, motion_eddy, outliers, trx, float16, labeled_only, qc, write_corrected, shell, labeler,
             device, workers, no_memory_check, json_path, quiet)
    except (ValueError, OSError, FetchError, ImageFileError) as e:        # the input, the disk, the network: one line
        if os.environ.get("TRACTLINE_DEBUG") == "1":
            raise
        hint = " (here: --device cpu, or --no-memory-check)" if "check_memory" in str(e) else ""
        raise click.ClickException(str(e) + hint) from e


def auto_shell(bval) -> float:
    """The b-value to track when none is given: the shell with the most volumes (motion_eddy.shell_groups, the larger
    b on a tie), as the median of its own b-values, so a shell stored as 1050 or jittered about 1150 is tracked at its
    own value, inside the tracker's +-50 selection, not at a rounded one."""
    import numpy as np
    from .motion_eddy import shell_groups
    b = np.asarray(bval, float)
    groups, nominal = shell_groups(b)
    dw = [(len(g), float(np.median(b[g]))) for g, n in zip(groups, nominal) if n > 0]
    if not dw:
        raise click.ClickException("the series has no diffusion-weighted volumes")
    return max(dw)[1]


def _run(dwi, partners, motion_eddy, outliers, trx, float16, labeled_only, qc, write_corrected, shell, labeler, device,
         workers, no_memory_check, json_path, quiet):
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
    timer = P.Timer(echo=None if quiet else "  ", stream=sys.stderr)       # standard output stays for --json -
    say(f"running on {dev}: shell b = {shell:g}" + (f", head motion and eddy currents ({motion_eddy})" if me else ""))
    corr, tg, labels = P.run(s, lab, timer, trx=trx, device=dev, workers=workers, shell=shell, motion_eddy=me,
                             positions="float16" if float16 else "float32", labeled_only=labeled_only)
    cfg = P.config_for(motion_eddy=me)
    summary = {"series": str(dwi), "corrected": corr.applied, "note": corr.note, "residual_left": corr.residual_left,
               "warnings": corr.warnings, "shell": shell, "device": dev,
               "seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages("", cfg)),
               "streamlines": len(tg.fibers), "labeled": int(labels.keep.sum()), "trx": str(trx) if trx else None}
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
@click.option("--model", type=click.Choice(["rapidparc", "hemiaug", "tractcloud", "all"]), default="all",
              show_default=True)
def fetch(model):
    """Download the labelers' weights into the cache now (they are otherwise fetched on first use): RapidParc's two
    models (6.6 MB each) and TractCloud's code and model (about 42 MB), each sha256-checked."""
    from . import fetch as FE
    for name in (["rapidparc", "hemiaug", "tractcloud"] if model == "all" else [model]):
        try:
            if name == "tractcloud":
                src, mod, centroid = FE.tractcloud(progress_for=_progress)
                click.echo(f"tractcloud: {src}, {mod}, {centroid}")
            else:
                click.echo(f"{name}: {FE.weights(name, progress=_progress(name))}")
        except (FE.FetchError, OSError) as e:                               # the network, or the cache's folder
            raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    sys.exit(main())
