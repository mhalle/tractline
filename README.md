# tractline

tractline turns a diffusion MRI scan into labeled white-matter tractography. One command corrects the scan's
distortions, runs two-tensor UKF tractography, and assigns each streamline to one of 42 white-matter tracts, or to
"Other". The tracts are those defined on the fiber clusters of the ORG atlas (Zhang et al. 2018), and the labels come
from deep-learning tract labelers (RapidParc by default, or TractCloud). tractline runs on an Apple GPU, an NVIDIA GPU
or the CPU, and writes a TRX file, an open tractography format that other tools read.

tractline is research software under active development. It is not reviewed or approved by any regulatory agency
and is not for clinical use.

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and git. This runs tractline on one diffusion
series without a separate install step:

```
uvx --from "tractline[cli] @ git+https://github.com/mhalle/tractline" \
    tractline run sub-01/dwi/sub-01_dir-AP_dwi.nii.gz --motion-eddy --trx out/sub-01.trx --qc qc/sub-01
```

This corrects the series for head motion and eddy currents, and for susceptibility distortion when it finds a series
acquired with the opposite phase encoding (see "What it reads"). It then tracks, labels, and writes:

- `out/sub-01.trx`: every streamline, in millimeters, grouped by tract. Each one also carries its tract label, the
  probability of that label and its margin over the next most likely tract, its atlas cluster, its length and its
  seed. Streamlines shorter than 40 mm are kept but belong to no tract.
- `qc/sub-01/`: the head motion and eddy current correction's QC: a table per volume (`motion_eddy_qc.tsv`), a
  summary (`motion_eddy_qc.json`) and a plot (`motion_eddy_qc.png`).

Without `--trx`, nothing is saved; `--json summary.json` also saves a summary of the run. The first run takes longer:
uv builds an environment for tractline, which includes PyTorch (a few hundred MB, a few GB on Linux with its CUDA
libraries), and tractline downloads its labeler's weights (6.6 MB). Both are cached, so later runs start at once.

**A shortcut for regular use.** `uvx` is uv's short form of `uv tool run`: it runs a tool from a cached environment
without installing it. If you use tractline often, install it once as a tool, and `tractline` becomes an ordinary
command:

```
uv tool install "tractline[cli] @ git+https://github.com/mhalle/tractline"
tractline run sub-01/dwi/sub-01_dir-AP_dwi.nii.gz --motion-eddy --trx out/sub-01.trx --qc qc/sub-01
uv tool upgrade tractline           # later: move to the newest release
uv tool uninstall tractline         # remove it
```

If the shell then cannot find `tractline`, `uv tool update-shell` adds uv's tool folder to your PATH.

`tractline run --help` lists every option. `tractline docs` prints this README, the user guide and the help of every
command as one Markdown text, which is handy to give to an LLM.

## What it reads

A diffusion series as NIfTI (`.nii` or `.nii.gz`) with its `.bval`, `.bvec` and JSON sidecar, usually in a BIDS
folder. The sidecar supplies what the corrections need: the phase-encoding direction, the readout time, the echo time
and the shim settings. tractline tracks one shell: the one with the most volumes, unless `--shell` names another. The
ORG atlas the labelers follow was built on b = 3000, so on a multi-shell scan `--shell 3000` (or the nearest shell)
matches it (the [user guide](docs/motion-eddy-qc-guide.md), "Tractography input").

The susceptibility correction needs a second series acquired with the opposite phase encoding, such as an AP/PA pair.
tractline finds it from the sidecars (`B0FieldIdentifier`, `IntendedFor`, or a reversed diffusion series in the same
folder), checks that the two really match, and tells you why if it cannot use them. You can also name it with
`--partners`. Without a usable pair, the scan is tracked without that correction, and the output says so.

**From DICOM**, convert with [dcm2niix](https://github.com/rordenlab/dcm2niix) first. It also runs with `uvx`:

```
mkdir -p nifti/sub-01
uvx dcm2niix@1.0.20260724 -b y -z y -f %p_%s_dwi -o nifti/sub-01 dicom/sub-01/
uvx --from "tractline[cli] @ git+https://github.com/mhalle/tractline" \
    tractline run nifti/sub-01/DTI_AP_10_dwi.nii.gz --motion-eddy --trx out/sub-01.trx
```

The file names end in `_dwi` so that a reversed series in the same folder is found and paired. Section 2 of the
[user guide](docs/motion-eddy-qc-guide.md) covers the cases that need a step more, such as a spin-echo field map as
the reversed series.

## What it does

1. **Gibbs ringing removal** (`--gibbs`, optional), DIPY's method, on the series as acquired, before the other steps.
2. **Susceptibility correction** with FSL topup's model, estimated from the b0 images of both phase-encoding
   directions. It agrees with topup and takes about 15 seconds on an Apple GPU, where topup takes about 11 minutes.
3. **Head motion and eddy current correction** (`--motion-eddy`, optional): FSL eddy's model with tractline's own
   fit, combined with the susceptibility correction into a single resampling of each volume. Slices with signal
   loss are replaced, as eddy's `--repol` does.
4. **Tractography**: two-tensor UKF tractography as Slicer's UKFTractography does it, with the settings used to build
   the ORG atlas: a seed in every voxel of the brain mask whose FA is above 0.1. The mask is DIPY's `median_otsu` of
   the mean b0, or SynthStrip's (`--mask synthstrip`, experimental), which kept the lower brainstem on ON-Harmony's
   scans, where `median_otsu` cut through it.
5. **Tract labels** from RapidParc (the default; 94.5 % tract accuracy on TractCloud's labeled test set) or
   TractCloud (`--labeler tractcloud`).

Each step is tractline's own implementation, checked against the original tool. The
[user guide](docs/motion-eddy-qc-guide.md) explains each one and how to read the QC.

Times for one scan (ON-Harmony sub-16974: 105 volumes at 2 mm) with `--motion-eddy --gibbs`, from the scan to
labeled streamlines:

| hardware | time |
|---|---|
| Apple M2 laptop (GPU) | 153 s |
| NVIDIA A10 | 36 s |

On an NVIDIA GPU the first run also compiles the GPU kernels, about two minutes. They are kept in Triton's cache
(`~/.triton/cache`), and `tractline warm` compiles them ahead of time. On the CPU every step runs, but slowly: the
motion and eddy correction there is estimated at 12 to 13 minutes, and has not yet been run end to end on a real
scan. A series too large for the GPU's memory (an HCP-sized acquisition on a 16 GB laptop, for example) is refused
before the run starts, with a message.

**On a cluster:** `tractline fetch` downloads the weights ahead of time, for nodes without internet access
(`TRACTLINE_CACHE` sets where they are kept; `TRACTLINE_OFFLINE=1` forbids downloads). Set `TRITON_CACHE_DIR` to a
folder the nodes share, so the GPU kernels are compiled once for all of them.

## Using the output

The TRX file is a zip of arrays, readable with [trx-python](https://github.com/tee-ar-ex/trx-python):

```python
from trx.trx_file_memmap import load

t = load("out/sub-01.trx")
print(sorted(t.groups))                     # the tract names
af = t.streamlines[t.groups["AF"]]          # the arcuate fasciculus's streamlines, in RAS millimeters
```

trx-python's `trx_convert_tractogram` converts it to `.tck`, `.trk` or `.vtk` for tools that do not read TRX.

Beside each TRX file, `out/sub-01.tractline.json` records how it was made: the pipeline's configuration, each step's
status (validated or experimental), the software versions, the device and GPU code each step ran, and the checksums
of the inputs, the labeler's weights and the TRX itself. Its `pipeline_id` names the algorithm and settings, the same
on any machine; its `execution_id` also covers the device, versions and weights, so two runs on the same input with
the same `execution_id` should give the same streamlines; it covers the operating system's build and, on NVIDIA GPUs,
the CUDA runtime and driver. Files are identified by their checksums, and their paths are recorded relative to the
BIDS dataset (or by file name), so the record can be shared without your folder names; `--record-absolute-paths`
records full paths for local bookkeeping.

## Choosing a version

- **No `@...`**, or `@main`: the newest release. The `main` branch moves only when a release is made.
- **A tag** such as `@v0.7.2`: one release, permanently. Pin a tag when a study's scans must all run on the same
  code: `uvx --from "tractline[cli] @ git+https://github.com/mhalle/tractline@v0.7.2" tractline ...`, or the same
  URL with `uv tool install`. `tractline --version` shows which version ran.
- **`@dev`**: the newest code, ahead of the latest release.

**For a study that must be reproducible exactly,** pin the dependencies too, not only tractline. A tag fixes
tractline's code, but `uvx` and `uv tool install` resolve the newest versions of PyTorch and the rest. The repository's
`uv.lock` pins every package, and running from a clone of the tag uses it:

```
git clone --branch v0.7.2 https://github.com/mhalle/tractline && cd tractline
uv run --locked --no-dev --extra cli tractline run ../sub-01/dwi/sub-01_dir-AP_dwi.nii.gz --motion-eddy --trx ../out/sub-01.trx
```

The provenance file also records the lock's checksum, so the environment can be rebuilt later.

`uvx` reuses the environment it built first, so after a new release `uvx --refresh --from ...` picks it up. An
unpinned `uv tool install` moves to the newest release with `uv tool upgrade tractline`; a pinned one stays on its
tag until you install again with another. [CHANGELOG.md](CHANGELOG.md) lists what each release changed.

## Using it from Python

Add tractline to a uv project, pinned to a release, with the `bids` extra for reading NIfTI and BIDS:

```
uv add git+https://github.com/mhalle/tractline --tag v0.7.2 --extra bids
```

Outside uv, `pip install "tractline[bids] @ git+https://github.com/mhalle/tractline@v0.7.2"` does the same.

```python
from tractline import bids, pipeline as P

s = bids.load("sub-01/dwi/sub-01_dir-AP_dwi.nii.gz")
print(s.pairing.message)                        # which series correct it, or why none can
r = P.run(s, None, P.Timer(), motion_eddy=True, trx="out/sub-01.trx")
corr, tractogram, labels = r                    # r.provenance: the record also written beside the TRX
```

As with the command, the device (`device=`) and the shell tracked (`shell=`) are chosen automatically unless given. On
the CPU, put the code under `if __name__ == "__main__":`: the tracker starts worker processes that re-import the
script.

The core package needs numpy and torch (and click, for the command). The extras add what the optional parts need:

| extra | for |
|---|---|
| `bids` | reading NIfTI and BIDS (nibabel, scipy) |
| `cli` | the `tractline` command (nibabel, scipy, matplotlib) |
| `plot` | the QC plot (matplotlib) |
| `nrrd` | the tracker's input and output as NRRD files (pynrrd) |
| `triton` | the GPU kernels on Linux with an NVIDIA GPU (PyPI's torch for Linux already includes it) |

The pipeline's steps can also be chosen one by one through a configuration (`tractline.config`), including opt-in
methods that the command does not expose yet: denoising (Patch2Self or MP-PCA) and CSD with parallel transport
tractography. See [docs/sota-pipeline-design.md](docs/sota-pipeline-design.md).

## Documentation

- [User guide](docs/motion-eddy-qc-guide.md): running tractline, what each step does, and reading the motion and eddy
  QC.
- [CHANGELOG.md](CHANGELOG.md): what changed in each release.
- [docs/labelers.md](docs/labelers.md): RapidParc and TractCloud as tractline runs them.
- [docs/pipeline.md](docs/pipeline.md): the pipeline as built, with timings.
- [docs/development.md](docs/development.md): working on tractline: setup, tests, data, releases.
- [bench/README.md](bench/README.md): the scripts behind the measurements, and their results.

## Development

```
git clone https://github.com/mhalle/tractline && cd tractline
git switch dev                      # work happens on dev; main holds the newest release
uv sync                             # the package and the development tools
uv run pytest                       # the test suite (about ten minutes)
uv run tractline run ...            # the command from the checkout
```

[docs/development.md](docs/development.md) has the rest: the data the tests and the bench use, the optional
dependency groups, the GPU tests and the release steps.

## Credits and license

Ron Kikinis contributed to tractline. tractline implements methods from FSL (topup, eddy), Slicer's UKFTractography,
DIPY, SynthStrip, RapidParc and TractCloud; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) credits them, including
the two files tractline redistributes (RapidParc's tract scheme and TractCloud's atlas centroid). tractline's own code
is under the Apache License 2.0 ([LICENSE](LICENSE)).
