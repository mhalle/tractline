# Working on tractline

## Setup

tractline is a uv project. Python's version is pinned in `.python-version` (3.12) and every dependency in `uv.lock`.

```
git clone https://github.com/mhalle/tractline && cd tractline
git switch dev
uv sync                             # the package, plus the dev group: pytest, ruff and the bench's packages
uv sync --extra triton              # + the tracker's Triton kernel on CUDA (Linux with an NVIDIA GPU)
uv run tractline --version          # the command, from the checkout
```

`uv run` syncs the environment before it runs anything, so after a pull `uv run pytest` is enough. Add a dependency
with `uv add` (or `uv add --group dev` for one only the bench needs), never by editing the environment by hand, so
that `uv.lock` stays the record.

Beyond the `dev` group, a few dependency groups are opt-in, each for a few bench scripts:

| group | for |
|---|---|
| `dipy` | DIPY as the mask's reference (`median_check.py`), the bootstrap's spherical harmonics, and the denoise slot's Patch2Self (with scikit-learn); CI installs it |
| `vtk` | VTK tractography files (`ukf_compare.py`, `ukf_bench.py`, `resample_check.py`); 0.5 GB |
| `sklearn` | `accuracy_tractcloud_test.py` |
| `rankfield` | `ensemble_pilot.py`, `ensemble_votes_test.py` (Python 3.12 or later) |
| `store` | `ensemble_store.py`: ensemble stores as duckn and zarr (Python 3.12 or later) |

For example, `uv run --group vtk bench/ukf_compare.py`. `rapidparc_check.py` runs RapidParc's own package in an
isolated environment uv builds for it (`uv run --with RapidParc==1.0.4`), since its pins do not fit the project's.

## Data

Scans, weights and other data live outside the repository, in `$TRACTOGRAPHY_DATA` (default
`~/tmp/data/tractography`; the layout is in `src/tractline/data.py`). An installed tractline does not need it: the
labelers' weights are downloaded on first use into a user cache (`tractline.fetch`; `$TRACTLINE_CACHE` moves the
cache, `TRACTLINE_OFFLINE=1` refuses to download). In a checkout, copies under `$TRACTOGRAPHY_DATA` are used first.

- **RapidParc's weights** (the default labeler): `RapidParc/rapidparc.safetensors` and `hemiaug.safetensors`,
  6.6 MB each, from [RapidParc's release](https://github.com/MedVisBonn/RapidParc/releases/tag/v1.0.0), checked
  against their sha256 when loaded. The tests use `rapidparc.safetensors` and skip without it.
- **TractCloud** (the optional labeler): its code in `TractCloud/src`
  ([SlicerDMRI/TractCloud](https://github.com/SlicerDMRI/TractCloud)) and its weights `TrainedModel/` from its v1.0.0
  release (its atlas centroid ships with tractline; the bench's own scripts read `TrainData_800clu800ol/` too).
- **The bench's data**: OpenNeuro ds001226 (BTC_preop, CC0) in `ds001226/`; ON-Harmony (ds004712) in `ds004712/`;
  the Stanford HARDI scan in `ukf/hardi/`; TractCloud's test split and HCP test tractogram from its release;
  dcm2niix's validation DICOM in `dicom/`. FSL's topup (for `topup_ref.py` and `cohort_topup.py`) in `fsl-env/`,
  from FSL's conda channel; FSL is not a Python package.

## Tests

```
uv run ruff check src bench tests   # undefined, shadowed and unused names
uv run pytest                       # about ten minutes on an M2
```

The suite runs the whole pipeline on a synthetic phantom (`tests/phantom.py`), checks each module against its
reference (DIPY, the torch float64 paths, the Slicer binary's conventions), the command end to end, the package with
numpy and torch alone, and the inputs the modules refuse. Tests that need an Apple GPU or CUDA skip without one.

GitHub's CI (`.github/workflows/ci.yml`) runs lint and the suite on Linux, on the CPU, on every push to `dev` and
`main` and on every pull request. The GPU paths need their GPUs:

- **Apple GPU (Metal)**: run the suite on a Mac.
- **CUDA (Triton)**: `modal run bench/motion_eddy/modal_cuda.py --tests` runs the GPU tests (motion and eddy, CPU
  against GPU, Gibbs) on an NVIDIA A10 through [Modal](https://modal.com) (a uv tool: `uv tool install modal`) and
  exits non-zero on a failure. The workflow `.github/workflows/gpu.yml` runs it on every push to `dev` once the
  repository has the secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (from `modal token new`); until then it
  notes that it did not run, and succeeds. It is not yet a required check: with the secrets set, making it one
  in the repository's branch rules gates releases on it. The `bench/modal_*.py` scripts
  time and check the rest of the pipeline on CUDA. From Python, `ukf_triton_block.warm()` and
  `motion_eddy_triton.warm()` compile the CUDA kernels ahead, as `tractline warm` does.

## Branches and releases

- **`dev`** is where work lands. Commit and push there; CI runs on every push.
- **`main`** is the newest release. It moves only when a release is made, so users who install with no version, or
  with `@main`, get code that has passed the release checks.
- **Tags** (`v0.5.1`, ...) are the releases.

To make a release:

1. On `dev`, with CI green, run the release checks: the suite on a Mac (Metal), the CUDA tests on Modal, and one real
   scan through `tractline run` (ON-Harmony 16974 with `--motion-eddy --gibbs`, compared with
   `bench/results/motion_eddy/m2_cli_gibbs_16974.json`).
2. Set the version in `src/tractline/__init__.py`, turn `CHANGELOG.md`'s "Unreleased" into the release, and update
   the version in the README's examples. Commit to `dev` and let CI pass on that commit.
3. With `dev` pushed and its CI green on the release commit, tag it and move `main` to it:
   `git tag -a vX.Y.Z -m "..."`, then `git push origin dev:main vX.Y.Z`. `main` only ever fast-forwards to a
   commit already on `dev`; nothing is committed on `main` itself.

## The package (`src/tractline`)

`pipeline.py` runs the pipeline in memory: correct, track, label, then optionally write TRX. Its docstring states the
subject it takes and the conventions every module follows (array layouts, units, devices). Two defaults to know:
`device="mps"` (pass `"cuda"` or `"cpu"` elsewhere) and `shell=2800.0` (pass the scan's own b-value). A script that
uses the CPU path needs an `if __name__ == "__main__":` guard, because the tracker spawns worker processes that
re-import the script; the tracker refuses to run without one. On the CPU the workers share their input through
shared memory; `TRACTOGRAPHY_SHARE=0` sends copies instead, for systems without it.

| module | what it does |
|---|---|
| `cli.py` | the `tractline` command (`run`, `fetch`, `warm`, `docs`) |
| `config.py` | the pipeline as a configuration: slots, their implementations, presets, and the config's hash |
| `bids.py` | a BIDS diffusion series as the pipeline's subject, with the series that correct it found and checked |
| `susceptibility.py` | the susceptibility field: FSL topup's model by Gauss-Newton, on the GPU or the CPU |
| `motion_eddy.py`, `motion_eddy_metal.py`, `motion_eddy_triton.py`, `mi.py` | head motion and eddy current correction, with Metal and Triton kernels |
| `gibbs.py`, `denoise.py` | Gibbs ringing removal (DIPY's method in torch); denoising through DIPY (Patch2Self, MP-PCA) |
| `prep.py`, `mask.py` | the tracker's input: one shell, gradients in RAS, DIPY's `median_otsu` mask (exactly, in torch) |
| `ukf.py`, `ukf_metal.py`, `ukf_triton_block.py` | UKF two-tensor tractography as the Slicer binary does it, with Metal and Triton kernels (`ukf_triton.py`, the first attempt, compiles too slowly to use) |
| `csd.py`, `ptt.py`, `ptt_metal.py`, `fields.py`, `sh.py`, `nearest.py` | CSD and parallel transport tractography (experimental) |
| `labelers/` | RapidParc (the default) and TractCloud; `base.py` holds what they share (the 40 mm cut, the 43-class scheme) |
| `resample.py` | TractCloud's streamline features |
| `trx.py` | the tractogram as TRX, with tract labels and probabilities |
| `provenance.py` | the run record written beside every TRX (`<name>.tractline.json`): the pipeline and execution IDs, steps' statuses, versions, device, inputs' and output's checksums |
| `fetch.py`, `data.py` | the weights downloaded on first use; where data lives |
| `ensemble.py`, `compact.py` | uncertainty from repeated runs, and compact run records (prototypes) |
| `t1check.py` | a measurement, not a pipeline step: the distortion left against the T1 (the `t1check` extra) |

## Further documents

- `docs/sota-pipeline-design.md`: the configurable pipeline's design.
- `docs/gpu-motion-eddy-design.md`: the head motion and eddy current correction's design and its checks.
- `docs/uncertainty-ensemble.md`, `docs/robustness-plan.md`, `docs/data-product.md`: the uncertainty and robustness
  work.
- `docs/validation-datasets.md`, `docs/prior-art.md`, `docs/prior-art-uncertainty.md`: data and literature.
- `docs/duckn-proposal.md`, `docs/dcm2niix-dmri-map.md`: what a duckn store would need to carry for tractline, and
  where dcm2niix finds each vendor's diffusion facts.
- `bench/results/NOTES.md`: the running journal of every experiment, newest last.
