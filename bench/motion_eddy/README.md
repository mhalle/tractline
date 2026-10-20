# Motion and eddy-current correction: prototype and checks

These are the experiments behind `docs/gpu-motion-eddy-design.md`. They are a prototype, not pipeline code, and
nothing in `src/` uses them. Results are in `bench/results/motion_eddy/`. Data is in `$TRACTOGRAPHY_DATA`, and the
synthetic scan with its eddy runs is in `derived/motion_eddy_synth/`.

| script | what | doc |
|---|---|---|
| `proto.py` | the estimator on a synthetic scan with known motion and eddy fields: batched per-volume Levenberg-Marquardt, leave-one-out SH prediction; `--save-synth` writes the scan for eddy | 2.1 |
| `compare_eddy.py` | eddy's `--dfields` and rotated b-vectors scored against the truth, as `proto.py` scores itself | 2.1 |
| `bench_pieces.py` | the time of one Gauss-Newton iteration's pieces on a device, at 96 x 96 x 50 x 100 and 104 x 104 x 72 x 105 | 3 |
| `bench_fast.py` | faster moments, and the cubic resample as 8 trilinear fetches (checked against the 64-tap sum) | 3 |
| `bench_moments.py` | moment layouts on MPS | 3 |
| `cpu_timing.py`, `headers.py` | the first CPU-only timing; the data headers | 3, 6.3 |

Reproduce (MPS; the GPU should be otherwise idle for timings):

    uv run --group dipy bench/motion_eddy/proto.py --device mps --acq-cost --cubic --outer 15 --inner 1   # final form: 0.103 voxels
    uv run --group dipy bench/motion_eddy/proto.py --device mps                                           # first form: 0.284
    uv run --group dipy bench/motion_eddy/proto.py --device mps --save-synth $TRACTOGRAPHY_DATA/derived/motion_eddy_synth

eddy (`fsl-eddy` 2602.0 in `$TRACTOGRAPHY_DATA/fsl-env`, installed by the standalone micromamba in `tools/bin`, as
`fsl-topup` was), from the synthetic scan's folder:

    export FSLDIR=$TRACTOGRAPHY_DATA/fsl-env FSLOUTPUTTYPE=NIFTI_GZ PATH=$FSLDIR/bin:$PATH
    eddy_cpu --imain=imain --mask=mask --acqp=acqp.txt --index=index.txt --bvecs=bvecs --bvals=bvals \
      --out=eddy_slm_none --flm=quadratic --slm=none --dfields --nthr=6 --data_is_shelled     # 36 min on 6 M2 threads
    uv run --group dipy bench/motion_eddy/compare_eddy.py $TRACTOGRAPHY_DATA/derived/motion_eddy_synth/eddy_slm_none

The saved eddy command lines are in `results/motion_eddy/eddy_synth_*.command.txt`. Seeds are fixed (`--seed 0`), so
`--save-synth` regenerates the same scan.
