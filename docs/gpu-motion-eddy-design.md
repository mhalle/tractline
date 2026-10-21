# GPU motion and eddy-current correction: design

Status: design, with a prototype on synthetic data (2026-10-20); nothing in the pipeline. Its open choices were settled by the author at the owner's
request ("Decisions taken"). Branch: `motion-eddy`, from `ismrm2027`. The pipeline is untouched; the
measured numbers come from the prototype and benchmarks in `bench/motion_eddy/` (results in
`bench/results/motion_eddy/`), everything else is an estimate with its assumptions stated.

## Summary

tractline corrects one static susceptibility field per scan (`susceptibility.estimate`, topup's model on the GPU) and
applies it to every diffusion volume. It does not register the diffusion volumes to each other, model eddy-current
fields, or rotate b-vectors (`docs/pipeline.md`, "Open"). FSL eddy does all three (Andersson & Sotiropoulos 2016).
This document proposes an opt-in step, `motion_eddy=`, that does the same in tractline's style: one batched
Levenberg-Marquardt Gauss-Newton fit over all volumes at once, in PyTorch, on MPS, CUDA or the CPU.

- **Model.** Each volume gets 6 rigid parameters and an eddy-current field in Hz in the scanner frame: a polynomial
  that is linear (4 parameters) or quadratic (10). eddy's own paper finds linear insufficient and quadratic
  substantially better, and quadratic is eddy's default. Both orders are the same code (a monomial basis), so this
  design moves quadratic into phase 1, with the second-level model fitted jointly (decision 7). The static field `h` stays in the head's frame. One sampling
  map per volume composes motion, `h` and the eddy field, followed by the Jacobian. The rigid rotation alone rotates
  the b-vector.
- **Reference.** Each volume is registered to a leave-one-out prediction from all the other volumes, iterated with
  resampling. Phase 1 uses Laplace-Beltrami-regularized spherical harmonics per shell. This is the model the
  project's bootstrap already fits. For a fixed gradient table, both it and eddy's Gaussian process are a G x G
  linear smoother applied to every voxel, so each prediction is one matmul.
- **Cost, measured on the M2** (section 3; the GPU was shared with image generation, so these are upper bounds):
  - one batched Gauss-Newton iteration in plain torch takes 0.40 s on a 2.5 mm, 100-volume scan and 0.74 s on a 2 mm,
    105-volume scan, about 2x the earlier estimate;
  - the prototype's whole estimate takes 14 s for 50 volumes;
  - the final cubic resample takes 2.9 s / 5.2 s, as eight trilinear fetches.

  The final form (below) took 103 s at first. Plain-torch fixes cut it to 20 s for 50 volumes. **Fused Metal kernels
  (`bench/motion_eddy/metal_step.py`) cut it to 2.4 s for 15 steps, or 4.0 s run to convergence, at eddy's accuracy**
  (section 3). **Measured on the full 105-volume scan (5 b0s, b1000, b2000): 6.9 s for 20 steps, including the final
  resample (0.32 s, also a Metal kernel)**, under the field estimate's 18 s. The CUDA (Triton) port is still to write.
- **The shells need tying together, and now are** (sections 2.2-2.3). Each shell's common transform is
  unobservable from its own leave-one-out prediction: its rigid position and its common eddy field along PE. An MI
  registration of each shell's corrected mean to the corrected mean b0 (`mi_torch.py`, on the GPU) with those 16
  parameters brings the between-shell error to eddy's level: 0.013 against 0.015 voxels on an aligned 105-volume
  synthetic scan. The fit plus anchor takes 11.7 s on the M2, against eddy's 78 min.
- **What remains is within-shell precision**: 0.086 voxels against eddy's 0.051 on that scan. eddy's predictor
  draws on all shells at once; ours uses each shell alone (decision 1).
- **On a real scan** (ON-Harmony sub-16974, 105 volumes, no field for either method; section 2.4), the fit agrees
  with eddy to 0.075 voxels RMS at the edge, in 18 s against eddy's 79 min. A rigid shell anchor by dipy's MI gives
  eddy's model-fit numbers almost exactly. Two cautions:
  - **The GPU MI is biased on real cross-contrast images.** It moved the b0s about 1.1 voxels off eddy's and lowered
    FA by 0.05. It is not usable until it matches dipy's MI there.
  - **The eddy-term anchor**, which fixed the synthetic scan, has no support on real data. There the rigid anchor is
    the evidence-backed choice, as in eddy.
- **Prototype against eddy, on one synthetic scan with known motion and eddy** (section 2.1). FSL eddy (`eddy_cpu`
  2602.0, installed natively) gets the edge displacement from 1.52 voxels RMS to 0.105, with b-vector rotation error
  0.13 deg, in 36 min on 6 CPU threads. The prototype's first form stopped at 0.24-0.28 voxels. Three changes bring it
  to **0.103 voxels and 0.12 deg on MPS** (20 s after the section 3 fixes, at 0.107 voxels), matching eddy:
  1. The residual is taken in acquisition space: the smooth prediction is moved into each volume's grid and compared
     with the raw data, so noisy data is never interpolated.
  2. The prediction is sampled with cubic B-splines, so its gradient is continuous.
  3. The prediction is rebuilt before every Gauss-Newton step.

  The second-level model turned out not to be needed here, for eddy or for the prototype. Leave-one-out helps by
  15-20 %.
- **Ensemble.** Each replicate reruns the step, like the field estimate. That costs about 5-10 % per run. The bigger
  change is how replicates are made: they should be generated by a forward model in acquisition space, so that they
  carry the scan's motion and the estimator has to find it again (section 5).
- **Effort.** Phase 1 (rigid plus quadratic eddy with the joint second-level model, b-vector rotation, the fused
  sample-and-moments kernel, cubic resample, the forward-model bootstrap, the eddy import path, opt-in flag, synthetic
  tests): about 4-5 weeks. Phase 2 (slice outlier replacement, GP predictor): about 2 weeks. Phase 3 (slice-to-volume): about 4-8 weeks, research-grade. Nothing here can be
  validated before the 28 Oct abstract deadline, and the GPU queue is busy. The abstract should rely on the CPU
  correlation check (`bench/calibration_motion_check.py`) and the planned DIPY affine test, and list this step as
  future work.
- **What is already known.** The CPU check (NOTES 2026-10-20, "motion and eddy-current sanity check") measured
  motion and eddy proxies on 83 pairs.
  - No evidence that eddy currents drive the calibration gap.
  - A weak motion-mismatch signal, plausibly under about 5 % of the gap, well under one coverage point.
  - It was correlation only: no data was corrected and rerun, and it says nothing about accuracy. ON-Harmony's eddy
    signature reaches 0.7-2 mm at the brain's edge, and its b0 motion up to 6.6 mm.
  - A same-site, other-unit rescan (same scanner model) costs about 3 extra points at 0.9, the same as ON-Harmony's
    other site. This is preliminary (`calibration_crossscanner.json`) and points at the scanner unit, not motion, as
    the larger term.
- **Phase 0, before any implementation: a direct test with eddy.** FSL eddy is run as a reference, as topup has
  been throughout, and fed tractline's own field so that only motion and eddy correction change. It runs on about 8
  pairs chosen for the most motion and eddy. It asks two separate questions:
  - does correction move the gap there by more than a point?
  - does it improve accuracy (tensor-fit residuals, alignment to the T1, rescan agreement)?

  The CPU check predicts no to the first. The second decides whether phases 1-2 go ahead, with accuracy and parity
  with eddy as the motivation, not the calibration gap. It takes about 3-4 days of work plus 16 eddy runs and about
  150 tracking runs.

## Decisions taken

The owner asked the design to settle these by judgment. Each entry gives the choice, the reason, and the measurement
that would reverse it. None of them changes the default pipeline: everything here sits behind the opt-in flag.

1. **Predictor: leave-one-out regularized SH per shell in phase 1; the GP only if it earns its place.**
   - *Reason*: it is one matrix per shell and the model the wild bootstrap already fits, so corrected data and
     replicates share one model. tractline tracks one shell at a time (section 1.4).
   - *Reversed if*: in phase 2, the GP's held-out prediction error on joined multi-shell series (B2000J, ON-Harmony
     b1000 + b2000) is at least 10 % lower. Then the GP becomes the predictor for multi-shell input only.
2. **Sequential: `h` is held at the existing estimate; motion and eddy are fitted against it.**
   - *Reason*: the field estimate is validated against topup and on 12 patients. Refitting it jointly would move every
     committed result and tie two convergence problems together. Susceptibility-by-movement is phase 3.
   - *Reversed if*: with each b0's fitted motion applied, `residual_left` stays above its warning level on scans where
     the step finds large rotations. That would mean `h` is being bent to absorb motion.
3. **Final interpolation: 3D cubic B-spline, one resample per volume; trilinear only inside the estimate.**
   - *Reason*: today's `apply` is cubic along the phase-encoding axis. Trilinear smooths each volume's noise by an
     amount that depends on its sub-voxel shift, which would put motion-dependent texture into the residuals the
     bootstrap flips.
   - *How*: a prefilter (as `prefilter`, on three axes) and a chunked 64-tap gather on the device in float32, float64 on
     the CPU. A custom kernel only if it measures slower than 2 s on the M2.
   - *Reversed if*: the cubic resample's cost exceeds the estimate's.
4. **Replicates: R1 (forward-model) whenever the flag is on; R0 unchanged when off; R2 conditional.**
   - *Reason*: R0 replicates contain no motion for a rerun motion step to find, so rerunning the step per replicate
     would measure nothing. R1 is the bootstrap that matches a pipeline which estimates motion. R2 changes what the
     envelope means (motion as a source of variation), so it must earn its place by closing the gap without
     over-covering.
   - *Status of R2*: dropped unless phase 0's eddy arm moves the gap by more than a point on the extreme pairs. The CPU
     check finds motion under about 5 % of the gap. The preliminary cross-scanner result puts a term of about 3
     points on the scanner unit, which motion trajectories cannot represent.
5. **Gauge: each shell's summed rigid parameters held to a one-time registration of the shell's mean to the mean b0.**
   - *How*: normalized gradient fields as the cost, since the mean DW image's CSF is dark where the b0's is bright and
     NGF ignores contrast sign. Mutual information is the fallback if NGF fails the synthetic test.
   - *Applies to*: every protocol, including one-b0 ones (MASiVar cohort II).
   - *Status (prototype, sections 2.2-2.3)*: needed, and working.
     - The cost is mutual information, as eddy's PEAS, not NGF. NGF on the shell means was off by up to 0.5 mm
       across the contrast.
     - The anchor fits 16 parameters per shell: the rigid 6 plus the shell's common eddy polynomial along PE. With the
       rigid 6 alone, the between-shell error was 0.105 voxels; with all 16, 0.013 (eddy: 0.015).
     - It runs once, after the fit (eddy's design). Running it every few steps did no better.
   - *Also considered*: a predictor that couples the shells (eddy's multi-shell GP, or a joint model across b), which
     makes the shells' relative position observable inside the fit itself. That would strengthen decision 1's case
     for the GP on multi-shell scans.
6. **Recalibration maps: relearned with the flag on, and the protocol descriptor gains a `motion_eddy` item.** A map
   learned without the flag refuses runs made with it, and the reverse.
7. **Eddy order: quadratic, per volume; the second-level model is an option, fitted jointly when used.**
   - *Reason*: eddy's finding favors quadratic. On the synthetic scan (50 directions, whole sphere), eddy gives 0.105
     voxels without a second-level model and 0.098 with a linear one. The prototype reaches 0.103 without one once its
     residual is in acquisition space. The rotation-shear tie that first looked like a reason for it (0.39 deg) came
     from the noise-interpolation bias, and the acquisition-space residual removes it (0.12 deg).
   - *When to use it*: few directions (B32, A32) or half-sphere sets. Then it is fitted jointly, `A [1, g_v]` shared
     by all volumes, by a Schur complement (section 2). A projection after each pass diverged (section 2.1).
     Phase 2 measures it on those protocols.
8. **FSL is the oracle, run wherever it is fastest, and phase 0 is built on it.** As for topup throughout this
   project (`bench/topup_ref.py`, `bench/cohort_topup.py`), running FSL to evaluate tractline is part of the plan.
   Where it runs:
   - eddy (`fsl-eddy`) installed from FSL's conda channel into `DATA/fsl-env`, beside `fsl-topup`;
   - `eddy_cpu` on Modal CPU containers;
   - `eddy_cuda` on Modal's NVIDIA GPUs;
   - fsl-metal's `eddy_metal` on the M2 once the queue is idle.

   A loader takes eddy's corrected series and `eddy_rotated_bvecs` in place of the step and records the source in the
   provenance.
   - *Reason*: the loader is about a day's work. It gives phase 0 (section 7), the comparison of section 6.1, and an
     eddy arm in the synthetic tests of section 6.2.
   - *eddy is fed tractline's field* (`--field`, a field image in Hz, and `--field_mat`, the rigid matrix between field and
     data; both in the eddy user guide, checked 2026-10-20), not topup's. Otherwise an eddy arm would change the field method and the motion and eddy
     correction at once.
9. **Our own code, honestly attributed.** This is not a commercial clean-room situation. FSL's source (eddy,
   topup), TORTOISEV4's and NiFreeze's may be read, studied and used as inspiration, as topup's was for
   `susceptibility.py`. The rules:
   - No code is copied from them, and none is mechanically translated or rewritten so as to hide where it came from.
   - Methods are free to use. If eddy uses Gauss-Newton, a Gaussian-process predictor or a quadratic eddy field, so
     may tractline.
   - Where a choice follows FSL's (or another tool's) implementation and not just its paper, the code's docstring
     says so, as `susceptibility.py` does for topup's schedule and conventions.
   - tractline distributes only code its authors wrote, claims no more than that, and complies with each license as
     written.
10. **Timing.**
    - **Before the 28 Oct abstract:** no implementation. Phase 0's eddy runs can start now, because they are
      CPU-bound or run on Modal and so do not compete for the M2's GPU. Their tracking runs join the queue after the
      current work. If they land in time, the abstract can cite the eddy arm; the plan does not depend on it.
    - **After the deadline:** phase 1 starts CPU-first (the float64 synthetic tests need no GPU) and moves to MPS
      when the queue is idle.

### What still needs the owner

These are not design choices.

- **GPU time**:
  - phase 0: about 150 tracking runs (about 8 pairs x (17 + rescans)), roughly 5-8 GPU hours on the M2;
  - phase 1's arms: the same per pair over all pairs;
  - all after the current queue.
- **Modal or CPU time** for phase 0's 16 eddy runs (A and rescan of about 8 pairs). Eddy per replicate (E-rerun) is
  conditional.

## 1. The model

### 1.1 Spaces and the composed transform

Write `x` for a point in the **reference space**. This is the undistorted frame of the b0 that the field estimate
holds fixed (volume 0 of `s.b0s`, the DWI's first b0). The static field `h(x)` (Hz) lives there: it moves with the
head. Write `y` for a point in the **scanner frame** of volume `v`, its acquired voxel grid in mm. The eddy-current
field `e_v(y)` (Hz) lives there, because it is made by the gradient coils, which do not move. Volume `v`'s observed
image `f_v` is then, to first order:

    reference x  --rigid M_v-->  y = R_v x + t_v  --distortion along PE-->  y + k_v [h(x) + e_v(y)] e_pe

so the corrected volume at `x` is

    u_v(x) = f_v( R_v x + t_v + k_v [h(x) + e_v(R_v x + t_v)] e_pe ) * (1 + k_v d/dpe [h + e_v])

Here `k_v = pe_sign * readout_s` (voxels per Hz: the existing `pe_scale`) and `e_pe` is the image's
phase-encoding axis. The order is: **rigid motion first, then both off-resonance displacements along the scanner's
phase-encoding axis, in one sample.** This is the composition `susceptibility.unwarp` already uses (`M_v(x) + d_v(x)
e_pe`). The only change is that the eddy term is added to the field and evaluated at the moved point, not at `x`.
Evaluating `h` at `x` and not at the moved point is the existing first-order approximation. It holds as long as
rotations are small compared with `h`'s smoothness (`h` is a B-spline with 4 mm knots at its finest), and eddy makes
the same assumption unless its susceptibility-by-movement option is used (Andersson et al. 2018).

Keeping the eddy field in Hz has three uses. It adds directly to `h`. Readout time and polarity enter once, through
`k_v`. Its Jacobian joins the existing one. For a linear field the Jacobian factor is constant over the volume.

### 1.2 Per-volume parameters

| block | parameters | phase |
|---|---|---|
| rigid | 3 translations (mm), 3 rotations (rad) about the grid center (`susceptibility.rigid`) | 1 |
| eddy, linear | `e_v(y) = c0 + c . y`: a constant (Hz) and 3 gradients (Hz/mm), i.e. translation, scale along PE and two shears (Haselgrove & Moore 1996; Jezzard et al. 1998) | 1 |
| eddy, quadratic | + 6 second-order terms (Hz/mm^2) | 1 (same monomial code) |
| eddy, cubic | + 10 third-order terms; eddy offers it, but its paper reports marginal gains at about twice the time | not planned |
| gain | optional: one intensity scale per volume, solved in closed form | 1 |

That gives P = 10 parameters per volume for the linear model (11 with the gain) and 16 for the quadratic one. The
task framing put quadratic in phase 2. Andersson & Sotiropoulos (2016) find linear insufficient, and the cost here is
only the moment reduction's 55 monomial products instead of 10, already counted in section 3. So phase 1 implements
the polynomial for any order and defaults to quadratic, with the second-level model (decision 7). Each volume's
parameterization is that of the CPU proxy fit in `bench/calibration_motion_check.py` (6 rigid + PE affine + gain).
Phase 1 differs from that fit in its prediction (leave-one-out, not a fit that includes the volume), in fitting all
volumes in one batch on the GPU, and in resampling only once.

**Confounds and their fixes** (each also handled by eddy):

- The eddy constant `c0` and the translation along PE make the same displacement. As topup's convention does for
  the field offset, the constant is modeled as a linear function of the volume's gradient vector across volumes,
  and the rest is attributed to motion. This mirrors eddy's default separation of field offset from movement.
- Per-volume eddy parameters are noisy with few directions, and they trade off against rotation (section 2.1).
  Phase 1 fits eddy's "second-level model" jointly. It constrains each eddy parameter to be a linear function of
  the diffusion gradient across volumes (quadratic as an option).
  That is the physics: eddy currents scale with the gradient (Zhuang et al. 2006).
- The DW volumes as a group, against the b0s. A common shift of all DW volumes leaves the leave-one-out
  predictions unchanged, so it is unobservable. Each shell's summed rigid parameters are held to an anchor. The anchor
  is a single rigid registration of the shell's mean (corrected) image to the mean b0, with a cross-contrast cost
  (normalized gradient fields, decision 5). This mirrors eddy's post-eddy alignment of shells.

eddy's `--dont_sep_offs_move` ("do not attempt to separate subject movement from field DC component") and
`--dont_peas` ("do not end with an alignment of shells to each other") turn off its versions of the first and third
fixes (eddy user guide, checked 2026-10-20).

### 1.3 b-vector rotation

Only the rigid rotation rotates the encoding direction. The eddy fields are distortions, not rotations of the
anatomy. The motion is estimated in the grid's mm frame (`gmm = grid * vox`), the frame of FSL's b-vectors except for
FSL's x flip when `det(affine) > 0`. So, with `F = diag(-1, 1, 1)` when the affine's determinant is positive and the
identity otherwise:

    bvec_v' = F R_v^T F bvec_v

This is computed before `prep.prepare` converts the b-vectors to RAS (`prep.py:37-40`). The transpose is there
because `R_v` maps reference to acquisition. A synthetic rotation test (section 6.2) pins the sign, since getting it
backward doubles the error instead of removing it. Leemans & Jones (2009) show what skipping the rotation costs.
`ukf.from_arrays` reads one gradient per volume, so per-volume rotated gradients need no tracker change.

### 1.4 The reference problem

DW volumes differ in contrast, so each needs a target with its own contrast. Three options:

| option | how | pros | cons |
|---|---|---|---|
| GP (eddy) | per-shell (and cross-shell) GP over gradient directions; hyperparameters by marginal likelihood (Andersson & Sotiropoulos 2015) | multi-shell sharing; no model order to choose; eddy's behavior | hyperparameter fit; more to validate |
| SH or tensor, leave-one-out (model-based registration: Andersson & Skare 2002; Bai & Alexander 2008) | per shell, `pred = W y`, `W` from the hat matrix with its diagonal removed (LOO) | one matrix per shell; the bootstrap's own model (`_bootstrap.sh_order_for`); closed-form LOO; QSIPrep's SHORELine does this with 3dSHORE (Cieslak et al. 2021) | per-shell only (phase 1); order and regularization to set |
| iterate fit and register | fit to the resampled data, register to the fit, repeat | simple | biased toward the volume's current position unless LOO; at high b, the noise drives the fit (Ben-Amitay et al. 2012) |

The table hides one simplification. For one gradient table, **every one of these predictors is linear in the data with
weights that depend only on the directions**. So each prediction is a G x G matrix applied to every masked voxel's
signal vector. For the GP, `W = K_* (K + s^2 I)^-1` and the LOO version comes in closed form from `(K + s^2 I)^-1`. For
SH, `W = B (B^T B + l L)^-1 B^T`, and `y_i - r_i / (1 - H_ii)` gives LOO. Either way, prediction on the GPU is one
`(N_mask x G) @ (G x G)` matmul per outer iteration: 6.6 GFLOP for 300 k voxels and 105 volumes, a few milliseconds.

**Leave-one-out matters, but less than first argued.** Without regularization, `sh_order_for` targets coefficients
at 0.6 of the directions, so the hat matrix's diagonal would average about 0.6. With the Laplace-Beltrami
regularization the prototype uses (order 6, l = 0.006, 50 directions), it averages 0.30. Measured on the prototype
(section 2.1), predicting each volume from all volumes, itself included, leaves an edge error 18 % larger than
leave-one-out (0.334 against 0.284 voxels RMS). That is worse, but it does not stall. Leave-one-out costs nothing, so
it stays.

**Decision 1: phase 1 uses leave-one-out SH per shell with Laplace-Beltrami regularization; phase 2 adds the
GP.** The reasons:

- It is the model the wild bootstrap already fits, so the corrected data and the replicates share one model (section
  5).
- `l` comes from generalized cross-validation in closed form.
- tractline tracks one shell at a time (`prep.prepare` keeps the b0s and one shell), so per-shell prediction is
  enough for every protocol in the queue.

The GP becomes worth its cost for joined multi-shell series (B2000J, the ON-Harmony b1000 + b2000 scan) and for few
directions. Phase 2 measures both predictors' held-out error on the data before picking. b0 volumes are predicted by
the mean of the other b0s. With one b0 (cohort II), that b0 is the anchor and is not registered.

The outer loop is eddy's. Resample all volumes with the current parameters, predict each one from the others, then
update all parameters by Gauss-Newton against the predictions, and repeat. The predictor matrices are computed once.
The predictions are recomputed after every resampling.

## 2. The optimizer

- **Unknowns**: `theta_v` (P per volume) for all G volumes. Given the predictions, the volumes do not interact, so the
  Gauss-Newton matrix is **block-diagonal**: G independent P x P systems. This is simpler than the field estimate's
  matrix-free PCG, which is needed there only because `h` has thousands of coupled coefficients. Here
  `torch.linalg.solve` on a (G, P, P) batch (on the CPU: 105 x 16 x 16) costs nothing.
- **Residual, in acquisition space**: `r_v(y) = a_v p_v(phi_v^-1(y)) J_v(y) - f_v(y)` over the acquired grid's mask.
  - `f_v` is the raw volume, never interpolated.
  - `p_v` is its LOO prediction (built in reference space from the corrected volumes), moved into volume v's grid.
  - `J_v` is the Jacobian factor; `a_v` is the optional gain.
  - `phi_v^-1` is found by a few fixed-point steps (`x <- x - (phi_v(x) - y)`, 4 in the prototype).

  The cost is SSD, as eddy's. Measured on the prototype (section 2.1): the reference-space residual `u_v(x) - p_v(x)`
  interpolates the noisy `f_v`, and with no motion it drifts 0.18 voxels and skews the rotation (0.39 deg). In
  acquisition space the drift is 0.08 and the rotation error 0.03 deg. This is also R1's forward model (section 5):
  the same operation, so the step and its bootstrap share it. NCC buys nothing, because the prediction has the
  volume's contrast. Phase 2 adds outlier weights (section 4).
- **Jacobian**: `d r / d theta ~ -grad p_v(phi_v^-1(y)) . d phi_v / d theta`, with `D phi ~ I` (small motion).
  - The gradient is the prediction's, which is smooth.
  - It is sampled with cubic B-splines (the 8-fetch form, autograd through it), so it is continuous; trilinear's
    piecewise-constant gradient cost about 10 % in the prototype.
  - `d phi / d theta` is analytic: the rotation's derivative times the moved coordinates, plus
    `k_v * monomial(y) * e_pe` for the eddy terms. `J^T J`
  comes from moments. For each volume, sum over voxels `g_a g_b m_i m_j` (g the 3-vector gradient, m the 4 or 10
  monomials of the moved point) and contract with the small parameter map on the CPU. The (G, N, P) Jacobian is
  never formed. The rotation's chain-rule factor varies only slowly with the parameters, so it is evaluated at the
  current iterate.
- **Levenberg-Marquardt per volume**: each volume has its own `mu`, starting at 1e-3, divided by 3 on success and
  multiplied by 8 on failure (the `gauss_newton()` schedule). Trial costs are evaluated for all volumes in one batched
  pass, and steps are accepted volume by volume.
- **Parameter scaling**: as `estimate`'s `MS` (1 mm, 0.01 rad), plus eddy terms scaled so that one unit is about one
  voxel at the brain's edge.
- **Mask**: `mask.median_otsu` (the pipeline's) on the field-corrected mean b0, dilated 2 voxels, in reference space.
  Computation is restricted to the mask's voxels, gathered once (`grid_sample` on a (G, 1, 1, N_mask, 3) grid), about
  25-35 % of the box at these fields of view (an assumption, not measured here).
- **Schedule: one Gauss-Newton step per prediction.** Rebuild the predictions (one resample of every volume and one
  matmul) before every step. Measured, at the same 15 steps: 15 x 1 reaches 0.103 voxels, and 5 x 3 reaches 0.200;
  even 10 x 3 only reaches 0.123. The prediction lags the parameters, and refreshing it is cheap. Smoothing, FWHM in
  voxels per step: 2, 2, 1, 1, then 0. The first steps can run 2x subsampled, on the CPU under MPS as the field
  estimate does (`coarse_device="auto"`); that is not yet tried in the prototype.
- **Second-level model (optional; decision 7).** To first order, a rotation about a non-PE axis moves voxels along
  PE linearly in the other coordinate, which is an eddy shear term. With few directions the data may not separate
  them.
  - Eddy's physics then breaks the tie: eddy coefficients follow the gradient direction, and motion does not. Each
    eddy coefficient is parameterized as `A [1, g_v]`, with A shared by all volumes (eddy's `--slm=linear`).
  - The volumes then couple only through A (10 x 4 = 40 shared unknowns). One Schur-complement solve per step keeps
    it cheap: eliminate the per-volume 6 x 6 blocks, solve the 40 x 40 system, substitute back.
  - A projection onto `[1, g]` after each pass diverged in the prototype (7-11 voxels). On 50 whole-sphere
    directions it is not needed (section 2.1).
- **Convergence**: a volume is frozen when its step moves no mask-edge voxel by more than 0.01 voxel, or its cost
  falls by less than 1e-4 relative, or no damping gives descent. An outer iteration ends when every volume is frozen
  or the cap is reached. The step ends when the parameters' largest change over an outer iteration is below 0.05 mm
  at the edge. Each volume's iterations, final cost and convergence flag are logged, as `gauss_newton()`'s `info`.

### 2.1 Prototype results (2026-10-20)

**Script**: `bench/motion_eddy/proto.py` (results in `bench/results/motion_eddy/proto_*.json`), on MPS.

**Setup**:
- The truth is the b = 2000 shell of ON-Harmony sub-16974's OXF1PRI001 scan (104 x 104 x 72, 2 mm, 50 directions over
  the whole sphere), replaced by its own SH fit (order 6, Laplace-Beltrami l = 0.006). The truth therefore has the
  scan's anatomy and diffusion contrast and no motion.
- Each volume is moved by a known rigid transform (1 mm, 1 deg SD per axis) and distorted along PE by a quadratic eddy
  field linear in the gradient. It is resampled with cubic B-splines and given Gaussian noise at the scan's own
  residual level (sigma 77). There is no static field.
- The estimator: per-volume rigid + quadratic eddy (P = 16), leave-one-out SH prediction, batched per-volume
  Levenberg-Marquardt, explicit Jacobian, the mask dilated 2 voxels (210 k voxels, 27 % of the box), 15 Gauss-Newton
  steps at full resolution. Its first form had the residual in reference space, trilinear sampling and 5 outer x 3
  steps. Its final form is section 2's: acquisition-space residual, cubic prediction, 15 x 1.
- **eddy on the same scan**: the same 50 DW volumes plus the scan's 5 b0s (rigid motion, b0 0 fixed), as acquired
  (`proto.py --save-synth`). `eddy_cpu` 2602.0 runs with `--flm=quadratic --nthr=6 --data_is_shelled --dfields`, and
  its displacement fields are scored the same way (`compare_eddy.py`). The fields use our sign convention directly.
- The metric is the displacement error at the mask's edge, after removing the transform common to all volumes (which
  the leave-one-out prediction cannot see).

| variant | edge error, voxels RMS / p95 | note |
|---|---|---|
| no correction | 1.52 / 2.57 | the simulated displacement |
| baseline (final FWHM 0) | 0.284 / 0.485 | 15 iterations, 14 s |
| baseline, 30 iterations | 0.251 / 0.428 | 27 s; still creeping down |
| no leave-one-out | 0.334 / 0.561 | 18 % worse |
| final FWHM 1.5 voxels | 0.243 / 0.406 | rotation error 0.39 deg RMS |
| no motion, final FWHM 0 / 1 / 1.5 | 0.182 / 0.150 / 0.101 | the estimator's own drift: a noise-interpolation bias |
| rigid only (no eddy simulated) | 0.290 / 0.508 | the eddy terms still free |
| eddy projected onto `[1, g]` after each pass | 7.3 / 8.9 | diverges: a second-level model must be joint |
| acquisition-space residual (trilinear, 5 x 3) | 0.222 / 0.356 | 91 s; no motion: 0.081, rotation 0.03 deg |
| + cubic prediction (5 x 3) | 0.200 / 0.333 | 98 s; no motion: 0.075 |
| + cubic, 10 x 3 | 0.123 / 0.210 | 219 s |
| **+ cubic, 15 x 1 (prediction rebuilt every step)** | **0.103 / 0.181** | **103 s; rotation error 0.117 deg RMS** |
| **eddy, `--slm=none`** | **0.105 / 0.186** | **36 min on 6 CPU threads; rotation error 0.129 deg** |
| eddy, `--slm=linear` | 0.098 / 0.168 | 35 min; rotation error 0.104 deg |
| final form with the section 3 fixes, 8 of 15 steps coarse (`--fast --coarse 8`) | 0.104 / 0.185 | 31 s; rotation error 0.118 deg |
| **the same, 12 of 15 steps coarse (`--fast`, the default)** | **0.107 / 0.191** | **20 s; rotation error 0.123 deg** |
| the same, all 15 steps coarse | 0.115 / 0.207 | 13 s; rotation error 0.134 deg |
| **fused Metal kernels (`--fast --metal --coarse 0`), 15 steps** | **0.103 / 0.181** | **2.4 s; rotation error 0.116 deg** |
| the same, to convergence (`--outer 40 --tol 0.05`: 25 steps) | 0.099 / 0.178 | 4.0 s; rotation error 0.099 deg |
| the same, no motion | 0.083 / 0.158 | 3.2 s (20 steps); rotation error 0.041 deg |
| the same, 2x the motion and eddy (3.0 voxels RMS true) | 0.111 / 0.201 | 30 steps, 4.6 s; rotation error 0.10 deg |
| the same, 3x (4.6 voxels RMS, 7.7 p95: about 9 mm and 3 deg) | 0.160 / 0.285 | 29 steps, 4.6 s; a floor (45 steps: 0.163); eddy not run at 3x |

The rotation errors are those of the rotated b-vectors against the truth's, as `F R^T F` (section 1.3). eddy's
`eddy_rotated_bvecs` match that convention.

What it shows:
- The design reaches eddy's accuracy on this scan, at phase 1's bounds: 0.10 voxel, and 0.12 deg against the 0.2
  deg target.
- The prediction's freshness, the residual's space and the interpolation each mattered more than the predictor (SH
  here, a GP in eddy) or the second-level model.
- In the final form the rigid and eddy parts still trade off (each 0.48 voxels off). That is the c0-against-PE-
  translation confound of section 1.2: harmless to the images and to the b-vectors, but the per-part numbers should
  not be read on their own.
- Not yet tested: the static field (`h = 0` here), real motion statistics, a half-sphere gradient set, cohort II's
  2.5 mm and 96 directions, and outliers.

### 2.2 The full scan: 105 volumes, three shells (2026-10-20)

**Setup**: `proto.py --all-shells`.
- All 105 volumes of the same ON-Harmony scan: 5 b0s, 50 at b1000, 50 at b2000.
- Each DW shell gets its own LOO-SH predictor (order 6); the b0s are predicted by the mean of the other b0s. The
  predictor matrix is block-diagonal.
- Eddy is simulated in proportion to the gradient amplitude (sqrt(b / 2000)) and is zero for the b0s. In the fit
  the b0s' eddy terms are fixed at 0, because eddy currents are zero at b = 0. Leaving them free cost 0.3 voxels in
  the b0 group.
- The fit is the Metal fast path, all steps at full resolution.
- The images are also checked: the corrected volumes (`apply_volumes`) against the truth inside the brain mask,
  with the truth moved by the estimate's common transform.

| steps | edge error, RMS / p95 | within shell | between shells | rotation | images: RMSE corrected / as acquired / noise | time, estimate + final resample |
|---|---|---|---|---|---|---|
| 20 | 0.139 / 0.252 | 0.093 | 0.202 | 0.095 deg | 111 / 288 / 84 | 6.6 + 0.32 = 6.9 s |
| 40 | 0.190 / 0.271 | 0.095 | 0.340 | 0.116 deg | 137 / 288 / 84 | 12.7 + 0.32 = 13.0 s |

What it shows:
- **Speed**: a full 105-volume, 2 mm scan takes 6.9 s on the M2 for 20 steps, with the final resample included.
- **Within each shell the fit is at eddy's level** (0.093 voxels).
- **Between shells it is not**, and more steps make it worse. Each shell's common transform is invisible to its own
  leave-one-out prediction, so it drifts with the noise. The shell means move apart along PE (b1000 +0.58 mm, b2000
  -0.71 mm after 40 steps); the eddy constants compensate within each shell (the c0-against-PE-translation
  confound), but not across shells.
- The images confirm it: corrected volumes are 111 against 288 as acquired (noise 84) at 20 steps, and worse at 40
  (137).
- **The fix** is decision 5's anchor, now working (section 2.3).
- The 50-volume comparison with eddy (section 2.1) scored the DW shell alone. There the shell's own common transform
  was removed with the gauge, so that comparison stands.

**What upstream does, and how the prototype compares** (2026-10-20):

- **eddy**, from its outputs on our synthetic scan (`*.eddy_post_eddy_shell_alignment_parameters`) and its user guide:
  - "between shell parameters ... calculated using MI between shell means", each shell registered rigidly to the mean
    b0 and applied (PEAS, on by default);
  - the same estimated from "interspersed b0-volumes", reported only;
  - a PE translation by MI, "calculated but not used";
  - "no movement is estimated between the first b=0 and the first DWI volume", which are assumed close in time;
  - the guide puts PEAS at about 0.2-0.3 mm.
- **eddy's result on our synthetic scan** (5 b0s + b2000, scored with its b0s): between groups 0.062 voxels, within
  0.101, overall 0.108.
- **TORTOISE (DIFFPREP)** sidesteps the gauge by registering every volume to the b0 by MI. **QSIPrep's SHORELine**
  fits one 3dSHORE model across all shells, which couples them inside the fit.

| shell anchor (105 volumes, 20 steps unless noted) | between shells | images (RMSE) | note |
|---|---|---|---|
| none | 0.202 | 111 | |
| NGF, Gauss-Newton (GPU), at the end | 0.255 | 110 | self-test: 94 % of a 0.7 mm shift, 84 % of a 0.29 deg rotation |
| NGF, Gauss-Newton, every 5 steps | 0.157 | 97 | the best images; 30 steps: 0.200, 104 |
| NGF on 5 b0s + b2000 (eddy's volumes), every 5 steps | 0.254 (none: 0.208) | | eddy: 0.062 |
| dipy rigid MI (CPU), at the end | 0.217 | 114 | self-test failed (no translation recovered); torch and dipy's OpenMP clash in one process |

The registrations' own errors, against what the truth needs: NGF on the shell means is off by up to 0.5 mm and 0.4
deg.

**A mechanical check** (`--anchor-truthtest`) rules out a convention or composition error in the anchor path:
- Each group's corrected mean is registered by the same NGF code to the noise-free truth's mean of that group,
  which has the same contrast.
- For the b0s it finds (-0.29, -0.67, 0.00) mm and 0.24 deg, against (-0.27, -0.64, 0.02) mm and 0.19 deg needed.
- For the DW shells, x, z and the rotations agree to about 0.02 mm and 0.02 deg. Along PE (y) the image shift agrees
  once the eddy constant is counted with the PE translation, because the two are confounded. The earlier "needed"
  printout counted the translation alone.

So the remaining error is the cross-contrast registration itself, in x and z (0.5-0.65 mm), not voxel conventions. Cross-contrast registration of a b2000 mean to a b0 mean is the weak link, and the prototype does not yet match
eddy's MI.

**Next, in order of expected payoff:**
1. **A same-contrast reference** (Nilsson et al. 2015). Extrapolate a b0-like image from the DW shells per voxel
   (mono-exponential: `S0 ~ S1000^2 / S2000`, from directional means). Register the DW group to the b0 mean by NGF or
   SSD, same contrast, where the self-test reaches 94 %. Register b1000 and b2000 to each other directly; their
   contrasts are close.
2. **A GPU MI registration** (Parzen-window histogram, as eddy's PEAS), if the extrapolation is not accurate enough.
3. **A predictor coupling the DW shells** (eddy's multi-shell GP; SHORELine's joint model). It fixes the
   b1000-b2000 offset inside the fit, but not the b0s', whose voxel-wise relation to the DW signal (the ADC) absorbs
   any shift under a predictor with weights shared across voxels.
4. **eddy's time assumption** as a fallback anchor: the first DW volume shares the adjacent b0's motion. Real scans
   have temporally smooth motion; the synthetic one does not.

**What carries over from `susceptibility.py`**: `rigid`, `smooth`, `subsample`, the `MS`-style scaling, the
Marquardt loop, the stopping rules, the `coarse_device` split, the autograd-through-`grid_sample` gradient,
`prefilter`/`sample_pe` (the cubic final resample, generalized to 3D), the float32-on-MPS / float64-on-CPU dtype
rule and the `exact_float32` context on CUDA. **What does not**: the per-volume Python loops (`for v in range(V -
1)`, `autograd.functional.jacobian` per volume). They are fine for 8 b0s and not for 105 volumes. Phase 1 replaces
them with batched (G, 3, 3) rotations and an analytic derivative.

### 2.3 The shell anchor: MI with the shell's eddy field, on a truth aligned across shells (2026-10-21)

**The truth had to change first.** The SH-fit truth of section 2.2 inherits the real scan's own offsets between its
groups. Registered by dipy's MI, the real b1000 and b2000 means sit (0.04, -0.22, -0.32) and (0.05, -0.27, -0.18) mm
from the real b0 mean, and single real b0s move by up to 0.2 mm. That is a floor of about 0.15 voxel under any
between-shell score, the size of the errors being chased.

`proto.py --tensor-truth` instead builds every volume from one tensor model on the b0 mean, `S0 exp(-b g'Dg)`, with D a
log-linear fit to the real data. Every shell then has the b0's edges exactly.

**The registration.** `mi_torch.py`, a Mattes-style MI in torch:
- a cubic B-spline Parzen joint histogram, built as a matmul, so it is differentiable;
- three levels; BFGS with backtracking;
- optionally the eddy polynomial along PE as 10 more parameters.

It was checked against dipy's MI (`mi_dipy.py`, in its own process, since dipy and torch clash over OpenMP):
- **known same-contrast shifts**: recovered to 0.01 mm and 0.007 deg (`mi_torch_check.py`);
- **the full-resolution level keeps sigma 1.5 voxels**: unsmoothed, trilinear sampling's shift-dependent blur moved
  MI's optimum by up to 0.15 mm (the known artifact of interpolation in MI).

**Results** (`--all-shells --tensor-truth --fast --metal --coarse 0 --outer 20`, 105 volumes unless noted):

| anchor | edge error, RMS | within shells | between shells | fit + anchor |
|---|---|---|---|---|
| none | 0.138 | 0.086 | 0.204 | 6.7 s |
| NGF (Gauss-Newton), every 5 steps | 0.119 | 0.103 | 0.111 | 12.9 s |
| dipy MI, rigid, at the end | 0.106 | 0.086 | 0.101 | 18.6 s |
| GPU MI, rigid, at the end | 0.107 | 0.086 | 0.105 | 10.6 s |
| **GPU MI, rigid + the shell's eddy field (16), at the end** | **0.086** | **0.086** | **0.013** | **11.7 s** |
| the same, 40 steps | 0.088 | 0.086 | 0.032 | 17.6 s |
| 55 volumes (b0s + b2000), GPU MI with eddy | 0.094 | 0.094 | 0.012 | 5.6 s |
| **eddy** (`eddy_cpu`, `--flm=quadratic --slm=none`, PEAS on) | **0.053** | **0.051** | **0.015** | **78 min, 6 threads** |

What it shows:
- **Between shells, the design now matches eddy** (0.013 against 0.015), in seconds.
- **The rigid anchor alone is not enough.** The shells' remaining error was in their common eddy terms (xy and z^2
  about +/-0.1-0.2 voxel, and a shared PE scale and curvature against the b0s). Those terms are as unobservable
  within a shell as its rigid position. Only the b0s, which have no eddy currents, can pin them.
- **Within shells eddy is better**: 0.051 against 0.086 voxels. More steps do not close it, since the within-shell
  error stays at 0.086.
- eddy's predictor is a GP across all shells, so each prediction draws on about twice the data ours does. The next
  test is decision 1's multi-shell predictor: an SH model per shell with the shells coupled, or eddy's GP.
- Its PEAS output on this scan applied (0.05, 1.41, 0.29) mm and (-1.09, 1.47, 0.59) deg to the b1000 shell. Its
  shells too drift far from the b0s during the fit; the alignment at the end is what fixes them.

### 2.4 A real scan, against eddy (2026-10-21)

**Setup**:
- The scan is ON-Harmony sub-16974 OXF1PRI001, all 105 volumes (5 b0s, 50 at b1000, 50 at b2000), 104 x 104 x 72
  at 2 mm, multiband 3.
- Neither method gets the static field (no `--topup` or `--field`), so they are compared on motion and eddy alone.
- eddy: `eddy_cpu --flm=quadratic --slm=none --dfields --nthr=6 --data_is_shelled`, 79 min.
- Ours: `proto.py --all-shells --fast --metal --coarse 0 --outer 20 --real <dir>`, 18 s with eddy sharing the CPU.

There is no truth, so `compare_real.py` measures:
- agreement with eddy: each volume's displacement at the brain edge, the common transform removed, also by group;
- each corrected set's leave-one-out residual per shell and single-tensor residual, RMS over the brain;
- mean FA where FA > 0.25 as acquired.

`between_shells.py` adds two truth-free checks: the b2000 mean registered to the b1000 mean, and an extrapolated b0
(S1000^2 / S2000) registered to the b0 mean.

| | agreement with eddy, RMS (b0s / b1000 / b2000 group offsets) | LOO residual b0 / b1000 / b2000 | tensor residual | mean FA |
|---|---|---|---|---|
| as acquired | | 181.0 / 108.6 / 92.1 | 91.5 | 0.475 |
| **eddy** | | **139.9 / 79.1 / 72.4** | **75.7** | **0.471** |
| ours, no anchor | **0.075** (0.138 / 0.035 / 0.048) | 146.4 / 80.5 / 74.4 | 77.2 | 0.477 |
| **ours, rigid anchor by dipy's MI** | 0.101 (0.194 / 0.068 / 0.082) | 146.4 / **79.1 / 72.6** | 76.2 | 0.476 |
| ours, rigid anchor by the GPU MI | 0.317 (1.10 / 0.145 / 0.250) | 146.4 / 79.2 / 68.7 | 80.2 | 0.421 |
| ours, GPU MI with the eddy terms | 0.328 (1.15 / 0.158 / 0.251) | 146.4 / 71.3 / 67.3 | 76.8 | 0.427 |

What it shows:
- **The fit agrees with eddy on real data.** Volume by volume within each shell, it differs from eddy by about 0.04
  voxels. Overall it is 0.075, and the rest is mostly the b0s' group offset, about the size of eddy's own shell
  alignment on this scan: (-0.01, 0.18, 0.04) mm for b1000 and (-0.02, 0.29, -0.00) mm for b2000, 0.03-0.12 deg.
- **With a rigid dipy-MI anchor, our corrected images fit the models as well as eddy's.** The DW residuals are equal
  to 0.2 %, and the tensor residual and FA are within 1 %. The b0 residual is 4.6 % higher; our b0s are anchored, not
  refitted, so that is expected.
- **The GPU MI is wrong here.** It moves the shells about a voxel against the b0s, and FA drops by 0.05, the
  signature of a b0-to-DW mismatch. Its self-test had warned: on this pair it disagreed with dipy by up to 0.6 mm. It
  was accurate on the synthetic scan, whose shells are the b0 times a smooth attenuation. Real b0 and DW means differ
  in contrast in ways the synthetic scan does not. The likely causes to examine are full-resolution smoothing
  (sigma 1.5), the Parzen intensity ranges, and masking of the moving image. **Until it matches dipy on real data,
  dipy's MI (CPU, its own process, about 10 s) is the anchor.**
- **The eddy-term anchor overfits on real data.** It wins the synthetic scan's between-shell score, but on real data
  it bends the shells to match real contrast differences. On real data the rigid anchor is the one with support,
  as in eddy's PEAS.
- **The between-shell checks** (`between_shells_16974.json`): b2000 against b1000 leaves 0.26 voxels as acquired
  and 0.24 with the eddy-term anchor. The extrapolated b0 against the b0 mean leaves 1.1 and 0.7. That check
  assumes mono-exponential decay, so it cannot separate misalignment from kurtosis. It was not rerun on eddy's or on
  the rigid-anchor sets.
- **b-vectors**: this affine is radiological (det < 0), so there is no x flip, and `R^T` fits eddy's rotated
  b-vectors best of the four conventions. Still, ours and eddy's differ by 0.29 deg RMS, about the size of eddy's
  whole rotation here (0.31 deg). The image displacements agree while the split between rotation and eddy shear
  does not; on a low-motion scan the rotations are weakly determined. It matters little at this size, but a
  higher-motion scan should be checked.

## 3. GPU mapping

**Targets**: MPS on the M2 and CUDA, both through plain PyTorch in phase 1, with one code path and the device as an
argument, as `estimate` already works. Custom kernels come only where measurement shows plain torch falls short. They
would use `torch.mps.compile_shader` on the M2 and Triton on CUDA, the split `ukf_metal.py` / `ukf_triton.py` already
uses.

| piece | plain torch | custom kernel (later, if measured worth it) |
|---|---|---|
| resample + position gradient | `grid_sample` (trilinear) + autograd | **phase 1**: fused sample, gradient and moment accumulation per volume: no (G, N, 3) gradient or (G, 6, N) products in memory. Plain torch spends most of an iteration on those temporaries (measured below) |
| `J^T J`, `J^T r` | einsum / bmm over moments, chunked by volumes | part of the fused kernel; fixed-order reductions (no atomics), so a run stays deterministic on a device, as the UKF kernels are |
| LOO prediction | one matmul per shell | none |
| batched P x P solves, LM bookkeeping | `torch.linalg.solve` on the CPU | none |
| final resample, 3D cubic B-spline | separable prefilter (as `prefilter`) + 8 trilinear `grid_sample` fetches with folded weights (Sigg & Hadwiger 2005; equal to the 64-tap sum to 1.6e-15, measured), batched over volumes | a 3D cubic sampler, if the 8-fetch form's 5 s (2 mm) matters |
| slice outlier statistics | reductions over (G, Z) | none |

**Memory** (float32; "box" means every voxel, "mask" means 30 % of them; ON-Harmony's mask dilated 2 voxels is 27 %,
measured):

| array | 96 x 96 x 50 x 100 (2.5 mm) | 104 x 104 x 72 x 105 (2 mm) |
|---|---|---|
| one 4D copy (box) | 184 MB | 327 MB |
| data + resampled + prediction + residual (box) | 0.74 GB | 1.31 GB |
| position gradient (G, N, 3) (box) | 0.55 GB | 0.98 GB |
| moment chunk (16 volumes x N x (6 + 55)) (box) | 1.8 GB | 3.0 GB |
| **peak, box, chunked** | **about 3 GB** | **about 5 GB** |
| **peak, mask-restricted, chunked** | **about 1 GB** | **about 1.7 GB** |
| final cubic resample (coefficients + output, box) | 0.37 GB | 0.65 GB |

Restricting to the mask is what makes this fit beside other GPU work in the M2's 16 GB of unified memory. The step
calls `release_memory` like the other stages.

**Run time, measured** (2026-10-20, M2, float32, MPS, mask 30 % of the box, P = 16; median of 5).

The GPU was shared with text and image generation, so these are upper bounds. The scripts are
`bench/motion_eddy/bench_pieces.py` and `bench_fast.py`; results are in `bench/results/motion_eddy/`.

| piece | 2.5 mm, 100 vol | 2 mm, 105 vol |
|---|---|---|
| one Gauss-Newton iteration (positions, sample + gradient, residual, moments) | 0.40 s | 0.74 s |
| of which sample + position gradient | 0.17 s | 0.30 s |
| of which forming the g g' products alone | 0.07 s | 0.13 s |
| moments, best plain-torch layout (products + K-chunked matmul) | 0.09 s | 0.16 s |
| a cost evaluation (LM trial) | 0.11 s | 0.19 s |
| LOO prediction (one matmul) | 6 ms | 9 ms |
| final cubic resample, 64-tap gather | 7.1 s | 12.7 s |
| final cubic resample, 8 trilinear fetches | 2.9 s | 5.2 s |
| prototype, first form (reference-space residual, trilinear; 50 volumes, 104 x 104 x 72, 15 steps) | | 14 s |
| prototype, final form (acquisition-space residual, cubic, 15 x 1) | | 103 s |
| the same with the fixes below (`--fast`: 12 coarse steps, then 3 at full resolution) | | 20 s |

**The final form, profiled and fixed** (`proto.py --profile`; 50 volumes, 104 x 104 x 72, 15 steps, MPS;
device-synchronized). Each fix was checked against the synthetic truth, and against eddy's 0.105 voxels:

| part | before | after | the fix |
|---|---|---|---|
| LM trials | 33 s (one per step, all accepted) | 0.9-2.5 s (7 trials) | a trial only in the first two steps and when a step moves an edge point more than 0.3 voxel; otherwise the step is taken |
| the inverse `phi^-1` | 23 s | 0.9-5.6 s | the rigid part in closed form (`x = c + R^T((y - e e_pe - c) vox - t) / vox`), 2 passes for the eddy term, which is evaluated as a written-out polynomial (no (G, N, 10) monomial array) |
| cubic sample + gradient | 32 s | 3.5-23 s | most steps on every other voxel per axis (1/8 of the points); autograd through the 8 fetches kept |
| Jacobian, H, b | 13 s | 1-7 s | the rotation and eddy columns at the acquired points `y` (fixed), not at `phi^-1(y)`; `bmm` |
| predictions, smoothing | 7 s | 6 s | smoothed volumes cached per FWHM |
| **total** | **109 s** | **20 s (12 coarse steps) / 31 s (8)** | accuracy 0.107 / 0.104 voxels, against 0.103 before and eddy's 0.105 |

Two planned fixes did not work as planned:
- **A gradient sampled from a finite-difference gradient volume** of the prediction. Its gradient came out 62 % as
  large as the cubic's true gradient, with a correlation of 0.86. Gauss-Newton converges to where the Jacobian it is
  given says the cost is flat, so that bias moved the answer: 0.70 voxels. The gradient must be the exact one.
  - A one-pass 64-tap form that accumulates the value and the exact gradient from each gathered coefficient
    (`cubic_value_grad`) matches autograd to 1e-4.
  - On MPS it is slower than autograd through the 8 fetches (35 s against 23 s), because MPS gathers are slow.
  - It is the natural shape for the fused kernel.
- **Moments in place of the Jacobian.** With 16 parameters, the per-voxel basis the moments need (g times
  [1, x, y, z], plus `g_pe` times the 6 quadratic monomials) has 18 terms, more than the parameters. So moments save
  nothing here.

What remains, at 20 s:
- The predictions take 4 s. A prediction is rebuilt every step, from all volumes resampled on the mask dilated 4
  voxels.
- Each full-resolution step costs about 2 s. Most of that is the 8-fetch cubic with its backward pass.

Scaled to 105 volumes, plain torch gives about 40 s on the M2.

**The fused Metal kernels** (`bench/motion_eddy/metal_step.py`, `torch.mps.compile_shader` as `tractline.ukf_metal`;
`proto.py --fast --metal`). Each is checked against a float64 reference (`metal_check.py`): H to 1.5e-5 relative, b
to 4e-6, the cost to 8e-8, the prefilter to 2.6e-7.

| kernel | what one thread does | replaces |
|---|---|---|
| `step_moments` | one point of one volume: the inverse (closed-form rigid, 2 eddy passes), the 64-tap cubic value and exact gradient (each coefficient read once), the 16 Jacobian entries, then J'J (upper 136), J'r and r^2 summed over the threadgroup (`simd_sum`, then threadgroup memory) and written per threadgroup; torch sums the partials in a fixed order (deterministic, no atomics) | the inverse, the 8-fetch sample with its backward pass, the (G, N, 16) Jacobian, `bmm` |
| `step_cost` | the same without the gradient, for the listed volumes | the LM trials' cost |
| `resample` | the forward map and a trilinear sample of the acquired volume, for the predictions | `grid_sample` and the torch transform |
| `prefilter_axis` | one line of the recursive cubic B-spline prefilter (pole sqrt(3) - 2, mirrored edges), on a crop to the head | three dense (n x n) matmuls on the full box |
| `gauss_axis` | one voxel of one axis of the separable Gaussian | `conv3d` |

Profile, 50 volumes, 15 steps (device-synchronized):

| part | plain torch, fixed (3 coarse steps) | Metal (no coarse steps) |
|---|---|---|
| inverse, sample + gradient, Jacobian, H, b | 35.6 s | 1.05 s |
| predictions (resample, matmul, prefilter) | 4.3 s | 0.85 s |
| smoothing | 1.9 s | 0.23 s |
| LM trials | 2.5 s | 0.2 s |
| **total** | **44.7 s** (20 s with 12 coarse steps, at 0.107 voxels) | **2.7 s with profiling, 2.4 s without** (0.103 voxels) |

So the fixed 15-step schedule is no longer the constraint, and the step runs to convergence. The stop: no volume's
step moves an edge point more than 0.05 voxel, two steps running, capped at 40. Rebuilding the prediction every step
leaves a jitter of a few hundredths of a voxel, so 0.01-0.03 never stop. The predictions' smoothing (the first two
steps, FWHM 2 and 1 voxels) made no difference on this scan, but stays for capture range on larger real motion.

| | 2.5 mm, 100 vol | 2 mm, 105 vol |
|---|---|---|
| estimate, plain torch, fixed (measured at 50 volumes, scaled by volumes x points) | 25-30 s | 40-45 s |
| **estimate, Metal** (2 mm: measured on the 105-volume scan, 20 steps; 2.5 mm scaled) | **about 3-4 s** | **6.6 s** |
| + final resample (Metal `apply_volumes`, measured at 2 mm: 0.32 s, against 5.2 s in torch) | 0.2 s | 0.32 s |
| **total, M2, Metal** | **about 4 s** | **6.9 s (measured)** |
| CUDA (A10 / L40S): Triton port not written; 3-5x the M2 by the field estimate's ratio | 2-3 s | 3-5 s |

For scale: the field estimate takes 18 s on the M2 and the UKF 27 s. With the Metal kernels the step costs less than
the field estimate. The final resample is the next piece to move into a kernel.

eddy's own CUDA version took 13 min for 128 volumes of 96 x 96 x 64 (quadratic model, a GeForce 750M; Andersson &
Sotiropoulos 2016). Its CPU version is "noticeably slower" (FSL user guide). Most of this design's advantage over eddy
comes from a lighter model: no per-volume GP refit, no susceptibility-by-movement, and Gauss-Newton on a few
parameters per volume. Faster hardware contributes less. The numbers should be re-measured on an idle GPU,
alternating variants as the 2026-10-02 notes do.

## 4. Outliers and within-volume motion

**Phase 2: slice dropout detection and replacement**, the approach of eddy's `--repol` (Andersson et al. 2016). After
each outer iteration:

1. For each volume and slice (each multiband slice group when `SliceTiming` gives one), compute the mean residual
   against the prediction in the mask, standardized across that shell's volumes.
2. Flag a slice when its z falls below -4 (signal loss: only the negative tail), as eddy's default.
3. Give flagged slices zero weight in the cost and in the predictor's fit. Missing data make the predictor depend
   on the voxel, but every voxel of a slice shares one missing pattern, so this costs one G x G solve per affected
   slice, not per voxel.
4. In the output, replace a flagged slice by its prediction, and record it in an outlier map.

SOLID (Sairanen et al. 2018) is a cheaper detector to compare against.

The bootstrap must treat replaced slices separately (section 5). Their residual is not noise.

**Phase 3, out of scope until phases 1-2 are validated: slice-to-volume motion** (Andersson et al. 2017; SHARD,
Christiaens et al. 2021). It needs the slice timing and multiband grouping. ON-Harmony's sidecars give `SliceTiming`
(72 entries, multiband 3). MASiVar cohort II ships no sidecars at all, so phase 3 could be validated on ON-Harmony
only, or on cohort II under an assumed acquisition order. Susceptibility-by-movement (Andersson et al. 2018) is phase
3 too.

## 5. Integration

**Where it sits**: in `pipeline.correct()`, between the field estimate and its application. The field estimate stays
as it is.

    denoise (opt-in) -> field estimate (b0s + reversed pair; unchanged) -> motion_eddy (opt-in, new)
      -> apply: one resample per volume (motion + h + e_v, Jacobian) -> prep (rotated bvecs) -> UKF -> label

- **The flag**: `run(..., motion_eddy=None)` and `correct(s, timer, device, motion_eddy=None)`. None keeps today's
  path bit for bit, and `S.apply` stays the applier when it is None. Values are `"rigid"`, `"linear"` and
  `"quadratic"` (all phase 1), with an options dict for the schedule and predictor, as `tracker=` takes settings. When
  there is no reversed pair (`can_correct` false), the step can still run with `h = 0`, as eddy runs without topup.
  So the flag is independent of the pair, and the note says which case ran.
- **b-vectors**: `track()` passes `s.bvec` to `prepare` today. With the step on, it passes `corr.bvec`, the
  rotated table. That one call site changes.
- **The applier**: a new `S.apply_volumes(dwi, h, motion, eddy, ...)` that composes each volume's map and samples
  once. Two interpolations (motion, then field) would smooth twice.
- **Provenance**: new `Correction` fields, all None when the step is off:

  | field | content |
  |---|---|
  | `volume_motion` | (G, 6) |
  | `eddy` | (G, k) Hz, Hz/mm |
  | `bvec` | (3, G) rotated |
  | `outliers` | (G, Z) bool |
  | `motion_eddy` | dict: model, predictor and its order or hyperparameters, schedule, per-volume iterations, convergence and final cost, the shell anchors, summary displacements (rigid and eddy, max and p95 at the mask edge; eddy-QC-style metrics, Bastiani et al. 2019), seconds |

  Its time is `timer.seconds["motion_eddy"]`, among `pipeline_stages`. A short line is added to `Correction.note`.
  The ensemble's `runs.json` log copies the summary displacements. Bench scripts already hash their module
  (`module_sha256`) and record `pipeline_settings(key)`, and those settings gain `motion_eddy`.
- **The ensemble, per replicate.** Each replicate already reruns the whole pipeline from its own data
  (`Runner.replicates` -> `Runner.one` -> `P.correct`). With the flag, it reruns motion and eddy too. Rerunning the
  step is easy. Generating the replicates correctly takes more care:

  - **R0 (today)**: `ShellWildBootstrap` fits SH to the series as acquired. Motion between volumes then shows up as
    model misfit in the residuals, mostly at edges, and those residuals get sign-flipped. That is texture, not
    motion. A replicate built this way contains no motion for the estimator to find.
  - **R1 (the flag's replicate design; decision 4)**: a forward-model bootstrap. Fit the predictor in reference space to the
    corrected data. Push the fit forward through each volume's estimated transform (motion, `h`, `e_v`) into
    acquisition space. Compute the residuals there, against the data as acquired, so they are never interpolated.
    The replicate is the pushed-forward fit plus the sign-flipped residuals. The replicate then carries the scan's
    own motion and eddy distortion, and the pipeline has to estimate them again. The spread of the estimates across
    replicates measures the estimator's precision, which comes free as a QC number. Replaced outlier slices take
    residuals borrowed from their neighbors, not their own.
  - **R2 (conditional; decision 4)**: R1, pushed forward with another plausible trajectory, for example the scan's trajectory
    block-permuted, or one from another scan of the protocol. This puts the effect of motion itself (interpolation,
    edges, gradient rotation) into the envelope. Whether that closes part of the calibration gap is the experiment.

  **Run-time effect**: a replicate takes 83-207 s now (`calibration_cohortII.py`'s per-protocol table, M2). The step
  adds about 3-12 s, and R1's forward push one more resample, about 1-3 s. That is +5-10 % per run, about +2-4 min per
  17-run ensemble on the M2, and +2-4 s per run on CUDA. The GPU, not the CPU, stays the ensemble's bottleneck.

## 6. Validation plan

### 6.1 Against FSL eddy

topup + eddy run on every scan in the calibration pairs: cohort II B2000, B32, A2000r19 and A32, and the ON-Harmony
sessions. They run as decision 8 lists. Settings: `--repol`; `--flm=linear` and then `quadratic`; `--slm=linear` for
the 32-direction protocols. The script that runs them records its commands, versions and times, as
`bench/topup_ref.py` does. eddy is fed tractline's field (decision 8). Phase 0's runs are the first of these. Through
the import path, tractline compares:

| quantity | from eddy | acceptance (eddy is not ground truth: these are agreement bounds) |
|---|---|---|
| per-volume rigid displacement at the mask edge | `eddy_parameters` | median abs difference <= 0.3 mm; p95 <= 0.8 mm |
| eddy displacement at the edge | `eddy_parameters` | median abs difference <= 0.3 mm |
| rotated b-vectors | `eddy_rotated_bvecs` | median angle <= 0.3 deg |
| corrected series | the corrected series | WM FA mean abs difference <= 0.02; NCC per volume >= 0.98 |
| outlier slices (phase 2) | `eddy_outlier_map` | Cohen's kappa >= 0.7 |

### 6.2 Synthetic, known truth (CPU, before any GPU time)

- **Truth**: a low-motion scan's own LOO-SH fit in reference space (ON-Harmony, with 5 b0s, and a cohort II B2000
  scan). The static field is the scan's own estimate.
- **Forward model**: the same one as R1, pushed forward with known parameters:
  - rotations up to 3 deg and translations up to 3 mm, as random walks plus jumps;
  - eddy fields linear in the gradient, up to 2 voxels at the edge at b = 2000, plus quadratic terms in phase 2;
  - noise at the scan's own SNR (`noise_check`);
  - phase 2: dropped slices (5 % of volumes, one slice group each).
- **A second, independent simulator**: Graham et al. (2016)'s simulation (spin history, realistic motion), so the
  test is not of the model against itself.
- **eddy on the same synthetic scans.** With a known truth, eddy stops being only an agreement target and becomes a
  benchmark. tractline's step should come within 1.5x of eddy's parameter error. Where it beats eddy, the reason
  should be understood before it is believed.

Acceptance (phase 1):

| quantity | bound |
|---|---|
| rigid displacement at the edge | RMS error <= 0.1 voxel |
| eddy displacement at the edge | RMS error <= 0.15 voxel |
| b-vector angle | <= 0.2 deg median |
| sign test (a pure 5 deg rotation) | recovered with the right sign |
| zero-motion input | identity to 0.02 voxel, with outputs within float32 rounding of today's `apply` when the step's parameters are zero |
| devices | same result on CPU (float64), MPS and CUDA (float32) to 0.02 voxel |

### 6.3 Effect on reproducibility and on the calibration gap (the question that matters)

Data on disk, read-only under `~/tmp/data/tractography`. Headers inspected for this document:

- **ON-Harmony** (ds004712; 20 subjects): 104 x 104 x 72 x 105 at 2.02 x 2.02 x 2.0 mm, uint16, phase encoding j-,
  readout 0.069 s, multiband 3, `SliceTiming` present. 13192 and 16974 have six OXF1PRI sessions (14229 and 16766:
  OXF2PRI).
- **MASiVar cohort II** (ds003416; 39 sessions on disk):
  - scanner B b2000n96: 96 x 96 x 48 x 97 at 2.5 mm, int16, no sidecars;
  - b1000n32: 96 x 96 x 48 x 33;
  - scanner A b2000n96: 112 x 112 x 48 x 97 at 1.89 x 1.89 x 2.2 mm.

What the CPU check already settled is listed in the summary. What remains open is any direct test with corrected
data, on the gap and above all on accuracy.

**Phase 0's pairs** (about 8), chosen by the CPU check's proxies, where correction has the most to change:

- ON-Harmony cross-site pairs 16974 and 10975, the extreme ones;
- the three OXF2PRI001 rescans with the most eddy (of the 9 that carry about 2.5 times the others'), with their
  OXF1PRI A;
- the same-scanner ON-Harmony session with the most b0 motion, with its A;
- one cohort II B2000 pair as a low-eddy control.

ON-Harmony has T1 images on disk, so accuracy can be judged against anatomy there (`tractline.t1check`). Cohort II
has none.

Design: the existing 17-run ensembles and rescans (`calibration_cohortII.py`, `calibration_onharmony.py`, mode-3
registration) are rerun with only the correction changed. The arms:

| arm | when | correction | replicates | what it answers |
|---|---|---|---|---|
| today | exists | field only | R0 | the baseline |
| E-fixed | phase 0, the 8 pairs | eddy once per scan, on tractline's field | R0 of the eddy-corrected series | does correction move the gap, and does it improve accuracy? |
| E-rerun | only if E-fixed moves the gap by more than 1 point | eddy per replicate (Modal) | R1 built from eddy's parameters | the gap with the correction's own variability in the ensemble |
| flag + R1 | phase 1, all pairs | tractline's step | R1 | the step's gap and accuracy against E-fixed's |
| flag + R2 | only with E-rerun | tractline's step | R2 | motion as a source of variation |

Accuracy measures (phase 0, E-fixed against today, on the same pairs):

- the tensor fit's residual RMS in white matter;
- the spread across volumes of each volume's edge alignment to the T1 (ON-Harmony; t1check's normalized gradient
  fields);
- the FA maps' test-retest agreement between A and the rescan;
- tract-label and envelope agreement between A and the rescan, without the recalibration map.

**Phase 0's verdict**, written down before the runs:

- **Gap**: a change of under 1 point at 0.9 on these pairs closes the question for the paper. Motion and eddy are
  then not the missing term, consistent with the CPU check, and E-rerun and R2 are not run.
- **Accuracy**: phase 1 goes ahead if any one of these holds on ON-Harmony:
  - the WM residual falls by 10 % or more;
  - the T1 alignment spread falls by 0.5 mm or more at its 95th percentile;
  - rescan agreement improves beyond its interval.

  Otherwise phase 1 drops behind other roadmap work, kept as a parity feature with eddy (rotated b-vectors, eddy
  outputs).

Measured (all arms that run):

- the as-stated gap (coverage minus target at 0.8 / 0.9 / 0.95), per stratum and pooled by within-stratum ranks, as
  `calibration_motion_check.py` pools;
- rescan agreement of tract labels and envelopes, without the recalibration map;
- the recalibration map relearned per arm (decision 6), with its leave-one-subject-out coverage.

The motion check's proxies sort scans by motion, and they set the prediction before any run: if they explain part
of the gap, the gap should shrink most where motion is largest.

Acceptance:

1. No stratum's gap gets worse beyond its bootstrap interval.
2. The pooled gap at 0.9 shrinks with an interval that excludes 0. Otherwise the result is reported as a null, which
   is informative for the paper too: preprocessing would then not be the missing term.
3. Rescan label agreement is unchanged or better.
4. Run-time cost is within section 3's estimate x 2.
5. flag + R1's accuracy measures are within 20 % of E-fixed's, and its gap within E-fixed's interval (E-rerun's,
   if that ran).

GPU budget: phase 0 about 150 tracking runs; phase 1 one arm over all pairs. Both wait for the current queue.

## 7. Phases, effort and risks

| phase | scope | effort (focused) | main risks |
|---|---|---|---|
| 0 | a direct test with eddy: install fsl-eddy (**done 2026-10-20**; validated on the synthetic scan, section 2.1); eddy on tractline's field for A and the rescan of about 8 extreme pairs (section 6.3); the import loader; arm E-fixed; the accuracy measures. **Gate**: the gap verdict settles the paper's question; the accuracy verdict decides phase 1 | 3-4 days of work, 16 eddy runs (36 min each for 55 volumes on 6 M2 threads, measured: about 10 h locally, or in parallel on Modal), about 150 tracking runs (5-8 GPU hours) | (resolved: eddy installed natively, 2026-10-20) FSL's channel has `fsl-eddy` 2602.0 for osx-arm64 (about 1 MB), and every dependency it lists is already in `DATA/fsl-env` from the topup install, so it adds one package through the same standalone micromamba, no conda; `eddy_cpu` is threaded (`--nthr`); eddy's `--field` input behaving as documented; few pairs, so the gap verdict is only coarse (it only needs to tell under 1 point from more) |
| 1 | rigid + quadratic eddy, acquisition-space residual (shared with R1's forward model), cubic prediction rebuilt every step, LOO-SH predictor, the joint second-level model as an option, shell anchors, b-vector rotation, the fused kernels (Metal: prototyped, `metal_step.py`; Triton: to port), `apply_volumes` (one cubic resample, 8 fetches), forward-model bootstrap (R1), flag and provenance, synthetic tests (starting from `bench/motion_eddy/proto.py`), CPU / MPS / CUDA | 4-5 weeks | the prototype matching eddy on one synthetic scan only (section 2.1), the 3x-motion floor (0.16 voxel at about 9 mm and 3 deg; eddy not run there); the Triton port's speed unmeasured; gauge and confounds (c0 against PE translation; one-b0 protocols); registration of noisy high-b volumes (b = 2000-2500 at 2.5 mm; Ben-Amitay et al. 2012, Nilsson et al. 2015); b-vector sign conventions; float32 rounding at small steps on MPS; GPU time contended by the queue |
| 2 | slice outliers and replacement, GP predictor (multi-shell, if decision 1's test passes), the full eddy comparison of section 6.1 | about 2 weeks | the GP's hyperparameter fit; outlier thresholds tuned on few protocols; the bootstrap's change moves every committed ensemble result (a new arm, not a replacement) |
| 3 | slice-to-volume, susceptibility-by-movement | 4-8 weeks | needs slice timing (absent for cohort II); identifiability per slice group; research-grade |

Cross-cutting risks:

- **Interpolation changes the noise.** Trilinear resampling with a sub-voxel shift that varies per volume smooths
  each volume's noise by a different amount. That shifts the residual structure the bootstrap sees, and so the
  envelopes. Decision 3 (cubic) limits it; a noise check (`noise_check`) before and after measures what is left.
- **The step will probably not move the gap.** The CPU check puts motion under about 5 % of it, and eddy at none.
  Its case is accuracy and parity with eddy. Phase 0 tests accuracy with eddy itself before any GPU code is written,
  and the step is built only if accuracy improves.
- **Calibration bookkeeping.** Maps learned without the flag must not be applied with it (decision 6).

## 8. Prior art and licensing

Bibliographic data was checked against Crossref and the abstracts against PubMed or Europe PMC on 2026-10-20.
Licensing is as stated on each project's own page or LICENSE file. The project's position, held throughout (topup
as the field estimate's reference): running FSL as a reference or oracle to evaluate tractline, in any environment,
complies with its license. No reading of the license seen so far says otherwise. Reading FSL's source to understand
its methods does too. What the license governs is FSL's code itself (its use, modification and redistribution).
tractline neither ships nor derives code from it, and attributes method choices taken from it (decision 9).

### Tools

| tool | what it does | license | GPU | usable by tractline |
|---|---|---|---|---|
| **FSL eddy** (Andersson & Sotiropoulos 2016; outliers 2016; slice-to-volume 2017; susceptibility-by-movement 2018) | GP prediction, rigid motion and polynomial eddy fields (`--flm` linear, quadratic (default) or cubic), second-level model, `--repol` | FSL license: free for non-commercial use only; commercial use (including research aimed at a product) through Oxford University Innovation; source distributed, derivatives restricted (https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html) | `eddy_cuda`, NVIDIA only (FSL eddy FAQ); up to 5x over the CPU (Maller et al. 2020) | the oracle: run in any environment (local conda env, Modal CPU or CUDA) and read through the import path (decision 8); its source read for method, none copied (decision 9) |
| **fsl-metal** (community port, https://github.com/neurolabusc/fsl-metal, announced June 2026) | `eddy_metal` on Apple silicon; float32 results "equivalent but not identical" | FSL license | Metal | an eddy oracle on the M2's GPU when the queue is idle; its self-reported speeds are not verified here, and its float32 results should be checked once against `eddy_cpu` |
| **TORTOISE / DIFFPREP** (Rohde et al. 2004; Pierpaoli et al. 2010; Irfanoglu et al. 2025, TORTOISEV4) | MI registration of each volume to the b0 with rigid motion and a quadratic or cubic eddy model, one interpolation | GPL-3.0 (https://github.com/QMICodeBase/TORTOISEV4) | CUDA build (`TORTOISEProcess_cuda`; `src/cuda_src`: MI, rigid and quadratic transforms, resampling) | read for method and compared against; its GPL code is not copied into Apache-2.0 tractline |
| **DIPY** (Garyfallidis et al. 2014) | `dipy.align.register_dwi_series` (1.3.0) and `motion_correction` (1.5.0): affine registration of each volume to the b0 (MI or CC); `reorient_bvecs` (Leemans & Jones 2009); no eddy model, no prediction | BSD-3-Clause | none | the planned CPU test of affine correction (background); a baseline arm in section 6.3 |
| **MRtrix3 `dwifslpreproc`** (Tournier et al. 2019) | a wrapper that runs topup and eddy with the surrounding data and metadata handling | MPL-2.0 (the wrapper; it calls FSL) | through eddy | a convenient way for a user to produce the eddy reference |
| **QSIPrep SHORELine** (Cieslak et al. 2021) | leave-one-out 3dSHORE (or tensor) prediction, registration, b-vector rotation, 2 iterations by default; motion only, no eddy; its docs mark it "scheduled for removal" in favor of TORTOISE | BSD-3-Clause | none | the precedent for section 1.4's LOO predictor |
| **NiFreeze** (formerly eddymotion; nipreps; doi:10.5281/zenodo.4680599) | open Python reimplementation of eddy-style motion and eddy correction on DIPY models | Apache-2.0 | not verified | the closest license-compatible prior art; worth reading before phase 1 |
| **Eddeep** (Legouhy et al. 2024) | deep-learning eddy-current correction (translator + registration networks, Keras) | Apache-2.0 | yes (inference) | related work; a learned model is not this design's approach |
| **FireANTs** (Jena et al. 2026), **deepali** (doi:10.5281/zenodo.8170161) | general GPU registration in PyTorch | Apache-2.0 | yes | patterns only; their generic optimizers do not exploit the block-diagonal per-volume structure |

No GPU implementation of eddy-style motion and eddy correction was found that is both open-source and under a
permissive license. That gap is what this design fills.

### Papers

1. Andersson JLR, Sotiropoulos SN. An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging. NeuroImage 2016;125:1063-1078. doi:10.1016/j.neuroimage.2015.10.019
2. Andersson JLR, Sotiropoulos SN. Non-parametric representation and prediction of single- and multi-shell diffusion-weighted MRI data using Gaussian processes. NeuroImage 2015;122:166-176. doi:10.1016/j.neuroimage.2015.07.067
3. Andersson JLR, Graham MS, Zsoldos E, Sotiropoulos SN. Incorporating outlier detection and replacement into a non-parametric framework for movement and distortion correction of diffusion MR images. NeuroImage 2016;141:556-572. doi:10.1016/j.neuroimage.2016.06.058
4. Andersson JLR, Graham MS, Drobnjak I, Zhang H, Filippini N, Bastiani M. Towards a comprehensive framework for movement and distortion correction of diffusion MR images: within volume movement. NeuroImage 2017;152:450-466. doi:10.1016/j.neuroimage.2017.02.085
5. Andersson JLR, Graham MS, Drobnjak I, Zhang H, Campbell J. Susceptibility-induced distortion that varies due to motion: correction in diffusion MR without acquiring additional data. NeuroImage 2018;171:277-295. doi:10.1016/j.neuroimage.2017.12.040
6. Andersson JLR, Skare S. A model-based method for retrospective correction of geometric distortions in diffusion-weighted EPI. NeuroImage 2002;16(1):177-199. doi:10.1006/nimg.2001.1039
7. Andersson JLR, Skare S, Ashburner J. How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging. NeuroImage 2003;20(2):870-888. doi:10.1016/S1053-8119(03)00336-7
8. Leemans A, Jones DK. The B-matrix must be rotated when correcting for subject motion in DTI data. Magn Reson Med 2009;61(6):1336-1349. doi:10.1002/mrm.21890
9. Rohde GK, Barnett AS, Basser PJ, Marenco S, Pierpaoli C. Comprehensive approach for correction of motion and distortion in diffusion-weighted MRI. Magn Reson Med 2004;51(1):103-114. doi:10.1002/mrm.10677
10. Jezzard P, Barnett AS, Pierpaoli C. Characterization of and correction for eddy current artifacts in echo planar diffusion imaging. Magn Reson Med 1998;39(5):801-812. doi:10.1002/mrm.1910390518
11. Haselgrove JC, Moore JR. Correction for distortion of echo-planar images used to calculate the apparent diffusion coefficient. Magn Reson Med 1996;36(6):960-964. doi:10.1002/mrm.1910360620
12. Zhuang J, Hrabe J, Kangarlu A, et al. Correction of eddy-current distortions in diffusion tensor images using the known directions and strengths of diffusion gradients. J Magn Reson Imaging 2006;24(5):1188-1193. doi:10.1002/jmri.20727
13. Ben-Amitay S, Jones DK, Assaf Y. Motion correction and registration of high b-value diffusion weighted images. Magn Reson Med 2012;67(6):1694-1702. doi:10.1002/mrm.23186
14. Nilsson M, Szczepankiewicz F, van Westen D, Hansson O. Extrapolation-based references improve motion and eddy-current correction of high b-value DWI data: application in Parkinson's disease dementia. PLoS One 2015;10(11):e0141825. doi:10.1371/journal.pone.0141825
15. Bai Y, Alexander DC. Model-based registration to correct for motion between acquisitions in diffusion MR imaging. ISBI 2008:947-950. doi:10.1109/ISBI.2008.4541154
16. Cieslak M, Cook PA, He X, et al. QSIPrep: an integrative platform for preprocessing and reconstructing diffusion MRI data. Nat Methods 2021;18(7):775-778. doi:10.1038/s41592-021-01185-5
17. Tournier J-D, Smith R, Raffelt D, et al. MRtrix3: a fast, flexible and open software framework for medical image processing and visualisation. NeuroImage 2019;202:116137. doi:10.1016/j.neuroimage.2019.116137
18. Garyfallidis E, Brett M, Amirbekian B, et al. Dipy, a library for the analysis of diffusion MRI data. Front Neuroinform 2014;8:8. doi:10.3389/fninf.2014.00008
19. Irfanoglu MO, Nayak A, Taylor P, Thai A, Pierpaoli C. TORTOISEV4: reimagining the NIH diffusion MRI processing pipeline. Imaging Neurosci 2025;3:IMAG.a.948. doi:10.1162/imag.a.948
20. Pierpaoli C, Walker L, Irfanoglu MO, et al. TORTOISE: an integrated software package for processing of diffusion MRI data. ISMRM 2010, abstract 1597.
21. Maller JJ, Grieve SM, Vogrin SJ, Welton T. Performance gains with Compute Unified Device Architecture-enabled eddy current correction for diffusion MRI. NeuroReport 2020;31(10):746-753. doi:10.1097/WNR.0000000000001475
22. Sairanen V, Leemans A, Tax CMW. Fast and accurate Slicewise OutLIer Detection (SOLID) with informed model estimation for diffusion MRI data. NeuroImage 2018;181:331-346. doi:10.1016/j.neuroimage.2018.07.003
23. Christiaens D, Cordero-Grande L, Pietsch M, et al. Scattered slice SHARD reconstruction for motion correction in multi-shell diffusion MRI. NeuroImage 2021;225:117437. doi:10.1016/j.neuroimage.2020.117437
24. Graham MS, Drobnjak I, Zhang H. Realistic simulation of artefacts in diffusion MRI for validating post-processing correction techniques. NeuroImage 2016;125:1079-1094. doi:10.1016/j.neuroimage.2015.11.006
25. Bastiani M, Cottaar M, Fitzgibbon SP, et al. Automated quality control for within and between studies diffusion MRI data using a non-parametric framework for movement and distortion correction. NeuroImage 2019;184:801-812. doi:10.1016/j.neuroimage.2018.09.073
26. Legouhy A, Callaghan R, Stee W, et al. Eddeep: fast eddy-current distortion correction for diffusion MRI with deep learning. MICCAI 2024, LNCS:152-161. doi:10.1007/978-3-031-72069-7_15
27. Jena R, Chaudhari P, Gee JC. Adaptive Riemannian optimization for multi-scale diffeomorphic matching (FireANTs). Nat Commun 2026;17:4774. doi:10.1038/s41467-026-72508-3
28. Cai LY, Yang Q, Kanakaraj P, et al. MASiVar: multisite, multiscanner, and multisubject acquisitions for studying variability in diffusion weighted MRI. Magn Reson Med 2021;86(6):3304-3320. doi:10.1002/mrm.28926
29. Warrington S, et al. ON-Harmony (ds004712). Sci Data 2025;12:609. doi:10.1038/s41597-025-04822-2

Not verified for this document: Bai & Alexander's (15) content beyond its title; fsl-metal's speed claims; whether NiFreeze uses a GPU; the TORTOISE NIH site's own license wording (the
repository says GPL-3.0).
