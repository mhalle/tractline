"""Phase 0 (docs/gpu-motion-eddy-design.md, sections 6.3 and 7): does motion and eddy-current correction move the
calibration gap on the pairs where it has the most to change, and does it improve accuracy?

The pairs (chosen by bench/results/calibration_motion_check.json's proxies, before any run): ON-Harmony cross-site
pairs (A = OXF1PRI001, rescan = OXF2PRI001) of 10975 and 16974 (the most b0 motion), 16793, 16794 and 16745 (the most
eddy-like displacement), 14221 (the least of both: the control), and the same-scanner pair 16766 (A = OXF2PRI001,
rescan = OXF2PRI002; the most b0 motion on one scanner). Each scan: the ensemble's series (5 b0s + b2000, 55 volumes).

Arms, both with the susceptibility correction applied once and the replicates drawn from the corrected series (so
the field does not vary across the 17 runs in either):
  T  today's correction (tractline's field) applied once; the replicates of that series;
  E  FSL eddy (eddy_cpu, quadratic, fed tractline's field) applied once; the replicates of eddy's series, with its
     rotated b-vectors.
T against E isolates what motion and eddy correction changes; the published arm (field re-estimated in every run)
stays the reference. The rescan is placed on A's grid by the published arm's registration (to_A from its runs.json),
so the registration is the same in all arms.

    python phase0.py prep [--subject ...]        # tractline's correction per scan, eddy's inputs (field in Hz)
    python phase0.py fieldcheck <scan dir>       # eddy on 10 volumes, 1 iteration: does it read the field as we do?
    python phase0.py run --arm T|E [--subject ...] [--runs 16]
    python phase0.py collect --arm T|E [--subject ...]
    python phase0.py evaluate
"""
import argparse, json, os, shutil, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import calibration_onharmony as CO                                              # noqa: E402
import calibration_runs as CR                                                   # noqa: E402
import calibration_cohortII as CII                                              # noqa: E402
from tractline import pipeline as P                                             # noqa: E402
from tractline.data import DATA                                                 # noqa: E402

PAIRS = {"10975": ("OXF1", "OXF2PRI001"), "16974": ("OXF1", "OXF2PRI001"), "16793": ("OXF1", "OXF2PRI001"),
         "16794": ("OXF1", "OXF2PRI001"), "16745": ("OXF1", "OXF2PRI001"), "14221": ("OXF1", "OXF2PRI001"),
         "16766": ("OXF2", "OXF2PRI002")}
OUT = DATA / "derived" / "phase0"
FSL = DATA / "fsl-env"
EDDY_MIN_FREE_GB = 25.0                # eddy stops below this (the tracking stage: calibration_runs.MIN_FREE_GB, 20)
EDDY_ARGS = ["--flm=quadratic", "--slm=none", "--data_is_shelled", "--nthr=6"]


def scans():
    """(subject, session) for every scan in the pairs: A and its rescan."""
    out = []
    for sub, (sc, tag) in PAIRS.items():
        out += [(sub, f"{CO.SCANNERS[sc]}001"), (sub, tag)]
    return out


def scan_dir(sub, ses):
    return OUT / f"{sub}_{ses}"


def prep(args):
    import nibabel as nib
    from tractline.mask import median_otsu
    for sub, ses in scans():
        if args.subject and sub not in args.subject:
            continue
        d = scan_dir(sub, ses)
        if (d / "done.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        s = CO.load_checked(sub, ses)
        timer = P.Timer()
        with P.exact_float32(CR.DEVICE):
            corr = P.correct(s, timer, CR.DEVICE)
        P.release_memory(CR.DEVICE)
        aff = s.affine
        nib.save(nib.Nifti1Image(np.asarray(s.dwi, np.float32), aff), d / "imain.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(corr.field_hz, np.float32), aff), d / "field_hz.nii.gz")
        np.save(d / "today_dwi.npy", np.asarray(corr.dwi, np.float32))           # arm T's series
        np.savetxt(d / "bvals", np.asarray(s.bval)[None], fmt="%g")
        np.savetxt(d / "bvecs", np.asarray(s.bvec), fmt="%.6f")
        pe = np.asarray(s.pe_vectors, float)[0]
        (d / "acqp.txt").write_text(" ".join(f"{v:g}" for v in pe) + f" {s.readout_s}\n")
        (d / "index.txt").write_text(" ".join(["1"] * len(s.bval)) + "\n")
        b0 = np.asarray(corr.dwi)[..., np.asarray(s.bval) < 50].mean(-1)
        _, mask = median_otsu(b0, median_radius=4, numpass=4)
        nib.save(nib.Nifti1Image(np.asarray(mask, np.uint8), aff), d / "mask.nii.gz")
        info = {"subject": sub, "session": ses, "scan": s.name, "shape": list(s.dwi.shape), "pe_vector": pe.tolist(),
                "readout_s": s.readout_s, "residual_left": corr.residual_left, "correct_seconds": timer.seconds,
                "seconds": round(time.time() - t0, 1)}
        (d / "done.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info), flush=True)


def _eddy(d, out, extra=()):
    env = {**os.environ, "FSLDIR": str(FSL), "FSLOUTPUTTYPE": "NIFTI_GZ", "PATH": f"{FSL / 'bin'}:{os.environ['PATH']}"}
    cmd = ["eddy_cpu", f"--imain={d / 'imain'}", f"--mask={d / 'mask'}", f"--acqp={d / 'acqp.txt'}", f"--index={d / 'index.txt'}",
           f"--bvecs={d / 'bvecs'}", f"--bvals={d / 'bvals'}", f"--field={d / 'field_hz'}", f"--out={out}", *EDDY_ARGS, *extra]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (Path(str(out) + ".log")).write_text(" ".join(cmd) + "\n\n" + r.stdout + r.stderr)
    return r.returncode, round(time.time() - t0, 1)


def eddy_all(args):
    """eddy on every prepared scan (skips those done), one at a time, 6 threads."""
    for sub, ses in scans():
        if args.subject and sub not in args.subject:
            continue
        d = scan_dir(sub, ses)
        if (d / "eddy.nii.gz").exists() or not (d / "done.json").exists():
            continue
        if CR.free_gb() < EDDY_MIN_FREE_GB:                                       # the disk is shared with the other work
            sys.exit(f"stopped: {CR.free_gb():.1f} GB free, below {EDDY_MIN_FREE_GB:g} GB")
        code, secs = _eddy(d, d / "eddy")
        print(f"{sub} {ses} eddy exit {code} {secs:.0f} s", flush=True)


def fieldcheck(args):
    """eddy on the first 10 volumes for one iteration, fed our field; its corrected b0 against ours (field only)."""
    import nibabel as nib
    d = Path(args.dir); t = d / "fieldcheck"; t.mkdir(exist_ok=True)
    img = nib.load(d / "imain.nii.gz"); bv = np.loadtxt(d / "bvals"); sel = np.r_[np.flatnonzero(bv < 50), np.flatnonzero(bv >= 50)[:5]]
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj, np.float32)[..., sel], img.affine), t / "imain.nii.gz")
    np.savetxt(t / "bvals", bv[sel][None], fmt="%g"); np.savetxt(t / "bvecs", np.loadtxt(d / "bvecs")[:, sel], fmt="%.6f")
    (t / "index.txt").write_text(" ".join(["1"] * len(sel)) + "\n")
    for f in ("acqp.txt", "mask.nii.gz", "field_hz.nii.gz"):
        shutil.copy(d / f, t / f)
    code, secs = _eddy(t, t / "eddy", ["--niter=1"])
    E = np.asarray(nib.load(t / "eddy.nii.gz").dataobj, np.float32)
    ours = np.load(d / "today_dwi.npy")[..., sel]
    raw = np.asarray(nib.load(t / "imain.nii.gz").dataobj, np.float32)
    m = np.asarray(nib.load(d / "mask.nii.gz").dataobj) > 0
    ncc = lambda a, b: float(np.corrcoef(a[m], b[m])[0, 1])
    out = {"eddy exit": code, "seconds": secs, "b0 0: NCC eddy vs ours (field only)": round(ncc(E[..., 0], ours[..., 0]), 4),
           "b0 0: NCC raw vs ours": round(ncc(raw[..., 0], ours[..., 0]), 4), "b0 0: NCC eddy vs raw": round(ncc(E[..., 0], raw[..., 0]), 4)}
    print(json.dumps(out), flush=True)
    (t / "check.json").write_text(json.dumps(out, indent=1))


class ArmRecords(CO.Records):
    """The published pair's records, in a folder of the arm's own: <pair>-phase0<arm>, with only the phase 0 rescan."""
    def __init__(self, subject, sc, arm, smoke=False):
        super().__init__(subject, sc, smoke)
        tag = PAIRS[subject][1]
        self.pair = replace(self.pair, other={tag: self.pair.other[tag]})
        self.dir = self.dir.with_name(self.dir.name + f"-phase0{arm}")


def series(arm, sub, ses, s):
    """The arm's corrected series and b-vectors for scan (sub, ses): T today's correction, E eddy's (rotated b-vectors)."""
    import nibabel as nib
    d = scan_dir(sub, ses)
    if arm == "T":
        return np.load(d / "today_dwi.npy"), np.asarray(s.bvec)
    return np.asarray(nib.load(d / "eddy.nii.gz").dataobj, np.float32), np.loadtxt(d / "eddy.eddy_rotated_bvecs")


class ArmRunner(CO.Runner):
    def __init__(self, subject, arm):
        sc = PAIRS[subject][0]
        self.key, self.shell, self.sc, self.arm = CO.KEY[sc], CO.SHELL, sc, arm
        self.R = ArmRecords(subject, sc, arm)
        self.subject, self.pair = subject, self.R.pair
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": CR.git_state(), "phase0 arm": arm, "pipeline": CO.pipeline_settings(),
                                        "arms": __doc__.split("Arms,")[1].split("T against E")[0].strip(),
                                        "registration": "the published arm's to_A (mode 3), reused"})
        self.labeler = None

    def reference(self):
        A = CO.load_checked(self.subject, self.pair.ref[0], self.key)
        dwi, bvec = series(self.arm, self.subject, self.pair.ref[0], A)
        Ax = replace(A, dwi=dwi, bvec=bvec, b0s=None)
        if not self.R.have("A"):
            self.log["A_scan"] = {"name": A.name, "series": A.series, "arm": self.arm, "shape": list(A.dwi.shape)}

            def companions(corr, tg):
                for name, arr in (("A_meanb0", CO.mean_b0(corr.dwi, A.bval)), ("A_mask", tg.mask)):
                    q = self.R.new(name, ".npy"); tmp = q.with_name(q.stem + ".partial.npy"); np.save(tmp, arr); tmp.replace(q)
            self.one(Ax, "A", corrected=dwi, before_record=companions)
        return Ax, np.load(self.R.path("A_meanb0", ".npy")), np.load(self.R.path("A_mask", ".npy"))

    def rescan(self, Ax):
        tag = PAIRS[self.subject][1]
        name = CII.rescan_tag(tag, CO.REG)
        if self.R.have(f"{name}_onA"):
            return
        pub = json.loads((CO.Records(self.subject, self.sc).dir / "runs.json").read_text())[f"{name}_scan"]
        B = CO.load_checked(self.subject, tag)
        dwi, bvec = series(self.arm, self.subject, tag, B)
        T = np.array(pub["to_A"])
        dwiA, bvecA = CO.onto_reference(dwi, bvec, B.affine, T, Ax.affine, Ax.dwi.shape[:3])
        self.log[f"{name}_scan"] = {**{k: pub[k] for k in ("to_A", "rotation_deg", "translation_mm", "ncc_after", "QC") if k in pub},
                                    "name": B.name, "term": CO.term_of(self.sc, tag), "session": tag, "arm": self.arm,
                                    "registration": "the published arm's (reused)"}
        self.save_log()
        self.one(replace(Ax, dwi=dwiA, bval=B.bval, bvec=bvecA, b0s=None), f"{name}_onA", corrected=dwiA)

    def replicates(self, Ax, mask, runs):
        from _bootstrap import ShellWildBootstrap
        need = [r for r in range(1, runs + 1) if not self.R.have(f"{CR.FLIP}/r{r:03d}")]
        if not need:
            return
        wb = ShellWildBootstrap(Ax.dwi, Ax.bval, Ax.bvec, extra_b0s=None, b0_noise="added", mask=mask, flip=CR.FLIP, fwhm=CR.FWHM)
        self.log.setdefault(f"bootstrap_{CR.FLIP}", wb.noise_check(mask))
        for r in need:
            dwi, _ = wb.replicate(r)
            self.one(replace(Ax, dwi=dwi, b0s=None), f"{CR.FLIP}/r{r:03d}", corrected=dwi)


PROTO = HERE / "proto.py"
PROTO_ARGS = ["--device", "mps", "--all-shells", "--fast", "--metal", "--coarse", "0", "--outer", "40", "--mask-output"]
# no shell anchor: on 14221's scan the MI anchor moved the b2000 shell 1.27 mm (its rescan and replicates 0.09-0.16):
# a wrong optimum. Without it each shell's mean pose stays as acquired (b0 placement vs eddy 0.08-0.15 voxel on the
# two real scans, sections 2.7 and 2.10). The anchor's robustness is a follow-up.
QUICK = ("10975", "16974", "16793", "14221")            # arm O's pairs: the most motion (2), the most eddy, the control


def motion_eddy(dwi, s, tmp):
    """Our GPU motion and eddy-current step (bench/motion_eddy/proto.py: 40 steps, no shell anchor,
    output masking, lambda 0.06) on a field-corrected series: (corrected series, rotated b-vectors, its summary)."""
    import nibabel as nib
    tmp.mkdir(parents=True, exist_ok=True)
    base = tmp / "in"
    nib.save(nib.Nifti1Image(np.asarray(dwi, np.float32), s.affine), str(base) + ".nii.gz")
    np.savetxt(str(base) + ".bval", np.asarray(s.bval)[None], fmt="%g"); np.savetxt(str(base) + ".bvec", np.asarray(s.bvec), fmt="%.6f")
    out = tmp / "out"
    r = subprocess.run([sys.executable, str(PROTO), "--dwi", str(base), "--pe", str(s.pe_axis), *PROTO_ARGS, "--real", str(out)],
                       capture_output=True, text=True, cwd=str(HERE.parent.parent))
    if r.returncode:
        raise RuntimeError(f"motion_eddy failed: {r.stderr[-800:]}")
    info = json.loads([l for l in r.stdout.splitlines() if l.startswith('{"data"')][-1])
    corr = np.asarray(nib.load(out / "ours.nii.gz").dataobj, np.float32)
    bvec = np.loadtxt(out / "ours_rotated_bvecs")
    return corr, bvec, {k: info.get(k) for k in ("estimate_s", "final_resample_s", "total_s", "shell_anchors (mm, deg)")}


class ORunner(ArmRunner):
    """Arm O: the published procedure, with our motion and eddy-current step after the field correction in every run.
    The field is re-estimated in every run as published, the replicates are the published ones (same scan, mask, seeds),
    and the rescan goes onto A's grid by the published registration: only the step differs."""
    def __init__(self, subject):
        super().__init__(subject, "O")
        self.pub = CO.Records(subject, self.sc).dir
        self.tmp = OUT / "arm_O_tmp" / subject

    def corrected(self, s):
        timer = P.Timer()
        with P.exact_float32(CR.DEVICE):
            c = P.correct(s, timer, CR.DEVICE)
        P.release_memory(CR.DEVICE)
        t0 = time.time()
        dwi, bvec, info = motion_eddy(c.dwi, s, self.tmp)
        return dwi, bvec, {"field": timer.seconds, "residual_left": c.residual_left, "motion_eddy": info,
                           "motion_eddy_wall_s": round(time.time() - t0, 1)}

    def reference(self):
        A = CO.load_checked(self.subject, self.pair.ref[0], self.key)
        if not self.R.have("A"):
            dwi, bvec, info = self.corrected(A)
            Ax = replace(A, dwi=dwi, bvec=bvec, b0s=None)
            self.log["A_scan"] = {"name": A.name, "series": A.series, "arm": "O", "shape": list(A.dwi.shape), "correction": info}

            def companions(corr, tg):
                for name, arr in (("A_meanb0", CO.mean_b0(corr.dwi, A.bval)), ("A_mask", tg.mask)):
                    q = self.R.new(name, ".npy"); tmp = q.with_name(q.stem + ".partial.npy"); np.save(tmp, arr); tmp.replace(q)
            self.one(Ax, "A", corrected=dwi, before_record=companions, extra={"correction": info})
        return A, np.load(self.R.path("A_meanb0", ".npy")), np.load(self.pub / "A_mask.npy")      # the published mask

    def rescan(self, A):
        tag = PAIRS[self.subject][1]
        name = CII.rescan_tag(tag, CO.REG)
        if self.R.have(f"{name}_onA"):
            return
        pub = json.loads((self.pub / "runs.json").read_text())[f"{name}_scan"]
        B = CO.load_checked(self.subject, tag)
        dwi, bvec, info = self.corrected(B)
        dwiA, bvecA = CO.onto_reference(dwi, bvec, B.affine, np.array(pub["to_A"]), A.affine, A.dwi.shape[:3])
        self.log[f"{name}_scan"] = {**{k: pub[k] for k in ("to_A", "rotation_deg", "translation_mm", "ncc_after", "QC") if k in pub},
                                    "name": B.name, "term": CO.term_of(self.sc, tag), "session": tag, "arm": "O", "correction": info,
                                    "registration": "the published arm's (reused)"}
        self.save_log()
        self.one(replace(A, dwi=dwiA, bval=B.bval, bvec=bvecA, b0s=None), f"{name}_onA", corrected=dwiA)

    def replicates(self, A, mask, runs):
        from _bootstrap import ShellWildBootstrap
        need = [r for r in range(1, runs + 1) if not self.R.have(f"{CR.FLIP}/r{r:03d}")]
        if not need:
            return
        wb = ShellWildBootstrap(A.dwi, A.bval, A.bvec, extra_b0s=A.b0s[..., len(A.b0_index):], b0_noise="added", mask=mask,
                                flip=CR.FLIP, fwhm=CR.FWHM)                     # as published (calibration_runs.Runner.replicates)
        for r in need:
            dwi, apa = wb.replicate(r)
            Ar = replace(A, dwi=dwi, b0s=np.concatenate([dwi[..., A.b0_index].astype(np.float64), apa], -1))
            cdwi, bvec, info = self.corrected(Ar)
            self.one(replace(Ar, dwi=cdwi, bvec=bvec, b0s=None), f"{CR.FLIP}/r{r:03d}", corrected=cdwi, extra={"correction": info})


def run(args):
    for sub in args.subject or (list(QUICK) if args.arm == "O" else list(PAIRS)):
        t0 = time.time()
        rn = ORunner(sub) if args.arm == "O" else ArmRunner(sub, args.arm)
        Ax, ref_b0, mask = rn.reference()
        rn.rescan(Ax)
        rn.replicates(Ax, mask, args.runs)
        print(f"{sub} arm {args.arm} done {time.time() - t0:.0f}s, {CR.free_gb():.1f} GB free", flush=True)


def collect(args):
    """calibration_onharmony.collect on the arm's folders (its Records swapped for the arm's)."""
    for sub in args.subject or list(PAIRS):
        sc = PAIRS[sub][0]
        CO.Records = lambda subject, sc_, smoke=False, _arm=args.arm: ArmRecords(subject, sc_, _arm, smoke)
        CO.collect(argparse.Namespace(subject=[sub], scanner=sc, smoke=False, runs=args.runs, force=args.force))


def evaluate(args):
    """The as-stated gap (coverage - target at 0.8 / 0.9 / 0.95) per pair and arm: published, T, E."""
    import importlib
    CC, CS = CO.CC, CO.CS
    Pub = importlib.reload(CO).Records
    rows = {}
    for sub, (sc, tag) in PAIRS.items():
        term = CO.term_of(sc, tag)
        r = {}
        for arm, d in (("published", Pub(sub, sc).dir), ("T", ArmRecords(sub, sc, "T").dir), ("E", ArmRecords(sub, sc, "E").dir),
                       ("O", ArmRecords(sub, sc, "O").dir)):
            p = d / CO.statements_name(args.runs + 1)
            if not p.exists():
                continue
            s = CC.Subject(p)
            if tag not in [t.split("reg")[0] for t in s.tags(term)]:
                continue
            cov = CS.pick(CS.stated(s, term)["coverage at stated level"])
            r[arm] = {q: round(cov[q] - float(q), 4) for q in cov}
        rows[sub] = {"pair": f"{CO.SCANNERS[sc]}001 -> {tag}", "term": term, "gap (coverage - target)": r}
        print(sub, json.dumps(rows[sub]), flush=True)
    arms = ("published", "T", "E", "O")
    mean = {a: {q: round(float(np.mean([v["gap (coverage - target)"][a][q] for v in rows.values() if a in v["gap (coverage - target)"]])), 4)
                for q in ("0.8", "0.9", "0.95")} for a in arms if any(a in v["gap (coverage - target)"] for v in rows.values())}
    diff = {q: round(float(np.mean([v["gap (coverage - target)"]["E"][q] - v["gap (coverage - target)"]["T"][q] for v in rows.values()
                                   if {"E", "T"} <= set(v["gap (coverage - target)"])])), 4) for q in ("0.8", "0.9", "0.95")}
    diffO = {q: round(float(np.mean([v["gap (coverage - target)"]["O"][q] - v["gap (coverage - target)"]["published"][q] for v in rows.values()
                                    if {"O", "published"} <= set(v["gap (coverage - target)"])])), 4) for q in ("0.8", "0.9", "0.95")}
    out = {"pairs": rows, "mean gap per arm": mean, "E - T, mean over pairs (the effect of motion and eddy correction)": diff,
           "O - published, mean over pairs (our step in every run)": diffO}
    print(json.dumps({k: out[k] for k in ("mean gap per arm", "O - published, mean over pairs (our step in every run)")}, indent=1))
    json.dump(out, open(HERE.parent / "results" / "motion_eddy" / "phase0_gap.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("prep"); a.add_argument("--subject", nargs="*")
    a = sp.add_parser("fieldcheck"); a.add_argument("dir")
    a = sp.add_parser("eddy"); a.add_argument("--subject", nargs="*")
    for c in ("run", "collect"):
        a = sp.add_parser(c); a.add_argument("--arm", choices=("T", "E", "O"), required=True); a.add_argument("--subject", nargs="*")
        a.add_argument("--runs", type=int, default=CR.RUNS); a.add_argument("--force", action="store_true")
    a = sp.add_parser("evaluate"); a.add_argument("--runs", type=int, default=CR.RUNS)
    args = ap.parse_args()
    {"prep": prep, "fieldcheck": fieldcheck, "eddy": eddy_all, "run": run, "collect": collect, "evaluate": evaluate}[args.cmd](args)
