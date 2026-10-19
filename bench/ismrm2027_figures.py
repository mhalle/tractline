"""Figures and summary statistics for the ISMRM 2027 abstract: calibrated uncertainty for automated tract
identification from a single diffusion scan (a 17-run whole-pipeline bootstrap ensemble, a per-protocol calibration
card learned on MASiVar cohort III, and the ISMRM 2015 phantom's systematic-error gap).

    uv run --group dipy --group rankfield python bench/ismrm2027_figures.py [all | fig1 | fig2 | fig3 | fig4 | fig5 | preview | stats]

Reads only existing results and records; runs no pipeline:
  - each subject's statements (calibration_card.py collect: DATA/ensemble/<subject>/card/<pair>/statements_R17.npz),
    refitted leave one subject out with calibration_card's own functions, and checked against the card's stored
    held-out summary (results/calibration_card_masivar_cIII.json);
  - results/robust_phantom.json ("envelopes, count-matched": the labeled 17-run envelopes and the bias-free
    ground-truth-subsample ensemble against the phantom's count-matched density presence);
  - for Fig 5, cIIIs04's stored run records (A, smooth/r001..r016 and the same-session rescan within_onA, on A's grid)
    and A's corrected mean b0: density presence of the CST per run, as collect computes it (checked against the
    statements' vote histograms).

Writes bench/results/ismrm2027/: fig1_pipeline, fig2_streamline_calibration, fig3_envelope_coverage,
fig4_phantom_gap, fig5_cst_envelopes, preview (each .pdf and .png at 300 dpi) and stats.json (95 % intervals by
resampling held-out subjects).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results/ismrm2027"
CARD_JSON = HERE / "results/calibration_card_masivar_cIII.json"
PHANTOM_JSON = HERE / "results/robust_phantom.json"
TERMS = ("same session", "other session")
TARGETS = (0.8, 0.9, 0.95)
BOOT, SEED = 10000, 20261019
EXAMPLE, TRACT, RESCAN = "cIIIs04", "CST", "within"        # Fig 5: the subject, the tract and the rescan shown

# Okabe-Ito, colorblind-safe
C = {"stated": "#D55E00", "card": "#0072B2", "pat16": "#E69F00", "biasfree": "#009E73", "gray": "#6E6E6E",
     "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "axes.linewidth": 0.8})
    return plt


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03)
        sizes[ext] = p.stat().st_size
        assert sizes[ext] < 2e6, f"{p} is {sizes[ext] / 1e6:.2f} MB, over the 2 MB limit"
    print(f"{name}: pdf {sizes['pdf'] / 1e3:.0f} kB, png {sizes['png'] / 1e3:.0f} kB", flush=True)


# ------------------------------------------------------------------------------------------- the held-out evaluation

def held_out():
    """Leave one subject out, per term, with calibration_card's own fit: per held-out subject the streamline
    reliability per vote count (as stated, card, PAT16's map), the ECEs, and the envelope coverage curves (as stated on
    the card's level grid; with the card at each target). Checked against the stored card's summary."""
    import calibration_card as CC
    from calibration_runs import CARD, Records
    subjects = [CC.Subject(Records(n).dir / "statements_R17.npz") for n in CARD]
    pat = CC.pat16_table()
    targets = np.round(np.arange(0.70, 0.9601, 0.005), 3)       # beyond 0.96 a fold's widest outline is reached
    out = {}
    for term in TERMS:
        rows = {}
        for s in subjects:
            if not s.tags(term):
                continue
            card = CC.fit_term([x for x in subjects if x is not s], term)
            R = s.R
            v, h = s.streamline(term)
            p_card = CC.apply_votes(card["streamline"], v, R)
            k = np.arange(R + 1)
            n = np.bincount(v, minlength=R + 1)
            obs = np.bincount(v, h.astype(float), minlength=R + 1) / np.maximum(n, 1)
            fold = np.array([np.nan if x is None else x for x in card["streamline"]["observed"]])
            curve = CC.term_curve(s, term)                                   # coverage at each stated level (GRID)
            lv = np.array([CC.level_for(card["_curve"], q)[0] for q in targets])
            widest = CC.term_shapes(s, term)["widest"]["coverage"]          # as evaluate(): a level not reached
            card_cov = np.array([curve[np.flatnonzero(CC.GRID == x)[0]] if x < 1 else widest for x in lv])
            at = lambda q: float(curve[np.flatnonzero(CC.GRID == round(q, 3))[0]])  # noqa: E731
            rows[s.name] = {
                "k": k, "n": n, "observed": obs, "stated": k / R, "card": fold, "pat16": pat(k / R),
                "ece": {"as stated": CC.ece(v / R, h)["ece"], "card": CC.ece(p_card, h)["ece"], "PAT16 map": CC.ece(pat(v / R), h)["ece"]},
                "grid": CC.GRID, "curve": curve, "targets": targets, "card curve": card_cov,
                "coverage": {f"{q:g}": {"as stated": at(q), "card": float(card_cov[np.flatnonzero(targets == q)[0]]),
                                        "card level": float(lv[np.flatnonzero(targets == q)[0]])} for q in TARGETS},
                "rescans": s.tags(term)}
        out[term] = rows
    check(out)
    return out


def check(ho):
    """The refit reproduces the stored card's held-out summary (means over subjects)."""
    summ = json.loads(CARD_JSON.read_text())["held-out evaluation (leave one subject out)"]["summary"]
    for term, rows in ho.items():
        s = summ[term]
        got = [np.mean([r["ece"][m] for r in rows.values()]) for m in ("as stated", "card", "PAT16 map")]
        assert np.allclose(got, s["streamline ECE: as stated / card / PAT16 map"], atol=6e-5), (term, got)
        for q in TARGETS:
            want = s["envelope coverage at target, as stated / card"][f"{q:g}"]
            got = [np.mean([r["coverage"][f"{q:g}"][m] for r in rows.values()]) for m in ("as stated", "card")]
            assert np.allclose(got, want, atol=6e-5), (term, q, got, want)
    print("held-out refit matches the stored card's summary", flush=True)


def boot_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    m = x[idx].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def stats(ho):
    rng = np.random.default_rng(SEED)
    out = {"what": "95 % percentile bootstrap intervals over held-out subjects (subjects resampled with replacement, "
                   f"{BOOT} draws, seed {SEED}) for the mean over subjects of each subject's leave-one-subject-out value. "
                   "Each subject's value comes from the card fitted on the other subjects (calibration_card.fit_term), "
                   "and the folds are not refitted inside the bootstrap. With 8 (same session) and 5 (other session) "
                   "subjects the intervals are coarse.",
           "source": "statements_R17.npz of the eight MASiVar cohort III subjects; the refit reproduces "
                     "results/calibration_card_masivar_cIII.json's held-out summary",
           "ECE": "streamline statement, P(a rescan gives the ensemble's mode), calibration_card.ECE_BINS bins",
           "coverage": "share of the rescan's density presence inside the envelope, pooled over the 42 named tracts "
                       "(weighted by the rescan's voxels); as stated: the envelope at stated level = target; card: at "
                       "the held-out fold's level for the target",
           "terms": {}}
    for term, rows in ho.items():
        names = sorted(rows)
        t = {"subjects": names, "n subjects": len(names), "ECE": {}, "coverage": {}}
        for m in ("as stated", "card", "PAT16 map"):
            x = [rows[n]["ece"][m] for n in names]
            t["ECE"][m] = {"mean": round(float(np.mean(x)), 4), "95 % CI": boot_ci(x, rng),
                           "per subject": {n: v for n, v in zip(names, x)}}
        for q in TARGETS:
            t["coverage"][f"{q:g}"] = {}
            for m in ("as stated", "card"):
                x = [rows[n]["coverage"][f"{q:g}"][m] for n in names]
                t["coverage"][f"{q:g}"][m] = {"mean": round(float(np.mean(x)), 4), "95 % CI": boot_ci(x, rng),
                                              "per subject": {n: round(v, 4) for n, v in zip(names, x)}}
            t["coverage"][f"{q:g}"]["card's stated level (per subject's fold)"] = {n: rows[n]["coverage"][f"{q:g}"]["card level"] for n in names}
        out["terms"][term] = t
    ph = phantom()
    out["phantom (ISMRM 2015, labeled presence, count-matched truth)"] = {
        f"{lv:g}": {"pipeline": ph["runs"][f"envelope {lv:g}"]["coverage"], "bias-free": ph["perfect ensemble"][f"envelope {lv:g}"]["coverage"],
                    "gap": round(ph["perfect ensemble"][f"envelope {lv:g}"]["coverage"] - ph["runs"][f"envelope {lv:g}"]["coverage"], 4)}
        for lv in (0.8, 0.9, 0.925, 0.95)}
    out["phantom note"] = "one simulated brain and one noise draw: no interval; per-bundle values in robust_phantom.json"
    if (OUT / "fig5.json").exists():
        out["fig5 example"] = json.loads((OUT / "fig5.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(out, indent=1))
    for term, t in out["terms"].items():
        print(term, {m: (v["mean"], v["95 % CI"]) for m, v in t["ECE"].items()})
        for q, c in t["coverage"].items():
            print("  ", q, {m: (c[m]["mean"], c[m]["95 % CI"]) for m in ("as stated", "card")})
    print(f"wrote {OUT / 'stats.json'}")


# ------------------------------------------------------------------------------------------- figures

def fig1():
    """The pipeline: one scan, 17 whole-pipeline runs, the ensemble, the card (or run-to-run only)."""
    plt = mpl()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(-1, 44); ax.axis("off")
    rnd = "round,pad=0.3,rounding_size=1.2"

    def box(x, y, w, h, title, sub=None, fc="#FFFFFF", ec=C["black"], stack=0, tc=C["black"], lw=1.0):
        for k in range(stack, 0, -1):
            ax.add_patch(FancyBboxPatch((x + 0.6 * k, y + 0.6 * k), w, h, boxstyle=rnd, fc=fc, ec=ec, lw=0.5, zorder=1))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=rnd, fc=fc, ec=ec, lw=lw, zorder=2))
        cy = y + h / 2
        if sub:
            ax.text(x + w / 2, cy + 1.9, title, ha="center", va="center", fontsize=7.5, weight="bold", color=tc, zorder=3)
            ax.text(x + w / 2, cy - 1.7, sub, ha="center", va="center", fontsize=6.5, color=tc, zorder=3, linespacing=1.2)
        else:
            ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=7.5, weight="bold", color=tc, zorder=3)

    def arrow(a, b, color=C["black"], text=None, tpos=None):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=1.0, color=color, zorder=4, shrinkA=0, shrinkB=0))
        if text:
            ax.text(*tpos, text, fontsize=7, ha="center", va="center", color=color, style="italic")

    # top row: the per-run pipeline, 17 times
    y1, h1 = 26, 10.5
    box(0.5, y1, 11, h1, "One scan", "DWI +\nreversed b0")
    ax.add_patch(FancyBboxPatch((14.5, y1 - 3), 85, h1 + 10, boxstyle="round,pad=0.2,rounding_size=1.5",
                                fc="#F2F7FB", ec=C["card"], lw=0.8, ls=(0, (3, 2)), zorder=0))
    ax.text(57, y1 + h1 + 4.3, "× 17: the scan and 16 bootstrap replicates, each through the whole pipeline",
            ha="center", va="center", fontsize=7.5, color=C["card"], weight="bold")
    xs = [(16, 19, "Bootstrap replicate", "wild bootstrap,\nsmooth signs (FWHM 4)"),
          (39, 17.5, "Field re-estimated", "susceptibility field\nfrom the b0 pair"),
          (60.5, 14, "UKF tracking", "b = 2000 shell"),
          (78.5, 16, "RapidParc", "42 named tracts\n+ Other")]
    for x, w, t, sub in xs:
        box(x, y1, w, h1, t, sub, stack=2)
    arrow((12.1, y1 + h1 / 2), (15.4, y1 + h1 / 2))
    for (x, w, *_), (x2, *_) in zip(xs[:-1], xs[1:]):
        arrow((x + w + 1.6, y1 + h1 / 2), (x2 - 0.4, y1 + h1 / 2))

    # bottom row: ensemble, card check, statements
    y2, h2 = 3, 12
    box(16, y2, 19, h2, "17-run ensemble", "votes (0–17) per seed,\nvoxel and point", fc="#F2F7FB", ec=C["card"])
    arrow((25.5, y1 - 3.2), (25.5, y2 + h2 + 0.5))
    dx, dy = 46, y2 + h2 / 2
    ax.add_patch(Polygon([[dx - 7.5, dy], [dx, dy + 7.4], [dx + 7.5, dy], [dx, dy - 7.4]], closed=True, fc="#FFFFFF",
                         ec=C["black"], lw=1.0, zorder=2))
    ax.text(dx, dy + 1.2, "Map for this", ha="center", va="center", fontsize=7, weight="bold", zorder=3)
    ax.text(dx, dy - 1.4, "protocol?", ha="center", va="center", fontsize=7, weight="bold", zorder=3)
    arrow((35.6, dy), (dx - 7.6, dy))
    yc = y2 + 7.2
    box(59, yc, 18, 8.6, "Recalibration map", "8 test-retest subjects\n(MASiVar cIII), held out", fc="#E6F0F8", ec=C["card"])
    box(80.3, yc, 19.2, 8.6, "Calibrated", "P(rescan, same session)\nP(rescan, other session)", fc="#E6F0F8", ec=C["card"])
    box(59, y2 - 2.6, 40, 5.4, "Run-to-run only: levels, not probabilities", fc="#FBEFE6", ec=C["stated"], tc=C["stated"])
    arrow((dx + 3.75, dy + 3.7), (58.4, yc + 4.3), text="yes", tpos=(52.5, yc + 6.6))
    arrow((dx + 3.75, dy - 3.7), (58.4, y2 + 0.1), color=C["stated"], text="no", tpos=(54.5, y2 - 0.6))
    arrow((77.6, yc + 4.3), (79.7, yc + 4.3))
    save(fig, "fig1_pipeline")
    plt.close(fig)


def reliability_panel(ax, rows, title, min_n=300, legend=True, thin=True, big=False):
    lw_main, lw_thin, ms = (3.0, 0.0, 0) if big else (1.8, 0.6, 3)
    ax.plot([0, 1], [0, 1], color=C["gray"], lw=0.8 if not big else 1.5, ls=":", zorder=1)
    methods = [("as stated", "stated", C["stated"], "As stated"), ("card", "card", C["card"], "With map")]
    if not big:
        methods.append(("PAT16 map", "pat16", C["pat16"], "PAT16 map"))
    for m, key, col, label in methods:
        xs, ys = [], []
        for r in rows.values():
            ok = (r["n"] >= min_n) & (r["k"] >= 1)
            x, y = r[key][ok], r["observed"][ok]
            xs.append(np.where(ok, r[key], np.nan)); ys.append(np.where(ok, r["observed"], np.nan))
            if thin and m != "PAT16 map":
                ax.plot(x, y, color=col, lw=lw_thin, alpha=0.45, zorder=2)
        X, Y = np.nanmean(xs, 0), np.nanmean(ys, 0)       # each subject weighted equally, per vote count
        e = np.mean([r["ece"][m] for r in rows.values()])
        lab = label if big else f"{label} (ECE {e:.3f})"
        ax.plot(X, Y, color=col, lw=lw_main, ls="--" if m == "PAT16 map" else "-", marker=None if big else "o",
                ms=ms, zorder=3, label=lab)
    ax.set_xlim(0.15, 1.0); ax.set_ylim(0.15, 1.0)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper left", frameon=False, handlelength=1.8)


def fig2(ho):
    plt = mpl()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.5))
    for ax, term in zip(axs, TERMS):
        rows = ho[term]
        reliability_panel(ax, rows, f"{term.capitalize().replace(' ', '-')} rescan ({len(rows)} subjects)")
        ax.set_xlabel("Stated P(rescan gives the ensemble's tract)")
    axs[0].set_ylabel("Observed rate, held-out subject")
    for ax, lab in zip(axs, "ab"):
        ax.text(-0.16, 1.04, lab, transform=ax.transAxes, fontsize=11, weight="bold")
    fig.tight_layout()
    save(fig, "fig2_streamline_calibration")
    plt.close(fig)


def fig3(ho):
    plt = mpl()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6))
    rng = np.random.default_rng(1)
    for ax, term in zip(axs, TERMS):
        rows = ho[term]
        ax.plot([0, 1], [0, 1], color=C["gray"], lw=0.8, ls=":", zorder=1)
        g = next(iter(rows.values()))["grid"]
        sel = (g >= 0.70) & (g <= 0.96)
        stated = np.mean([r["curve"] for r in rows.values()], 0)
        card = np.mean([r["card curve"] for r in rows.values()], 0)
        t = next(iter(rows.values()))["targets"]
        cov = {m: np.mean([[r["coverage"][f"{q:g}"][m] for q in TARGETS] for r in rows.values()], 0) for m in ("as stated", "card")}
        ax.plot(g[sel], stated[sel], color=C["stated"], lw=1.8, zorder=3,
                label="As stated (" + ", ".join(f"{c:.3f}" for c in cov["as stated"]) + ")")
        ax.plot(t, card, color=C["card"], lw=1.8, zorder=3,
                label="With map (" + ", ".join(f"{c:.3f}" for c in cov["card"]) + ")")
        for j, q in enumerate(TARGETS):
            for m, col, dx in (("as stated", C["stated"], -0.006), ("card", C["card"], 0.006)):
                y = [r["coverage"][f"{q:g}"][m] for r in rows.values()]
                x = q + dx + rng.uniform(-0.0025, 0.0025, len(y))
                ax.scatter(x, y, s=14, facecolor="white", edgecolor=col, lw=0.9, zorder=4)
        ax.set_xlim(0.70, 0.97); ax.set_ylim(0.70, 0.97)
        ax.set_aspect("equal")
        ax.set_xticks([0.7, 0.8, 0.9, 0.95]); ax.set_yticks([0.7, 0.8, 0.9, 0.95])
        ax.set_title(f"{term.capitalize().replace(' ', '-')} rescan ({len(rows)} subjects)")
        ax.set_xlabel("Stated envelope level (target coverage)")
        ax.legend(loc="upper left", frameon=False, title="coverage at 0.8, 0.9, 0.95", title_fontsize=7.5,
                  alignment="left", handlelength=1.6)
    axs[0].set_ylabel("Observed coverage of the rescan's tract")
    for ax, lab in zip(axs, "ab"):
        ax.text(-0.16, 1.04, lab, transform=ax.transAxes, fontsize=11, weight="bold")
    fig.tight_layout()
    save(fig, "fig3_envelope_coverage")
    plt.close(fig)


def phantom():
    return json.loads(PHANTOM_JSON.read_text())["envelopes, count-matched"]["pooled"]["labeled"]


def fig4():
    plt = mpl()
    em = json.loads(PHANTOM_JSON.read_text())["envelopes, count-matched"]
    pool, per = em["pooled"]["labeled"], em["per bundle"]
    lv = np.array([0.8, 0.9, 0.925, 0.95])
    runs = np.array([pool["runs"][f"envelope {x:g}"]["coverage"] for x in lv])
    free = np.array([pool["perfect ensemble"][f"envelope {x:g}"]["coverage"] for x in lv])
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    ax.plot([0, 1], [0, 1], color=C["gray"], lw=0.8, ls=":", zorder=1)
    rng = np.random.default_rng(2)
    for b in per.values():
        for x in lv:
            e = b["labeled"].get(f"envelope {x:g}")
            if not e:
                continue
            j = rng.uniform(-0.003, 0.003)
            ax.scatter(x - 0.004 + j, e["runs"]["coverage"], s=7, color=C["stated"], alpha=0.35, lw=0, zorder=2)
            ax.scatter(x + 0.004 + j, e["perfect ensemble"]["coverage"], s=7, color=C["biasfree"], alpha=0.35, lw=0, zorder=2)
    ax.plot(lv, free, color=C["biasfree"], lw=1.8, marker="o", ms=4, zorder=3, label="Bias-free ensemble\n(ground-truth subsamples)")
    ax.plot(lv, runs, color=C["stated"], lw=1.8, marker="o", ms=4, zorder=3, label="Pipeline, 17 runs\n(labeled presence)")
    for x, a, b in zip(lv, runs, free):
        if x in (0.8, 0.9, 0.95):
            ax.annotate("", xy=(x, a + 0.008), xytext=(x, b - 0.008),
                        arrowprops=dict(arrowstyle="<->", color=C["black"], lw=0.8, shrinkA=0, shrinkB=0))
            ax.text(x + 0.005, (a + b) / 2, f"−{100 * (b - a):.0f}\npts", fontsize=8, va="center", ha="left", linespacing=1.0)
    ax.set_xlim(0.77, 0.98); ax.set_ylim(0.25, 1.0)
    ax.set_xticks([0.8, 0.9, 0.95])
    ax.set_xlabel("Stated envelope level")
    ax.set_ylabel("Coverage of the count-matched truth")
    ax.set_title("ISMRM 2015 phantom, 14 bundles")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)
    fig.tight_layout()
    save(fig, "fig4_phantom_gap")
    plt.close(fig)


def example_presence():
    """cIIIs04's CST density presence per run (A and 16 replicates) and in the same-session rescan, on A's grid, as
    calibration_card.collect computes it; checked against the statements' vote histograms."""
    import nibabel as nib
    import calibration_card as CC
    from calibration_runs import Records
    from _masivar import scan_files
    from ensemble_grid_check import Grid, sample, RHO
    from tractline import ensemble as E
    R = Records(EXAMPLE)
    k = CC.TRACT_NAMES.index(TRACT)
    paths = R.ensemble("", 16)
    img = nib.load(scan_files(EXAMPLE, *R.pair.ref)["APP"][0][1])
    A0 = E.RunRecord.load(paths[0], logp=False)
    grid = Grid("dwi", img.affine, A0.shape)

    def pres(p):
        rec = E.RunRecord.load(p, logp=False, fibers=True)
        Pt, Ot = sample(rec.fibers)
        sel = CC.L[rec.cluster][Ot] == k
        return grid.density(grid.counts(Pt[sel], Ot[sel])).reshape(grid.shape) >= RHO

    votes = sum(pres(p).astype(np.int16) for p in paths)
    target = pres(R.path(f"{RESCAN}_onA"))
    z = np.load(R.dir / "statements_R17.npz")
    cnt = np.bincount(votes.ravel(), minlength=18)
    hit = np.bincount(votes[target], minlength=18)
    assert (cnt == z["e_cnt"][k]).all() and (hit == z[f"e_hit_{RESCAN}"][k]).all(), "presence differs from collect's"
    b0 = np.load(R.path("A_meanb0", ".npy"))
    return votes, target, b0, img.affine, grid.h, z["e_cnt"][k]


def fig5(ho):
    """cIIIs04's CST: the card's 80 % and 95 % envelopes (the held-out fold's same-session levels) on A's mean b0, with
    the same-session rescan's CST outline."""
    import calibration_card as CC
    plt = mpl()
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    votes, target, b0, aff, h, cnt = example_presence()
    row = ho["same session"][EXAMPLE]
    levels = {q: row["coverage"][f"{q:g}"]["card level"] for q in (0.8, 0.95)}
    cut = {q: CC.thresholds_from_counts(cnt, [lv])[0] for q, lv in levels.items()}
    env = {q: votes >= c for q, c in cut.items()}
    cov = {q: float((env[q] & target).sum() / target.sum()) for q in env}
    # slices: coronal through the most of the 95 % envelope; axial at the internal capsule's level (most voxels of the
    # 95 % envelope in the slab 25-50 % of the envelope's height)
    jy = int(np.argmax(env[0.95].sum((0, 2))))
    zs = np.flatnonzero(env[0.95].any((0, 1)))
    lo, hi = zs.min() + int(0.25 * (zs.max() - zs.min())), zs.min() + int(0.5 * (zs.max() - zs.min()))
    kz = lo + int(np.argmax(env[0.95].sum((0, 1))[lo:hi + 1]))
    # LAS voxel order: i -> left, j -> anterior, k -> superior. Shown in radiological convention (the subject's left
    # on the image's right): i increases to the right as stored.
    vmax = np.percentile(b0[b0 > 0], 99.5)
    cm95 = ListedColormap([C["sky"]]); cm80 = ListedColormap([C["card"]])
    views = [("Axial", lambda a: a[:, :, kz].T, (h[0], h[1])), ("Coronal", lambda a: a[:, jy, :].T, (h[0], h[2]))]
    crops = []
    for _, sl, (hx, hy) in views:                                    # crop each slice to the head
        B = sl(b0)
        on = B > 0.05 * vmax
        ys_, xs_ = np.flatnonzero(on.any(1)), np.flatnonzero(on.any(0))
        crops.append(((max(xs_.min() - 2, 0) * hx, min(xs_.max() + 3, B.shape[1]) * hx),
                      (max(ys_.min() - 2, 0) * hy, min(ys_.max() + 3, B.shape[0]) * hy)))
    ratios = [(x[1] - x[0]) / (y[1] - y[0]) for x, y in crops]       # equal panel heights
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 6.8 / (sum(ratios) + 0.15) + 0.75), gridspec_kw={"width_ratios": ratios})
    for ax, (name, sl, (hx, hy)), (xl, yl) in zip(axs, views, crops):
        B = sl(b0)
        ny, nx = B.shape
        ext = (0, nx * hx, 0, ny * hy)
        ax.imshow(B, cmap="gray", vmin=0, vmax=vmax, origin="lower", extent=ext, interpolation="bilinear")
        ax.imshow(np.ma.masked_where(~sl(env[0.95]), np.ones_like(B)), cmap=cm95, alpha=0.8, origin="lower", extent=ext, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~sl(env[0.8]), np.ones_like(B)), cmap=cm80, alpha=0.85, origin="lower", extent=ext, interpolation="nearest")
        T = sl(target).astype(float)
        xx = (np.arange(nx) + 0.5) * hx; yy = (np.arange(ny) + 0.5) * hy
        ax.contour(xx, yy, T, levels=[0.5], colors=[C["yellow"]], linewidths=1.3)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name)
        ax.text(xl[0] + 3, yl[1] - 4, "R", color="white", fontsize=10, weight="bold", va="top")
        ax.text(xl[1] - 3, yl[1] - 4, "L", color="white", fontsize=10, weight="bold", va="top", ha="right")
    handles = [Patch(fc=C["card"], alpha=0.85, label=f"80 % envelope (covers {cov[0.8]:.2f})"),
               Patch(fc=C["sky"], alpha=0.8, label=f"95 % envelope (covers {cov[0.95]:.2f})"),
               Line2D([], [], color=C["yellow"], lw=1.6, label="Same-session rescan's CST")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.4, columnspacing=1.2)
    fig.suptitle(f"{EXAMPLE}, corticospinal tract: recalibrated envelopes from one scan", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig5_cst_envelopes")
    plt.close(fig)
    info = {"subject": EXAMPLE, "tract": TRACT, "rescan": RESCAN, "term": "same session",
            "card's stated levels (held-out fold, without this subject)": {f"{q:g}": lv for q, lv in levels.items()},
            "vote cut (of 17)": {f"{q:g}": int(c) for q, c in cut.items()},
            "coverage of the rescan's CST (this tract alone; per-tract statements were not validated)": {f"{q:g}": round(c, 4) for q, c in cov.items()},
            "envelope size over the rescan's CST": {f"{q:g}": round(float(env[q].sum() / target.sum()), 3) for q in env},
            "slices (A's voxel indices)": {"axial k": int(kz), "coronal j": int(jy)}}
    (OUT / "fig5.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def preview(ho):
    """A simple, legible reliability diagram: same-session rescans, as stated against with the card."""
    plt = mpl()
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 18})
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    reliability_panel(ax, ho["same session"], None, legend=False, thin=False, big=True)
    ax.set_xticks([0.2, 0.6, 1.0]); ax.set_yticks([0.2, 0.6, 1.0])
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Observed on a rescan")
    ax.text(0.63, 0.40, "As stated", color=C["stated"], fontsize=20, weight="bold", ha="left")
    ax.text(0.22, 0.78, "With map", color=C["card"], fontsize=20, weight="bold", ha="left")
    ax.spines["left"].set_linewidth(1.5); ax.spines["bottom"].set_linewidth(1.5)
    fig.tight_layout()
    save(fig, "preview")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=("all", "fig1", "fig2", "fig3", "fig4", "fig5", "preview", "stats"))
    a = ap.parse_args().what
    if a in ("all", "fig1"):
        fig1()
    if a in ("all", "fig4"):
        fig4()
    if a in ("all", "fig2", "fig3", "fig5", "preview", "stats"):
        ho = held_out()
        if a in ("all", "fig2"):
            fig2(ho)
        if a in ("all", "fig3"):
            fig3(ho)
        if a in ("all", "preview"):
            preview(ho)
        if a in ("all", "fig5"):
            fig5(ho)
        if a in ("all", "stats"):
            stats(ho)
