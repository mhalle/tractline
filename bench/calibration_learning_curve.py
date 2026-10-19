"""A learning curve for the calibration card (the protocol-specific recalibration map): how many test-retest
subjects does a card need, and does it improve with more?

    uv run --group dipy --group rankfield python bench/calibration_learning_curve.py [run | figure | all]

Reads only the stored statements (calibration_card.py collect: DATA/ensemble/<subject>/card/<pair>/statements_R17.npz
of the eight MASiVar cohort III subjects); runs no pipeline and no tracking.

Per term (same session: 8 subjects; other session: the 5 with other-session rescans) and per held-out subject, the
card is refitted on training sets of k of the other subjects (k = 1..7 same session, 1..4 other session): every
combination when there are at most MAX_DRAWS, else MAX_DRAWS random ones (seed SEED; here every k has at most 35, so
all combinations are used). The fit is calibration_card.fit_term's streamline, point and envelope maps (fit_votes on
the subjects' pooled rescans, each subject weighted equally; the mean of the subjects' coverage curves), without the
shape summaries, which the held-out scores do not use; at k = all others it reproduces the stored card's held-out
values (checked). Scored on the held-out subject's own rescans of the term, as calibration_card.evaluate does:
  - streamline ECE and point ECE (calibration_card.ECE_BINS);
  - envelope coverage error at targets 0.8, 0.9, 0.95: the held-out coverage at the fold's stated level for the
    target (the widest outline when the training curve never reaches it) minus the target.
References: k = 0, as stated (the vote share; the envelope at stated level = target), and PAT16's double-bootstrap
vote table (results/ensemble_double_smooth.json; for the point statement it is applied to the point's vote share,
outside the statement it was fitted for). Spread: the 10th and 90th percentiles over (held-out subject, training
set) pairs; every held-out subject has the same number of training sets at each k, so each weighs equally.

Statement count against subject count: each training subject's statements thinned to FRACTIONS (10 %, 1 %) of its
streamline seeds and points (rows kept at random) and of its envelope voxels (per tract and vote count, the kept
voxels Binomial(count, f), and of those the rescan's presence voxels hypergeometric; each rescan thinned on its own,
so a subject's rescans do not share kept voxels exactly), THIN_REPS independent thinnings per fraction. The held-out
subject is scored on all of its statements.

Between-subject variance: per term and statement, each subject's own observed rate per vote count (and envelope
coverage at stated levels); the between-subject variance, the mean within-subject binomial variance (seeds and
points treated as independent, which they are not: a lower bound on the within-subject noise), and their
difference tau^2 (method of moments). A k-subject map's error about the population is about tau / sqrt(k); a new
subject deviates from the population by tau whatever k is, so the held-out error scales as tau * sqrt(1 + 1/k).

Writes results/calibration_learning_curve.json and results/ismrm2027/learning_curve.{png,pdf}.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_card as CC  # noqa: E402

OUT = HERE / "results/calibration_learning_curve.json"
CARD_JSON = HERE / "results/calibration_card_masivar_cIII.json"
TERMS = CC.TERMS
TARGETS = (0.8, 0.9, 0.95)
MAX_DRAWS, SEED = 50, 20261019
FRACTIONS = (1.0, 0.1, 0.01)
THIN_REPS = 5
KEY_K = (9, 13, 17)                        # vote counts reported in the variance table


# ----------------------------------------------------------------------------------------------- per-subject data

class Data:
    """A subject's statements for one term, reduced to what the maps and the held-out scores use."""
    def __init__(self, subj, term, s=None, p=None, rows=None):
        self.name, self.R, self.term = subj.name, subj.R, term
        self.s = s if s is not None else subj.streamline(term)
        self.p = p if p is not None else subj.point(term)
        rows = rows if rows is not None else [(c, h) for c, h, _, _ in subj.envelopes(term)]
        self.rows = rows
        self.curve = np.mean([CC.coverage_curve(c, h) for c, h in rows], 0)

    def widest(self):
        """Coverage of the widest outline (any run), as calibration_card.term_shapes gives it."""
        _, cuts = CC.shape_cuts(self.rows[0][0])
        return float(np.mean([CC.kept(h, cuts["widest"]).sum() / max(h.sum(), 1) for _, h in self.rows]))


def thin(d, subj, f, rng):
    """d with f of its streamline seeds, points and envelope voxels kept (at random)."""
    if f >= 1:
        return d
    def keep(vh):
        m = rng.random(len(vh[0])) < f
        return vh[0][m], vh[1][m]
    rows = []
    for c, h in d.rows:
        c2 = rng.binomial(c, f)
        h2 = np.zeros_like(h)
        ok = c > 0
        h2[ok] = rng.hypergeometric(h[ok], c[ok] - h[ok], c2[ok])
        rows.append((c2, h2))
    return Data(subj, d.term, keep(d.s), keep(d.p), rows)


def fit(train):
    R = train[0].R
    return {"streamline": CC.fit_votes([d.s for d in train], R), "point": CC.fit_votes([d.p for d in train], R),
            "curve": np.mean([d.curve for d in train], 0)}


def at(curve, lv):
    return float(curve[np.flatnonzero(CC.GRID == round(lv, 3))[0]])


def score(card, test, widest, pat):
    """The held-out subject's scores with the card (or as stated / PAT16's map when card is a string)."""
    R = test.R
    (vs, hs), (vp, hp) = test.s, test.p
    if card == "as stated":
        ps, pp = vs / R, vp / R
        cov = {q: at(test.curve, q) for q in TARGETS}
    elif card == "PAT16 map":
        ps, pp = pat(vs / R), pat(vp / R)
        cov = None
    else:
        ps, pp = CC.apply_votes(card["streamline"], vs, R), CC.apply_votes(card["point"], vp, R)
        cov = {}
        for q in TARGETS:
            lv = CC.level_for(card["curve"], q)[0]
            cov[q] = at(test.curve, lv) if lv < 1 else widest
    out = {"streamline ECE": CC.ece(ps, hs)["ece"], "point ECE": CC.ece(pp, hp)["ece"]}
    if cov:
        out.update({f"coverage error at {q:g}": cov[q] - q for q in TARGETS})
    return out


def summarize(rows):
    """rows: list of score dicts -> per metric mean, 10th and 90th percentile (and mean |x| for coverage error)."""
    out = {}
    for m in rows[0]:
        x = np.array([r[m] for r in rows], float)
        out[m] = {"mean": round(float(x.mean()), 4), "p10": round(float(np.percentile(x, 10)), 4),
                  "p90": round(float(np.percentile(x, 90)), 4)}
        if m.startswith("coverage"):
            out[m]["mean |error|"] = round(float(np.abs(x).mean()), 4)
    out["n"] = len(rows)
    return out


def draws(others, k, rng):
    combos = list(itertools.combinations(range(len(others)), k))
    if len(combos) > MAX_DRAWS:
        combos = [combos[i] for i in sorted(rng.choice(len(combos), MAX_DRAWS, replace=False))]
    return combos


# ----------------------------------------------------------------------------------------------- variance

def subject_rates(groups, R):
    """Per subject the observed rate and count per vote count (R + 1,)."""
    rates, ns = [], []
    for v, h in groups:
        n = np.bincount(v, minlength=R + 1)
        rates.append(np.bincount(v, h.astype(float), minlength=R + 1) / np.maximum(n, 1))
        ns.append(n)
    return np.array(rates), np.array(ns)


def variance(data, R):
    out = {}
    for stmt in ("streamline", "point"):
        rate, n = subject_rates([getattr(d, stmt[0]) for d in data], R)
        ok = (n >= 30).all(0) & (np.arange(R + 1) >= 1)
        between = rate.var(0, ddof=1)
        within = (rate * (1 - rate) / np.maximum(n, 1)).mean(0)
        tau2 = np.maximum(between - within, 0)
        share = n.sum(0) / n.sum()
        w = np.where(ok, share, 0); w = w / w.sum()
        rows = {int(k): {"mean observed": round(float(rate[:, k].mean()), 4), "between-subject SD": round(float(np.sqrt(between[k])), 4),
                         "within-subject binomial SE": round(float(np.sqrt(within[k])), 4), "tau": round(float(np.sqrt(tau2[k])), 4),
                         "per subject": {d.name: round(float(rate[i, k]), 4) for i, d in enumerate(data)}} for k in KEY_K if ok[k]}
        tau = float(np.sqrt((w * tau2).sum()))
        out[stmt] = {"per vote count": rows,
                     "tau, statement-weighted over vote counts (>= 30 statements in every subject)": round(tau, 4),
                     "within-subject binomial SE, statement-weighted": round(float(np.sqrt((w * within).sum())), 4),
                     "between-subject SD, statement-weighted": round(float(np.sqrt((w * between).sum())), 4)}
    env = {}
    for lv in TARGETS:
        x = np.array([at(d.curve, lv) for d in data])
        env[f"{lv:g}"] = {"mean coverage at stated level": round(float(x.mean()), 4), "between-subject SD": round(float(x.std(ddof=1)), 4),
                          "per subject": {d.name: round(float(v), 4) for d, v in zip(data, x)}}
    lvl = {f"{q:g}": {d.name: CC.level_for(d.curve, q)[0] for d in data} for q in TARGETS}
    out["envelope"] = {"coverage at stated level": env, "the subject's own stated level for target": lvl,
                       "note": "within-subject voxel noise not separated: voxels are spatially correlated"}
    return out


# ----------------------------------------------------------------------------------------------- run

def run():
    from calibration_runs import CARD, Records
    subjects = [CC.Subject(Records(n).dir / "statements_R17.npz") for n in CARD]
    pat = CC.pat16_table()
    stored = json.loads(CARD_JSON.read_text())["held-out evaluation (leave one subject out)"]["per subject"]
    result = {"what": __doc__.split("\n\n")[0], "method": __doc__.split("\n\n", 2)[2].split("\n\nWrites")[0],
              "source": "statements_R17.npz of the eight MASiVar cohort III subjects (calibration_card.py collect)",
              "seed": SEED, "max draws": MAX_DRAWS, "fractions": list(FRACTIONS), "thinning replicates": THIN_REPS,
              "terms": {}}
    for ti, term in enumerate(TERMS):
        subj = [s for s in subjects if s.tags(term)]
        data = [Data(s, term) for s in subj]
        widest = [d.widest() for d in data]
        R = subj[0].R
        kmax = len(subj) - 1
        rng = np.random.default_rng([SEED, ti])
        refs = {"as stated (k = 0)": summarize([score("as stated", d, w, pat) for d, w in zip(data, widest)]),
                "PAT16 map": summarize([score("PAT16 map", d, w, pat) for d, w in zip(data, widest)])}
        refs["PAT16 map"]["note"] = "PAT16's streamline vote table; on the point statement applied outside its fit"
        combos = {i: {k: draws([j for j in range(len(subj)) if j != i], k, rng) for k in range(1, kmax + 1)} for i in range(len(subj))}
        curve = {}
        for f in FRACTIONS:
            reps = 1 if f >= 1 else THIN_REPS
            pools = []
            for r in range(reps):
                trng = np.random.default_rng([SEED, ti, int(round(f * 1000)), r])
                pools.append([thin(d, s, f, trng) for d, s in zip(data, subj)])
            curve[f"{f:g}"] = {}
            for k in range(1, kmax + 1):
                rows = []
                for i, test in enumerate(data):
                    others = [j for j in range(len(subj)) if j != i]
                    for c in combos[i][k]:
                        for pool in pools:
                            card = fit([pool[others[j]] for j in c])
                            sc = score(card, test, widest[i], pat)
                            rows.append(sc)
                            if f >= 1 and k == kmax:                       # the stored card's held-out values
                                st = stored[test.name][term]
                                want = [st["streamline"]["card"]["ece"], st["point"]["card"]["ece"]] + \
                                       [st["envelope"][f"{q:g}"]["card: coverage"] - q for q in TARGETS]
                                got = [sc["streamline ECE"], sc["point ECE"]] + [sc[f"coverage error at {q:g}"] for q in TARGETS]
                                assert np.allclose(got, want, atol=6e-5), (term, test.name, got, want)
                curve[f"{f:g}"][str(k)] = {**summarize(rows), "training sets per held-out subject": len(combos[0][k]) * reps}
                print(term, f"f={f:g}", f"k={k}", {m: v["mean"] for m, v in curve[f"{f:g}"][str(k)].items() if isinstance(v, dict)}, flush=True)
        result["terms"][term] = {
            "subjects": [s.name for s in subj], "rescans": {s.name: s.tags(term) for s in subj},
            "statements per subject": {d.name: {"streamline seeds": int(len(d.s[0])), "points": int(len(d.p[0])),
                                                "envelope presence voxels (rescans summed)": int(sum(h.sum() for _, h in d.rows))} for d in data},
            "references": refs, "learning curve (fraction of each training subject's statements -> k)": curve,
            "between-subject variance": variance(data, R)}
    print("k = all others reproduces the stored card's held-out values", flush=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT}")
    return result


# ----------------------------------------------------------------------------------------------- figure

def figure(res):
    from ismrm2027_figures import mpl, save, C
    plt = mpl()
    tc = {0.8: "#009E73", 0.9: C["card"], 0.95: "#CC79A7"}               # Okabe-Ito green, blue, reddish purple
    fc = {"1": (C["card"], "-", "all statements"), "0.1": (C["sky"], "--", "10 % of statements"),
          "0.01": (C["gray"], ":", "1 % of statements")}
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4), sharex=True)
    for col, term in enumerate(TERMS):
        t = res["terms"][term]
        lc = t["learning curve (fraction of each training subject's statements -> k)"]
        refs = t["references"]
        kmax = max(int(k) for k in lc["1"])
        hline = lambda ax, y, **kw: ax.hlines(y, -0.4, kmax + 0.3, **kw)  # noqa: E731
        for row, m in enumerate(("streamline ECE", "point ECE")):
            ax = axes[row, col]
            st = refs["as stated (k = 0)"][m]
            hline(ax, st["mean"], color=C["stated"], lw=1.0, ls="--", zorder=1, label="as stated")
            ax.errorbar([0], [st["mean"]], yerr=[[st["mean"] - st["p10"]], [st["p90"] - st["mean"]]], fmt="o", ms=4,
                        color=C["stated"], capsize=2, lw=1, zorder=3)
            pt = refs["PAT16 map"][m]
            hline(ax, pt["mean"], color=C["pat16"], lw=1.0, ls="-.", zorder=1,
                       label="PAT16 map")
            for f, (colr, ls, lab) in fc.items():
                ks = sorted(int(k) for k in lc[f])
                y = np.array([lc[f][str(k)][m]["mean"] for k in ks])
                lo = np.array([lc[f][str(k)][m]["p10"] for k in ks]); hi = np.array([lc[f][str(k)][m]["p90"] for k in ks])
                if f == "1":
                    ax.fill_between(ks, lo, hi, color=colr, alpha=0.18, lw=0, zorder=1)
                ax.plot(ks, y, color=colr, ls=ls, lw=1.8 if f == "1" else 1.2, marker="o" if f == "1" else None, ms=3.5,
                        zorder=4 if f == "1" else 2, label=f"card, {lab}")
            top = max([st["p90"], pt["mean"]] + [lc[f][k][m]["mean"] for f in lc for k in lc[f]])
            ax.set_ylim(0, top * 1.08)
            ax.set_ylabel(m)
            if row == 0:
                ax.set_title(f"{term} ({len(t['subjects'])} subjects)")
            if col == 0 and row == 0:
                ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.74), fontsize=7, handlelength=1.8)
            if col == 0 and row == 1:
                ax.text(kmax + 0.3, pt["mean"] + 0.0015, "PAT16 map, applied off its fit", ha="right", va="bottom", fontsize=6.5, color=C["gray"])
        ax = axes[2, col]
        hline(ax, 0, color=C["black"], lw=0.6, zorder=1)
        f = lc["1"]
        ks = sorted(int(k) for k in f)
        for qi, q in enumerate(TARGETS):
            m = f"coverage error at {q:g}"
            dx = (qi - 1) * 0.13
            st = refs["as stated (k = 0)"][m]
            y = np.array([f[str(k)][m]["mean"] for k in ks])
            lo = np.array([f[str(k)][m]["p10"] for k in ks]); hi = np.array([f[str(k)][m]["p90"] for k in ks])
            ax.errorbar(np.array(ks) + dx, y, yerr=[y - lo, hi - y], color=tc[q], lw=1.5, marker="o", ms=3.5, capsize=1.5,
                        elinewidth=0.8, zorder=4, label=f"card, target {q:g}")
            hline(ax, st["mean"], color=tc[q], lw=0.8, ls="--", zorder=1)
            ax.errorbar([dx], [st["mean"]], yerr=[[st["mean"] - st["p10"]], [st["p90"] - st["mean"]]], fmt="s", ms=4,
                        color=tc[q], capsize=2, lw=1, zorder=3)
        ax.set_ylabel("envelope coverage − target")
        ax.set_xlabel("training subjects, k")
        if col == 1:
            ax.plot([], [], color=C["gray"], lw=0.8, ls="--", label="as stated (k = 0, dashed)")
            ax.legend(frameon=False, loc="center right", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks(range(0, 8))
        ax.set_xticklabels(["0\nas stated"] + [str(k) for k in range(1, 8)])
        ax.set_xlim(-0.4, 7.4)
        ax.grid(axis="y", color="#E5E5E5", lw=0.5, zorder=0)
    fig.text(0.5, -0.005, "Held out one subject at a time; card fitted on every set of k of the others. Lines: mean; "
             "bands and bars: 10th–90th percentile over held-out subjects and training sets.", ha="center", fontsize=7, color=C["gray"])
    fig.tight_layout(h_pad=0.8)
    save(fig, "learning_curve")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", nargs="?", default="all", choices=("all", "run", "figure"))
    a = ap.parse_args()
    res = run() if a.what in ("all", "run") else json.loads(OUT.read_text())
    if a.what in ("all", "figure"):
        figure(res)


if __name__ == "__main__":
    main()
