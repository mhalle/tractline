"""Do the named streamlines run where their bundle should? FastSurfer's anatomy against the ensemble and the rescans
(NOTES 2026-10-09, "Endpoints against FastSurfer's anatomy").

    uv run --group dipy --group rankfield bench/ensemble_endpoints.py

Anatomy: FastSurfer (VINN, asegdkt: aseg + DKT cortex, lateralized; 2.5.4) run through haversack on Modal from each
subject's T1, restored to labels on the T1's grid (DATA/fastsurfer/<subject>_asegdkt.seg.nrrd; the ranked store beside
it). The T1 is registered rigidly to the diffusion data: PAT16 by tumor_on_dwi()'s cached transform (the T1 to the
corrected mean b0), cIIIs04 by bench/_register.py (FastSurfer's conformed T1 to A's corrected mean b0, A_meanb0.npy);
bench/endpoints_overlay.py draws the check.

Per streamline, in the T1:
  - each end's region: the nearest cerebral gray-matter label (cortex, deep gray, ventral DC, brainstem; not the
    cerebellar cortex, which basal temporal and occipital ends would otherwise reach across the tentorium) within
    NEAR_MM, else none; an end within EDGE_MM of the DWI's inferior face is "fov-edge" (the field of view cuts the
    brainstem);
  - each end's side: which side of the midsagittal plane it lies on (the plane midway between the two hemispheres'
    cortex), not its nearest label's hemisphere, since medial surfaces lie within NEAR_MM of each other;
  - which deep structures its path enters (thalamus, striatum), by side: cIIIs04 from the full streamlines (compact
    records), PAT16 from its 20-point paths (format 1 records), which can miss a short passage.
"Consistent" means the streamline fits its bundle's rule in RULES (a first pass from the bundles' usual definitions,
revised after an independent review, NOTES):
  - pair: one end in each of two region sets, on one side;
  - within: both ends in the union of the two sets, on one side (the superficial pairs, whose DKT lobe borders are
    uncertain where U-fibers sit);
  - callosal: both ends in the segment's cortex, on opposite sides;
  - via: the path enters a deep structure and one end is in the lobe, on the structure's side (the thalamic and
    striatal classes, which the ORG atlas names by the structures their fibers pass, not by where tracking stops);
  - cingulum: one end in cingulate, parahippocampal or entorhinal cortex or the hippocampus, the other there or in
    the medial cortex next to it, on one side.
An end in the insula leaves an AF, UF or IOFF streamline unscored (the insula borders all three). Not scored: the
capsules, corona radiata, PLIC, MdLF, SO (n = 28 in a run; it never entered the striatum), the cerebellar classes and
Other.

  - per bundle: the consistent share, in A's run, over A's 17 runs, and on each rescan;
  - cIIIs04, seeds labeled in all 17 runs and on a rescan: does a consistent A streamline keep its name on the rescan
    more often than an inconsistent one, pooled and within bundles (weighted by the bundle's scored streamlines); and
    the mode of the runs whose version is consistent against the mode of all runs, as a predictor of the rescan.
Writes results/ensemble_endpoints.json and DATA/fastsurfer/<subject>_t1_to_dwi.npz.
"""
import json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import nibabel as nib
import nrrd
from scipy.ndimage import distance_transform_edt
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import TRACT_NAMES, LUT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402

FS = DATA / "fastsurfer"
NEAR_MM = 4.0
EDGE_MM = 3.0

# DKT cortex (left 1000+, right 2000+: the code minus the hemisphere base); DKT has no frontal pole, temporal pole or
# banks of the STS
CTX = dict(caudalanteriorcingulate=2, caudalmiddlefrontal=3, cuneus=5, entorhinal=6, fusiform=7, inferiorparietal=8,
           inferiortemporal=9, isthmuscingulate=10, lateraloccipital=11, lateralorbitofrontal=12, lingual=13,
           medialorbitofrontal=14, middletemporal=15, parahippocampal=16, paracentral=17, parsopercularis=18,
           parsorbitalis=19, parstriangularis=20, pericalcarine=21, postcentral=22, posteriorcingulate=23, precentral=24,
           precuneus=25, rostralanteriorcingulate=26, rostralmiddlefrontal=27, superiorfrontal=28, superiorparietal=29,
           superiortemporal=30, supramarginal=31, transversetemporal=34, insula=35)
INSULA = CTX["insula"]
C = lambda *names: frozenset(CTX[n] for n in names)                           # noqa: E731
FRONTAL = C("caudalmiddlefrontal", "lateralorbitofrontal", "medialorbitofrontal", "parsopercularis", "parsorbitalis",
            "parstriangularis", "precentral", "rostralmiddlefrontal", "superiorfrontal", "paracentral")
ORBITO = C("lateralorbitofrontal", "medialorbitofrontal", "parsorbitalis", "rostralanteriorcingulate")
IFG = C("parsopercularis", "parstriangularis", "parsorbitalis", "precentral", "caudalmiddlefrontal")
TEMPORAL = C("superiortemporal", "middletemporal", "inferiortemporal", "transversetemporal", "fusiform", "entorhinal",
             "parahippocampal")
LATTEMP = C("superiortemporal", "middletemporal", "inferiortemporal", "transversetemporal")
OCCIPITAL = C("cuneus", "lateraloccipital", "lingual", "pericalcarine")
PARIETAL = C("inferiorparietal", "postcentral", "precuneus", "superiorparietal", "supramarginal", "paracentral")
MOTOR = C("precentral", "postcentral", "paracentral", "superiorfrontal", "caudalmiddlefrontal")
# aseg (left, right)
ASEG = dict(thalamus=(10, 49), caudate=(11, 50), putamen=(12, 51), pallidum=(13, 52), hippocampus=(17, 53),
            amygdala=(18, 54), accumbens=(26, 58), ventraldc=(28, 60))
BRAINSTEM = 16
SUB = lambda *names: frozenset("aseg:" + n for n in names)                    # noqa: E731
DEEP = {"thalamus": (ASEG["thalamus"],),
        "striatum": (ASEG["caudate"], ASEG["putamen"], ASEG["pallidum"], ASEG["accumbens"])}
CING_CORE = C("caudalanteriorcingulate", "isthmuscingulate", "posteriorcingulate", "rostralanteriorcingulate",
              "parahippocampal", "entorhinal") | SUB("hippocampus")
CING_NEXT = C("superiorfrontal", "paracentral", "precuneus", "medialorbitofrontal", "lingual", "fusiform")

pair = lambda a, b: ("pair", a, b)                                            # noqa: E731
within = lambda a, b: ("within", a, b)                                        # noqa: E731
callosal = lambda s: ("callosal", s, s)                                       # noqa: E731
via = lambda structure, lobe: ("via", structure, lobe)                        # noqa: E731
RULES = {
    "CST": pair(MOTOR, frozenset({"brainstem", "fov-edge"}) | SUB("ventraldc")),
    "AF": pair(IFG | C("rostralmiddlefrontal", "postcentral"), LATTEMP | C("supramarginal")),
    "SLF-I": pair(C("superiorfrontal", "paracentral", "caudalmiddlefrontal", "precentral"),
                  C("superiorparietal", "precuneus", "postcentral")),
    "SLF-II": pair(C("caudalmiddlefrontal", "rostralmiddlefrontal", "precentral", "parsopercularis", "parstriangularis"),
                   C("inferiorparietal", "supramarginal", "superiorparietal")),
    "SLF-III": pair(IFG | C("rostralmiddlefrontal"), C("supramarginal", "inferiorparietal", "postcentral")),
    "ILF": pair(OCCIPITAL | C("inferiorparietal", "fusiform"), TEMPORAL | SUB("amygdala", "hippocampus")),
    "IOFF": pair(OCCIPITAL | C("inferiorparietal", "superiorparietal", "fusiform"),
                 FRONTAL - C("precentral", "paracentral", "caudalmiddlefrontal")),
    "UF": pair(C("superiortemporal", "middletemporal", "inferiortemporal", "entorhinal") | SUB("amygdala"),
               ORBITO | IFG | C("superiorfrontal", "rostralmiddlefrontal")),
    "CB": ("cingulum", CING_CORE, CING_CORE | CING_NEXT),
    "TF": via("thalamus", FRONTAL), "TO": via("thalamus", OCCIPITAL), "TP": via("thalamus", PARIETAL),
    "TT": via("thalamus", TEMPORAL), "SF": via("striatum", FRONTAL), "SP": via("striatum", PARIETAL),
    "Sup-F": pair(FRONTAL, FRONTAL), "Sup-P": pair(PARIETAL, PARIETAL), "Sup-O": pair(OCCIPITAL, OCCIPITAL),
    "Sup-T": pair(TEMPORAL, TEMPORAL),
    "Sup-FP": within(FRONTAL, PARIETAL), "Sup-PO": within(PARIETAL, OCCIPITAL), "Sup-PT": within(PARIETAL, TEMPORAL),
    "Sup-OT": within(OCCIPITAL, TEMPORAL),
    "CC1": callosal(C("lateralorbitofrontal", "medialorbitofrontal", "parsorbitalis", "superiorfrontal", "rostralmiddlefrontal")),
    "CC2": callosal(C("lateralorbitofrontal", "medialorbitofrontal", "parsorbitalis", "superiorfrontal", "rostralmiddlefrontal")),
    "CC3": callosal(C("superiorfrontal", "caudalmiddlefrontal", "precentral", "parsopercularis")),
    "CC4": callosal(C("precentral", "paracentral", "caudalmiddlefrontal")),
    "CC5": callosal(C("postcentral", "paracentral", "superiorparietal")),
    "CC6": callosal(PARIETAL | C("superiortemporal", "transversetemporal")),
    "CC7": callosal(OCCIPITAL | C("precuneus", "fusiform", "inferiortemporal")),
}
INSULA_UNSCORED = {"AF", "UF", "IOFF"}


def region_of(label):
    """FreeSurfer label -> region key (DKT offset int, "aseg:<name>", "brainstem") or None."""
    if 1000 <= label < 3000:
        return label % 1000
    if label == BRAINSTEM:
        return "brainstem"
    for name, pr in ASEG.items():
        if label in pr:
            return "aseg:" + name
    return None


class Anatomy:
    """The label map on the T1 grid; each voxel's nearest cerebral gray label within NEAR_MM; the midsagittal x."""
    def __init__(self, subject):
        lab, h = nrrd.read(str(FS / f"{subject}_asegdkt.seg.nrrd"))
        lps = np.eye(4)
        lps[:3, :3], lps[:3, 3] = np.asarray(h["space directions"], float).T, h["space origin"]
        self.affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps                       # RAS, as the T1's NIfTI header
        self.inv = np.linalg.inv(self.affine)
        gray = (lab >= 1000) | np.isin(lab, [BRAINSTEM, *[x for p in ASEG.values() for x in p]])
        dist, idx = distance_transform_edt(~gray, sampling=np.sqrt((self.affine[:3, :3] ** 2).sum(0)), return_indices=True)
        self.near = np.where(dist <= NEAR_MM, lab[tuple(idx)], 0).astype(np.int32)
        self.lab = lab
        xs = []
        for lo, hi in ((1000, 2000), (2000, 3000)):
            ijk = np.argwhere((lab >= lo) & (lab < hi))[::50]
            xs.append(float((ijk @ self.affine[:3, :3].T + self.affine[:3, 3])[:, 0].mean()))
        self.mid_x = sum(xs) / 2

    def voxels(self, world):
        ijk = np.rint(world @ self.inv[:3, :3].T + self.inv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < self.lab.shape), 1)
        return ijk, ok

    def nearest(self, world):
        ijk, ok = self.voxels(world)
        out = np.zeros(len(world), np.int32)
        out[ok] = self.near[tuple(ijk[ok].T)]
        return out

    def raw(self, world):
        ijk, ok = self.voxels(world)
        out = np.zeros(len(world), np.int32)
        out[ok] = self.lab[tuple(ijk[ok].T)]
        return out


def ends_info(paths_or_fibers, anat, to_t1, dwi_affine):
    """Per streamline: end regions (a, b), sides (0 left / 1 right), and the deep structures its path enters by side."""
    if isinstance(paths_or_fibers, np.ndarray):
        pts = [p for p in paths_or_fibers]
    else:
        pts = paths_or_fibers
    lens = np.array([len(p) for p in pts])
    allp = np.concatenate(pts).astype(np.float64)
    owner = np.repeat(np.arange(len(pts)), lens)
    start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    e = np.concatenate([allp[start], allp[start + lens - 1]])
    vz = float(np.sqrt((dwi_affine[:3, 2] ** 2).sum()))
    k = (e @ np.linalg.inv(dwi_affine)[:3, :3].T + np.linalg.inv(dwi_affine)[:3, 3])[:, 2]
    edge = (k + 0.5) * vz <= EDGE_MM
    et1 = e @ to_t1[:3, :3].T + to_t1[:3, 3]
    lab = anat.nearest(et1)
    reg = np.array([region_of(int(x)) for x in lab], object)
    reg[edge & (lab == 0)] = "fov-edge"                   # no cerebral gray near an end the field of view cut
    side = (et1[:, 0] > anat.mid_x).astype(np.int8)
    n = len(pts)
    raw = anat.raw(allp @ to_t1[:3, :3].T + to_t1[:3, 3])
    enters = {}
    for name, pairs in DEEP.items():
        for s in (0, 1):
            hit = np.isin(raw, [p[s] for p in pairs])
            enters[(name, s)] = np.bincount(owner[hit], minlength=n) > 0
    return reg[:n], reg[n:], side[:n], side[n:], enters


def consistent(tract, ra, rb, sa, sb, enters):
    """(n,) consistent and scored, by RULES."""
    n = len(tract)
    ok = np.zeros(n, bool)
    scored = np.zeros(n, bool)
    for t in np.unique(tract):
        name = TRACT_NAMES[t]
        rule = RULES.get(name)
        if rule is None:
            continue
        kind, A, B = rule
        for r in np.flatnonzero(tract == t):
            a, b = ra[r], rb[r]
            if name in INSULA_UNSCORED and INSULA in (a, b):
                continue
            scored[r] = True
            if kind == "pair":
                ok[r] = (sa[r] == sb[r] or bool({a, b} & {"brainstem", "fov-edge"})) and ((a in A and b in B) or (a in B and b in A))
            elif kind == "within":
                ok[r] = sa[r] == sb[r] and a in (A | B) and b in (A | B)
            elif kind == "callosal":
                ok[r] = sa[r] != sb[r] and a in A and b in A
            elif kind == "cingulum":
                ok[r] = sa[r] == sb[r] and ((a in A and b in B) or (b in A and a in B))
            elif kind == "via":
                for end, s in ((a, sa[r]), (b, sb[r])):
                    if end in B and enters[(A, int(s))][r]:
                        ok[r] = True
    return ok, scored


def register_cIIIs04(d, aff):
    out = FS / "cIIIs04_t1_to_dwi.npz"
    if out.exists():
        z = np.load(out)
        return z["t1_to_dwi"], float(z["ncc"])
    conf = nib.load(FS / "ranked_cIIIs04_brain" / "conformed_1mm.nii.gz")
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(a, moving=np.asarray(conf.dataobj, np.float32), moving_affine=conf.affine,
                 static=np.load(d / "A_meanb0.npy").astype(np.float32), static_affine=aff)
        subprocess.run([sys.executable, str(HERE / "_register.py"), str(a), str(b)], check=True)
        z = np.load(b)
        T, ncc = z["to_static"], float(z["ncc"])
    np.savez(out, t1_to_dwi=T, ncc=ncc)
    return T, ncc


def score(rec, anat, to_t1, dwi_affine):
    geo = rec.fibers if rec.fibers is not None else rec.paths
    ra, rb, sa, sb, enters = ends_info(geo, anat, to_t1, dwi_affine)
    tract = LUT[rec.cluster]
    ok, scored = consistent(tract, ra, rb, sa, sb, enters)
    return tract, ok, scored


def per_bundle(tract, ok, scored):
    return {TRACT_NAMES[t]: [int((tract[scored] == t).sum()), round(float(ok[scored & (tract == t)].mean()), 3)]
            for t in np.unique(tract[scored])}


def pooled(tract, ok, scored, superficial=True):
    m = scored & (superficial | ~np.isin(tract, [TRACT_NAMES.index(n) for n in TRACT_NAMES if n.startswith("Sup-")]))
    return round(float(ok[m].mean()), 3)


def main():
    from _masivar import scan_files
    t0 = time.time()
    res = {"near_mm": NEAR_MM, "edge_mm": EDGE_MM, "rules": {k: v[0] for k, v in RULES.items()}}

    # cIIIs04: full streamlines
    pair_name = "cIIIs04-2.1"
    subject, (ses, scan), rescans, shells, _ = PAIRS[pair_name]
    d = DATA / "ensemble" / subject / "retest" / pair_name
    aff = nib.load(scan_files(subject, ses, scan, shells)["APP"][0][1]).affine
    T, ncc = register_cIIIs04(d, aff)
    to_t1 = np.linalg.inv(T)
    anat = Anatomy("cIIIs04")
    runs = [d / "A.npz"] + [d / "smooth" / f"r{r:03d}.npz" for r in range(1, 17)]
    keys, tr, okk, sc = [], [], [], []
    for p in runs:
        rec = E.RunRecord.load(p, logp=False, fibers=True)
        t, ok, s = score(rec, anat, to_t1, aff)
        keys.append(rec.keys); tr.append(t); okk.append(ok); sc.append(s)
        del rec
    print(f"cIIIs04 runs {time.time() - t0:.0f}s", flush=True)
    A = {"registration NCC (T1 to mean b0; negative: the contrasts run opposite)": round(ncc, 3),
         "midsagittal x in the T1 (mm)": round(anat.mid_x, 1)}
    A["A's run: per bundle [scored streamlines, consistent share]"] = per_bundle(tr[0], okk[0], sc[0])
    allt, allok, alls = np.concatenate(tr), np.concatenate(okk), np.concatenate(sc)
    A["17 runs pooled: consistent share, all scored / without the superficial classes"] = [pooled(allt, allok, alls),
                                                                                           pooled(allt, allok, alls, False)]
    A["17 runs pooled: per bundle"] = per_bundle(allt, allok, alls)
    common = keys[0]
    for k in keys[1:]:
        common = np.intersect1d(common, k)
    idx = [np.searchsorted(k, common) for k in keys]
    for k, i in zip(keys, idx):
        assert (k[i] == common).all()
    T17 = np.stack([tr[i][idx[i]] for i in range(17)], 1)
    OK17 = np.stack([okk[i][idx[i]] for i in range(17)], 1)
    S17 = np.stack([sc[i][idx[i]] for i in range(17)], 1)
    rows17 = np.repeat(np.arange(len(common)), 17)
    cnt = np.zeros((len(common), E.N_TRACTS), int)
    np.add.at(cnt, (rows17, T17.ravel()), 1)
    mode_all = cnt.argmax(1)
    w = (OK17 & S17).ravel()
    cntp = np.zeros_like(cnt)
    np.add.at(cntp, (rows17[w], T17.ravel()[w]), 1)
    has_p = cntp.sum(1) > 0
    mode_p = np.where(has_p, cntp.argmax(1), mode_all)
    resc = {}
    for tag in rescans:
        rec = E.RunRecord.load(d / f"{tag}_onA.npz", logp=False, fibers=True)
        t, ok, s = score(rec, anat, to_t1, aff)
        r = {"per bundle": per_bundle(t, ok, s), "consistent share (scored)": pooled(t, ok, s)}
        pos = np.minimum(np.searchsorted(rec.keys, common), len(rec.keys) - 1)
        hit = rec.keys[pos] == common
        rt = np.where(hit, t[pos], -1)
        del rec
        a_sc = S17[:, 0] & hit
        keep = rt == T17[:, 0]
        good, bad = a_sc & OK17[:, 0], a_sc & ~OK17[:, 0]
        r["A's streamline: keeps its name on the rescan, consistent / inconsistent (n, n)"] = [
            round(float(keep[good].mean()), 3), round(float(keep[bad].mean()), 3), int(good.sum()), int(bad.sum())]
        diffs, weights, neg = [], [], []
        for b in np.unique(T17[a_sc, 0]):
            g, h = good & (T17[:, 0] == b), bad & (T17[:, 0] == b)
            if g.sum() >= 20 and h.sum() >= 20:
                diffs.append(keep[g].mean() - keep[h].mean()); weights.append(g.sum() + h.sum())
                if diffs[-1] < 0:
                    neg.append(TRACT_NAMES[b])
        r["within bundles (>= 20 each side): keep-name difference, weighted / bundles / negative"] = [
            round(float(np.average(diffs, weights=weights)), 3), len(diffs), neg]
        r["inconsistent A streamlines that keep their name on the rescan"] = round(float(keep[bad].mean()), 3)
        r["rescan gives the mode, all seeds: consistent-runs mode / all-runs mode"] = [
            round(float((rt[hit] == mode_p[hit]).mean()), 3), round(float((rt[hit] == mode_all[hit]).mean()), 3)]
        resc[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k != "per bundle"}), flush=True)
    A["rescans"] = resc
    res["cIIIs04"] = A

    # PAT16: 20-point paths (format 1 records)
    z = np.load(DATA / "ensemble" / "PAT16" / "tumor_dwi.npz")
    to_t1 = np.linalg.inv(z["t1_to_dwi"])
    pat_aff = nib.load(DATA / "ds001226/sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi.nii.gz").affine
    anat = Anatomy("PAT16")
    first = DATA / "ensemble" / "PAT16" / "added-smooth"
    tr, okk, sc = [], [], []
    for r in range(17):
        rec = E.RunRecord.load(first / f"r{r:03d}.npz", logp=False)
        t, ok, s = score(rec, anat, to_t1, pat_aff)
        tr.append(t); okk.append(ok); sc.append(s)
    allt, allok, alls = np.concatenate(tr), np.concatenate(okk), np.concatenate(sc)
    res["PAT16"] = {"17 runs pooled: consistent share, all scored / without the superficial classes":
                    [pooled(allt, allok, alls), pooled(allt, allok, alls, False)],
                    "17 runs pooled: per bundle": per_bundle(allt, allok, alls),
                    "note": "20-point paths (passage through the thalamus or striatum can be missed); FastSurfer is "
                            "trained on healthy brains and its labels near the tumor are unchecked"}
    (HERE / "results/ensemble_endpoints.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(A["17 runs pooled: consistent share, all scored / without the superficial classes"]),
          json.dumps(res["PAT16"]["17 runs pooled: consistent share, all scored / without the superficial classes"]),
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
