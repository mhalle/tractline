"""Do the named streamlines end where their bundle should? FastSurfer's anatomy against the ensemble and the rescans
(NOTES 2026-10-09, "Endpoints against FastSurfer's anatomy").

    uv run --group dipy --group rankfield bench/ensemble_endpoints.py

Anatomy: FastSurfer (VINN, asegdkt: aseg + DKT cortex, lateralized; 2.5.4) run through haversack on Modal from each
subject's T1, restored to labels on the T1's grid (DATA/fastsurfer/<subject>_asegdkt.seg.nrrd; the ranked store beside
it). The T1 is registered rigidly to the diffusion data: PAT16 by tumor_on_dwi()'s cached transform (the T1 to the
corrected mean b0), cIIIs04 by bench/_register.py (FastSurfer's conformed T1 to A's corrected mean b0, A_meanb0.npy).

Each streamline's two ends (the first and last points of its path, in the DWI's world mm) are carried into the T1 and
given the nearest gray-matter label (cortex, subcortical gray, brainstem, cerebellar cortex) within NEAR_MM; farther
is "none". A named streamline is plausible when its ends fit its bundle's rule in ENDS: one end in each of two region
sets, in one hemisphere (association, projection, superficial), or both ends cortical in opposite hemispheres (the
corpus callosum). The cingulum's ends may be in either hemisphere: the medial surfaces lie within NEAR_MM of each
other across the midline. The rules are a first pass from the bundles' usual definitions, written here; bundles without one
(capsules, corona radiata, PLIC, MdLF, cerebellar) are not scored. Other is not scored.

  - per bundle: the share of its streamlines with plausible ends, in A's run, over A's 17 runs, and on each rescan;
  - cIIIs04, seeds labeled in all 17 runs and on a rescan: does a plausible A streamline keep its name on the rescan
    more often than an implausible one; and does the mode of the runs whose version has plausible ends (where any
    has) predict the rescan's name better than the mode of all runs.
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
OTHER = len(TRACT_NAMES) - 1

# DKT cortex (left 1000+, right 2000+: the code minus the hemisphere base) and aseg structures, by region
CTX = dict(caudalanteriorcingulate=2, caudalmiddlefrontal=3, cuneus=5, entorhinal=6, fusiform=7, inferiorparietal=8,
           inferiortemporal=9, isthmuscingulate=10, lateraloccipital=11, lateralorbitofrontal=12, lingual=13,
           medialorbitofrontal=14, middletemporal=15, parahippocampal=16, paracentral=17, parsopercularis=18,
           parsorbitalis=19, parstriangularis=20, pericalcarine=21, postcentral=22, posteriorcingulate=23, precentral=24,
           precuneus=25, rostralanteriorcingulate=26, rostralmiddlefrontal=27, superiorfrontal=28, superiorparietal=29,
           superiortemporal=30, supramarginal=31, transversetemporal=34, insula=35)
C = lambda *names: {CTX[n] for n in names}                                    # noqa: E731
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
CINGULATE = C("caudalanteriorcingulate", "isthmuscingulate", "posteriorcingulate", "rostralanteriorcingulate",
              "precuneus", "superiorfrontal", "parahippocampal", "entorhinal", "medialorbitofrontal")
# aseg (left, right): thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens, ventral DC
ASEG = dict(thalamus=(10, 49), caudate=(11, 50), putamen=(12, 51), pallidum=(13, 52), hippocampus=(17, 53),
            amygdala=(18, 54), accumbens=(26, 58), ventraldc=(28, 60), cerebellum=(8, 47))
BRAINSTEM = 16
SUB = lambda *names: {"aseg:" + n for n in names}                             # noqa: E731
STRIATUM = SUB("caudate", "putamen", "pallidum", "accumbens")

ENDS = {      # bundle: (end set A, end set B), one hemisphere; "callosal" for the corpus callosum
    "CST": (MOTOR, {"brainstem"} | SUB("ventraldc")),
    "AF": (IFG, LATTEMP | C("supramarginal")),
    "SLF-I": (C("superiorfrontal", "paracentral", "caudalmiddlefrontal", "precentral"), C("superiorparietal", "precuneus")),
    "SLF-II": (C("caudalmiddlefrontal", "rostralmiddlefrontal", "precentral"), C("inferiorparietal", "supramarginal", "superiorparietal")),
    "SLF-III": (IFG | C("rostralmiddlefrontal"), C("supramarginal", "inferiorparietal", "postcentral")),
    "ILF": (OCCIPITAL | C("inferiorparietal"), TEMPORAL | SUB("amygdala", "hippocampus")),
    "IOFF": (OCCIPITAL | C("inferiorparietal", "superiorparietal", "fusiform"), FRONTAL),
    "UF": (C("superiortemporal", "middletemporal", "inferiortemporal", "entorhinal") | SUB("amygdala"), ORBITO | IFG),
    "CB": (CINGULATE | SUB("hippocampus"), CINGULATE | SUB("hippocampus"), "either hemisphere"),   # the medial surfaces
                                     # lie within NEAR_MM of each other, so an end's nearest label may be across
    "TF": (SUB("thalamus"), FRONTAL), "TO": (SUB("thalamus"), OCCIPITAL), "TP": (SUB("thalamus"), PARIETAL),
    "TT": (SUB("thalamus"), TEMPORAL), "SF": (STRIATUM, FRONTAL), "SO": (STRIATUM, OCCIPITAL), "SP": (STRIATUM, PARIETAL),
    "Sup-F": (FRONTAL, FRONTAL), "Sup-P": (PARIETAL, PARIETAL), "Sup-O": (OCCIPITAL, OCCIPITAL), "Sup-T": (TEMPORAL, TEMPORAL),
    "Sup-FP": (FRONTAL, PARIETAL), "Sup-PO": (PARIETAL, OCCIPITAL), "Sup-PT": (PARIETAL, TEMPORAL), "Sup-OT": (OCCIPITAL, TEMPORAL),
    **{f"CC{i}": "callosal" for i in range(1, 8)},
}


def region_of(label):
    """FreeSurfer label -> (region key, hemisphere 0 left / 1 right / None)."""
    if 1000 <= label < 2000:
        return label - 1000, 0
    if 2000 <= label < 3000:
        return label - 2000, 1
    if label == BRAINSTEM:
        return "brainstem", None
    for name, (lh, rh) in ASEG.items():
        if label in (lh, rh):
            return "aseg:" + name, 0 if label == lh else 1
    return None, None


class Anatomy:
    """The label map on the T1 grid, with each voxel's nearest gray-matter label (within NEAR_MM)."""
    def __init__(self, subject):
        lab, h = nrrd.read(str(FS / f"{subject}_asegdkt.seg.nrrd"))
        lps = np.eye(4)
        lps[:3, :3], lps[:3, 3] = np.asarray(h["space directions"], float).T, h["space origin"]
        self.affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps                       # RAS, as the T1's NIfTI header
        gray = (lab >= 1000) | np.isin(lab, [BRAINSTEM, *[x for p in ASEG.values() for x in p]])
        dist, idx = distance_transform_edt(~gray, sampling=np.sqrt((self.affine[:3, :3] ** 2).sum(0)), return_indices=True)
        self.near = np.where(dist <= NEAR_MM, lab[tuple(idx)], 0).astype(np.int32)
        self.lab = lab

    def label(self, world_t1):
        ijk = np.rint((world_t1 - self.affine[:3, 3]) @ np.linalg.inv(self.affine[:3, :3]).T).astype(int)
        ok = np.all((ijk >= 0) & (ijk < self.near.shape), 1)
        out = np.zeros(len(ijk), np.int32)
        out[ok] = self.near[tuple(ijk[ok].T)]
        return out


def plausible(tract, la, lb):
    """(n,) tract ids and end labels -> bool (n,), and scored (n,) (a rule exists)."""
    ok = np.zeros(len(tract), bool)
    scored = np.zeros(len(tract), bool)
    regs = [region_of(int(x)) for x in np.unique(np.concatenate([la, lb]))]
    look = dict(zip(np.unique(np.concatenate([la, lb])).tolist(), regs))
    for t in np.unique(tract):
        rule = ENDS.get(TRACT_NAMES[t])
        if rule is None:
            continue
        rows = np.flatnonzero(tract == t)
        scored[rows] = True
        for r in rows:
            (ra, ha), (rb, hb) = look[int(la[r])], look[int(lb[r])]
            if rule == "callosal":
                ok[r] = isinstance(ra, int) and isinstance(rb, int) and ha != hb
                continue
            A, B = rule[:2]
            same = len(rule) > 2 or ha is None or hb is None or ha == hb
            ok[r] = same and ((ra in A and rb in B) or (ra in B and rb in A))
    return ok, scored


def ends(rec, to_t1):
    p = rec.paths
    e = np.concatenate([p[:, 0], p[:, -1]])
    w = e @ to_t1[:3, :3].T + to_t1[:3, 3]
    return w[: len(p)], w[len(p):]


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


def score(rec, anat, to_t1):
    a, b = ends(rec, to_t1)
    la, lb = anat.label(a), anat.label(b)
    tract = LUT[rec.cluster]
    ok, scored = plausible(tract, la, lb)
    return tract, ok, scored, la, lb


def per_bundle(tract, ok, scored):
    out = {}
    for t in np.unique(tract[scored]):
        m = tract == t
        out[TRACT_NAMES[t]] = [int(m.sum()), round(float(ok[m].mean()), 3)]
    return out


def main():
    from _masivar import scan_files
    t0 = time.time()
    res = {"near_mm": NEAR_MM, "rules": {k: ("callosal" if v == "callosal" else "two region sets") for k, v in ENDS.items()}}

    # cIIIs04
    pair = "cIIIs04-2.1"
    subject, (ses, scan), rescans, shells, _ = PAIRS[pair]
    d = DATA / "ensemble" / subject / "retest" / pair
    aff = nib.load(scan_files(subject, ses, scan, shells)["APP"][0][1]).affine
    T, ncc = register_cIIIs04(d, aff)
    to_t1 = np.linalg.inv(T)
    anat = Anatomy("cIIIs04")
    runs = [d / "A.npz"] + [d / "smooth" / f"r{r:03d}.npz" for r in range(1, 17)]
    keys, tr, okk, sc = [], [], [], []
    for p in runs:
        rec = E.RunRecord.load(p, logp=False)
        t, ok, s, _, _ = score(rec, anat, to_t1)
        keys.append(rec.keys); tr.append(t); okk.append(ok); sc.append(s)
    A = {"registration NCC (T1 to mean b0; negative: the contrasts run opposite)": round(ncc, 3)}
    A["A's run: per bundle [streamlines, plausible share]"] = per_bundle(tr[0], okk[0], sc[0])
    allt, allok, alls = np.concatenate(tr), np.concatenate(okk), np.concatenate(sc)
    A["17 runs pooled: plausible share of scored named streamlines"] = round(float(allok[alls].mean()), 3)
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
    cnt = np.zeros((len(common), E.N_TRACTS), int)
    np.add.at(cnt, (np.repeat(np.arange(len(common)), 17), T17.ravel()), 1)
    mode_all = cnt.argmax(1)
    cntp = np.zeros_like(cnt)
    w = OK17 & S17
    np.add.at(cntp, (np.repeat(np.arange(len(common)), 17)[w.ravel()], T17.ravel()[w.ravel()]), 1)
    has_p = cntp.sum(1) > 0
    mode_p = np.where(has_p, cntp.argmax(1), mode_all)
    resc = {}
    for tag in rescans:
        rec = E.RunRecord.load(d / f"{tag}_onA.npz", logp=False)
        t, ok, s, _, _ = score(rec, anat, to_t1)
        r = {"per bundle": per_bundle(t, ok, s), "plausible share (scored)": round(float(ok[s].mean()), 3)}
        pos = np.searchsorted(rec.keys, common)
        pos = np.minimum(pos, len(rec.keys) - 1)
        hit = rec.keys[pos] == common
        rt = np.where(hit, t[pos], -1)
        a_named = S17[:, 0] & hit
        keep = rt == T17[:, 0]
        r["A's streamline scored: keeps its name on the rescan, plausible / implausible ends"] = [
            round(float(keep[a_named & OK17[:, 0]].mean()), 3), round(float(keep[a_named & ~OK17[:, 0]].mean()), 3),
            int((a_named & OK17[:, 0]).sum()), int((a_named & ~OK17[:, 0]).sum())]
        sel = hit & has_p & (mode_p != mode_all)
        r["where the plausible runs' mode differs from all runs' mode: rescan gives plausible mode / all mode / n"] = [
            round(float((rt[sel] == mode_p[sel]).mean()), 3), round(float((rt[sel] == mode_all[sel]).mean()), 3), int(sel.sum())]
        r["rescan gives the mode, all seeds: plausible-runs mode / all-runs mode"] = [
            round(float((rt[hit] == mode_p[hit]).mean()), 3), round(float((rt[hit] == mode_all[hit]).mean()), 3)]
        resc[tag] = r
        print(tag, json.dumps({k: v for k, v in r.items() if k != "per bundle"}), flush=True)
    A["rescans"] = resc
    res["cIIIs04"] = A
    print(f"cIIIs04 done {time.time() - t0:.0f}s", flush=True)

    # PAT16: 17 runs, format 1 records (20-point paths; their first and last points are the ends)
    z = np.load(DATA / "ensemble" / "PAT16" / "tumor_dwi.npz")
    to_t1 = np.linalg.inv(z["t1_to_dwi"])
    anat = Anatomy("PAT16")
    first = DATA / "ensemble" / "PAT16" / "added-smooth"
    tr, okk, sc = [], [], []
    for r in range(17):
        rec = E.RunRecord.load(first / f"r{r:03d}.npz", logp=False)
        t, ok, s, _, _ = score(rec, anat, to_t1)
        tr.append(t); okk.append(ok); sc.append(s)
    allt, allok, alls = np.concatenate(tr), np.concatenate(okk), np.concatenate(sc)
    res["PAT16"] = {"17 runs pooled: plausible share of scored named streamlines": round(float(allok[alls].mean()), 3),
                    "17 runs pooled: per bundle": per_bundle(allt, allok, alls),
                    "note": "FastSurfer is trained on healthy brains; its labels near the tumor are unchecked"}
    (HERE / "results/ensemble_endpoints.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res["cIIIs04"][k] for k in res["cIIIs04"] if "pooled: plausible" in k or "NCC" in k}),
          res["PAT16"]["17 runs pooled: plausible share of scored named streamlines"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
