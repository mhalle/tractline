"""The labelers' accuracy on TractCloud's own test split (its release v1.0.0, TrainData_800clu800ol/
test.pickle: 20 HCP subjects x 10,000 streamlines, 15-point features, ORG-atlas cluster labels - the
split RapidParc's paper also reports on: TractCloud 92.12 % accuracy / 90.22 % macro F1, RapidParc
94.44 / 93.2).

Each subject is one tractogram: its 10,000 streamlines labeled together (their context drawn from them),
by TractCloud at upstream's inference context (k_global 80, 10 % local subsample) and at the trained one
(500, all streamlines: 10,000 candidates), RapidParc and its hemiaug model; seeds 0 and 1. Pooled over
the 200,000 streamlines: tract accuracy and macro F1 over the 43 classes (42 tracts + Other), cluster
accuracy (1,600), and seed 0 against seed 1.

    python bench/accuracy_tractcloud_test.py

Writes results/accuracy_tractcloud_test.json.
"""
import json, pickle, time
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import f1_score
from tractline.data import DATA
from tractline.labelers import tractcloud, rapidparc
from tractline.labelers.tractcloud import inf

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1)


def tc_clusters(lab, feat, seed, k_global, ds_rate, batch=1024):
    """TractCloud's cluster argmax for one subject's (n, 15, 3) features (the Labeler's forward)."""
    np.random.seed(seed)
    ds = inf.RealDataDataset(inf.center_tractography(feat, lab.center), k=20, k_global=k_global, k_ds_rate=ds_rate)
    Pf = torch.from_numpy(ds.feat).float().to(lab.device).transpose(2, 1).contiguous()
    L = torch.from_numpy(ds.local_feat).float().to(lab.device).transpose(2, 1).contiguous()
    G = torch.from_numpy(ds.global_feat).float().to(lab.device).transpose(2, 1).contiguous()
    out = []
    with torch.no_grad():
        for a in range(0, len(ds), batch):
            b = min(len(ds), a + batch)
            out.append(lab.model(Pf[a:b], torch.cat((L[a:b], G.expand(b - a, -1, -1, -1)), 3)).view(-1, 1600).argmax(1).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    d = pickle.load(open(DATA / "TrainData_800clu800ol/test.pickle", "rb"))
    feat, label, subj = d["feat"].astype(np.float32), d["label"], d["subject_id"]
    subjects = list(dict.fromkeys(subj.tolist()))
    tc80, tc500 = tractcloud.Labeler("mps", upstream=True), tractcloud.Labeler("mps")
    rp, rph = rapidparc.Labeler("mps"), rapidparc.Labeler("mps", model="hemiaug")
    lut = rp.lut
    runs = {"tractcloud_80": lambda f, s: tc_clusters(tc80, f, s, 80, 0.1),
            "tractcloud_500": lambda f, s: tc_clusters(tc500, f, s, 500, min(1.0, tractcloud.LOCAL_CANDIDATES / len(f))),
            "rapidparc": lambda f, s: rp.logits(torch.from_numpy(f), s).argmax(1).numpy(),
            "rapidparc_hemiaug": lambda f, s: rph.logits(torch.from_numpy(f), s).argmax(1).numpy()}
    truth_t = lut[label]
    res = {"streamlines": len(label), "subjects": len(subjects), "labelers": {}}
    for name, fn in runs.items():
        pred = {s: np.empty(len(label), np.int64) for s in SEEDS}
        t0 = time.time()
        for sid in subjects:
            idx = np.flatnonzero(subj == sid)
            for s in SEEDS:
                pred[s][idx] = fn(feat[idx], s)
        row = {"seconds_per_subject_draw": round((time.time() - t0) / (len(subjects) * len(SEEDS)), 2)}
        for s in SEEDS:
            pt = lut[pred[s]]
            row[f"seed {s}"] = {"tract_accuracy": round(float((pt == truth_t).mean()) * 100, 2),
                                "tract_macro_f1": round(float(f1_score(truth_t, pt, average="macro")) * 100, 2),
                                "cluster_accuracy": round(float((pred[s] == label).mean()) * 100, 2),
                                "other_fraction": round(float((pt == 42).mean()), 4)}
        row["seed_agreement_tract"] = round(float((lut[pred[0]] == lut[pred[1]]).mean()), 4)
        res["labelers"][name] = row
        print(name, json.dumps(row), flush=True)
    res["truth_other_fraction"] = round(float((truth_t == 42).mean()), 4)
    (HERE / "results/accuracy_tractcloud_test.json").write_text(json.dumps(res, indent=1))
    print("truth Other fraction", res["truth_other_fraction"])
