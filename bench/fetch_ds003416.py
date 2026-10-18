"""Rescans for the bootstrap's test-retest check (NOTES 2026-10-08, "The bootstrap against real rescans"): a slice of
OpenNeuro ds003416, MASiVar (CC0; Cai et al., MRM 2021, doi:10.1002/mrm.28926; Philips Achieva 3T, raw BIDS).

    uv run bench/fetch_ds003416.py

What is read (raw series only, whole, with their .bval/.bvec/.json; no T1, no PreQual derivatives):
  sub-cIIIs01 (scanner B, b1000 x40 + b2000 x56, each shell its own series, plus a reversed "APA" series):
    ses-s1Bx1 scans 2 and 4 (2.5 mm iso) and scan 1 (2.1 x 2.1 x 2.2 mm); ses-s1Bx3 scans 1 and 2 (2.1 mm).
  sub-cIIIs02, 03, 05, 06, 07 and 08 (scanner B, the same protocol; bench/calibration_runs.py, the calibration card):
    every scan of the sessions below, 2.1 x 2.1 x 2.2 mm only (GRID). cIIIs02 ses-s1Bx5 scan 1's b2000 series is
    2.2 mm iso and is not read, so that scan is incomplete here (its rescan is scan 2).
    WANT fetches every scan of those sessions, more than the card uses (calibration_runs.CARD): the other sessions'
    scan 2 and cIIIs02 ses-s1Bx5 scan 1's b1000 and APA series are not used. They are kept for completeness, for later
    use (more rescans, or a second same-session pair), at about 30 MB a scan.
  sub-cIs1 (scanner A, 96 directions a shell; b1000, b2000 and b3000 of its five shells):
    ses-s1Ax1 scans 1 and 2, ses-s1Ax2 scan 1 (consecutive days), 2.5 mm iso.
A scan is the run's first digit (run-<scan><acquisition>). The file list comes from the dataset's GitHub mirror,
the files from OpenNeuro's S3 bucket. Writes $TRACTOGRAPHY_DATA/ds003416/<BIDS path> and ds003416/fetched.json.
"""
import json, re, urllib.request
from tractline.data import DATA

S3 = "https://s3.amazonaws.com/openneuro.org/ds003416"
GH = "https://api.github.com/repos/OpenNeuroDatasets/ds003416/git/trees/HEAD?recursive=1"
OUT = DATA / "ds003416"
WANT = {  # (subject, session): {scan: shells}
    ("cIIIs01", "s1Bx1"): {1: (1000, 2000), 2: (1000, 2000), 4: (1000, 2000)},
    ("cIIIs01", "s1Bx3"): {1: (1000, 2000), 2: (1000, 2000)},
    ("cIs1", "s1Ax1"): {1: (1000, 2000, 3000), 2: (1000, 2000, 3000)},
    ("cIs1", "s1Ax2"): {1: (1000, 2000, 3000)},
    **{(sub, ses): {1: (1000, 2000), 2: (1000, 2000)} for sub, sessions in (
        ("cIIIs02", ("s1Bx1", "s1Bx4", "s1Bx5")), ("cIIIs03", ("s1Bx1",)), ("cIIIs05", ("s1Bx1", "s1Bx2")),
        ("cIIIs06", ("s1Bx1",)), ("cIIIs07", ("s1Bx1",)), ("cIIIs08", ("s1Bx1", "s1Bx2"))) for ses in sessions},
}
GRID = {sub: "21x21x22" for sub in ("cIIIs02", "cIIIs03", "cIIIs05", "cIIIs06", "cIIIs07", "cIIIs08")}   # read at this resolution only
NAME = re.compile(r"sub-(\w+?)/ses-(\w+?)/dwi/sub-\w+?_ses-\w+?_acq-b(\d+)n(\d+)r(\w+?)pe(APP|APA)_run-(\d)\d\d_dwi\.(nii\.gz|bval|bvec|json)$")


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def wanted(path):
    m = NAME.match(path)
    if not m:
        return False
    sub, ses, b, res, pe, scan = m.group(1), m.group(2), int(m.group(3)), m.group(5), m.group(6), int(m.group(7))
    shells = WANT.get((sub, ses), {}).get(scan)
    return shells is not None and (pe == "APA" or b in shells) and GRID.get(sub, res) == res


if __name__ == "__main__":
    paths = sorted(t["path"] for t in json.loads(get(GH))["tree"] if wanted(t["path"]))
    log, total = {}, 0
    for p in ["dataset_description.json", "README"] + paths:
        dst = OUT / p
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = get(f"{S3}/{p}")
            dst.write_bytes(data)
        log[p] = dst.stat().st_size
        total += log[p]
        print(f"{log[p] / 1e6:8.2f} MB  {p}", flush=True)
    (OUT / "fetched.json").write_text(json.dumps({"source": S3, "files": log, "bytes": total}, indent=1))
    print(f"{len(log)} files, {total / 1e6:.1f} MB")
