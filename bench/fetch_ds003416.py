"""Rescans for the bootstrap's test-retest check (NOTES 2026-10-08, "The bootstrap against real rescans"): a slice of
OpenNeuro ds003416, MASiVar (CC0; Cai et al., MRM 2021, doi:10.1002/mrm.28926; Philips Achieva 3T, raw BIDS).

    uv run bench/fetch_ds003416.py [--dry-run]     # --dry-run: list what would be fetched, with S3's sizes

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
  sub-cIIs00..cIIs04 (cohort II, site 1 only, scanners A and B; bench/calibration_cohortII.py): every DWI series at
    2.5 mm iso (GRID 25x25x25) of sessions s1Ax1, s1Ax2, s1Bx1 and s1Bx2 (cIIs00: s1Ax1 and s1Bx1 only): b1000n96,
    b1500n96, b2000n96, b2500n96 and b1000n32 (APP), and every 2.5 mm reversed series (APA, b1000n3; two or three a
    session). On scanner A only cIIs00's series are all 2.5 mm: cIIs01-04's 96-direction shells there are
    1.9 x 1.9 x 2.2 mm (r19x19x22) and their reversed series r06x06x25. Of cIIs01-04's scanner A sessions (SCANNER_A)
    are read: b1000n32 (2.5 mm), b2000n96 at 1.9 mm, and the reversed series (r06x06x25) nearest in acquisition
    number to each of them (ties: the later), one or two a session; not the other 1.9 mm shells nor the third reversed
    series. Site 2, T1s and derivatives are not read.
  sub-cIIs01..cIIs04, site 3 (SITE3; Siemens Skyra 3T, session s3x1; bench/calibration_crosssite.py): the
    b2000n96r25x25x25 APP series and the b1000n3r25x25x25 APA series nearest it in acquisition number (ties: the
    later), with their .bval/.bvec; nothing else of site 3 (cIIs00 has no site 3 APA series and is not read there).
A scan is the run's first digit (run-<scan><acquisition>). The file list comes from the dataset's GitHub mirror,
the files from OpenNeuro's S3 bucket. Writes $TRACTOGRAPHY_DATA/ds003416/<BIDS path> and ds003416/fetched.json. Every file's size on disk is then checked against
OpenNeuro's S3 listing (ListObjectsV2 under each subject's prefix); a file fetched in this run whose size differs is
fetched again, once (a file already on disk is never replaced), and the run fails if any differs. fetched.json records the listing's size beside each file.
"""
import json, re, sys, urllib.request
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
    **{(sub, ses): {1: (1000, 1500, 2000, 2500)} for sub in ("cIIs00", "cIIs01", "cIIs02", "cIIs03", "cIIs04")
       for ses in (("s1Ax1", "s1Bx1") if sub == "cIIs00" else ("s1Ax1", "s1Ax2", "s1Bx1", "s1Bx2"))},
}
GRID = {**{sub: "21x21x22" for sub in ("cIIIs02", "cIIIs03", "cIIIs05", "cIIIs06", "cIIIs07", "cIIIs08")},   # read at this resolution only
        **{sub: "25x25x25" for sub in ("cIIs00", "cIIs01", "cIIs02", "cIIs03", "cIIs04")}}
SCANNER_A = {"subjects": ("cIIs01", "cIIs02", "cIIs03", "cIIs04"), "sessions": ("s1Ax1", "s1Ax2"),
             "APP": {"b2000n96": "19x19x22"}, "APA": "06x06x25", "nearest to": ("b2000n96", "b1000n32")}
SITE3 = {"subjects": ("cIIs01", "cIIs02", "cIIs03", "cIIs04"), "sessions": ("s3x1",),
         "APP": {"b2000n96": "25x25x25"}, "APA": "25x25x25", "nearest to": ("b2000n96",)}
NAME = re.compile(r"sub-(\w+?)/ses-(\w+?)/dwi/sub-\w+?_ses-\w+?_acq-b(\d+)n(\d+)r(\w+?)pe(APP|APA)_run-(\d)\d\d_dwi\.(nii\.gz|bval|bvec|json)$")


def scanner_a(tree_paths, spec=None):
    """cIIs01-04's scanner A extras (SCANNER_A): b2000n96 at 1.9 mm and the reversed series nearest b2000n96 and
    b1000n32, with their .bval/.bvec (cohort II has no sidecars). Another spec of the same shape (SITE3) picks alike."""
    spec = spec or SCANNER_A
    rx = re.compile(r"sub-(\w+?)/ses-(\w+?)/dwi/sub-\w+?_ses-\w+?_acq-(b\d+n\d+)r(\w+?)pe(APP|APA)_run-(\d{3})_dwi\.nii\.gz$")
    by = {}
    for p in tree_paths:
        m = rx.match(p)
        if m and m.group(1) in spec["subjects"] and m.group(2) in spec["sessions"]:
            by.setdefault((m.group(1), m.group(2)), []).append((m.group(3), m.group(4), m.group(5), int(m.group(6)), p))
    out = []
    for key, rows in sorted(by.items()):
        app = {(a, r): (run, p) for a, r, pe, run, p in rows if pe == "APP"}
        apa = [(run, p) for a, r, pe, run, p in rows if pe == "APA" and r == spec["APA"]]
        pick = [app[(a, r)][1] for a, r in spec["APP"].items()]
        for a in spec["nearest to"]:
            run = next(v[0] for (aa, _), v in app.items() if aa == a)
            pick.append(min(apa, key=lambda x: (abs(x[0] - run), -x[0]))[1])
        out += [q[:-7] + ext for q in sorted(set(pick)) for ext in (".nii.gz", ".bval", ".bvec")]
    return out


def site3(tree_paths):
    """cIIs01-04's site 3 extras (SITE3): b2000n96 at 2.5 mm and the 2.5 mm reversed series nearest it."""
    return scanner_a(tree_paths, SITE3)


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


def s3_sizes(prefix):
    """{path under ds003416/: size} from S3's ListObjectsV2 under ds003416/<prefix>."""
    import urllib.parse, xml.etree.ElementTree as ET
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    out, token = {}, None
    while True:
        q = {"list-type": "2", "prefix": f"ds003416/{prefix}", "max-keys": "1000", **({"continuation-token": token} if token else {})}
        root = ET.fromstring(get("https://s3.amazonaws.com/openneuro.org?" + urllib.parse.urlencode(q)))
        for c in root.iter(f"{ns}Contents"):
            out[c.find(f"{ns}Key").text[len("ds003416/"):]] = int(c.find(f"{ns}Size").text)
        token = root.findtext(f"{ns}NextContinuationToken")
        if not token:
            return out


def fetch(p):
    dst = OUT / p
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.write_bytes(get(f"{S3}/{p}"))
    tmp.replace(dst)


if __name__ == "__main__":
    tree = [t["path"] for t in json.loads(get(GH))["tree"]]
    have = set(tree)
    extra = scanner_a(tree) + site3(tree)
    assert all(p in have for p in extra), [p for p in extra if p not in have]
    paths = sorted({p for p in tree if wanted(p)} | set(extra))
    listing = {}
    for sub in sorted({p.split("/")[0] for p in paths}):
        listing.update(s3_sizes(sub + "/"))
    listing.update({k: v for k, v in s3_sizes("").items() if k in ("dataset_description.json", "README")})
    if "--dry-run" in sys.argv[1:]:          # what a run would fetch (not on disk), with S3's sizes; nothing written
        todo = [p for p in ["dataset_description.json", "README"] + paths if not (OUT / p).exists()]
        for p in todo:
            print(f"{(listing.get(p) or 0) / 1e6:8.2f} MB  {p}")
        print(f"{len(todo)} files to fetch, {sum(listing.get(p) or 0 for p in todo) / 1e6:.1f} MB (S3 listing); "
              f"{sum(p not in listing for p in todo)} not in the listing")
        raise SystemExit(0)
    log, total, bad = {}, 0, []
    for p in ["dataset_description.json", "README"] + paths:
        dst = OUT / p
        new = not dst.exists()
        if new:
            fetch(p)
        if new and p in listing and dst.stat().st_size != listing[p]:     # a file on disk before this run is never replaced
            print(f"size {dst.stat().st_size} != S3's {listing[p]}: fetching again  {p}", flush=True)
            fetch(p)
        log[p] = {"bytes": dst.stat().st_size, "S3 listing bytes": listing.get(p)}
        if log[p]["bytes"] != log[p]["S3 listing bytes"]:
            bad.append(p)
        total += log[p]["bytes"]
        print(f"{log[p]['bytes'] / 1e6:8.2f} MB  {p}", flush=True)
    (OUT / "fetched.json").write_text(json.dumps({"source": S3, "files": log, "bytes": total,
                                                  "size check": "every file against S3's listing" if not bad else {"differ": bad}}, indent=1))
    print(f"{len(log)} files, {total / 1e6:.1f} MB; " + ("every size equal to S3's listing" if not bad else f"{len(bad)} differ from S3's listing"))
    if bad:
        raise SystemExit(f"sizes differ from S3's listing: {bad}")
