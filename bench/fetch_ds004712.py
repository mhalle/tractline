"""Same-model, cross-site rescans for the calibration study (bench/calibration_onharmony.py): a slice of OpenNeuro
ds004712, ON-Harmony v2.0.1 (CC0; Warrington et al., Sci Data 12:609, 2025, doi:10.1038/s41597-025-04822-2).

    uv run bench/fetch_ds004712.py [--dry-run]     # --dry-run: list what would be fetched, with S3's sizes

What is read: the two Oxford Siemens Prismas only, every session of each (ses-OXF1PRI00*, WIN-FMRIB, 32-channel coil;
ses-OXF2PRI00*, WIN-OHBA, 64-channel coil), all 20 subjects. Of each session: dwi/*_dir-AP_dwi.{nii.gz,bval,bvec,json}
(the UK Biobank-style b1000/b2000 series), dwi/*_dir-PA_dwi.{nii.gz,bval,bvec,json} (the reversed series) and
anat/*_T1w.nii.gz; plus the top-level README, participants.tsv, dataset_description.json and scan_delays.csv.
Nothing else (no T2w, rfMRI, SWI, defacing masks, T1w sidecars, other scanners).

The file list and the sizes come from OpenNeuro's S3 listing (ListObjectsV2 under each subject's prefix), the files
from the same bucket. Writes $TRACTOGRAPHY_DATA/ds004712/<BIDS path> and ds004712/fetched.json. Every file's size on
disk is checked against the listing; a file fetched in this run whose size differs is fetched again, once (a file
already on disk is never replaced), and the run fails if any differs. fetched.json records the listing's size beside
each file.
"""
import json, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from tractline.data import DATA

DS = "ds004712"
S3 = f"https://s3.amazonaws.com/openneuro.org/{DS}"
OUT = DATA / DS
TOP = ("dataset_description.json", "README", "participants.tsv", "scan_delays.csv")
NAME = re.compile(r"sub-(\d+)/ses-(OXF[12]PRI00\d)/(dwi/sub-\1_ses-\2_dir-(AP|PA)_dwi\.(nii\.gz|bval|bvec|json)"
                  r"|anat/sub-\1_ses-\2_T1w\.nii\.gz)$")
WORKERS = 4


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def s3_sizes(prefix, delimiter=None):
    """{path under ds004712/: size} from S3's ListObjectsV2 under ds004712/<prefix> (and the common prefixes when a
    delimiter is given)."""
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    out, pre, token = {}, [], None
    while True:
        q = {"list-type": "2", "prefix": f"{DS}/{prefix}", "max-keys": "1000", **({"delimiter": delimiter} if delimiter else {}),
             **({"continuation-token": token} if token else {})}
        root = ET.fromstring(get("https://s3.amazonaws.com/openneuro.org?" + urllib.parse.urlencode(q)))
        for c in root.iter(f"{ns}Contents"):
            out[c.find(f"{ns}Key").text[len(DS) + 1:]] = int(c.find(f"{ns}Size").text)
        pre += [c.findtext(f"{ns}Prefix")[len(DS) + 1:] for c in root.iter(f"{ns}CommonPrefixes")]
        token = root.findtext(f"{ns}NextContinuationToken")
        if not token:
            return (out, pre) if delimiter else out


def fetch(p):
    dst = OUT / p
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    tmp.write_bytes(get(f"{S3}/{p}"))
    tmp.replace(dst)


if __name__ == "__main__":
    top, subjects = s3_sizes("", "/")
    subjects = sorted(s for s in subjects if re.fullmatch(r"sub-\d+/", s))
    listing = {k: v for k, v in top.items() if k in TOP}
    for s in subjects:
        listing.update({k: v for k, v in s3_sizes(s).items() if NAME.match(k)})
    paths = [p for p in TOP if p in listing] + sorted(p for p in listing if p not in TOP)
    if "--dry-run" in sys.argv[1:]:          # what a run would fetch (not on disk), with S3's sizes; nothing written
        todo = [p for p in paths if not (OUT / p).exists()]
        for p in todo:
            print(f"{listing[p] / 1e6:8.2f} MB  {p}")
        ses = sorted({tuple(p.split("/")[:2]) for p in paths if "/" in p})
        print(f"{len(subjects)} subjects, {len(ses)} sessions; {len(todo)} files to fetch, "
              f"{sum(listing[p] for p in todo) / 1e9:.2f} GB (S3 listing); top-level files missing: "
              f"{[p for p in TOP if p not in listing]}")
        raise SystemExit(0)

    def one(p):
        dst = OUT / p
        new = not dst.exists()
        if new:
            fetch(p)
        if new and dst.stat().st_size != listing[p]:     # a file on disk before this run is never replaced
            print(f"size {dst.stat().st_size} != S3's {listing[p]}: fetching again  {p}", flush=True)
            fetch(p)
        n = dst.stat().st_size
        print(f"{n / 1e6:8.2f} MB  {p}", flush=True)
        return p, {"bytes": n, "S3 listing bytes": listing[p]}

    with ThreadPoolExecutor(WORKERS) as ex:
        log = dict(ex.map(one, paths))
    log = {p: log[p] for p in paths}
    bad = [p for p, v in log.items() if v["bytes"] != v["S3 listing bytes"]]
    total = sum(v["bytes"] for v in log.values())
    (OUT / "fetched.json").write_text(json.dumps({"source": S3, "version": "2.0.1 (S3's current; dataset_description.json's DatasetDOI)",
                                                  "files": log, "bytes": total,
                                                  "size check": "every file against S3's listing" if not bad else {"differ": bad}}, indent=1))
    print(f"{len(log)} files, {total / 1e9:.2f} GB; " + ("every size equal to S3's listing" if not bad else f"{len(bad)} differ from S3's listing"))
    if bad:
        raise SystemExit(f"sizes differ from S3's listing: {bad}")
