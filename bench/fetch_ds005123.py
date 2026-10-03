"""A slice of OpenNeuro ds005123 (CC0; Smith, Sharp, Dachs et al.; Siemens Prisma, doi:10.18112/openneuro.ds005123.v1.1.3)
for the phase-encoding tests: per subject, the diffusion series' first two volumes (both b = 0) and the
spin-echo field maps acquired for it in both directions (acq-dwi, dir-AP and dir-PA), with their sidecars.

The diffusion series (145 volumes, ~190 MB) is streamed from its start and decompressed only until its two
leading b0s are in hand (~4 MB read); they are written as a 2-volume NIfTI with the series' own header. The
field maps (~3 MB each) and the sidecars are fetched whole. Public, from OpenNeuro's S3 bucket; nothing
else is read.

    uv run bench/fetch_ds005123.py [--subjects 12]

Writes $TRACTOGRAPHY_DATA/ds005123/sub-*/{dwi,fmap}/ and ds005123/fetched.json (what was read, bytes).
"""
import argparse, gzip, json, struct, urllib.request, zlib
import numpy as np
from tractline.data import DATA

S3 = "https://s3.amazonaws.com/openneuro.org/ds005123"
GH = "https://api.github.com/repos/OpenNeuroDatasets/ds005123/git/trees/HEAD?recursive=1"
OUT = DATA / "ds005123"
TYPE_BYTES = {2: 1, 4: 2, 8: 4, 16: 4, 64: 8, 256: 1, 512: 2, 768: 4}


def get(url, limit=None):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read() if limit is None else r.read(limit)


def leading_volumes(url, n):
    """The first n volumes of a gzipped NIfTI-1, streamed: (header bytes with dim[4] = n, the volumes' bytes,
    compressed bytes read)."""
    z, raw, read = zlib.decompressobj(16 + zlib.MAX_WBITS), b"", 0
    need = None
    with urllib.request.urlopen(url, timeout=60) as r:
        while need is None or len(raw) < need:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            read += len(chunk)
            raw += z.decompress(chunk)
            if need is None and len(raw) >= 352:
                assert struct.unpack("<i", raw[:4])[0] == 348, "not a little-endian NIfTI-1"
                dim = struct.unpack("<8h", raw[40:56])
                datatype = struct.unpack("<h", raw[70:72])[0]
                offset = int(struct.unpack("<f", raw[108:112])[0])
                need = offset + n * dim[1] * dim[2] * dim[3] * TYPE_BYTES[datatype]
    assert need is not None and len(raw) >= need, (url, len(raw), need)
    hdr = bytearray(raw[:offset])
    dim = list(struct.unpack("<8h", hdr[40:56])); dim[4] = n
    hdr[40:56] = struct.pack("<8h", *dim)
    return bytes(hdr), raw[offset:need], read


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", type=int, default=12); args = ap.parse_args()
    paths = [t["path"] for t in json.loads(get(GH))["tree"]]
    def has(sub):
        return all(f"{sub}/{p}" in paths for p in (f"dwi/{sub}_dwi.nii.gz", f"fmap/{sub}_acq-dwi_dir-AP_epi.nii.gz",
                                                   f"fmap/{sub}_acq-dwi_dir-PA_epi.nii.gz"))
    subs = sorted({p.split("/")[0] for p in paths if p.startswith("sub-")})
    chosen = [s for s in subs if has(s)][:args.subjects]
    log = {"source": S3, "subjects": {}}
    for sub in chosen:
        rec = {}
        for d in ("dwi", "fmap"):
            (OUT / sub / d).mkdir(parents=True, exist_ok=True)
        for rel in (f"dwi/{sub}_dwi.json", f"dwi/{sub}_dwi.bval", f"dwi/{sub}_dwi.bvec",
                    f"fmap/{sub}_acq-dwi_dir-AP_epi.json", f"fmap/{sub}_acq-dwi_dir-AP_epi.nii.gz",
                    f"fmap/{sub}_acq-dwi_dir-PA_epi.json", f"fmap/{sub}_acq-dwi_dir-PA_epi.nii.gz"):
            data = get(f"{S3}/{sub}/{rel}")
            (OUT / sub / rel).write_bytes(data)
            rec[rel] = len(data)
        bval = np.array((OUT / sub / f"dwi/{sub}_dwi.bval").read_text().split(), float)
        n = int(np.argmax(bval >= 50))                                      # the leading b0s
        assert n >= 1, (sub, bval[:5])
        hdr, vol, read = leading_volumes(f"{S3}/{sub}/dwi/{sub}_dwi.nii.gz", n)
        with gzip.open(OUT / sub / f"dwi/{sub}_desc-leadingb0_dwi.nii.gz", "wb") as f:
            f.write(hdr); f.write(vol)
        rec[f"dwi/{sub}_dwi.nii.gz (first {n} volumes, streamed)"] = read
        log["subjects"][sub] = rec
        print(sub, n, "b0s;", round(sum(rec.values()) / 1e6, 1), "MB read", flush=True)
    log["total_mb"] = round(sum(sum(r.values()) for r in log["subjects"].values()) / 1e6, 1)
    (OUT / "fetched.json").write_text(json.dumps(log, indent=1))
    print("total", log["total_mb"], "MB")
