"""Manifest of the PFF FC World Cup 2022 files that bancos/pff.py actually reads.

The raw dataset is not redistributed. Third parties download it from the source and verify it against
manifest.csv (path,bytes,sha256) with --check.

  --make            scan the data root and write manifest.csv
  --check           verify a downloaded folder against manifest.csv (exit code 1 if anything is off)
  --selftest        run the checks on temporary files

Relevant files (same ones pff.py opens), per match id found in Metadata/:
  Metadata/<id>.json, Rosters/<id>.json, Event Data/<id>.json, Tracking Data/<id>.jsonl.bz2
Everything else (PDF specifications, players.csv, competitions.csv, cache/, hidden files) is ignored.
"""
import argparse
import csv
import hashlib
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import onside_paths  # noqa: E402

DEFAULT_ROOT = Path(onside_paths.PFF_DIR)           # ONSIDE_PFF_DIR, default ./data/pff
DEFAULT_MANIFEST = HERE / "manifest.csv"

# folder -> file name template, one file per match id
PATTERNS = {
    "Metadata": "{}.json",
    "Rosters": "{}.json",
    "Event Data": "{}.json",
    "Tracking Data": "{}.jsonl.bz2",
}
HEADER = ["path", "bytes", "sha256"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def match_ids(root):
    """Match ids taken from Metadata/*.json, the same source pff.py uses for MATCHES."""
    d = root / "Metadata"
    return sorted((p.stem for p in d.glob("*.json") if p.stem.isdigit()), key=int) if d.is_dir() else []


def relevant(root):
    """Relative posix paths of the files the loader reads, for the ids present in root."""
    return [f"{folder}/{tpl.format(mid)}" for mid in match_ids(root) for folder, tpl in PATTERNS.items()]


def extras(root, known):
    """Files in the relevant folders that look like data for an id but are not in `known`."""
    out = []
    for folder, tpl in PATTERNS.items():
        d = root / folder
        if not d.is_dir():
            continue
        suffix = tpl.format("")
        for p in sorted(d.iterdir()):
            if p.is_file() and p.name.endswith(suffix) and p.name[:-len(suffix)].isdigit():
                rel = f"{folder}/{p.name}"
                if rel not in known:
                    out.append(rel)
    return out


def _hash_rows(root, rels):
    def one(rel):
        p = root / rel
        return rel, p.stat().st_size, sha256(p)
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(one, rels))


def make(root, manifest):
    rels = relevant(root)
    if not rels:
        sys.exit(f"no Metadata/*.json under {root}")
    missing = [r for r in rels if not (root / r).is_file()]
    if missing:
        sys.exit(f"{len(missing)} files missing under {root}, e.g. {missing[0]}")
    rows = _hash_rows(root, rels)
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"{manifest}: {len(rows)} files, {sum(r[1] for r in rows) / 1e9:.2f} GB")
    return 0


def read_manifest(manifest):
    with open(manifest, newline="") as fh:
        return {r["path"]: (int(r["bytes"]), r["sha256"]) for r in csv.DictReader(fh)}


def check(root, manifest, quiet=False):
    exp = read_manifest(manifest)
    missing, bad = [], []
    present = []
    for rel, (size, _) in exp.items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif p.stat().st_size != size:
            bad.append((rel, "size"))
        else:
            present.append(rel)
    for rel, _, digest in _hash_rows(root, present):
        if digest != exp[rel][1]:
            bad.append((rel, "sha256"))
    extra = extras(root, exp)
    log = (lambda *a: None) if quiet else print
    for rel in missing:
        log(f"MISSING  {rel}")
    for rel, why in bad:
        log(f"DIFFERS  {rel} ({why})")
    for rel in extra:
        log(f"EXTRA    {rel}")
    ok = not (missing or bad or extra)
    log(f"{len(exp)} expected, {len(missing)} missing, {len(bad)} differ, {len(extra)} extra: "
        + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        root, man = Path(tmp) / "pff", Path(tmp) / "manifest.csv"
        for folder, tpl in PATTERNS.items():
            (root / folder).mkdir(parents=True)
            for mid in ("10", "11"):
                (root / folder / tpl.format(mid)).write_bytes(f"{folder}{mid}".encode())
        # irrelevant files must stay out of the manifest and never count as extras
        (root / "cache").mkdir()
        (root / "cache" / "10_tracking_v3.npz").write_bytes(b"x")
        (root / "players.csv").write_bytes(b"x")
        (root / "Event Data" / "spec.pdf").write_bytes(b"x")
        (root / "Event Data" / ".DS_Store").write_bytes(b"x")
        assert make(root, man) == 0
        exp = read_manifest(man)
        assert len(exp) == 8 and not any(k.startswith("cache") or k.endswith(".pdf") for k in exp), exp
        assert check(root, man, quiet=True) == 0
        (root / "Rosters" / "11.json").write_bytes(b"tampered")          # other hash
        assert check(root, man, quiet=True) == 1
        (root / "Rosters" / "11.json").write_bytes(b"Rosters11")
        assert check(root, man, quiet=True) == 0
        (root / "Tracking Data" / "11.jsonl.bz2").unlink()               # missing
        assert check(root, man, quiet=True) == 1
        (root / "Tracking Data" / "11.jsonl.bz2").write_bytes(b"Tracking Data11")
        (root / "Metadata" / "12.json").write_bytes(b"{}")               # relevant extra
        assert check(root, man, quiet=True) == 1
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=f"data root (default {DEFAULT_ROOT})")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.root.is_dir():
        sys.exit(f"{a.root} is not a directory")
    return make(a.root, a.manifest) if a.make else check(a.root, a.manifest)


if __name__ == "__main__":
    sys.exit(main())
