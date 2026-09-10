#!/usr/bin/env python3
"""
profile_raw_archive.py

Inventory and profile a raw data tree. Answers, with numbers:
  1. how many files, how big, how deep
  2. what file TYPES exist (extension x count x size)
  3. size distribution: buckets, zero-byte files, largest files
  4. per-folder x extension matrix
  5. naming conventions: delimiters, case style, token vocabulary,
     subject IDs encoded in filenames, timepoint/phase tokens, campaign tokens
  6. duplicates: same basename in >1 folder, byte-identical content (md5)

Writes file_inventory.csv (one row per file) so any number printed here can be
traced back to the exact files that produced it.

Usage:
    python3 profile_raw_archive.py <root_dir> [--outdir OUT] [--no-hash]

Example:
    python3 profile_raw_archive.py \
      "/Users/rtscott2/Desktop/AI/20260429/BR TFM Project/bedrest_tfm/03_raw_downloads" \
      --outdir ~/Desktop/AI/20260429/archive_profile
"""

import os
import re
import csv
import sys
import json
import hashlib
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# Noise to ignore everywhere (macOS cruft, resource forks, lock files).
IGNORE_NAMES = {".DS_Store", "Icon\r", "Thumbs.db", "desktop.ini"}
IGNORE_PATH_PARTS = {"__MACOSX"}
IGNORE_PREFIXES = ("._", "~$")

# Subject ID forms used across C1/C3/C11.
RE_NUM_SUBJ = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")
RE_C1G_SUBJ = re.compile(r"C1G\d{3,4}", re.IGNORECASE)

# Timepoint / study-phase vocabulary that shows up in FILENAMES.
PHASE_TOKENS = {
    "pre", "post", "post1", "post2", "pre1", "pre2", "in", "intest",
    "bdc", "br", "brday", "day", "baseline", "recovery", "screen", "screening",
    "test", "test1", "test2", "test3", "test4", "visit", "session", "trial",
    "avg", "avgs", "obsv", "summary", "all", "raw", "final", "draft",
}

SIZE_BUCKETS = [
    (0, "0 B (empty)"),
    (1, "1 B - 1 KB"),
    (1024, "1 KB - 10 KB"),
    (10 * 1024, "10 KB - 100 KB"),
    (100 * 1024, "100 KB - 1 MB"),
    (1024 ** 2, "1 MB - 10 MB"),
    (10 * 1024 ** 2, "10 MB - 100 MB"),
    (100 * 1024 ** 2, "100 MB+"),
]


def human(n):
    """Bytes to a short human string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def bucket_of(size):
    label = SIZE_BUCKETS[0][1]
    for lo, lab in SIZE_BUCKETS:
        if size >= lo:
            label = lab
    return label


def is_noise(path: Path):
    if path.name in IGNORE_NAMES:
        return True
    if path.name.startswith(IGNORE_PREFIXES):
        return True
    if IGNORE_PATH_PARTS & set(path.parts):
        return True
    return False


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return ""
    return h.hexdigest()


def case_style(stem):
    """Coarse naming-case classification of a filename stem."""
    letters = [c for c in stem if c.isalpha()]
    if not letters:
        return "no_letters"
    up = sum(c.isupper() for c in letters)
    lo = sum(c.islower() for c in letters)
    if up == len(letters):
        return "UPPER"
    if lo == len(letters):
        return "lower"
    return "Mixed"


def delimiters_in(stem):
    d = set()
    if "_" in stem:
        d.add("underscore")
    if "-" in stem:
        d.add("hyphen")
    if " " in stem:
        d.add("space")
    if "." in stem:
        d.add("dot")
    return d or {"none"}


def tokenize(stem):
    return [t for t in re.split(r"[_\-\s\.\(\)\[\]]+", stem) if t]


def scan(root, do_hash=True):
    rows = []
    root = Path(root)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_noise(p):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        folder = parts[0] if len(parts) > 1 else "<root>"
        stem = p.stem
        ext = p.suffix.lower().lstrip(".") or "<none>"
        rows.append({
            "folder": folder,
            "relpath": str(rel),
            "filename": p.name,
            "stem": stem,
            "ext": ext,
            "bytes": size,
            "depth": len(parts) - 1,
            "md5": md5_of(p) if do_hash else "",
        })
    return rows


def print_header(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Profile a raw data tree.")
    ap.add_argument("root", help="root directory to scan recursively")
    ap.add_argument("--outdir", default=".", help="where to write file_inventory.csv")
    ap.add_argument("--no-hash", action="store_true", help="skip md5 (faster, no content-duplicate detection)")
    ap.add_argument("--top", type=int, default=25, help="how many rows in top-N listings")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")

    rows = scan(root, do_hash=not args.no_hash)
    if not rows:
        sys.exit("no files found")

    total_bytes = sum(r["bytes"] for r in rows)
    folders = sorted({r["folder"] for r in rows})
    n_dirs = sum(1 for p in Path(root).rglob("*") if p.is_dir() and not (IGNORE_PATH_PARTS & set(p.parts)))

    # ---------------------------------------------------------------- totals
    print_header("1. TOTALS")
    print(f"root                 {root}")
    print(f"files                {len(rows)}")
    print(f"total size           {human(total_bytes)}  ({total_bytes:,} bytes)")
    print(f"top-level folders    {len(folders)}")
    print(f"subdirectories (all) {n_dirs}")
    depth = Counter(r["depth"] for r in rows)
    print("nesting depth (dirs below root -> files):")
    for d in sorted(depth):
        print(f"  depth {d}: {depth[d]:>5} files")

    # ------------------------------------------------------------ file types
    print_header("2. FILE TYPES")
    by_ext = defaultdict(list)
    for r in rows:
        by_ext[r["ext"]].append(r["bytes"])
    print(f"{'ext':<12} {'count':>6} {'pct':>6} {'total':>10} {'median':>10} {'min':>9} {'max':>10}")
    for ext, sizes in sorted(by_ext.items(), key=lambda kv: -len(kv[1])):
        sizes_sorted = sorted(sizes)
        med = sizes_sorted[len(sizes_sorted) // 2]
        print(f"{ext:<12} {len(sizes):>6} {100*len(sizes)/len(rows):>5.1f}% "
              f"{human(sum(sizes)):>10} {human(med):>10} {human(sizes_sorted[0]):>9} {human(sizes_sorted[-1]):>10}")
    print(f"\ndistinct extensions: {len(by_ext)}")

    # ------------------------------------------------------ size distribution
    print_header("3. SIZE DISTRIBUTION")
    buckets = Counter(bucket_of(r["bytes"]) for r in rows)
    for _, lab in SIZE_BUCKETS:
        if buckets.get(lab):
            print(f"  {lab:<18} {buckets[lab]:>5}")
    empties = [r for r in rows if r["bytes"] == 0]
    print(f"\nzero-byte files: {len(empties)}")
    for r in empties[:args.top]:
        print(f"  {r['relpath']}")
    if len(empties) > args.top:
        print(f"  ... +{len(empties) - args.top} more (see file_inventory.csv)")

    print(f"\nlargest {args.top} files:")
    for r in sorted(rows, key=lambda r: -r["bytes"])[:args.top]:
        print(f"  {human(r['bytes']):>9}  {r['relpath']}")

    # --------------------------------------------------- folder x ext matrix
    print_header("4. PER-FOLDER BREAKDOWN (files, size, extensions)")
    fstats = defaultdict(lambda: {"n": 0, "bytes": 0, "ext": Counter()})
    for r in rows:
        f = fstats[r["folder"]]
        f["n"] += 1
        f["bytes"] += r["bytes"]
        f["ext"][r["ext"]] += 1
    print(f"{'folder':<22} {'files':>6} {'size':>10}  extensions")
    for folder, st in sorted(fstats.items(), key=lambda kv: -kv[1]["n"]):
        exts = ", ".join(f"{e}:{c}" for e, c in st["ext"].most_common())
        print(f"{folder:<22} {st['n']:>6} {human(st['bytes']):>10}  {exts}")

    # ----------------------------------------------------------- nomenclature
    print_header("5. NAMING CONVENTIONS")

    delim = Counter()
    for r in rows:
        for d in delimiters_in(r["stem"]):
            delim[d] += 1
    print("delimiter usage (a filename can use more than one):")
    for d, c in delim.most_common():
        print(f"  {d:<12} {c:>5}  ({100*c/len(rows):.1f}% of files)")

    cases = Counter(case_style(r["stem"]) for r in rows)
    print("\ncase style of filename stem:")
    for c, n in cases.most_common():
        print(f"  {c:<12} {n:>5}")

    lens = sorted(len(r["filename"]) for r in rows)
    print(f"\nfilename length: min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")
    print("longest filenames:")
    for r in sorted(rows, key=lambda r: -len(r["filename"]))[:5]:
        print(f"  [{len(r['filename'])}] {r['relpath']}")

    # token vocabulary
    tokens = Counter()
    for r in rows:
        tokens.update(t.lower() for t in tokenize(r["stem"]))
    print(f"\ndistinct filename tokens: {len(tokens)}")
    print(f"top {args.top} tokens:")
    for t, c in tokens.most_common(args.top):
        print(f"  {c:>5}  {t}")

    # leading token = de facto naming prefix / study code
    lead = Counter()
    for r in rows:
        tk = tokenize(r["stem"])
        if tk:
            lead[tk[0]] += 1
    print(f"\nleading token (study/prefix code), top {args.top}:")
    for t, c in lead.most_common(args.top):
        print(f"  {c:>5}  {t}")

    # phase / timepoint tokens present in names
    phase_hits = Counter()
    for r in rows:
        for t in tokenize(r["stem"]):
            tl = t.lower()
            if tl in PHASE_TOKENS:
                phase_hits[tl] += 1
    print("\ntimepoint/phase tokens appearing in filenames:")
    for t, c in phase_hits.most_common():
        print(f"  {c:>5}  {t}")

    # subject IDs encoded in filenames
    subj_files = []
    subj_ids = Counter()
    for r in rows:
        ids = set(RE_NUM_SUBJ.findall(r["stem"])) | {m.upper() for m in RE_C1G_SUBJ.findall(r["stem"])}
        if ids:
            subj_files.append((r, sorted(ids)))
            subj_ids.update(ids)
    print(f"\nfiles with a 3-4 digit or C1Gxxxx token in the NAME: {len(subj_files)} "
          f"({100*len(subj_files)/len(rows):.1f}%)")
    print("  (candidates for filename-encoded subject IDs; year-like tokens are noise)")
    byfolder = Counter(r["folder"] for r, _ in subj_files)
    for f, c in byfolder.most_common():
        print(f"  {c:>5}  {f}")

    # extension-less and odd names
    noext = [r for r in rows if r["ext"] == "<none>"]
    if noext:
        print(f"\nfiles with no extension: {len(noext)}")
        for r in noext[:args.top]:
            print(f"  {r['relpath']}")

    weird = [r for r in rows if re.search(r"[^A-Za-z0-9_\-\. ()\[\]+&%,']", r["filename"])]
    if weird:
        print(f"\nfilenames with unusual characters: {len(weird)}")
        for r in weird[:args.top]:
            print(f"  {r['relpath']}")

    # ------------------------------------------------------------ duplicates
    print_header("6. DUPLICATES")
    byname = defaultdict(list)
    for r in rows:
        byname[r["filename"].lower()].append(r["relpath"])
    dupname = {k: v for k, v in byname.items() if len(v) > 1}
    print(f"filenames appearing in more than one location: {len(dupname)}")
    for k, v in sorted(dupname.items(), key=lambda kv: -len(kv[1]))[:args.top]:
        print(f"  x{len(v)}  {k}")
        for p in v[:4]:
            print(f"        {p}")
        if len(v) > 4:
            print(f"        ... +{len(v)-4} more")

    if not args.no_hash:
        byhash = defaultdict(list)
        for r in rows:
            if r["md5"]:
                byhash[r["md5"]].append(r)
        dupes = {h: v for h, v in byhash.items() if len(v) > 1}
        wasted = sum(v[0]["bytes"] * (len(v) - 1) for v in dupes.values())
        print(f"\nbyte-identical content groups: {len(dupes)}  "
              f"({sum(len(v) for v in dupes.values())} files, {human(wasted)} redundant)")
        for h, v in sorted(dupes.items(), key=lambda kv: -kv[1][0]["bytes"] * (len(kv[1]) - 1))[:args.top]:
            print(f"  x{len(v)}  {human(v[0]['bytes']):>9}  {v[0]['filename']}")
            for r in v[:4]:
                print(f"        {r['relpath']}")
            if len(v) > 4:
                print(f"        ... +{len(v)-4} more")

    # ------------------------------------------------------------------ write
    inv_path = os.path.join(outdir, "file_inventory.csv")
    with open(inv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["folder", "relpath", "filename", "stem", "ext", "bytes", "depth", "md5"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "root": root,
        "n_files": len(rows),
        "total_bytes": total_bytes,
        "n_top_level_folders": len(folders),
        "extensions": {e: len(s) for e, s in sorted(by_ext.items(), key=lambda kv: -len(kv[1]))},
        "zero_byte_files": len(empties),
        "duplicate_name_groups": len(dupname),
    }
    sum_path = os.path.join(outdir, "archive_profile_summary.json")
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=1)

    print_header("WROTE")
    print(inv_path)
    print(sum_path)


if __name__ == "__main__":
    main()
