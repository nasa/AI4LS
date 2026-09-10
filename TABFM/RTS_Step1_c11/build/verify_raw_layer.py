#!/usr/bin/env python3
"""
verify_raw_layer.py

Closes the four open questions in the raw archive layer. Reads only; extracts
nothing; modifies nothing.

CHECK 1  Nested zips (119 in MR079G, 11.7 MB, never opened).
         List every entry WITHOUT extracting: file types inside, uncompressed
         size, subject IDs and dates parsed from the zip name, and whether the
         inner filenames already exist on disk (i.e. is this redundant or is it
         data no build has ever seen). Roster intersection reported.

CHECK 2  md5sum verification (140 checksum files, never verified).
         Parse GNU ("<hash>  path") and BSD ("MD5 (path) = <hash>") formats,
         resolve each entry to a real file, recompute, compare. Reports OK,
         MISMATCH, MISSING TARGET, and files covered by no checksum at all.

CHECK 3  Bundle vs extraction completeness.
         For each of the 140 top-level bundles, compare the zip's own manifest
         against what is on disk in the matching extracted folder. Name-level
         and size-level. Proves extraction was complete, not just that a folder
         with the right name exists.

CHECK 4  Cross-folder duplicate assays.
         Every filename+content collision that spans two different experiment
         folders (e.g. BRSMTReflex vs BRSMFSR monosynaptic reflex). Distinguishes
         "same assay filed twice" from "two distinct assays" by comparing the
         full file sets, not just the overlapping ones.

Outputs (to --outdir):
    raw_layer_report.json          machine-readable everything
    nested_zip_manifest.csv        one row per entry inside every nested zip
    md5_verification.csv           one row per checksum entry, with verdict

Usage:
    python3 verify_raw_layer.py <root_dir> [--outdir OUT] [--roster FILE]

Example:
    python3 verify_raw_layer.py \
      "/Users/rtscott2/Desktop/AI/20260429/BR TFM Project/bedrest_tfm/03_raw_downloads" \
      --outdir ~/Desktop/AI/20260720/20260728/raw_layer_check
"""

import os
import re
import csv
import sys
import json
import zipfile
import hashlib
import argparse
from collections import Counter, defaultdict
from pathlib import Path

IGNORE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_PREFIXES = ("._", "~$")
IGNORE_PARTS = {"__MACOSX"}

# Verified C11 roster (46). Used only to report intersection, never to filter.
C11_ROSTER = {
    "5210", "5297", "5803", "6213", "6319", "6546", "6791", "6947", "7574", "7750", "8936",
    "5159", "5160", "5188", "5627", "6403", "6611", "6877", "7036", "7152", "7326", "7350",
    "7707", "8010", "8072", "8713", "8784", "8837", "9667", "9682", "9713",
    "5673", "6187", "6464", "7548", "8177", "8179", "9023", "9793", "9633", "6559", "9217",
    "5016", "8930", "9011", "9751",
}

RE_MD5_GNU = re.compile(r"^([0-9a-fA-F]{32})\s+[\*\s]?(.+?)\s*$")
RE_MD5_BSD = re.compile(r"^MD5\s*\((.+?)\)\s*=\s*([0-9a-fA-F]{32})\s*$")
RE_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
RE_ID4 = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


def is_noise(p: Path):
    return (p.name in IGNORE_NAMES
            or p.name.startswith(IGNORE_PREFIXES)
            or bool(IGNORE_PARTS & set(p.parts)))


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def index_tree(root: Path):
    """All real files, indexed by relpath and by basename."""
    by_rel, by_base = {}, defaultdict(list)
    for p in root.rglob("*"):
        if not p.is_file() or is_noise(p):
            continue
        rel = str(p.relative_to(root))
        by_rel[rel] = p
        by_base[p.name].append(p)
    return by_rel, by_base


def zip_entries(zpath):
    """(entries, error). entries = list of dicts. Never extracts."""
    out = []
    try:
        with zipfile.ZipFile(zpath) as z:
            for i in z.infolist():
                if i.is_dir():
                    continue
                nm = i.filename
                if any(part in IGNORE_PARTS for part in Path(nm).parts):
                    continue
                if Path(nm).name.startswith(IGNORE_PREFIXES) or Path(nm).name in IGNORE_NAMES:
                    continue
                out.append({
                    "name": nm,
                    "basename": Path(nm).name,
                    "ext": (Path(nm).suffix.lower().lstrip(".") or "<none>"),
                    "size": i.file_size,
                    "compressed": i.compress_size,
                    "crc": f"{i.CRC:08x}",
                    "modified": "%04d-%02d-%02d" % i.date_time[:3],
                })
    except (zipfile.BadZipFile, OSError) as e:
        return [], str(e)[:200]
    return out, ""


# ---------------------------------------------------------------- CHECK 1 + 3
def collect_zips(root: Path):
    """Split zips into top-level bundles (depth 1) and nested (depth >= 2)."""
    top, nested = [], []
    for p in root.rglob("*.zip"):
        if is_noise(p):
            continue
        depth = len(p.relative_to(root).parts) - 1
        (top if depth == 1 else nested).append(p)
    return sorted(top), sorted(nested)


def check_nested(nested, root, by_base, outdir):
    hdr("CHECK 1: NESTED ZIPS (never extracted)")
    if not nested:
        print("none found")
        return {"n_zips": 0}

    rows, per_zip, errs = [], [], []
    ext_all, inner_total = Counter(), 0
    already_on_disk = 0
    roster_hits, all_ids, dates = set(), Counter(), []

    for z in nested:
        rel = str(z.relative_to(root))
        entries, err = zip_entries(z)
        if err:
            errs.append((rel, err))
            continue
        stem = z.stem
        ids = set(RE_ID4.findall(stem))
        all_ids.update(ids)
        roster_hits |= (ids & C11_ROSTER)
        m = RE_DATE.search(stem)
        if m:
            dates.append(m.group(0))

        inner_bytes = sum(e["size"] for e in entries)
        inner_total += inner_bytes
        on_disk = sum(1 for e in entries if e["basename"] in by_base)
        already_on_disk += on_disk
        for e in entries:
            ext_all[e["ext"]] += 1
            rows.append({
                "zip_relpath": rel, "zip_folder": z.relative_to(root).parts[0],
                "zip_subject_ids": ";".join(sorted(ids)),
                "zip_date": m.group(0) if m else "",
                "inner_name": e["name"], "inner_ext": e["ext"],
                "inner_bytes": e["size"], "inner_crc": e["crc"],
                "basename_exists_on_disk": int(e["basename"] in by_base),
            })
        per_zip.append({"zip": rel, "n_entries": len(entries), "inner_bytes": inner_bytes,
                        "n_basenames_already_on_disk": on_disk})

    print(f"nested zips        {len(nested)}")
    print(f"entries inside     {len(rows)}")
    print(f"uncompressed total {human(inner_total)}")
    print(f"inner file types   {dict(ext_all.most_common())}")
    print(f"\ninner basenames that ALSO exist as loose files on disk: "
          f"{already_on_disk} / {len(rows)}")
    if len(rows):
        pct = 100 * already_on_disk / len(rows)
        verdict = ("REDUNDANT (contents already extracted elsewhere)" if pct > 95
                   else "NEW DATA (contents not present on disk)" if pct < 5
                   else "PARTIAL overlap, inspect")
        print(f"  -> {verdict}")

    if dates:
        print(f"\ndate range in zip names: {min(dates)} to {max(dates)}  (n={len(dates)})")
    print(f"distinct 3-4 digit IDs in zip names: {len(all_ids)}")
    print(f"of those, on the C11 roster: {len(roster_hits)}"
          + (f"  {sorted(roster_hits)}" if roster_hits else ""))
    if not roster_hits:
        print("  -> no C11 subjects in these zips; they are C1/C3 era")

    if errs:
        print(f"\nunreadable zips: {len(errs)}")
        for r, e in errs[:10]:
            print(f"  {r}: {e}")

    print("\nlargest 8 nested zips by uncompressed content:")
    for z in sorted(per_zip, key=lambda r: -r["inner_bytes"])[:8]:
        print(f"  {human(z['inner_bytes']):>9}  {z['n_entries']:>3} entries  {z['zip']}")

    path = os.path.join(outdir, "nested_zip_manifest.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["zip_relpath", "zip_folder", "zip_subject_ids", "zip_date",
                            "inner_name", "inner_ext", "inner_bytes", "inner_crc",
                            "basename_exists_on_disk"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")

    return {"n_zips": len(nested), "n_entries": len(rows), "inner_bytes": inner_total,
            "inner_ext": dict(ext_all), "basenames_already_on_disk": already_on_disk,
            "roster_ids_in_zip_names": sorted(roster_hits),
            "distinct_ids_in_zip_names": len(all_ids),
            "date_min": min(dates) if dates else "", "date_max": max(dates) if dates else "",
            "unreadable": errs}


def check_extraction(top, root, by_rel, outdir):
    hdr("CHECK 3: BUNDLE MANIFEST vs EXTRACTED FOLDER")
    results, n_ok, n_bad = [], 0, 0
    for z in top:
        rel = str(z.relative_to(root))
        folder = z.relative_to(root).parts[0]
        target_dir = z.parent / z.stem
        entries, err = zip_entries(z)
        if err:
            results.append({"bundle": rel, "status": "unreadable_zip", "detail": err})
            n_bad += 1
            continue

        zip_names = {Path(e["name"]).name: e["size"] for e in entries}
        if not target_dir.is_dir():
            results.append({"bundle": rel, "status": "no_extracted_folder",
                            "n_in_zip": len(zip_names), "n_on_disk": 0})
            n_bad += 1
            continue

        disk = {p.name: p.stat().st_size for p in target_dir.rglob("*")
                if p.is_file() and not is_noise(p)}
        missing = sorted(set(zip_names) - set(disk))
        extra = sorted(set(disk) - set(zip_names))
        size_mismatch = sorted(n for n in set(zip_names) & set(disk)
                               if zip_names[n] != disk[n])
        status = "complete" if not (missing or size_mismatch) else "incomplete"
        if status == "complete":
            n_ok += 1
        else:
            n_bad += 1
        results.append({
            "bundle": rel, "folder": folder, "status": status,
            "n_in_zip": len(zip_names), "n_on_disk": len(disk),
            "missing_from_disk": missing[:20], "n_missing": len(missing),
            "extra_on_disk": extra[:20], "n_extra": len(extra),
            "size_mismatch": size_mismatch[:20], "n_size_mismatch": len(size_mismatch),
        })

    print(f"bundles checked   {len(top)}")
    print(f"complete          {n_ok}")
    print(f"problems          {n_bad}")
    for r in results:
        if r["status"] != "complete":
            print(f"\n  [{r['status']}] {r['bundle']}")
            if r.get("n_missing"):
                print(f"     missing from disk ({r['n_missing']}): {r['missing_from_disk']}")
            if r.get("n_size_mismatch"):
                print(f"     size mismatch ({r['n_size_mismatch']}): {r['size_mismatch']}")
            if r.get("n_extra"):
                print(f"     extra on disk ({r['n_extra']}): {r['extra_on_disk'][:5]}")
    if n_bad == 0:
        print("\n  every bundle fully extracted, all sizes match")
    return results


# ---------------------------------------------------------------------- CHECK 2
def parse_md5_file(path):
    """Return list of (hash, target_string). Handles GNU and BSD formats."""
    out = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = RE_MD5_BSD.match(line)
        if m:
            out.append((m.group(2).lower(), m.group(1).strip()))
            continue
        m = RE_MD5_GNU.match(line)
        if m:
            out.append((m.group(1).lower(), m.group(2).strip()))
    return out


def check_md5(root, by_rel, by_base, outdir):
    hdr("CHECK 2: MD5SUM VERIFICATION")
    sumfiles = sorted(p for p in root.rglob("*.md5sum") if not is_noise(p))
    print(f"checksum files: {len(sumfiles)}")

    rows = []
    verdicts = Counter()
    cache = {}
    covered = set()

    for sf in sumfiles:
        entries = parse_md5_file(sf)
        if not entries:
            verdicts["unparseable_file"] += 1
            rows.append({"md5_file": str(sf.relative_to(root)), "target": "",
                         "expected": "", "actual": "", "verdict": "UNPARSEABLE"})
            continue
        for expected, target in entries:
            tnorm = target.replace("\\", "/").lstrip("./")
            hit = None
            if tnorm in by_rel:
                hit = by_rel[tnorm]
            else:
                cands = by_base.get(Path(tnorm).name, [])
                if len(cands) == 1:
                    hit = cands[0]
                elif len(cands) > 1:
                    same = [c for c in cands
                            if c.relative_to(root).parts[0] == sf.relative_to(root).parts[0]]
                    hit = same[0] if len(same) == 1 else None
            if hit is None:
                verdicts["MISSING_TARGET"] += 1
                rows.append({"md5_file": str(sf.relative_to(root)), "target": target,
                             "expected": expected, "actual": "", "verdict": "MISSING_TARGET"})
                continue
            key = str(hit)
            if key not in cache:
                try:
                    cache[key] = md5_of(hit)
                except OSError as e:
                    cache[key] = f"ERR:{e}"
            actual = cache[key]
            v = "OK" if actual == expected else "MISMATCH"
            verdicts[v] += 1
            covered.add(str(hit.relative_to(root)))
            rows.append({"md5_file": str(sf.relative_to(root)),
                         "target": str(hit.relative_to(root)),
                         "expected": expected, "actual": actual, "verdict": v})

    print("\nverdicts:")
    for v, n in verdicts.most_common():
        print(f"  {v:<16} {n}")

    bad = [r for r in rows if r["verdict"] == "MISMATCH"]
    if bad:
        print(f"\nMISMATCHES ({len(bad)}) -- content differs from recorded checksum:")
        for r in bad[:25]:
            print(f"  {r['target']}")
    miss = [r for r in rows if r["verdict"] == "MISSING_TARGET"]
    if miss:
        print(f"\nMISSING TARGETS ({len(miss)}) -- checksum names a file not on disk:")
        for r in miss[:15]:
            print(f"  [{r['md5_file']}] -> {r['target']}")

    # coverage: which real files no checksum touches
    all_files = {r for r in by_rel if not r.endswith(".md5sum")}
    uncovered = sorted(all_files - covered)
    print(f"\ncoverage: {len(covered)} of {len(all_files)} non-checksum files verified "
          f"({100*len(covered)/max(len(all_files),1):.1f}%)")
    if uncovered:
        bf = Counter(Path(u).parts[0] for u in uncovered)
        print(f"unverified files by folder (top 12): {dict(bf.most_common(12))}")

    path = os.path.join(outdir, "md5_verification.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["md5_file", "target", "expected", "actual", "verdict"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")

    return {"n_checksum_files": len(sumfiles), "verdicts": dict(verdicts),
            "n_verified": len(covered), "n_files_total": len(all_files),
            "n_unverified": len(uncovered),
            "mismatches": [r["target"] for r in bad]}


# ---------------------------------------------------------------------- CHECK 4
def check_cross_folder_dupes(root, by_rel, outdir):
    hdr("CHECK 4: CROSS-FOLDER DUPLICATE ASSAYS")
    byhash = defaultdict(list)
    for rel, p in by_rel.items():
        if rel.endswith((".md5sum", ".zip")):
            continue
        try:
            byhash[md5_of(p)].append(rel)
        except OSError:
            continue

    cross = defaultdict(lambda: {"n": 0, "bytes": 0, "examples": []})
    for h, v in byhash.items():
        if len(v) < 2:
            continue
        folders = {r.split("/")[0] for r in v}
        if len(folders) < 2:
            continue
        pair = " <-> ".join(sorted(folders))
        cross[pair]["n"] += 1
        cross[pair]["bytes"] += by_rel[v[0]].stat().st_size * (len(v) - 1)
        if len(cross[pair]["examples"]) < 5:
            cross[pair]["examples"].append(v)

    if not cross:
        print("no byte-identical files span two experiment folders")
        return {}

    print(f"folder pairs sharing identical content: {len(cross)}\n")
    detail = {}
    for pair, st in sorted(cross.items(), key=lambda kv: -kv[1]["n"]):
        a, b = pair.split(" <-> ")
        fa = {r for r in by_rel if r.startswith(a + "/") and not r.endswith((".md5sum", ".zip", ".json"))}
        fb = {r for r in by_rel if r.startswith(b + "/") and not r.endswith((".md5sum", ".zip", ".json"))}
        print(f"{pair}")
        print(f"  identical files          {st['n']}  ({human(st['bytes'])} redundant)")
        print(f"  {a} total data files     {len(fa)}")
        print(f"  {b} total data files     {len(fb)}")
        share = st["n"] / min(len(fa), len(fb)) if min(len(fa), len(fb)) else 0
        print(f"  overlap as share of smaller folder: {100*share:.0f}%")
        print(f"  -> {'SAME ASSAY FILED TWICE' if share > 0.8 else 'PARTIAL overlap; folders hold distinct data beyond the shared files'}")
        for ex in st["examples"][:3]:
            print(f"     {ex}")
        print()
        detail[pair] = {"identical_files": st["n"], "redundant_bytes": st["bytes"],
                        f"{a}_data_files": len(fa), f"{b}_data_files": len(fb),
                        "overlap_share_of_smaller": round(share, 3),
                        "examples": st["examples"][:5]}
    return detail


def main():
    ap = argparse.ArgumentParser(description="Verify the raw archive layer.")
    ap.add_argument("root")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--roster", default=None, help="optional file of subject IDs, one per line")
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root))
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    if args.roster:
        with open(args.roster) as fh:
            C11_ROSTER.clear()
            C11_ROSTER.update(ln.strip() for ln in fh if ln.strip())

    print(f"root: {root}")
    by_rel, by_base = index_tree(root)
    print(f"indexed {len(by_rel)} files")

    top, nested = collect_zips(root)
    print(f"top-level bundles: {len(top)}   nested zips: {len(nested)}")

    r1 = check_nested(nested, root, by_base, outdir)
    r2 = check_md5(root, by_rel, by_base, outdir)
    r3 = check_extraction(top, root, by_rel, outdir)
    r4 = check_cross_folder_dupes(root, by_rel, outdir)

    report = {
        "root": str(root),
        "n_files_indexed": len(by_rel),
        "check1_nested_zips": r1,
        "check2_md5": r2,
        "check3_extraction": r3,
        "check4_cross_folder_duplicates": r4,
    }
    rp = os.path.join(outdir, "raw_layer_report.json")
    with open(rp, "w") as fh:
        json.dump(report, fh, indent=1, default=str)

    hdr("VERDICT")
    print(f"nested zips        {r1.get('n_zips',0)} zips, {r1.get('n_entries',0)} entries, "
          f"{r1.get('basenames_already_on_disk',0)} already on disk")
    v = r2.get("verdicts", {})
    print(f"md5                {v.get('OK',0)} OK, {v.get('MISMATCH',0)} MISMATCH, "
          f"{v.get('MISSING_TARGET',0)} missing target, {r2.get('n_unverified',0)} files uncovered")
    print(f"extraction         {sum(1 for r in r3 if r.get('status')=='complete')}/{len(r3)} bundles complete")
    print(f"cross-folder dupes {len(r4)} folder pairs")
    print(f"\nwrote {rp}")


if __name__ == "__main__":
    main()
