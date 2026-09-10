#!/usr/bin/env python3
"""
audit_csv_headers.py

Opens every CSV in the tree, reads the header row plus a small value sample, and
reports how consistently the data is named. Reads only; writes nothing back.

Answers:

  A. SUBJECT COLUMN. Which exact spelling does each file use? Counts and file
     lists for every variant found (Subject, SUBJECT, Subject ID, SubjectID,
     subject_id, Subj, ID, Participant, ...). Tiered by confidence so a bare
     "ID" is never silently treated as a subject key.
       - files with exactly one subject-like column      -> keyable
       - files with none                                 -> blind spot
       - files with two or more                          -> ambiguous, needs a rule
     Also reports the column POSITION (is Subject always first?) and the VALUE
     FORMAT in that column (3-4 digit, C1Gxxxx, C3x_nnnn, mixed, non-subject).

  B. NOMENCLATURE DRIFT. Every distinct raw header string in the archive, and
     which of them collapse to the same normalized form (case, spaces,
     underscores, BOM, quotes stripped). A collision group like
     {"Test_Phase", "Test Phase", "TEST_PHASE"} is one concept spelled three
     ways: that is the drift, measured.

  C. FILE-LEVEL ANOMALIES. Encoding needed to read, delimiter, BOM present,
     duplicate header names within one file, blank/unnamed columns, header
     with leading or trailing whitespace, zero-row files.

Outputs (to --outdir):
    header_audit.csv        one row per CSV: subject col, spelling, position,
                            value format, n_cols, n_rows, encoding, delimiter
    header_vocabulary.csv   one row per distinct raw header: count, n_files,
                            normalized form, folders it appears in
    header_report.json      machine-readable summary of everything printed

Usage:
    python3 audit_csv_headers.py <root_dir> [--outdir OUT] [--sample-rows 50]

Example:
    python3 audit_csv_headers.py \
      "/Users/rtscott2/Desktop/AI/20260429/BR TFM Project/bedrest_tfm/03_raw_downloads" \
      --outdir ~/Desktop/AI/20260720/20260728/header_audit
"""

import os
import re
import csv
import sys
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

IGNORE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_PREFIXES = ("._", "~$")
IGNORE_PARTS = {"__MACOSX"}

ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

# ---------------------------------------------------------------------------
# Subject-column classification, tiered by confidence.
#   STRONG   the header names a subject outright
#   MEDIUM   plausible subject key, but the word "subject" is absent
#   WEAK     generic identifiers that MIGHT be a subject key or might be a row
#            serial. Reported separately and never counted as keyable on its own.
# ---------------------------------------------------------------------------
STRONG = {
    "subject", "subjectid", "subject id", "subject_id", "subject-id",
    "subjectnumber", "subject number", "subject no", "subject_no",
    "subjectcode", "subject code", "subj", "subjno", "subj_id", "subjid",
    "subject name", "subjectname",
}
MEDIUM = {
    "participant", "participantid", "participant id", "participant_id",
    "volunteer", "volunteerid", "volunteer id", "test subject", "testsubject",
    "crewmember", "crew member", "sub", "sub_id", "subid", "case", "caseid",
}
WEAK = {
    "id", "code", "number", "no", "num", "record", "recordid", "record id",
    "identifier", "uid", "key",
}

# Value-format signatures for whatever sits in the subject column.
RE_NUM34 = re.compile(r"^\d{3,4}$")
RE_C1G = re.compile(r"^C1G\d{3,4}$", re.IGNORECASE)
RE_C3X = re.compile(r"^C3[A-H]_?\d{3,4}$", re.IGNORECASE)
RE_FLOATY = re.compile(r"^\d{3,4}\.0+$")


def is_noise(p: Path):
    return (p.name in IGNORE_NAMES
            or p.name.startswith(IGNORE_PREFIXES)
            or bool(IGNORE_PARTS & set(p.parts)))


def norm(h):
    """Normalized header: strip BOM, quotes, whitespace, collapse separators, lower."""
    s = str(h).replace("\ufeff", "").strip().strip('"').strip("'").strip()
    s = re.sub(r"[\s_\-]+", " ", s).strip().lower()
    return s


def classify_header(h):
    n = norm(h)
    if n in STRONG:
        return "STRONG"
    if n in MEDIUM:
        return "MEDIUM"
    if n in WEAK:
        return "WEAK"
    return ""


def sniff_delim(line):
    counts = {",": line.count(","), ";": line.count(";"), "\t": line.count("\t"), "|": line.count("|")}
    d = max(counts, key=counts.get)
    return d if counts[d] > 0 else ","


def read_head(path, sample_rows):
    """
    Return (header_list, rows, encoding_used, delimiter, bom, error).
    Reads only the first sample_rows+1 lines.
    """
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as fh:
                first = fh.readline()
                if first == "":
                    return [], [], enc, ",", False, "empty_file"
                bom = first.startswith("\ufeff")
                delim = sniff_delim(first)
                fh.seek(0)
                rdr = csv.reader(fh, delimiter=delim)
                try:
                    header = next(rdr)
                except StopIteration:
                    return [], [], enc, delim, bom, "empty_file"
                rows = []
                for i, r in enumerate(rdr):
                    if i >= sample_rows:
                        break
                    rows.append(r)
                return header, rows, enc, delim, bom, ""
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return [], [], enc, ",", False, f"{type(e).__name__}: {str(e)[:120]}"
    return [], [], "?", ",", False, "undecodable"


def value_format(vals):
    """Classify the value shape of a candidate subject column."""
    vs = [str(v).replace("\ufeff", "").strip().strip('"').strip()
          for v in vals if str(v).strip() != ""]
    if not vs:
        return "empty"
    tags = Counter()
    for v in vs:
        if RE_NUM34.match(v):
            tags["num3-4"] += 1
        elif RE_FLOATY.match(v):
            tags["num3-4_as_float"] += 1
        elif RE_C1G.match(v):
            tags["C1Gxxxx"] += 1
        elif RE_C3X.match(v):
            tags["C3x_nnnn"] += 1
        elif re.match(r"^\d+$", v):
            tags["other_numeric"] += 1
        else:
            tags["non_subject"] += 1
    if len(tags) == 1:
        return next(iter(tags))
    return "MIXED(" + ",".join(f"{k}:{c}" for k, c in tags.most_common()) + ")"


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Audit CSV header nomenclature.")
    ap.add_argument("root")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--sample-rows", type=int, default=50)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root))
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    csvs = sorted(p for p in root.rglob("*.csv") if not is_noise(p))
    print(f"root: {root}")
    print(f"csv files: {len(csvs)}")

    per_file = []
    raw_header_count = Counter()
    raw_header_files = defaultdict(set)
    raw_header_folders = defaultdict(set)
    errors = []

    for p in csvs:
        rel = str(p.relative_to(root))
        folder = p.relative_to(root).parts[0]
        header, rows, enc, delim, bom, err = read_head(p, args.sample_rows)
        if err:
            errors.append((rel, err))
            per_file.append({
                "folder": folder, "relpath": rel, "error": err, "encoding": enc,
                "delimiter": "", "bom": int(bom), "n_cols": 0, "n_sample_rows": 0,
                "subject_header_raw": "", "subject_tier": "", "subject_pos": "",
                "subject_value_format": "", "n_strong": 0, "n_medium": 0, "n_weak": 0,
                "dup_headers": "", "blank_headers": 0, "ws_padded_headers": 0,
            })
            continue

        for h in header:
            raw_header_count[h] += 1
            raw_header_files[h].add(rel)
            raw_header_folders[h].add(folder)

        tiers = [(i, h, classify_header(h)) for i, h in enumerate(header)]
        strong = [(i, h) for i, h, t in tiers if t == "STRONG"]
        medium = [(i, h) for i, h, t in tiers if t == "MEDIUM"]
        weak = [(i, h) for i, h, t in tiers if t == "WEAK"]

        pick = (strong or medium or weak)
        pick_tier = "STRONG" if strong else ("MEDIUM" if medium else ("WEAK" if weak else ""))
        if pick:
            idx, hname = pick[0]
            vals = [r[idx] for r in rows if len(r) > idx]
            vfmt = value_format(vals)
        else:
            idx, hname, vfmt = "", "", ""

        dupes = [h for h, c in Counter(norm(x) for x in header).items() if c > 1 and h]
        blanks = sum(1 for h in header if norm(h) == "" or norm(h).startswith("unnamed"))
        wspad = sum(1 for h in header if str(h) != str(h).strip())

        per_file.append({
            "folder": folder, "relpath": rel, "error": "", "encoding": enc,
            "delimiter": {"\t": "TAB"}.get(delim, delim), "bom": int(bom),
            "n_cols": len(header), "n_sample_rows": len(rows),
            "subject_header_raw": hname, "subject_tier": pick_tier,
            "subject_pos": idx, "subject_value_format": vfmt,
            "n_strong": len(strong), "n_medium": len(medium), "n_weak": len(weak),
            "dup_headers": ";".join(dupes), "blank_headers": blanks,
            "ws_padded_headers": wspad,
        })

    ok = [r for r in per_file if not r["error"]]

    # ------------------------------------------------------------------- A
    hdr("A. SUBJECT COLUMN: WHICH SPELLING, AND HOW MANY FILES CAN BE KEYED")

    keyable = [r for r in ok if r["subject_tier"] in ("STRONG", "MEDIUM")]
    weak_only = [r for r in ok if r["subject_tier"] == "WEAK"]
    none = [r for r in ok if r["subject_tier"] == ""]
    ambiguous = [r for r in ok if r["n_strong"] + r["n_medium"] > 1]

    print(f"csvs read successfully        {len(ok)} of {len(csvs)}")
    print(f"  keyable (STRONG or MEDIUM)  {len(keyable)}  ({100*len(keyable)/max(len(ok),1):.1f}%)")
    print(f"  WEAK identifier only        {len(weak_only)}")
    print(f"  no subject-like column      {len(none)}")
    print(f"  two or more subject cols    {len(ambiguous)}  (ambiguous, needs a rule)")

    print("\nexact spellings found, by tier:")
    spell = Counter((r["subject_tier"], r["subject_header_raw"]) for r in ok if r["subject_tier"])
    for tier in ("STRONG", "MEDIUM", "WEAK"):
        sub = [(h, c) for (t, h), c in spell.items() if t == tier]
        if not sub:
            continue
        print(f"\n  [{tier}]")
        for h, c in sorted(sub, key=lambda x: -x[1]):
            shown = repr(h) if (h != h.strip() or "\ufeff" in h) else h
            print(f"    {c:>5}  {shown}")

    print("\nposition of the subject column (0 = first):")
    for pos, c in Counter(r["subject_pos"] for r in keyable).most_common(10):
        print(f"  col {pos:<3} {c:>5}")

    print("\nvalue format inside the subject column:")
    for v, c in Counter(r["subject_value_format"] for r in keyable).most_common(15):
        print(f"  {c:>5}  {v}")

    print("\nsubject-column spelling by folder (folders using more than one spelling):")
    byfolder = defaultdict(Counter)
    for r in ok:
        if r["subject_tier"] in ("STRONG", "MEDIUM"):
            byfolder[r["folder"]][r["subject_header_raw"]] += 1
    multi = {f: c for f, c in byfolder.items() if len(c) > 1}
    if multi:
        for f, c in sorted(multi.items(), key=lambda kv: -len(kv[1])):
            print(f"  {f:<16} {dict(c)}")
    else:
        print("  none: every folder is internally consistent")

    if none:
        print(f"\nfiles with NO subject-like column ({len(none)}), by folder:")
        for f, c in Counter(r["folder"] for r in none).most_common():
            print(f"  {c:>5}  {f}")
        print("  examples:")
        for r in none[:8]:
            print(f"    {r['relpath']}")

    if ambiguous:
        print(f"\nfiles with MORE THAN ONE subject-like column ({len(ambiguous)}):")
        for r in ambiguous[:15]:
            print(f"  {r['relpath']}  (strong={r['n_strong']} medium={r['n_medium']})")

    if weak_only:
        print(f"\nfiles whose only identifier is WEAK ({len(weak_only)}) -- verify before keying:")
        for f, c in Counter(r["folder"] for r in weak_only).most_common(10):
            print(f"  {c:>5}  {f}")
        for r in weak_only[:6]:
            print(f"    {r['relpath']}  header={r['subject_header_raw']!r}  fmt={r['subject_value_format']}")

    # ------------------------------------------------------------------- B
    hdr("B. NOMENCLATURE DRIFT: SAME CONCEPT, DIFFERENT SPELLING")

    print(f"distinct raw header strings across the archive: {len(raw_header_count)}")
    groups = defaultdict(list)
    for h in raw_header_count:
        groups[norm(h)].append(h)
    collisions = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"normalized concepts:                           {len(groups)}")
    print(f"concepts spelled more than one way:            {len(collisions)}")
    drift_cols = sum(raw_header_count[h] for v in collisions.values() for h in v)
    print(f"column instances affected by drift:            {drift_cols}")

    print(f"\ntop {args.top} drift groups (by how many files they touch):")
    ranked = sorted(collisions.items(),
                    key=lambda kv: -sum(len(raw_header_files[h]) for h in kv[1]))
    for nconcept, variants in ranked[:args.top]:
        nfiles = sum(len(raw_header_files[h]) for h in variants)
        print(f"\n  '{nconcept}'  ->  {len(variants)} spellings, {nfiles} files")
        for h in sorted(variants, key=lambda x: -raw_header_count[x]):
            shown = repr(h) if (h != h.strip() or "\ufeff" in h) else h
            fol = sorted(raw_header_folders[h])
            print(f"      {raw_header_count[h]:>5}  {shown:<40} {fol[:4]}"
                  + (f" +{len(fol)-4}" if len(fol) > 4 else ""))

    print("\n  (every file carrying each variant is listed in drift_examples.csv)")

    print(f"\nmost common headers overall (top {args.top}):")
    for h, c in raw_header_count.most_common(args.top):
        print(f"  {c:>5}  in {len(raw_header_files[h]):>4} files  {h}")

    # ------------------------------------------------------------------- C
    hdr("C. FILE-LEVEL ANOMALIES")
    print("encoding required to read:")
    for e, c in Counter(r["encoding"] for r in ok).most_common():
        print(f"  {c:>5}  {e}")
    print("\ndelimiter:")
    for d, c in Counter(r["delimiter"] for r in ok).most_common():
        print(f"  {c:>5}  {d}")
    print(f"\nfiles with a BOM:                 {sum(r['bom'] for r in ok)}")
    print(f"files with whitespace-padded headers: {sum(1 for r in ok if r['ws_padded_headers'])}")
    print(f"files with blank/unnamed columns:     {sum(1 for r in ok if r['blank_headers'])}")
    print(f"files with duplicate header names:    {sum(1 for r in ok if r['dup_headers'])}")
    zero = [r for r in ok if r["n_sample_rows"] == 0]
    print(f"files with a header but zero data rows: {len(zero)}")
    for r in zero[:10]:
        print(f"    {r['relpath']}")

    ncols = sorted(r["n_cols"] for r in ok)
    if ncols:
        print(f"\ncolumn count: min {ncols[0]}, median {ncols[len(ncols)//2]}, max {ncols[-1]}")
        widest = sorted(ok, key=lambda r: -r["n_cols"])[:5]
        print("widest files:")
        for r in widest:
            print(f"  {r['n_cols']:>5} cols  {r['relpath']}")

    if errors:
        print(f"\nunreadable csvs: {len(errors)}")
        for rel, e in errors[:15]:
            print(f"  {rel}: {e}")

    # ---------------------------------------------------------------- write
    hdr("WROTE")
    f1 = os.path.join(outdir, "header_audit.csv")
    with open(f1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_file[0].keys()))
        w.writeheader()
        w.writerows(per_file)
    print(f1)

    f2 = os.path.join(outdir, "header_vocabulary.csv")
    with open(f2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["raw_header", "normalized", "n_occurrences", "n_files",
                    "n_folders", "folders", "is_drift_variant", "example_files"])
        for h, c in raw_header_count.most_common():
            n = norm(h)
            ex = sorted(raw_header_files[h])[:3]
            w.writerow([h, n, c, len(raw_header_files[h]), len(raw_header_folders[h]),
                        ";".join(sorted(raw_header_folders[h])), int(len(groups[n]) > 1),
                        " | ".join(ex)])
    print(f2)

    # Every file that carries a drift variant, one row per (variant, file), so a
    # drift group can be opened and compared side by side without re-scanning.
    f4 = os.path.join(outdir, "drift_examples.csv")
    with open(f4, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["normalized_concept", "n_variants", "raw_header", "variant_rank",
                    "n_files_with_this_variant", "folder", "relpath"])
        for nconcept, variants in ranked:
            ordered = sorted(variants, key=lambda x: -raw_header_count[x])
            for rank, h in enumerate(ordered, 1):
                for rel in sorted(raw_header_files[h]):
                    w.writerow([nconcept, len(variants), h, rank,
                                len(raw_header_files[h]), rel.split("/")[0], rel])
    print(f4)

    report = {
        "root": str(root),
        "n_csv": len(csvs),
        "n_readable": len(ok),
        "subject_column": {
            "keyable": len(keyable), "weak_only": len(weak_only),
            "none": len(none), "ambiguous": len(ambiguous),
            "spellings": {f"{t}|{h}": c for (t, h), c in spell.items()},
            "value_formats": dict(Counter(r["subject_value_format"] for r in keyable)),
            "folders_with_multiple_spellings": {f: dict(c) for f, c in multi.items()},
            "files_without_subject_column": [r["relpath"] for r in none],
        },
        "nomenclature": {
            "distinct_raw_headers": len(raw_header_count),
            "normalized_concepts": len(groups),
            "concepts_with_multiple_spellings": len(collisions),
            "column_instances_affected": drift_cols,
            "drift_groups": {k: sorted(v) for k, v in list(ranked[:100])},
        },
        "anomalies": {
            "encodings": dict(Counter(r["encoding"] for r in ok)),
            "delimiters": dict(Counter(r["delimiter"] for r in ok)),
            "bom_files": sum(r["bom"] for r in ok),
            "ws_padded_header_files": sum(1 for r in ok if r["ws_padded_headers"]),
            "blank_header_files": sum(1 for r in ok if r["blank_headers"]),
            "dup_header_files": sum(1 for r in ok if r["dup_headers"]),
            "zero_row_files": len(zero),
            "unreadable": errors,
        },
    }
    f3 = os.path.join(outdir, "header_report.json")
    with open(f3, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f3)


if __name__ == "__main__":
    main()
