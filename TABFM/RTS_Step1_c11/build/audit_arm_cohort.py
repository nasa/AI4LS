#!/usr/bin/env python3
"""
audit_arm_cohort.py

Finds every place an experimental ARM, COHORT, or INTERVENTION label is recorded
anywhere in the raw archive, and measures how consistently it is recorded.

Arm assignment is the one variable that must never enter a feature matrix, so
this maps its full surface area: columns, values, filenames, and bundle names.

  1. ARM-BEARING COLUMNS. Every column whose normalized header names a group,
     arm, cohort, treatment, condition, randomization, dose, or supplement.
     Reports the exact spellings found and how many files carry each.
     Guards against false positives: a column named "Group" that holds muscle
     groups or numeric codes is flagged rather than counted as an arm.

  2. VALUE VOCABULARY AND DRIFT. Every distinct arm value in the archive, then
     grouped by normalized form so CONTROL / Control / Ctrl / CTRL collapse to
     one concept with four spellings. This is the arm-level equivalent of the
     header drift audit.

  3. INTERVENTION LEXICON. Which trial arms exist across the whole archive,
     matched against a lexicon covering exercise (control, exercise, flywheel,
     resistance, aerobic, treadmill, cycle), pharmacological (testosterone,
     placebo, supplement, dose), and artificial gravity. Reports each folder's
     intervention vocabulary so multi-axis designs are visible: a subject can
     carry an exercise arm AND a supplementation arm.

  4. PER-SUBJECT CONFLICTS. For every subject, all arm labels found across all
     files and folders. A subject carrying two or more distinct normalized
     labels is a conflict. Reported overall and restricted to the C11 roster,
     with the authoritative source (BEDREST_FTT) shown alongside so each
     disagreement is legible.

  5. ARM ENCODED OUTSIDE COLUMNS. Filenames and bundle/folder names containing
     an arm token. These are the leakage path that column-level rules miss,
     because the builder folds the filename into every column name it creates.

Outputs (to --outdir):
    arm_columns.csv        one row per (file, arm column): header, n values, sample
    arm_values.csv         one row per distinct raw value: count, files, folders
    arm_by_subject.csv     one row per (subject, folder, source_file, raw label)
    arm_conflicts.csv      subjects carrying more than one normalized label
    arm_report.json        machine-readable summary

Usage:
    python3 audit_arm_cohort.py <root_dir> [--outdir OUT] [--roster FILE]

Example:
    python3 audit_arm_cohort.py \
      "/Users/rtscott2/Desktop/AI/20260429/BR TFM Project/bedrest_tfm/03_raw_downloads" \
      --outdir ~/Desktop/AI/20260720/20260728/arm_audit
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
ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

C11_ROSTER = {
    "5210", "5297", "5803", "6213", "6319", "6546", "6791", "6947", "7574", "7750", "8936",
    "5159", "5160", "5188", "5627", "6403", "6611", "6877", "7036", "7152", "7326", "7350",
    "7707", "8010", "8072", "8713", "8784", "8837", "9667", "9682", "9713",
    "5673", "6187", "6464", "7548", "8177", "8179", "9023", "9793", "9633", "6559", "9217",
    "5016", "8930", "9011", "9751",
}

# Headers that may name an arm. Compared in normalized form (lower, separators
# collapsed to single spaces, BOM and quotes stripped).
ARM_HEADERS = {
    "group", "groupname", "group name", "grouplabel", "group label", "grp",
    "arm", "study arm", "studyarm", "treatment arm", "trial arm",
    "cohort", "cohortname", "cohort name",
    "treatment", "treatmentgroup", "treatment group", "tx",
    "condition", "assignment", "randomization", "randomisation", "randomized group",
    "intervention", "protocol group", "subject group", "test group",
    "dose", "dosage", "supplement", "supplementation", "drug", "agent",
    "placebo", "regimen",
}

# Subject-column forms (from the header audit: these are the ones that occur).
SUBJECT_HEADERS = {
    "subject", "subjectid", "subject id", "subject number", "subjectnumber",
    "subj", "subjno", "subjid", "subject code", "subjectcode",
}

# Intervention lexicon, for classifying what an arm value actually means.
LEXICON = {
    "control": ["control", "ctrl", "ctl", "con", "cont", "sedentary", "sed", "no exercise"],
    "exercise": ["exercise", "exer", "ex", "exr", "trained", "training", "active"],
    "flywheel": ["flywheel", "fly wheel", "fw", "iras", "irats"],
    "resistance": ["resistance", "resist", "rt", "strength", "weight"],
    "aerobic": ["aerobic", "aero", "cardio", "cycle", "cycling", "treadmill", "supine cycle"],
    "combined": ["ex&t", "ex and t", "exercise and testosterone", "combined", "comb"],
    "testosterone": ["testosterone", "test supp", "testost", "androgen", " t ", "t only", "t-only"],
    "placebo": ["placebo", "plac", "pbo", "sham", "vehicle"],
    "artificial_gravity": ["artificial gravity", "ag", "centrifuge", "cent"],
    "nutrition": ["diet", "protein", "nutrition", "supplement", "amino"],
    "hdt": ["hdt", "head down", "bed rest", "bedrest", "br"],
}

# Tokens that mark an arm inside a FILENAME or BUNDLE name.
NAME_TOKENS = {
    "control", "ctrl", "exercise", "exer", "flywheel", "testosterone", "placebo",
    "treatment", "treated", "untreated", "sham", "resistive", "resistance",
    "aerobic", "sedentary", "intervention", "arm", "cohort", "group",
}

RE_NUM34 = re.compile(r"^\d{3,4}$")
RE_C1G = re.compile(r"^C1G\d{3,4}$", re.IGNORECASE)
RE_FLOATY = re.compile(r"^(\d{3,4})\.0+$")


def is_noise(p: Path):
    return (p.name in IGNORE_NAMES
            or p.name.startswith(IGNORE_PREFIXES)
            or bool(IGNORE_PARTS & set(p.parts)))


def norm_h(h):
    s = str(h).replace("\ufeff", "").strip().strip('"').strip("'").strip()
    return re.sub(r"[\s_\-]+", " ", s).strip().lower()


def norm_v(v):
    """Normalize an arm VALUE for concept grouping: lower, strip punctuation/space."""
    s = str(v).replace("\ufeff", "").strip().strip('"').strip("'").strip()
    s = re.sub(r"[\s_\-\.]+", " ", s).strip().lower()
    return s


def canon_subject(v):
    s = str(v).replace("\ufeff", "").strip().strip('"').strip()
    if s.upper().startswith("C1G"):
        return s.upper()
    m = RE_FLOATY.match(s)
    if m:
        return m.group(1)
    return s


def is_subject_value(v):
    s = canon_subject(v)
    return bool(RE_NUM34.match(s) or RE_C1G.match(s))


def classify_value(v):
    """Map an arm value to intervention categories via the lexicon."""
    n = " " + norm_v(v) + " "
    hits = []
    for cat, toks in LEXICON.items():
        for t in toks:
            tt = t if t.startswith(" ") else " " + t
            tt = tt if tt.endswith(" ") else tt + " "
            if tt in n or norm_v(v) == t.strip():
                hits.append(cat)
                break
    return hits


def read_csv_rows(path, max_rows=100000):
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as fh:
                rdr = csv.reader(fh)
                try:
                    header = next(rdr)
                except StopIteration:
                    return None, [], enc
                rows = []
                for i, r in enumerate(rdr):
                    if i >= max_rows:
                        break
                    rows.append(r)
                return header, rows, enc
        except UnicodeDecodeError:
            continue
        except Exception:
            return None, [], enc
    return None, [], "?"


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="Audit arm/cohort labeling across the archive.")
    ap.add_argument("root")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--roster", default=None)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root))
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")
    roster = set(C11_ROSTER)
    if args.roster:
        with open(args.roster) as fh:
            roster = {ln.strip() for ln in fh if ln.strip()}

    csvs = sorted(p for p in root.rglob("*.csv") if not is_noise(p))
    print(f"root: {root}")
    print(f"csv files: {len(csvs)}")

    col_rows = []                       # per (file, arm column)
    val_count = Counter()               # raw value -> occurrences
    val_files = defaultdict(set)
    val_folders = defaultdict(set)
    val_headers = defaultdict(set)
    subj_rows = []                      # (subject, folder, file, header, raw)
    suspicious = []                     # arm-named columns holding non-arm content

    for p in csvs:
        rel = str(p.relative_to(root))
        folder = p.relative_to(root).parts[0]
        header, rows, enc = read_csv_rows(p)
        if not header:
            continue

        nh = [norm_h(h) for h in header]
        arm_idx = [i for i, h in enumerate(nh) if h in ARM_HEADERS]
        if not arm_idx:
            continue
        subj_idx = next((i for i, h in enumerate(nh) if h in SUBJECT_HEADERS), None)

        for ai in arm_idx:
            vals = [r[ai] for r in rows if len(r) > ai and str(r[ai]).strip() != ""]
            if not vals:
                continue
            uniq = Counter(norm_v(v) for v in vals)
            # False-positive guard: a "Group"/"Dose" column that is purely numeric,
            # or that has as many distinct values as rows, is not an arm label.
            numericish = sum(1 for v in uniq if re.match(r"^[\d\.]+$", v))
            looks_numeric = numericish == len(uniq)
            too_many = len(uniq) > max(12, 0.5 * len(vals))
            flag = ""
            if looks_numeric:
                flag = "numeric_values"
            elif too_many:
                flag = "high_cardinality"

            col_rows.append({
                "folder": folder, "relpath": rel, "arm_header_raw": header[ai],
                "arm_header_norm": nh[ai], "n_rows": len(vals),
                "n_distinct": len(uniq), "has_subject_col": int(subj_idx is not None),
                "flag": flag,
                "sample_values": " | ".join(list(dict.fromkeys(
                    str(v).strip() for v in vals))[:8]),
            })
            if flag:
                suspicious.append((rel, header[ai], flag, list(uniq)[:6]))
                continue

            for r in rows:
                if len(r) <= ai:
                    continue
                raw = str(r[ai]).strip()
                if raw == "":
                    continue
                val_count[raw] += 1
                val_files[raw].add(rel)
                val_folders[raw].add(folder)
                val_headers[raw].add(header[ai])
                if subj_idx is not None and len(r) > subj_idx and is_subject_value(r[subj_idx]):
                    subj_rows.append({
                        "subject": canon_subject(r[subj_idx]), "folder": folder,
                        "source_file": rel, "arm_header": header[ai], "arm_raw": raw,
                        "arm_norm": norm_v(raw),
                    })

    # ------------------------------------------------------------------ 1
    hdr("1. ARM-BEARING COLUMNS")
    print(f"files carrying at least one arm-named column: "
          f"{len({r['relpath'] for r in col_rows})}")
    print(f"(file, column) pairs: {len(col_rows)}")
    print("\nexact header spellings:")
    for (raw, nrm), c in Counter((r["arm_header_raw"], r["arm_header_norm"])
                                 for r in col_rows).most_common():
        print(f"  {c:>5}  {raw:<24} -> '{nrm}'")

    print("\nby folder:")
    for f, c in Counter(r["folder"] for r in col_rows).most_common():
        heads = sorted({r["arm_header_raw"] for r in col_rows if r["folder"] == f})
        print(f"  {c:>5}  {f:<16} {heads}")

    if suspicious:
        print(f"\nEXCLUDED as not-an-arm ({len(suspicious)} columns): numeric or high-cardinality")
        for rel, h, flag, sample in suspicious[:12]:
            print(f"  [{flag}] {h!r} in {rel}")
            print(f"        sample: {sample}")

    # ------------------------------------------------------------------ 2
    hdr("2. ARM VALUE VOCABULARY AND DRIFT")
    print(f"distinct raw arm values: {len(val_count)}")
    groups = defaultdict(list)
    for v in val_count:
        groups[norm_v(v)].append(v)
    print(f"normalized concepts:     {len(groups)}")
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"concepts with >1 spelling: {len(multi)}")

    print(f"\nall arm values (top {args.top} by occurrence):")
    print(f"  {'n':>7}  {'files':>6}  value")
    for v, c in val_count.most_common(args.top):
        cats = classify_value(v)
        tag = ("  [" + ",".join(cats) + "]") if cats else "  [UNMATCHED]"
        print(f"  {c:>7}  {len(val_files[v]):>6}  {v!r}{tag}")

    if multi:
        print("\nspelling drift within a single concept:")
        for k, vs in sorted(multi.items(), key=lambda kv: -sum(val_count[x] for x in kv[1])):
            print(f"  '{k}' -> {sorted(vs)}")

    # concept-level rollup across the lexicon
    print("\nintervention concepts present, after normalizing spelling:")
    concept = defaultdict(Counter)
    for v, c in val_count.items():
        for cat in (classify_value(v) or ["UNCLASSIFIED"]):
            concept[cat][norm_v(v)] += c
    for cat, spellings in sorted(concept.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(spellings.values())
        print(f"  {tot:>7}  {cat:<18} {dict(spellings.most_common(6))}")

    # ------------------------------------------------------------------ 3
    hdr("3. INTERVENTION VOCABULARY BY FOLDER (multi-axis designs)")
    byfolder = defaultdict(Counter)
    for v, c in val_count.items():
        for f in val_folders[v]:
            byfolder[f][norm_v(v)] += c
    for f in sorted(byfolder):
        vocab = byfolder[f]
        cats = sorted({c for v in vocab for c in classify_value(v)})
        print(f"\n  {f}")
        print(f"    values     {dict(vocab.most_common(10))}")
        print(f"    concepts   {cats}")

    # ------------------------------------------------------------------ 4
    hdr("4. PER-SUBJECT ARM CONFLICTS")
    bysubj = defaultdict(set)
    bysubj_detail = defaultdict(list)
    for r in subj_rows:
        bysubj[r["subject"]].add(r["arm_norm"])
        bysubj_detail[r["subject"]].append(r)

    print(f"subjects with at least one arm label: {len(bysubj)}")
    conflicts = {s: v for s, v in bysubj.items() if len(v) > 1}
    print(f"subjects with MORE THAN ONE distinct label: {len(conflicts)}")

    roster_lab = {s: v for s, v in bysubj.items() if s in roster}
    roster_conf = {s: v for s, v in conflicts.items() if s in roster}
    print(f"\nC11 roster subjects with an arm label:  {len(roster_lab)} of {len(roster)}")
    print(f"C11 roster subjects with a CONFLICT:    {len(roster_conf)}")

    if roster_conf:
        print("\nC11 conflicts (subject -> label: sources):")
        for s in sorted(roster_conf)[:25]:
            print(f"\n  {s}")
            per = defaultdict(set)
            for r in bysubj_detail[s]:
                per[r["arm_norm"]].add(f"{r['folder']}/{r['arm_header']}")
            for lab in sorted(per):
                print(f"     {lab:<24} {sorted(per[lab])}")

    # authoritative source comparison
    auth = "BEDREST_FTT"
    auth_map = {}
    for r in subj_rows:
        if r["folder"] == auth:
            auth_map.setdefault(r["subject"], set()).add(r["arm_norm"])
    print(f"\nauthoritative source {auth}: {len(auth_map)} subjects labeled")
    if auth_map:
        disagree = []
        for s, labs in auth_map.items():
            other = {r["arm_norm"] for r in bysubj_detail[s] if r["folder"] != auth}
            if other and not (other & labs):
                disagree.append((s, sorted(labs), sorted(other)))
        print(f"subjects where another folder disagrees with {auth}: {len(disagree)}")
        for s, a, o in disagree[:20]:
            print(f"  {s}: {auth}={a}  other={o}")

    # ------------------------------------------------------------------ 5
    hdr("5. ARM ENCODED OUTSIDE COLUMNS (filenames and bundle names)")
    fn_hits, bundle_hits = [], set()
    for p in csvs:
        rel = str(p.relative_to(root))
        parts = p.relative_to(root).parts
        stem_toks = set(re.split(r"[_\-\s\.]+", p.stem.lower()))
        hit = sorted(stem_toks & NAME_TOKENS)
        if hit:
            fn_hits.append((rel, hit))
        if len(parts) > 1:
            btoks = set(re.split(r"[_\-\s\.]+", parts[1].lower()))
            bh = sorted(btoks & NAME_TOKENS)
            if bh:
                bundle_hits.add((parts[0], parts[1], tuple(bh)))

    print(f"CSV filenames containing an arm token: {len(fn_hits)}")
    for f, c in Counter(r.split("/")[0] for r, _ in fn_hits).most_common():
        print(f"  {c:>5}  {f}")
    print("\n  examples:")
    for rel, hit in fn_hits[:15]:
        print(f"    {hit}  {rel}")

    print(f"\nBUNDLE (extracted folder) names containing an arm token: {len(bundle_hits)}")
    for folder, bundle, hit in sorted(bundle_hits):
        print(f"    {list(hit)}  {folder}/{bundle}")

    print("\n  NOTE: the builder names columns measure::filetag::timepoint, so an arm")
    print("  token in a FILENAME propagates into every column derived from that file.")

    # ---------------------------------------------------------------- write
    hdr("WROTE")
    f1 = os.path.join(outdir, "arm_columns.csv")
    with open(f1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(col_rows[0].keys()) if col_rows else
                           ["folder", "relpath", "arm_header_raw", "arm_header_norm",
                            "n_rows", "n_distinct", "has_subject_col", "flag", "sample_values"])
        w.writeheader()
        w.writerows(col_rows)
    print(f1)

    f2 = os.path.join(outdir, "arm_values.csv")
    with open(f2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["raw_value", "normalized", "concepts", "n_occurrences",
                    "n_files", "n_folders", "folders", "headers"])
        for v, c in val_count.most_common():
            w.writerow([v, norm_v(v), ";".join(classify_value(v)) or "UNCLASSIFIED",
                        c, len(val_files[v]), len(val_folders[v]),
                        ";".join(sorted(val_folders[v])), ";".join(sorted(val_headers[v]))])
    print(f2)

    f3 = os.path.join(outdir, "arm_by_subject.csv")
    with open(f3, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject", "folder", "source_file",
                                           "arm_header", "arm_raw", "arm_norm"])
        w.writeheader()
        for r in subj_rows:
            w.writerow(r)
    print(f3)

    f4 = os.path.join(outdir, "arm_conflicts.csv")
    with open(f4, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "on_c11_roster", "n_distinct_labels", "labels", "sources"])
        for s, labs in sorted(conflicts.items()):
            srcs = sorted({f"{r['folder']}/{r['arm_header']}={r['arm_raw']}"
                           for r in bysubj_detail[s]})
            w.writerow([s, int(s in roster), len(labs), ";".join(sorted(labs)), ";".join(srcs)])
    print(f4)

    report = {
        "root": str(root), "n_csv": len(csvs),
        "arm_columns": {
            "files_with_arm_column": len({r["relpath"] for r in col_rows}),
            "header_spellings": dict(Counter(r["arm_header_raw"] for r in col_rows)),
            "by_folder": dict(Counter(r["folder"] for r in col_rows)),
            "excluded_not_arm": len(suspicious),
        },
        "values": {
            "distinct_raw": len(val_count), "normalized_concepts": len(groups),
            "concepts_with_multiple_spellings": {k: sorted(v) for k, v in multi.items()},
            "top_values": {v: c for v, c in val_count.most_common(60)},
            "by_concept": {k: dict(v) for k, v in concept.items()},
        },
        "subjects": {
            "labeled": len(bysubj), "conflicts": len(conflicts),
            "roster_labeled": len(roster_lab), "roster_conflicts": len(roster_conf),
            "authoritative_source": auth, "authoritative_labeled": len(auth_map),
        },
        "name_encoding": {
            "csv_filenames_with_arm_token": len(fn_hits),
            "bundles_with_arm_token": [f"{a}/{b}" for a, b, _ in sorted(bundle_hits)],
        },
    }
    f5 = os.path.join(outdir, "arm_report.json")
    with open(f5, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f5)


if __name__ == "__main__":
    main()
