"""
build_master_v6_1_inclusive.py

v6.1 change from v6 (feature-matrix cleanup, so the WIDE master is measurements
only and safe to hand to modeling):
    1. Non-measurement columns are excluded from the wide feature matrix:
       arm/group (GroupName is arm-assignment leakage; the project forbids arm as
       a predictor), Date and Campaign (provenance, not biomarkers), and the
       static descriptors Age/Sex/Gender/Height. Weight is deliberately KEPT:
       body mass is a real longitudinal measurement, not a static descriptor.
       Subject covariates and arm are sourced from the authoritative C11 cohort
       composition file, NOT re-derived from these noisy measurement-file columns
       (Age/Height carry within-subject inconsistency; GroupName carries up to 3
       conflicting arm labels per subject). Removing them cut ~3,900 junk columns
       from the C11 master (11,453 -> 7,545), residual leakage 0.
    2. The known-scrambled BRSMVJ Post2 file is excluded entirely (wide and long).
       Any OTHER file whose name contains POST2 (e.g. MR080G POST2) is KEPT and
       flagged at run time, since only BRSMVJ Post2 is the corrupted file.
    3. The wide master is the clean analysis matrix. The long table remains the
       complete traceable substrate: it retains every field (including covariates
       and provenance) so nothing is lost and any value stays traceable.

v6 change from v5 (single fix; values, subject counts, and the long table are
unchanged):
    Phantom timepoint-as-measure columns removed. In v5, widen_file excluded
    only the minimal observation key from the melted measures, so any timepoint
    column a file carried BEYOND that key (e.g. Test_Phase and BR_Day riding
    alongside a Session key) was swept in as a phantom measure: a column named
    Test_Phase::...  holding the string POST_TEST, which SHAP would then score
    as if the name of the timepoint predicted the outcome. Same failure mode as
    the arm leakage already removed. v6 excludes EVERY declared timepoint column
    from measures and folds them all into the column name, matching what the long
    table already did. The wide reshape and the long export now share one
    definition of a timepoint column (declared_timepoint_cols). Removes the ~1,500
    phantom columns from the C11 master; drops no real measure and changes no
    value; the long table is byte-identical to v5.

v3 INCLUSIVE variant. Paired A/B test against build_master_v3_longitudinal.py.

  This variant does NOT apply a longitudinal gate. It keeps every usable file,
  whether a subject was measured once (cross-sectional) or many times
  (longitudinal). Longitudinal shape is recorded as metadata only.

  Its twin, build_master_v3_longitudinal.py, is identical except it re-adds
  James v1's rule: keep ONLY files where subjects appear more than once, and
  drop cross-sectional files. The two scripts differ by exactly one constant
  (LONGITUDINAL_ONLY) and the gate it controls, so any difference in output is
  attributable solely to that choice.

Evolution of James Casaletto's build_master.py (v1).

Purpose (unchanged from v1's intent):
    Walk a raw data tree, find every subject-keyed CSV, and assemble ONE
    master table keyed by subject, with every value traceable back to the
    measure, the source file, and the observation (timepoint) it came from.
    Campaign (C1/C3/C11) and science-question selection happen LATER, as
    downstream filters on this master. The master itself is built to be
    maximally inclusive and fully traceable.

What changed from v1, and why (from the peer review):

  STRUCTURAL
  1. Subject key is normalized before use. v1 matched the header exactly as
     'Subject', which silently dropped files headed 'SUBJECT', 'Subject ID',
     or BOM-prefixed. v2 normalizes headers (strip BOM/quotes/space, lower)
     and matches a small allowed set, so those files survive.
  2. Subject VALUES are validated. Real subjects are 3-4 digit numeric IDs
     (C3/C11) or the C1Gxxxx form (C1). Non-subject artifacts that leak into
     subject columns (SUP, UM, C3A, group codes A1-L2, malformed CIG0003)
     are excluded from becoming rows and logged, not silently merged.
  3. The master is maximally inclusive. v1 hard-gated to longitudinal-only
     files (dropping every single-timepoint measure). v2 keeps BOTH
     cross-sectional and longitudinal files; longitudinal vs cross-sectional
     is recorded as metadata for the downstream science-question step to use.
  4. The observation key is carried into the column name. v1 computed the
     key then discarded it, so two timepoints of one measure would collide.
     v2 names every column measure::file::obskey so each timepoint is a
     distinct, self-describing column.

  BUGS FIXED
  5. build_master_table was a stub (computed df_sample, returned empty). Now
     implemented as a real per-file pivot + outer-join merge.
  6. Column namescheme used a stale loop variable `f` instead of `file`.
  7. get_unique_identifier_files read a global instead of its parameter.
  8. Minimal-key search now truly minimal + deterministic (sorted columns,
     stop at smallest working combo).
  9. Duplicate `import pandas` removed.

  AUDITABILITY
  10. Writes master to disk AND writes a manifest describing every file:
      kept/excluded, why, subject count, obs key, longitudinal flag,
      rows-per-subject shape. The master inherits many inclusion decisions;
      the manifest makes them inspectable and reproducible.

Usage:
    python3 build_master_v6_inclusive.py <root_dir> [--outdir OUT] [--max-combo N]
"""

import os
import sys
import re
import json
import argparse
import itertools
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------------
# Subject key handling
# ----------------------------------------------------------------------------

# ---- A/B TOGGLE (the ONLY functional difference between the two v3 scripts) ---
# False  -> inclusive: keep both cross-sectional and longitudinal files.
# True   -> longitudinal-only: drop files where every subject appears once.
LONGITUDINAL_ONLY = False
# ------------------------------------------------------------------------------

# Density cap. If the resolved observation key yields more than this many
# timepoints per subject, the file is densely longitudinal (e.g. daily intake
# across 70+ bed-rest days). Wide-pivoting it would add thousands of columns,
# and how to summarize it (mean, AUC, change-from-baseline) is a science
# decision, not build-time plumbing. Such files are flagged 'dense_longitudinal'
# and held OUT of the wide master, logged so the science step can decide.
# Overridable with --max-density.
MAX_TIMEPOINTS_PER_SUBJECT = 30

# Built-in Campaign 11 subject roster (46 subjects, verified). Use --c11 to
# filter the master to exactly these subjects. Without --c11 (or --roster), the
# master stays campaign-agnostic and includes every subject found (C1/C3/C11),
# per the maximally-inclusive design; the roster filter is a downstream select.
C11_ROSTER = {
    "5210", "5297", "5803", "6213", "6319", "6546", "6791", "6947", "7574", "7750", "8936",
    "5159", "5160", "5188", "5627", "6403", "6611", "6877", "7036", "7152", "7326", "7350",
    "7707", "8010", "8072", "8713", "8784", "8837", "9667", "9682", "9713",
    "5673", "6187", "6464", "7548", "8177", "8179", "9023", "9793", "9633", "6559", "9217",
    "5016", "8930", "9011", "9751",
}

# Accepted subject-column header forms, compared after normalization.
SUBJECT_HEADER_FORMS = {"subject", "subject id", "subjectid", "subject_id"}

# Canonical name we give the subject column once found.
SUBJECT_ID = "Subject"

# What a real subject value looks like: 3-4 digit numeric (C3/C11) or C1Gxxxx.
_NUMERIC_SUBJECT = re.compile(r"^[0-9]{3,4}$")
_C1G_SUBJECT = re.compile(r"^C1G[0-9]{3,4}$", re.IGNORECASE)


def normalize_header(h):
    """Strip BOM, quotes, whitespace; lowercase. For header matching only."""
    return h.replace("\ufeff", "").strip().strip('"').strip().lower()


def clean_value(v):
    """Normalize a cell value to a comparable string."""
    if v is None:
        return ""
    return str(v).replace("\ufeff", "").strip().strip('"').strip()


def is_real_subject(v):
    """True if the cleaned value is a plausible subject ID, not an artifact."""
    v = clean_value(v)
    return bool(_NUMERIC_SUBJECT.match(v) or _C1G_SUBJECT.match(v))


def canon_subject(v):
    """
    Canonical form for roster comparison only (not for storage).
    Numeric IDs are int-normalized so leading zeros do not break matching:
    '0992', '992', and '992.0' all canonicalize to '992'. Campaign-1 codes are
    upper-cased. This lets a C3 roster written with zero-padded codes (0224,
    0446, 0992) match the master's parsed values (224, 446, 992). It never
    merges distinct subjects: subject numbers are unique, so 0992 and 992 are
    the same person.
    """
    s = clean_value(v)
    if s.upper().startswith("C1G"):
        return s.upper()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def find_subject_column(columns):
    """Return the original column name that is the subject key, or None."""
    for c in columns:
        if normalize_header(c) in SUBJECT_HEADER_FORMS:
            return c
    return None


# ----------------------------------------------------------------------------
# Robust IO (kept from v1, encoding fallback was sound)
# ----------------------------------------------------------------------------

def read_csv_robust(path):
    """Read a CSV trying several encodings. Returns a DataFrame or raises."""
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # empty file, bad CSV, etc. Let caller decide; surface once.
            raise e
    raise ValueError(f"Could not decode {path}: {last_err}")


def get_files_from_pattern(root, pattern):
    """Recursively collect files matching pattern, skipping macOS cruft."""
    out = []
    for p in Path(root).rglob(pattern):
        s = str(p)
        if "__MACOSX" in s:
            continue
        out.append(s)
    return out


# ----------------------------------------------------------------------------
# Per-file inspection: subject column, values, observation key, shape
# ----------------------------------------------------------------------------

# Columns that must never be used as an observation key: row serials and the
# like. A real observation key is a SHARED timepoint/condition label (Session,
# Day, Visit, Side), not a per-row identifier or a measurement value.
KEY_DENYLIST = {"id", "index", "row", "rownum", "record", "recordid", "uid"}

# Declared observation-key vocabulary. The key that turns one subject into
# multiple rows is a TIMEPOINT or within-subject CONDITION label, never a
# measurement. Uniqueness alone cannot tell the two apart: a jump-height
# column also makes rows unique, and auto-picking it buries the measurement
# in the column NAME and explodes the table. So we only accept keys from this
# curated set (normalized form). Extend it as new assays introduce new
# legitimate timepoint columns. A file whose rows cannot be made unique by
# any of these is logged 'unresolved_key' for a human to inspect, NOT keyed
# on a measurement.
TIMEPOINT_KEYS = {
    "test_phase", "phase",
    "br_day", "br day", "bdc_day", "day",
    "session", "trial", "test", "visit", "period", "week", "timepoint",
    "side", "leg", "limb", "region", "rep", "repeat", "set",
    # added after mining unresolved files: legitimate within-test condition
    # labels (NOT measurements, NOT arm/group which would be leakage).
    "position",        # isokinetic: e.g. "Extension at 60 deg/sec"
    "stage", "stage (watts)",  # aerobic test stage: "25%", "Cool Down"
    # v4 additions (recover latent virus + AD ASTRA behavioral):
    "time_period",     # latent virus: "Before"/"During"/"After"
    "daycount",        # AD ASTRA behavioral: integer day index
}

# v4: tidy-file signature. Some assays store data in tidy/long form: one column
# NAMES the variable and another HOLDS the number, e.g. immune panels with
# Unit="GRANULOCYTES (%)" and Value=41.2. The standard path assumes each measure
# is its own column, so subject x timepoint is not unique (many variables share
# a timepoint) and the file is wrongly logged 'unresolved_key'. v4 detects the
# pair and treats the variable column as part of the measure name.
VALUE_COL_NAMES = {
    "value", "result", "measurement", "reading", "conc", "concentration",
    "level", "amount", "count", "copies",
}
VARIABLE_COL_NAMES = {
    "unit", "analyte", "test", "parameter", "marker", "measure", "assay",
    "variable", "component", "analyte_name", "measurement_type",
}
# Between-subject / leakage columns never used as measure, key, or variable.
LEAKAGE_COL_NAMES = {"group", "grouplabel", "arm", "cohort", "treatment", "condition_arm"}

# v6.1: columns that are NOT physiological measurements and must stay OUT of the
# wide feature matrix. Arm/group is leakage; date/campaign are provenance;
# age/sex/gender/height are static subject descriptors sourced from the
# authoritative cohort file. Weight is intentionally ABSENT: body mass is a real
# longitudinal measurement. Excluded from the WIDE matrix only; the long table
# keeps them as a complete traceable substrate.
NON_MEASURE_COLS = {
    "groupname", "group", "grouplabel", "arm", "cohort", "treatment", "condition_arm",
    "date", "campaign", "age", "sex", "gender", "height",
}

# v6.1: files known to be corrupt/scrambled and always discarded (spec). Matched
# by name so the rule survives path changes. Only the BRSMVJ Post2 vertical-jump
# file is corrupted; other POST2-named files are legitimate post-bedrest tests
# and are kept (and flagged at run time).
def is_scrambled_file(path):
    t = normalize_header(file_tag(path))
    return "brsmvj" in t and "post2" in t


def find_observation_key(df, subject_col, max_combo=3):
    """
    Find the minimal, deterministic observation key: the smallest set of
    DECLARED timepoint/condition columns (see TIMEPOINT_KEYS) that, together
    with the subject, makes each row unique.

    Guards:
      - only columns in TIMEPOINT_KEYS are eligible (prevents keying on a
        measurement or a row serial such as ID)
      - denylisted row-identifier columns are dropped
      - columns unique across the whole file (row serials) are dropped

    Returns a tuple of column names (empty if subject alone is already
    unique), or None if no declared-key combo up to max_combo resolves
    duplicates -> caller logs 'unresolved_key'.
    """
    # subject alone already unique -> cross-sectional, empty key
    if df.duplicated(subset=[subject_col]).sum() == 0:
        return tuple()

    n = len(df)
    candidates = sorted(
        c for c in df.columns
        if c != subject_col
        and normalize_header(c) in TIMEPOINT_KEYS
        and normalize_header(c) not in KEY_DENYLIST
        and df[c].nunique(dropna=False) < n  # exclude global row-unique columns
    )
    # search smallest combos first; deterministic via sorted candidates
    for r in range(1, max_combo + 1):
        for combo in itertools.combinations(candidates, r):
            if df.duplicated(subset=[subject_col] + list(combo)).sum() == 0:
                return combo
    return None  # unresolved within max_combo


def detect_tidy(df):
    """
    v4: detect a tidy/long file where one column NAMES the variable and another
    HOLDS the value (e.g. immune panels: Unit='GRANULOCYTES (%)', Value=41.2).
    Returns (variable_col, value_col) or None.

    Guards: both a recognized value column and a recognized variable column must
    be present, and the variable column must be repeated categorical labels
    (analyte names), not a per-row-unique serial.
    """
    norm = {c: normalize_header(c) for c in df.columns}
    val_cols = [c for c in df.columns if norm[c] in VALUE_COL_NAMES]
    var_cols = [c for c in df.columns if norm[c] in VARIABLE_COL_NAMES]
    if not val_cols or not var_cols:
        return None
    var_col, val_col = var_cols[0], val_cols[0]
    if df[var_col].nunique(dropna=False) >= len(df):
        return None  # variable column is row-unique -> not a tidy label column
    return (var_col, val_col)


def inspect_file(path, max_combo=3, max_density=MAX_TIMEPOINTS_PER_SUBJECT):
    """
    Inspect one CSV. Returns a dict of metadata and, if usable, the cleaned
    DataFrame with a canonical Subject column of real subjects only.
    """
    info = {
        "file": path,
        "status": None,          # kept | no_subject_col | empty | unresolved_key | read_error
        "reason": "",
        "subject_col_raw": None,
        "n_subjects": 0,
        "n_rows_total": 0,
        "n_rows_real": 0,
        "n_artifact_rows": 0,
        "obs_key": None,
        "longitudinal": None,
        "rows_per_subject": None,  # "balanced:k" | "ragged" | "single"
        "shape_detail": None,      # per-subject row-count distribution (see below)
        "timepoints_per_subject": None,
        "tidy": None,              # v4: {"variable","value","obs_key"} if tidy/long
        "df": None,
    }

    # v6.1: known-scrambled file -> discard entirely (out of wide AND long).
    if is_scrambled_file(path):
        info["status"] = "excluded_scrambled"
        info["reason"] = "known-scrambled file (spec: always discard)"
        return info

    try:
        df = read_csv_robust(path)
    except Exception as e:
        info["status"] = "read_error"
        info["reason"] = str(e)[:200]
        return info

    if df.shape[0] == 0 or df.shape[1] == 0:
        info["status"] = "empty"
        return info

    subj = find_subject_column(df.columns)
    if subj is None:
        info["status"] = "no_subject_col"
        return info
    info["subject_col_raw"] = subj

    # canonicalize subject column, split real vs artifact
    df = df.copy()
    df[subj] = df[subj].map(clean_value)
    info["n_rows_total"] = len(df)
    real_mask = df[subj].map(is_real_subject)
    info["n_artifact_rows"] = int((~real_mask).sum())
    df = df[real_mask]
    info["n_rows_real"] = len(df)

    if len(df) == 0:
        info["status"] = "no_real_subjects"
        return info

    if subj != SUBJECT_ID:
        df = df.rename(columns={subj: SUBJECT_ID})

    subjects = df[SUBJECT_ID]
    info["n_subjects"] = subjects.nunique()

    # v4: retain the cleaned frame for EVERY file with real subjects, regardless
    # of whether it becomes wide-eligible. The long table is built from this and
    # captures files the wide master cannot (dense daily, replicate-laden assays
    # like latent virus). The wide master still uses only status=='kept' files.
    info["df"] = df

    # longitudinal shape
    counts = subjects.value_counts()
    if (counts == 1).all():
        info["longitudinal"] = False
        info["rows_per_subject"] = "single"
    else:
        info["longitudinal"] = True
        uniq_counts = set(counts.tolist())
        info["rows_per_subject"] = (
            f"balanced:{next(iter(uniq_counts))}" if len(uniq_counts) == 1 else "ragged"
        )

    # Ragged-shape detail. Expose the per-subject row-count distribution so the
    # downstream science-question step can decide how to treat unevenness with
    # real numbers in hand. We deliberately do NOT pad subjects to a common set
    # of timepoints (that fabricates completeness: an invented blank looks the
    # same as a real gap) and do NOT collapse timepoints to canonical ones (that
    # is an assay-specific science decision, not build-time plumbing). The master
    # keeps the data as measured; this only quantifies the raggedness.
    cvals = sorted(int(c) for c in counts.tolist())
    info["shape_detail"] = {
        "min_rows_per_subject": cvals[0],
        "max_rows_per_subject": cvals[-1],
        "median_rows_per_subject": int(pd.Series(cvals).median()),
        "distinct_row_counts": sorted(set(cvals)),
        "n_distinct_row_counts": len(set(cvals)),  # 1 = balanced, >1 = ragged
    }

    # Longitudinal gate (A/B toggle). In longitudinal-only mode, drop any file
    # where every subject appears exactly once. Logged with a reason so the
    # comparison against the inclusive variant is fully legible.
    if LONGITUDINAL_ONLY and info["longitudinal"] is False:
        info["status"] = "skipped_cross_sectional"
        info["reason"] = "LONGITUDINAL_ONLY: every subject appears once"
        return info

    # v4: tidy/long branch. If a variable-name + value pair is present, the
    # observation key is the declared TIMEPOINT columns only; the variable
    # column becomes part of the measure name at widen time. Density is measured
    # on subject x timepoint (ignoring the variable), so a panel of 15 analytes
    # across 6 timepoints reads as ~6 timepoints/subject, not 90.
    tidy = detect_tidy(df)
    if tidy:
        var_col, val_col = tidy
        tp_key = [
            c for c in df.columns
            if normalize_header(c) in TIMEPOINT_KEYS
            and normalize_header(c) not in KEY_DENYLIST
            and c != var_col
        ]
        ns = df[SUBJECT_ID].nunique()
        n_tp = df.groupby([SUBJECT_ID] + tp_key).ngroups if tp_key else ns
        per_subj = n_tp / ns if ns else 0
        info["timepoints_per_subject"] = round(per_subj, 1)
        info["obs_key"] = tp_key
        info["tidy"] = {"variable": var_col, "value": val_col, "obs_key": tp_key}
        if per_subj > max_density:
            info["status"] = "dense_longitudinal"
            info["reason"] = (
                f"tidy key {tp_key} implies ~{per_subj:.0f} timepoints/subject "
                f"(> {max_density}); needs aggregation before wide pivot"
            )
            return info
        info["status"] = "kept"
        info["df"] = df
        return info

    key = find_observation_key(df, SUBJECT_ID, max_combo=max_combo)
    if key is None:
        info["status"] = "unresolved_key"
        info["reason"] = f"no unique key within max_combo={max_combo}"
        return info

    info["obs_key"] = list(key)

    # Density guard: how many timepoints per subject does this key imply?
    ns = df[SUBJECT_ID].nunique()
    n_tuples = df.groupby([SUBJECT_ID] + list(key)).ngroups if key else ns
    per_subj = n_tuples / ns if ns else 0
    info["timepoints_per_subject"] = round(per_subj, 1)
    if per_subj > max_density:
        info["status"] = "dense_longitudinal"
        info["reason"] = (
            f"key {list(key)} implies ~{per_subj:.0f} timepoints/subject "
            f"(> {max_density}); needs aggregation before wide pivot"
        )
        return info

    info["status"] = "kept"
    info["df"] = df
    return info


# ----------------------------------------------------------------------------
# Master assembly
# ----------------------------------------------------------------------------

def file_tag(path):
    """Short, portable file identity for column provenance (basename, no ext)."""
    return os.path.basename(path).rsplit(".csv", 1)[0]


def obskey_label(row, obs_key):
    """Encode this row's observation-key values, e.g. 'Visit=2|Side=Right'."""
    if not obs_key:
        return ""
    return "|".join(f"{k}={clean_value(row[k])}" for k in obs_key)


def declared_timepoint_cols(df, exclude=None):
    """
    Every column in this file whose (normalized) name is a DECLARED timepoint/
    condition label (TIMEPOINT_KEYS), minus the row-serial denylist and any
    caller exclusion (the tidy variable column). This is the SINGLE definition
    of "which columns are timepoints", shared by the wide reshape (widen_file)
    and the long export (to_long), so a timepoint is treated identically in both
    tables and the two paths cannot drift.

    A timepoint column is never a measurement. It is folded into the column name
    / timepoint label and excluded from the melted measures. This is the fix for
    the phantom timepoint-as-measure columns: a file can carry more timepoint
    columns than the minimal observation key needs to make rows unique (e.g.
    Session keys the rows while Test_Phase and BR_Day ride along), and v5 swept
    those extras into the melt as if they were measured biomarkers, producing a
    column literally named Test_Phase::... whose value is the string POST_TEST.
    """
    exclude = set(exclude or ())
    return [
        c for c in df.columns
        if c not in exclude
        and normalize_header(c) in TIMEPOINT_KEYS
        and normalize_header(c) not in KEY_DENYLIST
    ]


def widen_file(info):
    """
    Turn one inspected long file into a wide, subject-indexed frame.
    Columns are named  measure::filetag::obskey  so every value is traceable
    to measure, source file, and timepoint.

    Vectorized: build the obs-key label per row, melt measures to long, form
    the composite column name, then pivot. No row-by-row assignment.
    """
    df = info["df"].copy()
    obs_key = info["obs_key"]
    tag = file_tag(info["file"])

    # v4 tidy branch: measure name comes from the variable column's values, the
    # number from the value column. Column = varvalue::tag::timepoint.
    if info.get("tidy"):
        var_col = info["tidy"]["variable"]
        val_col = info["tidy"]["value"]
        if obs_key:
            label = df[obs_key].apply(
                lambda r: "|".join(f"{k}={clean_value(r[k])}" for k in obs_key), axis=1
            )
        else:
            label = pd.Series([""] * len(df), index=df.index)
        df["__label__"] = label
        var_clean = df[var_col].map(clean_value)
        df["__col__"] = var_clean + "::" + tag
        has = df["__label__"] != ""
        df.loc[has, "__col__"] = var_clean[has] + "::" + tag + "::" + df.loc[has, "__label__"]
        wide = df.pivot_table(
            index=SUBJECT_ID, columns="__col__", values=val_col, aggfunc="first"
        )
        wide.columns.name = None
        return wide

    # v6 FIX (phantom timepoint-as-measure). Both the measure exclusion and the
    # row label use EVERY declared timepoint column present, not just the minimal
    # obs_key that made rows unique. obs_key stays the uniqueness/density key
    # (set in inspect_file); it is a subset of tp_cols. Using the full set here:
    #   - removes every timepoint column from the melt, so none can appear as a
    #     phantom measure column (the 1,501-column artifact), and
    #   - folds the full timepoint context into the column name, matching exactly
    #     what the long table already encodes (to_long uses the same helper).
    # Adding columns to the label can only make it finer, never collide, so pivot
    # uniqueness is preserved. No values change; no real measure is dropped.
    tp_cols = declared_timepoint_cols(df)

    measure_cols = [
        c for c in df.columns
        if c != SUBJECT_ID and c not in tp_cols
        and normalize_header(c) not in KEY_DENYLIST       # drop ID/index/serials
        and normalize_header(c) not in LEAKAGE_COL_NAMES  # drop group/arm (leakage)
        and normalize_header(c) not in NON_MEASURE_COLS   # v6.1: drop covariates+provenance (Weight kept)
    ]
    if not measure_cols:
        return pd.DataFrame(index=pd.Index(sorted(df[SUBJECT_ID].unique()), name=SUBJECT_ID))

    # one label string per row encoding this row's FULL timepoint/observation,
    # built from every declared timepoint column present (superset of obs_key).
    if tp_cols:
        label = df[tp_cols].apply(
            lambda r: "|".join(f"{k}={clean_value(r[k])}" for k in tp_cols), axis=1
        )
    else:
        label = pd.Series([""] * len(df), index=df.index)
    df["__label__"] = label

    long = df.melt(
        id_vars=[SUBJECT_ID, "__label__"],
        value_vars=measure_cols,
        var_name="__measure__",
        value_name="__value__",
    )
    # composite, self-describing column name
    has_label = long["__label__"] != ""
    long["__col__"] = long["__measure__"] + "::" + tag
    long.loc[has_label, "__col__"] = (
        long.loc[has_label, "__measure__"] + "::" + tag + "::" + long.loc[has_label, "__label__"]
    )

    wide = long.pivot_table(
        index=SUBJECT_ID,
        columns="__col__",
        values="__value__",
        aggfunc="first",
    )
    wide.columns.name = None
    return wide


def source_folder(path, root):
    """Top-level experiment folder for a file, relative to the scan root."""
    try:
        rel = os.path.relpath(path, root)
        return rel.split(os.sep)[0]
    except Exception:
        return "?"


LONG_COLS = [SUBJECT_ID, "folder", "source_file", "variable", "timepoint", "value"]


def to_long(info, root):
    """
    Emit one file's data in tidy long form: one row per observed value, columns
    Subject, folder, source_file, variable, timepoint, value. NaN values are
    dropped so the long table holds only real observations.

    Works for ANY file with real subjects, including files the wide master
    rejects (dense, or replicate-laden / non-uniquely-keyable). The timepoint
    label is built from every DECLARED timepoint/condition column present, so
    replicate and period columns are preserved as context rather than forced
    into a unique key. Uniqueness is not required in long form.
    """
    df = info.get("df")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=LONG_COLS)
    df = df.copy()
    tag = file_tag(info["file"])
    folder = source_folder(info["file"], root)

    # dimension columns = declared timepoint/condition columns present in file.
    # Same definition the wide reshape uses (declared_timepoint_cols), so a
    # timepoint is treated identically in both tables.
    var_excl = (info["tidy"]["variable"],) if info.get("tidy") else ()
    dims = declared_timepoint_cols(df, exclude=var_excl)
    if dims:
        tp = df[dims].apply(
            lambda r: "|".join(f"{k}={clean_value(r[k])}" for k in dims), axis=1
        )
    else:
        tp = pd.Series([""] * len(df), index=df.index)
    df["__tp__"] = tp

    if info.get("tidy"):
        var_col = info["tidy"]["variable"]
        val_col = info["tidy"]["value"]
        out = pd.DataFrame({
            SUBJECT_ID: df[SUBJECT_ID].values, "folder": folder, "source_file": tag,
            "variable": df[var_col].map(clean_value).values,
            "timepoint": df["__tp__"].values, "value": df[val_col].values,
        })
    else:
        measure_cols = [
            c for c in df.columns
            if c != SUBJECT_ID and c not in dims and c != "__tp__"
            and normalize_header(c) not in LEAKAGE_COL_NAMES
            and normalize_header(c) not in KEY_DENYLIST
        ]
        if not measure_cols:
            return pd.DataFrame(columns=LONG_COLS)
        mlt = df.melt(
            id_vars=[SUBJECT_ID, "__tp__"], value_vars=measure_cols,
            var_name="variable", value_name="value",
        )
        out = pd.DataFrame({
            SUBJECT_ID: mlt[SUBJECT_ID].values, "folder": folder, "source_file": tag,
            "variable": mlt["variable"].values,
            "timepoint": mlt["__tp__"].values, "value": mlt["value"].values,
        })
    return out[out["value"].notna()]


def build_long(infos, root, roster=None):
    """
    Concatenate the long form of EVERY file with real subjects (not just wide-
    eligible ones), so the long table is the complete traceable substrate.
    """
    parts = [to_long(info, root) for info in infos if info.get("df") is not None]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=LONG_COLS)
    longdf = pd.concat(parts, ignore_index=True)
    if roster is not None:
        canon_roster = {canon_subject(r) for r in roster}
        longdf = longdf[longdf[SUBJECT_ID].map(lambda s: canon_subject(s) in canon_roster)]
    return longdf


def reconcile_completeness(all_infos, longdf, root, roster):
    """
    v5 completeness reconciliation (data-engineering standard: compare captured
    output against the source to PROVE nothing was dropped).

    For each source folder:
      expected  = distinct (roster-filtered) subjects that appear in ANY raw
                  file's subject column in that folder, regardless of whether
                  the file was kept, dense, or unresolved.
      captured  = distinct (roster-filtered) subjects present in the long table
                  for that folder.
      gap       = expected - captured  (subjects seen in raw data but with no
                  observation captured -> a real completeness failure).

    Returns a list of dicts, one per folder, sorted by gap size then name.
    """
    from collections import defaultdict
    canon_roster = {canon_subject(r) for r in roster} if roster else None

    expected = defaultdict(set)
    for info in all_infos:
        df = info.get("df")
        if df is None:
            continue
        folder = source_folder(info["file"], root)
        for s in df[SUBJECT_ID].unique():
            cs = canon_subject(s)
            if canon_roster is None or cs in canon_roster:
                expected[folder].add(cs)

    captured = defaultdict(set)
    if longdf is not None and len(longdf):
        for folder, grp in longdf.groupby("folder"):
            captured[folder] = {canon_subject(s) for s in grp[SUBJECT_ID].unique()}

    rows = []
    for folder in sorted(set(expected) | set(captured)):
        exp = expected.get(folder, set())
        cap = captured.get(folder, set())
        gap = sorted(exp - cap)
        rows.append({
            "folder": folder,
            "expected_subjects": len(exp),
            "captured_subjects": len(cap),
            "gap_n": len(gap),
            "gap_subjects": gap,
        })
    rows.sort(key=lambda r: (-r["gap_n"], r["folder"]))
    return rows


def audit_no_subject_files(all_infos, root, roster):
    """
    v5 blind-spot audit. Files with NO subject column can still hold real data
    with the subject encoded in the FILENAME (e.g. bone DXA/QCT per-subject
    files). A column-only scan cannot see them, so they would silently vanish.
    This lists every no_subject_col file and flags any roster subject IDs found
    in its name as candidates for filename-subject recovery.
    """
    canon_roster = {canon_subject(r) for r in roster} if roster else None
    out = []
    for info in all_infos:
        if info["status"] != "no_subject_col":
            continue
        base = file_tag(info["file"])
        folder = source_folder(info["file"], root)
        found = []
        if canon_roster:
            for tok in re.findall(r"\d{3,4}", base):
                if canon_subject(tok) in canon_roster:
                    found.append(canon_subject(tok))
        out.append({"folder": folder, "file": base, "roster_ids_in_name": sorted(set(found))})
    return out


def build_master(kept_infos, roster=None):
    """
    Outer-join every widened file on Subject into one master table.
    If roster is given, keep only those subjects (rows) in the final master.
    """
    master = None
    for info in kept_infos:
        wide = widen_file(info)
        master = wide if master is None else master.join(wide, how="outer")
    if master is None:
        return pd.DataFrame()
    master = master.sort_index()
    if roster is not None:
        canon_roster = {canon_subject(r) for r in roster}
        present = [s for s in master.index if canon_subject(s) in canon_roster]
        master = master.loc[present]
    # v5: drop all-empty columns. After the roster filter, columns sourced from
    # other-campaign files (their rows removed) and per-subject-file fragments
    # are entirely NaN. Dropping them is zero data loss (every value is missing)
    # and removes the empty padding seen on manual review. The complete record
    # lives in the long table regardless; this only slims the wide view.
    n_before = master.shape[1]
    master = master.dropna(axis=1, how="all")
    master.attrs["empty_cols_dropped"] = n_before - master.shape[1]
    return master


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a subject-keyed, traceable master table.")
    ap.add_argument("root", help="root directory to scan recursively")
    ap.add_argument("--outdir", default=".", help="where to write master + manifest")
    ap.add_argument("--max-combo", type=int, default=3, help="max obs-key column combo size")
    ap.add_argument("--max-density", type=int, default=MAX_TIMEPOINTS_PER_SUBJECT, help="max timepoints/subject before a file is held out as dense_longitudinal")
    ap.add_argument("--c11", action="store_true", help="filter master to the built-in 46-subject Campaign 11 roster")
    ap.add_argument("--roster", default=None, help="path to a file of subject IDs (one per line) to filter the master to")
    ap.add_argument("--no-long", action="store_true", help="skip writing the long-format table (wide master only)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    csv_files = get_files_from_pattern(args.root, "*.csv")
    print(f"csv files found: {len(csv_files)}")

    manifest = []
    kept = []
    all_infos = []
    for path in csv_files:
        info = inspect_file(path, max_combo=args.max_combo, max_density=args.max_density)
        # manifest entry without the DataFrame
        entry = {k: v for k, v in info.items() if k != "df"}
        manifest.append(entry)
        all_infos.append(info)  # retains df for the long builder
        if info["status"] == "kept":
            kept.append(info)

    # summary
    from collections import Counter
    status_counts = Counter(m["status"] for m in manifest)
    print("file status breakdown:")
    for s, n in status_counts.most_common():
        print(f"  {s:<18} {n}")
    print(f"files kept for master: {len(kept)}")

    # v6.1: flag non-scrambled POST2-named files (kept) so legitimacy is verified.
    other_post2 = sorted({file_tag(m["file"]) for m in manifest
                          if "post2" in normalize_header(file_tag(m["file"]))
                          and m["status"] != "excluded_scrambled"})
    if other_post2:
        print(f"NOTE: POST2-named files KEPT (verify legitimacy): {other_post2}")

    roster = None
    if args.c11:
        roster = C11_ROSTER
    elif args.roster:
        with open(args.roster) as rf:
            roster = {ln.strip() for ln in rf if ln.strip()}
    master = build_master(kept, roster=roster)
    if roster is not None:
        canon_roster = {canon_subject(r) for r in roster}
        canon_present = {canon_subject(s) for s in master.index}
        found = sorted(canon_roster & canon_present)
        print(f"roster filter applied: {len(found)}/{len(canon_roster)} roster subjects present in master")
        missing = sorted(canon_roster - canon_present)
        if missing:
            print(f"  roster subjects NOT in master: {missing}")
    dropped = master.attrs.get("empty_cols_dropped", 0)
    print(f"master shape: {master.shape[0]} subjects x {master.shape[1]} columns "
          f"(dropped {dropped} all-empty columns)")

    master_path = os.path.join(args.outdir, "master_table.csv")
    manifest_path = os.path.join(args.outdir, "master_manifest.json")
    master.to_csv(master_path)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"wrote {master_path}")
    print(f"wrote {manifest_path}")

    # v4: long-format export (tall, NaN-dropped). Same data, cheaper to select.
    longdf = None
    if not args.no_long:
        longdf = build_long(all_infos, args.root, roster=roster)
        long_path = os.path.join(args.outdir, "master_long.csv")
        longdf.to_csv(long_path, index=False)
        print(f"wrote {long_path}  ({len(longdf)} observed values)")

    # v4: honest per-system coverage, attributed by TRUE source folder (not by
    # filename prefix). Uses the long table when available, else the manifest.
    if longdf is not None and len(longdf):
        cov = longdf.groupby("folder")[SUBJECT_ID].nunique().sort_values(ascending=False)
        print("\nper-system coverage (subjects with any observed value):")
        for folder, n in cov.items():
            print(f"  {n:>3}  {folder}")

    # v5: completeness reconciliation. Proves every subject seen in raw data was
    # captured. This is the answer to "did we catch all the data" -> a number.
    if longdf is not None:
        recon = reconcile_completeness(all_infos, longdf, args.root, roster)
        total_gap = sum(r["gap_n"] for r in recon)
        print("\ncompleteness reconciliation (expected vs captured, per folder):")
        print(f"  {'exp':>4} {'cap':>4} {'gap':>4}  folder")
        for r in recon:
            flag = "" if r["gap_n"] == 0 else "  <-- GAP"
            print(f"  {r['expected_subjects']:>4} {r['captured_subjects']:>4} {r['gap_n']:>4}  {r['folder']}{flag}")
        if total_gap == 0:
            print(f"  RECONCILED: every subject seen in raw data was captured (0 gaps across {len(recon)} folders)")
        else:
            print(f"  {total_gap} subject-folder gaps: subjects present in raw data but not captured (see report)")

        # v5: blind-spot audit for files with no subject column (filename-encoded)
        audit = audit_no_subject_files(all_infos, args.root, roster)
        flagged = [a for a in audit if a["roster_ids_in_name"]]
        if flagged:
            print(f"\nno-subject-column files with roster IDs in their NAME "
                  f"({len(flagged)} files, candidates for filename-subject recovery):")
            byf = {}
            for a in flagged:
                byf.setdefault(a["folder"], set()).update(a["roster_ids_in_name"])
            for folder in sorted(byf):
                print(f"  {folder}: {len(byf[folder])} roster subjects encoded in filenames")

        report = {"reconciliation": recon, "no_subject_column_audit": audit}
        report_path = os.path.join(args.outdir, "reconciliation_report.json")
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
