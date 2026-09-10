#!/usr/bin/env python3
"""
build_totalfat_target_v2.py

Aim 1 Run 1 (redo) feature tables with DXA TOTAL FAT CHANGE as the outcome.
v2 applies peer-review fixes 1-3. Fix 4 is deliberately NOT applied -- see below.

CHANGES FROM v1
---------------
FIX 1  Drop single-subject columns. 36 features came from per-subject CRF files
       and carry the subject ID in the column name (…crf_cft70_vitals_5160,
       …waterintake_summary_8072, …vitals_8936). Each has exactly n=1 non-null.
       They are useless as features AND act as a subject fingerprint: a model
       keying on "this column is populated" has identified an individual, not
       learned physiology. This is the filename-encoded-subject blind spot from
       the build manifest surfacing inside the feature matrix.

FIX 2  One outcome per file. v1 shipped total_fat_pct_change and
       total_fat_abs_change_g side by side. They are the same quantity up to a
       per-subject scale factor, so "all columns except the target" leaks the
       answer. Percent is primary (matches the published +4.2%); absolute is
       written to its own file for reference.

FIX 3  Collapse exact duplicate columns. The vertical-jump source carried two
       names for one measure (max_acceleration__g__ / maxaccel__g__, and the
       velocity pair, for both pre1 and pre2). First occurrence in column order
       is kept; the duplicate is dropped and logged.

NOT APPLIED -- FIX 4, and why it matters
----------------------------------------
Timepoint provenance is absent from the inherited flattened column names. The
master encodes measure::file::timepoint; the flattened names keep measure and a
truncated file tag only. So baseline status CANNOT be verified from this table --
`activityboard____bedrest_ftt_cft70_hr_acrosstests_datatab` gives no timepoint.
Fixing this means regenerating from master_table.csv, which would break strict
X-comparability with the BMD run already executing. Held by decision.

CONSEQUENCE: any write-up must state that features are asserted baseline by
construction of the upstream flattening step, not verified in this artifact.

LEAKAGE EXCLUSIONS (unchanged from v1, 134 columns)
--------------------------------------------------
  direct     total_fat____idxa_cft70_body_composition_total = Pre value of y
  algebraic  131 DXA body-composition cols; total fat = tissue - fat free, and
             regional fat sums to total fat, same scan
  arm proxy  LH and FSH; bimodal and near-deterministic for the testosterone arm
             (Dillon 2018 Table 1: TEX ~0.5 vs CON 2.7-2.9). Arm never enters X.

OUTCOME (named before features were touched)
-------------------------------------------
  total_fat_pct_change = 100 * (Fat[Post] - Fat[Pre]) / Fat[Pre]
  source: Total Fat::BEDREST_IRATS_iDXA_CFT70_Body_Composition_Total_obsv
  units of the raw columns are GRAMS (Pre mean 17,112; range 5,711-29,556)

NO IMPUTATION. NaN passes through; TabPFN handles it. Missingness is reported.

Usage:
  python3 build_totalfat_target_v2.py \
      --bmd-table c11_totalhipBMD_change_features_all.csv \
      --shortlist c11_totalhipBMD_change_features_shortlist.csv \
      --master    master_table.csv \
      --outdir    .
"""

import argparse
import os
import re
import sys

import pandas as pd

FAT_PRE = "Total Fat::BEDREST_IRATS_iDXA_CFT70_Body_Composition_Total_obsv::Test=Pre"
FAT_POST = "Total Fat::BEDREST_IRATS_iDXA_CFT70_Body_Composition_Total_obsv::Test=Post"

OLD_OUTCOME = "totalhip_BMD_change"
Y_PCT = "total_fat_pct_change"
Y_ABS = "total_fat_abs_change_g"
SUBJ = "subject"

PUBLISHED_PCT, PUBLISHED_DZ = 4.2, 0.47

DXA_TOKENS = ("body_composition", "idxa")
ARM_PROXY_TOKENS = ("fsh", "lh__miu_ml")
SUBJECT_ID_RE = re.compile(r"_(\d{4})$")   # FIX 1


def canon(v):
    s = str(v).strip()
    if s.upper().startswith("C1G"):
        return s.upper()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def drop_reason(col):
    """Reason to exclude a feature, or None to keep."""
    c = col.lower()
    if c == "total_fat____idxa_cft70_body_composition_total":
        return "LEAKAGE-direct: Pre value of the outcome variable"
    if any(t in c for t in DXA_TOKENS):
        return "LEAKAGE-algebraic: DXA body-composition (component/complement of total fat)"
    if any(t in c for t in ARM_PROXY_TOKENS):
        return "LEAKAGE-arm-proxy: LH/FSH near-deterministic for testosterone arm"
    m = SUBJECT_ID_RE.search(c)
    if m:
        return f"SINGLE-SUBJECT: per-subject CRF column for {m.group(1)} (n=1, subject fingerprint)"
    return None


def dz_paired(diff):
    d = diff.dropna()
    sd = d.std(ddof=1)
    return float("nan") if sd == 0 else d.mean() / sd


def find_duplicate_cols(df, cols):
    """Exact-duplicate feature columns. Returns {dropped: kept}, first-seen wins."""
    seen, dup = {}, {}
    for c in cols:
        key = tuple(pd.util.hash_pandas_object(df[c].fillna(-9e99), index=False))
        if key in seen:
            dup[c] = seen[key]
        else:
            seen[key] = c
    return dup


def main():
    ap = argparse.ArgumentParser(description="Build fat-change target tables (v2, peer-review fixes 1-3).")
    ap.add_argument("--bmd-table", required=True)
    ap.add_argument("--shortlist", default=None)
    ap.add_argument("--master", required=True)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    feat = pd.read_csv(args.bmd_table)
    mas = pd.read_csv(args.master, low_memory=False).copy()
    print(f"feature table : {feat.shape[0]} x {feat.shape[1]}  ({os.path.basename(args.bmd_table)})")
    print(f"master        : {mas.shape[0]} x {mas.shape[1]}  ({os.path.basename(args.master)})")

    for c in (FAT_PRE, FAT_POST):
        if c not in mas.columns:
            sys.exit(f"FATAL: master missing outcome source:\n  {c}")
    if SUBJ not in feat.columns:
        sys.exit(f"FATAL: no '{SUBJ}' column in feature table")

    # ---------------------------------------------------------------- outcome
    mas["_s"] = mas["Subject"].map(canon)
    fat = mas.loc[:, ["_s", FAT_PRE, FAT_POST]].dropna(subset=[FAT_PRE, FAT_POST]).copy()
    if fat["_s"].duplicated().any():
        sys.exit("FATAL: duplicate subjects in master fat columns")
    fat[Y_ABS] = fat[FAT_POST] - fat[FAT_PRE]
    fat[Y_PCT] = 100.0 * fat[Y_ABS] / fat[FAT_PRE]

    feat["_s"] = feat[SUBJ].map(canon)
    fs, ms = set(feat["_s"]), set(fat["_s"])
    print(f"\nsubject join  : features={len(fs)}  fat Pre+Post={len(ms)}  intersect={len(fs & ms)}")
    if fs - ms:
        print(f"  dropped (no outcome)   : {sorted(fs - ms)}")
    if ms - fs:
        print(f"  unused (not in X)      : {sorted(ms - fs)}")
    if not fs & ms:
        sys.exit("FATAL: empty subject intersection")

    # ----------------------------------------------------------- feature cuts
    all_feats = [c for c in feat.columns if c not in (SUBJ, OLD_OUTCOME, "_s")]
    leak = {c: r for c in all_feats if (r := drop_reason(c))}
    stage1 = [c for c in all_feats if c not in leak]
    dup = find_duplicate_cols(feat, stage1)                     # FIX 3
    keeps = [c for c in stage1 if c not in dup]

    print(f"\nfeatures in            : {len(all_feats)}")
    print(f"  dropped, leakage      : {sum(1 for r in leak.values() if r.startswith('LEAKAGE'))}")
    print(f"  dropped, single-subj  : {sum(1 for r in leak.values() if r.startswith('SINGLE'))}")
    print(f"  dropped, duplicate    : {len(dup)}")
    print(f"features out           : {len(keeps)}")

    by_reason = {}
    for c, r in leak.items():
        tag = r.split(":")[0]
        by_reason.setdefault(tag, []).append((c, r))
    for tag in sorted(by_reason, key=lambda k: -len(by_reason[k])):
        items = sorted(by_reason[tag])
        print(f"\n  [{len(items)}] {tag}")
        for c, r in items[:4]:
            print(f"        {c}")
        if len(items) > 4:
            print(f"        ... and {len(items) - 4} more")
    if dup:
        print(f"\n  [{len(dup)}] EXACT-DUPLICATE (kept <- dropped)")
        for drop_c, kept_c in sorted(dup.items()):
            print(f"        {kept_c}  <-  {drop_c}")

    # -------------------------------------------------------------- assemble
    base = (
        feat.loc[:, ["_s", SUBJ] + keeps]
        .merge(fat.loc[:, ["_s", Y_PCT, Y_ABS]], on="_s", how="inner")
        .drop(columns=["_s"])
        .sort_values(SUBJ)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------- sanity + report
    print("\n" + "=" * 64)
    print("OUTCOME SANITY CHECK  (DXA total fat, Pre -> Post, grams)")
    print("=" * 64)
    print(f"  n                : {len(base)}")
    print(f"  mean pct change  : {base[Y_PCT].mean():+.2f}%   published {PUBLISHED_PCT:+.1f}%")
    print(f"  dz (paired)      : {dz_paired(base[Y_ABS]):+.3f}     published {PUBLISHED_DZ:+.2f}")
    print(f"  mean abs change  : {base[Y_ABS].mean():+.1f} g")
    print(f"  pct range        : {base[Y_PCT].min():+.1f}% to {base[Y_PCT].max():+.1f}%")
    q1, q3 = base[Y_PCT].quantile([.25, .75])
    iqr = q3 - q1
    o = base.loc[(base[Y_PCT] < q1 - 1.5 * iqr) | (base[Y_PCT] > q3 + 1.5 * iqr), [SUBJ, Y_PCT, Y_ABS]]
    if len(o):
        print(f"  IQR outliers ({len(o)}) -- VERIFY AGAINST RAW ARCHIVE BEFORE MODELING:")
        for _, r in o.iterrows():
            print(f"        subject {int(r[SUBJ])}: {r[Y_PCT]:+.1f}%  ({r[Y_ABS]:+.0f} g)")
    if len(base) != 37:
        print(f"  ** n={len(base)}; Cromwell Table 3 gives 37 completers. Apply the same")
        print("     cohort cut to bone and fat, or the two runs are not comparable. **")

    X = base.loc[:, keeps]
    cov = X.notna().mean()
    print("\n" + "=" * 64)
    print("MISSINGNESS (no imputation; NaN passed to TabPFN)")
    print("=" * 64)
    print(f"  overall missing        : {X.isna().mean().mean():.1%}")
    print(f"  complete columns       : {(cov == 1.0).sum()} / {len(keeps)}")
    print(f"  >=80% coverage         : {(cov >= 0.80).sum()}")
    print(f"  >=50% coverage         : {(cov >= 0.50).sum()}")
    print(f"  <25% coverage          : {(cov < 0.25).sum()}")
    print(f"  complete-case rows     : {X.dropna().shape[0]} / {len(base)}")
    print("  -> pre-register the coverage threshold before running.")

    # ------------------------------------------------------------ FIX 2 write
    outs = []
    p = os.path.join(args.outdir, "c11_totalfat_pct_change_features_all.csv")
    base.drop(columns=[Y_ABS]).to_csv(p, index=False)
    outs.append((p, base.shape[0], len(keeps) + 2, "PRIMARY: percent change"))

    p = os.path.join(args.outdir, "c11_totalfat_abs_change_features_all.csv")
    base.drop(columns=[Y_PCT]).to_csv(p, index=False)
    outs.append((p, base.shape[0], len(keeps) + 2, "reference: absolute grams"))

    if args.shortlist:
        sl = pd.read_csv(args.shortlist)
        sl_feats = [c for c in sl.columns if c not in (SUBJ, OLD_OUTCOME)]
        sl_keep = [c for c in sl_feats if not drop_reason(c) and c in base.columns]
        sl_lost = [c for c in sl_feats if drop_reason(c)]
        p = os.path.join(args.outdir, "c11_totalfat_pct_change_features_shortlist.csv")
        base.loc[:, [SUBJ] + sl_keep + [Y_PCT]].to_csv(p, index=False)
        outs.append((p, base.shape[0], len(sl_keep) + 2, f"shortlist ({len(sl_keep)} feats)"))
        if sl_lost:
            print(f"\n  shortlist lost {len(sl_lost)} of {len(sl_feats)} features to leakage: {sl_lost}")
            print("  ** fat shortlist != bone shortlist; do not compare across outcomes **")

    print("\n" + "=" * 64)
    print("WROTE")
    print("=" * 64)
    for p, r, c, note in outs:
        print(f"  {r} x {c:<4} {note}")
        print(f"        {p}")

    print("\n" + "=" * 64)
    print("RUN LOG LINE")
    print("=" * 64)
    print("  script      : build_totalfat_target_v2.py")
    print(f"  outcome     : {Y_PCT} (DXA total fat Pre->Post, percent)")
    print(f"  n subjects  : {base.shape[0]}")
    print(f"  n features  : {len(keeps)} (from {len(all_feats)}; "
          f"{len(leak)} leakage/single-subject, {len(dup)} duplicate)")
    print("  caveat      : timepoint provenance not verifiable in this artifact (fix 4 deferred)")
    print("  model run   : none in this script")


if __name__ == "__main__":
    main()
