#!/usr/bin/env python3
"""
build_aim1_targets_v3.py

ONE builder for both Aim 1 Test 1 outcomes, so the two tables are constructed
identically and are legitimately comparable. Supersedes build_totalfat_target_v2.py
and build_twp_target_v1.py.

FIXES OVER THOSE TWO SCRIPTS -- all three found in peer review
--------------------------------------------------------------
FIX A  FILTER ORDER. v1/v2 removed constant and duplicate columns from the
       38-row inherited feature table, then joined the outcome, which for TWP
       cuts the cohort to 35. Four columns became constant and two became exact
       duplicates only AFTER that cut, and survived into the model table. All
       structural filters now run on the FINAL cohort, after the join.

FIX B  NEAR-DUPLICATES. Exact hashing misses the same measurement recorded in
       different units or exported twice. 13 pairs at |r| > 0.9999 were present,
       including force_n vs force_lbs (r = 1.0000000000, ratio 4.448 = N/lbf),
       fall-recovery HR duplicated across two source files, and iDXA measures
       appearing in both the BEDREST_IRATS and NNX10AP86G exports. These do not
       destabilise a TabPFN fit the way they would an OLS coefficient, but they
       consume context and split feature attribution, so importance rankings
       become misleading. Removed at |r| > 0.999 on the final cohort,
       first-occurrence-wins, every pair logged.

FIX C  BOTH OUTCOMES, ONE CODE PATH. Leakage is outcome-specific, so the
       exclusion list is per-outcome, but every structural rule is shared.

OUTCOMES
--------
fat  total_fat_pct_change = 100 * (Fat[Post] - Fat[Pre]) / Fat[Pre]
     n=38. Published +4.2% / dz +0.47 (37 completers); here +3.64% / +0.439.
     Percent is safe: fat is strictly positive, mean 17,112 g, min 5,711 g.
     Leakage: the whole DXA body-composition block. Total fat = tissue - fat
     free, and regional fat sums to total fat, all from one scan. Baseline total
     fat is literally the Pre value of the outcome.

twp  twp_abs_change = TWP[POST BR_Day=0 Session=D] - TWP[PRE BR_Day=11 Session=A]
     n=35. Reproduces published -60.7% (change of group means) and dz -1.35.
     Pairing chosen by matching BOTH statistics across all 12 pre/post session
     combinations; no other combination matches.
     ABSOLUTE, not percent: 5 of 35 subjects have post <= 0, so percent change
     is unstable and reaches -1086% for one subject. The three percent
     definitions disagree by 43 points (mean-of-subject -86.0%, median -42.5%,
     of-means -60.7%).
     Leakage: baseline TWP (direct), Percent_Correct_Steps (r +0.83/+0.64/+0.57
     vs baseline TWP -- the accuracy term of the composite), and LineTest trial
     total Time (r to -0.61 -- the speed term). Torso RMS (|r| <= 0.45) and
     LineTest heart rate (|r| <= 0.30) are retained: verified independent, not
     assumed. The DXA block is NOT leakage against a balance outcome, so twp
     keeps far more features than fat.

SHARED EXCLUSIONS
-----------------
  LH and FSH. Bimodal, near-deterministic for the testosterone arm (Dillon 2018
  Table 1: TEX ~0.5 vs CON 2.7-2.9). Arm never enters a feature matrix.

  36 per-subject CRF columns carrying a subject ID in the column name
  (…crf_cft70_vitals_5160 and similar), each with exactly n=1. Useless as
  features and a subject fingerprint: a model keying on "this column is
  populated" has identified an individual, not learned physiology.

NO IMPUTATION anywhere. NaN passes through to TabPFN. Missingness is reported so
the coverage-threshold decision stays explicit and downstream.

Usage:
  python3 build_aim1_targets_v3.py \
      --bmd-table c11_totalhipBMD_change_features_all.csv \
      --master    master_table.csv \
      --outdir    . \
      --outcome   both
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

SUBJ = "subject"
OLD_OUTCOME = "totalhip_BMD_change"

FAT_PRE = "Total Fat::BEDREST_IRATS_iDXA_CFT70_Body_Composition_Total_obsv::Test=Pre"
FAT_POST = "Total Fat::BEDREST_IRATS_iDXA_CFT70_Body_Composition_Total_obsv::Test=Post"
TB = "TWP::BEDREST_FTT_CFT70_LineTest_TWP_obsv::"
TWP_PRE = TB + "Test_Phase=PRE_TEST|BR_Day=11|Session=A"
TWP_POST = TB + "Test_Phase=POST_TEST|BR_Day=0|Session=D"

ARM_PROXY = ("fsh", "lh__miu_ml")
SUBJECT_ID_RE = re.compile(r"_(\d{4})$")
NEAR_DUP_R = 0.999

SPEC = {
    "fat": {
        "y": "total_fat_pct_change",
        "src": (FAT_PRE, FAT_POST),
        "published": {"stat": "mean of per-subject percent", "value": 4.2, "dz": 0.47, "n": 37},
        "outfile": "c11_totalfat_pct_change_features_all.csv",
    },
    "twp": {
        "y": "twp_abs_change",
        "src": (TWP_PRE, TWP_POST),
        "published": {"stat": "percent change of group means", "value": -60.7, "dz": -1.35, "n": 35},
        "outfile": "c11_twp_change_features_all.csv",
    },
}

FAT_DIRECT = "total_fat____idxa_cft70_body_composition_total"
TWP_DIRECT = ("twp____bedrest_ftt_cft70_linetest_twp",)
TWP_COMPONENT = ("percent_correct_steps____bedrest_ftt_cft70_linetest_pctcorrectste",
                 "time__sec__bedrest_ftt_cft70_data_linetest_trialtot")


def canon(v):
    s = str(v).strip()
    if s.upper().startswith("C1G"):
        return s.upper()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def leak_reason(col, which):
    c = col.lower()
    if which == "fat":
        if c == FAT_DIRECT:
            return "LEAKAGE-direct: Pre value of the outcome"
        if "body_composition" in c or "idxa" in c:
            return "LEAKAGE-algebraic: DXA body-composition (component/complement of total fat)"
    else:
        if c in TWP_DIRECT:
            return "LEAKAGE-direct: baseline value of the outcome"
        if c in TWP_COMPONENT:
            return "LEAKAGE-component: constituent of the TWP composite (verified |r|>0.5)"
    if any(t in c for t in ARM_PROXY):
        return "LEAKAGE-arm-proxy: LH/FSH near-deterministic for testosterone arm"
    m = SUBJECT_ID_RE.search(c)
    if m:
        return f"SINGLE-SUBJECT: per-subject CRF column for {m.group(1)} (n=1, fingerprint)"
    return None


def structural_filters(X):
    """FIX A + FIX B. Run on the FINAL cohort only. Returns (keep, log)."""
    log = []
    cols = list(X.columns)

    nun = X.nunique(dropna=True)
    const = [c for c in cols if nun[c] <= 1]
    for c in const:
        log.append((c, f"CONSTANT on final cohort (n_nonnull={int(X[c].notna().sum())})"))
    cols = [c for c in cols if c not in const]

    seen, exact = {}, []
    for c in cols:
        k = tuple(pd.util.hash_pandas_object(X[c].fillna(-9e99), index=False))
        if k in seen:
            exact.append(c)
            log.append((c, f"EXACT-DUPLICATE of {seen[k]}"))
        else:
            seen[k] = c
    cols = [c for c in cols if c not in exact]

    # FIX B: unit conversions and re-exports. O(k^2) on ~600 cols is fine.
    A = X.loc[:, cols]
    drop_near = set()
    for i, c1 in enumerate(cols):
        if c1 in drop_near:
            continue
        s1 = A[c1]
        for c2 in cols[i + 1:]:
            if c2 in drop_near:
                continue
            s2 = A[c2]
            m = s1.notna() & s2.notna()
            if m.sum() < 15:
                continue
            v1, v2 = s1[m], s2[m]
            if v1.nunique() < 2 or v2.nunique() < 2:
                continue
            r = abs(np.corrcoef(v1, v2)[0, 1])
            if r > NEAR_DUP_R:
                drop_near.add(c2)
                log.append((c2, f"NEAR-DUPLICATE of {c1} (|r|={r:.8f})"))
    cols = [c for c in cols if c not in drop_near]
    return cols, log


def build(which, feat, mas, outdir):
    spec = SPEC[which]
    ycol, (src_pre, src_post) = spec["y"], spec["src"]
    print("\n" + "#" * 68)
    print(f"# OUTCOME: {which}   ->  {ycol}")
    print("#" * 68)

    for c in (src_pre, src_post):
        if c not in mas.columns:
            sys.exit(f"FATAL: master missing\n  {c}")

    o = mas.loc[:, ["_s", src_pre, src_post]].dropna(subset=[src_pre, src_post]).copy()
    if o["_s"].duplicated().any():
        sys.exit("FATAL: duplicate subjects in outcome source")
    if which == "fat":
        o[ycol] = 100.0 * (o[src_post] - o[src_pre]) / o[src_pre]
    else:
        o[ycol] = o[src_post] - o[src_pre]

    fs, ms = set(feat["_s"]), set(o["_s"])
    print(f"subject join: features={len(fs)}  outcome={len(ms)}  final cohort={len(fs & ms)}")
    if fs - ms:
        print(f"  dropped, no outcome : {sorted(fs - ms)}")

    all_feats = [c for c in feat.columns if c not in (SUBJ, OLD_OUTCOME, "_s")]
    leak = {c: r for c in all_feats if (r := leak_reason(c, which))}
    stage1 = [c for c in all_feats if c not in leak]

    merged = (feat.loc[:, ["_s", SUBJ] + stage1]
              .merge(o.loc[:, ["_s", src_pre, src_post, ycol]], on="_s", how="inner")
              .drop(columns=["_s"]).sort_values(SUBJ).reset_index(drop=True))

    keeps, slog = structural_filters(merged.loc[:, stage1])   # FIX A: after join

    print(f"\nfeatures in : {len(all_feats)}")
    tags = {}
    for c, r in leak.items():
        tags.setdefault(r.split(":")[0], []).append(c)
    for t in sorted(tags, key=lambda k: -len(tags[k])):
        print(f"  {len(tags[t]):>4}  {t}")
    stags = {}
    for c, r in slog:
        stags.setdefault(r.split(" of ")[0].split(" (")[0], []).append((c, r))
    for t in sorted(stags, key=lambda k: -len(stags[k])):
        print(f"  {len(stags[t]):>4}  {t}   [FIX A/B, post-cohort]")
        for c, r in sorted(stags[t]):
            print(f"          {c[:52]:<52} {r.split(': ')[-1][:44]}")
    print(f"features out: {len(keeps)}")

    pre, post, ch = merged[src_pre], merged[src_post], merged[ycol]
    pub = spec["published"]
    print(f"\nREPRODUCTION  ({pub['stat']})")
    obs = ch.mean() if which == "fat" else 100 * (post.mean() - pre.mean()) / pre.mean()
    dz = (post - pre).mean() / (post - pre).std(ddof=1)
    print(f"  n {len(merged)} (published {pub['n']})   value {obs:+.2f} (published {pub['value']:+.1f})"
          f"   dz {dz:+.3f} (published {pub['dz']:+.2f})")
    if which == "twp":
        print(f"  post<=0 in {int((post <= 0).sum())}/{len(merged)} subjects -> absolute change used")

    X = merged.loc[:, keeps]
    cov = X.notna().mean()
    print(f"\nMISSINGNESS  overall {X.isna().mean().mean():.1%} | complete cols "
          f"{(cov == 1.0).sum()} | >=0.80 {(cov >= 0.80).sum()} | >=0.50 {(cov >= 0.50).sum()}"
          f" | complete rows {X.dropna().shape[0]}/{len(merged)}")

    out = merged.loc[:, [SUBJ] + keeps + [ycol]]
    p = os.path.join(outdir, spec["outfile"])
    out.to_csv(p, index=False)
    print(f"\nwrote {p}   {out.shape[0]} x {out.shape[1]}  ({len(keeps)} features)")

    ap_ = os.path.join(outdir, f"c11_{which}_outcome_audit.csv")
    merged.loc[:, [SUBJ, src_pre, src_post, ycol]].rename(
        columns={src_pre: "pre", src_post: "post"}).to_csv(ap_, index=False)
    print(f"wrote {ap_}   (pre/post audit trail, NOT for modelling)")
    return {"outcome": ycol, "n": out.shape[0], "features": len(keeps),
            "leak": len(leak), "structural": len(slog)}


def main():
    ap = argparse.ArgumentParser(description="Build Aim 1 Test 1 target tables (v3).")
    ap.add_argument("--bmd-table", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--outcome", default="both", choices=["fat", "twp", "both"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    feat = pd.read_csv(args.bmd_table)
    mas = pd.read_csv(args.master, low_memory=False).copy()
    feat["_s"] = feat[SUBJ].map(canon)
    mas["_s"] = mas["Subject"].map(canon)
    print(f"feature table {feat.shape[0]} x {feat.shape[1]} | master {mas.shape[0]} x {mas.shape[1]}")

    which = ["fat", "twp"] if args.outcome == "both" else [args.outcome]
    res = [build(w, feat, mas, args.outdir) for w in which]

    print("\n" + "=" * 68)
    print("RUN LOG LINES")
    print("=" * 68)
    for r in res:
        print(f"  build_aim1_targets_v3.py | outcome={r['outcome']} | n={r['n']} "
              f"| feats={r['features']} | dropped: leakage={r['leak']} structural={r['structural']}")
    print("\n  caveat: timepoint provenance not verifiable in the inherited feature")
    print("  names; baseline status asserted upstream, not checked here.")


if __name__ == "__main__":
    main()
