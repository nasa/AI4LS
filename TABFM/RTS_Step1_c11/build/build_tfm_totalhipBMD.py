#!/usr/bin/env python3
"""
build_tfm_totalhipBMD.py
========================
Deterministic builder for the C11 (NASA bed rest, UTMB) TabPFN-ready matrix.

Scientific question (Aim 1): can a tabular foundation model (TabPFN-3) give
stable, better-than-null predictions of the *change* in total-hip bone mineral
density (BMD) across a bed-rest campaign, at small n with heavy missingness?

  TARGET  y = totalhip_BMD_post - totalhip_BMD_pre   (g/cm^2, continuous)
            totalhip_BMD = mean(Femoral Left Total Hip, Femoral Right Total Hip)
  ROWS    one row per subject with valid Pre AND Post total-hip BMD (n ~ 38)
  X       Pre-phase numeric features, two tables on the SAME rows / same y:
            * everything (clean)  ~ all clean, non-leaking, non-dropped-dim cols
            * shortlist           ~ physiologist short-list (lean mass, weight,
                                    height, VO2peak, VCO2peak)

GROUND TRUTH = the raw archive (03_raw_downloads). master_long.csv is known to
have SILENTLY DROPPED a descriptor dimension (e.g. the ROI column of the iDXA
Hips file), so the target is rebuilt from the RAW Hips file, and any long-form
measure whose within-cell multiplicity is NOT explained by a genuine
trial/session/day key is treated as a dropped-dimension measure and EXCLUDED.

Rules (user-approved):
  * No imputation. NaN where missing. No mean/median fill, ever.
  * 0.0 -> NaN for EVERY DXA/BMD feature column (a 0.0 bone-density value is
    physiologically impossible). Non-DXA 0.0 values are left untouched.
  * Genuine trials (multiplicity explained by a varying key) -> MAX-of-trials.
  * Leakage: exclude ALL hip/femoral BMD columns from features.
  * Group/countermeasure arm and demographics are NOT features.
  * Deterministic: sorted rows/cols, fixed float format, SHA256 of outputs.

Run:  python build_tfm_totalhipBMD.py
Self-contained: reloads everything from disk; no notebook kernel state needed.
"""

import os
import re
import json
import hashlib
import datetime as _dt

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (overridable via env for testing)
# ----------------------------------------------------------------------------
RAW_HIPS = os.environ.get(
    "C11_RAW_HIPS",
    os.path.join(
        os.environ.get("C11_RAW_ARCHIVE", "data/raw"),
        "BEDREST_iRATS/BEDREST_IRATS_iDXA_CFT70/",
        "BEDREST_IRATS_iDXA_CFT70_Body_Composition_Hips_obsv.csv",
    ),
)
LONG_V2 = os.environ.get("C11_LONG_V2", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "c11_normalized_long_v2.csv"))
INV_V2 = os.environ.get("C11_INV_V2", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "c11_measure_inventory_v2.csv"))
OUTDIR = os.environ.get("C11_OUTDIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "bmd"))

FLOAT_FMT = "%.6f"
SEED = 42  # documented for the downstream modeling step (not used in this build)

# Descriptor keys that legitimately explode a measure across columns.
KEY_COLS = ["key_Trial", "key_Session", "key_BR_DAY", "key_TEST",
            "key_TEST_index", "key_TEST_type", "key_phase", "key_other"]

# Physiologist short-list concepts -> candidate measure_label substrings.
SHORTLIST = {
    "lean_mass": ["total lean", "lean"],
    "body_weight": ["weight", "wt"],
    "height": ["height", "ht"],
    "vo2_peak": ["vo2pk", "rel vo2pk", "%vo2max @ vt"],
    "vco2_peak": ["vco2pk"],
}

# Bone-biochemistry concepts we EXPECT but that are absent in C11 (flagged).
EXPECTED_ABSENT = ["calcium", "pth", "parathyroid", "vitamin d", "25-oh",
                   "ntx", "ctx", "p1np", "osteocalcin", "bone turnover",
                   "bone-specific alkaline", "bsap", "dpd", "deoxypyridinoline"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text):
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_").lower()
    return re.sub(r"_+", "_", s)


def source_tag(source_file):
    base = os.path.basename(str(source_file))
    base = re.sub(r"\.csv$", "", base, flags=re.I)
    base = re.sub(r"_(obsv|avgs)$", "", base, flags=re.I)
    base = re.sub(r"^(BEDREST[_ ]?)?(IRATS|iRATS)[_ ]?", "", base, flags=re.I)
    base = re.sub(r"^(CFT70|C11)[_ ]?", "", base, flags=re.I)
    return slugify(base)[:40] or "src"


# ----------------------------------------------------------------------------
# 1. TARGET from the RAW Hips file (ground truth)
# ----------------------------------------------------------------------------
def build_target():
    raw = pd.read_csv(RAW_HIPS, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    need = {"Subject", "ROI", "Unit", "Test", "Value"}
    missing = need - set(raw.columns)
    if missing:
        raise ValueError(f"Raw Hips file missing expected columns: {missing}")

    th = raw[raw["ROI"].isin(["Femoral Left Total Hip",
                              "Femoral Right Total Hip"])].copy()
    th = th[th["Test"].isin(["Pre", "Post"])].copy()
    th["val"] = pd.to_numeric(th["Value"], errors="coerce")
    n_zero = int((th["val"] == 0.0).sum())
    th.loc[th["val"] == 0.0, "val"] = np.nan  # 0.0 BMD is impossible -> missing

    th["side"] = np.where(th["ROI"].str.contains("Left"), "L", "R")
    piv = th.pivot_table(index="Subject", columns=["Test", "side"],
                         values="val", aggfunc="mean")
    piv.columns = [f"{t}_{s}" for t, s in piv.columns]
    for c in ["Pre_L", "Pre_R", "Post_L", "Post_R"]:
        if c not in piv.columns:
            piv[c] = np.nan

    piv["totalhip_BMD_pre"] = piv[["Pre_L", "Pre_R"]].mean(axis=1, skipna=True)
    piv["totalhip_BMD_post"] = piv[["Post_L", "Post_R"]].mean(axis=1, skipna=True)
    piv.loc[piv[["Pre_L", "Pre_R"]].isna().all(axis=1), "totalhip_BMD_pre"] = np.nan
    piv.loc[piv[["Post_L", "Post_R"]].isna().all(axis=1), "totalhip_BMD_post"] = np.nan
    piv["totalhip_BMD_change"] = piv["totalhip_BMD_post"] - piv["totalhip_BMD_pre"]

    audit = piv.reset_index().rename(columns={"Subject": "subject"})
    audit["subject"] = audit["subject"].astype(str)
    audit = audit[["subject", "Pre_L", "Pre_R", "totalhip_BMD_pre",
                   "Post_L", "Post_R", "totalhip_BMD_post", "totalhip_BMD_change"]]
    audit = audit.sort_values("subject").reset_index(drop=True)

    valid = audit.dropna(subset=["totalhip_BMD_pre", "totalhip_BMD_post"]).copy()
    target_subjects = sorted(valid["subject"].tolist())
    y = valid.set_index("subject")["totalhip_BMD_change"].sort_index()

    stats = {
        "target_subjects_n": int(len(target_subjects)),
        "target_zero_cells_converted": n_zero,
        "target_change_mean": float(y.mean()),
        "target_change_sd": float(y.std(ddof=1)),
        "target_change_min": float(y.min()),
        "target_change_max": float(y.max()),
        "target_net_loss_n": int((y < 0).sum()),
    }
    return audit, y, target_subjects, stats


# ----------------------------------------------------------------------------
# 2. Load long_v2 + inventory; detect dropped-dimension (affected) pairs
# ----------------------------------------------------------------------------
def load_long():
    long = pd.read_csv(LONG_V2, dtype=str, keep_default_na=False)
    long["value_numeric"] = pd.to_numeric(long["value_numeric"], errors="coerce")
    for c in KEY_COLS:
        if c not in long.columns:
            long[c] = ""
    return long


def load_inv():
    inv = pd.read_csv(INV_V2, dtype=str, keep_default_na=False)
    naf = set(inv.loc[inv["not_a_feature"].str.upper() == "TRUE", "measure_label"])
    return inv, naf


def detect_affected_pairs(long):
    df = long.copy()
    df["keysig"] = df[KEY_COLS].astype(str).agg("|".join, axis=1)
    grp = df.groupby(["source_file", "measure_label", "subject", "timepoint_raw"])
    agg = grp.agg(n_rows=("keysig", "size"), n_key=("keysig", "nunique")).reset_index()
    bad = agg[agg["n_rows"] > agg["n_key"]]
    affected = set(zip(bad["source_file"], bad["measure_label"]))
    return affected, agg


def is_hip_femoral_bmd(source_file, measure_label):
    s = str(source_file).lower()
    m = str(measure_label).lower()
    hip_src = ("hip" in s) and any(t in s for t in ["bmd", "bmc", "bone", "composition"])
    femoral_m = any(t in m for t in ["femoral", "troch", "ward", "neck"])
    hip_m = ("hip" in m) and ("bmd" in m or "bone" in m)
    return hip_src or femoral_m or hip_m


# Demographics / identifiers / campaign tags are NOT features (user rule).
# Age & Sex are subject descriptors; 'Iteration' is a campaign-iteration tag;
# the iDXA/MR079G Demographics files only carry HT/WT/Age/Sex/Iteration.
DEMO_LABELS = {"age", "sex", "iteration", "subject", "id", "group", "groupname"}
DEMO_SRC_TOKEN = "demographic"


def is_demographic(source_file, measure_label):
    s = str(source_file).lower()
    m = str(measure_label).strip().lower()
    return (m in DEMO_LABELS) or (DEMO_SRC_TOKEN in s)


# ----------------------------------------------------------------------------
# 3. FEATURES
# ----------------------------------------------------------------------------
def build_features(long, naf, affected, target_subjects):
    feat = long[(long["phase_coarse"] == "Pre")].copy()
    feat = feat[~feat["measure_label"].isin(naf)].copy()

    feat["pair"] = list(zip(feat["source_file"], feat["measure_label"]))
    feat = feat[~feat["pair"].isin(affected)].copy()
    leak_mask = feat.apply(lambda r: is_hip_femoral_bmd(r["source_file"],
                                                        r["measure_label"]), axis=1)
    n_leak_pairs = feat.loc[leak_mask, "pair"].nunique()
    feat = feat[~leak_mask].copy()

    demo_mask = feat.apply(lambda r: is_demographic(r["source_file"],
                                                    r["measure_label"]), axis=1)
    n_demo_pairs = feat.loc[demo_mask, "pair"].nunique()
    feat = feat[~demo_mask].copy()

    # Amendment 2: 0.0 -> NaN for DXA/BMD feature columns only
    is_dxa = feat["source_file"].str.lower().str.contains("dxa")
    zero_dxa = is_dxa & (feat["value_numeric"] == 0.0)
    dxa_zero_by_pair = feat.loc[zero_dxa].groupby("pair").size().to_dict()
    feat.loc[zero_dxa, "value_numeric"] = np.nan

    # Anthropometric guard: a 0.0 height/weight is physiologically impossible
    # (distinct from legitimately-zero scores like POMS). Blank those to NaN.
    mlow = feat["measure_label"].str.strip().str.lower()
    is_anthro = mlow.isin(["height", "ht", "weight", "wt"])
    zero_anthro = is_anthro & (feat["value_numeric"] == 0.0)
    n_anthro_zero = int(zero_anthro.sum())
    feat.loc[zero_anthro, "value_numeric"] = np.nan

    feat_t = feat[feat["subject"].isin(target_subjects)].copy()

    # collapse genuine trials -> MAX of value_numeric per (pair, subject)
    collapsed = (feat_t.groupby(["source_file", "measure_label", "unit", "subject"],
                                dropna=False)["value_numeric"]
                 .agg(["max", "count", "nunique"])
                 .reset_index())
    collapsed = collapsed.rename(columns={"max": "value"})

    pair_diag = (collapsed.groupby(["source_file", "measure_label", "unit"])
                 .agg(n_subj=("value", lambda s: int(s.notna().sum())),
                      n_multi=("count", lambda s: int((s > 1).sum())),
                      max_reps=("count", "max"))
                 .reset_index())

    keep_pairs = pair_diag[pair_diag["n_subj"] >= 1][
        ["source_file", "measure_label", "unit"]]
    collapsed = collapsed.merge(keep_pairs, on=["source_file", "measure_label", "unit"],
                                how="inner")

    # One column per distinct measure identity (measure_label, unit, source_file).
    # Build the slug at the MEASURE level (not per row) so all subjects sharing a
    # measure collapse into a single column. Suffix only on a genuine identity
    # collision (two different measures mapping to the same readable slug).
    meas = (collapsed[["source_file", "measure_label", "unit"]].drop_duplicates()
            .sort_values(["source_file", "measure_label", "unit"]).reset_index(drop=True))
    meas["slug_base"] = (meas["measure_label"].map(slugify) + "__" +
                         meas["unit"].map(slugify) + "__" +
                         meas["source_file"].map(source_tag))
    meas_counts = {}
    mslugs = []
    for b in meas["slug_base"]:
        meas_counts[b] = meas_counts.get(b, 0) + 1
        mslugs.append(b if meas_counts[b] == 1 else f"{b}__{meas_counts[b]}")
    meas["slug"] = mslugs
    n_collisions = int((meas["slug_base"].duplicated(keep=False)).sum())

    collapsed = collapsed.merge(meas, on=["source_file", "measure_label", "unit"],
                                how="left")

    wide = collapsed.pivot_table(index="subject", columns="slug", values="value",
                                 aggfunc="first")
    wide = wide.reindex(index=target_subjects)
    wide = wide.reindex(columns=sorted(wide.columns))
    wide.index.name = "subject"

    dictionary = (meas[["slug", "measure_label", "unit", "source_file"]]
                  .sort_values("slug").reset_index(drop=True))
    dictionary = dictionary.merge(
        pair_diag.rename(columns={"n_subj": "n_subjects_with_value",
                                  "n_multi": "n_subjects_multitrial",
                                  "max_reps": "max_trials_in_cell"}),
        on=["source_file", "measure_label", "unit"], how="left")
    dictionary["is_dxa"] = dictionary["source_file"].str.lower().str.contains("dxa")
    dictionary["dxa_zero_to_nan"] = dictionary.apply(
        lambda r: int(dxa_zero_by_pair.get((r["source_file"], r["measure_label"]), 0)),
        axis=1)

    miss = pd.DataFrame({"slug": wide.columns,
                         "n_missing": wide.isna().sum().values})
    miss["n_present"] = len(wide) - miss["n_missing"]
    dictionary = dictionary.merge(miss, on="slug", how="left")

    stats = {
        "feature_columns_n": int(wide.shape[1]),
        "slug_identity_collisions_n": n_collisions,
        "leak_pairs_excluded_n": int(n_leak_pairs),
        "demo_pairs_excluded_n": int(n_demo_pairs),
        "dxa_zero_cells_converted_n": int(sum(dxa_zero_by_pair.values())),
        "dxa_columns_with_zero_n": int(len(dxa_zero_by_pair)),
        "anthro_zero_cells_converted_n": n_anthro_zero,
        "features_ge20_subj": int((dictionary["n_present"] >= 20).sum()),
        "features_ge30_subj": int((dictionary["n_present"] >= 30).sum()),
        "features_singleton": int((dictionary["n_present"] == 1).sum()),
    }
    return wide, dictionary, stats


# Preferred source_file substrings per concept, ranked. The mr080g exercise
# file records height/weight in cm/kg (clean, no zeros); the vertical-jump file
# records height in inches and had an impossible 0.0, so it is ranked last.
SHORTLIST_SRC_PREF = {
    "lean_mass": ["body_composition_total", "body_composition_legs", "body_composition_android"],
    "body_weight": ["mr080g_cft70_pre_all", "mr080g", "vitals", "vertical_jump"],
    "height": ["mr080g_cft70_pre_all", "mr080g", "vitals", "vertical_jump"],
    "vo2_peak": ["mr080g_cft70_pre_all", "mr080g"],
    "vco2_peak": ["mr080g_cft70_pre_all", "mr080g"],
}


def build_shortlist(wide, dictionary):
    chosen = {}
    for concept, pats in SHORTLIST.items():
        pref = SHORTLIST_SRC_PREF.get(concept, [])

        def rank(slug):
            s = slug.lower()
            for i, p in enumerate(pref):
                if p in s:
                    return i
            return len(pref)

        cands = []
        for slug in wide.columns:
            lab = dictionary.loc[dictionary["slug"] == slug, "measure_label"]
            lab = lab.iloc[0].lower() if len(lab) else ""
            if any(p in lab for p in pats):
                n_present = int(wide[slug].notna().sum())
                cands.append((slug, n_present))
        if cands:
            # preferred source first, then best coverage, then slug (deterministic)
            cands.sort(key=lambda t: (rank(t[0]), -t[1], t[0]))
            chosen[concept] = cands[0]
    cols = [v[0] for k, v in sorted(chosen.items())]
    short = wide[cols].copy() if cols else pd.DataFrame(index=wide.index)
    return short, chosen


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    log = []

    def say(msg):
        print(msg)
        log.append(msg)

    say(f"[build] {_dt.datetime.now().isoformat(timespec='seconds')}")
    say(f"[build] RAW_HIPS = {RAW_HIPS}")
    say(f"[build] LONG_V2  = {LONG_V2}")
    say(f"[build] INV_V2   = {INV_V2}")
    say(f"[build] OUTDIR   = {OUTDIR}")

    audit, y, target_subjects, tstats = build_target()
    say(f"[target] subjects with valid Pre&Post total-hip BMD: {tstats['target_subjects_n']}")
    say(f"[target] 0.0->NaN cells in Pre/Post total hip: {tstats['target_zero_cells_converted']}")
    say(f"[target] change mean={tstats['target_change_mean']:.6f} "
        f"sd={tstats['target_change_sd']:.6f} "
        f"range=[{tstats['target_change_min']:.6f},{tstats['target_change_max']:.6f}] "
        f"net_loss={tstats['target_net_loss_n']}/{tstats['target_subjects_n']}")

    long = load_long()
    inv, naf = load_inv()
    affected, _agg = detect_affected_pairs(long)
    say(f"[features] dropped-dimension (affected) pairs excluded: {len(affected)}")

    wide, dictionary, fstats = build_features(long, naf, affected, target_subjects)
    say(f"[features] clean feature columns (>=1 value among 38): {fstats['feature_columns_n']}")
    say(f"[features] hip/femoral-BMD leak pairs excluded: {fstats['leak_pairs_excluded_n']}")
    say(f"[features] demographic/identifier pairs excluded: {fstats['demo_pairs_excluded_n']}")
    say(f"[features] DXA 0.0->NaN cells: {fstats['dxa_zero_cells_converted_n']} "
        f"across {fstats['dxa_columns_with_zero_n']} columns")
    say(f"[features] anthropometric (height/weight) 0.0->NaN cells: "
        f"{fstats['anthro_zero_cells_converted_n']}")
    say(f"[features] coverage: >=20 subj={fstats['features_ge20_subj']}, "
        f">=30 subj={fstats['features_ge30_subj']}, singletons={fstats['features_singleton']}")

    all_tbl = wide.copy()
    all_tbl["totalhip_BMD_change"] = y
    short, chosen = build_shortlist(wide, dictionary)
    short_tbl = short.copy()
    short_tbl["totalhip_BMD_change"] = y

    say("[shortlist] canonical columns chosen (concept: slug [n_present]):")
    for concept, (slug, n_pres) in sorted(chosen.items()):
        say(f"    {concept:12s}: {slug}  [n={n_pres}]")

    all_labels = " ".join(dictionary["measure_label"].str.lower().unique())
    absent_found = [c for c in EXPECTED_ABSENT if c in all_labels]
    say(f"[shortlist] bone-biochemistry predictors present in C11 clean set: "
        f"{absent_found if absent_found else 'NONE (calcium/PTH/vitamin-D/turnover markers absent)'}")

    excluded_rows = []
    for (sf, ml) in sorted(affected):
        excluded_rows.append({"source_file": sf, "measure_label": ml,
                              "reason": "dropped_dimension"})
    excluded = pd.DataFrame(excluded_rows, columns=["source_file", "measure_label", "reason"])

    p_all = os.path.join(OUTDIR, "c11_totalhipBMD_change_features_all.csv")
    p_short = os.path.join(OUTDIR, "c11_totalhipBMD_change_features_shortlist.csv")
    p_audit = os.path.join(OUTDIR, "c11_totalhipBMD_target_audit.csv")
    p_dict = os.path.join(OUTDIR, "c11_tfm_feature_dictionary.csv")
    p_excl = os.path.join(OUTDIR, "c11_tfm_excluded_columns.csv")
    p_stats = os.path.join(OUTDIR, "build_stats.json")
    p_log = os.path.join(OUTDIR, "run_log_tfm_c11_totalhipBMD.txt")

    all_tbl.to_csv(p_all, float_format=FLOAT_FMT)
    short_tbl.to_csv(p_short, float_format=FLOAT_FMT)
    audit.to_csv(p_audit, index=False, float_format=FLOAT_FMT)
    dictionary.to_csv(p_dict, index=False)
    excluded.to_csv(p_excl, index=False)

    stats = {"target": tstats, "features": fstats,
             "shortlist": {k: {"slug": v[0], "n_present": v[1]}
                           for k, v in sorted(chosen.items())},
             "absent_bone_biochem": absent_found,
             "seed_documented_for_modeling": SEED}
    with open(p_stats, "w") as f:
        json.dump(stats, f, indent=2)

    say("\n[missingness] per-feature (present/total) -- top 40 most complete:")
    dd = dictionary.sort_values(["n_present", "slug"], ascending=[False, True])
    for _, r in dd.head(40).iterrows():
        say(f"    {r['n_present']:>2}/38  {r['slug']}")
    say(f"[missingness] ... ({dictionary.shape[0]} features total; "
        f"see c11_tfm_feature_dictionary.csv for full per-feature counts)")

    say("\n[sha256]")
    for p in [p_all, p_short, p_audit, p_dict, p_excl, p_stats]:
        say(f"    {sha256(p)}  {os.path.basename(p)}")

    with open(p_log, "w") as f:
        f.write("\n".join(log) + "\n")

    say(f"\n[build] DONE -> {OUTDIR}")
    return stats


if __name__ == "__main__":
    main()
