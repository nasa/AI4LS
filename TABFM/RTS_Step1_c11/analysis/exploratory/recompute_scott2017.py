#!/usr/bin/env python3
"""
recompute_scott2017.py — Recompute Scott 2017 ultrasound-vs-MRI concordance
statistics from master_long.csv.

Scott 2017 (JCSM 8:475-81, DOI 10.1002/jcsm.12172) compared panoramic
ultrasound to MRI for measuring muscle CSA change during 70-day bed rest.
This script reproduces 10 published statistics from the raw archive data.

METHODOLOGY (from the paper):
  - Change scores: each timepoint value minus BR-6 (PRE_TEST|BR_Day=6) baseline
  - Two groups combined (exercise + sedentary), arm-blind
  - Quadriceps = combined CSA (all four heads)
  - Gastrocnemius = lateral + medial gastrocnemius summed
  - Rectus femoris = separate (internal null)
  - CCC: Lin's concordance correlation coefficient on paired change scores
  - Bland-Altman: diff = MRI_change - US_change, 95% LoA = mean +/- 1.96*SD
  - Atrophy = change < 0, hypertrophy = change > 0 (directional dichotomization)
  - Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP)
  - PPV = TP/(TP+FP), NPV = TN/(TN+FN)

PUBLISHED VALUES (Table 1 + text):
  Quadriceps:    CCC 0.78, Sens 73.7%, Spec 74.2%, PPV 66.7%, NPV 80.0%
  Gastrocnemius: CCC 0.37, Sens 83.1%, Spec 33.0%, PPV 72.5%, NPV 47.9%
  Rectus femoris: CCC = not published (internal null, reproduced as ~0)
  LoA bounds: +/- 5 cm^2 (quadriceps), +/- 3 cm^2 (gastrocnemius)

OUTPUT:
  results/scott2017_concordance_summary.csv — summary statistics only
  (no per-subject values; subject-level boundary rule applies)

USAGE:
    python code/analysis/recompute_scott2017.py
    python code/analysis/recompute_scott2017.py --master /path/to/master_long.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Muscles with both MRI and Ultrasound
MUSCLES_DUAL = ["Quadriceps", "Rectus_Femoris"]
GASTROCNEMII = ["Lateral_Gastrocnemius", "Medial_Gastrocnemius"]

BASELINE_TIMEPOINT = "Test_Phase=PRE_TEST|BR_Day=6"


def lins_ccc(x, y):
    """Lin's concordance correlation coefficient.

    CCC = 2 * cov(x,y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

    Measures agreement between two methods (1 = perfect agreement).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return 0.0
    return 2 * cov_xy / denom


def bland_altman(x, y):
    """Bland-Altman analysis.

    Returns (mean_diff, sd_diff, lower_loa, upper_loa).
    diff = x - y (MRI - US), LoA = mean +/- 1.96 * SD.
    """
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    lower = mean_diff - 1.96 * sd_diff
    upper = mean_diff + 1.96 * sd_diff
    return mean_diff, sd_diff, lower, upper


def diagnostic_accuracy(mri_change, us_change):
    """Compute sensitivity, specificity, PPV, NPV.

    Atrophy = change < 0 (MRI reference).
    Hypertrophy = change > 0 (MRI reference).
    """
    mri_change = np.asarray(mri_change, dtype=float)
    us_change = np.asarray(us_change, dtype=float)

    # Dichotomize: atrophy (True) vs hypertrophy (False)
    mri_atrophy = mri_change < 0
    us_atrophy = us_change < 0

    tp = np.sum(mri_atrophy & us_atrophy)      # MRI atrophy, US atrophy
    fn = np.sum(mri_atrophy & ~us_atrophy)      # MRI atrophy, US hypertrophy
    tn = np.sum(~mri_atrophy & ~us_atrophy)     # MRI hypertrophy, US hypertrophy
    fp = np.sum(~mri_atrophy & us_atrophy)      # MRI hypertrophy, US atrophy

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def load_muscle_data(df, modality, muscle):
    """Load data for one muscle and modality from master_long.

    Returns a DataFrame with Subject, timepoint, value (float).
    """
    prefix = "MRI" if modality == "MRI" else "Ultrasound"
    source_file = f"BEDREST_IRATS_MRI_ULTRASOUND_CFT70_{prefix}_{muscle}"
    sub = df[(df["folder"] == "BEDREST_iRATS") &
             (df["source_file"] == source_file)].copy()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub[["Subject", "timepoint", "value"]].copy()
    sub = sub.dropna(subset=["value"])
    return sub


def compute_change_scores(df_mri, df_us):
    """Compute change scores (value - baseline) for paired MRI/US data.

    Returns (mri_change, us_change, n_pairs).
    """
    # Pivot to wide: Subject x timepoint -> value
    mri_wide = df_mri.pivot_table(index="Subject", columns="timepoint",
                                   values="value", aggfunc="first")
    us_wide = df_us.pivot_table(index="Subject", columns="timepoint",
                                 values="value", aggfunc="first")

    # Get baseline (BR-6)
    mri_baseline = mri_wide[BASELINE_TIMEPOINT]
    us_baseline = us_wide[BASELINE_TIMEPOINT]

    # Compute change scores for all non-baseline timepoints
    non_baseline_tp = [tp for tp in mri_wide.columns if tp != BASELINE_TIMEPOINT]

    mri_changes = []
    us_changes = []

    for subj in mri_wide.index:
        if subj not in us_wide.index:
            continue
        mri_base = mri_baseline.get(subj)
        us_base = us_baseline.get(subj)
        if pd.isna(mri_base) or pd.isna(us_base):
            continue
        for tp in non_baseline_tp:
            mri_val = mri_wide.loc[subj, tp] if tp in mri_wide.columns else np.nan
            us_val = us_wide.loc[subj, tp] if tp in us_wide.columns else np.nan
            if pd.isna(mri_val) or pd.isna(us_val):
                continue
            mri_changes.append(mri_val - mri_base)
            us_changes.append(us_val - us_base)

    return np.array(mri_changes), np.array(us_changes), len(mri_changes)


def compute_combined_gastrocnemius(df, modality):
    """Load lateral + medial gastrocnemius and sum them per subject/timepoint."""
    lat = load_muscle_data(df, modality, "Lateral_Gastrocnemius")
    med = load_muscle_data(df, modality, "Medial_Gastrocnemius")

    # Merge on Subject + timepoint, sum values
    merged = lat.merge(med, on=["Subject", "timepoint"], suffixes=("_lat", "_med"))
    merged["value"] = merged["value_lat"] + merged["value_med"]
    return merged[["Subject", "timepoint", "value"]]


def main():
    parser = argparse.ArgumentParser(
        description="Recompute Scott 2017 concordance statistics from master_long.csv")
    parser.add_argument("--master", default=None,
                        help="Path to master_long.csv")
    parser.add_argument("--output", default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    master_path = args.master or os.environ.get(
        "MASTER_LONG_PATH",
        os.path.join(REPO_ROOT, "data", "master_long.csv"),
    )
    output_path = args.output or os.path.join(
        REPO_ROOT, "results", "scott2017_concordance_summary.csv")

    if not os.path.exists(master_path):
        print(f"ERROR: master_long.csv not found at {master_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading master_long.csv from {master_path}...", flush=True)
    df = pd.read_csv(master_path, low_memory=False)
    print(f"  Total rows: {len(df)}", flush=True)

    # Published values for comparison
    published = {
        "Quadriceps_CCC": 0.78,
        "Quadriceps_sensitivity": 73.7,
        "Quadriceps_specificity": 74.2,
        "Quadriceps_PPV": 66.7,
        "Quadriceps_NPV": 80.0,
        "Quadriceps_LoA_lower": -5.0,
        "Quadriceps_LoA_upper": 5.0,
        "Gastrocnemius_CCC": 0.37,
        "Gastrocnemius_sensitivity": 83.1,
        "Gastrocnemius_specificity": 33.0,
        "Gastrocnemius_PPV": 72.5,
        "Gastrocnemius_NPV": 47.9,
        "Gastrocnemius_LoA_lower": -3.0,
        "Gastrocnemius_LoA_upper": 3.0,
        "Rectus_Femoris_CCC": None,  # internal null, not published
    }

    results = []

    # ── Quadriceps ──────────────────────────────────────────────────────
    print("\n=== Quadriceps ===", flush=True)
    mri_q = load_muscle_data(df, "MRI", "Quadriceps")
    us_q = load_muscle_data(df, "Ultrasound", "Quadriceps")
    mri_ch, us_ch, n = compute_change_scores(mri_q, us_q)
    ccc_q = lins_ccc(mri_ch, us_ch)
    md_q, sd_q, lo_q, hi_q = bland_altman(mri_ch, us_ch)
    diag_q = diagnostic_accuracy(mri_ch, us_ch)
    print(f"  n_pairs={n}, CCC={ccc_q:.3f}", flush=True)
    print(f"  Bland-Altman: bias={md_q:.3f}, SD={sd_q:.3f}, "
          f"LoA=[{lo_q:.3f}, {hi_q:.3f}]", flush=True)
    print(f"  Sens={diag_q['sensitivity']*100:.1f}%, "
          f"Spec={diag_q['specificity']*100:.1f}%, "
          f"PPV={diag_q['ppv']*100:.1f}%, "
          f"NPV={diag_q['npv']*100:.1f}%", flush=True)
    print(f"  Published: CCC=0.78, Sens=73.7%, Spec=74.2%, PPV=66.7%, NPV=80.0%", flush=True)

    results.extend([
        {"muscle": "Quadriceps", "statistic": "CCC", "published": 0.78,
         "recomputed": round(ccc_q, 3), "n_pairs": n},
        {"muscle": "Quadriceps", "statistic": "sensitivity_pct", "published": 73.7,
         "recomputed": round(diag_q["sensitivity"] * 100, 1), "n_pairs": n},
        {"muscle": "Quadriceps", "statistic": "specificity_pct", "published": 74.2,
         "recomputed": round(diag_q["specificity"] * 100, 1), "n_pairs": n},
        {"muscle": "Quadriceps", "statistic": "PPV_pct", "published": 66.7,
         "recomputed": round(diag_q["ppv"] * 100, 1), "n_pairs": n},
        {"muscle": "Quadriceps", "statistic": "NPV_pct", "published": 80.0,
         "recomputed": round(diag_q["npv"] * 100, 1), "n_pairs": n},
        {"muscle": "Quadriceps", "statistic": "LoA_lower", "published": -5.0,
         "recomputed": round(lo_q, 3), "n_pairs": n},
        {"muscle": "Quadriceps", "statistic": "LoA_upper", "published": 5.0,
         "recomputed": round(hi_q, 3), "n_pairs": n},
    ])

    # ── Gastrocnemius (lateral + medial combined) ───────────────────────
    print("\n=== Gastrocnemius (lateral + medial combined) ===", flush=True)
    mri_g = compute_combined_gastrocnemius(df, "MRI")
    us_g = compute_combined_gastrocnemius(df, "Ultrasound")
    mri_ch_g, us_ch_g, n_g = compute_change_scores(mri_g, us_g)
    ccc_g = lins_ccc(mri_ch_g, us_ch_g)
    md_g, sd_g, lo_g, hi_g = bland_altman(mri_ch_g, us_ch_g)
    diag_g = diagnostic_accuracy(mri_ch_g, us_ch_g)
    print(f"  n_pairs={n_g}, CCC={ccc_g:.3f}", flush=True)
    print(f"  Bland-Altman: bias={md_g:.3f}, SD={sd_g:.3f}, "
          f"LoA=[{lo_g:.3f}, {hi_g:.3f}]", flush=True)
    print(f"  Sens={diag_g['sensitivity']*100:.1f}%, "
          f"Spec={diag_g['specificity']*100:.1f}%, "
          f"PPV={diag_g['ppv']*100:.1f}%, "
          f"NPV={diag_g['npv']*100:.1f}%", flush=True)
    print(f"  Published: CCC=0.37, Sens=83.1%, Spec=33.0%, PPV=72.5%, NPV=47.9%", flush=True)

    results.extend([
        {"muscle": "Gastrocnemius", "statistic": "CCC", "published": 0.37,
         "recomputed": round(ccc_g, 3), "n_pairs": n_g},
        {"muscle": "Gastrocnemius", "statistic": "sensitivity_pct", "published": 83.1,
         "recomputed": round(diag_g["sensitivity"] * 100, 1), "n_pairs": n_g},
        {"muscle": "Gastrocnemius", "statistic": "specificity_pct", "published": 33.0,
         "recomputed": round(diag_g["specificity"] * 100, 1), "n_pairs": n_g},
        {"muscle": "Gastrocnemius", "statistic": "PPV_pct", "published": 72.5,
         "recomputed": round(diag_g["ppv"] * 100, 1), "n_pairs": n_g},
        {"muscle": "Gastrocnemius", "statistic": "NPV_pct", "published": 47.9,
         "recomputed": round(diag_g["npv"] * 100, 1), "n_pairs": n_g},
        {"muscle": "Gastrocnemius", "statistic": "LoA_lower", "published": -3.0,
         "recomputed": round(lo_g, 3), "n_pairs": n_g},
        {"muscle": "Gastrocnemius", "statistic": "LoA_upper", "published": 3.0,
         "recomputed": round(hi_g, 3), "n_pairs": n_g},
    ])

    # ── Rectus Femoris (internal null) ──────────────────────────────────
    print("\n=== Rectus Femoris (internal null) ===", flush=True)
    mri_rf = load_muscle_data(df, "MRI", "Rectus_Femoris")
    us_rf = load_muscle_data(df, "Ultrasound", "Rectus_Femoris")
    mri_ch_rf, us_ch_rf, n_rf = compute_change_scores(mri_rf, us_rf)
    ccc_rf = lins_ccc(mri_ch_rf, us_ch_rf)
    print(f"  n_pairs={n_rf}, CCC={ccc_rf:.3f}", flush=True)
    print(f"  Published: not published (internal null, expected ~0)", flush=True)

    results.append({
        "muscle": "Rectus_Femoris", "statistic": "CCC", "published": None,
        "recomputed": round(ccc_rf, 3), "n_pairs": n_rf,
    })

    # ── Save summary CSV ────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved summary to {output_path}", flush=True)

    # ── Comparison table ────────────────────────────────────────────────
    print("\n=== Comparison: Published vs Recomputed ===", flush=True)
    print(f"{'Muscle':<16} {'Statistic':<16} {'Published':>10} "
          f"{'Recomputed':>10} {'Match':>6}", flush=True)
    print("-" * 62, flush=True)
    for _, row in results_df.iterrows():
        pub = row["published"]
        rec = row["recomputed"]
        if pub is not None and not np.isnan(pub):
            # For CCC, match within 0.05; for percentages, within 5%; for LoA, check inside bounds
            if "LoA" in row["statistic"]:
                if "lower" in row["statistic"]:
                    match = rec >= pub  # lower LoA should be >= published lower bound
                else:
                    match = rec <= pub  # upper LoA should be <= published upper bound
                match_str = "INSIDE" if match else "OUTSIDE"
            elif "CCC" in row["statistic"]:
                match = abs(rec - pub) < 0.05
                match_str = "YES" if match else "NO"
            else:
                match = abs(rec - pub) < 5.0
                match_str = "YES" if match else "NO"
            print(f"{row['muscle']:<16} {row['statistic']:<16} {pub:>10.1f} "
                  f"{rec:>10.1f} {match_str:>6}", flush=True)
        else:
            print(f"{row['muscle']:<16} {row['statistic']:<16} {'N/A':>10} "
                  f"{rec:>10.1f} {'NULL':>6}", flush=True)


if __name__ == "__main__":
    main()
