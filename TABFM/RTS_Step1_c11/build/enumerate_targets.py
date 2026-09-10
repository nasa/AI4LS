#!/usr/bin/env python3
"""
enumerate_targets.py

Reads the C11 master and emits a catalogue of every target that can be run,
one row per target, ranked by effect size. Replaces writing a bespoke builder
script per outcome.

The master encodes every column as  measure::source_file::timepoint. That is
enough structure to enumerate targets mechanically:

  Shape 2 (before-to-after change)
      group columns by (measure, source_file), classify each timepoint as
      PRE / IN / POST, then every (PRE, POST) pair with enough paired subjects
      is a candidate. Effect size and percent change computed on the spot.

  Shape 1 (same-time prediction)
      any (measure, file, timepoint) with enough subjects is a candidate
      outcome; the count of OTHER measures observed at the same timepoint is
      the available feature breadth.

  Shape 3 (next-point prediction)
      any (measure, file) with enough IN_TEST timepoints and enough subjects
      measured at all of them.

Two columns are emitted that exist to prevent the defects found by hand:

  leakage_family  the source_file. Every other column from the same file is a
                  leakage candidate against this outcome and must be screened,
                  not assumed safe. This is how Percent_Correct_Steps (r=0.83
                  with baseline TWP) would have been caught automatically.

  pct_safe        whether percent change is usable. False when any post value
                  is <= 0, because percent change then becomes unstable. This
                  is the tandem walk case, where one subject computed to -1086%
                  and the three percent definitions disagreed by 43 points.

Usage:
  python3 enumerate_targets.py --master master_table.csv --outdir . [--min-n 20]
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

SUBJ = "Subject"

# Timepoint phase, read off the label. PRE = before bed rest, IN = during,
# POST = recovery. Files vary in casing and separator, so match loosely.
def phase(tp):
    t = tp.upper()
    if "PRE" in t or "SCREEN" in t or re.search(r"TEST=PRE", t):
        return "PRE"
    if "POST" in t:
        return "POST"
    if "IN_TEST" in t or "IN-TEST" in t:
        return "IN"
    return "OTHER"


def br_day(tp):
    """Bed-rest day if the label carries one, else NaN. Used only for ordering."""
    m = re.search(r"BR[_ ]?DAY\s*=\s*([0-9.]+)", tp, re.I)
    return float(m.group(1)) if m else np.nan


def split_col(c):
    p = c.split("::")
    if len(p) == 3:
        return p[0], p[1], p[2]
    if len(p) == 2:
        return p[0], p[1], ""
    return c, "", ""


def dz(diff):
    d = pd.Series(diff).dropna()
    s = d.std(ddof=1)
    return float("nan") if not s else d.mean() / s


def main():
    ap = argparse.ArgumentParser(description="Catalogue runnable targets from the master.")
    ap.add_argument("--master", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--min-n", type=int, default=20,
                    help="minimum paired subjects for a target to be listed")
    ap.add_argument("--min-timepoints", type=int, default=4,
                    help="minimum IN_TEST timepoints for a Shape 3 candidate")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    m = pd.read_csv(a.master, low_memory=False)
    if SUBJ not in m.columns:
        raise SystemExit(f"FATAL: no '{SUBJ}' column in {a.master}")
    print(f"master: {m.shape[0]} subjects x {m.shape[1]} columns")

    # numeric only -- a target must be a number
    num = m.select_dtypes("number")
    cols = [c for c in num.columns if c != SUBJ]
    print(f"numeric measure columns: {len(cols)}")

    groups = defaultdict(list)          # (measure, file) -> [(timepoint, col)]
    at_timepoint = defaultdict(set)     # (file, timepoint) -> {measure}
    for c in cols:
        meas, f, tp = split_col(c)
        groups[(meas, f)].append((tp, c))
        at_timepoint[(f, tp)].add(meas)

    print(f"distinct measure x file groups: {len(groups)}")

    rows = []

    # ---------------------------------------------------------------- Shape 2
    for (meas, f), items in groups.items():
        pres = [(tp, c) for tp, c in items if phase(tp) == "PRE"]
        posts = [(tp, c) for tp, c in items if phase(tp) == "POST"]
        if not pres or not posts:
            continue
        # deepest post (largest BR day) and the pre with the most coverage:
        # matches how the verified targets were chosen by hand
        pres.sort(key=lambda x: -num[x[1]].notna().sum())
        posts.sort(key=lambda x: (-(br_day(x[0]) if not np.isnan(br_day(x[0])) else -1),
                                  -num[x[1]].notna().sum()))
        for tp_pre, c_pre in pres[:2]:
            for tp_post, c_post in posts[:2]:
                d = pd.DataFrame({"pre": num[c_pre], "post": num[c_post]}).dropna()
                if len(d) < a.min_n:
                    continue
                pct_safe = bool((d["post"] > 0).all() and (d["pre"] > 0).all())
                pct = (100 * (d["post"].mean() - d["pre"].mean()) / d["pre"].mean()
                       if d["pre"].mean() else float("nan"))
                rows.append({
                    "target_id": f"s2_{re.sub(r'[^A-Za-z0-9]+','_',meas).strip('_').lower()[:34]}",
                    "shape": 2, "measure": meas, "leakage_family": f,
                    "pre_col": c_pre, "post_col": c_post,
                    "n_paired": len(d),
                    "group_pct_of_means": round(pct, 2),
                    "effect_size": round(dz(d["post"] - d["pre"]), 3),
                    "abs_effect_size": round(abs(dz(d["post"] - d["pre"])), 3),
                    "pct_safe": pct_safe,
                    "recommended_scale": "percent" if pct_safe else "absolute",
                    "family_size": len(items),
                    "notes": "",
                })

    # ---------------------------------------------------------------- Shape 1
    for (f, tp), measures in at_timepoint.items():
        if len(measures) < 2 or phase(tp) == "OTHER":
            continue
        for meas in measures:
            c = f"{meas}::{f}::{tp}" if tp else f"{meas}::{f}"
            if c not in num.columns:
                continue
            n = int(num[c].notna().sum())
            if n < a.min_n:
                continue
            rows.append({
                "target_id": f"s1_{re.sub(r'[^A-Za-z0-9]+','_',meas).strip('_').lower()[:34]}",
                "shape": 1, "measure": meas, "leakage_family": f,
                "pre_col": "", "post_col": c, "n_paired": n,
                "group_pct_of_means": float("nan"), "effect_size": float("nan"),
                "abs_effect_size": float("nan"), "pct_safe": True,
                "recommended_scale": "absolute",
                "family_size": len(measures),
                "notes": f"same-timepoint; {len(measures) - 1} co-measured in this file at {tp}",
            })

    # ---------------------------------------------------------------- Shape 3
    for (meas, f), items in groups.items():
        ins = [(br_day(tp), c) for tp, c in items if phase(tp) == "IN"]
        ins = [(d, c) for d, c in ins if not np.isnan(d)]
        if len(ins) < a.min_timepoints:
            continue
        ins.sort()
        sub = num[[c for _, c in ins]].dropna()
        if len(sub) < a.min_n:
            continue
        rows.append({
            "target_id": f"s3_{re.sub(r'[^A-Za-z0-9]+','_',meas).strip('_').lower()[:34]}",
            "shape": 3, "measure": meas, "leakage_family": f,
            "pre_col": ins[0][1], "post_col": ins[-1][1], "n_paired": len(sub),
            "group_pct_of_means": float("nan"), "effect_size": float("nan"),
            "abs_effect_size": float("nan"), "pct_safe": True,
            "recommended_scale": "absolute", "family_size": len(ins),
            "notes": f"{len(ins)} in-bedrest timepoints, days {int(ins[0][0])}-{int(ins[-1][0])}",
        })

    cat = pd.DataFrame(rows)
    if cat.empty:
        raise SystemExit("no targets found -- check --min-n and the master path")

    # de-duplicate: one row per (target_id, shape), keep the largest effect / n
    cat = (cat.sort_values(["shape", "abs_effect_size", "n_paired"], ascending=[True, False, False])
              .drop_duplicates(subset=["target_id", "shape"], keep="first")
              .reset_index(drop=True))

    p = os.path.join(a.outdir, "c11_target_registry.csv")
    cat.to_csv(p, index=False)

    print()
    for s in sorted(cat["shape"].unique()):
        sub = cat[cat["shape"] == s]
        print(f"Shape {s}: {len(sub)} candidate targets")
    print(f"\nwrote {p}")

    s2 = cat[cat["shape"] == 2].nlargest(12, "abs_effect_size")
    if len(s2):
        print("\ntop Shape 2 targets by absolute effect size:")
        print(f"  {'n':>3} {'pct':>8} {'effect':>7} {'scale':>9}  measure / file")
        for _, r in s2.iterrows():
            print(f"  {r.n_paired:>3} {r.group_pct_of_means:>8.1f} {r.effect_size:>7.2f} "
                  f"{r.recommended_scale:>9}  {r.measure[:30]} / {r.leakage_family[:26]}")

    unsafe = cat[(cat["shape"] == 2) & (~cat["pct_safe"])]
    print(f"\n{len(unsafe)} Shape 2 targets must use ABSOLUTE change "
          f"(a pre or post value is <= 0, so percent is unstable)")


if __name__ == "__main__":
    main()
