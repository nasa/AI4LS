"""
Flatten a CSV with multiple rows per Subject into one row per Subject,
prefixing each measurement column with its Test_Phase.

Handles the case where a subject has MULTIPLE rows within the SAME
Test_Phase (e.g. several PRE_TEST rows) via the STRATEGY setting below.
"""

import pandas as pd
import sys

# ---- Settings -----------------------------------------------------------

# How to handle multiple rows for the same Subject + Test_Phase:
#   "index"   -> keep every row, number duplicates: "PRE_TEST_1:Pre-Hgb (g/dL)", "PRE_TEST_2:..."
#   "mean"    -> average the measurements within each phase: "PRE_TEST:Pre-Hgb (g/dL)"
#   "first"   -> keep only the first row of each phase
#   "last"    -> keep only the last row of each phase
STRATEGY = "mean"

# Columns that identify the subject (kept once per output row, not prefixed)
ID_COLS = ["Subject", "Group"]

# Column that splits the measurements
PHASE_COL = "Test_Phase"

# Columns to ignore entirely (e.g. a raw row-id that has no meaning per-subject)
DROP_COLS = ["ID"]
# --------------------------------------------------------------------------


def main():
    INPUT_CSV=sys.argv[1]
    OUTPUT_CSV=sys.argv[2]
    df = pd.read_csv(INPUT_CSV)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    measurement_cols = [c for c in df.columns if c not in ID_COLS + [PHASE_COL]]

    if STRATEGY == "mean":
        grouped = (
            df.groupby(ID_COLS + [PHASE_COL], as_index=False)[measurement_cols]
            .mean()
        )
        grouped["_occ"] = 1  # single "occurrence" per phase

    elif STRATEGY in ("first", "last"):
        keep = "first" if STRATEGY == "first" else "last"
        grouped = (
            df.groupby(ID_COLS + [PHASE_COL], as_index=False)
            .agg(keep)
        )
        grouped = grouped[ID_COLS + [PHASE_COL] + measurement_cols]
        grouped["_occ"] = 1

    elif STRATEGY == "index":
        df = df.copy()
        df["_occ"] = df.groupby(ID_COLS + [PHASE_COL]).cumcount() + 1
        grouped = df

    else:
        raise ValueError(f"Unknown STRATEGY: {STRATEGY}")

    # Build the prefixed column name for each (phase, occurrence) combo
    max_occ = grouped.groupby(PHASE_COL)["_occ"].max().to_dict()

    def prefix(phase, occ):
        if STRATEGY == "index" and max_occ.get(phase, 1) > 1:
            return f"{phase}_{occ}"
        return phase

    grouped["_phase_label"] = grouped.apply(
        lambda r: prefix(r[PHASE_COL], r["_occ"]), axis=1
    )

    # Pivot: one row per Subject (+ other id cols), columns = "<phase_label>:<measure>"
    wide_frames = []
    for label, sub in grouped.groupby("_phase_label"):
        sub = sub.set_index(ID_COLS)[measurement_cols]
        sub.columns = [f"{label}:{c}" for c in sub.columns]
        wide_frames.append(sub)

    result = pd.concat(wide_frames, axis=1).reset_index()

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(result)} rows and {len(result.columns)} columns.")


if __name__ == "__main__":
    main()
