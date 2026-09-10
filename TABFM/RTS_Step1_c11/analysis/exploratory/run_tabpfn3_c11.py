#!/usr/bin/env python3
"""
run_tabpfn3_c11.py

Run TabPFN-3 regression on one C11 subject-by-feature table from the command
line. Reproduces the Aim-1 test structure used in the 2026-07-30 runs:
coverage-filtered features, K-fold or leave-one-out cross-validation, a
mean-guess floor, and an optional shuffle-outcome null.

Design notes:
  - NO IMPUTATION. TabPFN ingests NaN natively. Missing cells are passed
    through as NaN, so per-variable missingness is never papered over with a
    column mean. Missingness is reported, not filled.
  - Arm assignment never enters the feature block. Any column resembling
    arm / group / cohort is dropped before fitting and the drop is printed.
  - Metrics are computed on pooled out-of-fold predictions, so every subject
    is scored by a model that never saw them.
  - The resolved model checkpoint is recorded, not just "default", so the run
    log pins the actual weights.

Usage:
  python3 run_tabpfn3_c11.py \
      --table "/path/to/c11_totalfat_pct_change_features_all.csv" \
      --outcome total_fat_pct_change \
      --outdir ./tabpfn3_runs \
      --cv 5 --seed 42 --min-coverage 0.8

  python3 run_tabpfn3_c11.py --table ... --outcome twp_abs_change \
      --cv loo --n-shuffle 20
"""

import argparse
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


RULE = "=" * 78
THIN = "-" * 78

# Columns that must never become features. Arm/group is assignment leakage.
LEAKAGE_PAT = re.compile(
    r"^(group|groupname|grouplabel|arm|cohort|treatment|condition_arm)$", re.I
)
ID_PAT = re.compile(r"^(subject|subject[_ ]?id|subjectid|id|index)$", re.I)


def banner(text):
    print(RULE)
    print(text)
    print(RULE)


def resolve_device(requested):
    """cuda if present, else cpu. MPS is skipped: the TabPFN-3 checkpoint has
    asymmetric query/KV head counts and the MPS attention path is unreliable."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def env_block(device):
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
    }
    try:
        import torch
        env["torch"] = torch.__version__
        if device == "cuda":
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        env["torch"] = "not installed"
    try:
        import tabpfn
        env["tabpfn"] = getattr(tabpfn, "__version__", "unknown")
    except Exception:
        env["tabpfn"] = "not installed"
    return env


def make_regressor(device, seed):
    """Build a TabPFNRegressor, passing only kwargs this installed version
    accepts. Keeps the script working across the 8.x line without pinning."""
    import inspect
    from tabpfn import TabPFNRegressor

    wanted = {
        "device": device,
        "random_state": seed,
        "ignore_pretraining_limits": True,  # >200 features triggers subsampling
    }
    sig = inspect.signature(TabPFNRegressor.__init__).parameters
    kwargs = {k: v for k, v in wanted.items() if k in sig}
    return TabPFNRegressor(**kwargs), kwargs


def resolved_checkpoint(model):
    """Best-effort read of the actual weights used, so the run log is pinned."""
    for attr in ("model_path", "model_path_", "checkpoint_", "model_name_"):
        v = getattr(model, attr, None)
        if v:
            return str(v)
    cfg = getattr(model, "config_", None) or getattr(model, "config", None)
    if isinstance(cfg, dict):
        for k in ("model_path", "checkpoint", "model_name"):
            if cfg.get(k):
                return str(cfg[k])
    return "default (unresolved)"


def scores(y_true, y_pred):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "spearman_r": float(spearmanr(y_true, y_pred)[0]),
    }


def build_splitter(cv, n, seed):
    from sklearn.model_selection import KFold, LeaveOneOut
    if str(cv).lower() == "loo":
        return LeaveOneOut(), n, "loo"
    k = int(cv)
    return KFold(n_splits=k, shuffle=True, random_state=seed), k, str(k)


def cross_val_oof(X, y, device, seed, cv, verbose=True):
    """Pooled out-of-fold predictions. Returns (preds, n_fits, checkpoint)."""
    splitter, n_fits, _ = build_splitter(cv, len(y), seed)
    oof = np.full(len(y), np.nan, dtype=float)
    ckpt = "default (unresolved)"

    for i, (tr, te) in enumerate(splitter.split(X), start=1):
        model, _ = make_regressor(device, seed)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
        if i == 1:
            ckpt = resolved_checkpoint(model)
        if verbose:
            print(f"  fit {i:>3}/{n_fits}   train={len(tr):>3}  test={len(te):>3}",
                  flush=True)
    return oof, n_fits, ckpt


def main():
    ap = argparse.ArgumentParser(
        description="TabPFN-3 regression on a C11 feature table."
    )
    ap.add_argument("--table", required=True, help="path to the feature CSV")
    ap.add_argument("--outcome", required=True, help="outcome column name")
    ap.add_argument("--outdir", default=".", help="where to write the summary JSON")
    ap.add_argument("--cv", default="5", help="'5' for 5-fold, or 'loo'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-coverage", type=float, default=0.8,
                    help="keep features with at least this fraction non-null")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--n-shuffle", type=int, default=0,
                    help="shuffle-outcome null refits (0 = skip)")
    args = ap.parse_args()

    t0 = time.time()
    device = resolve_device(args.device)
    env = env_block(device)

    banner("TabPFN-3  |  NASA LSDA bed rest  |  Campaign 11  |  Aim 1")
    for k in ("timestamp_utc", "python", "platform", "torch", "gpu",
              "tabpfn", "device"):
        if k in env:
            print(f"  {k:<16} {env[k]}")
    if env["tabpfn"] == "not installed":
        sys.exit("\nERROR: tabpfn is not installed. See setup instructions.")
    print(RULE)

    # ---- load -------------------------------------------------------------
    df = pd.read_csv(args.table)
    print(f"\ntable            {os.path.basename(args.table)}")
    print(f"raw shape        {df.shape[0]} rows x {df.shape[1]} columns")

    if args.outcome not in df.columns:
        sys.exit(f"ERROR: outcome '{args.outcome}' not in table. "
                 f"First columns: {list(df.columns[:8])}")

    df = df[df[args.outcome].notna()].copy()
    y = df[args.outcome].astype(float).to_numpy()
    print(f"outcome          {args.outcome}")
    print(f"subjects scored  {len(y)}  "
          f"(mean {y.mean():.3f}, sd {y.std(ddof=1):.3f})")

    # ---- feature block ----------------------------------------------------
    drop_leak = [c for c in df.columns if LEAKAGE_PAT.match(str(c).strip())]
    drop_id = [c for c in df.columns if ID_PAT.match(str(c).strip())]
    feat = df.drop(columns=set(drop_leak + drop_id + [args.outcome]),
                   errors="ignore")
    feat = feat.apply(pd.to_numeric, errors="coerce")

    coverage = feat.notna().mean()
    keep = coverage[coverage >= args.min_coverage].index
    keep = [c for c in keep if feat[c].nunique(dropna=True) > 1]
    dropped = feat.shape[1] - len(keep)
    X = feat[keep].to_numpy(dtype=float)

    print(THIN)
    if drop_leak:
        print(f"arm/group columns dropped (leakage): {drop_leak}")
    print(f"features kept    {len(keep)}")
    print(f"features dropped {dropped}   "
          f"(coverage < {args.min_coverage:.0%} or zero variance)")
    print(f"missing cells    {np.isnan(X).mean():.1%} of the kept matrix, "
          f"passed to TabPFN as NaN (no imputation)")
    print(THIN)

    # ---- observed ---------------------------------------------------------
    _, _, cv_label = build_splitter(args.cv, len(y), args.seed)
    print(f"\ncross-validation  cv={cv_label}  seed={args.seed}  device={device}")
    oof, n_fits, ckpt = cross_val_oof(X, y, device, args.seed, args.cv)
    observed = scores(y, oof)
    print(f"\nresolved checkpoint  {ckpt}")

    # ---- mean-guess floor -------------------------------------------------
    splitter, _, _ = build_splitter(args.cv, len(y), args.seed)
    floor_pred = np.full(len(y), np.nan)
    for tr, te in splitter.split(X):
        floor_pred[te] = y[tr].mean()
    floor = scores(y, floor_pred)

    # ---- shuffle null -----------------------------------------------------
    null = None
    if args.n_shuffle > 0:
        print(f"\nshuffle-outcome null: {args.n_shuffle} refits")
        rng = np.random.default_rng(args.seed)
        maes = []
        for s in range(args.n_shuffle):
            ys = rng.permutation(y)
            p, _, _ = cross_val_oof(X, ys, device, args.seed, args.cv,
                                    verbose=False)
            maes.append(scores(ys, p)["mae"])
            print(f"  shuffle {s + 1:>3}/{args.n_shuffle}  mae={maes[-1]:.4f}",
                  flush=True)
        maes = np.array(maes)
        null = {
            "n_shuffle": int(args.n_shuffle),
            "mae_mean": float(maes.mean()),
            "mae_sd": float(maes.std(ddof=1)),
            "p_value_mae": float((maes <= observed["mae"]).mean()),
        }

    # ---- report -----------------------------------------------------------
    print("\n" + RULE)
    print("RESULT")
    print(RULE)
    print(f"  {'metric':<12}{'TabPFN-3':>14}{'mean-guess floor':>20}")
    print(f"  {THIN[:44]}")
    for m in ("mae", "rmse", "r2", "pearson_r", "spearman_r"):
        print(f"  {m:<12}{observed[m]:>14.4f}{floor[m]:>20.4f}")
    beat = observed["mae"] < floor["mae"]
    print(f"\n  beats mean-guess floor on MAE:  {'YES' if beat else 'NO'}")
    if null:
        print(f"  shuffle null MAE  {null['mae_mean']:.4f} "
              f"+/- {null['mae_sd']:.4f}   p={null['p_value_mae']:.3f}")
    print(f"  elapsed  {time.time() - t0:.1f}s")
    print(RULE)

    # ---- write ------------------------------------------------------------
    env["model_version_requested"] = "default"
    env["model_checkpoint_resolved"] = ckpt
    summary = {
        "environment": env,
        "config": {
            "table": os.path.basename(args.table),
            "table_path": os.path.abspath(args.table),
            "outcome": args.outcome,
            "min_coverage": args.min_coverage,
            "n_features_kept": len(keep),
            "n_features_dropped": int(dropped),
            "arm_columns_dropped": drop_leak,
            "imputation": "none (NaN passed through)",
            "cv": cv_label,
            "n_fits": int(n_fits),
            "seed": args.seed,
            "n_shuffle": int(args.n_shuffle),
        },
        "observed": observed,
        "mean_guess_floor": floor,
        "shuffle_null": null,
    }

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.basename(args.table).rsplit(".csv", 1)[0]
    out = os.path.join(
        args.outdir, f"summary_{stem}_{cv_label}_seed{args.seed}_local.json"
    )
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out}\n")


if __name__ == "__main__":
    main()
