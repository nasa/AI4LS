#!/usr/bin/env python3
"""
run_tabpfn_test1.py

Aim 1, Run 1, Test 1: held-out prediction of DXA total-fat change from baseline
features, scored against the mean-guess floor.

WHAT THIS DOES
--------------
Leave-one-out by default: hide one subject's outcome, train TabPFN on the other
37, predict the held-out one, record the error. Repeat 38 times. That gives one
observed skill score. The comparison that matters is against the mean-guess
floor -- a model that always predicts the training mean. Beat the floor or the
model is not using the data.

  observed skill   TabPFN with access to 454 baseline features
  mean-guess floor always predict the training-fold mean
  shuffle null     outcomes randomly reassigned to subjects, LOOCV repeated

WHY THE FLOOR IS THE RIGHT BASELINE, NOT R^2 ALONE
--------------------------------------------------
At n=38 with 454 features, R^2 against the grand mean is optimistic because the
grand mean is computed with the held-out point included. The floor here is
recomputed per fold from the 37 training subjects only, so it is a genuine
out-of-sample baseline. A negative R^2 is normal and expected; what matters is
whether MAE beats the floor's MAE.

THE SHUFFLE NULL IS NOT OPTIONAL BEFORE REPORTING
-------------------------------------------------
With 454 features and 38 rows, correlations of |r| > 0.7 arise from noise alone
(the audit found r=0.765 on n=11). --n-shuffle permutes the outcome and reruns
the whole CV, giving the distribution of skill under no real signal. Do not
report an effect, or any feature ranking, until observed skill sits outside that
distribution. Default is 0 (off) so the first pass is quick; the script warns.

COST
----
LOOCV is 38 sequential TabPFN fits. On one GPU (T4 or better) that is ~1 hour
for a table this size. On CPU expect substantially longer. Use --cv 5 for a
5-fold screen (5 fits) to decide whether the full LOOCV is worth the time.
--n-shuffle N multiplies total fits by (N+1).

Usage:
  # quick 5-fold screen
  python3 run_tabpfn_test1.py --table c11_totalfat_pct_change_features_all.csv --cv 5

  # full Test 1
  python3 run_tabpfn_test1.py --table c11_totalfat_pct_change_features_all.csv --cv loo

  # full Test 1 + null
  python3 run_tabpfn_test1.py --table c11_totalfat_pct_change_features_all.csv \
      --cv loo --n-shuffle 20
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SUBJ = "subject"


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def describe_env(requested_device, model_version):
    """Resolve device and report the exact stack. Version pinning matters:
    if the package default moves from v3 to v4, an unpinned rerun is not
    comparable to this one."""
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if requested_device != "auto":
            dev = requested_device
        elif torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        info["device"] = dev
        if dev == "cuda":
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        sys.exit("FATAL: torch not installed. Run: pip install --upgrade tabpfn")

    try:
        import tabpfn
        info["tabpfn"] = getattr(tabpfn, "__version__", "unknown")
    except ImportError:
        sys.exit("FATAL: tabpfn not installed. Run: pip install --upgrade tabpfn")

    info["model_version_requested"] = model_version
    return info


def make_regressor(model_version, device, seed):
    """Build a TabPFNRegressor, pinning the checkpoint version when asked.
    Falls back cleanly and reports what it actually built."""
    from tabpfn import TabPFNRegressor

    kw = {}
    for name, val in (("device", device), ("random_state", seed)):
        kw[name] = val

    if model_version and model_version != "default":
        try:
            from tabpfn.constants import ModelVersion
            mv = getattr(ModelVersion, model_version)
            try:
                return TabPFNRegressor.create_default_for_version(mv, **kw), model_version
            except TypeError:
                return TabPFNRegressor.create_default_for_version(mv), model_version
        except (ImportError, AttributeError) as e:
            print(f"  WARNING: could not pin version '{model_version}' ({e}); using package default")

    try:
        return TabPFNRegressor(**kw), "package-default"
    except TypeError:
        return TabPFNRegressor(), "package-default"


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    out = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
    if len(y_true) > 2 and np.std(y_pred) > 0:
        # numpy/pandas only -- avoids a scipy dependency for two correlations.
        out["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        rt = pd.Series(y_true).rank().to_numpy()   # .rank() averages ties
        rp = pd.Series(y_pred).rank().to_numpy()
        out["spearman_r"] = float(np.corrcoef(rt, rp)[0, 1])
    else:
        out["pearson_r"] = out["spearman_r"] = float("nan")
    return out


def folds_for(n, cv, seed):
    """LOO or shuffled k-fold index pairs."""
    idx = np.arange(n)
    if cv == "loo":
        return [(np.delete(idx, i), np.array([i])) for i in idx]
    k = int(cv)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    return [(np.setdiff1d(idx, te), np.sort(te)) for te in np.array_split(perm, k)]


def run_cv(X, y, cv, seed, model_version, device, label, quiet=False):
    """One full CV pass. Returns (preds, floor_preds, elapsed_seconds)."""
    n = len(y)
    preds = np.full(n, np.nan)
    floor = np.full(n, np.nan)
    t0 = time.time()
    for j, (tr, te) in enumerate(folds_for(n, cv, seed), 1):
        model, _ = make_regressor(model_version, device, seed)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
        floor[te] = y[tr].mean()          # per-fold training mean: true baseline
        if not quiet:
            el = time.time() - t0
            print(f"    {label} fold {j:>2}  ({el:6.1f}s elapsed, "
                  f"~{el / j * (len(folds_for(n, cv, seed)) - j):6.1f}s left)", flush=True)
    return preds, floor, time.time() - t0


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TabPFN held-out test with mean-guess floor.")
    ap.add_argument("--table", required=True, help="feature CSV; last column is the outcome")
    ap.add_argument("--outcome", default=None, help="outcome column (default: last column)")
    ap.add_argument("--min-coverage", type=float, default=0.80,
                    help="drop features with non-null fraction below this (default 0.80)")
    ap.add_argument("--cv", default="loo", help="'loo' or an integer k (default loo)")
    ap.add_argument("--n-shuffle", type=int, default=0,
                    help="permutation nulls; 0 skips (results NOT reportable without it)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--model-version", default="default",
                    help="'default', or a tabpfn.constants.ModelVersion name e.g. V2_6")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 66)
    print("AIM 1 / RUN 1 / TEST 1 -- held-out prediction vs mean-guess floor")
    print("=" * 66)

    env = describe_env(args.device, args.model_version)
    for k, v in env.items():
        print(f"  {k:<26} {v}")
    device = env["device"]
    if device == "cpu":
        print("\n  WARNING: running on CPU. TabPFN documents a GPU (T4 minimum) as")
        print("  recommended. LOOCV will be slow; consider --cv 5 first.")

    # ------------------------------------------------------------------ data
    d = pd.read_csv(args.table)
    outcome = args.outcome or d.columns[-1]
    if outcome not in d.columns:
        sys.exit(f"FATAL: outcome '{outcome}' not in table")
    print(f"\n  table    {os.path.basename(args.table)}  {d.shape[0]} x {d.shape[1]}")
    print(f"  outcome  {outcome}")

    other_targets = [c for c in d.columns if "change" in c and c != outcome]
    if other_targets:
        sys.exit(f"FATAL: a second outcome column is present and would leak: {other_targets}\n"
                 f"       Use the single-outcome table.")

    d = d[d[outcome].notna()].reset_index(drop=True)
    subjects = d[SUBJ].values if SUBJ in d.columns else np.arange(len(d))
    y = d[outcome].to_numpy(float)
    Xdf = d.drop(columns=[c for c in (SUBJ, outcome) if c in d.columns])
    Xdf = Xdf.select_dtypes("number")

    cov = Xdf.notna().mean()
    keep = cov[cov >= args.min_coverage].index.tolist()
    dropped = len(Xdf.columns) - len(keep)
    Xdf = Xdf.loc[:, keep]
    print(f"  features {len(keep)} kept, {dropped} dropped at coverage >= {args.min_coverage:.2f}")
    print(f"  missing  {Xdf.isna().mean().mean():.1%} of retained cells (NaN passed to TabPFN)")
    print(f"  outcome  mean {y.mean():+.2f}  sd {y.std(ddof=1):.2f}  "
          f"range {y.min():+.1f} to {y.max():+.1f}")

    if len(keep) == 0:
        sys.exit("FATAL: no features survived the coverage filter")

    X = Xdf.to_numpy(float)
    np.random.seed(args.seed)

    cv = args.cv if args.cv == "loo" else int(args.cv)
    n_fits = len(y) if cv == "loo" else int(cv)
    print(f"\n  scheme   {'LOOCV' if cv == 'loo' else f'{cv}-fold'}  ->  {n_fits} fits"
          f"{f' x {args.n_shuffle + 1} (incl. nulls)' if args.n_shuffle else ''}")
    print(f"  seed     {args.seed}")

    # ------------------------------------------------------------- observed
    print("\n  running observed CV ...")
    preds, floor, elapsed = run_cv(X, y, cv, args.seed, args.model_version, device, "obs")
    m_obs, m_floor = metrics(y, preds), metrics(y, floor)

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    hdr = f"  {'metric':<12}{'TabPFN':>12}{'mean-floor':>14}{'better?':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k, lower_is_better in (("mae", True), ("rmse", True), ("r2", False)):
        a, b = m_obs[k], m_floor[k]
        win = (a < b) if lower_is_better else (a > b)
        print(f"  {k:<12}{a:>12.4f}{b:>14.4f}{'YES' if win else 'no':>10}")
    print(f"  {'pearson_r':<12}{m_obs['pearson_r']:>12.4f}{'--':>14}{'':>10}")
    print(f"  {'spearman_r':<12}{m_obs['spearman_r']:>12.4f}{'--':>14}{'':>10}")

    skill = 1 - m_obs["mae"] / m_floor["mae"]
    print(f"\n  MAE improvement over floor : {skill:+.1%}")
    verdict = ("BEATS the floor" if m_obs["mae"] < m_floor["mae"]
               else "does NOT beat the floor -- model is not using the features")
    print(f"  Verdict                    : {verdict}")
    print(f"  Wall clock                 : {elapsed / 60:.1f} min ({elapsed / n_fits:.1f}s/fit)")

    # ---------------------------------------------------------------- nulls
    null_maes = []
    if args.n_shuffle > 0:
        print(f"\n  running {args.n_shuffle} shuffle nulls ...")
        rng = np.random.default_rng(args.seed)
        for s in range(1, args.n_shuffle + 1):
            ysh = rng.permutation(y)
            p, _, el = run_cv(X, ysh, cv, args.seed, args.model_version, device, f"null{s}", quiet=True)
            null_maes.append(metrics(ysh, p)["mae"])
            print(f"    null {s:>2}/{args.n_shuffle}  MAE {null_maes[-1]:.4f}  ({el / 60:.1f} min)", flush=True)
        nm = np.array(null_maes)
        p_emp = float((nm <= m_obs["mae"]).sum() + 1) / (len(nm) + 1)
        print("\n" + "=" * 66)
        print("SHUFFLE NULL")
        print("=" * 66)
        print(f"  null MAE      mean {nm.mean():.4f}  sd {nm.std(ddof=1):.4f}  "
              f"min {nm.min():.4f}")
        print(f"  observed MAE  {m_obs['mae']:.4f}")
        print(f"  empirical p   {p_emp:.4f}   ({(nm <= m_obs['mae']).sum()}/{len(nm)} nulls "
              f"as good or better)")
        print(f"  reading       {'observed skill is outside the null' if p_emp < 0.05 else 'observed skill is INSIDE the null -- do not report as signal'}")
    else:
        print("\n  ** NO SHUFFLE NULL RUN. With 454 candidate features at n=38, skill this")
        print("     size can arise from noise. Do not report this result, or any feature")
        print("     ranking from it, until --n-shuffle has been run. **")

    # ---------------------------------------------------------------- write
    tag = f"{os.path.splitext(os.path.basename(args.table))[0]}_{cv}_seed{args.seed}"
    pred_path = os.path.join(args.outdir, f"predictions_{tag}.csv")
    pd.DataFrame({
        SUBJ: subjects, "y_true": y, "y_pred_tabpfn": preds, "y_pred_floor": floor,
        "abs_err_tabpfn": np.abs(preds - y), "abs_err_floor": np.abs(floor - y),
    }).to_csv(pred_path, index=False)

    summ = {
        "environment": env,
        "config": {
            "table": os.path.basename(args.table), "outcome": outcome,
            "min_coverage": args.min_coverage, "n_features_kept": len(keep),
            "n_features_dropped": dropped, "cv": str(cv), "n_fits": n_fits,
            "seed": args.seed, "n_shuffle": args.n_shuffle,
        },
        "observed": m_obs, "mean_guess_floor": m_floor,
        "mae_improvement_over_floor": skill,
        "beats_floor": bool(m_obs["mae"] < m_floor["mae"]),
        "elapsed_sec": elapsed,
        "shuffle_null_maes": null_maes,
        "shuffle_empirical_p": (float((np.array(null_maes) <= m_obs["mae"]).sum() + 1)
                                / (len(null_maes) + 1)) if null_maes else None,
        "caveats": [
            "Timepoint provenance not verifiable in the feature table; baseline status "
            "asserted by upstream construction, not checked here.",
            "n=38; Cromwell Table 3 gives 37 completers. Cohort cut unresolved.",
            "3 IQR outliers in the outcome (subjects 5188, 6187, 6791) not yet verified "
            "against the raw archive.",
        ],
    }
    summ_path = os.path.join(args.outdir, f"summary_{tag}.json")
    with open(summ_path, "w") as fh:
        json.dump(summ, fh, indent=2)

    print(f"\n  wrote {pred_path}")
    print(f"  wrote {summ_path}")

    print("\n" + "=" * 66)
    print("RUN LOG LINE")
    print("=" * 66)
    print(f"  {env['timestamp_utc']} | run_tabpfn_test1.py | {os.path.basename(args.table)} "
          f"| outcome={outcome} | cv={cv} | feats={len(keep)} | seed={args.seed} "
          f"| tabpfn={env['tabpfn']} | model={args.model_version} | device={device} "
          f"| MAE={m_obs['mae']:.4f} floor={m_floor['mae']:.4f} | out=summary_{tag}.json")


if __name__ == "__main__":
    main()
