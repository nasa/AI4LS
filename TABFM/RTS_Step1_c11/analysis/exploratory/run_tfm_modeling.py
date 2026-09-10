#!/usr/bin/env python3
"""
TabPFN-3 Aim-1 modeling run — C11 total-hip BMD change.

Order (per launch steer):
  1. Descriptive metrics: 10x5 repeated k-fold CV (R2/RMSE/MAE/Spearman + CIs) + mean-predictor floor.
     Both tables. Headline result; does not depend on the null.
  2. Permutation null (LOOCV, matched scheme), everything table FIRST, shortlist SECOND.
     Chunked (25 shuffles/chunk), resumable, running empirical p written after every chunk.

Null test statistics: Spearman(pred, actual) and negative MAE (not R2).
Observed and null use the IDENTICAL LOOCV scheme.
10x5 repeated-CV metrics are DESCRIPTIVE ONLY and are never compared to the null.

Determinism: torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8,
seed pinned (random_state=42). Verified byte-identical across fresh fits on this A10G.
"""
import os, sys, json, time, hashlib, warnings
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
warnings.filterwarnings('ignore')
from tabpfn import TabPFNRegressor
from sklearn.model_selection import RepeatedKFold
from scipy.stats import spearmanr

# ---------------- pins ----------------
SEED = 42
N_EST = 32
DEVICE = 'cuda'
CKPT = os.environ.get('TABPFN_CKPT', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tabpfn-v3-regressor-v3_default.ckpt'))
TABPFN_VERSION = '8.2.0'
OUTDIR = os.environ.get('C11_OUTDIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'bmd'))
# np.savez / HDF5-style writes fail on S3-backed mounts (Invalid argument).
# Write checkpoints to local /workspace, then copy to the S3 shared workspace.
CKPT_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tfm_null_ckpt')
CKPT_DIR = os.environ.get('TABPFN_CKPT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tfm_null_ckpt'))
os.makedirs(CKPT_LOCAL, exist_ok=True)
TARGET = 'totalhip_BMD_change'
CHUNK = 25
N_SHUFFLES = 1000

TABLES = {
    'everything': f'{OUTDIR}/c11_totalhipBMD_change_features_all.csv',
    'shortlist':  f'{OUTDIR}/c11_totalhipBMD_change_features_shortlist.csv',
}

# Log to local file, then copy to results on completion.
LOG_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs', 'run_log_tfm_modeling.txt')
LOG = f'{OUTDIR}/run_log_tfm_modeling.txt'
_lf = open(LOG_LOCAL, 'a', buffering=1)
def say(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    _lf.write(line + '\n')

def sync_log():
    import shutil
    try:
        _lf.flush()
        shutil.copy(LOG_LOCAL, LOG)
    except Exception as e:
        print(f'[warn] log sync failed: {e}', flush=True)

def sha256(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:n]

def make_model():
    return TabPFNRegressor(model_path=CKPT, device=DEVICE, n_estimators=N_EST,
                           random_state=SEED, show_progress_bar=False)

def loocv_preds(X, y):
    """LOOCV out-of-fold predictions. Deterministic fold order (subject order)."""
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        m = make_model()
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i+1])[0]
    return preds

def metrics_block(ytrue, ypred):
    mask = ~(np.isnan(ytrue) | np.isnan(ypred))
    yt, yp = ytrue[mask], ypred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    sp = float(spearmanr(yt, yp).statistic) if mask.sum() > 2 else np.nan
    return dict(r2=r2, rmse=rmse, mae=mae, spearman=sp, n=int(mask.sum()))

def neg_mae(ytrue, ypred):
    return -float(np.mean(np.abs(ytrue - ypred)))

def bootstrap_ci(ytrue, ypred, fn, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(ytrue)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(ytrue[idx])) < 2:
            continue
        vals.append(fn(ytrue[idx], ypred[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

# ---------------- descriptive: 10x5 repeated CV ----------------
def descriptive(name, X, y, subjects):
    say(f'[descriptive] {name}: 10x5 repeated CV ...')
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    oof = np.full(len(y), np.nan)
    # accumulate per-repeat OOF then average predictions per subject
    per_repeat = []
    for rep in range(10):
        oof_r = np.full(len(y), np.nan)
        for tr, te in rkf.split(X):
            pass  # placeholder; we iterate manually below
        break
    # Manual repeated k-fold to control repeats
    from sklearn.model_selection import KFold
    rng = np.random.default_rng(SEED)
    for rep in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        oof_r = np.full(len(y), np.nan)
        for tr, te in kf.split(X):
            m = make_model()
            m.fit(X[tr], y[tr])
            oof_r[te] = m.predict(X[te])
        per_repeat.append(oof_r)
    per_repeat = np.array(per_repeat)          # (10, n)
    oof_mean = np.nanmean(per_repeat, axis=0)  # mean prediction per subject
    mb = metrics_block(y, oof_mean)
    # CIs by subject bootstrap on the mean-OOF predictions
    r2_lo, r2_hi = bootstrap_ci(y, oof_mean, lambda a, b: metrics_block(a, b)['r2'])
    rmse_lo, rmse_hi = bootstrap_ci(y, oof_mean, lambda a, b: metrics_block(a, b)['rmse'])
    mae_lo, mae_hi = bootstrap_ci(y, oof_mean, lambda a, b: metrics_block(a, b)['mae'])
    sp_lo, sp_hi = bootstrap_ci(y, oof_mean, lambda a, b: metrics_block(a, b)['spearman'])
    say(f'[descriptive] {name}: R2={mb["r2"]:.4f} RMSE={mb["rmse"]:.5f} MAE={mb["mae"]:.5f} Spearman={mb["spearman"]:.4f}')
    row = dict(table=name, **mb,
               r2_ci_lo=r2_lo, r2_ci_hi=r2_hi, rmse_ci_lo=rmse_lo, rmse_ci_hi=rmse_hi,
               mae_ci_lo=mae_lo, mae_ci_hi=mae_hi, spearman_ci_lo=sp_lo, spearman_ci_hi=sp_hi)
    return row, oof_mean, per_repeat

def mean_floor(y):
    """Mean-predictor floor via LOOCV (predict leave-one-out training mean)."""
    n = len(y)
    preds = np.array([np.mean(y[np.arange(n) != i]) for i in range(n)])
    return metrics_block(y, preds), preds

# ---------------- permutation null (LOOCV, chunked, resumable) ----------------
def null_ckpt_local(name):
    return f'{CKPT_LOCAL}/null_{name}.npz'

def null_ckpt_path(name):
    # resume source: prefer S3 copy (survives machine recreation), fall back to local
    import shutil
    s3 = f'{CKPT_DIR}/null_{name}.npz'
    loc = null_ckpt_local(name)
    if os.path.exists(s3) and not os.path.exists(loc):
        try:
            os.makedirs(CKPT_LOCAL, exist_ok=True)
            shutil.copy(s3, loc)
        except Exception:
            pass
    return loc

def save_ckpt(name, done, null_sp, null_nm):
    import shutil
    np.savez(null_ckpt_local(name), start=done,
             null_sp=np.array(null_sp), null_nm=np.array(null_nm))
    try:
        os.makedirs(CKPT_DIR, exist_ok=True)
        shutil.copy(null_ckpt_local(name), f'{CKPT_DIR}/null_{name}.npz')
    except Exception as e:
        say(f'[warn] ckpt S3 sync failed (local copy intact): {e}')

def run_null(name, X, y):
    say(f'[null] {name}: observed LOOCV ...')
    obs_preds = loocv_preds(X, y)
    obs_spearman = float(spearmanr(y, obs_preds).statistic)
    obs_negmae = neg_mae(y, obs_preds)
    say(f'[null] {name}: observed Spearman={obs_spearman:.4f}  -MAE={obs_negmae:.6f}')

    # resume
    start = 0
    null_sp, null_nm = [], []
    if os.path.exists(null_ckpt_path(name)):
        d = np.load(null_ckpt_path(name))
        start = int(d['start'])
        null_sp = list(d['null_sp']); null_nm = list(d['null_nm'])
        say(f'[null] {name}: RESUME from shuffle {start} ({len(null_sp)} null stats cached)')

    rng = np.random.default_rng(SEED + 1000)  # fixed shuffle stream
    # advance rng past already-done shuffles for determinism across resume
    for _ in range(start):
        rng.permutation(len(y))

    results_csv = f'{OUTDIR}/tfm_results_null.csv'
    for s in range(start, N_SHUFFLES):
        yp = rng.permutation(y)
        preds = loocv_preds(X, yp)
        null_sp.append(float(spearmanr(yp, preds).statistic))
        null_nm.append(neg_mae(yp, preds))

        done = s + 1
        if done % CHUNK == 0 or done == N_SHUFFLES:
            save_ckpt(name, done, null_sp, null_nm)
            # running empirical p (one-sided, +1 correction)
            arr_sp = np.array(null_sp); arr_nm = np.array(null_nm)
            p_sp = (1 + np.sum(arr_sp >= obs_spearman)) / (len(arr_sp) + 1)
            p_nm = (1 + np.sum(arr_nm >= obs_negmae)) / (len(arr_nm) + 1)
            sp_p95 = float(np.percentile(arr_sp, 95))
            nm_p95 = float(np.percentile(arr_nm, 95))
            gap_sp = obs_spearman - sp_p95   # observed minus null 95th pct (Spearman)
            gap_nm = obs_negmae - nm_p95     # observed minus null 95th pct (-MAE)
            rec = dict(table=name, n_shuffles=done,
                       obs_spearman=obs_spearman, obs_neg_mae=obs_negmae,
                       null_spearman_mean=float(arr_sp.mean()), null_spearman_sd=float(arr_sp.std()),
                       null_spearman_p95=sp_p95,
                       null_negmae_mean=float(arr_nm.mean()), null_negmae_sd=float(arr_nm.std()),
                       null_negmae_p95=nm_p95,
                       obs_minus_null95_spearman=gap_sp,
                       obs_minus_null95_neg_mae=gap_nm,
                       emp_p_spearman=p_sp, emp_p_neg_mae=p_nm,
                       p_floor=1.0 / (done + 1))
            # write/refresh running results (one row per table, latest)
            if os.path.exists(results_csv):
                rdf = pd.read_csv(results_csv)
                rdf = rdf[rdf['table'] != name]
            else:
                rdf = pd.DataFrame()
            rdf = pd.concat([rdf, pd.DataFrame([rec])], ignore_index=True)
            rdf.to_csv(results_csv, index=False)
            say(f'[null] {name}: {done}/{N_SHUFFLES} shuffles | running p(Spearman)={p_sp:.4f}  p(-MAE)={p_nm:.4f}  (floor {rec["p_floor"]:.4f}) | obs-null95 gap: Spearman={gap_sp:+.4f}  -MAE={gap_nm:+.6f}')
            sync_log()

    return obs_preds, obs_spearman, obs_negmae, np.array(null_sp), np.array(null_nm)

# ---------------- main ----------------
def main():
    say('=' * 70)
    say('TabPFN-3 Aim-1 modeling run START')
    say(f'PINS: tabpfn=={TABPFN_VERSION}  device={DEVICE}  n_estimators={N_EST}  seed={SEED}')
    say(f'checkpoint={os.path.basename(CKPT)}  sha256={sha256(CKPT)}')
    say(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}')
    say('determinism: torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8 (byte-identical on A10G, verified)')
    say('null scheme: LOOCV matched observed/null; stats=Spearman & -MAE; descriptive=10x5 (not compared to null)')
    say('=' * 70)

    which = sys.argv[1] if len(sys.argv) > 1 else 'all'

    # ---- Phase 1: descriptive + mean floor (both tables) ----
    if which in ('all', 'descriptive'):
        metrics_rows = []
        oof_store = {}
        for name, path in TABLES.items():
            df = pd.read_csv(path, index_col=0)
            X = df.drop(columns=[TARGET]).values
            y = df[TARGET].values
            subjects = df.index.astype(str).tolist()
            row, oof_mean, per_repeat = descriptive(name, X, y, subjects)
            # mean-predictor floor
            floor_mb, floor_preds = mean_floor(y)
            say(f'[floor] {name}: mean-predictor R2={floor_mb["r2"]:.4f} RMSE={floor_mb["rmse"]:.5f} MAE={floor_mb["mae"]:.5f} Spearman={floor_mb["spearman"]:.4f}')
            row['floor_r2'] = floor_mb['r2']; row['floor_rmse'] = floor_mb['rmse']
            row['floor_mae'] = floor_mb['mae']; row['floor_spearman'] = floor_mb['spearman']
            metrics_rows.append(row)
            oof_store[name] = (subjects, y, oof_mean, floor_preds)
        mdf = pd.DataFrame(metrics_rows)
        mdf.to_csv(f'{OUTDIR}/tfm_results_metrics.csv', index=False)
        say('[descriptive] wrote tfm_results_metrics.csv')
        # OOF predictions
        for name, (subjects, y, oof_mean, floor_preds) in oof_store.items():
            pd.DataFrame({'subject': subjects, 'y_true': y,
                          'oof_pred_10x5': oof_mean, 'floor_pred': floor_preds}
                         ).to_csv(f'{OUTDIR}/tfm_oof_predictions_{name}.csv', index=False)
        say('[descriptive] wrote tfm_oof_predictions_*.csv')
        sync_log()

    # ---- Phase 2: permutation null ----
    order = ['everything', 'shortlist'] if which in ('all', 'null') else [which]
    if which in ('all', 'null', 'everything', 'shortlist'):
        for name in order:
            path = TABLES[name]
            df = pd.read_csv(path, index_col=0)
            X = df.drop(columns=[TARGET]).values
            y = df[TARGET].values
            obs_preds, obs_sp, obs_nm, null_sp, null_nm = run_null(name, X, y)
            # save full null distribution
            pd.DataFrame({'null_spearman': null_sp, 'null_neg_mae': null_nm}
                         ).to_csv(f'{OUTDIR}/tfm_null_distribution_{name}.csv', index=False)
            say(f'[null] {name}: wrote tfm_null_distribution_{name}.csv ({len(null_sp)} shuffles)')

    say('TabPFN-3 Aim-1 modeling run COMPLETE')
    sync_log()

if __name__ == '__main__':
    main()
