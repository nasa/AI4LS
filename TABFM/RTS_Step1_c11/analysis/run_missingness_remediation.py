#!/usr/bin/env python3
"""
Missingness Remediation + Fold-Safe Test 2 — OBSERVED-ONLY stage (no nulls).
Replicates the battery v2-final harness exactly (TabPFN, seed 42, n_est=32,
deterministic LOOCV subject-index order) so comparisons are apples-to-apples.

Components:
  B      : mask-only baseline on FULL 628 features' NaN indicators (571 retained)
  B-core : mask-only on the 271 common-core features' NaN indicators
  A      : common-core restriction (actual values of the 271 features)
  R1     : fold-safe Test 2 (arm-residualize y within each LOOCV training fold)

All classifier outputs report balanced accuracy + per-class confusion matrix.
tabpfn.__version__ logged explicitly (original battery metadata had None).
"""
import os, json, time, hashlib, warnings
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# Licensed TabPFN access + cached weights (mirrors run_with_env.sh)
# TABPFN_TOKEN: set a licensed token in the environment before running.
# Do NOT hardcode tokens in this file.
if not os.environ.get('TABPFN_TOKEN'):
    raise SystemExit('TABPFN_TOKEN not set. Export a licensed token first (see README).')
os.environ.setdefault('TABPFN_NO_BROWSER', '1')
os.makedirs(os.path.expanduser('~/.cache/tabpfn'), exist_ok=True)
with open(os.path.expanduser('~/.cache/tabpfn/auth_token'), 'w') as _f:
    _f.write(os.environ['TABPFN_TOKEN'])
os.chmod(os.path.expanduser('~/.cache/tabpfn/auth_token'), 0o600)
os.environ.setdefault('TABPFN_CKPT', '/mnt/shared-workspace/tfm_battery_v2/weights/tabpfn-v3-regressor-v3_default.ckpt')
os.environ.setdefault('TABPFN_CLF_CKPT', '/mnt/shared-workspace/tfm_battery_v2/weights/tabpfn-v3-classifier-v3_default.ckpt')
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import tabpfn

SEED = 42
N_EST = 32
DEVICE = 'cuda'
SHARED = '/mnt/shared-workspace/tfm_battery_v2'
DATA_DIR = f'{SHARED}/data'
BMD_TABLE = f'{DATA_DIR}/c11_totalhipBMD_change_features_all.csv'
ARM_MAP = f'{DATA_DIR}/c11_subject_arm_map.csv'
OUT_DIR = '/mnt/results/tfm_battery_v2/phase4_missingness_remediation'
os.makedirs(OUT_DIR, exist_ok=True)

def say(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)

def _filter_kwargs(wanted, sig_params):
    return {k: v for k, v in wanted.items() if k in sig_params}

def make_classifier():
    import inspect
    from tabpfn import TabPFNClassifier
    wanted = {'device': DEVICE, 'random_state': SEED, 'n_estimators': N_EST,
              'ignore_pretraining_limits': True, 'show_progress_bar': False}
    ckpt = os.environ.get('TABPFN_CLF_CKPT', '')
    if ckpt and os.path.exists(ckpt):
        wanted['model_path'] = ckpt
    sig = inspect.signature(TabPFNClassifier.__init__).parameters
    return TabPFNClassifier(**_filter_kwargs(wanted, sig))

def make_regressor():
    import inspect
    from tabpfn import TabPFNRegressor
    wanted = {'device': DEVICE, 'random_state': SEED, 'n_estimators': N_EST,
              'ignore_pretraining_limits': True, 'show_progress_bar': False}
    ckpt = os.environ.get('TABPFN_CKPT', '')
    if ckpt and os.path.exists(ckpt):
        wanted['model_path'] = ckpt
    sig = inspect.signature(TabPFNRegressor.__init__).parameters
    return TabPFNRegressor(**_filter_kwargs(wanted, sig))

def loocv_preds_clf(X, y):
    n = len(y)
    preds = np.full(n, fill_value=None, dtype=object)
    for i in range(n):
        tr = np.arange(n) != i
        m = make_classifier()
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i+1])[0]
    return preds

def clf_metrics(ytrue, ypred):
    yt = np.array(ytrue); yp = np.array(ypred)
    labels = sorted(np.unique(yt))
    cm = confusion_matrix(yt, yp, labels=labels)
    return {'accuracy': float(accuracy_score(yt, yp)),
            'balanced_accuracy': float(balanced_accuracy_score(yt, yp)),
            'labels': labels,
            'confusion_matrix': cm.tolist()}

# ---------------- load data ----------------
df = pd.read_csv(BMD_TABLE, index_col=0)
y_reg = df['totalhip_BMD_change'].values
X_df = df.drop(columns=['totalhip_BMD_change'])
subjects = df.index.tolist()
feat_cols = X_df.columns.tolist()
arm_map = pd.read_csv(ARM_MAP)
arm_dict = dict(zip(arm_map['subject'], arm_map['arm']))
arms_all = np.array([arm_dict[s] for s in subjects])
n = len(subjects)
say(f'Loaded BMD table: {n} subjects x {len(feat_cols)} features; tabpfn {tabpfn.__version__}')

# ---------------- feature sets ----------------
X_full = X_df.values.astype(np.float32)                       # 628
P = pd.DataFrame({a: X_df[arms_all==a].notna().sum(axis=0) for a in ['CONTROL','EXERCISE','FLY']})
core_mask = (P['CONTROL']>=1)&(P['EXERCISE']>=1)&(P['FLY']>=1)
core_cols = core_mask[core_mask].index.tolist()
X_core = X_df[core_cols].values.astype(np.float32)            # 271
say(f'Common core: {len(core_cols)} features')

def mask_matrix(Xdf):
    M = Xdf.isna().astype(int)
    colsum = M.sum(axis=0)
    keep = (colsum>0)&(colsum<len(M))   # drop zero-variance (never- or always-missing)
    return M.loc[:, keep].values.astype(np.float32), int((~keep).sum()), int(keep.sum())

M_full, drop_full, keep_full = mask_matrix(X_df)              # B
M_core, drop_core, keep_core = mask_matrix(X_df[core_cols])   # B-core
say(f'Mask full: {keep_full} indicators (dropped {drop_full}); Mask core: {keep_core} (dropped {drop_core})')

PAIRS = [('CONTROL_vs_EXERCISE','CONTROL','EXERCISE'),
         ('CONTROL_vs_FLY','CONTROL','FLY'),
         ('EXERCISE_vs_FLY','EXERCISE','FLY')]

def run_classifiers(Xmat, tag):
    """Run C1 (3-class) + 3 pairs on feature matrix Xmat. Return results dict."""
    res = {}
    # C1 3-class
    say(f'[{tag}] C1 3-class ({Xmat.shape[1]} feats, n={n})...')
    preds = loocv_preds_clf(Xmat, arms_all)
    met = clf_metrics(arms_all, preds)
    pd.DataFrame({'subject': subjects, 'y_true': arms_all, 'y_pred': preds}
                 ).to_csv(f'{OUT_DIR}/{tag}_c1_predictions.csv', index=False)
    res['c1'] = met
    say(f'[{tag}] C1 bacc={met["balanced_accuracy"]:.4f} acc={met["accuracy"]:.4f}')
    # pairs
    for pname, a, b in PAIRS:
        m = (arms_all==a)|(arms_all==b)
        Xp = Xmat[m]; yp = arms_all[m]
        subj_p = [subjects[i] for i in range(n) if m[i]]
        say(f'[{tag}] {pname} (n={m.sum()})...')
        preds = loocv_preds_clf(Xp, yp)
        met = clf_metrics(yp, preds)
        pd.DataFrame({'subject': subj_p, 'y_true': yp, 'y_pred': preds}
                     ).to_csv(f'{OUT_DIR}/{tag}_{pname}_predictions.csv', index=False)
        res[pname] = met
        say(f'[{tag}] {pname} bacc={met["balanced_accuracy"]:.4f} acc={met["accuracy"]:.4f}')
    return res

# ---------------- R1: fold-safe Test 2 ----------------
def run_fold_safe_test2():
    say('[R1] Fold-safe Test 2 (residualize within each training fold)...')
    preds = np.full(n, np.nan)
    y_resid_heldout = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        # arm means from TRAINING subjects only
        train_arms = arms_all[tr]
        train_means = {a: y_reg[tr][train_arms==a].mean() for a in np.unique(train_arms)}
        y_resid_train = np.array([y_reg[tr][j] - train_means[train_arms[j]] for j in range(tr.sum())])
        # held-out residual uses its own arm's TRAINING mean
        y_resid_heldout[i] = y_reg[i] - train_means[arms_all[i]]
        m = make_regressor()
        m.fit(X_full[tr], y_resid_train)
        preds[i] = m.predict(X_full[i:i+1])[0]
    rho = float(spearmanr(y_resid_heldout, preds).statistic)
    say(f'[R1] Fold-safe rho = {rho:.6f} (original full-sample-resid rho = 0.558598)')
    return rho, y_resid_heldout, preds

# ---------------- execute ----------------
t0 = time.time()
results = {'metadata': {'seed': SEED, 'n_estimators': N_EST, 'device': DEVICE,
                        'tabpfn_version': tabpfn.__version__, 'torch_version': torch.__version__,
                        'n_subjects': n, 'n_features_full': len(feat_cols),
                        'n_features_core': len(core_cols),
                        'mask_full_indicators': keep_full, 'mask_core_indicators': keep_core,
                        'stage': 'observed_only_no_nulls'}}

say('=== B: mask-only (full 628 indicators) ===')
results['B_mask_full'] = run_classifiers(M_full, 'B_maskfull')
say('=== B-core: mask-only (common-core indicators) ===')
results['Bcore_mask_core'] = run_classifiers(M_core, 'Bcore_maskcore')
say('=== A: common-core restriction (271 features) ===')
results['A_common_core'] = run_classifiers(X_core, 'A_core')

say('=== R1: fold-safe Test 2 ===')
rho_r1, yres_ho, preds_r1 = run_fold_safe_test2()
results['R1_fold_safe_test2'] = {'fold_safe_rho': rho_r1, 'original_rho': 0.5585976594733492,
                                  'delta': rho_r1 - 0.5585976594733492}
pd.DataFrame({'subject': subjects, 'arm': arms_all, 'y_resid_heldout': yres_ho, 'y_pred': preds_r1}
             ).to_csv(f'{OUT_DIR}/R1_fold_safe_test2_predictions.csv', index=False)

with open(f'{OUT_DIR}/remediation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
say(f'ALL DONE in {(time.time()-t0)/60:.1f} min. Results -> {OUT_DIR}/remediation_results.json')
