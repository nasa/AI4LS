#!/usr/bin/env python3
"""
PHASE5 NULL RUN — Directive PHASE5_NULLS_AND_CLOSEOUT_v1, component B.
Exactly two nulls, 50 shuffles each, seed 42, n_est=32, tabpfn.__version__ logged,
checkpoint every 10 shuffles. gpu-battery-6.

  B1 : R1 fold-safe Test 2 null. Observed rho = 0.6646. Permute y WITHIN the
       same fold-safe pipeline: per shuffle, permute the target, then residualize
       using TRAINING-fold arm means per fold (identical to the observed R1 run).
       Does NOT reuse the original Test 2 null (pipeline differs).
  B2 : A-core CONTROL vs FLY null. Observed bacc = 0.784 (271 features, n=19).
       Label-permutation null, stratified by arm, balanced accuracy as statistic.

Checkpointing: after every 10 shuffles, write partial null stats + all per-shuffle
values to /mnt/shared-workspace/tfm_battery_v2/phase5_nulls_checkpoint.json so the
run can resume after hibernation/OOM without redoing completed shuffles.
"""
import os, json, time, warnings
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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
from sklearn.metrics import balanced_accuracy_score
import tabpfn

SEED = 42
N_EST = 32
DEVICE = 'cuda'
N_SHUFFLES = 50
CHECKPOINT_EVERY = 10
SHARED = '/mnt/shared-workspace/tfm_battery_v2'
DATA_DIR = f'{SHARED}/data'
BMD_TABLE = f'{DATA_DIR}/c11_totalhipBMD_change_features_all.csv'
ARM_MAP = f'{DATA_DIR}/c11_subject_arm_map.csv'
OUT_DIR = '/mnt/results/tfm_battery_v2/phase5_nulls'
# Checkpoint lives on local /workspace (POSIX-safe); mirrored to shared mount via
# plain copy. os.replace / atomic rename is NOT implemented on the S3-backed mount.
CKPT_LOCAL = '/workspace/phase5_nulls_checkpoint.json'
CKPT = f'{SHARED}/phase5_nulls_checkpoint.json'
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

# ---------------- load data (identical to remediation) ----------------
df = pd.read_csv(BMD_TABLE, index_col=0)
y_reg = df['totalhip_BMD_change'].values.astype(np.float64)
X_df = df.drop(columns=['totalhip_BMD_change'])
subjects = df.index.tolist()
arm_map = pd.read_csv(ARM_MAP)
arm_dict = dict(zip(arm_map['subject'], arm_map['arm']))
arms_all = np.array([arm_dict[s] for s in subjects])
n = len(subjects)
X_full = X_df.values.astype(np.float32)

# common core (identical construction to remediation)
P = pd.DataFrame({a: X_df[arms_all==a].notna().sum(axis=0) for a in ['CONTROL','EXERCISE','FLY']})
core_mask = (P['CONTROL']>=1)&(P['EXERCISE']>=1)&(P['FLY']>=1)
core_cols = core_mask[core_mask].index.tolist()
X_core = X_df[core_cols].values.astype(np.float32)
say(f'Loaded: n={n}, full={X_full.shape[1]}, core={X_core.shape[1]}; tabpfn {tabpfn.__version__}')

# ---------------- B1: fold-safe Test 2 null ----------------
def fold_safe_rho(y_target):
    """Observed-equivalent fold-safe Test 2 rho for a given target vector."""
    preds = np.full(n, np.nan)
    y_resid_heldout = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        train_arms = arms_all[tr]
        train_means = {a: y_target[tr][train_arms==a].mean() for a in np.unique(train_arms)}
        y_resid_train = np.array([y_target[tr][j] - train_means[train_arms[j]] for j in range(tr.sum())])
        y_resid_heldout[i] = y_target[i] - train_means[arms_all[i]]
        m = make_regressor()
        m.fit(X_full[tr], y_resid_train)
        preds[i] = m.predict(X_full[i:i+1])[0]
    return float(spearmanr(y_resid_heldout, preds).statistic)

# ---------------- B2: A-core CONTROL vs FLY null ----------------
cf_mask = (arms_all=='CONTROL')|(arms_all=='FLY')
X_cf = X_core[cf_mask]
y_cf = arms_all[cf_mask]
n_cf = int(cf_mask.sum())

def loocv_bacc(Xmat, ylabels):
    nn = len(ylabels)
    preds = np.full(nn, fill_value=None, dtype=object)
    for i in range(nn):
        tr = np.arange(nn) != i
        m = make_classifier()
        m.fit(Xmat[tr], ylabels[tr])
        preds[i] = m.predict(Xmat[i:i+1])[0]
    return float(balanced_accuracy_score(np.array(ylabels), np.array(preds)))

def stratified_permute(labels, rng):
    """DEPRECATED / BUGGY: permuted within each arm, returning labels unchanged.
    Kept only for reference. Use unrestricted_permute for the label null."""
    lab = np.array(labels).copy()
    for a in np.unique(lab):
        idx = np.where(lab==a)[0]
        lab[idx] = rng.permutation(lab[idx])
    return lab

def unrestricted_permute(labels, rng):
    """Correct label-permutation null: freely shuffle ALL labels, breaking any
    feature->label association while preserving the class-count split."""
    return rng.permutation(np.array(labels))

# ---------------- checkpoint helpers ----------------
def load_ckpt():
    # Prefer local checkpoint; fall back to the shared-mount mirror.
    for path in (CKPT_LOCAL, CKPT):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                continue
    return {'B1_null_rhos': [], 'B2_null_bacc': [], 'done': False}

def save_ckpt(state):
    # Write local (POSIX) checkpoint directly, then mirror to shared mount with a
    # plain copy (no os.replace on S3-backed FUSE).
    with open(CKPT_LOCAL, 'w') as f:
        json.dump(state, f)
    try:
        import shutil
        shutil.copyfile(CKPT_LOCAL, CKPT)
    except Exception as e:
        say(f'[ckpt] mirror to shared mount failed (local copy safe): {e}')

# ---------------- execute ----------------
t0 = time.time()
state = load_ckpt()
rng = np.random.default_rng(SEED)
# Advance RNG past any already-completed shuffles to keep permutations distinct
# and reproducible: regenerate the full permutation schedule deterministically.
perm_schedule_B1 = [np.random.default_rng(SEED*1000 + s) for s in range(N_SHUFFLES)]
perm_schedule_B2 = [np.random.default_rng(SEED*2000 + s) for s in range(N_SHUFFLES)]

say(f'=== B1: fold-safe Test 2 null ({N_SHUFFLES} shuffles) ===')
start_b1 = len(state['B1_null_rhos'])
if start_b1 > 0:
    say(f'Resuming B1 at shuffle {start_b1}')
for s in range(start_b1, N_SHUFFLES):
    r = perm_schedule_B1[s]
    y_perm = r.permutation(y_reg)   # permute target; fold-safe residualization happens inside
    rho = fold_safe_rho(y_perm)
    state['B1_null_rhos'].append(rho)
    say(f'[B1] shuffle {s+1}/{N_SHUFFLES} rho={rho:.4f}')
    if (s+1) % CHECKPOINT_EVERY == 0:
        save_ckpt(state)
        say(f'[B1] checkpoint at {s+1} shuffles')
save_ckpt(state)

say(f'=== B2: A-core CONTROL vs FLY null ({N_SHUFFLES} shuffles, n={n_cf}) ===')
start_b2 = len(state['B2_null_bacc'])
if start_b2 > 0:
    say(f'Resuming B2 at shuffle {start_b2}')
for s in range(start_b2, N_SHUFFLES):
    r = perm_schedule_B2[s]
    y_perm = unrestricted_permute(y_cf, r)   # FIXED: was stratified_permute (degenerate)
    bacc = loocv_bacc(X_cf, y_perm)
    state['B2_null_bacc'].append(bacc)
    say(f'[B2] shuffle {s+1}/{N_SHUFFLES} bacc={bacc:.4f}')
    if (s+1) % CHECKPOINT_EVERY == 0:
        save_ckpt(state)
        say(f'[B2] checkpoint at {s+1} shuffles')
save_ckpt(state)

# ---------------- summarize ----------------
OBS_B1 = 0.6646216785258114
OBS_B2 = 0.7841  # A-core CONTROL vs FLY balanced accuracy (observed)
b1 = np.array(state['B1_null_rhos'])
b2 = np.array(state['B2_null_bacc'])
# empirical p with floor annotation: (1 + #{null >= obs}) / (1 + N)
p_b1 = float((1 + np.sum(b1 >= OBS_B1)) / (1 + len(b1)))
p_b2 = float((1 + np.sum(b2 >= OBS_B2)) / (1 + len(b2)))

summary = {
    'metadata': {'seed': SEED, 'n_estimators': N_EST, 'device': DEVICE,
                 'tabpfn_version': tabpfn.__version__, 'torch_version': torch.__version__,
                 'n_shuffles': N_SHUFFLES, 'stage': 'phase5_nulls',
                 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
    'B1_fold_safe_test2': {'observed_rho': OBS_B1,
                            'null_rhos': b1.tolist(),
                            'null_mean': float(b1.mean()), 'null_sd': float(b1.std()),
                            'null_p95': float(np.percentile(b1, 95)),
                            'empirical_p': p_b1, 'p_floor': 1.0/(1+len(b1))},
    'B2_acore_CONTROL_vs_FLY': {'observed_bacc': OBS_B2,
                                 'null_bacc': b2.tolist(),
                                 'null_mean': float(b2.mean()), 'null_sd': float(b2.std()),
                                 'null_p95': float(np.percentile(b2, 95)),
                                 'empirical_p': p_b2, 'p_floor': 1.0/(1+len(b2))},
}
_local_res = '/workspace/phase5_nulls_results.json'
with open(_local_res, 'w') as f:
    json.dump(summary, f, indent=2)
import shutil
shutil.copyfile(_local_res, f'{OUT_DIR}/phase5_nulls_results.json')
state['done'] = True
save_ckpt(state)
say(f'B1: obs={OBS_B1:.4f} null_mean={b1.mean():.4f} p={p_b1:.4f} (floor {1.0/(1+len(b1)):.4f})')
say(f'B2: obs={OBS_B2:.4f} null_mean={b2.mean():.4f} p={p_b2:.4f} (floor {1.0/(1+len(b2)):.4f})')
say(f'ALL DONE in {(time.time()-t0)/60:.1f} min. Results -> {OUT_DIR}/phase5_nulls_results.json')
