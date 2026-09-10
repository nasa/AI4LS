#!/usr/bin/env python3
"""
TFM Battery v2-final — C11 Bed Rest Project
Pre-registered execution script. Zero spec modifications during run.

Priority order:
  1. prerequisite  — BMD LOOCV, must reproduce +0.376
  2. test1         — arm-stratified null (500 shuffles)
  3. c1            — 3-class ARM classifier (200 shuffles)
  4. test3         — Reading C check (post-hoc, no fits)
  5. test2         — arm-residualized null (500 shuffles)
  6. c2            — binary ARM sub-classifiers (3 pairs, 200 each)
  7. c3            — classification group importance (16 groups, 50 each)
  8. c4            — classification width sweep (optional)
  9. fat_rerun     — 628 features, LOOCV, 500-shuffle null
  10. tandem_rerun — 628 features, LOOCV, 500-shuffle null
  11. conditional  — if fat/tandem p<0.05, Tests 1-3 on that target

Seed policy: model seed 42 in every fit. Permutation seed[i] = i (np.random.RandomState(i)).
Checkpoints every 50 shuffles to /mnt/shared-workspace/tfm_battery_v2/.
"""
import os, sys, json, time, hashlib, warnings, re
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, precision_recall_fscore_support,
                              roc_auc_score)

# ---------------- pins ----------------
SEED = 42
N_EST = 32
DEVICE = 'cuda'
PREREG_VERSION = 'v2-final'
CHUNK = 50  # checkpoint every 50 shuffles

# Paths — data on shared workspace (accessible from GPU machine)
SHARED = '/mnt/shared-workspace/tfm_battery_v2'
DATA_DIR = f'{SHARED}/data'
RESULTS_DIR = '/mnt/results/tfm_battery_v2'
CKPT_DIR = f'{SHARED}/checkpoints'
BMD_TABLE = f'{DATA_DIR}/c11_totalhipBMD_change_features_all.csv'
ARM_MAP = f'{DATA_DIR}/c11_subject_arm_map.csv'
FAT_REBUILT = f'{RESULTS_DIR}/rebuilt_tables/c11_fat_rebuilt_628features.csv'
TANDEM_REBUILT = f'{RESULTS_DIR}/rebuilt_tables/c11_tandem_rebuilt_628features.csv'
FEATURE_DICT = f'{DATA_DIR}/c11_tfm_feature_dictionary.csv'

# Regex guards
LEAKAGE_PAT = re.compile(r"^(group|groupname|grouplabel|arm|cohort|treatment|condition_arm)$", re.I)
ID_PAT = re.compile(r"^(subject|subject[_ ]?id|subjectid|id|index)$", re.I)

# Checkpoint: try env var, then default (TabPFN v3 auto-downloads if no path given)
CKPT_REG = os.environ.get('TABPFN_CKPT', '')

# Logging
def say(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)

def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def _filter_kwargs(wanted, sig_params):
    """Pass only kwargs the installed version accepts."""
    return {k: v for k, v in wanted.items() if k in sig_params}

def make_metadata(input_table_path, component):
    """Build metadata dict embedded in every output."""
    import platform
    return {
        'prereg_version': PREREG_VERSION,
        'component': component,
        'model_seed': SEED,
        'n_estimators': N_EST,
        'permutation_seed_policy': 'seed[i] = i, model_seed = 42, recorded per row',
        'input_table': os.path.basename(input_table_path),
        'input_table_md5': md5_file(input_table_path),
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
            'tabpfn': None,  # filled at runtime
        },
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

# ---------------- model factories ----------------
def make_regressor():
    """Build TabPFNRegressor matching BMD primary run settings.
    Uses flexible kwarg filtering for cross-version compatibility."""
    import inspect
    from tabpfn import TabPFNRegressor
    wanted = {
        'device': DEVICE,
        'random_state': SEED,
        'n_estimators': N_EST,
        'ignore_pretraining_limits': True,  # >200 features triggers subsampling
        'show_progress_bar': False,
    }
    if CKPT_REG and os.path.exists(CKPT_REG):
        wanted['model_path'] = CKPT_REG
    sig = inspect.signature(TabPFNRegressor.__init__).parameters
    kwargs = _filter_kwargs(wanted, sig)
    return TabPFNRegressor(**kwargs)

def make_classifier():
    """Build TabPFNClassifier with same settings."""
    import inspect
    from tabpfn import TabPFNClassifier
    wanted = {
        'device': DEVICE,
        'random_state': SEED,
        'n_estimators': N_EST,
        'ignore_pretraining_limits': True,
        'show_progress_bar': False,
    }
    clf_ckpt = os.environ.get('TABPFN_CLF_CKPT', '')
    if clf_ckpt and os.path.exists(clf_ckpt):
        wanted['model_path'] = clf_ckpt
    sig = inspect.signature(TabPFNClassifier.__init__).parameters
    kwargs = _filter_kwargs(wanted, sig)
    return TabPFNClassifier(**kwargs)

# ---------------- LOOCV ----------------
def loocv_preds_reg(X, y):
    """LOOCV regression predictions. Deterministic fold order (subject index)."""
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        m = make_regressor()
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i+1])[0]
    return preds

def loocv_preds_clf(X, y, return_proba=False):
    """LOOCV classification predictions. Deterministic fold order."""
    n = len(y)
    preds = np.full(n, fill_value=None, dtype=object)
    probas = []
    classes_seen = sorted(np.unique(y))
    for i in range(n):
        tr = np.arange(n) != i
        m = make_classifier()
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i+1])[0]
        if return_proba:
            try:
                p = m.predict_proba(X[i:i+1])[0]
                probas.append(p)
            except Exception:
                probas.append(None)
    if return_proba:
        return preds, probas, classes_seen
    return preds

# ---------------- metrics ----------------
def spearman(ytrue, ypred):
    mask = ~(np.isnan(ytrue) | np.isnan(ypred))
    if mask.sum() < 3:
        return np.nan
    return float(spearmanr(ytrue[mask], ypred[mask]).statistic)

def clf_metrics(ytrue, ypred, classes=None):
    """Classification metrics dict."""
    yt = np.array(ytrue)
    yp = np.array(ypred)
    acc = float(accuracy_score(yt, yp))
    bacc = float(balanced_accuracy_score(yt, yp))
    if classes is None:
        classes = sorted(np.unique(np.concatenate([yt, yp])))
    cm = confusion_matrix(yt, yp, labels=classes).tolist()
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=classes, zero_division=0)
    return {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'confusion_matrix': cm,
        'classes': list(classes),
        'per_class': {
            c: {'precision': float(prec[i]), 'recall': float(rec[i]),
                'f1': float(f1[i]), 'support': int(sup[i])}
            for i, c in enumerate(classes)
        }
    }

# ---------------- checkpointing ----------------
def ckpt_path(name):
    return f'{CKPT_DIR}/{name}_ckpt.npz'

def save_ckpt(name, start, null_metrics, obs_metric):
    """Save checkpoint to shared workspace (survives session death).
    Writes to local disk first (NPZ/ZIP needs random-access writes),
    then copies to S3-backed shared storage."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    local_path = f'/workspace/_ckpt_{name}.npz'
    np.savez(local_path,
             start=start,
             null_metrics=np.array(null_metrics),
             obs_metric=obs_metric)
    import shutil
    shutil.copy(local_path, ckpt_path(name))

def load_ckpt(name):
    """Load checkpoint if exists. Returns (start, null_metrics, obs_metric) or None."""
    p = ckpt_path(name)
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return int(d['start']), list(d['null_metrics']), float(d['obs_metric'])
    return None

def save_null_csv(name, null_metrics, obs_metric, shuffle_seeds, out_dir):
    """Save full null distribution CSV with per-shuffle seeds."""
    df = pd.DataFrame({
        'shuffle_index': range(len(null_metrics)),
        'permutation_seed': shuffle_seeds[:len(null_metrics)],
        'null_metric': null_metrics,
    })
    df.to_csv(f'{out_dir}/{name}_null.csv', index=False)

def save_results_json(name, results, metadata, out_dir):
    """Save results JSON with embedded metadata."""
    results['metadata'] = metadata
    with open(f'{out_dir}/{name}_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

# ---------------- domain groups (from run_tfm_probes.py) ----------------
def assign_group(source_file, measure_label):
    s = source_file.upper(); m = measure_label.upper()
    if m.startswith('BMD_') or m.startswith('BMC_'): return 'Bone_mineral_regional'
    if 'DXA' in s or 'BODY_COMPOSITION' in s: return 'DXA_soft_tissue'
    if 'AMINO_ACID' in s: return 'Amino_acids'
    if 'IMMUNE' in s: return 'Immune'
    if 'IMMULITE' in s: return 'Immunoassay_IMMULITE'
    if 'ECHO' in s: return 'Echo_cardiac'
    if ('PLASMA_VOLUME' in s or 'BLOODVOLUME' in s or 'REDCELLVOLUME' in s or
        'PLASMAVOLUME' in s or 'HEMATOCRIT' in s or 'HEMOGLOBIN' in s): return 'Blood_volume'
    if 'VITALS' in s or 'WATERINTAKE' in s or 'DAILY_INTAKE' in s: return 'Vitals_intake'
    if 'VSI' in s: return 'Muscle_size_VSI'
    if ('VERTICAL_JUMP' in s or 'LEGPRESS' in s or 'UPPERBODY' in s or 'KNEEEXT' in s or
        'TORQGEN' in s or 'MSR_' in s or 'PEGBOARD' in s or 'JUMP' in s or
        'INTERPTWITCH' in s): return 'Muscle_function'
    if 'MRI' in s or 'ULTRASOUND' in s: return 'MRI_ultrasound_muscle'
    if 'POMS' in s or 'MFSI' in s: return 'Psych_POMS_MFSI'
    if ('NCC958' in s or 'CARDROTATION' in s or 'CUBEROTATION' in s or 'DIGIT' in s or
        'DUALTASK' in s or 'TAPPING' in s or 'FMT_' in s or 'RODANDFRAME' in s or
        'BIMANUAL' in s or 'COUNT' in s): return 'Cognitive_NCC'
    if 'SCREEN' in s or 'PRE_ALL' in s: return 'Screening_fitness'
    if ('HR_ACROSS' in s or 'HR_' in s or 'RR_' in s or 'SBP' in s or 'DBP' in s or
        'MAP_' in s or 'PULSEPRESSURE' in s): return 'Cardiovascular_hemo'
    if ('STANDTEST' in s or 'LINETEST' in s or 'DVA_' in s or 'EGRESS' in s or 'EGR' in s or
        'LADDER' in s or 'ROCKS' in s or 'ACTBOARD' in s or 'STANDFZ' in s or 'SOT' in s or
        'EQ_SCORE' in s or 'LC_' in s or 'FALLRECOVERY' in s or 'FALLREC' in s): return 'Balance_mobility_FTT'
    return 'OTHER'

def get_groups(feature_columns):
    """Map feature columns to domain groups using feature dictionary."""
    fd = pd.read_csv(FEATURE_DICT)
    slugs = fd['slug'].tolist()
    assert slugs == feature_columns, 'Feature order mismatch with dictionary'
    groups = np.array([assign_group(sf, ml) for sf, ml in zip(fd['source_file'], fd['measure_label'])])
    return groups

# ---------------- data loading ----------------
def load_bmd():
    """Load BMD table, return X, y, subjects, feature_cols."""
    df = pd.read_csv(BMD_TABLE, index_col=0)
    y = df['totalhip_BMD_change'].values
    X_df = df.drop(columns=['totalhip_BMD_change'])
    # Exclude leakage/ID columns
    keep = [c for c in X_df.columns if not LEAKAGE_PAT.match(c) and not ID_PAT.match(c)]
    X_df = X_df[keep]
    return X_df.values.astype(np.float32), y, df.index.tolist(), X_df.columns.tolist()

def load_arm_labels(subjects):
    """Load arm labels for subjects."""
    arm = pd.read_csv(ARM_MAP)
    arm_dict = dict(zip(arm['subject'], arm['arm']))
    return np.array([arm_dict[s] for s in subjects])

def load_rebuilt(table_path, target_col='target'):
    """Load rebuilt fat/tandem table."""
    df = pd.read_csv(table_path, index_col=0)
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    keep = [c for c in X_df.columns if not LEAKAGE_PAT.match(c) and not ID_PAT.match(c)]
    X_df = X_df[keep]
    return X_df.values.astype(np.float32), y, df.index.tolist(), X_df.columns.tolist()

# ================================================================
# COMPONENT 1: PREREQUISITE — BMD LOOCV reproduction gate
# ================================================================
def run_prerequisite():
    say('[prerequisite] Starting BMD LOOCV reproduction gate')
    out_dir = f'{RESULTS_DIR}/phase2_arm_aware'
    os.makedirs(out_dir, exist_ok=True)

    X, y, subjects, feat_cols = load_bmd()
    say(f'[prerequisite] Data: {X.shape[0]} subjects, {X.shape[1]} features')

    t0 = time.time()
    preds = loocv_preds_reg(X, y)
    elapsed = time.time() - t0
    obs_sp = spearman(y, preds)
    say(f'[prerequisite] LOOCV Spearman = {obs_sp:.7f} (elapsed {elapsed:.1f}s)')
    say(f'[prerequisite] Expected: +0.3762039 (to 3 dp: +0.376)')

    # Save predictions
    pred_df = pd.DataFrame({'subject': subjects, 'y_true': y, 'y_pred': preds})
    pred_df.to_csv(f'{out_dir}/bmd_loocv_predictions.csv', index=False)

    # Gate check
    reproduced = round(obs_sp, 3) == 0.376
    say(f'[prerequisite] Reproduced to 3dp: {reproduced}')

    meta = make_metadata(BMD_TABLE, 'prerequisite')
    meta['environment']['tabpfn'] = __import__('tabpfn').__version__
    meta['loocv_elapsed_sec'] = elapsed

    results = {
        'observed_spearman': obs_sp,
        'expected_spearman': 0.3762039,
        'reproduced_3dp': reproduced,
        'gate_passed': reproduced,
        'n_subjects': len(y),
        'n_features': X.shape[1],
    }
    save_results_json('prerequisite', results, meta, out_dir)

    if not reproduced:
        say('[prerequisite] GATE FAILED. Stopping battery. Report BLOCKED.')
        return False
    say('[prerequisite] GATE PASSED. Proceeding to next component.')
    return True

# ================================================================
# COMPONENT 2: TEST 1 — arm-stratified null (500 shuffles)
# ================================================================
def run_test1():
    say('[test1] Starting arm-stratified null (500 shuffles)')
    out_dir = f'{RESULTS_DIR}/phase2_arm_aware'
    os.makedirs(out_dir, exist_ok=True)

    X, y, subjects, feat_cols = load_bmd()
    arms = load_arm_labels(subjects)
    arm_names = sorted(np.unique(arms))
    n = len(y)
    N_SHUFFLES = 500

    # Observed
    obs_preds = loocv_preds_reg(X, y)
    obs_sp = spearman(y, obs_preds)
    say(f'[test1] Observed Spearman = {obs_sp:.7f}')

    # Resume
    ck = load_ckpt('test1')
    if ck:
        start, null_sps, _ = ck
        say(f'[test1] RESUME from shuffle {start}')
    else:
        start, null_sps = 0, []

    for s in range(start, N_SHUFFLES):
        # Permutation seed = shuffle index
        rng = np.random.RandomState(s)
        y_perm = y.copy()
        for arm in arm_names:
            mask = arms == arm
            y_perm[mask] = y[mask][rng.permutation(mask.sum())]

        preds = loocv_preds_reg(X, y_perm)
        null_sps.append(spearman(y_perm, preds))

        done = s + 1
        if done % CHUNK == 0 or done == N_SHUFFLES:
            save_ckpt('test1', done, null_sps, obs_sp)
            p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
            say(f'[test1] {done}/{N_SHUFFLES} | running p = {p_val:.4f}')

    # Save final
    seeds = list(range(N_SHUFFLES))
    save_null_csv('test1', null_sps, obs_sp, seeds, out_dir)
    p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
    meta = make_metadata(BMD_TABLE, 'test1')
    results = {
        'observed_spearman': obs_sp,
        'n_shuffles': len(null_sps),
        'empirical_p': p_val,
        'null_mean': float(np.mean(null_sps)),
        'null_sd': float(np.std(null_sps)),
        'null_p95': float(np.percentile(null_sps, 95)),
        'arm_names': arm_names,
    }
    save_results_json('test1', results, meta, out_dir)
    say(f'[test1] COMPLETE. p = {p_val:.4f}')
    return results

# ================================================================
# COMPONENT 3: C1 — 3-class ARM classifier (200 shuffles)
# ================================================================
def run_c1():
    say('[c1] Starting 3-class ARM classifier (200 shuffles)')
    out_dir = f'{RESULTS_DIR}/phase1_classification'
    os.makedirs(out_dir, exist_ok=True)

    X, y_reg, subjects, feat_cols = load_bmd()
    arms = load_arm_labels(subjects)
    n = len(arms)
    N_SHUFFLES = 200

    # Observed
    say('[c1] Running observed LOOCV...')
    obs_preds = loocv_preds_clf(X, arms)
    obs_metrics = clf_metrics(arms, obs_preds)
    obs_acc = obs_metrics['accuracy']
    obs_bacc = obs_metrics['balanced_accuracy']
    say(f'[c1] Observed accuracy = {obs_acc:.4f}, balanced accuracy = {obs_bacc:.4f}')

    # Save predictions
    pd.DataFrame({'subject': subjects, 'y_true': arms, 'y_pred': obs_preds}
                 ).to_csv(f'{out_dir}/arm_classifier_predictions.csv', index=False)

    # Resume
    ck = load_ckpt('c1')
    if ck:
        start, null_accs, _ = ck
        say(f'[c1] RESUME from shuffle {start}')
    else:
        start, null_accs = 0, []

    for s in range(start, N_SHUFFLES):
        rng = np.random.RandomState(s)
        arms_perm = arms[rng.permutation(n)]
        preds = loocv_preds_clf(X, arms_perm)
        null_accs.append(float(accuracy_score(arms_perm, preds)))

        done = s + 1
        if done % CHUNK == 0 or done == N_SHUFFLES:
            save_ckpt('c1', done, null_accs, obs_acc)
            p_val = (1 + np.sum(np.array(null_accs) >= obs_acc)) / (len(null_accs) + 1)
            say(f'[c1] {done}/{N_SHUFFLES} | running p = {p_val:.4f}')

    seeds = list(range(N_SHUFFLES))
    save_null_csv('c1', null_accs, obs_acc, seeds, out_dir)
    p_val = (1 + np.sum(np.array(null_accs) >= obs_acc)) / (len(null_accs) + 1)
    meta = make_metadata(BMD_TABLE, 'c1')
    results = {
        'observed_accuracy': obs_acc,
        'observed_balanced_accuracy': obs_bacc,
        'observed_metrics': obs_metrics,
        'n_shuffles': len(null_accs),
        'empirical_p': p_val,
        'null_mean': float(np.mean(null_accs)),
        'null_sd': float(np.std(null_accs)),
        'base_rate': float(np.max(np.bincount(np.searchsorted(sorted(np.unique(arms)), arms)))),
    }
    save_results_json('c1', results, meta, out_dir)
    say(f'[c1] COMPLETE. p = {p_val:.4f}')
    return results

# ================================================================
# COMPONENT 4: TEST 3 — Reading C check (post-hoc, no fits)
# ================================================================
def run_test3():
    say('[test3] Starting Reading C check (post-hoc)')
    out_dir = f'{RESULTS_DIR}/phase2_arm_aware'
    os.makedirs(out_dir, exist_ok=True)

    # Load LOOCV predictions from prerequisite
    pred_path = f'{out_dir}/bmd_loocv_predictions.csv'
    if not os.path.exists(pred_path):
        say('[test3] BLOCKED: prerequisite predictions not found')
        return None
    preds_df = pd.read_csv(pred_path)
    subjects = preds_df['subject'].tolist()
    y_true = preds_df['y_true'].values
    y_pred = preds_df['y_pred'].values
    abs_error = np.abs(y_true - y_pred)

    # Load arm labels
    arms = load_arm_labels(subjects)

    # Load body composition variables
    bmd = pd.read_csv(BMD_TABLE, index_col=0)
    h_col = 'height____mr080g_cft70_pre_all'
    w_col = 'weight____mr080g_cft70_pre_all'
    bmi = bmd[w_col] / (bmd[h_col] / 100) ** 2

    test3_vars = [
        ('total_lean____idxa_cft70_body_composition_total', 'Total lean mass', bmd),
        ('weight____mr080g_cft70_pre_all', 'Body weight', bmd),
        ('height____mr080g_cft70_pre_all', 'Height', bmd),
        ('BMI_COMPUTED', 'BMI (computed)', None),
        ('fat____idxa_cft70_body_composition_trunk', 'Trunk fat', bmd),
        ('lean____idxa_cft70_body_composition_legs', 'Leg lean mass', bmd),
        ('fat____idxa_cft70_body_composition_legs', 'Leg fat', bmd),
        ('total_fat____idxa_cft70_body_composition_total', 'Total fat', bmd),
    ]

    arm_names = ['CONTROL', 'EXERCISE', 'FLY']
    bonferroni_threshold = 0.10 / 16  # 8 vars x 2 testable arms
    rows = []

    for slug, label, src_df in test3_vars:
        if slug == 'BMI_COMPUTED':
            values = bmi.values
        else:
            values = src_df[slug].values

        for arm in arm_names:
            mask = arms == arm
            n_arm = mask.sum()
            n_valid = np.sum(~np.isnan(values[mask]))

            if n_valid < 10:
                rows.append({
                    'arm': arm, 'variable': slug, 'label': label,
                    'n_arm': int(n_arm), 'n_valid': int(n_valid),
                    'rho': np.nan, 'p_value': np.nan,
                    'p_bonferroni': np.nan, 'significant': False,
                    'status': 'UNTESTABLE: n_valid < 10',
                    'bonferroni_threshold': bonferroni_threshold,
                })
                continue

            e = abs_error[mask & ~np.isnan(values)]
            v = values[mask & ~np.isnan(values)]
            if len(e) < 3:
                rows.append({
                    'arm': arm, 'variable': slug, 'label': label,
                    'n_arm': int(n_arm), 'n_valid': int(n_valid),
                    'rho': np.nan, 'p_value': np.nan,
                    'p_bonferroni': np.nan, 'significant': False,
                    'status': 'UNTESTABLE: insufficient paired data',
                    'bonferroni_threshold': bonferroni_threshold,
                })
                continue

            rho, pval = spearmanr(v, e)
            p_bonf = min(pval * 16, 1.0)  # Bonferroni: 16 tests
            rows.append({
                'arm': arm, 'variable': slug, 'label': label,
                'n_arm': int(n_arm), 'n_valid': int(n_valid),
                'rho': float(rho), 'p_value': float(pval),
                'p_bonferroni': float(p_bonf),
                'significant': bool(p_bonf < bonferroni_threshold),
                'status': 'TESTABLE',
                'bonferroni_threshold': bonferroni_threshold,
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(f'{out_dir}/test3_reading_c_check.csv', index=False)

    n_testable = (results_df['status'] == 'TESTABLE').sum()
    n_untestable = (results_df['status'].str.startswith('UNTESTABLE')).sum()
    n_sig = (results_df['significant'] == True).sum()
    say(f'[test3] {n_testable} testable, {n_untestable} untestable, {n_sig} significant')

    meta = make_metadata(BMD_TABLE, 'test3')
    results = {
        'n_testable': int(n_testable),
        'n_untestable': int(n_untestable),
        'n_significant': int(n_sig),
        'bonferroni_threshold': bonferroni_threshold,
        'n_tests': 16,
        'untestable_list': results_df[results_df['status'].str.startswith('UNTESTABLE')][['arm', 'variable', 'status']].to_dict('records'),
    }
    save_results_json('test3', results, meta, out_dir)
    say('[test3] COMPLETE')
    return results

# ================================================================
# COMPONENT 5: TEST 2 — arm-residualized null (500 shuffles)
# ================================================================
def run_test2():
    say('[test2] Starting arm-residualized null (500 shuffles)')
    out_dir = f'{RESULTS_DIR}/phase2_arm_aware'
    os.makedirs(out_dir, exist_ok=True)

    X, y, subjects, feat_cols = load_bmd()
    arms = load_arm_labels(subjects)
    arm_names = sorted(np.unique(arms))
    n = len(y)
    N_SHUFFLES = 500

    # Residualize: y_resid = y - arm_mean
    y_resid = y.copy()
    for arm in arm_names:
        mask = arms == arm
        y_resid[mask] = y[mask] - np.mean(y[mask])
    say(f'[test2] y_resid: mean={np.mean(y_resid):.6f}, SD={np.std(y_resid):.6f}')

    # Observed
    obs_preds = loocv_preds_reg(X, y_resid)
    obs_sp = spearman(y_resid, obs_preds)
    say(f'[test2] Observed Spearman = {obs_sp:.7f}')

    # Resume
    ck = load_ckpt('test2')
    if ck:
        start, null_sps, _ = ck
        say(f'[test2] RESUME from shuffle {start}')
    else:
        start, null_sps = 0, []

    for s in range(start, N_SHUFFLES):
        rng = np.random.RandomState(s)
        y_perm = y_resid[rng.permutation(n)]
        preds = loocv_preds_reg(X, y_perm)
        null_sps.append(spearman(y_perm, preds))

        done = s + 1
        if done % CHUNK == 0 or done == N_SHUFFLES:
            save_ckpt('test2', done, null_sps, obs_sp)
            p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
            say(f'[test2] {done}/{N_SHUFFLES} | running p = {p_val:.4f}')

    seeds = list(range(N_SHUFFLES))
    save_null_csv('test2', null_sps, obs_sp, seeds, out_dir)
    p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
    meta = make_metadata(BMD_TABLE, 'test2')
    results = {
        'observed_spearman': obs_sp,
        'n_shuffles': len(null_sps),
        'empirical_p': p_val,
        'null_mean': float(np.mean(null_sps)),
        'null_sd': float(np.std(null_sps)),
        'null_p95': float(np.percentile(null_sps, 95)),
        'y_resid_mean': float(np.mean(y_resid)),
        'y_resid_sd': float(np.std(y_resid)),
    }
    save_results_json('test2', results, meta, out_dir)
    say(f'[test2] COMPLETE. p = {p_val:.4f}')
    return results

# ================================================================
# COMPONENT 6: C2 — binary ARM sub-classifiers (3 pairs, 200 each)
# ================================================================
def run_c2():
    say('[c2] Starting binary ARM sub-classifiers (3 pairs, 200 shuffles each)')
    out_dir = f'{RESULTS_DIR}/phase1_classification'
    os.makedirs(out_dir, exist_ok=True)

    X_all, y_reg, subjects_all, feat_cols = load_bmd()
    arms_all = load_arm_labels(subjects_all)

    pairs = [
        ('CONTROL_vs_EXERCISE', 'CONTROL', 'EXERCISE'),
        ('CONTROL_vs_FLY', 'CONTROL', 'FLY'),
        ('EXERCISE_vs_FLY', 'EXERCISE', 'FLY'),
    ]
    N_SHUFFLES = 200
    all_results = {}

    for pair_name, arm_a, arm_b in pairs:
        say(f'[c2] {pair_name}: {arm_a} vs {arm_b}')
        mask = (arms_all == arm_a) | (arms_all == arm_b)
        X = X_all[mask]
        arms = arms_all[mask]
        n = len(arms)
        say(f'[c2] {pair_name}: n = {n}')

        # Observed
        obs_preds = loocv_preds_clf(X, arms)
        obs_metrics = clf_metrics(arms, obs_preds)
        obs_acc = obs_metrics['accuracy']
        say(f'[c2] {pair_name}: observed accuracy = {obs_acc:.4f}')

        # Save predictions
        pair_subjects = [subjects_all[i] for i in range(len(subjects_all)) if mask[i]]
        pd.DataFrame({'subject': pair_subjects, 'y_true': arms, 'y_pred': obs_preds}
                     ).to_csv(f'{out_dir}/arm_binary_{pair_name}_predictions.csv', index=False)

        # Resume
        ck_name = f'c2_{pair_name}'
        ck = load_ckpt(ck_name)
        if ck:
            start, null_accs, _ = ck
            say(f'[c2] {pair_name}: RESUME from shuffle {start}')
        else:
            start, null_accs = 0, []

        for s in range(start, N_SHUFFLES):
            rng = np.random.RandomState(s)
            arms_perm = arms[rng.permutation(n)]
            preds = loocv_preds_clf(X, arms_perm)
            null_accs.append(float(accuracy_score(arms_perm, preds)))

            done = s + 1
            if done % CHUNK == 0 or done == N_SHUFFLES:
                save_ckpt(ck_name, done, null_accs, obs_acc)
                p_val = (1 + np.sum(np.array(null_accs) >= obs_acc)) / (len(null_accs) + 1)
                say(f'[c2] {pair_name}: {done}/{N_SHUFFLES} | running p = {p_val:.4f}')

        seeds = list(range(N_SHUFFLES))
        save_null_csv(f'c2_{pair_name}', null_accs, obs_acc, seeds, out_dir)
        p_val = (1 + np.sum(np.array(null_accs) >= obs_acc)) / (len(null_accs) + 1)

        meta = make_metadata(BMD_TABLE, f'c2_{pair_name}')
        results = {
            'pair': pair_name,
            'arm_a': arm_a, 'arm_b': arm_b,
            'n': n,
            'observed_accuracy': obs_acc,
            'observed_balanced_accuracy': obs_metrics['balanced_accuracy'],
            'observed_metrics': obs_metrics,
            'n_shuffles': len(null_accs),
            'empirical_p': p_val,
            'null_mean': float(np.mean(null_accs)),
            'null_sd': float(np.std(null_accs)),
        }
        save_results_json(f'c2_{pair_name}', results, meta, out_dir)
        all_results[pair_name] = results
        say(f'[c2] {pair_name}: COMPLETE. p = {p_val:.4f}')

    return all_results

# ================================================================
# COMPONENT 7: C3 — classification group importance (16 groups, 50 each)
# ================================================================
def run_c3():
    say('[c3] Starting classification group importance (16 groups, 50 shuffles each)')
    out_dir = f'{RESULTS_DIR}/phase1_classification'
    os.makedirs(out_dir, exist_ok=True)

    X, y_reg, subjects, feat_cols = load_bmd()
    arms = load_arm_labels(subjects)
    groups = get_groups(feat_cols)
    group_names = sorted(np.unique(groups))
    n = len(arms)
    N_SHUFFLES_PER_GROUP = 50

    # Observed baseline accuracy
    say('[c3] Computing observed baseline accuracy...')
    obs_preds = loocv_preds_clf(X, arms)
    obs_acc = float(accuracy_score(arms, obs_preds))
    say(f'[c3] Baseline accuracy = {obs_acc:.4f}')

    rows = []
    for g in group_names:
        cols = np.where(groups == g)[0]
        n_feat = len(cols)
        say(f'[c3] Group: {g} ({n_feat} features)')

        # Resume
        ck_name = f'c3_{g}'
        ck = load_ckpt(ck_name)
        if ck:
            start, drops, _ = ck
            say(f'[c3] {g}: RESUME from shuffle {start}')
        else:
            start, drops = 0, []

        for s in range(start, N_SHUFFLES_PER_GROUP):
            rng = np.random.RandomState(s)
            X_perm = X.copy()
            X_perm[:, cols] = X[rng.permutation(n)][:, cols]
            preds = loocv_preds_clf(X_perm, arms)
            perm_acc = float(accuracy_score(arms, preds))
            drops.append(obs_acc - perm_acc)

            done = s + 1
            if done % CHUNK == 0 or done == N_SHUFFLES_PER_GROUP:
                save_ckpt(ck_name, done, drops, obs_acc)

        mean_drop = float(np.mean(drops))
        sd_drop = float(np.std(drops))
        rows.append({
            'group': g, 'n_features': n_feat,
            'obs_drop_mean': mean_drop, 'obs_drop_sd': sd_drop,
            'n_shuffles': len(drops),
        })
        say(f'[c3] {g}: drop = {mean_drop:+.4f} +/- {sd_drop:.4f}')

    results_df = pd.DataFrame(rows).sort_values('obs_drop_mean', ascending=False)
    results_df.to_csv(f'{out_dir}/arm_classifier_group_importance.csv', index=False)

    meta = make_metadata(BMD_TABLE, 'c3')
    results = {
        'baseline_accuracy': obs_acc,
        'n_groups': len(group_names),
        'n_shuffles_per_group': N_SHUFFLES_PER_GROUP,
        'top_group': results_df.iloc[0]['group'],
        'top_drop': float(results_df.iloc[0]['obs_drop_mean']),
    }
    save_results_json('c3', results, meta, out_dir)
    say('[c3] COMPLETE')
    return results

# ================================================================
# COMPONENT 8: C4 — classification width sweep (optional)
# ================================================================
def run_c4():
    say('[c4] Starting classification width sweep (optional)')
    out_dir = f'{RESULTS_DIR}/phase1_classification'
    os.makedirs(out_dir, exist_ok=True)

    X, y_reg, subjects, feat_cols = load_bmd()
    arms = load_arm_labels(subjects)
    n = len(arms)
    rng = np.random.default_rng(SEED)

    widths = [5, 25, 100, 250]
    n_draws = 5
    rows = []

    for w in widths:
        for d in range(n_draws):
            cols = rng.choice(X.shape[1], size=w, replace=False)
            preds = loocv_preds_clf(X[:, cols], arms)
            acc = float(accuracy_score(arms, preds))
            bacc = float(balanced_accuracy_score(arms, preds))
            rows.append({'width': w, 'draw': d, 'accuracy': acc, 'balanced_accuracy': bacc})
            say(f'[c4] width={w} draw={d}: acc={acc:.4f} bacc={bacc:.4f}')

    results_df = pd.DataFrame(rows)
    results_df.to_csv(f'{out_dir}/arm_classifier_width_sweep.csv', index=False)
    say('[c4] COMPLETE')
    return results_df

# ================================================================
# COMPONENT 9: FAT RERUN — 628 features, LOOCV, 500-shuffle null
# ================================================================
def run_fat_rerun():
    say('[fat_rerun] Starting fat A10G rerun (628 features, 500 shuffles)')
    out_dir = f'{RESULTS_DIR}/phase3_fat_tandem'
    os.makedirs(out_dir, exist_ok=True)

    X, y, subjects, feat_cols = load_rebuilt(FAT_REBUILT)
    n = len(y)
    N_SHUFFLES = 500
    say(f'[fat_rerun] Data: {n} subjects, {X.shape[1]} features')

    # Observed
    obs_preds = loocv_preds_reg(X, y)
    obs_sp = spearman(y, obs_preds)
    say(f'[fat_rerun] Observed Spearman = {obs_sp:.7f}')

    pd.DataFrame({'subject': subjects, 'y_true': y, 'y_pred': obs_preds}
                 ).to_csv(f'{out_dir}/fat_rerun_predictions.csv', index=False)

    # Resume
    ck = load_ckpt('fat_rerun')
    if ck:
        start, null_sps, _ = ck
        say(f'[fat_rerun] RESUME from shuffle {start}')
    else:
        start, null_sps = 0, []

    for s in range(start, N_SHUFFLES):
        rng = np.random.RandomState(s)
        y_perm = y[rng.permutation(n)]
        preds = loocv_preds_reg(X, y_perm)
        null_sps.append(spearman(y_perm, preds))

        done = s + 1
        if done % CHUNK == 0 or done == N_SHUFFLES:
            save_ckpt('fat_rerun', done, null_sps, obs_sp)
            p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
            say(f'[fat_rerun] {done}/{N_SHUFFLES} | running p = {p_val:.4f}')

    seeds = list(range(N_SHUFFLES))
    save_null_csv('fat_rerun', null_sps, obs_sp, seeds, out_dir)
    p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
    meta = make_metadata(FAT_REBUILT, 'fat_rerun')
    results = {
        'observed_spearman': obs_sp,
        'n_shuffles': len(null_sps),
        'empirical_p': p_val,
        'null_mean': float(np.mean(null_sps)),
        'null_sd': float(np.std(null_sps)),
        'n_subjects': n,
        'n_features': X.shape[1],
    }
    save_results_json('fat_rerun', results, meta, out_dir)
    say(f'[fat_rerun] COMPLETE. p = {p_val:.4f}')
    return results

# ================================================================
# COMPONENT 10: TANDEM RERUN — 628 features, LOOCV, 500-shuffle null
# ================================================================
def run_tandem_rerun():
    say('[tandem_rerun] Starting tandem A10G rerun (628 features, 500 shuffles)')
    out_dir = f'{RESULTS_DIR}/phase3_fat_tandem'
    os.makedirs(out_dir, exist_ok=True)

    X, y, subjects, feat_cols = load_rebuilt(TANDEM_REBUILT)
    n = len(y)
    N_SHUFFLES = 500
    say(f'[tandem_rerun] Data: {n} subjects, {X.shape[1]} features')

    # Observed
    obs_preds = loocv_preds_reg(X, y)
    obs_sp = spearman(y, obs_preds)
    say(f'[tandem_rerun] Observed Spearman = {obs_sp:.7f}')

    pd.DataFrame({'subject': subjects, 'y_true': y, 'y_pred': obs_preds}
                 ).to_csv(f'{out_dir}/tandem_rerun_predictions.csv', index=False)

    # Resume
    ck = load_ckpt('tandem_rerun')
    if ck:
        start, null_sps, _ = ck
        say(f'[tandem_rerun] RESUME from shuffle {start}')
    else:
        start, null_sps = 0, []

    for s in range(start, N_SHUFFLES):
        rng = np.random.RandomState(s)
        y_perm = y[rng.permutation(n)]
        preds = loocv_preds_reg(X, y_perm)
        null_sps.append(spearman(y_perm, preds))

        done = s + 1
        if done % CHUNK == 0 or done == N_SHUFFLES:
            save_ckpt('tandem_rerun', done, null_sps, obs_sp)
            p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
            say(f'[tandem_rerun] {done}/{N_SHUFFLES} | running p = {p_val:.4f}')

    seeds = list(range(N_SHUFFLES))
    save_null_csv('tandem_rerun', null_sps, obs_sp, seeds, out_dir)
    p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
    meta = make_metadata(TANDEM_REBUILT, 'tandem_rerun')
    results = {
        'observed_spearman': obs_sp,
        'n_shuffles': len(null_sps),
        'empirical_p': p_val,
        'null_mean': float(np.mean(null_sps)),
        'null_sd': float(np.std(null_sps)),
        'n_subjects': n,
        'n_features': X.shape[1],
    }
    save_results_json('tandem_rerun', results, meta, out_dir)
    say(f'[tandem_rerun] COMPLETE. p = {p_val:.4f}')
    return results

# ================================================================
# COMPONENT 11: CONDITIONAL — Tests 1-3 on fat/tandem if p<0.05
# ================================================================
def run_conditional(target_name, table_path, n_subjects):
    """Run Tests 1-3 on a conditional target (fat or tandem)."""
    say(f'[conditional] Starting arm-aware tests on {target_name}')
    out_dir = f'{RESULTS_DIR}/phase4_conditional'
    os.makedirs(out_dir, exist_ok=True)

    X, y, subjects, feat_cols = load_rebuilt(table_path)
    arms = load_arm_labels(subjects)
    arm_names = sorted(np.unique(arms))
    n = len(y)
    N_SHUFFLES = 500

    # --- Test 1: arm-stratified null ---
    say(f'[conditional/{target_name}] Test 1: arm-stratified null')
    obs_preds = loocv_preds_reg(X, y)
    obs_sp = spearman(y, obs_preds)

    ck_name = f'cond_{target_name}_test1'
    ck = load_ckpt(ck_name)
    if ck:
        start, null_sps, _ = ck
    else:
        start, null_sps = 0, []

    for s in range(start, N_SHUFFLES):
        rng = np.random.RandomState(s)
        y_perm = y.copy()
        for arm in arm_names:
            mask = arms == arm
            y_perm[mask] = y[mask][rng.permutation(mask.sum())]
        preds = loocv_preds_reg(X, y_perm)
        null_sps.append(spearman(y_perm, preds))
        done = s + 1
        if done % CHUNK == 0 or done == N_SHUFFLES:
            save_ckpt(ck_name, done, null_sps, obs_sp)
            p_val = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)
            say(f'[conditional/{target_name}] test1: {done}/{N_SHUFFLES} | p = {p_val:.4f}')

    save_null_csv(f'cond_{target_name}_test1', null_sps, obs_sp, list(range(N_SHUFFLES)), out_dir)
    t1_p = (1 + np.sum(np.array(null_sps) >= obs_sp)) / (len(null_sps) + 1)

    # --- Test 2: arm-residualized null ---
    say(f'[conditional/{target_name}] Test 2: arm-residualized null')
    y_resid = y.copy()
    for arm in arm_names:
        mask = arms == arm
        y_resid[mask] = y[mask] - np.mean(y[mask])

    obs_preds2 = loocv_preds_reg(X, y_resid)
    obs_sp2 = spearman(y_resid, obs_preds2)

    ck_name = f'cond_{target_name}_test2'
    ck = load_ckpt(ck_name)
    if ck:
        start, null_sps2, _ = ck
    else:
        start, null_sps2 = 0, []

    for s in range(start, N_SHUFFLES):
        rng = np.random.RandomState(s)
        y_perm = y_resid[rng.permutation(n)]
        preds = loocv_preds_reg(X, y_perm)
        null_sps2.append(spearman(y_perm, preds))
        done = s + 1
        if done % CHUNK == 0 or done == N_SHUFFLES:
            save_ckpt(ck_name, done, null_sps2, obs_sp2)
            p_val = (1 + np.sum(np.array(null_sps2) >= obs_sp2)) / (len(null_sps2) + 1)
            say(f'[conditional/{target_name}] test2: {done}/{N_SHUFFLES} | p = {p_val:.4f}')

    save_null_csv(f'cond_{target_name}_test2', null_sps2, obs_sp2, list(range(N_SHUFFLES)), out_dir)
    t2_p = (1 + np.sum(np.array(null_sps2) >= obs_sp2)) / (len(null_sps2) + 1)

    # --- Test 3: Reading C check ---
    say(f'[conditional/{target_name}] Test 3: Reading C check')
    abs_error = np.abs(y - obs_preds)
    bmd = pd.read_csv(BMD_TABLE, index_col=0)
    # Align BMD to conditional subjects
    bmd_aligned = bmd.loc[subjects]
    h_col = 'height____mr080g_cft70_pre_all'
    w_col = 'weight____mr080g_cft70_pre_all'
    bmi = bmd_aligned[w_col] / (bmd_aligned[h_col] / 100) ** 2

    test3_vars = [
        ('total_lean____idxa_cft70_body_composition_total', 'Total lean mass', bmd_aligned),
        ('weight____mr080g_cft70_pre_all', 'Body weight', bmd_aligned),
        ('height____mr080g_cft70_pre_all', 'Height', bmd_aligned),
        ('BMI_COMPUTED', 'BMI (computed)', None),
        ('fat____idxa_cft70_body_composition_trunk', 'Trunk fat', bmd_aligned),
        ('lean____idxa_cft70_body_composition_legs', 'Leg lean mass', bmd_aligned),
        ('fat____idxa_cft70_body_composition_legs', 'Leg fat', bmd_aligned),
        ('total_fat____idxa_cft70_body_composition_total', 'Total fat', bmd_aligned),
    ]

    bonferroni_threshold = 0.10 / 16
    rows = []
    for slug, label, src_df in test3_vars:
        if slug == 'BMI_COMPUTED':
            values = bmi.values
        else:
            values = src_df[slug].values
        for arm in ['CONTROL', 'EXERCISE', 'FLY']:
            mask = arms == arm
            n_arm = mask.sum()
            n_valid = np.sum(~np.isnan(values[mask]))
            if n_valid < 10:
                rows.append({
                    'target': target_name, 'arm': arm, 'variable': slug, 'label': label,
                    'n_arm': int(n_arm), 'n_valid': int(n_valid),
                    'rho': np.nan, 'p_value': np.nan, 'p_bonferroni': np.nan,
                    'significant': False, 'status': 'UNTESTABLE: n_valid < 10',
                    'bonferroni_threshold': bonferroni_threshold,
                })
                continue
            e = abs_error[mask & ~np.isnan(values)]
            v = values[mask & ~np.isnan(values)]
            if len(e) < 3:
                rows.append({
                    'target': target_name, 'arm': arm, 'variable': slug, 'label': label,
                    'n_arm': int(n_arm), 'n_valid': int(n_valid),
                    'rho': np.nan, 'p_value': np.nan, 'p_bonferroni': np.nan,
                    'significant': False, 'status': 'UNTESTABLE: insufficient paired data',
                    'bonferroni_threshold': bonferroni_threshold,
                })
                continue
            rho, pval = spearmanr(v, e)
            p_bonf = min(pval * 16, 1.0)
            rows.append({
                'target': target_name, 'arm': arm, 'variable': slug, 'label': label,
                'n_arm': int(n_arm), 'n_valid': int(n_valid),
                'rho': float(rho), 'p_value': float(pval), 'p_bonferroni': float(p_bonf),
                'significant': bool(p_bonf < bonferroni_threshold), 'status': 'TESTABLE',
                'bonferroni_threshold': bonferroni_threshold,
            })

    test3_df = pd.DataFrame(rows)
    test3_df.to_csv(f'{out_dir}/cond_{target_name}_test3_reading_c_check.csv', index=False)

    meta = make_metadata(table_path, f'conditional_{target_name}')
    results = {
        'target': target_name,
        'test1_p': t1_p, 'test1_obs_spearman': obs_sp,
        'test2_p': t2_p, 'test2_obs_spearman': obs_sp2,
        'test3_n_significant': int((test3_df['significant'] == True).sum()),
        'test3_n_untestable': int((test3_df['status'].str.startswith('UNTESTABLE')).sum()),
    }
    save_results_json(f'conditional_{target_name}', results, meta, out_dir)
    say(f'[conditional/{target_name}] COMPLETE. test1 p={t1_p:.4f}, test2 p={t2_p:.4f}')
    return results

# ================================================================
# MAIN — run components in priority order
# ================================================================
def main():
    say('=' * 70)
    say(f'TFM Battery {PREREG_VERSION} START')
    say(f'SEED={SEED}  N_EST={N_EST}  DEVICE={DEVICE}  CHUNK={CHUNK}')
    say(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        say(f'GPU={torch.cuda.get_device_name(0)}')
    say('=' * 70)

    # Allow running a single component
    component = sys.argv[1] if len(sys.argv) > 1 else 'all'

    components = [
        ('prerequisite', run_prerequisite),
        ('test1', run_test1),
        ('c1', run_c1),
        ('test3', run_test3),
        ('test2', run_test2),
        ('c2', run_c2),
        ('c3', run_c3),
        ('c4', run_c4),
        ('fat_rerun', run_fat_rerun),
        ('tandem_rerun', run_tandem_rerun),
    ]

    if component == 'all':
        # Run prerequisite gate first
        if not run_prerequisite():
            say('BATTERY BLOCKED: prerequisite gate failed')
            return

        # Run components in priority order (skip prerequisite, already done)
        for name, func in components[1:]:
            try:
                func()
            except Exception as e:
                say(f'[ERROR] {name}: {e}')
                say(f'[ERROR] {name}: reporting UNCLASSIFIED, continuing to next component')
                continue

        # Check conditionals
        say('[conditional] Checking fat/tandem results for p<0.05...')
        fat_path = f'{RESULTS_DIR}/phase3_fat_tandem/fat_rerun_results.json'
        tandem_path = f'{RESULTS_DIR}/phase3_fat_tandem/tandem_rerun_results.json'

        for target_name, path, table, n_subj in [
            ('fat', fat_path, FAT_REBUILT, 38),
            ('tandem', tandem_path, TANDEM_REBUILT, 35),
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                if r.get('empirical_p', 1.0) < 0.05:
                    say(f'[conditional] {target_name} p={r["empirical_p"]:.4f} < 0.05. Running arm-aware tests.')
                    try:
                        run_conditional(target_name, table, n_subj)
                    except Exception as e:
                        say(f'[ERROR] conditional/{target_name}: {e}')
                        say(f'[ERROR] conditional/{target_name}: UNCLASSIFIED, continuing')
                else:
                    say(f'[conditional] {target_name} p={r.get("empirical_p", "N/A")} >= 0.05. No conditional needed.')
            else:
                say(f'[conditional] {target_name}: results not found, skipping')

        say('TFM Battery COMPLETE')

    elif component == 'prerequisite':
        run_prerequisite()
    elif component == 'test1':
        run_test1()
    elif component == 'c1':
        run_c1()
    elif component == 'test3':
        run_test3()
    elif component == 'test2':
        run_test2()
    elif component == 'c2':
        run_c2()
    elif component == 'c3':
        run_c3()
    elif component == 'c4':
        run_c4()
    elif component == 'fat_rerun':
        run_fat_rerun()
    elif component == 'tandem_rerun':
        run_tandem_rerun()
    elif component == 'conditional_fat':
        run_conditional('fat', FAT_REBUILT, 38)
    elif component == 'conditional_tandem':
        run_conditional('tandem', TANDEM_REBUILT, 35)
    else:
        say(f'Unknown component: {component}')
        sys.exit(1)

if __name__ == '__main__':
    main()
