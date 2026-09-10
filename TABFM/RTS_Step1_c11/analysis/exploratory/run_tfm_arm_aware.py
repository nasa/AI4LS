#!/usr/bin/env python3

# *** PAUSED — DO NOT RUN WITHOUT EXPLICIT ACKNOWLEDGEMENT ***
# This script is shipped PAUSED. Arm-aware LOOCV conflates treatment effect
# with generalization. Observed reads were prose-sourced, no output CSVs survived.
# To unpause: set C11_ARM_AWARE_CONFIRMED=1 and remove this notice.
# Status: PAUSED since 2026-08-03.

"""
TabPFN-3 C11 total-hip BMD change — ARM-AWARE robustness re-analysis (WS2).

Motivation: the pooled n=38 analysis mixes three treatment arms (CONTROL sedentary,
EXERCISE, FLY). The target (total-hip BMD change) is itself arm-graded (controls lose
most), and the driving features (Screening_fitness / CPET) are treatment-exposed. So the
pooled rank signal could partly read off exercise-group membership rather than intrinsic
biology. This script runs three leakage-safe probes that reuse the exact LOOCV +
permutation-null machinery from run_tfm_probes.py:

  WS2a  armstrat   : LOOCV + 1000-shuffle y-null WITHIN CONTROL (n=11) and EXERCISE (n=19).
                     Tests whether rank-ordering signal survives inside a single arm.
  WS2b  armadj     : pooled LOOCV + 1000-shuffle null, but Spearman computed on WITHIN-ARM
                     ranks (rank y and preds inside each arm, then pooled Spearman). Tests
                     whether the signal is more than arm-mean separation.
  WS2c  armimp     : grouped permutation importance with the ARM LABEL added as a 17th
                     group, vs Screening_fitness. If arm ~ Screening_fitness -> treatment
                     readout; if Screening_fitness >> arm -> beyond-treatment information.

Leakage-safe: every permutation / refit happens INSIDE the LOOCV loop (training folds only
see training rows). Deterministic seeds. No imputation. Archive = ground truth.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings('ignore')

# ---------------- pins (match run_tfm_probes.py / run_tfm_modeling.py) ----------------
SEED = 42
N_EST = 32
DEVICE = 'cuda'
CKPT_DEFAULT = os.environ.get('TABPFN_CKPT', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tabpfn-v3-regressor-v3_default.ckpt'))
OUTDIR = os.environ.get('C11_OUTDIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'bmd'))
CKPT_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tfm_arm_ckpt')
CKPT_DIR = os.environ.get('TABPFN_CKPT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tfm_arm_ckpt'))
os.makedirs(CKPT_LOCAL, exist_ok=True)
TARGET = 'totalhip_BMD_change'
EVERYTHING = f'{OUTDIR}/c11_totalhipBMD_change_features_all.csv'
DICT = f'{OUTDIR}/c11_tfm_feature_dictionary.csv'
ARMMAP = f'{OUTDIR}/c11_subject_arm_map.csv'

N_SHUFFLES = 1000      # arm-stratified + arm-adjusted nulls
IMP_REPEATS = 5        # observed repeats per group (importance)
IMP_NULL_B = 50        # global y-shuffles for importance null

LOG_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs', 'run_log_tfm_arm.txt')
LOG = f'{OUTDIR}/run_log_tfm_arm.txt'
_lf = open(LOG_LOCAL, 'a', buffering=1)

def say(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    _lf.write(line + '\n')

def sync_log():
    import shutil
    _lf.flush()
    try:
        shutil.copy(LOG_LOCAL, LOG)
    except Exception as e:
        say(f'[warn] log sync failed: {e}')

def make_model(ckpt=CKPT_DEFAULT):
    return TabPFNRegressor(model_path=ckpt, device=DEVICE, n_estimators=N_EST,
                           random_state=SEED, show_progress_bar=False)

def loocv_preds(X, y, col_perm=None):
    """LOOCV out-of-fold predictions, deterministic fold order (subject order).
    If col_perm=(cols, perm), apply that SAME row-permutation to those columns of X
    before fitting (grouped permutation importance)."""
    if col_perm is not None:
        cols, perm = col_perm
        X = X.copy()
        X[:, cols] = X[perm][:, cols]
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        m = make_model()
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i+1])[0]
    return preds

def sp_stat(y, p):
    return float(spearmanr(y, p).statistic)

def neg_mae(y, p):
    return -float(np.mean(np.abs(y - p)))

def within_arm_spearman(y, p, arms):
    """Spearman on within-arm ranks: rank y and p inside each arm, pool, correlate.
    Removes between-arm mean separation; tests within-arm rank-ordering only."""
    yr = np.empty_like(y, dtype=float); pr = np.empty_like(p, dtype=float)
    for a in pd.unique(arms):
        idx = np.where(arms == a)[0]
        yr[idx] = rankdata(y[idx]); pr[idx] = rankdata(p[idx])
    return float(spearmanr(yr, pr).statistic)

# ---------------- checkpoint helpers ----------------
def ckpt_local(name):
    return f'{CKPT_LOCAL}/{name}.npz'

def ckpt_load(name):
    import shutil
    s3 = f'{CKPT_DIR}/{name}.npz'; loc = ckpt_local(name)
    if os.path.exists(s3) and not os.path.exists(loc):
        try:
            os.makedirs(CKPT_LOCAL, exist_ok=True); shutil.copy(s3, loc)
        except Exception:
            pass
    if os.path.exists(loc):
        return np.load(loc, allow_pickle=True)
    return None

def ckpt_save(name, **arrays):
    import shutil
    np.savez(ckpt_local(name), **arrays)
    try:
        os.makedirs(CKPT_DIR, exist_ok=True)
        shutil.copy(ckpt_local(name), f'{CKPT_DIR}/{name}.npz')
    except Exception as e:
        say(f'[warn] ckpt S3 sync failed (local intact): {e}')

# ---------------- domain groups (identical to run_tfm_probes.assign_group) ----------------
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

def load_everything_with_arms():
    df = pd.read_csv(EVERYTHING, index_col=0)
    df.index = df.index.astype(str)
    am = pd.read_csv(ARMMAP, dtype={'subject': str}).set_index('subject')
    # align arms to the everything-table subject order
    arms = am.reindex(df.index)['arm'].values
    assert not pd.isna(arms).any(), 'unmapped subject in everything table'
    X = df.drop(columns=[TARGET]).values
    y = df[TARGET].values
    feats = df.columns[:-1].tolist()
    d = pd.read_csv(DICT)
    assert feats == d['slug'].tolist(), 'feature order mismatch with dictionary'
    groups = np.array([assign_group(sf, ml) for sf, ml in zip(d['source_file'], d['measure_label'])])
    return X, y, feats, groups, arms, df.index.tolist()

# ================= WS2a: arm-stratified permutation nulls =================
def probe_armstrat():
    X, y, feats, groups, arms, subjects = load_everything_with_arms()
    rows = []
    for arm in ['CONTROL', 'EXERCISE']:
        idx = np.where(arms == arm)[0]
        Xa, ya = X[idx], y[idx]
        n = len(ya)
        say(f'[armstrat] === {arm} (n={n}) ===')
        state = ckpt_load(f'armstrat_{arm}')
        if state is not None:
            obs_sp = float(state['obs_sp']); obs_nm = float(state['obs_nm'])
            null_sp = list(state['null_sp']); null_nm = list(state['null_nm'])
            say(f'[armstrat] {arm} RESUME obs_sp={obs_sp:.4f} nulls={len(null_sp)}')
        else:
            preds = loocv_preds(Xa, ya)
            obs_sp = sp_stat(ya, preds); obs_nm = neg_mae(ya, preds)
            null_sp, null_nm = [], []
            say(f'[armstrat] {arm} observed Spearman={obs_sp:.4f}  -MAE={obs_nm:.6f}')
            ckpt_save(f'armstrat_{arm}', obs_sp=obs_sp, obs_nm=obs_nm,
                      null_sp=np.array(null_sp), null_nm=np.array(null_nm))
        rng = np.random.default_rng(SEED + (7000 if arm == 'CONTROL' else 8000))
        for _ in range(len(null_sp)):
            rng.permutation(n)
        for s in range(len(null_sp), N_SHUFFLES):
            yp = rng.permutation(ya)
            preds = loocv_preds(Xa, yp)
            null_sp.append(sp_stat(yp, preds)); null_nm.append(neg_mae(yp, preds))
            done = s + 1
            if done % 100 == 0 or done == N_SHUFFLES:
                ckpt_save(f'armstrat_{arm}', obs_sp=obs_sp, obs_nm=obs_nm,
                          null_sp=np.array(null_sp), null_nm=np.array(null_nm))
                arr = np.array(null_sp)
                p = (1 + np.sum(arr >= obs_sp)) / (len(arr) + 1)
                say(f'[armstrat] {arm} {done}/{N_SHUFFLES} | running p={p:.4f}')
                sync_log()
        arr_sp = np.array(null_sp); arr_nm = np.array(null_nm)
        p_sp = (1 + np.sum(arr_sp >= obs_sp)) / (len(arr_sp) + 1)
        p_nm = (1 + np.sum(arr_nm >= obs_nm)) / (len(arr_nm) + 1)
        rows.append(dict(arm=arm, n=n, n_shuffles=len(null_sp),
                         obs_spearman=obs_sp, obs_neg_mae=obs_nm,
                         null_spearman_mean=float(arr_sp.mean()), null_spearman_sd=float(arr_sp.std()),
                         null_spearman_p95=float(np.percentile(arr_sp, 95)),
                         obs_minus_nullmean_spearman=float(obs_sp - arr_sp.mean()),
                         z_spearman=float((obs_sp - arr_sp.mean()) / (arr_sp.std() + 1e-12)),
                         emp_p_spearman=float(p_sp), emp_p_neg_mae=float(p_nm),
                         p_floor=1.0 / (len(null_sp) + 1)))
        pd.DataFrame({'null_spearman': null_sp, 'null_neg_mae': null_nm}).to_csv(
            f'{OUTDIR}/tfm_null_distribution_armstrat_{arm.lower()}.csv', index=False)
        say(f'[armstrat] {arm}: obs_sp={obs_sp:.4f} null={arr_sp.mean():+.4f}+-{arr_sp.std():.4f} '
            f'z={rows[-1]["z_spearman"]:+.2f} p={p_sp:.4f}')
    pd.DataFrame(rows).to_csv(f'{OUTDIR}/tfm_results_null_byarm.csv', index=False)
    say('[armstrat] wrote tfm_results_null_byarm.csv')
    sync_log()

# ================= WS2b: arm-adjusted (within-arm rank) null =================
def probe_armadj():
    X, y, feats, groups, arms, subjects = load_everything_with_arms()
    n = len(y)
    say(f'[armadj] pooled n={n}, within-arm rank Spearman')
    state = ckpt_load('armadj')
    if state is not None:
        obs_wa = float(state['obs_wa']); obs_sp = float(state['obs_sp'])
        null_wa = list(state['null_wa'])
        say(f'[armadj] RESUME obs_wa={obs_wa:.4f} nulls={len(null_wa)}')
    else:
        preds = loocv_preds(X, y)
        obs_sp = sp_stat(y, preds)
        obs_wa = within_arm_spearman(y, preds, arms)
        null_wa = []
        say(f'[armadj] observed pooled Spearman={obs_sp:.4f}  within-arm Spearman={obs_wa:.4f}')
        ckpt_save('armadj', obs_sp=obs_sp, obs_wa=obs_wa, null_wa=np.array(null_wa))
    rng = np.random.default_rng(SEED + 9000)
    for _ in range(len(null_wa)):
        rng.permutation(n)
    for s in range(len(null_wa), N_SHUFFLES):
        yp = rng.permutation(y)
        preds = loocv_preds(X, yp)
        null_wa.append(within_arm_spearman(yp, preds, arms))
        done = s + 1
        if done % 100 == 0 or done == N_SHUFFLES:
            ckpt_save('armadj', obs_sp=obs_sp, obs_wa=obs_wa, null_wa=np.array(null_wa))
            arr = np.array(null_wa)
            p = (1 + np.sum(arr >= obs_wa)) / (len(arr) + 1)
            say(f'[armadj] {done}/{N_SHUFFLES} | running p={p:.4f}')
            sync_log()
    arr = np.array(null_wa)
    p = (1 + np.sum(arr >= obs_wa)) / (len(arr) + 1)
    rec = dict(table='everything_armadjusted', n=n, n_shuffles=len(null_wa),
               obs_pooled_spearman=obs_sp, obs_withinarm_spearman=obs_wa,
               null_withinarm_mean=float(arr.mean()), null_withinarm_sd=float(arr.std()),
               null_withinarm_p95=float(np.percentile(arr, 95)),
               obs_minus_nullmean=float(obs_wa - arr.mean()),
               z_withinarm=float((obs_wa - arr.mean()) / (arr.std() + 1e-12)),
               emp_p_withinarm=float(p), p_floor=1.0 / (len(null_wa) + 1))
    pd.DataFrame([rec]).to_csv(f'{OUTDIR}/tfm_results_null_armadjusted.csv', index=False)
    pd.DataFrame({'null_withinarm_spearman': null_wa}).to_csv(
        f'{OUTDIR}/tfm_null_distribution_armadjusted.csv', index=False)
    say(f'[armadj] wrote tfm_results_null_armadjusted.csv | obs_wa={obs_wa:.4f} '
        f'null={arr.mean():+.4f}+-{arr.std():.4f} z={rec["z_withinarm"]:+.2f} p={p:.4f}')
    sync_log()

# ================= WS2c: importance with ARM as a group =================
def probe_armimp():
    X, y, feats, groups, arms, subjects = load_everything_with_arms()
    n = len(y)
    # one-hot encode arm (3 cols) and append as a new group 'ARM_LABEL'
    arm_dummies = pd.get_dummies(arms).astype(float).values  # n x 3
    X2 = np.hstack([X, arm_dummies])
    groups2 = np.concatenate([groups, np.array(['ARM_LABEL'] * arm_dummies.shape[1])])
    uniq = sorted(pd.unique(groups2).tolist())
    say(f'[armimp] {len(uniq)} groups incl ARM_LABEL: ' +
        ', '.join(f'{g}({int((groups2==g).sum())})' for g in uniq))

    state = ckpt_load('armimp')
    if state is not None:
        base_sp = float(state['base_sp'])
        obs = {k: list(state[f'obs_{k}']) for k in uniq}
        null = {k: list(state[f'null_{k}']) for k in uniq}
        done_b = int(state['done_b'])
        say(f'[armimp] RESUME base_sp={base_sp:.4f} done_b={done_b}')
    else:
        base_preds = loocv_preds(X2, y)
        base_sp = sp_stat(y, base_preds)
        say(f'[armimp] baseline (with arm features) Spearman={base_sp:.4f}')
        obs, null, done_b = {}, {}, 0
        rng_obs = np.random.default_rng(SEED + 2100)
        for g in uniq:
            cols = np.where(groups2 == g)[0]
            drops = []
            for r in range(IMP_REPEATS):
                perm = rng_obs.permutation(n)
                preds = loocv_preds(X2, y, col_perm=(cols, perm))
                drops.append(base_sp - sp_stat(y, preds))
            obs[g] = drops
            say(f'[armimp] obs {g}: drop_sp={np.mean(drops):+.4f}')
            null[g] = []
        ckpt_save('armimp', base_sp=base_sp, done_b=0,
                  **{f'obs_{k}': np.array(v) for k, v in obs.items()},
                  **{f'null_{k}': np.array(v) for k, v in null.items()})

    rng_null = np.random.default_rng(SEED + 3100)
    for _ in range(done_b):
        rng_null.permutation(n)
    for b in range(done_b, IMP_NULL_B):
        yp = rng_null.permutation(y)
        bp = loocv_preds(X2, yp)
        bsp = sp_stat(yp, bp)
        for g in uniq:
            cols = np.where(groups2 == g)[0]
            perm = rng_null.permutation(n)
            preds = loocv_preds(X2, yp, col_perm=(cols, perm))
            null[g].append(bsp - sp_stat(yp, preds))
        done_b = b + 1
        if done_b % 10 == 0 or done_b == IMP_NULL_B:
            ckpt_save('armimp', base_sp=base_sp, done_b=done_b,
                      **{f'obs_{k}': np.array(v) for k, v in obs.items()},
                      **{f'null_{k}': np.array(v) for k, v in null.items()})
            say(f'[armimp] null shuffle {done_b}/{IMP_NULL_B} banked')
            sync_log()

    rows = []
    for g in uniq:
        o = np.array(obs[g]); nn = np.array(null[g])
        p = (1 + np.sum(nn >= o.mean())) / (len(nn) + 1) if len(nn) else np.nan
        rows.append(dict(group=g, n_features=int((groups2 == g).sum()),
                         obs_drop_spearman=float(o.mean()), obs_drop_spearman_sd=float(o.std()),
                         null_drop_spearman_mean=float(nn.mean()) if len(nn) else np.nan,
                         null_drop_spearman_sd=float(nn.std()) if len(nn) else np.nan,
                         sep_spearman=float(o.mean() - (nn.mean() if len(nn) else 0.0)),
                         z_spearman=float((o.mean() - nn.mean()) / (nn.std() + 1e-12)) if len(nn) else np.nan,
                         emp_p_spearman=float(p), p_floor=1.0 / (len(nn) + 1) if len(nn) else np.nan))
    rdf = pd.DataFrame(rows).sort_values('obs_drop_spearman', ascending=False).reset_index(drop=True)
    rdf['rank_by_obs_drop_spearman'] = np.arange(1, len(rdf) + 1)
    rdf.to_csv(f'{OUTDIR}/tfm_group_importance_witharm.csv', index=False)
    for g in ['ARM_LABEL', 'Screening_fitness', 'Bone_mineral_regional']:
        r = rdf[rdf.group == g].iloc[0]
        say(f'[armimp] {g}: obs_drop={r.obs_drop_spearman:+.4f} null={r.null_drop_spearman_mean:+.4f} '
            f'z={r.z_spearman:+.2f} p={r.emp_p_spearman:.4f} rank={int(r.rank_by_obs_drop_spearman)}/{len(rdf)}')
    say('[armimp] wrote tfm_group_importance_witharm.csv')
    sync_log()

def main():
    say('=' * 70)
    say('TabPFN-3 C11 ARM-AWARE re-analysis START')
    say(f'torch={torch.__version__} cuda={torch.cuda.is_available()} '
        f'gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}')
    say('=' * 70)
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    order = ['armstrat', 'armadj', 'armimp'] if which == 'all' else [which]
    for probe in order:
        if probe == 'armstrat':
            probe_armstrat()
        elif probe == 'armadj':
            probe_armadj()
        elif probe == 'armimp':
            probe_armimp()
    say('TabPFN-3 C11 ARM-AWARE re-analysis COMPLETE')
    sync_log()

if __name__ == '__main__':
    main()
