#!/usr/bin/env python3
"""
TabPFN-3 Aim-1 follow-up probes — C11 total-hip BMD change (everything table).

Three approved cheap probes, run in background in this order:
  1. importance : grouped permutation importance inside LOOCV + shuffle-null on importances.
  2. width      : feature-width breakpoint sweep (random + coverage-matched).
  3. ood        : OOD-checkpoint sensitivity arm (observed LOOCV + small permutation null).

Leakage-safe: every permutation / refit happens INSIDE the LOOCV loop (training folds only
are ever fit; the held-out subject is only ever predicted). No imputation. R2 is never used
as a null statistic. n_estimators pinned at 32.

Determinism: torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8,
seed pinned (random_state=42). Same pins as run_tfm_modeling.py.

S3-note: np.savez / append fail on /mnt mounts. Checkpoints + log go to local /workspace,
then shutil.copy to the S3 shared workspace / results (same pattern as run_tfm_modeling.py).
"""
import os, sys, json, time, hashlib, warnings
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
torch.use_deterministic_algorithms(True)
warnings.filterwarnings('ignore')
from tabpfn import TabPFNRegressor
from scipy.stats import spearmanr

# ---------------- pins (match run_tfm_modeling.py) ----------------
SEED = 42
N_EST = 32
DEVICE = 'cuda'
CKPT_DEFAULT = os.environ.get('TABPFN_CKPT', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tabpfn-v3-regressor-v3_default.ckpt'))
CKPT_OOD = os.environ.get('TABPFN_CKPT_OOD', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tabpfn-v3-regressor-v3_20260506_ood.ckpt'))
TABPFN_VERSION = '8.2.0'
OUTDIR = os.environ.get('C11_OUTDIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'bmd'))
CKPT_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tfm_probe_ckpt')
CKPT_DIR = os.environ.get('TABPFN_CKPT_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'checkpoints', 'tfm_probe_ckpt'))
os.makedirs(CKPT_LOCAL, exist_ok=True)
TARGET = 'totalhip_BMD_change'
EVERYTHING = f'{OUTDIR}/c11_totalhipBMD_change_features_all.csv'
DICT = f'{OUTDIR}/c11_tfm_feature_dictionary.csv'

# probe parameters
IMP_REPEATS = 5      # observed repeats per group (mean+sd)
IMP_NULL_B = 50      # global y-shuffles for the importance shuffle-null
WIDTHS = [5, 25, 100, 250]   # 628 = full table (reuse observed, no refit)
WIDTH_DRAWS = 5
OOD_N_SHUFFLES = 200

LOG_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs', 'run_log_tfm_probes.txt')
LOG = f'{OUTDIR}/run_log_tfm_probes.txt'
_lf = open(LOG_LOCAL, 'a', buffering=1)
def say(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    _lf.write(line + '\n')

def sync_log():
    import shutil
    try:
        _lf.flush(); shutil.copy(LOG_LOCAL, LOG)
    except Exception as e:
        print(f'[warn] log sync failed: {e}', flush=True)

def sha256(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:n]

def make_model(ckpt):
    return TabPFNRegressor(model_path=ckpt, device=DEVICE, n_estimators=N_EST,
                           random_state=SEED, show_progress_bar=False)

def loocv_preds(X, y, ckpt=CKPT_DEFAULT, col_perm=None):
    """LOOCV out-of-fold predictions. Deterministic fold order (subject order).
    If col_perm is given as (cols, perm), apply that SAME row-permutation to those
    columns of X before fitting (used for grouped permutation importance)."""
    if col_perm is not None:
        cols, perm = col_perm
        X = X.copy()
        X[:, cols] = X[perm][:, cols]
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        m = make_model(ckpt)
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i+1])[0]
    return preds

def sp_stat(y, p):
    return float(spearmanr(y, p).statistic)

def neg_mae(y, p):
    return -float(np.mean(np.abs(y - p)))

# ---------------- checkpoint helpers (local then S3 copy) ----------------
def ckpt_local(name):
    return f'{CKPT_LOCAL}/{name}.npz'

def ckpt_load(name):
    import shutil
    s3 = f'{CKPT_DIR}/{name}.npz'
    loc = ckpt_local(name)
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

# ---------------- domain groups ----------------
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

def load_everything():
    df = pd.read_csv(EVERYTHING, index_col=0)
    X = df.drop(columns=[TARGET]).values
    y = df[TARGET].values
    feats = df.columns[:-1].tolist()
    d = pd.read_csv(DICT)
    assert feats == d['slug'].tolist(), 'feature order mismatch with dictionary'
    groups = [assign_group(sf, ml) for sf, ml in zip(d['source_file'], d['measure_label'])]
    groups = np.array(groups)
    return X, y, feats, groups

# ================= PROBE 1: grouped permutation importance =================
def probe_importance():
    say('[importance] loading everything table + groups')
    X, y, feats, groups = load_everything()
    uniq = sorted(pd.unique(groups).tolist())
    say(f'[importance] {len(uniq)} groups: ' + ', '.join(f'{g}({int((groups==g).sum())})' for g in uniq))
    n = len(y)

    # resume state
    state = ckpt_load('importance')
    if state is not None:
        base_sp = float(state['base_sp']); base_nm = float(state['base_nm'])
        # stored as object arrays of [drops_sp_list, drops_nm_list]; coerce back to lists
        obs = {k: [list(state[f'obs_{k}'][0]), list(state[f'obs_{k}'][1])] for k in uniq}
        null = {k: [list(state[f'null_{k}'][0]), list(state[f'null_{k}'][1])] for k in uniq}
        done_b = int(state['done_b'])
        say(f'[importance] RESUME: base_sp={base_sp:.4f} done_b={done_b}')
    else:
        say('[importance] baseline observed LOOCV ...')
        base_preds = loocv_preds(X, y)
        base_sp = sp_stat(y, base_preds); base_nm = neg_mae(y, base_preds)
        say(f'[importance] baseline Spearman={base_sp:.4f}  -MAE={base_nm:.6f}')
        obs, null, done_b = {}, {}, 0
        # observed importance: R repeats per group
        rng_obs = np.random.default_rng(SEED + 2000)
        for g in uniq:
            cols = np.where(groups == g)[0]
            drops_sp, drops_nm = [], []
            for r in range(IMP_REPEATS):
                perm = rng_obs.permutation(n)
                preds = loocv_preds(X, y, col_perm=(cols, perm))
                drops_sp.append(base_sp - sp_stat(y, preds))
                drops_nm.append(base_nm - neg_mae(y, preds))
                say(f'[importance] obs {g} rep{r+1}/{IMP_REPEATS}: drop_sp={drops_sp[-1]:+.4f}')
            obs[g] = (drops_sp, drops_nm)
            ckpt_save('importance', base_sp=base_sp, base_nm=base_nm, done_b=0,
                      **{f'obs_{k}': np.array(v, dtype=object) for k, v in obs.items()},
                      **{f'null_{k}': np.array([], dtype=object) for k in uniq})
        null = {k: ([], []) for k in uniq}

    # shuffle-null on importances: B global y-shuffles
    rng_null = np.random.default_rng(SEED + 3000)
    for _ in range(done_b):
        rng_null.permutation(n)  # advance stream deterministically
    for b in range(done_b, IMP_NULL_B):
        yp = rng_null.permutation(y)
        # baseline under this shuffle
        bp = loocv_preds(X, yp)
        b_sp = sp_stat(yp, bp); b_nm = neg_mae(yp, bp)
        for g in uniq:
            cols = np.where(groups == g)[0]
            perm = rng_null.permutation(n)
            preds = loocv_preds(X, yp, col_perm=(cols, perm))
            null[g][0].append(b_sp - sp_stat(yp, preds))
            null[g][1].append(b_nm - neg_mae(yp, preds))
        done_b = b + 1
        ckpt_save('importance', base_sp=base_sp, base_nm=base_nm, done_b=done_b,
                  **{f'obs_{k}': np.array(obs[k], dtype=object) for k in uniq},
                  **{f'null_{k}': np.array(null[k], dtype=object) for k in uniq})
        say(f'[importance] null shuffle {done_b}/{IMP_NULL_B} banked')
        sync_log()

    # assemble results
    rows = []
    for g in uniq:
        o_sp = np.array(obs[g][0]); o_nm = np.array(obs[g][1])
        n_sp = np.array(null[g][0]); n_nm = np.array(null[g][1])
        # empirical p: fraction of null drops >= observed mean drop (one-sided, +1)
        p_sp = (1 + np.sum(n_sp >= o_sp.mean())) / (len(n_sp) + 1)
        p_nm = (1 + np.sum(n_nm >= o_nm.mean())) / (len(n_nm) + 1)
        rows.append(dict(group=g, n_features=int((groups == g).sum()),
                         obs_drop_spearman=float(o_sp.mean()), obs_drop_spearman_sd=float(o_sp.std()),
                         null_drop_spearman_mean=float(n_sp.mean()), null_drop_spearman_sd=float(n_sp.std()),
                         sep_spearman=float(o_sp.mean() - n_sp.mean()),
                         z_spearman=float((o_sp.mean() - n_sp.mean()) / (n_sp.std() + 1e-12)),
                         emp_p_spearman=float(p_sp), p_floor=1.0 / (len(n_sp) + 1),
                         obs_drop_negmae=float(o_nm.mean()), obs_drop_negmae_sd=float(o_nm.std()),
                         null_drop_negmae_mean=float(n_nm.mean()), null_drop_negmae_sd=float(n_nm.std()),
                         sep_negmae=float(o_nm.mean() - n_nm.mean()),
                         emp_p_neg_mae=float(p_nm)))
    rdf = pd.DataFrame(rows).sort_values('obs_drop_spearman', ascending=False).reset_index(drop=True)
    rdf['rank_by_obs_drop_spearman'] = np.arange(1, len(rdf) + 1)
    rdf.to_csv(f'{OUTDIR}/tfm_group_importance.csv', index=False)
    say('[importance] wrote tfm_group_importance.csv')
    # flag bone group
    bone = rdf[rdf['group'] == 'Bone_mineral_regional'].iloc[0]
    say(f'[importance] Bone_mineral_regional: obs_drop_sp={bone["obs_drop_spearman"]:+.4f} '
        f'null={bone["null_drop_spearman_mean"]:+.4f}+-{bone["null_drop_spearman_sd"]:.4f} '
        f'sep={bone["sep_spearman"]:+.4f} z={bone["z_spearman"]:+.2f} p={bone["emp_p_spearman"]:.4f} '
        f'rank={int(bone["rank_by_obs_drop_spearman"])}/{len(rdf)}')
    sync_log()

# ================= PROBE 2: feature-width breakpoint sweep =================
def probe_width():
    say('[width] loading everything table')
    X, y, feats, groups = load_everything()
    d = pd.read_csv(DICT)
    coverage = d['n_present'].values.astype(float)
    prob = coverage / coverage.sum()   # coverage-matched sampling weights
    n = len(y)

    state = ckpt_load('width')
    if state is not None:
        rows = [tuple(r) for r in state['rows']]
        done_key = set((int(r[0]), int(r[1])) for r in rows)
        say(f'[width] RESUME: {len(rows)} draws banked')
    else:
        rows, done_key = [], set()

    rng = np.random.default_rng(SEED + 4000)
    for W in WIDTHS:
        for draw in range(WIDTH_DRAWS):
            if (W, draw) in done_key:
                continue
            cols = rng.choice(X.shape[1], size=W, replace=False, p=prob)
            preds = loocv_preds(X[:, cols], y)
            sp = sp_stat(y, preds); nm = neg_mae(y, preds)
            rows.append((W, draw, int(W), sp, nm))
            done_key.add((W, draw))
            ckpt_save('width', rows=np.array(rows, dtype=object))
            say(f'[width] W={W} draw={draw+1}/{WIDTH_DRAWS}: Spearman={sp:+.4f}  -MAE={nm:+.6f}')
            sync_log()

    wdf = pd.DataFrame(rows, columns=['width', 'draw', 'n_features', 'spearman', 'neg_mae'])
    # add full-table observed (628) from the main run for reference
    full = pd.DataFrame([{'width': 628, 'draw': 0, 'n_features': 628,
                          'spearman': 0.3762038995750089, 'neg_mae': -0.012086301265701064}])
    wdf = pd.concat([wdf, full], ignore_index=True)
    wdf.to_csv(f'{OUTDIR}/tfm_width_sweep.csv', index=False)
    summ = wdf.groupby('width').agg(n=('spearman', 'size'),
                                    spearman_mean=('spearman', 'mean'), spearman_sd=('spearman', 'std'),
                                    negmae_mean=('neg_mae', 'mean'), negmae_sd=('neg_mae', 'std')).reset_index()
    summ.to_csv(f'{OUTDIR}/tfm_width_sweep_summary.csv', index=False)
    say('[width] wrote tfm_width_sweep.csv + _summary.csv')
    say('[width] summary: ' + '; '.join(f'W={int(r.width)}: sp={r.spearman_mean:+.3f}' for r in summ.itertuples()))
    sync_log()

# ================= PROBE 3: OOD sensitivity arm =================
def probe_ood():
    say(f'[ood] OOD checkpoint sha256={sha256(CKPT_OOD)}')
    X, y, feats, groups = load_everything()
    n = len(y)

    state = ckpt_load('ood')
    if state is not None:
        obs_sp = float(state['obs_sp']); obs_nm = float(state['obs_nm'])
        null_sp = list(state['null_sp']); null_nm = list(state['null_nm'])
        say(f'[ood] RESUME: obs_sp={obs_sp:.4f} nulls={len(null_sp)}')
    else:
        say('[ood] observed LOOCV (OOD checkpoint) ...')
        preds = loocv_preds(X, y, ckpt=CKPT_OOD)
        obs_sp = sp_stat(y, preds); obs_nm = neg_mae(y, preds)
        null_sp, null_nm = [], []
        say(f'[ood] observed Spearman={obs_sp:.4f}  -MAE={obs_nm:.6f}')
        ckpt_save('ood', obs_sp=obs_sp, obs_nm=obs_nm,
                  null_sp=np.array(null_sp), null_nm=np.array(null_nm))

    rng = np.random.default_rng(SEED + 5000)
    for _ in range(len(null_sp)):
        rng.permutation(n)
    for s in range(len(null_sp), OOD_N_SHUFFLES):
        yp = rng.permutation(y)
        preds = loocv_preds(X, yp, ckpt=CKPT_OOD)
        null_sp.append(sp_stat(yp, preds)); null_nm.append(neg_mae(yp, preds))
        done = s + 1
        if done % 25 == 0 or done == OOD_N_SHUFFLES:
            ckpt_save('ood', obs_sp=obs_sp, obs_nm=obs_nm,
                      null_sp=np.array(null_sp), null_nm=np.array(null_nm))
            arr = np.array(null_sp)
            p = (1 + np.sum(arr >= obs_sp)) / (len(arr) + 1)
            say(f'[ood] {done}/{OOD_N_SHUFFLES} shuffles | running p(Spearman)={p:.4f} (floor {1/(done+1):.4f})')
            sync_log()

    arr_sp = np.array(null_sp); arr_nm = np.array(null_nm)
    p_sp = (1 + np.sum(arr_sp >= obs_sp)) / (len(arr_sp) + 1)
    p_nm = (1 + np.sum(arr_nm >= obs_nm)) / (len(arr_nm) + 1)
    rec = dict(table='everything_ood', checkpoint=os.path.basename(CKPT_OOD), n_shuffles=len(null_sp),
               obs_spearman=obs_sp, obs_neg_mae=obs_nm,
               null_spearman_mean=float(arr_sp.mean()), null_spearman_sd=float(arr_sp.std()),
               null_spearman_p95=float(np.percentile(arr_sp, 95)),
               obs_minus_nullmean_spearman=float(obs_sp - arr_sp.mean()),
               z_spearman=float((obs_sp - arr_sp.mean()) / (arr_sp.std() + 1e-12)),
               emp_p_spearman=float(p_sp), emp_p_neg_mae=float(p_nm), p_floor=1.0 / (len(null_sp) + 1))
    pd.DataFrame([rec]).to_csv(f'{OUTDIR}/tfm_results_null_ood.csv', index=False)
    pd.DataFrame({'null_spearman': null_sp, 'null_neg_mae': null_nm}).to_csv(
        f'{OUTDIR}/tfm_null_distribution_everything_ood.csv', index=False)
    say(f'[ood] wrote tfm_results_null_ood.csv | obs_sp={obs_sp:.4f} null_mean={arr_sp.mean():.4f} '
        f'z={rec["z_spearman"]:+.2f} p={p_sp:.4f}')
    sync_log()

# ---------------- main ----------------
def main():
    say('=' * 70)
    say('TabPFN-3 Aim-1 follow-up probes START')
    say(f'PINS: tabpfn=={TABPFN_VERSION} device={DEVICE} n_estimators={N_EST} seed={SEED} deterministic')
    say(f'default ckpt sha256={sha256(CKPT_DEFAULT)}')
    say(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}')
    say('=' * 70)
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    order = ['importance', 'width', 'ood'] if which == 'all' else [which]
    for probe in order:
        if probe == 'importance':
            probe_importance()
        elif probe == 'width':
            probe_width()
        elif probe == 'ood':
            probe_ood()
    say('TabPFN-3 Aim-1 follow-up probes COMPLETE')
    sync_log()

if __name__ == '__main__':
    main()
