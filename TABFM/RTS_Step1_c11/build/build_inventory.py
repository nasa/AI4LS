#!/usr/bin/env python3
"""
C11 measure-level inventory + normalized long table  --  v2 (correction pass).
STRUCTURE ONLY. No collapse / filter / impute / model / recommend.
Substrate: master_long.csv (Subject, folder, source_file, variable, timepoint, value).
Ground truth: 03_raw_downloads archive (used only to RE-READ explicit phase labels).

v2 changes vs v1 (both defects; validated in notebook cells 50-62):
  DEFECT 2  -- split key_TEST -> key_TEST_index (numeric) / key_TEST_type (non-numeric).
  DEFECT 1  -- phase_coarse recovery by LABEL RE-READ ONLY (M0 keep -> M1 raw column ->
              M2 Test word -> M3 Time_Period), plus phase_source / phase_conflict.
              NO day calendar is invented (a global day->phase calendar cannot exist;
              see PLAN.md / ActualBedRestTestDay.csv).
M0 logic (split_measure_unit, PHASE_MAP, classify_key, parse_timepoint, parse_numeric)
is UNCHANGED from v1 so phase_coarse before recovery is byte-identical to v1.
"""
import pandas as pd
import re
import os
import sys
from collections import defaultdict

ML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'master_long.csv')
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'c11_normalized_long_v2.csv')

# ---- Unit parsing ------------------------------------------------------
UNIT_PAREN_RE = re.compile(r'^(?P<base>.*?)\s*\((?P<unit>[^()]+)\)\s*$')
MACHINE_UNIT_MAP = {
    'cm': 'cm', 'in': 'in', 'mm': 'mm', 'm': 'm',
    'w': 'W', 'wkg': 'W/kg', 'ms': 'm/s', 'g': 'g',
    'mmhg': 'mmHg', 'pct': '%', 'kg': 'kg', 'bpm': 'bpm',
    'mlkgmin': 'ml/kg/min', 'lmin': 'L/min', 'l': 'L',
}
def canon_unit(u):
    if u is None:
        return ''
    return u.strip()

def split_measure_unit(variable):
    """Return (measure_label_canonical, unit). Strip a clear unit suffix only."""
    v = variable.strip()
    m = UNIT_PAREN_RE.match(v)
    if m:
        base = m.group('base').strip()
        unit = canon_unit(m.group('unit'))
        return base, unit
    if '_' in v:
        head, tail = v.rsplit('_', 1)
        tl = tail.lower()
        if tl in MACHINE_UNIT_MAP:
            return head, MACHINE_UNIT_MAP[tl]
    return v, ''

# ---- Phase mapping for M0 (v1 behaviour, UNCHANGED) --------------------
PHASE_MAP = {
    'pre_test': 'Pre', 'pre-test': 'Pre', 'pre-bedrest': 'Pre',
    'pre1': 'Pre', 'pre2': 'Pre', 'screen': 'Pre',
    'bedrest': 'BR', 'in_test': 'BR',
    'post_test': 'Post', 'post-bedrest': 'Post', 'post1': 'Post', 'post2': 'Post',
    'nan': 'unknown', '': 'unknown',
}
PHASE_KEYS = {'test_phase', 'visit'}  # keys whose VALUE encodes phase (M0)

# ---- Timepoint key classification (UNCHANGED) --------------------------
def classify_key(raw_key):
    k = raw_key.strip().lower()
    if k == 'trial':
        return 'Trial'
    if k == 'session':
        return 'Session'
    if k in ('br day', 'br_day', 'br day', 'day', 'daycount', 'br day'):
        return 'BR_DAY'
    if k.replace(' ', '_') in ('br_day',):
        return 'BR_DAY'
    if k in ('test',):
        return 'TEST'
    if k in ('test_phase', 'visit'):
        return 'phase'
    if k in ('time_period', 'region'):
        return 'other'
    return 'other'

def parse_timepoint(tp):
    """Return dict: {key_type: key_value, ...} plus phase_coarse (M0). Verbatim values."""
    keys = {}
    phase = 'unknown'
    if tp is None or tp == '':
        return keys, phase
    for part in tp.split('|'):
        if '=' not in part:
            keys.setdefault('other', part.strip())
            continue
        k, v = part.split('=', 1)
        k = k.strip()
        v = v.strip()
        kt = classify_key(k)
        if k.lower() in PHASE_KEYS:
            mapped = PHASE_MAP.get(v.lower(), 'unknown')
            if mapped != 'unknown':
                phase = mapped
        if kt in keys:
            keys[kt] = keys[kt] + ';' + v
        else:
            keys[kt] = v
    return keys, phase

# ---- Numeric parsing (UNCHANGED) ---------------------------------------
NUM_RE = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')
def parse_numeric(val):
    """Return (value_numeric_or_blank, nonnumeric_flag)."""
    s = val.strip()
    if s == '':
        return '', False
    s2 = s.replace(',', '') if re.match(r'^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$', s) else s
    if NUM_RE.match(s2):
        try:
            return float(s2), False
        except ValueError:
            return '', True
    return '', True

def pnum(s):
    """Numeric parse returning float or None (used for the key_TEST split)."""
    s = str(s).strip()
    if s == '':
        return None
    s2 = s.replace(',', '') if re.match(r'^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$', s) else s
    return float(s2) if NUM_RE.match(s2) else None

# ---- DEFECT 1: phase recovery by LABEL RE-READ ONLY --------------------
# Priority M0 (kept) -> M1 (raw phase column) -> M2 (Test word) -> M3 (Time_Period).
ARCH = os.environ.get('C11_RAW_ARCHIVE', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'raw'))
# The 6 source_files whose RAW csv carries an explicit phase column that
# master_long dropped when it packed the timepoint.
M1_FILES = ['NNX10AP86G_CFT70_Amino_Acids', 'NNX10AP86G_CFT70_OGTT',
            'NNX10AP86G_CFT70_IMMULITE', 'NNX10AP86G_CFT70_LIPIDS',
            'NCC958SA02802_CFT70_SOT5HeadMovementEQ', 'CRF_CFT70_FARUMEDLOGS_FINAL']
PHASE_COL_TOKENS = {'test phase', 'test_phase', 'phase',
                    'session_test_phase', 'actual_test_phase', 'visit'}
DAY_COL_TOKENS = {'br day', 'br_day', 'day', 'daycount'}
# Recovery map: expanded vs M0 to also catch bare 'pre'/'post' phase WORDS
# that appear as Test values (M2) and Pre/Post variants in raw phase columns.
RECOVERY_PHASE_MAP = {
    'pre_test': 'Pre', 'pre-test': 'Pre', 'pre-bedrest': 'Pre',
    'pre1': 'Pre', 'pre2': 'Pre', 'screen': 'Pre', 'pre': 'Pre',
    'bedrest': 'BR', 'in_test': 'BR',
    'post_test': 'Post', 'post-bedrest': 'Post', 'post1': 'Post',
    'post2': 'Post', 'post': 'Post',
}
def map_phase(v):
    return RECOVERY_PHASE_MAP.get(str(v).strip().lower(), None)

def build_raw_index(arch=ARCH):
    """filename_stem -> path for every csv under the archive (first wins on dup stems)."""
    idx = {}
    for root, _, files in os.walk(arch):
        for fn in files:
            if fn.lower().endswith('.csv'):
                idx.setdefault(fn[:-4], os.path.join(root, fn))
    return idx

def _read_raw(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')

def build_m1_lookup(sf, raw_index):
    """(subject, day) -> phase, ONLY where the mapping is unambiguous among REAL phases.
    Empty/NA/unmappable phases are dropped BEFORE the uniqueness test, and any
    (subject, day) key that still maps to >1 real phase is dropped (stays unknown)."""
    p = raw_index.get(sf)
    if not p:
        return {}
    df = _read_raw(p)
    cols_l = {c.lower().strip(): c for c in df.columns}
    subj_c = cols_l.get('subject') or cols_l.get('id')
    phase_c = next((cols_l[t] for t in PHASE_COL_TOKENS if t in cols_l), None)
    day_c = next((cols_l[t] for t in DAY_COL_TOKENS if t in cols_l), None)
    if not (subj_c and phase_c and day_c):
        return {}
    df['_ph'] = df[phase_c].map(map_phase)
    real = df[df['_ph'].notna()].copy()
    lut = {}
    for (s, d), grp in real.groupby([subj_c, day_c]):
        phs = set(grp['_ph'])
        if len(phs) == 1:
            lut[(str(s).strip(), str(d).strip())] = next(iter(phs))
    return lut

def recover_phase(long_df, raw_index):
    """Fill unknown phase_coarse by label re-read. Adds phase_source / phase_conflict.
    Returns (long_df, stats). Keeps M0 label on conflict, sets phase_conflict=TRUE."""
    L = long_df
    n_unknown_before = int((L['phase_coarse'] == 'unknown').sum())
    m1_luts = {sf: build_m1_lookup(sf, raw_index) for sf in M1_FILES}

    phase_new = L['phase_coarse'].tolist()
    phase_src = ['label' if p != 'unknown' else 'unresolved' for p in phase_new]
    phase_conflict = ['FALSE'] * len(L)
    rec = {'M1': 0, 'M2': 0, 'M3': 0}
    conflicts = 0

    subj = L['subject'].values
    sfile = L['source_file'].values
    day = L['key_BR_DAY'].values
    testtype = L['key_TEST_type'].values
    other = L['key_other'].values
    m3map = {'before': 'Pre', 'during': 'BR', 'after': 'Post'}

    for i in range(len(L)):
        cur = phase_new[i]
        cand = None
        via = None
        sf = sfile[i]
        lut = m1_luts.get(sf)
        if lut:
            cand = lut.get((str(subj[i]).strip(), str(day[i]).strip()))
            if cand:
                via = 'M1'
        if cand is None:
            p = map_phase(testtype[i])
            if p:
                cand = p
                via = 'M2'
        if cand is None:
            o = str(other[i]).strip().lower()
            if o in m3map:
                cand = m3map[o]
                via = 'M3'
        if cand is None:
            continue
        if cur == 'unknown':
            phase_new[i] = cand
            phase_src[i] = 'label'
            rec[via] += 1
        else:
            if cur != cand:
                phase_conflict[i] = 'TRUE'
                conflicts += 1

    L['phase_coarse'] = phase_new
    L['phase_source'] = phase_src            # {label, unresolved}; day_calendar defined but UNUSED
    L['phase_conflict'] = phase_conflict
    n_unknown_after = sum(1 for p in phase_new if p == 'unknown')
    return L, dict(n_unknown_before=n_unknown_before, rec=rec, conflicts=conflicts,
                   n_unknown_after=n_unknown_after)

# ---- Output column order (v1 + additive v2 columns) --------------------
COL_ORDER = ['subject', 'measure_label', 'unit', 'source_file', 'folder', 'timepoint_raw',
             'phase_coarse', 'phase_source', 'phase_conflict',
             'value_raw', 'value_numeric', 'nonnumeric_flag',
             'key_Trial', 'key_Session', 'key_BR_DAY', 'key_TEST',
             'key_TEST_index', 'key_TEST_type', 'key_phase', 'key_other']

def main():
    print('Loading master_long...')
    ml = pd.read_csv(ML_PATH, dtype=str, keep_default_na=False)
    print(f'  {len(ml):,} rows')

    key_type_order = ['Trial', 'Session', 'BR_DAY', 'TEST', 'phase', 'other']
    rows = []
    for r in ml.itertuples(index=False):
        subject = r.Subject
        folder = r.folder
        source_file = r.source_file
        variable = r.variable
        tp_raw = r.timepoint
        value_raw = r.value
        measure_label, unit = split_measure_unit(variable)
        keys, phase = parse_timepoint(tp_raw)
        vnum, nonnum = parse_numeric(value_raw)
        row = {
            'subject': subject,
            'measure_label': measure_label,
            'unit': unit,
            'source_file': source_file,
            'folder': folder,
            'timepoint_raw': tp_raw,
            'phase_coarse': phase,
            'value_raw': value_raw,
            'value_numeric': vnum,
            'nonnumeric_flag': nonnum,
        }
        for kt in key_type_order:
            row['key_' + kt] = keys.get(kt, '')
        rows.append(row)
    long_df = pd.DataFrame(rows)
    print(f'Normalized long table (pre-recovery / M0): {long_df.shape}')

    # DEFECT 2: split key_TEST -> index (numeric) / type (non-numeric)
    long_df['key_TEST_index'] = long_df['key_TEST'].map(lambda v: v if pnum(v) is not None else '')
    long_df['key_TEST_type'] = long_df['key_TEST'].map(
        lambda v: '' if (v == '' or pnum(v) is not None) else v)
    n_idx = int((long_df['key_TEST_index'] != '').sum())
    n_type = int((long_df['key_TEST_type'] != '').sum())
    print(f'DEFECT 2 key_TEST split: index rows={n_idx:,}, type rows={n_type:,}')

    # DEFECT 1: phase recovery (label re-read only)
    print('DEFECT 1 phase recovery (M0 keep -> M1 raw col -> M2 Test word -> M3 Time_Period)...')
    raw_index = build_raw_index()
    long_df, st = recover_phase(long_df, raw_index)
    print(f'  unknown before : {st["n_unknown_before"]:,}')
    print(f'  recovered M1   : {st["rec"]["M1"]:,}')
    print(f'  recovered M2   : {st["rec"]["M2"]:,}')
    print(f'  recovered M3   : {st["rec"]["M3"]:,}')
    print(f'  total recovered: {sum(st["rec"].values()):,}')
    print(f'  unknown after  : {st["n_unknown_after"]:,}')
    print(f'  phase_conflict : {st["conflicts"]:,}')

    long_df = long_df[COL_ORDER]
    return long_df

if __name__ == '__main__':
    df = main()
    df.to_csv(OUT_PATH, index=False)
    print(f'wrote {OUT_PATH}  shape={df.shape}')
