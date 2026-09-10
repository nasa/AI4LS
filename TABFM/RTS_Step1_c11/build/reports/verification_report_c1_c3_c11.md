# C1/C3/C11 Data Verification Report — Archive Ground Truth

**Version 1.1 — 2026-08-02** (reconciled to final curated tables). Source: `03_raw_downloads.zip` (3,555 CSVs, 140 datasets). The raw archive is the only ground truth.

> **Reconciliation note (v1.1).** v1.0 reported intermediate coverage figures (C3 spine=21, C3 plasma volume=24) computed during exploratory scanning before the measure dictionary and phase-mapping were finalized. The numbers below are the **definitive** counts from the final curated long table (`long_all_raw.csv`), after correct phase normalization, canonical-source selection, and 0.0→NaN cleaning. Wide modeling tables have slightly smaller n for some endpoints because they additionally drop featureless subjects (e.g. C3_8715, plasma-volume-only) — see `conflicts_audit_trail.csv` CONF-016.

## VERIFIED CORE-4 ENDPOINT COVERAGE (subjects with BOTH pre & post, non-null, measurement level)

| Endpoint | C11 | C3 | C1 | Notes |
|---|---|---|---|---|
| Total Hip BMD | 38 | 16 | 0 | C3A/C3B unrecoverable (no sub-region BMC/area archived); C1 has no Total Hip BMD column |
| Spine BMD (L1-4) | 38 | 26 | 0 | C1 DXA has no phase column; 18 C3 rows have blank TEST_PHASE (excluded) |
| VO2peak (relative) | 37 | 14 | 0 | C1 absolute L/min only, no body weight in file; C3 uses 4 column conventions; **aggregated as MAX over GXT stages** |
| Plasma Volume | 36 | 20 | 0 | C1 PRE only; C11 Results.csv has only 6 subjects — MUST use all_subjects.csv |

## VERIFIED C11-RICH ENDPOINT COVERAGE

| Endpoint | C11 pre+post | Aggregation |
|---|---|---|
| LV mass (3D echo) | 36 | mean |
| Vertical Jump MaxPower | 36 | max over trials |
| FTT Egress Time | 35 | mean |
| MRI Quadriceps volume | 27 | mean |
| OGTT Glucose | 24 | **AUC over 0–120 min** (sessions averaged per timepoint first) |

## Wide modeling table n (after leakage guard + featureless-subject drop)

| Endpoint | n_total | n_C11 | n_C3 |
|---|---|---|---|
| Total Hip BMD | 54 | 38 | 16 |
| Spine BMD L1-4 | 63 | 38 | 25 |
| VO2peak (rel) | 51 | 37 | 14 |
| Plasma Volume | 55 | 36 | 19 |
| LV mass | 36 | 36 | 0 |
| Jump MaxPower | 36 | 36 | 0 |
| MRI Quadriceps | 26 | 26 | 0 |
| OGTT Glucose AUC | 24 | 24 | 0 |
| FTT Egress Time | 35 | 35 | 0 |

## DATA-INTEGRITY LAND MINES CONFIRMED (must handle in curated build)

1. **Arm-label conflict**: subject 6546 = CONTROL in iDXA/MRI/FTT/PV/NNX10AP86G but EXERCISE in MR080G. Majority + Cromwell 2018 => CONTROL. MR080G GROUP column unreliable for 6546.
2. **Plasma Volume file trap**: BRSMPV_CFT70_PLASMA_VOLUME_Results.csv contains only 6 subjects; all_subjects.csv has 37. Using Results.csv silently drops 31 subjects.
3. **BRSMPV vs FTT PlasmaVolume discrepancy**: 17/70 subject-phase pairs differ by >0.05 L (max 0.30 L), corr 0.98. Two sources NOT identical — canonical source = BRSMPV (dedicated CO-rebreathing assay).
4. **C3 VO2peak column chaos**: 4 conventions ('VO2 ml/kg/min', 'VO2 (ml/kg/min)', 'RVO2', 'RVO2 (ml/kg/min)') across 91 Peak files. Normalized.
5. **C3 subject ID artifacts**: '7352.0' float; campaign code in filename not SUBJECT column (SUBJECT col sometimes = campaign code e.g. 'C3A').
6. **C1 subject ID typo**: 'CIG0003' (letter I) vs 'C1G0003' (digit 1).
7. **C1 contributes ~nothing to pre/post change endpoints**: only 3 subjects (C1G0001-3), VO2/PV PRE only, DXA no phase. C1 value is bone biochemistry (MR016G: PTH, Osteocalcin, BSAP, 25-OH-VitD) which C11 LACKS.
8. **Phase-label babel**: iDXA 'Pre/Test1-4/Post'; MR080G 'PRE/POST1/POST2'; BRSMPV/FTT/C3 'PRE_TEST/IN_TEST/POST_TEST'; C1 DXA none.
9. **C3 Total Hip BMD partial**: iterations C3A/C3B lack the column (subjects 7772, 7352 partially). UNRECOVERABLE — flag, do not infer.

## SUBJECT OVERLAP

- C11 ∩ C3: NONE (disjoint cohorts). C11 ∩ C1: NONE.
- Pooling = independent cohorts, NOT repeated measures. Campaign-held-out validation is legitimate.
- **ID collision**: subject 7152 exists in both C11 and C3 as different people → composite `subject_uid = campaign_subjectid` used everywhere.

## ARM ASSIGNMENTS (C11, archive-verified)

- CONTROL=11, EXERCISE=19 (iRAT 10 + ExT 8, combined in demographics), FLY=8. Total 38.
- NNX10AP86G IMMULITE Treatment (testosterone blinding): PLACEBO / EXERCISE A / EXERCISE B / DROPPED, 25 subjects.
- **C3 arm = BEDREST (inference)**: no arm/GROUP column exists in C3 DXA/PV/VO2 files; C3 is treated as a single-arm 60-day HDT campaign. This is an inference from the *absence* of arm structure, not positive confirmation — see README caveats.
