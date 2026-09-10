# Curated NASA Bed Rest Master Data Asset — C1 / C3 / C11

**Version 1.0 — 2026-08-02.** Built directly from the raw LSDA archive (`03_raw_downloads.zip`, 3,555 CSVs, 140 datasets). The raw archive is the only ground truth; no published value was allowed to override an archive value, and nothing unrecoverable was inferred.

This asset exists because the raw LSDA archive is **not internally consistent** — it contains conflicting arm labels, a plasma-volume file that silently drops 31 of 37 subjects, four different column names for the same VO2peak quantity, three different phase-label schemes, subject-ID typos and float artifacts, and cross-campaign ID collisions. Feeding the raw files directly to a tabular foundation model would silently corrupt every downstream result. Every one of these land mines is documented, resolved where the evidence supports it, and flagged where it does not.

---

## Files in this asset

| File | Rows × Cols | Contents |
|---|---|---|
| `roster_c1_c3_c11.csv` | 109 × 14 | Canonical subject roster. One row per subject: `subject_uid` (composite `campaign_subjectid`), campaign, raw subject_id, arm (resolved), raw arm labels, arm_conflict flag, arm_resolution, sex/age/ht/wt where archived, in_core4 flag. |
| `measure_dictionary.csv` | 16 × 11 | Harmonization dictionary. Maps each canonical measure → per-campaign source file, value-column variant(s), unit, phase-label map, aggregation rule, notes. **This drives all extraction.** |
| `long_c11.csv` / `long_c3.csv` / `long_c1.csv` | 2892 / 1185 / 45 | Per-campaign curated long tables: `subject_uid, campaign, subject_id, measure, phase (PRE/IN/POST), raw_phase, value, unit, br_day, source_file, value_col`. |
| `long_all_campaigns.csv` | 4122 × 12 | Merged cross-campaign long view with `arm` joined on. |
| `wide_modeling_tables/wide_<endpoint>.csv` | 9 files | Per-endpoint subject × feature wide tables ready for TFM input: 8 leakage-guarded Pre-phase features + `<endpoint>_change` target + arm. |
| `conflicts_audit_trail.csv` | 15 × 11 | Every confirmed discrepancy: both values, the resolution, the evidence, severity, action taken. |
| `extraction_audit.csv` | 12 × 4 | Files that failed or partially extracted. |
| `verification_report_c1_c3_c11.md` | — | Archive-ground-truth coverage verification. |
| `manifest.json` | — | SHA256 checksums, dimensions, and sizes for every file (byte-level reproducibility). |

---

## Subject roster (verified against archive)

| Campaign | Subjects | Arm structure | In core-4 (pre+post on ≥1 core endpoint) |
|---|---|---|---|
| **C11** (CFT70, 70-day 6° HDT) | 39 | CONTROL=11, EXERCISE=19 (iRAT 10 + ExT 8), FLY=8 | 38 |
| **C3** (60-day HDT) | 67 (43 screen-only) | BEDREST (single arm, no countermeasure) | 24 |
| **C1** (60-day HDT) | 3 | CONTROL (single arm) | 0 |

- **C11 ∩ C3 = none; C11 ∩ C1 = none.** The campaigns are disjoint cohorts, so pooling is legitimate and campaign-held-out validation is meaningful.
- **ID collision:** subject `7152` exists in both C11 and C3 as *different people*. All tables use the composite key `subject_uid = campaign_subjectid` to prevent silent merging.
- Subject `5016` (C11, Exercise) has MRI data but no iDXA scan; added to roster with `in_core4=NO`.

## Endpoint coverage (verified, subjects with BOTH pre & post, non-null)

| Endpoint | C11 | C3 | C1 | Wide table n |
|---|---|---|---|---|
| Total Hip BMD (g/cm²) | 38 | 16 | 0 | 54 |
| Spine BMD L1-4 (g/cm²) | 38 | 25 | 0 | 63 |
| VO2peak relative (ml/kg/min) | 37 | 14 | 0 | 51 |
| Plasma Volume (L) | 36 | 19 | 0 | 55 |
| LV mass (g) | 36 | — | — | 36 |
| Vertical Jump Max Power (W) | 36 | — | — | 36 |
| FTT Egress Time (s) | 35 | — | — | 35 |
| MRI Quadriceps volume (cm³) | 26 | — | — | 26 |
| OGTT Glucose (mg/dL) | 24 | — | — | 24 |

The four **core-4** endpoints (hip BMD, spine BMD, VO2peak, plasma volume) are the only ones with cross-campaign (C11+C3) coverage and are the primary targets for the pooled TFM study. The five **C11-rich** endpoints are single-campaign secondary targets.

---

## Data-integrity land mines (all confirmed against the archive)

These are the reasons a naive read of the LSDA archive would produce wrong results. Full detail in `conflicts_audit_trail.csv`.

1. **CRITICAL — Plasma-volume file trap (C11).** `BRSMPV_CFT70_PLASMA_VOLUME_Results.csv` contains only **6 of 37** subjects. The complete file is `..._all_subjects.csv`. Using the wrong file silently drops 31 subjects. We use `all_subjects.csv`.
2. **HIGH — Arm-label conflict (C11 subject 6546).** Recorded as CONTROL in iDXA, MRI, FTT, plasma-volume, and NNX10AP86G, but EXERCISE in MR080G. Resolved to **CONTROL** (5 of 6 sources + Cromwell 2018 Table 3).
3. **HIGH — C3 VO2peak column chaos.** Four different column names (`VO2 ml/kg/min`, `VO2 (ml/kg/min)`, `RVO2`, `RVO2 (ml/kg/min)`) across 91 Peak files. Normalized to a single canonical column.
4. **HIGH — C3A/C3B Total Hip BMD unrecoverable.** These iterations lack the `Total Hip BMD` column, and the sub-region BMC/area needed to derive it is not archived. **Flagged, not inferred.**
5. **HIGH — Phase-label babel.** Four schemes: `Pre/Test 1-4/Post` (iDXA), `PRE/POST1/POST2` (MR080G), `PRE_TEST/IN_TEST/POST_TEST` (BRSMPV/FTT/C3), and none (C1 DXA). Normalized to canonical `PRE/IN/POST`.
6. **MEDIUM — BRSMPV vs FTT plasma-volume discrepancy.** 17/70 subject-phase pairs differ by >0.05 L (max 0.30 L, r=0.98). BRSMPV (the dedicated CO-rebreathing assay) is canonical; FTT retained as cross-check.
7. **MEDIUM — C3 arm is an inference, not confirmed.** Clinical-lab `GROUP` columns contain lab-requisition panel names (e.g. "CHEMISTRY PROFILE FOR FLIGHT"), not study arms. No C3 DXA/PV/VO2 file carries any arm/GROUP column, so we label all C3 subjects `arm=BEDREST` on the **inference** that C3 was a single-arm 60-day HDT campaign. This is inferred from the *absence* of arm structure, not positive confirmation; if C3 had countermeasure arms not recorded in these files, the labels would be wrong. Treat C3 arm as unknown-by-design.
8. **LOW — ID artifacts.** C3 `7352.0` float; C1 `CIG0003` (letter I) typo. Normalized.
9. **C1 contributes nothing to core-4 change endpoints** (VO2/PV are PRE-only, DXA has no phase labels). C1's unique value is bone biochemistry (PTH, osteocalcin, BSAP, 25-OH-vitD) that C11 entirely lacks. **Scope note:** although the stated intent was to include C1 fully, only these 3 bone-biochemistry measures were curated for C1's 3 subjects in this pass; C1's ~33 other datasets (tilt, catecholamines, lactulose, etc.) were not extracted. Extending C1 is a documented follow-on, not part of this asset.

## Curation rules applied (carried from prior verified work)

- Raw archive is ground truth; no published value overrides it.
- **No imputation.** Missing stays `NaN` (TFMs handle it natively).
- **0.0 → NaN** for DXA/BMD and VO2peak (physiologically impossible = missing).
- Genuine repeated trials → max (jump) or mean (sessions), per the measure dictionary.
- Dropped-dimension descriptor pairs excluded (e.g. ROI for BMD).
- Demographics/identifiers are never features.
- **Leakage guard:** the target's own baseline and same-region measures are excluded from features.
- Only Pre-phase values are used as features; the target is always `Post − Pre`.
- Deterministic builds; SHA256 checksums in `manifest.json`.

## Strengths

- Every value is traceable to `(source_file, value_col, raw_phase)`.
- Cross-campaign pooling is legitimate (disjoint cohorts, composite keys).
- Core-4 endpoints have verified pre+post coverage on 51–63 subjects.
- All conflicts resolved with documented evidence, or flagged as unrecoverable.

## Weaknesses / limitations (be honest in the manuscript)

- **Small N.** C11=38, C3≤25 per endpoint. This is below the smallest published TabPFN benchmark; results must be characterized empirically, not compared to published Elo/AUC.
- **C1 is essentially unusable** for change endpoints (3 subjects, PRE-only).
- **C3A/C3B Total Hip BMD is unrecoverable** — those subjects are lost for that endpoint.
- **BRSMPV vs FTT plasma volume** disagree by up to 0.3 L on some subjects; we picked one canonical source, but the discrepancy is real and documented.
- **C3 arm is inferred, not confirmed** (see land mine 7): no arm column exists in C3 measurement files, so `arm=BEDREST` is an assumption from absent structure. Arm is therefore informative only for C11 and should be treated as unknown-by-design for C3.
- **Pooled absolute PRE features mix instruments/conventions.** C11 hip BMD is iDXA mean-of-L/R; C3 is DXA left-hip-only. Within-subject change targets largely cancel instrument offset, but the pooled *absolute* PRE feature values do not — cross-campaign feature comparisons should be interpreted with this caveat (or campaign included as a feature).
- **Aggregation is measure-specific and now validated**: VO2peak = max over GXT stages (warmup stages are not peak), OGTT glucose = AUC over 0–120 min, jump = max over trials, others = mean. An earlier pass mean-aggregated VO2peak/OGTT and was corrected after independent review.
- 18 C3 spine rows have genuinely blank `TEST_PHASE` and are excluded from change endpoints.
- The C11 `EXERCISE` arm combines iRAT and Exercise+Testosterone; the testosterone blinding (NNX10AP86G `Treatment`) is available for only 25 subjects.

## Model-selection rationale (for the TFM study built on this asset)

Given n≈38–63 per endpoint, we prioritize TFMs that work in the very-low-data regime: **TabPFN v2/v2.5/3** (prior-free, strong small-data performance, already validated on this exact data in the Aim-1 pilot), **TabICL v2**, **Mitra**, and **TabPFN-Wide**, with **XGBoost** as a non-TFM internal baseline. Primary metric = **Spearman rank correlation of predicted vs actual change against a LOOCV permutation null** (the validated Aim-1 protocol), because continuous physiological change at this N does not support the CAD paper's AUROC/AUPRC framing without fragile dichotomization. Cross-model concordance = pairwise Kendall's tau across SHAP feature rankings.

---

*Provenance: built by Biomni (Phylo) from `03_raw_downloads.zip`. All extraction code is deterministic and available. Every value carries `source_file` + `value_col` for audit. Checksums in `manifest.json`.*
