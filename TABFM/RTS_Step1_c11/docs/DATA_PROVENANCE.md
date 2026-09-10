# DATA_PROVENANCE.md — Where Every Number Comes From

**Status:** verified against stored artifacts. Scope: the outcome variable and the 628 predictors in `data/`. Result statistics live in `results/`; this document covers data lineage only.

## 1. The outcome (what is predicted)

**Absolute change in total hip bone mineral density (BMD), Post − Pre, in g/cm².**

| Field | Value |
|---|---|
| Archive file | `BEDREST_iRATS/BEDREST_IRATS_iDXA_CFT70/BEDREST_IRATS_iDXA_CFT70_Body_Composition_Hips_obsv.csv` |
| Scanner | iDXA (GE Lunar) |
| Measure | BMD (g/cm²) |
| Region of interest | Mean of Left and Right Total Hip |
| Pre timepoint | Test = Pre (before bed rest) |
| Post timepoint | Test = Post (after 70 days of 6-degree head-down tilt) |
| Target variable | Absolute change: Post − Pre (g/cm²) |
| Subjects with paired data | n = 38 |

Subject 5016 was dropped (no hip DXA data). Cohort: 37 male, 1 female (subject 5210, CONTROL). Arms: CONTROL 11, EXERCISE 19 (Exercise A and Exercise+Testosterone pooled), FLY 8 (flywheel resistive exercise).

The canonical definition for every measure (source file, value column, ROI, units, phase mapping) is in `results/measure_dictionary.csv`; the full candidate-outcome registry with pair counts and leakage families is in `results/c11_target_registry.csv`.

## 2. The predictors (what it is predicted with)

628 features, all **pre-bed-rest (baseline)**, one row per subject. Selection pipeline:

1. Raw archive: 4,809 files across 140 datasets, campaigns C1/C3/C11 (verified inventory: `results/archive_dataset_inventory.csv`). Count disclosure: the LSDA catalog listed 4,834 files; the 25-file difference is not itemizable from preserved artifacts (downloader inventory JSON not preserved; regenerable by re-running the downloader's inventory step). Both numbers are disclosed; neither is silently replaced.
2. C11 source files flattened into a master table (43 subjects × 7,545 columns, 32.1% of cells filled — `build/C11_data_dictionary.md`)
3. Coverage filter: features present in ≥80% of subjects (`min_coverage = 0.8`)
4. PRE-only filter: baseline timepoints only (leakage guard against post data)
5. Leakage-family exclusion: the hip iDXA source file (6 columns) removed
6. Result: 628 features

Feature counts by source dataset, verified against the delivered matrix header:

| Source dataset | n | Measures |
|---|---|---|
| BEDREST_FTT functional battery | 141 | Balance, mobility, egress course, strength, jump, heart rate across tests, dynamic visual acuity |
| NNX10AP86G iDXA (whole-body regional) | 109 | Regional BMD, BMC, fat, lean, tissue |
| BRSMCF 2D echo | 54 | Cardiac structure and function |
| NNX10AP86G amino acids | 40 | Plasma amino acid panel |
| MR080G screening | 37 | VO2peak and fitness testing, screening visit |
| CRF vitals / water intake | 36 | Daily vitals and fluid intake |
| MR080G pre | 36 | Fitness battery, pre-bed-rest visit |
| BRSMImmune | 35 | Immune cell phenotyping and activation |
| BRSMPV plasma volume | 27 | Plasma, blood, red-cell volume; hemoglobin |
| iDXA body composition (regional) | 23 | Android/gynoid/arms/legs/trunk/total fat and lean |
| BRSMVSI | 19 | Viral reactivation (EBV, CMV) and cortisol |
| NCC cognitive battery | 18 | Digit symbol, card/cube rotation, tapping, pegboard, rod-and-frame |
| BRSMVJ vertical jump | 14 | Jump mechanics |
| MRI/ultrasound muscle | 13 | Quadriceps, hamstrings, gastrocnemius, soleus, adductors volumes |
| NNX10AP86G POMS/MFSI | 13 | Mood and fatigue inventories |
| NNX10AP86G IMMULITE | 12 | Endocrine/immune assays |
| MR016G daily intake | 1 | Daily energy intake |
| **Total** | **628** | |

**Provenance honesty note.** The verifiable endpoints of the build are the archive inventory (4,809 files, 140 datasets) and the final table (38 × 628, re-read and re-counted). Intermediate build counts from the original feature-engineering run are quoted in older project documents, but the build scripts and intermediate tables are not preserved, so those numbers are deliberately not restated here.

## 3. Reproducibility

| Parameter | Value |
|---|---|
| Model | TabPFN-3, `n_estimators=32`, model seed 42 (every fit) |
| Python | 3.11.13 |
| PyTorch | 2.13.0+cu130 |
| GPU | NVIDIA A10 |
| LOOCV fold order | Deterministic (subject index 0..37) |
| Permutation seeds | `seed[i] = i` (battery); `42*1000+i`, `42*2000+i` (phase-5 nulls) |
| Input table MD5 | `ffd3c4fec75ab2d627985e03c162dddb` (recorded in every battery JSON) |

TabPFN-3 weights are gated and non-commercial; they are not in this repo. Set `TABPFN_TOKEN` from a registered PriorLabs account.

## 4. References

- Cromwell RL, Scott JM, Downs M, et al. (2018) Overview of the NASA 70-day Bed Rest Study. Med Sci Sports Exerc. doi:10.1249/mss.0000000000001617
- Ploutz-Snyder LL, Downs M, Goetchius E, et al. (2018) Exercise Training Mitigates Multisystem Deconditioning during Bed Rest. Med Sci Sports Exerc. doi:10.1249/mss.0000000000001618
- Smith SM, et al. (2014) Calcium kinetics during bed rest with artificial gravity and exercise countermeasures. Osteoporos Int. doi:10.1007/s00198-014-2754-x
- Hollmann N, et al. (2025) Nature 637, 319–326. (TabPFN)
