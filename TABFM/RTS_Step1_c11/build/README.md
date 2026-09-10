# build/ — Data engineering: raw LSDA archive → modeling matrix

**Status:** complete and frozen for Step 1. Scripts are archived as-run; they carry the paths of the original build machine, so expect to set the environment overrides they document (for example `C11_RAW_ARCHIVE`, `C11_RAW_HIPS`, `C11_OUTDIR`) rather than editing code.

## Run order

| # | Script | What it does | Key output |
|---|---|---|---|
| 1 | `lsda_downloader.py` | Bulk-downloads the raw archive for campaigns C1, C3, C11 from NASA LSDA (open access) | the raw file tree (not in this repo) |
| 2 | `build_inventory.py` | Measure-level inventory and normalized long table. Structure only: no collapse, filter, impute, or model | long table + inventory |
| 3 | `profile_raw_archive.py` | File-level inventory and profile, with numbers | `reports/archive_profile_summary.json` |
| 4 | `verify_raw_layer.py` | Read-only verification of the raw layer (nested zips, checksums, open questions) | `reports/verification_report_c1_c3_c11.md` |
| 5 | `audit_csv_headers.py` | Opens every CSV; audits encodings, BOMs, blank/duplicate headers | `reports/header_report.json` |
| 6 | `audit_arm_cohort.py` | Finds every arm/cohort/intervention label in the archive (arm must never enter a feature matrix) | `reports/arm_report.json` |
| 7 | `build_master_v6_1.py` | Flattens the C11 source files into the master table (43 subjects × 7,545 columns) | `master_c11_v6_1.csv` (rebuildable; not shipped) |
| 8 | `enumerate_targets.py` | Catalogs every runnable outcome from the master, with pair counts and leakage families | `results/c11_target_registry.csv` |
| 9 | `build_tfm_totalhipBMD.py` | Builds the hip-BMD-change modeling matrix. **Reads ROI directly from the raw Hips file** | `data/c11_totalhipBMD_change_features_all.csv` |
| 9 | `build_aim1_targets_v3.py` | Builds the other Aim-1 outcome tables | target tables |
| 9 | `build_totalfat_target.py` | Builds the fat-mass-change outcome table | target table |

## Two things a rebuilder must know

1. **D9 — the master build drops the hip ROI dimension.** `build_master_v6_1.py` does not include the hip DXA ROI column in `TIMEPOINT_KEYS`, so the four hip regions (left/right neck, left/right total hip) collapse to one arbitrary value in `master_c11_v6_1.csv`. Fix scheduled for the Step 2 pooled rebuild (v6.2): add `"roi"` to `TIMEPOINT_KEYS`. **Impact on Step 1: none** — the hip modeling matrix comes from `build_tfm_totalhipBMD.py`, which reads ROI directly from the raw Hips file. Verified by recomputing all 38 targets from the raw archive: 38/38 match the shipped matrix exactly.
2. **The master table is not in this repo.** It is rebuildable from the open-access archive with scripts 1–7. The analysis-ready products it feeds (the 38 × 628 modeling matrix, the target registry, the feature dictionary) **are** shipped, in `data/` and `results/`.

## Reports

`reports/` holds the build-time evidence: the archive profile, the header and arm audits, the raw-layer verification report, the curated-master README, and the C11 master report. `C11_data_dictionary.md` (the 32.1% fill source) belongs alongside them and is pending re-upload.
