# RUN_LOG.md

One line per executed run. Model is TabPFN-3 with `n_estimators=32`, model seed 42, deterministic algorithms on, unless stated otherwise. Every battery JSON records the input table MD5 `ffd3c4fec75ab2d627985e03c162dddb`.

Path convention: scripts named without a directory live in `analysis/` (registered battery) or `analysis/exploratory/` (aim1 scripts); build scripts live in `build/`. Output paths are repo-relative and exact.

| Date (UTC) | Script | Run | Output | Model / version / seed |
|---|---|---|---|---|
| 2026-08-16 | `setup_gpu_machine.sh` | Fresh GPU machine (A10G); pinned env installed; TabPFN-3 regressor + classifier checkpoints cached to shared storage with SHA256 recorded | `weights/` (not committed, license) | tabpfn 8.2.0, torch 2.13.0+cu130, Python 3.11.13 |
| 2026-08-20 08:06 | `run_battery.py` | Prerequisite reproduction gate — headline correlation matched the exploratory-run value (expected 0.3762039, observed 0.3762038995750089) before battery start | `results/phase2_arm_aware/prerequisite_results.json` | TabPFN-3, seed 42, n_est 32 |
| 2026-08-20 08:06 | `run_battery.py` | Test 1: arm-stratified permutation null, 500 shuffles | `results/phase2_arm_aware/test1_results.json`, `test1_null.csv` | same |
| 2026-08-20 08:08 | `run_battery.py` | C1: arm classifier (3-class) with null | `results/phase1_classification/c1_results.json`, `c1_null.csv` | same |
| 2026-08-20 08:10 | `run_battery.py` | Test 2: arm-residualized target, 500 shuffles | `results/phase2_arm_aware/test2_results.json`, `test2_null.csv` | same |
| 2026-08-20 | `run_battery.py` | Test 3: Reading C per-arm error screen (16 testable) | `results/phase2_arm_aware/test3_results.json`, `test3_reading_c_check.csv` | same |
| 2026-08-20 | `run_battery.py` | C2: pairwise arm classifiers with nulls; C3 control | `results/phase1_classification/c2_*`, `c3_results.json` | same |
| 2026-08-20 11:48 | fat rerun (run_battery family) | Negative control: fat-mass change | `results/phase3_fat_tandem/fat_rerun_*` | same |
| 2026-08-21 01:38 | tandem rerun (run_battery family) | Negative control: tandem-walk performance | `results/phase3_fat_tandem/tandem_rerun_*` | same |
| 2026-08-26/27 | `run_missingness_remediation.py` | Phase 4: A-core / B-maskfull / Bcore-maskcore / R1 fold-safe | `results/phase4_missingness_remediation/` | same |
| 2026-08-27 03:17 | `run_nulls_phase5.py` | Phase 5 nulls: B1 fold-safe regression null (50 shuffles), B2 A-core CONTROL vs FLY label null (50 shuffles) | `results/phase5_nulls/phase5_nulls_results.json` | same |
| (exploratory, dates not preserved) | aim1 scripts | Width sweep (5 draws/width), 1,000-shuffle unstratified null, 16-group permutation importance | collated in `results/all_tfm_results_master.csv` | tabpfn 8.2.0, seed 42, n_est 32 |

**Documentation gaps, flagged not filled:** the exploratory aim1 run dates are not preserved in the workspace. Battery result JSONs record the tabpfn version field as null; the version anchors are the 2026-08-16 install record (8.2.0) and the exploratory-run metadata in the master table (8.2.0). A later reproduction session ran 8.3.0 and matched the headline result.

**Machine change record (2026-08-16):** prior GPU machine lost; machine-local TabPFN weight cache did not survive. Reinstalled pinned env, re-downloaded checkpoints via the authenticated/licensed path, verified regressor SHA256 prefix `311ce18d97e9533d` (match), recorded classifier prefix `d0d865d54dfbc524` (first use). Documented swap, not a silent restart.
