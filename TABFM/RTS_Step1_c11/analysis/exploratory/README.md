# analysis/exploratory/ — Pre-registration scripts

**Status:** historical record. These are the exploratory scripts that ran before the analysis plan was registered. They are kept for transparency and mined for the findings that motivated the registered battery. They are **not** the registered analysis.

## What lives here and why

| Script | What it did |
|---|---|
| `run_tfm_modeling.py` | The exploratory headline run: TabPFN-3, leave-one-person-out, hip bone density change. Produced the correlation (0.3762…) that the registered battery later reproduced exactly as its entrance gate. |
| `run_tfm_probes.py` | Early probe runs around the headline (width subsets, sanity checks). |
| `run_tfm_arm_aware.py` | First attempt at the treatment-group-aware tests. One of its null runs was paused partway and never completed (deviation D3 in the main README); the registered battery's phase 5 is the completed replacement. |
| `run_tabpfn_test1.py`, `run_tabpfn3_c11.py` | Earliest feasibility runs on the hip outcome. |
| `run_9endpoint_reconstructed.py`, `recompute_scott2017.py` | Reconstruction checks against published values from earlier bed-rest analyses. |
| `make_figures.py`, `make_tables.py` | Figure and table generation for the exploratory phase. |
| `aim1_test1_tabpfn_colab.ipynb` | The original Colab notebook for the first feasibility test. |

## How to read these

- **Trust hierarchy:** the registered battery (`analysis/run_battery.py`, `run_missingness_remediation.py`, `run_nulls_phase5.py`) is the authoritative record. Where an exploratory script and the battery disagree, the battery governs, and the difference is documented in the main README's "Known limitations, deviations, and corrections".
- **Not turnkey:** these scripts carry the paths and machine assumptions of the exploratory sessions (Colab T4, early A10G). They are archived as-written, token-scanned before release, and are not expected to run unmodified.
- **Why keep them at all:** the deviations they contain (the paused null run, the lost rich-grid script) are part of the scientific record. Deleting them would make the corrections section of the README unverifiable.
