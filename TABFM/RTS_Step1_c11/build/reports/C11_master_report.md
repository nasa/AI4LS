# C11 Master: Coverage and Script Evolution

**Master:** 43 analytical subjects, 7,545 measurement columns, all 18 C11-bearing
folders. Built by `build_master_v6_1.py`, wide matrix is measurements only.

**Coverage:** all 46 roster subjects located. 8 PURE + 10 MIXED = 18 folders.
Per-folder C11 subject counts below are verified against the completeness
reconciliation (0 gaps across all 18 folders).

---

## PURE C11 (roster subjects only, no other campaign)

| Folder | Experiment | C11 subj (of 46) |
|---|---|---|
| BEDREST_iRATS | Integrated resistance + aerobic training | 39 |
| BRSMVJ | Vertical jump | 38 |
| BEDREST_FTT | Functional task test | 36 |
| NNX10AP86G | Testosterone supplementation | 25 |
| NNX12AB40G | AD ASTRA (behavioral) | 19 |
| NCC958SA02802 | Neuro-mapping (MRI) | 18 |
| BRSMQCT | Quantitative CT (bone) | 5 |
| BRSMiDXA | DXA (bone) | 5 |

## MIXED (C11 rows interleaved with other campaigns; extract by roster filter)

| Folder | Experiment | C11 subj | Also contains |
|---|---|---|---|
| MR079G | Isokinetic strength | 42 | C1, C3 |
| BRSMCF | Cardiovascular function | 38 | C1, C3 |
| MR080G | Aerobic capacity | 38 | C1, C3 |
| BRSMPV | Plasma volume | 37 | C1, C3 |
| BRSMLVR | Latent virus reactivation | 30 | C3 |
| BRSMImmune | General immunity | 29 | C3 |
| BRSMVSI | Viral-specific immunity | 29 | C3 |
| CRF | Case report forms (vitals, meals, meds) | 26 | C1, C3 |
| MR016G | Clinical/mineral | 14 | C1, C3 |
| BRSMVF | Visual function | 1 | C1, C3 |

The MIXED folders are why the roster filter matters: C11 rows sit alongside C1
and C3 rows in the same files, so the master is assembled campaign-agnostic and
`--c11` selects the C11 subjects out.

---

## Script evolution (from James Casaletto's `build_master.py`)

**James's baseline (`build_master.py`).** The core pipeline:
1. Recursively find every CSV in the tree.
2. Keep files that have a `Subject` column.
3. Keep only balanced-longitudinal files (every subject the same row count); drop
   cross-sectional files.
4. Find the minimal column combination that makes `Subject` + combo unique (the
   observation key).
5. Read robustly across encodings (utf-8, cp1252, latin-1).

The observation key was found but then discarded, and the master-assembly
function was a stub, so no table was produced yet.

**What was built on it (v2 to v6.1), to the point:**
1. Implemented the master assembly James stubbed: pivot each file to wide, then
   outer-join every file on `Subject`.
2. Named every column `measure :: source_file :: timepoint`, so the same measure
   at two timepoints stays two distinct columns instead of colliding.
3. Normalized the subject header (`SUBJECT`, `Subject ID`, BOM-prefixed) and
   validated subject values, rejecting group codes and artifacts that leak into
   the subject column.
4. Made the build inclusive: keep cross-sectional AND longitudinal files; record
   longitudinal shape as metadata. (v1 dropped every single-timepoint measure.)
5. Added the long-format substrate (complete traceable record) and a tidy-panel
   branch for files that store variable-name and value in separate columns.
6. Added a completeness reconciliation (expected vs captured subjects per folder)
   and a per-file decision manifest.
7. Added a density guard (hold out daily-dense files like intake logs) and
   unresolved-key logging for files no timepoint vocabulary resolves.
8. Added roster filtering (`--c11` and `--roster FILE`) with leading-zero
   normalization, ready for the pooled C3+C11 run.
9. **v6:** removed phantom timepoint columns. Timepoint fields beyond the minimal
   key were leaking into the matrix as if they were measurements; the fix
   excludes every timepoint field from the measures and folds them into the
   column name.
10. **v6.1:** wide matrix is measurements only. Dropped arm/GroupName, Date,
    Campaign, and the static descriptors Age/Sex/Gender/Height; kept Weight as a
    real measurement; excluded the known-scrambled BRSMVJ Post2 file. Result:
    11,453 columns down to 7,545, residual leakage 0.
