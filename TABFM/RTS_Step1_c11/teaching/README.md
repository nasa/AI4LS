# teaching/ — Finished educational assets

**Status:** finished assets only. Drafts, worksheets, and the teaching manuscript stay out by design; nothing here is a working document.

This folder teaches the ideas behind the analysis to students and reviewers who want the concepts without running the battery. Everything here is derived from the verified results in `results/`; nothing here introduces new numbers.

| Subfolder | Contents |
|---|---|
| `figures/` | The S1–S5 figure sets (PNG + SVG): the controls, the anatomy of a null distribution, group importance and the width sweep, four-bar summaries, and decision aids |
| `pipeline_teaching/` | Five figures explaining the pipeline itself, each with the Python script that generated it |
| `scripts/` | Thirteen small self-contained teaching scripts — one idea each: floor p-values, off-center nulls under leave-one-out, leakage audits, width sweeps, sign flips at small n, and more |
| `tables/` | The teaching tables (CSV + rendered PNG/SVG) |

Suggested entry points: `figures/S2_null_histogram.png` for what a permutation null is, `scripts/offcenter_null_loocv.py` for why these nulls are not centered at zero, and `figures/S5_decision_tree.png` for how the bias battery's readings A/B/C were adjudicated.
