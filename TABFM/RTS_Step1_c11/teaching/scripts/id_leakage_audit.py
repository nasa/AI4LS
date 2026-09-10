#!/usr/bin/env python3
"""
id_leakage_audit.py
Concept: leakage means the model gets information it should not have -
for example, if the data rows are ORDERED by the outcome, a model that
can see row position can 'predict' the outcome without learning any
biology. A leakage audit checks whether identifiers or row order carry
outcome information. The fix is trivial (shuffle the rows); the danger
is not checking.

Toy setup: 60 subjects, 30 noise measurements. The outcome is noise.
In version A the rows are SORTED by the outcome (worst case: row index
leaks the answer). In version B the same rows are shuffled. We give a
model ONLY the row index as input and see whether it can 'predict' the
outcome.

Run: python3 id_leakage_audit.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n = 60
y = rng.normal(size=n)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

# Version A: rows sorted by outcome -> row index is a perfect predictor
order_sorted = np.argsort(y)
y_sorted = y[order_sorted]
row_index = np.arange(n)
leak_score = spearman(y_sorted, row_index)

# Version B: same values, shuffled rows -> row index knows nothing
perm = rng.permutation(n)
y_shuffled = y[perm]
clean_score = spearman(y_shuffled, row_index)

print(f"Model input: ROW INDEX ONLY (no measurements at all).")
print(f"Sorted rows:   rank correlation of row index with outcome = {leak_score:+.3f}")
print(f"Shuffled rows: rank correlation of row index with outcome = {clean_score:+.3f}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  In the sorted file, a model handed nothing but the row number")
print("  achieves a perfect +1.000 - it 'predicts' every subject's")
print("  outcome without seeing a single measurement. That is leakage:")
print("  the answer was hidden in the filing order. After shuffling the")
print("  same rows, the identical model scores about zero. The lesson")
print("  cuts both ways: (1) before trusting any result, check that")
print("  identifiers, row order, and file structure carry no outcome")
print("  information; (2) when an audit finds a correlation between an")
print("  identifier and the outcome, the correct response is to trace")
print("  the file's construction history, not to believe the model.")
print("  This is exactly the audit the project ran on subject IDs, arm")
print("  labels, and row order.")
