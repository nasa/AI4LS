#!/usr/bin/env python3
"""
cv_scheme_variance.py
Concept: the cross-validation scheme (the rule for splitting data into
training parts and testing parts) changes the answer you get, even on the
SAME data with the SAME model.

Toy setup: 38 made-up subjects, 50 made-up measurements per subject, and a
true relationship planted between the measurements and an outcome. We score
the same model three ways:
  1. a single random split (one 80/20 division),
  2. repeated 5-fold cross-validation (five divisions, each subject tested
     once per round, repeated 10 times with different divisions),
  3. leave-one-out cross-validation (38 rounds; each round tests exactly one
     subject and trains on the other 37).
The planted relationship is identical throughout; only the scoring rule
changes. Watch how much the single-split answer moves around.

Run: python3 cv_scheme_variance.py   (CPU only, a few seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n, p = 38, 50
X = rng.normal(size=(n, p))
beta = np.zeros(p)
beta[:5] = 1.0                      # only 5 of 50 measurements truly matter
y = X @ beta + rng.normal(scale=1.5, size=n)  # moderate noise on purpose

def fit_predict(Xtr, ytr, Xte):
    # ridge regression: a linear model (a weighted sum of measurements)
    # with a penalty that keeps weights small so it does not chase noise
    lam = 10.0
    Xtr_c = Xtr - Xtr.mean(0); ytr_c = ytr - ytr.mean()
    Xte_c = Xte - Xtr.mean(0)
    w = np.linalg.solve(Xtr_c.T @ Xtr_c + lam * np.eye(p), Xtr_c.T @ ytr_c)
    return Xte_c @ w + ytr.mean()

def spearman(a, b):
    # rank correlation: replace values by their ranks, then correlate.
    # +1 = perfect ordering, 0 = no ordering, -1 = perfectly reversed.
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

# 1. single random split, done 6 times to show the spread
print("Single random 80/20 split, repeated 6 times (only the split changes):")
singles = []
for s in range(6):
    r = np.random.default_rng(100 + s)
    idx = r.permutation(n)
    te, tr = idx[:8], idx[8:]
    pred = fit_predict(X[tr], y[tr], X[te])
    rho = spearman(y[te], pred)
    singles.append(rho)
    print(f"  split {s+1}: rank correlation = {rho:+.3f}")
print(f"  -> answers range from {min(singles):+.3f} to {max(singles):+.3f} on identical data\n")

# 2. repeated 5-fold cross-validation
print("Repeated 5-fold cross-validation (10 rounds of 5 folds):")
reps = []
for rep in range(10):
    r = np.random.default_rng(1000 + rep)
    idx = r.permutation(n)
    folds = np.array_split(idx, 5)
    pred = np.empty(n)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        pred[f] = fit_predict(X[tr], y[tr], X[f])
    reps.append(spearman(y, pred))
print(f"  10 round answers: mean {np.mean(reps):+.3f}, "
      f"spread {min(reps):+.3f} to {max(reps):+.3f}\n")

# 3. leave-one-out
pred = np.empty(n)
for i in range(n):
    tr = np.setdiff1d(np.arange(n), [i])
    pred[i] = fit_predict(X[tr], y[tr], X[[i]])
print(f"Leave-one-out cross-validation: rank correlation = {spearman(y, pred):+.3f}")
print(f"  -> one fixed answer; no random split involved, so no split-to-split spread\n")

print("PLAIN-LANGUAGE TAKEAWAY:")
print("  The data and the model never changed in this demo. Only the rule for")
print("  who-gets-tested-when changed, and the reported score moved with it.")
print("  A single split can land anywhere in a wide range by luck of the draw;")
print("  repeated folds and leave-one-out are far more stable. This is why the")
print("  project reports which scheme produced every number, and why numbers")
print("  from different schemes (e.g. 5-fold vs leave-one-out) are not")
print("  directly comparable.")
