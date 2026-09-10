#!/usr/bin/env python3
"""
block_deletion_importance.py
Concept: when measurements are correlated (move together), testing
importance one column at a time can miss a whole family of important
measurements - because the remaining copies cover for the one you
removed. Deleting an entire BLOCK (a whole family of related
measurements) at once reveals what single-column tests hide.

Toy setup: 50 subjects, 30 measurements in 3 families of 10. Within a
family, all 10 measurements are near-duplicates (correlated). Only
family 1 truly drives the outcome. We compare:
  - single-column permutation: shuffle ONE measurement, see how much
    the score drops (each family member has 9 backups, so drops are tiny)
  - block deletion: remove a WHOLE family, refit, and see the drop.

Run: python3 block_deletion_importance.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n, p, fam_size = 50, 30, 10
latent = rng.normal(size=(n, 3))
X = np.empty((n, p))
for f in range(3):
    X[:, f*fam_size:(f+1)*fam_size] = (latent[:, [f]]
                                       + 0.1 * rng.normal(size=(n, fam_size)))
y = latent[:, 0] + 0.3 * rng.normal(size=n)   # ONLY family 1 matters

def fit_predict(Xtr, ytr, Xte):
    lam = 1.0
    Xc = Xtr - Xtr.mean(0); yc = ytr - ytr.mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xtr.shape[1]), Xc.T @ yc)
    return (Xte - Xtr.mean(0)) @ w + ytr.mean()

def cv_r2(X, y):
    idx = np.random.default_rng(3).permutation(n)
    pred = np.empty(n)
    for f in np.array_split(idx, 5):
        tr = np.setdiff1d(idx, f)
        pred[f] = fit_predict(X[tr], y[tr], X[f])
    ss_res = np.sum((y - pred) ** 2)
    return 1 - ss_res / np.sum((y - y.mean()) ** 2)

base = cv_r2(X, y)
print(f"Full-data cross-validated R-squared: {base:+.3f}  "
      f"(only family 1 truly matters)\n")

print("Single-column permutation (shuffle one measurement, re-score):")
for j in [0, 5, 15, 25]:
    Xp = X.copy()
    Xp[:, j] = rng.permutation(Xp[:, j])
    drop = base - cv_r2(Xp, y)
    fam = j // fam_size + 1
    print(f"  column {j:>2} (family {fam}): score drop = {drop:+.3f}")

print("\nBlock deletion (remove a whole family of 10, refit, re-score):")
for f in range(3):
    cols = np.setdiff1d(np.arange(p), np.arange(f*fam_size, (f+1)*fam_size))
    drop = base - cv_r2(X[:, cols], y)
    print(f"  remove family {f+1}: score drop = {drop:+.3f}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  Shuffling any single measurement barely dents the score, even")
print("  inside the one family that truly matters, because its nine")
print("  near-duplicates cover for it. Removing the whole family at once")
print("  exposes the truth: the family-1 block causes a large drop and")
print("  the other two families cause almost none. Single-column tests")
print("  answer 'does this exact column add anything beyond its")
print("  copies?'; block tests answer 'does this FAMILY of measurements")
print("  matter?'. When measurements move together - as biological")
print("  measurements usually do - only the block test matches the")
print("  question the team actually asked.")
