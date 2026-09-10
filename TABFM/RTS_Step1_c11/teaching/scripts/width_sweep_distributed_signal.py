#!/usr/bin/env python3
"""
width_sweep_distributed_signal.py
Concept: when real signal is spread thinly across MANY measurements
(each one carrying a tiny piece), a model that sees only a few
measurements can look useless while the same model seeing all of them
works. "Feature width" (how many measurements the model is allowed to
use) can therefore change the conclusion.

Toy setup: 80 subjects, 200 measurements. The outcome depends on 100 of
the measurements, each contributing an equally tiny amount - no single
measurement is informative on its own. We hand the model the first
5 / 25 / 50 / 100 / 200 measurements (in a fixed arbitrary order, so
small widths contain only a few of the 100 true contributors) and score
rank correlation with 5-fold cross-validation at each width.

Run: python3 width_sweep_distributed_signal.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n, p, n_signal = 80, 200, 100
X = rng.normal(size=(n, p))
beta = np.zeros(p)
beta[:n_signal] = np.sqrt(4.0 / n_signal)   # 100 equally weak contributors
y = X @ beta + rng.normal(size=n)

feat_order = np.random.default_rng(11).permutation(p)  # fixed arbitrary order

def cv_score(Xsub, y, lam=10.0, folds=5):
    idx = np.random.default_rng(7).permutation(len(y))
    pred = np.empty(len(y))
    for f in np.array_split(idx, folds):
        tr = np.setdiff1d(idx, f)
        Xtr, Xte = Xsub[tr], Xsub[f]
        Xc = Xtr - Xtr.mean(0); yc = y[tr] - y[tr].mean()
        w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xsub.shape[1]), Xc.T @ yc)
        pred[f] = (Xte - Xtr.mean(0)) @ w + y[tr].mean()
    ra = np.argsort(np.argsort(y)); rb = np.argsort(np.argsort(pred))
    return np.corrcoef(ra, rb)[0, 1]

print(f"True setup: outcome depends on {n_signal} weak measurements, "
      f"none informative alone.\n")
print(f"{'width (measurements kept)':>26}{'rank correlation':>20}")
for w in [5, 25, 50, 100, 200]:
    rho = cv_score(X[:, feat_order[:w]], y)
    print(f"{w:>26}{rho:>+20.3f}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  The signal never changed - only the number of measurements the")
print("  model was allowed to see. With a handful of measurements the")
print("  model finds nothing (even scoring below zero, which cross-")
print("  validation noise does), because each measurement alone is")
print("  nearly worthless. As the width grows, the tiny contributions")
print("  add up and the score climbs. A 'the model failed' conclusion")
print("  at one width can reverse at another width on the same data.")
print("  This is why the project ran its width sweep (5, 25, 100, 250,")
print("  628 measurements) before trusting any single-width result -")
print("  and why the full-width score of +0.376, not the near-zero")
print("  narrow-width scores, is the headline.")
