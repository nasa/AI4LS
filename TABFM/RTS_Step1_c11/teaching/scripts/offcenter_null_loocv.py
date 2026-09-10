#!/usr/bin/env python3
"""
offcenter_null_loocv.py
Concept: under leave-one-out cross-validation with many more
measurements than subjects, the chance level for rank correlation is
NOT zero - it is clearly negative. This is the single most
counterintuitive fact in this project: a model scoring 0.0 is actually
ABOVE chance here.

Why it happens: with 628 measurements and only 38 subjects, the fitted
model leans heavily toward the average of the training subjects. When
subject i is held out, the training average excludes subject i, so the
prediction for subject i gets pulled slightly AWAY from subject i's own
value - high true values get slightly under-predicted and low true
values slightly over-predicted. That built-in anti-correlation pushes
the chance level below zero.

Toy setup: 38 made-up subjects, 628 made-up measurements, outcome is
PURE NOISE (no true relationship at all). We run leave-one-out with a
ridge model (a linear model with a penalty that keeps weights small),
score the rank correlation, then shuffle the outcome labels 500 times
and re-score each time to build the null distribution (the collection
of scores pure chance produces).

Run: python3 offcenter_null_loocv.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n, p = 38, 628
X = rng.normal(size=(n, p))
y = rng.normal(size=n)          # pure noise: no true relationship exists

ALPHA = 1000.0                  # ridge penalty strength

def loocv_predictions(X, y):
    pred = np.empty(len(y))
    for i in range(len(y)):
        tr = np.setdiff1d(np.arange(len(y)), [i])
        Xtr, ytr = X[tr], y[tr]
        mu_x, mu_y = Xtr.mean(0), ytr.mean()
        Xc = Xtr - mu_x
        w = np.linalg.solve(Xc.T @ Xc + ALPHA * np.eye(p), Xc.T @ (ytr - mu_y))
        pred[i] = (X[i] - mu_x) @ w + mu_y
    return pred

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

obs = spearman(y, loocv_predictions(X, y))

k = 500
nulls = np.empty(k)
for s in range(k):
    y_shuf = rng.permutation(y)
    nulls[s] = spearman(y_shuf, loocv_predictions(X, y_shuf))

mean_null, sd_null = nulls.mean(), nulls.std(ddof=1)
p95 = np.quantile(nulls, 0.95)
z = (obs - mean_null) / sd_null
wins = np.sum(nulls >= obs)
p_counted = (wins + 1) / (k + 1)

print(f"Setup: {n} subjects, {p} measurements, outcome is pure noise.")
print(f"Observed rank correlation on unshuffled noise: {obs:+.3f}")
print(f"Null distribution from {k} shuffles: center {mean_null:+.3f}, "
      f"spread {sd_null:.3f}, 95th percentile {p95:+.3f}")
print(f"Distance of observed from null center, in spread units (z): {z:+.2f}")
print(f"Counted p-value: {p_counted:.4f}  (floor with {k} shuffles is {1/(k+1):.4f})")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  Nothing real exists in this toy data - the outcome is noise by")
print("  construction. Yet the chance scores do not center on 0. They")
print(f"  center near {mean_null:+.2f}. The leave-one-out machinery itself")
print("  bends predictions slightly against each held-out subject, so")
print("  'no skill' shows up as a NEGATIVE number, not zero.")
print("  This is exactly why the project measures its own null")
print("  distribution (center near -0.39) instead of assuming chance is")
print("  zero: a project score of 0.0, which sounds like failure, is")
print("  actually well ABOVE the measured chance level. Judging these")
print("  results against a textbook zero would misread above-chance")
print("  models as useless and barely-chance models as harmful.")
