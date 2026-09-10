#!/usr/bin/env python3
"""
controls_planted_signal.py
Concept: positive and negative controls are the basic trust checks for
a pipeline. A POSITIVE control is a case where the answer is known in
advance to be YES - the pipeline MUST find it, or the pipeline is
broken. A NEGATIVE control is a case where the answer is known in
advance to be NO - the pipeline must NOT find anything, or it cries
wolf.

Toy setup: 40 subjects, 100 noise measurements. We run the same model
and the same leave-one-out scoring three times:
  1. positive control: outcome is BUILT from measurement 0 (signal
     exists by construction) - the pipeline must score well above its
     shuffle null,
  2. negative control: outcome is fresh noise, unrelated to everything
     - the pipeline must score inside its shuffle null,
  3. each case is compared against 200 label shuffles of itself.

Run: python3 controls_planted_signal.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n, p = 40, 100
X = rng.normal(size=(n, p))

def loocv(X, y):
    lam = 10.0
    pred = np.empty(n)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        Xtr, ytr = X[tr], y[tr]
        Xc = Xtr - Xtr.mean(0); yc = ytr - ytr.mean()
        w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T @ yc)
        pred[i] = (X[i] - Xtr.mean(0)) @ w + ytr.mean()
    return pred

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

def run_case(y, label):
    obs = spearman(y, loocv(X, y))
    nulls = np.array([spearman(rng.permutation(y), loocv(X, rng.permutation(y)))
                      for _ in range(200)])
    z = (obs - nulls.mean()) / nulls.std(ddof=1)
    print(f"{label}: observed {obs:+.3f} | null {nulls.mean():+.3f} "
          f"+/- {nulls.std(ddof=1):.3f} | distance z = {z:+.2f}")
    return z

y_pos = X[:, 0] * 2.0 + rng.normal(scale=0.5, size=n)   # built from column 0
y_neg = rng.normal(size=n)                              # unrelated noise

z_pos = run_case(y_pos, "POSITIVE control (signal planted)")
z_neg = run_case(y_neg, "NEGATIVE control (nothing planted)")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print(f"  The positive control must light up (it did: z = {z_pos:+.1f}, far")
print("  above its null) - if it had not, every other result from this")
print("  pipeline would be untrustworthy, because a pipeline that misses")
print("  a known signal will miss real ones. The negative control must")
print(f"  stay quiet (it did: z = {z_neg:+.1f}, inside its null) - if it")
print("  had lit up, the pipeline would be manufacturing findings from")
print("  noise. Running both, on the same code path as the real")
print("  analysis, is how the project checks that its machinery can")
print("  both detect truth and reject noise before any real result is")
print("  believed.")
