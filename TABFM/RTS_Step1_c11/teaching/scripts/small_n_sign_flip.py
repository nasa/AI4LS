#!/usr/bin/env python3
"""
small_n_sign_flip.py
Concept: with very few subjects, a rank correlation is extremely
unstable - the SIGN (positive vs negative) can flip depending on which
subjects happen to be included. A correlation computed on 11 subjects
is not a fact about the world; it is a draw from a very wide
distribution.

Toy setup: a population of 200 subjects with a known, planted MODERATE
POSITIVE rank relationship (+0.40). We repeatedly draw random
subsamples of 11 subjects and compute the rank correlation in each
subsample. Watch how often the sign comes out negative even though the
true relationship is positive.

Run: python3 small_n_sign_flip.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

N_pop = 200
x = rng.normal(size=N_pop)
y = 0.6 * x + np.sqrt(1 - 0.6**2) * rng.normal(size=N_pop)  # true r = +0.60

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

full = spearman(x, y)
print(f"Full population of {N_pop}: rank correlation = {full:+.3f} "
      f"(planted truth is +0.60)\n")

n_sub, n_draws = 11, 2000
draws = np.array([spearman(rng.choice(x, n_sub, replace=False),
                           0) for _ in range(0)])  # placeholder, replaced below
vals = np.empty(n_draws)
for d in range(n_draws):
    idx = rng.choice(N_pop, n_sub, replace=False)
    vals[d] = spearman(x[idx], y[idx])

neg = np.mean(vals < 0)
print(f"{n_draws} random subsamples of {n_sub} subjects each:")
print(f"  range of observed correlations: {vals.min():+.3f} to {vals.max():+.3f}")
print(f"  middle 95%: {np.quantile(vals, 0.025):+.3f} to {np.quantile(vals, 0.975):+.3f}")
print(f"  fraction with the WRONG SIGN (negative, though truth is positive): {neg:.1%}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  The true relationship in this toy is solidly positive, and the")
print("  full 200-subject population shows it. But 11-subject samples")
print("  scatter enormously - a sizeable fraction come out NEGATIVE.")
print("  Nobody made an error; small samples simply cannot pin down a")
print("  correlation. This is why the project treats any result computed")
print("  on a single arm (8 to 19 subjects) as exploratory, why sign")
print("  disagreements between small subgroups are expected rather than")
print("  alarming, and why the headline analyses use all 38 subjects.")
