#!/usr/bin/env python3
"""
stratified_vs_global_permutation.py
Concept: when subjects come from groups with different average outcomes
(here: three bed-rest arms), HOW you shuffle matters. Shuffling labels
across the whole cohort (global permutation) breaks the group structure,
so it tests a different question than the one asked. Shuffling within
each group separately (stratified / arm-aware permutation) keeps the
group structure intact and tests only the within-group skill.

Toy setup: 36 subjects in three groups of 12. Group A averages -1,
group B averages 0, group C averages +1 on the outcome. One measurement
flags the group, so a model can rank subjects ACROSS groups - but the
outcome within any group is pure noise, so the model has ZERO
within-group skill by construction. We score the POOLED rank
correlation (all 36 subjects together) and judge it against two nulls:
200 global shuffles and 200 within-group shuffles.

Run: python3 stratified_vs_global_permutation.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n_per, p = 12, 20
groups = np.repeat([0, 1, 2], n_per)
n = len(groups)
group_mean = np.array([-1.0, 0.0, 1.0])

X = rng.normal(size=(n, p))
X[:, 0] += group_mean[groups] * 3.0          # measurement 0 flags the group
y = group_mean[groups] + rng.normal(scale=0.5, size=n)  # outcome = group + noise

def fit_predict(Xtr, ytr, Xte):
    lam = 20.0
    Xc = Xtr - Xtr.mean(0); yc = ytr - ytr.mean()
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(p), Xc.T @ yc)
    return (Xte - Xtr.mean(0)) @ w + ytr.mean()

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

def loocv(X, y):
    pred = np.empty(n)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        pred[i] = fit_predict(X[tr], y[tr], X[[i]])[0]
    return pred

pred = loocv(X, y)
obs = spearman(y, pred)   # pooled score: includes cross-group ranking

k = 200
null_global = np.empty(k)
null_strat = np.empty(k)
for s in range(k):
    # global: shuffle all 36 labels together - breaks group structure
    null_global[s] = spearman(rng.permutation(y), pred)
    # stratified: shuffle labels only inside each group - keeps it
    y_strat = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y_strat[idx] = rng.permutation(y[idx])
    null_strat[s] = spearman(y_strat, pred)

z_global = (obs - null_global.mean()) / null_global.std(ddof=1)
z_strat = (obs - null_strat.mean()) / null_strat.std(ddof=1)

print(f"Truth by construction: model ranks ACROSS groups perfectly,")
print(f"but has ZERO skill ranking people within the same group.\n")
print(f"Pooled rank correlation (real data): {obs:+.3f}")
print(f"Global-shuffle null:     center {null_global.mean():+.3f} +/- {null_global.std(ddof=1):.3f}  -> z = {z_global:+.2f}")
print(f"Stratified-shuffle null: center {null_strat.mean():+.3f} +/- {null_strat.std(ddof=1):.3f}  -> z = {z_strat:+.2f}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  Same data, same model, same real score - but the two nulls")
print("  tell opposite stories. The global shuffle breaks the group")
print("  pattern, so the real score looks spectacularly significant")
print("  against it. The stratified shuffle keeps each group's labels")
print("  inside that group, so the model's cross-group ranking skill is")
print("  present in EVERY shuffle - and the real score lands right on")
print("  top of that null, correctly reporting 'no within-group skill'.")
print("  When a cohort has built-in groups with different averages -")
print("  like the three bed-rest arms - a global shuffle can credit the")
print("  model for simply knowing which arm a subject was in. The")
print("  arm-aware (stratified) null is the honest test of whether the")
print("  model says anything about individuals beyond their arm.")
