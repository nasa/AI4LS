#!/usr/bin/env python3
"""
multiplicity_chance_hits.py
Concept: when you run many tests at once, some will look significant by
pure luck. With a 1-in-20 threshold (p < 0.05), running 35 tests on
pure noise yields about 1-2 "hits" on average - with zero real signal
anywhere. This is the multiplicity problem, and it is why the project
applies corrections (like Bonferroni: divide the threshold by the
number of tests) before believing any single hit in a batch.

Toy setup: 35 independent fake experiments, each with 30 subjects and
pure-noise data (no real signal anywhere). Each experiment gets a
counted p-value from 200 label shuffles - the same machinery the
project uses. We count how many of the 35 come out below 0.05, and how
many survive the Bonferroni threshold 0.05/35.

Run: python3 multiplicity_chance_hits.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n_tests, n_subj, p_feat, k = 35, 30, 20, 200

def one_experiment(rng):
    X = rng.normal(size=(n_subj, p_feat))
    y = rng.normal(size=n_subj)          # pure noise: nothing to find
    def loocv_score(yv):
        pred = np.empty(n_subj)
        for i in range(n_subj):
            tr = np.setdiff1d(np.arange(n_subj), [i])
            mu = yv[tr].mean()
            # simple ridge
            Xc = X[tr] - X[tr].mean(0)
            w = np.linalg.solve(Xc.T @ Xc + 10*np.eye(p_feat), Xc.T @ (yv[tr]-mu))
            pred[i] = (X[i]-X[tr].mean(0)) @ w + mu
        ra = np.argsort(np.argsort(yv)); rb = np.argsort(np.argsort(pred))
        return np.corrcoef(ra, rb)[0, 1]
    obs = loocv_score(y)
    wins = sum(loocv_score(rng.permutation(y)) >= obs for _ in range(k))
    return (wins + 1) / (k + 1)

pvals = np.array([one_experiment(rng) for _ in range(n_tests)])
hits_raw = np.sum(pvals < 0.05)
bonf = 0.05 / n_tests
hits_bonf = np.sum(pvals < bonf)

print(f"{n_tests} experiments, all pure noise, each with a counted p-value "
      f"from {k} shuffles.")
print(f"Smallest few p-values: {np.sort(pvals)[:5].round(4)}")
print(f"Tests below 0.05 (uncorrected): {hits_raw} of {n_tests}")
print(f"Bonferroni threshold: 0.05/{n_tests} = {bonf:.5f}")
print(f"Tests below the Bonferroni threshold: {hits_bonf} of {n_tests}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  Every one of these 35 experiments contains zero real signal -")
print("  the data are noise by construction. Yet some experiments still")
print("  produce small p-values, because a 1-in-20 threshold invites")
print("  about one lucky hit per twenty tests. The Bonferroni threshold")
print("  (0.05 divided by the number of tests) exists precisely to")
print("  absorb this luck. So when the project ran 35 related tests and")
print("  one cell came out at p = 0.03, the correct reaction is not")
print("  'a discovery' but 'this is what chance looks like in a batch")
print("  this size' - and the recorded result (that cell does not")
print("  survive the corrected threshold) reflects exactly that.")
