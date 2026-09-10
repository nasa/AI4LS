#!/usr/bin/env python3
"""
counted_pvalue_convergence.py
Concept: a permutation p-value is a COUNT, not a formula output. You
shuffle the labels k times, recompute the score each time, and count how
many shuffles beat the real score: p = (wins + 1) / (k + 1). Two facts
follow: (1) the estimate gets steadier as k grows, and (2) there is a
hard floor - with k shuffles you can never report below 1/(k+1), no
matter how strong the real result is.

Toy setup: a planted true score of +0.40. Shuffled scores are drawn from
a bell curve centered at -0.39 with spread 0.28 (matching the shape of
this project's measured nulls). We watch the counted p-value at
k = 20, 50, 100, 200, 500, 1000, 5000.

Run: python3 counted_pvalue_convergence.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

true_score = 0.40
null_center, null_spread = -0.39, 0.28

print(f"Real score: {true_score:+.2f}.  Shuffled scores center at {null_center:+.2f} "
      f"(this project's nulls really do sit near -0.39, not 0).\n")
print(f"{'shuffles k':>12}{'floor 1/(k+1)':>16}{'counted p':>14}")
for k in [20, 50, 100, 200, 500, 1000, 5000]:
    nulls = rng.normal(null_center, null_spread, size=k)
    wins = np.sum(nulls >= true_score)
    p = (wins + 1) / (k + 1)
    print(f"{k:>12}{1/(k+1):>16.5f}{p:>14.5f}"
          + ("   <- at the floor" if wins == 0 else ""))

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  Two lessons. First, the counted p-value wobbles when k is small")
print("  and settles as k grows - it is an estimate based on a count, not")
print("  an exact number. Second, the floor: with 20 shuffles the best")
print("  you can ever say is p = 1/21 = 0.048, even if ZERO shuffles beat")
print("  the real score. With 500 shuffles the floor is 1/501 = 0.002.")
print("  So when a project result reports p = 0.008 from 500 shuffles,")
print("  that is 4 wins out of 501 - a real count - and no result from")
print("  that run could ever have come out below 0.002. The '+1' in the")
print("  formula exists so that p is never reported as exactly zero,")
print("  which would falsely claim the result is impossible by chance.")
