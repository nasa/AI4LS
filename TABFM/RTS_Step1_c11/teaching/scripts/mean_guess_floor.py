#!/usr/bin/env python3
"""
mean_guess_floor.py
Concept: R-squared has a built-in reference point that people forget:
guessing the average outcome for every single subject scores R-squared
= 0 BY CONSTRUCTION. A model must beat that trivial strategy to score
above zero, and a negative R-squared means the model did WORSE than
simply naming the average for everyone.

Toy setup: 38 made-up outcomes. We score three "predictors":
  1. the mean-guess: predict the average for everyone (the floor),
  2. a noisy useless predictor: random numbers,
  3. a decent predictor: truth plus moderate noise.

Run: python3 mean_guess_floor.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n = 38
y = rng.normal(size=n)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot

mean_guess = np.full(n, y.mean())
random_pred = rng.normal(size=n)                      # knows nothing
decent_pred = y + rng.normal(scale=0.7, size=n)       # tracks truth loosely

print(f"Mean-guess (predict the average for all 38): R-squared = {r_squared(y, mean_guess):+.4f}")
print(f"Random numbers (no information at all):      R-squared = {r_squared(y, random_pred):+.4f}")
print(f"Decent predictor (truth + moderate noise):   R-squared = {r_squared(y, decent_pred):+.4f}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  The mean-guess uses no information about any subject, yet it")
print("  scores exactly 0 - that is where zero comes from. The random")
print("  predictor, which also knows nothing, scores NEGATIVE, because")
print("  its wrong guesses are worse than just naming the average.")
print("  So R-squared = 0 does not mean 'the model learned nothing")
print("  unusual happened'; it means 'the model exactly tied the")
print("  dumbest possible strategy'. And a negative R-squared - which")
print("  appears in several project results - means the model would")
print("  have been beaten by a colleague who ignored the data and")
print("  guessed the average for everyone.")
