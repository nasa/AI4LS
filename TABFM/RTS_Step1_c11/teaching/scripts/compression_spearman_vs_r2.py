#!/usr/bin/env python3
"""
compression_spearman_vs_r2.py
Concept: rank correlation (Spearman) and R-squared measure different
things. A predictor can keep a PERFECT rank correlation while its
R-squared collapses - even below zero - if its output values are
compressed or shifted relative to the true values.

Toy setup: 38 made-up true outcome values. Three made-up predictors all
rank the subjects in EXACTLY the correct order:
  A. perfect predictor: prediction = truth
  B. compressed predictor: prediction = mean + 0.225 * (truth - mean)
     (right order, but values squeezed to 22.5% of the true spread -
     this mirrors the project's observed prediction spread ratio)
  C. compressed AND shifted predictor: B minus a constant offset
     (right order, right relative spacing, but every value too low)
Watch Spearman stay at +1.00 for all three while R-squared falls apart.

Run: python3 compression_spearman_vs_r2.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n = 38
y = rng.normal(loc=0.0, scale=1.0, size=n)   # made-up true outcomes
y = y - y.mean()

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

def r_squared(y_true, y_pred):
    # R-squared: 1 minus (unexplained spread / total spread).
    # 1.0 = perfect; 0 = no better than guessing the mean every time;
    # negative = WORSE than guessing the mean.
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot

pred_A = y.copy()
pred_B = y.mean() + 0.225 * (y - y.mean())
pred_C = pred_B - 1.5

print(f"{'predictor':<38}{'Spearman (order)':>18}{'R-squared (values)':>20}")
for name, pred in [("A: exact", pred_A),
                   ("B: compressed to 22.5% spread", pred_B),
                   ("C: compressed + shifted low", pred_C)]:
    print(f"{name:<38}{spearman(y, pred):>+18.3f}{r_squared(y, pred):>20.3f}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  All three predictors put the 38 subjects in exactly the right")
print("  order, so all three score a perfect rank correlation of +1.000.")
print("  But R-squared asks a harder question: are the predicted VALUES")
print("  close to the true values? Predictor B squeezes every prediction")
print("  toward the average, and its R-squared drops to about 0.40.")
print("  Predictor C is additionally shifted too low, and its R-squared")
print("  goes NEGATIVE - worse than guessing the average for everyone -")
print("  while still ranking everyone perfectly.")
print("  So: a high rank correlation with a low or negative R-squared is")
print("  not a contradiction. It means the model knows WHO is above whom")
print("  but not BY HOW MUCH. In this project the observed predictions")
print("  vary only about 22.5% as much as the true values, which is why")
print("  the project judges models primarily on rank correlation.")
