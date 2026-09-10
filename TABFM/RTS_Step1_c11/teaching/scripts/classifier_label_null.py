#!/usr/bin/env python3
"""
classifier_label_null.py
Concept: for a classifier (a model that sorts subjects into named
categories instead of predicting a number), the chance level for
accuracy is NOT 1/number-of-classes and NOT exactly the biggest class's
share - under cross-validation it is its own measurable quantity, and
it must be measured by shuffling labels, not assumed.

Toy setup: 38 subjects in three uneven classes (11 / 19 / 8, matching
this project's arm sizes). Measurements are pure noise - no real
signal. A nearest-centroid classifier (assign each subject to the class
whose average measurement profile is closest) is scored by leave-one-out.
Then labels are shuffled 200 times to build the null distribution of
accuracy.

Run: python3 classifier_label_null.py   (CPU only, seconds)
"""
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n, p = 38, 50
y = np.array([0]*11 + [1]*19 + [2]*8)   # class sizes 11, 19, 8
X = rng.normal(size=(n, p))             # pure noise: no class signal

def loocv_predict(X, y):
    pred = np.empty(n, dtype=int)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        centroids = np.array([X[tr][y[tr] == c].mean(0) for c in np.unique(y)])
        d = np.sum((centroids - X[i]) ** 2, axis=1)
        pred[i] = np.unique(y)[np.argmin(d)]
    return pred

def balanced_accuracy(y_true, y_pred):
    # average of the per-class recall: each class counts equally,
    # so a model that ignores the small class is penalized
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls))

pred = loocv_predict(X, y)
acc = np.mean(pred == y)
bacc = balanced_accuracy(y, pred)
base_rate = np.max(np.bincount(y)) / n

k = 200
null_acc = np.empty(k)
for s in range(k):
    y_shuf = rng.permutation(y)
    null_acc[s] = np.mean(loocv_predict(X, y_shuf) == y_shuf)

print(f"Class sizes: 11 / 19 / 8.  Biggest class share (base rate): {base_rate:.3f}")
print(f"Naive guess for chance accuracy (1/3): {1/3:.3f}")
print(f"Measured null accuracy over {k} shuffles: "
      f"{null_acc.mean():.3f} +/- {null_acc.std(ddof=1):.3f}")
print(f"This noise run's own accuracy: {acc:.3f}   balanced accuracy: {bacc:.3f}")

print()
print("PLAIN-LANGUAGE TAKEAWAY:")
print("  Three different numbers all claim to be 'chance' here: 0.333")
print("  (one over the number of classes), 0.500 (always naming the")
print("  biggest class), and the measured shuffle null near "
      f"{null_acc.mean():.2f}.")
print("  Only the third is honest for this exact model and this exact")
print("  cross-validation scheme, because the machinery itself (uneven")
print("  classes, leave-one-out, this classifier) shifts where chance")
print("  lands. The project therefore never assumes a chance level for")
print("  classification - it measures one by shuffling labels, exactly")
print("  as this script does. Balanced accuracy is reported alongside")
print("  plain accuracy so that a model cannot look good by ignoring")
print("  the smallest class.")
