"""
significance.py  ->  feature_importance_service/src/significance.py

Label-permutation test for permutation importance (in the spirit of Altmann et al. 2010, "PIMP").

Observed:  fit the model as trained; permutation importance of each gene on the held-out test
           samples (drop in the score when that gene's values are shuffled).
Null:      shuffle the class labels of all samples (train and test together, so class counts are
           kept), refit a clone of the same model (same hyperparameters, same log/scale pipeline)
           on the training samples, and compute permutation importance on the test samples with the
           shuffled labels. Repeat n_permutations times. No gene is related to the shuffled labels,
           so these importances show how large an importance gets by chance.
p-value:   pooled over genes and permutations, p_g = (1 + #{null >= observed_g}) / (1 + n_null).
           Pooling gives a fine p-value resolution (100 permutations x 500 genes = 50,000 null
           values), which per-gene nulls of 100 values each could not; it treats the genes' null
           importances as exchangeable, which is reasonable here because every gene is z-scored
           inside the model's pipeline.
padj:      Benjamini-Hochberg across genes. significant = padj < alpha and observed importance > 0.

Permutation importance is computed with one prediction call per chunk of genes (all shuffled
copies stacked), which is what makes 100 permutations affordable.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

logger = logging.getLogger(__name__)

SCORINGS = ("accuracy", "brier")
MAX_STACK_VALUES = 20_000_000          # rows x genes per prediction call (~160 MB of float64)


def _score(model, X: pd.DataFrame, y: np.ndarray, scoring: str, classes: np.ndarray) -> np.ndarray:
    """Score per block of len(y) rows (X is several shuffled copies stacked); higher = better."""
    n = len(y)
    k = len(X) // n
    if scoring == "accuracy":
        pred = np.asarray(model.predict(X)).reshape(k, n)
        return (pred == y[None, :]).mean(axis=1)
    proba = np.asarray(model.predict_proba(X))                                  # (k*n, n_classes)
    onehot = (y[:, None] == classes[None, :]).astype(float)                     # (n, n_classes)
    err = ((proba.reshape(k, n, -1) - onehot[None]) ** 2).sum(axis=2).mean(axis=1)
    return -err                                                                 # negative Brier score


def permutation_importances(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int, rng: np.random.Generator,
                            scoring: str = "accuracy") -> np.ndarray:
    """Mean drop in score over n_repeats shuffles, for every gene. Same idea as
    sklearn.inspection.permutation_importance, but one predict call per chunk of genes."""
    n, g = X.shape
    classes = np.asarray(getattr(model, "classes_", np.unique(y)))
    base = _score(model, X, y, scoring, classes)[0]
    values = X.to_numpy(dtype=float)
    per_chunk = max(1, MAX_STACK_VALUES // max(1, n * n_repeats * g))
    out = np.empty(g)
    for start in range(0, g, per_chunk):
        cols = range(start, min(start + per_chunk, g))
        blocks = []
        for j in cols:
            for _ in range(n_repeats):
                b = values.copy()
                b[:, j] = b[rng.permutation(n), j]
                blocks.append(b)
        stacked = pd.DataFrame(np.vstack(blocks), columns=X.columns)
        scores = _score(model, stacked, y, scoring, classes).reshape(len(cols), n_repeats)
        out[start:start + len(cols)] = base - scores.mean(axis=1)
    return out


def _bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def _single_thread(est):
    """Inside the null loop run estimators single-threaded (parallelism is across permutations)."""
    try:
        params = est.get_params()
        upd = {k: 1 for k in params if k == "n_jobs" or k.endswith("__n_jobs")}
        if upd:
            est.set_params(**upd)
    except Exception:
        pass
    return est


def _null_once(model, X_train, X_test, y_all, n_train, n_repeats, scoring, seed):
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(y_all)
    m = _single_thread(clone(model))
    m.fit(X_train, y_perm[:n_train])
    return permutation_importances(m, X_test, y_perm[n_train:], n_repeats, rng, scoring)


def permutation_test(model, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_test,
                     n_permutations: int = 100, n_repeats: int = 5, alpha: float = 0.05,
                     scoring: str = "accuracy", random_state: int = 42,
                     n_jobs: Optional[int] = None) -> Dict:
    """Returns {"table": DataFrame(feature_name, importance, pvalue, padj, significant, rank), "metadata": {...}}.
    `model` must already be fitted on (X_train, y_train)."""
    if scoring not in SCORINGS:
        raise ValueError(f"scoring must be one of {SCORINGS}")
    if scoring == "brier" and not hasattr(model, "predict_proba"):
        logger.warning(f"{type(model).__name__} has no predict_proba; using accuracy")
        scoring = "accuracy"
    t0 = time.time()
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    rng = np.random.default_rng(random_state)
    observed = permutation_importances(model, X_test, y_test, n_repeats, rng, scoring)

    n_jobs = n_jobs if n_jobs is not None else int(os.environ.get("FI_PERM_N_JOBS", "-1"))
    y_all = np.concatenate([y_train, y_test])
    seeds = np.random.SeedSequence(random_state).spawn(n_permutations)
    seeds = [int(s.generate_state(1)[0]) for s in seeds]
    def run_serial():
        return [_null_once(model, X_train, X_test, y_all, len(y_train), n_repeats, scoring, s) for s in seeds]
    if n_jobs == 1:
        null = run_serial()
    else:
        from joblib import Parallel, delayed
        try:
            # Send this module's code to the worker processes by value, so they don't have to import
            # "src.significance" themselves (a different "src" package earlier on sys.path broke that).
            try:
                import cloudpickle                              # newer joblib: a separate package
            except ImportError:
                from joblib.externals import cloudpickle        # older joblib: vendored
            cloudpickle.register_pickle_by_value(sys.modules[__name__])
        except Exception:
            pass
        try:
            null = Parallel(n_jobs=n_jobs)(delayed(_null_once)(model, X_train, X_test, y_all, len(y_train),
                                                               n_repeats, scoring, s) for s in seeds)
        except Exception as e:
            logger.warning(f"Parallel permutation test failed ({type(e).__name__}: {e}); running serially")
            null = run_serial()
    null = np.sort(np.concatenate(null))
    n_null = len(null)
    n_ge = n_null - np.searchsorted(null, observed, side="left")          # null values >= observed
    p = (1 + n_ge) / (1 + n_null)
    p = np.where(observed > 0, p, 1.0)
    padj = _bh(p)
    table = pd.DataFrame({"feature_name": list(X_test.columns), "importance": observed, "pvalue": p, "padj": padj})
    table["significant"] = (table["padj"] < alpha) & (table["importance"] > 0)
    table = table.sort_values(["importance", "pvalue"], ascending=[False, True]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    meta = {"n_permutations": n_permutations, "n_repeats": n_repeats, "alpha": alpha, "scoring": scoring,
            "n_significant": int(table["significant"].sum()), "n_test_samples": len(y_test),
            "null_q95": float(np.quantile(null, 0.95)), "null_q99": float(np.quantile(null, 0.99)),
            "seconds": round(time.time() - t0, 1)}
    logger.info(f"Permutation test ({n_permutations} label shuffles, {scoring}): "
                f"{meta['n_significant']} of {len(table)} genes significant at padj < {alpha} in {meta['seconds']} s")
    return {"table": table, "metadata": meta}
