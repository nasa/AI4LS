"""
cross_validation.py  ->  ml_service/src/cross_validation.py

K-fold cross-validation and optional GridSearchCV tuning, run on the TRAINING split only.
The held-out test set is never passed in here; it is scored once, afterwards, by
ModelTrainer.train_model.

Every candidate is the full Pipeline([log?, scale?, model]) from build_pipeline, so the
transforms are refit inside each fold (no fold leakage).

Two modes:
  tune=False  cross_validate() the pipeline as configured -> unbiased CV estimate
              of the training procedure.
  tune=True   GridSearchCV over a small per-algorithm grid; the best candidate is chosen
              on mean CV score of `scoring`. Its CV metrics are slightly optimistic
              (they were used for selection); the held-out test score is the honest one.

Parallelism: GridSearchCV / cross_validate use ML_CV_N_JOBS (env, default -1). When it is
not 1, the estimator's own n_jobs (xgboost, random forest) is set to 1 inside CV to avoid
oversubscribing cores; the final model keeps its original setting.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate

from src.preprocessing import MODEL_STEP, final_estimator

logger = logging.getLogger(__name__)

Grid = Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]]

# Small grids on purpose: few samples, thousands of genes. Parameter names are the
# estimator's own (no "model__" prefix); they are prefixed automatically.
DEFAULT_PARAM_GRIDS: Dict[str, Grid] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [5000],
    },
    "random_forest": {
        "n_estimators": [100, 300],
        "max_features": ["sqrt", 0.1],
        "min_samples_leaf": [1, 3],
    },
    "xgboost": {
        "n_estimators": [100, 300],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.05, 0.1],
    },
    "svm": [
        {"kernel": ["linear"], "C": [0.01, 0.1, 1.0]},
        {"kernel": ["rbf"], "C": [0.1, 1.0, 10.0], "gamma": ["scale"]},
    ],
    "neural_network": {
        "hidden_layer_sizes": [[50], [100]],
        "alpha": [1e-4, 1e-3, 1e-2],
    },
    "gradient_boosting": {
        "n_estimators": [100, 300],
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
    },
}

DEFAULT_SCORING = "f1_weighted"
NON_FEATURE_COLUMNS = ["source_dataset"]


def numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Same column handling as ModelTrainer.train_model, so CV sees the same features."""
    X = X.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in X.columns])
    return X.select_dtypes(include=[np.number])


def _prefix_grid(grid: Grid) -> List[Dict[str, List[Any]]]:
    """Add the 'model__' prefix, wrap scalars in lists, turn hidden_layer_sizes lists into tuples."""
    out = []
    for g in grid if isinstance(grid, list) else [grid]:
        pg = {}
        for key, vals in g.items():
            vals = vals if isinstance(vals, list) else [vals]
            name = key.split("__", 1)[1] if key.startswith(f"{MODEL_STEP}__") else key
            if name == "hidden_layer_sizes":
                vals = [tuple(v) if isinstance(v, (list, tuple)) else (v,) for v in vals]
            pg[f"{MODEL_STEP}__{name}"] = vals
        out.append(pg)
    return out


def resolve_param_grid(algorithm: str, param_grids_json: str = "") -> Optional[Grid]:
    """Per-algorithm override from param_grids_json if present, else the default grid."""
    if param_grids_json:
        overrides = json.loads(param_grids_json)
        if not isinstance(overrides, dict):
            raise ValueError("param_grids_json must be a JSON object keyed by algorithm")
        if algorithm in overrides:
            return overrides[algorithm]
    return DEFAULT_PARAM_GRIDS.get(algorithm)


def _n_jobs() -> int:
    try:
        return int(os.environ.get("ML_CV_N_JOBS", "-1"))
    except ValueError:
        return -1


def _scorers(y: pd.Series, primary: str) -> Dict[str, str]:
    scorers = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
    if pd.Series(y).nunique() == 2:
        scorers["roc_auc"] = "roc_auc"
    if primary not in scorers:
        scorers[primary] = primary
    return scorers


def _strip_prefix(params: Dict[str, Any]) -> Dict[str, str]:
    return {k.split("__", 1)[1] if "__" in k else k: str(v) for k, v in params.items()}


def cross_validate_and_tune(
    pipeline,
    algorithm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    tune: bool = False,
    cv_folds: int = 5,
    cv_repeats: int = 1,
    random_state: int = 0,
    scoring: str = DEFAULT_SCORING,
    param_grids_json: str = "",
) -> Dict[str, Any]:
    """
    Returns:
      estimator    UNFITTED pipeline to train on all of X_train (best params applied if tuned)
      cv_metrics   {"<metric>_mean", "<metric>_std", "n_splits", "n_repeats", ...}; {} if CV skipped
      best_params  {param: str(value)} chosen by the grid search; {} if not tuned
      tuned        True if a grid search ran and its best params were applied
    """
    result: Dict[str, Any] = {"estimator": pipeline, "cv_metrics": {}, "best_params": {}, "tuned": False}
    X = numeric_features(X_train)

    # Stratified folds need at least n_splits samples of every class in the training set.
    min_class = int(pd.Series(y_train).value_counts().min())
    n_splits = min(max(int(cv_folds), 2), min_class)
    if n_splits < 2:
        logger.warning(f"[{algorithm}] smallest class has {min_class} training sample(s); "
                       "skipping cross-validation and tuning")
        return result
    if n_splits < cv_folds:
        logger.warning(f"[{algorithm}] cv_folds reduced from {cv_folds} to {n_splits} "
                       f"(smallest class has {min_class} training samples)")
    n_repeats = max(int(cv_repeats), 1)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scorers = _scorers(y_train, scoring)
    n_jobs = _n_jobs()

    base = clone(pipeline)
    if n_jobs != 1 and "n_jobs" in final_estimator(base).get_params():
        base.set_params(**{f"{MODEL_STEP}__n_jobs": 1})

    grid = resolve_param_grid(algorithm, param_grids_json) if tune else None
    if tune and not grid:
        logger.warning(f"[{algorithm}] no parameter grid defined; running CV without tuning")

    metrics: Dict[str, float] = {}
    if grid:
        search = GridSearchCV(base, _prefix_grid(grid), scoring=scorers, refit=False,
                              cv=cv, n_jobs=n_jobs, error_score=np.nan)
        search.fit(X, y_train)
        res = search.cv_results_
        primary = np.asarray(res[f"mean_test_{scoring}"], dtype=float)
        if np.all(np.isnan(primary)):
            raise RuntimeError(f"[{algorithm}] all {len(primary)} grid candidates failed during CV")
        best = int(np.nanargmax(primary))
        best_params = res["params"][best]
        for name in scorers:
            metrics[f"{name}_mean"] = float(res[f"mean_test_{name}"][best])
            metrics[f"{name}_std"] = float(res[f"std_test_{name}"][best])
        metrics["n_candidates"] = float(len(primary))
        result.update(
            estimator=clone(pipeline).set_params(**best_params),
            best_params=_strip_prefix(best_params),
            tuned=True,
        )
    else:
        res = cross_validate(base, X, y_train, cv=cv, scoring=scorers, n_jobs=n_jobs, error_score=np.nan)
        for name in scorers:
            scores = np.asarray(res[f"test_{name}"], dtype=float)
            metrics[f"{name}_mean"] = float(np.nanmean(scores)) if not np.all(np.isnan(scores)) else float("nan")
            metrics[f"{name}_std"] = float(np.nanstd(scores)) if not np.all(np.isnan(scores)) else float("nan")

    metrics["n_splits"] = float(n_splits)
    metrics["n_repeats"] = float(n_repeats)
    result["cv_metrics"] = metrics

    logger.info(f"[{algorithm}] {'tuned ' if result['tuned'] else ''}CV {n_splits}x{n_repeats}: "
                f"{scoring}={metrics.get(f'{scoring}_mean', float('nan')):.3f}"
                f"±{metrics.get(f'{scoring}_std', float('nan')):.3f}"
                + (f", best={result['best_params']}" if result["tuned"] else ""))
    return result
