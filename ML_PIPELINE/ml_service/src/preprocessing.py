"""
preprocessing.py  ->  ml_service/src/preprocessing.py

Wraps each estimator in an sklearn Pipeline so the transforms are fit on the
training split only (no train/test leakage) and always run in a safe order:
log first, then standardize, regardless of the order in trans_list.

The pipeline only contains sklearn/numpy objects (np.log1p, StandardScaler),
so a saved model can be unpickled by the feature importance service without
needing this module installed there.

Feature importance:
  - permutation_importance works on the Pipeline unchanged; features stay as genes.
  - RFE needs to be pointed at the final estimator: use rfe_importance_getter(model).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

TRANSFORM_CODES = {"l": "log", "s": "standardize"}
MODEL_STEP = "model"


def parse_trans_list(trans_list: Optional[str]) -> list[str]:
    """'s,l' -> ['l', 's'] (validated, safe order)."""
    codes = {c.strip().lower() for c in (trans_list or "").split(",") if c.strip()}
    unknown = codes - TRANSFORM_CODES.keys()
    if unknown:
        raise ValueError(f"Unknown transformation codes {sorted(unknown)}; "
                         f"supported: {sorted(TRANSFORM_CODES)}")
    return [c for c in ("l", "s") if c in codes]


def build_pipeline(estimator, trans_list: Optional[str]) -> Pipeline:
    """Wrap estimator as Pipeline([log?, scale?, model])."""
    steps = []
    for code in parse_trans_list(trans_list):
        if code == "l":
            steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        elif code == "s":
            steps.append(("scale", StandardScaler()))
    steps.append((MODEL_STEP, estimator))
    return Pipeline(steps)


def check_input_for_log(X, trans_list: Optional[str]) -> None:
    """log1p needs non-negative input (raw or normalized counts). Fail loudly otherwise."""
    if "l" not in parse_trans_list(trans_list):
        return
    values = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    if np.isnan(values).any():
        raise ValueError("Input contains NaN; log transform would propagate it.")
    if (values < 0).any():
        n_bad = int((values < 0).any(axis=0).sum())
        raise ValueError(f"{n_bad} features have negative values; the log transform expects counts. "
                         "Was this dataset already transformed upstream?")


def final_estimator(model):
    """The underlying estimator, whether or not model is a Pipeline."""
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


def rfe_importance_getter(model) -> Optional[str]:
    """
    importance_getter string for sklearn RFE, or None if the estimator has neither
    coef_ nor feature_importances_ (e.g. RBF SVC) -> skip RFE for that model.
    Call on a FITTED model.
    """
    est = final_estimator(model)
    prefix = f"named_steps.{MODEL_STEP}." if isinstance(model, Pipeline) else ""
    if hasattr(est, "feature_importances_"):
        return prefix + "feature_importances_"
    if hasattr(est, "coef_"):
        return prefix + "coef_"
    return None
