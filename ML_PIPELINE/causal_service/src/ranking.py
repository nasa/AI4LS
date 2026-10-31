"""
Combine CRISP's per-method results into one gene ranking.

  consensus (default)
      For each causal method that produced coefficients: scale |coefficient| by the method's
      largest |coefficient| (so 0 stays 0 and the top gene is 1), and give a vote to its top_k
      genes. Genes are ranked by votes, then by mean scaled |coefficient| across those methods.
      consensus = votes >= a majority of the methods. Mirrors the ML ensemble's consensus
      (votes, then mean rank). Coefficient signs aren't comparable across CRISP's methods
      (linear coefficients vs network sensitivities), so the direction of each gene is computed
      from the data afterwards (crisp_runner.py).

  crisp_v1.1
      CRISP v1.1's own ensemble (utils.plotting.get_ensemble_results): MinMax-scale each method's
      coefficients to [-1, 1], weight by the share of methods with the gene in their top 20,
      rank by |weighted coefficient|. Kept for comparison with earlier CRISP results. Note that
      MinMax scaling moves zero, so for all-positive coefficients the least important gene maps
      to -1 and ranks with the most important one.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

CAUSAL_CODES = {"ICP", "NLICP", "IRM", "LIRM"}


def consensus_ranking(methods: List[Dict], top_k: int) -> pd.DataFrame:
    used = [m for m in methods if m["causal"] and m["coefficients"]
            and np.nanmax(np.abs(m["coefficients"])) > 0]
    rows = {}
    for m in used:
        c = np.asarray(m["coefficients"], dtype=float)
        c = np.nan_to_num(c)
        scaled = np.abs(c) / np.abs(c).max()
        order = [i for i in np.argsort(-scaled) if scaled[i] > 0][:top_k]
        for i in order:
            g = m["features"][i]
            r = rows.setdefault(g, {"gene": g, "votes": 0, "scaled_sum": 0.0})
            r["votes"] += 1
            r["scaled_sum"] += float(scaled[i])
    n = len(used)
    if not rows:
        return pd.DataFrame(columns=["gene", "rank", "score", "votes", "n_methods", "consensus"])
    df = pd.DataFrame(rows.values())
    df["score"] = df["scaled_sum"] / max(n, 1)
    df["n_methods"] = n
    df["consensus"] = df["votes"] >= (n // 2 + 1)
    df = df.sort_values(["votes", "score"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["gene", "rank", "score", "votes", "n_methods", "consensus"]]


def crisp_v11_ranking(raw_results: List[Dict]) -> pd.DataFrame:
    """CRISP's original ensemble on the raw results_for_bucket entries (needs CRISP on sys.path)."""
    from utils.plotting import get_ensemble_results, NON_CAUSAL
    usable = []
    for r in raw_results:
        if "method" not in r or r["method"] in NON_CAUSAL:
            continue
        c = r.get("coefficients")
        if c is None or isinstance(c, float):     # no solution / scalar: get_ensemble_results can't use it
            continue
        if isinstance(c, list) and len(c) == 1 and isinstance(c[0], list):
            r = {**r, "coefficients": c[0]}
        if r.get("test_acc") is None:
            r = {**r, "test_acc": 1.0}
        usable.append(r)
    n = len(usable)
    if not usable:
        return pd.DataFrame(columns=["gene", "rank", "score", "votes", "n_methods", "consensus"])
    fw = get_ensemble_results(usable)
    df = pd.DataFrame({
        "gene": fw["feature"].astype(str).values,
        "score": fw["weighted_coefficient"].astype(float).abs().values,
        "votes": fw["count_models"].astype(int).values,
    })
    df = df[df["score"] > 0].sort_values("score", ascending=False).reset_index(drop=True)
    df["n_methods"] = n
    df["consensus"] = df["votes"] >= (n // 2 + 1)
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["gene", "rank", "score", "votes", "n_methods", "consensus"]]
