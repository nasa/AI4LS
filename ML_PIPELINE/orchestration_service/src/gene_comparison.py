"""
gene_comparison.py

Compare the ML ensemble's consensus important genes against DESeq2
differentially expressed genes (DEGs).

Drop into: orchestration_service/src/gene_comparison.py

Inputs are deliberately flexible:
  consensus_genes : list of gene names, OR dict {gene: votes/score},
                    OR DataFrame with a gene column (+ optional votes/score column)
  deseq2_results  : DataFrame or list of dicts with a gene column plus
                    log2FoldChange and padj columns (common name variants accepted)
  universe        : all genes that were tested (genes in the filtered dataset).
                    Needed for the enrichment p-value; if omitted, the union of
                    the DESeq2 result genes and consensus genes is used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd
from scipy.stats import hypergeom

logger = logging.getLogger(__name__)

_GENE_COLS = ["gene", "gene_id", "gene_symbol", "symbol", "feature", "feature_name", "name"]
_LFC_COLS = ["log2FoldChange", "log2_fold_change", "log2fc", "l2fc", "lfc"]
_PADJ_COLS = ["padj", "p_adj", "adj_pvalue", "fdr", "qvalue"]
_SCORE_COLS = ["votes", "vote_count", "n_models", "score", "importance", "mean_importance", "consensus_score"]


def _find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")
    return None


def _normalize_consensus(consensus) -> pd.DataFrame:
    """Return DataFrame[gene, ml_score, ml_rank]."""
    if isinstance(consensus, pd.DataFrame):
        df = consensus.copy()
        gcol = _find_col(df, _GENE_COLS, required=False)
        if gcol is None:  # gene names may be the index
            df = df.reset_index().rename(columns={"index": "gene"})
            gcol = "gene"
        scol = _find_col(df, _SCORE_COLS, required=False)
        out = pd.DataFrame({"gene": df[gcol].astype(str)})
        out["ml_score"] = df[scol].values if scol else range(len(df), 0, -1)
    elif isinstance(consensus, dict):
        out = pd.DataFrame({"gene": [str(g) for g in consensus], "ml_score": list(consensus.values())})
    else:  # ordered iterable, most important first
        genes = [str(g) for g in consensus]
        out = pd.DataFrame({"gene": genes, "ml_score": range(len(genes), 0, -1)})

    out = out.drop_duplicates("gene").sort_values("ml_score", ascending=False).reset_index(drop=True)
    out["ml_rank"] = range(1, len(out) + 1)
    return out


def _normalize_deseq2(deseq2_results) -> pd.DataFrame:
    """Return DataFrame[gene, log2FoldChange, padj] for all tested genes."""
    df = pd.DataFrame(deseq2_results).copy()
    gcol = _find_col(df, _GENE_COLS, required=False)
    if gcol is None:
        df = df.reset_index().rename(columns={"index": "gene"})
        gcol = "gene"
    lfc = _find_col(df, _LFC_COLS)
    padj = _find_col(df, _PADJ_COLS)
    out = pd.DataFrame({
        "gene": df[gcol].astype(str),
        "log2FoldChange": pd.to_numeric(df[lfc], errors="coerce"),
        "padj": pd.to_numeric(df[padj], errors="coerce"),
    })
    return out.drop_duplicates("gene").reset_index(drop=True)


def compare_genes(
    consensus_genes: Union[List[str], Dict[str, float], pd.DataFrame],
    deseq2_results: Union[pd.DataFrame, List[dict]],
    universe: Optional[Iterable[str]] = None,
    padj_threshold: float = 0.05,
    log2fc_threshold: float = 0.0,
    top_n_ml: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Compare ML consensus genes with DESeq2 significant genes.

    Returns a dict with:
      overlap, ml_only, deseq2_only : sorted gene lists
      n_ml, n_deseq2, n_overlap, n_universe
      jaccard                       : |A∩B| / |A∪B|
      hypergeom_pvalue              : P(overlap >= observed) by chance
      expected_overlap              : overlap expected by chance
      fold_enrichment               : observed / expected
      table                         : DataFrame, one row per gene in either set
    """
    ml = _normalize_consensus(consensus_genes)
    if top_n_ml is not None:
        ml = ml.head(top_n_ml)
    de = _normalize_deseq2(deseq2_results)

    sig_mask = (de["padj"] < padj_threshold) & (de["log2FoldChange"].abs() > log2fc_threshold)
    deg = de[sig_mask]

    ml_set = set(ml["gene"])
    deg_set = set(deg["gene"])

    if universe is not None:
        universe_set = {str(g) for g in universe}
    else:
        universe_set = set(de["gene"]) | ml_set
        logger.warning("No gene universe supplied; using DESeq2-tested genes ∪ ML genes "
                       f"({len(universe_set)} genes) for the enrichment test.")

    # Genes outside the universe can't be tested; drop them from both sets
    dropped = (ml_set | deg_set) - universe_set
    if dropped:
        logger.warning(f"{len(dropped)} genes not in universe were ignored: {sorted(dropped)[:10]}...")
    ml_set &= universe_set
    deg_set &= universe_set

    overlap = ml_set & deg_set
    union = ml_set | deg_set
    N, K, n, k = len(universe_set), len(deg_set), len(ml_set), len(overlap)

    expected = (n * K / N) if N else 0.0
    pval = float(hypergeom.sf(k - 1, N, K, n)) if N and n and K else 1.0
    jaccard = (k / len(union)) if union else 0.0
    fold = (k / expected) if expected > 0 else float("nan")

    # Merged per-gene table
    table = pd.DataFrame({"gene": sorted(union)})
    table = table.merge(ml[["gene", "ml_rank", "ml_score"]], on="gene", how="left")
    table = table.merge(de, on="gene", how="left")
    table["in_ml"] = table["gene"].isin(ml_set)
    table["in_deseq2"] = table["gene"].isin(deg_set)
    table["category"] = table.apply(
        lambda r: "both" if r.in_ml and r.in_deseq2 else ("ml_only" if r.in_ml else "deseq2_only"), axis=1
    )
    table["direction"] = table["log2FoldChange"].apply(
        lambda x: "up" if pd.notna(x) and x > 0 else ("down" if pd.notna(x) and x < 0 else "")
    )
    cat_order = {"both": 0, "ml_only": 1, "deseq2_only": 2}
    table = table.sort_values(
        by=["category", "ml_rank", "padj"],
        key=lambda s: s.map(cat_order) if s.name == "category" else s,
        na_position="last",
    ).reset_index(drop=True)

    result = {
        "overlap": sorted(overlap),
        "ml_only": sorted(ml_set - deg_set),
        "deseq2_only": sorted(deg_set - ml_set),
        "n_ml": n,
        "n_deseq2": K,
        "n_overlap": k,
        "n_universe": N,
        "jaccard": round(jaccard, 4),
        "expected_overlap": round(expected, 3),
        "fold_enrichment": round(fold, 3) if fold == fold else None,
        "hypergeom_pvalue": pval,
        "padj_threshold": padj_threshold,
        "log2fc_threshold": log2fc_threshold,
        "table": table,
    }

    logger.info(
        f"Gene comparison: ML={n}, DESeq2={K}, overlap={k} "
        f"(expected {expected:.2f}, p={pval:.3g}, Jaccard={jaccard:.3f})"
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "gene_comparison.csv", index=False)
        summary = {k_: v for k_, v in result.items() if k_ != "table"}
        pd.Series(summary).to_json(out / "gene_comparison_summary.json", indent=2)
        logger.info(f"Saved gene comparison to {out}")

    return result
