# src/go_enrichment.py
"""GO enrichment (ORA on gene lists + GSEA on rankings) using R clusterProfiler via rpy2."""

import logging
import math
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

R_SCRIPT = Path("/app/r_scripts/go_enrichment.R")
_DUP_SUFFIX = re.compile(r"_\d+$")   # ConvertFeatureNames renames duplicate symbols Gene_1, Gene_2


def clean_gene(gene: str) -> str:
    """Undo the _1/_2 de-duplication suffix; leave Ensembl IDs alone."""
    g = str(gene).strip()
    return g if g.startswith("ENS") else _DUP_SUFFIX.sub("", g)


def _na(v, default=0.0):
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


class GOEnrichment:
    def __init__(self, output_base_path: str = "/app/results", r_script: Path = R_SCRIPT):
        self.output_base_path = Path(output_base_path)
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        if not r_script.exists():
            raise FileNotFoundError(f"R script not found: {r_script}")
        import rpy2.robjects as ro
        self._ro = ro
        ro.r(f'source("{r_script}")')
        self._r_func = ro.globalenv["run_go_enrichment"]
        logger.info(f"Sourced R script: {r_script}")

    def _call_r(self, lists_path, rankings_path, universe_path, output_dir, organism, ontologies,
                p_adjust_method, padj_cutoff, qvalue_cutoff, min_gs_size, max_gs_size,
                do_simplify, simplify_cutoff) -> int:
        ro = self._ro
        res = self._r_func(str(lists_path), str(rankings_path), str(universe_path), str(output_dir),
                           organism, ro.StrVector(list(ontologies)), p_adjust_method,
                           float(padj_cutoff), float(qvalue_cutoff), int(min_gs_size),
                           int(max_gs_size), bool(do_simplify), float(simplify_cutoff))
        return int(res[0]) if res is not None and len(res) else 0

    def run(
        self,
        gene_lists: Dict[str, Sequence[str]],
        rankings: Dict[str, Dict],            # name -> {"genes": [...], "scores": [...], "score_type": "std"|"pos"}
        universe: Sequence[str],
        organism: str = "mmu",
        ontologies: Sequence[str] = ("BP",),
        p_adjust_method: str = "BH",
        padj_cutoff: float = 0.05,
        qvalue_cutoff: float = 0.2,
        min_gs_size: int = 10,
        max_gs_size: int = 500,
        simplify: bool = True,
        simplify_cutoff: float = 0.7,
        analysis_id: Optional[str] = None,
    ) -> Dict:
        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        output_dir = self.output_base_path / f"go_enrichment_{analysis_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        universe_clean = sorted({clean_gene(g) for g in universe if str(g).strip()})
        lists_rows = [(name, clean_gene(g)) for name, genes in gene_lists.items() for g in genes]
        rank_rows = []
        for name, r in rankings.items():
            st = r.get("score_type", "std")
            for g, s in zip(r["genes"], r["scores"]):
                rank_rows.append((name, clean_gene(g), float(s), st))

        lists_path = output_dir / "lists.csv"
        rankings_path = output_dir / "rankings.csv"
        universe_path = output_dir / "universe.csv"
        pd.DataFrame(lists_rows, columns=["name", "gene"]).drop_duplicates().to_csv(lists_path, index=False)
        pd.DataFrame(rank_rows, columns=["name", "gene", "score", "score_type"]).to_csv(rankings_path, index=False)
        pd.DataFrame({"gene": universe_clean}).to_csv(universe_path, index=False)

        logger.info(f"GO enrichment {analysis_id}: lists={list(gene_lists)} rankings={list(rankings)} "
                    f"universe={len(universe_clean)} organism={organism} ontologies={list(ontologies)}")
        n_terms = self._call_r(lists_path, rankings_path, universe_path, output_dir, organism, ontologies,
                               p_adjust_method, padj_cutoff, qvalue_cutoff, min_gs_size, max_gs_size,
                               simplify, simplify_cutoff)

        terms_df = pd.read_csv(output_dir / "go_enrichment_results.csv")
        summary_df = pd.read_csv(output_dir / "go_enrichment_summary.csv")
        terms = [{
            "analysis": str(r["analysis"]), "method": str(r["method"]), "ontology": str(r["ontology"]),
            "term_id": str(r["term_id"]), "description": str(r["description"]),
            "set_size": int(_na(r["set_size"])), "count": int(_na(r["count"])),
            "gene_ratio": "" if pd.isna(r["gene_ratio"]) else str(r["gene_ratio"]),
            "bg_ratio": "" if pd.isna(r["bg_ratio"]) else str(r["bg_ratio"]),
            "enrichment_score": _na(r["enrichment_score"]), "nes": _na(r["nes"]),
            "pvalue": _na(r["pvalue"], 1.0), "padj": _na(r["padj"], 1.0), "qvalue": _na(r["qvalue"], 1.0),
            "genes": [] if pd.isna(r["genes"]) else str(r["genes"]).split("/"),
        } for _, r in terms_df.iterrows()]
        summaries = [{
            "analysis": str(r["analysis"]), "method": str(r["method"]), "ontology": str(r["ontology"]),
            "n_input": int(_na(r["n_input"])), "n_mapped": int(_na(r["n_mapped"])),
            "n_terms": int(_na(r["n_terms"])), "note": "" if pd.isna(r["note"]) else str(r["note"]),
        } for _, r in summary_df.iterrows()]
        logger.info(f"✓ GO enrichment {analysis_id}: {n_terms} terms")
        return {"terms": terms, "summaries": summaries, "output_dir": str(output_dir)}
