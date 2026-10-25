# orchestration_service/src/enrichment_models.py
"""Configuration and request models for GO enrichment.

`EnrichmentConfig` is the `"enrichment": {...}` block of the pipeline config. Add it to
PipelineConfig in models.py so it's checked when a run starts:

    from src.enrichment_models import EnrichmentConfig
    class PipelineConfig(BaseModel):
        ...
        enrichment: EnrichmentConfig = EnrichmentConfig()
"""
import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

OraList = Literal["ml_consensus", "deseq2_significant", "deseq2_up", "deseq2_down", "overlap"]
GseaRanking = Literal["deseq2", "ml_importance"]


class EnrichmentConfig(BaseModel):
    enabled: bool = False
    methods: List[Literal["ora", "gsea"]] = ["ora", "gsea"]
    ontologies: List[Literal["BP", "MF", "CC"]] = ["BP"]
    ora_lists: List[OraList] = ["ml_consensus", "deseq2_significant", "overlap"]
    gsea_rankings: List[GseaRanking] = ["deseq2", "ml_importance"]
    organism: Literal["auto", "mmu", "hsa"] = "auto"
    p_adjust_method: Literal["BH", "BY", "bonferroni", "holm", "hochberg", "hommel", "fdr", "none"] = "BH"
    padj_cutoff: float = Field(0.05, gt=0, le=1)
    qvalue_cutoff: float = Field(0.2, gt=0, le=1)
    min_gs_size: int = Field(10, ge=1)
    max_gs_size: int = Field(500, ge=1)
    simplify: bool = True
    simplify_cutoff: float = Field(0.7, gt=0, le=1)
    top_n_terms: int = Field(20, ge=1)


class RankingIn(BaseModel):
    scores: Dict[str, float]                      # gene -> score
    score_type: Literal["std", "pos"] = "std"


class EnrichmentRunRequest(BaseModel):
    config: EnrichmentConfig
    universe: List[str]
    gene_lists: Dict[str, List[str]] = {}
    rankings: Dict[str, RankingIn] = {}
    analysis_id: Optional[str] = None


_MOUSE_SYMBOL = re.compile(r"^[A-Z][a-z0-9][A-Za-z0-9-]*$")   # Acta1, Gm15564
_HUMAN_SYMBOL = re.compile(r"^[A-Z][A-Z0-9-]+$")              # ACTA1


def detect_organism(genes: List[str]) -> str:
    """'mmu' or 'hsa' from Ensembl prefixes, else from symbol capitalisation."""
    ens_mouse = sum(1 for g in genes if g.startswith("ENSMUSG"))
    ens_human = sum(1 for g in genes if g.startswith("ENSG"))
    if ens_mouse or ens_human:
        return "mmu" if ens_mouse >= ens_human else "hsa"
    mouse = sum(1 for g in genes if _MOUSE_SYMBOL.match(g))
    human = sum(1 for g in genes if _HUMAN_SYMBOL.match(g))
    return "hsa" if human > mouse else "mmu"
