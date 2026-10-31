# orchestration_service/src/causal_models.py
"""Configuration for the CRISP causal ensemble (causal_service): the "causal": {...} block of
PipelineConfig.

CRISP's invariance methods look for genes whose relationship to the factor (e.g. Space Flight vs
Ground Control) is the same in every environment (here: every OSD study), so the genes it
reports are ones whose flight signal holds across studies, missions and labs.
"""
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

CrispModel = Literal["ICP", "NLICP", "IRM", "LIRM", "ERM", "RF"]


class CausalConfig(BaseModel):
    enabled: bool = False
    # ICP/NLICP = (non-)linear invariant causal prediction, IRM/LIRM = (linear) invariant risk
    # minimization. ERM and RF are CRISP's non-causal baselines (reported, never ranked).
    models: List[CrispModel] = ["ICP", "NLICP", "IRM", "LIRM"]
    environment_column: str = "source_dataset"          # the OSD study each sample came from
    # ICP, NLICP and LIRM run on the top max_features genes of a feature-reduction IRM. Larger values
    # let more genes through but cost time and memory (NLICP's defining-set step can run out of memory).
    max_features: int = Field(20, ge=2, le=200)
    ranking_method: Literal["consensus", "crisp_v1.1"] = "consensus"
    top_k: int = Field(0, ge=0)                         # consensus: votes for each method's top_k; 0 = max_features
    seed: int = 12
    test_fraction: float = Field(0.2, gt=0, lt=0.5)
    val_fraction: float = Field(0.1, gt=0, lt=0.5)
    model_options: Dict[str, Dict[str, Any]] = {}       # per-model overrides, e.g. {"ICP": {"alpha": 0.01}}
    reuse: bool = True                                  # reuse a cached result for the same data + CRISP settings
    timeout_minutes: int = Field(360, ge=1)
    min_votes: int = Field(0, ge=0)                     # notebook: genes with >= min_votes; 0 = majority of methods
