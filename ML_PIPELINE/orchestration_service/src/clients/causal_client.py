# orchestration_service/src/clients/causal_client.py
"""Client for causal_service (CRISP). Stubs: src/generated/causal_service_pb2*.py, made from
causal_service/proto/causal_service.proto the usual way (import fixed to `from src.generated import ...`)."""
import gzip
import io
import json
import logging
import math
from typing import Dict, List, Optional

import grpc
import pandas as pd

from src.generated import causal_service_pb2
from src.generated.causal_service_pb2_grpc import CausalServiceStub

logger = logging.getLogger(__name__)
MAX_MESSAGE = 500 * 1024 * 1024


def _num(x: float) -> Optional[float]:
    return None if x is None or math.isnan(x) else float(x)


class CausalServiceClient:
    def __init__(self, service_url: str):
        self.service_url = service_url
        self.channel = grpc.insecure_channel(service_url, options=[
            ("grpc.max_send_message_length", MAX_MESSAGE),
            ("grpc.max_receive_message_length", MAX_MESSAGE),
            ("grpc.keepalive_time_ms", 60_000),           # CRISP calls can run for hours
            ("grpc.keepalive_permit_without_calls", 1),
        ])
        self.stub = CausalServiceStub(self.channel)

    def run_crisp(self, df: pd.DataFrame, analysis_id: str, group_column: str, control_value: str,
                  treatment_value: str, environment_column: str = "source_dataset",
                  models: List[str] = None, max_features: int = 20, seed: int = 12,
                  test_fraction: float = 0.2, val_fraction: float = 0.1, ranking_method: str = "consensus",
                  top_k: int = 0, model_options: Dict = None, timeout_minutes: int = 360) -> Dict:
        """Run CRISP on df (samples x genes + metadata). Returns a plain dict; success=False on failure."""
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as f:
            f.write(df.to_csv().encode())
        request = causal_service_pb2.CRISPRequest(
            analysis_id=analysis_id, dataset_csv_gz=buf.getvalue(), group_column=group_column,
            control_value=control_value, treatment_value=treatment_value,
            environment_column=environment_column, models=models or [], max_features=max_features,
            seed=seed, test_fraction=test_fraction, val_fraction=val_fraction, ranking_method=ranking_method,
            top_k=top_k, model_options_json=json.dumps(model_options or {}),
            timeout_seconds=int(timeout_minutes * 60))
        try:
            r = self.stub.RunCRISP(request, timeout=timeout_minutes * 60 + 300)
        except grpc.RpcError as e:
            logger.error(f"gRPC error in run_crisp: {e.code()} - {e.details()}")
            return {"success": False, "error": f"{e.code().name}: {e.details()}"}
        if not r.success:
            return {"success": False, "error": r.error_message, "log_tail": r.log_tail,
                    "runtime_seconds": r.runtime_seconds}
        return {
            "success": True,
            "ranking_method": r.ranking_method,
            "methods": [{
                "method": m.method, "code": m.code, "causal": m.causal, "features": list(m.features),
                "coefficients": list(m.coefficients), "pvals": list(m.pvals), "pvalue": _num(m.pvalue),
                "test_acc": _num(m.test_acc), "val_acc": _num(m.val_acc), "note": m.note,
            } for m in r.methods],
            "ranking": [{
                "gene": g.gene, "rank": g.rank, "score": g.score, "votes": g.votes, "n_methods": g.n_methods,
                "direction": g.direction, "consensus": g.consensus, "log2fc": _num(g.log2fc),
                "consistent": g.consistent,
            } for g in r.ranking],
            "environments": [{"name": e.name, "n_control": e.n_control, "n_treatment": e.n_treatment,
                              "used": e.used, "note": e.note} for e in r.environments],
            "n_samples": r.n_samples, "n_genes": r.n_genes, "normalization": r.normalization,
            "runtime_seconds": r.runtime_seconds, "notes": list(r.notes),
        }

    def close(self):
        self.channel.close()
