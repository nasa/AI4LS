"""gRPC servicer: runs each CRISP analysis in a child process (src/crisp_runner.py)."""
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from src import causal_service_pb2, causal_service_pb2_grpc

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get("CRISP_RESULTS_DIR", "/app/results"))
DEFAULT_TIMEOUT = int(os.environ.get("CRISP_TIMEOUT_SECONDS", str(6 * 3600)))
MAX_CONCURRENT = int(os.environ.get("CRISP_MAX_CONCURRENT", "1"))   # CRISP is CPU- and memory-heavy
LOG_TAIL_LINES = 60


def _nan(x):
    return float("nan") if x is None else float(x)


def _tail(path: Path, n=LOG_TAIL_LINES) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-n:])
    except OSError:
        return ""


class CausalServiceImpl(causal_service_pb2_grpc.CausalServiceServicer):
    def __init__(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self._slots = threading.Semaphore(MAX_CONCURRENT)
        logger.info(f"CausalService ready (results in {RESULTS_DIR}, max {MAX_CONCURRENT} concurrent run(s), "
                    f"timeout {DEFAULT_TIMEOUT}s)")

    def RunCRISP(self, request, context):
        analysis_id = request.analysis_id or uuid.uuid4().hex[:12]
        workdir = RESULTS_DIR / f"crisp_{analysis_id}"
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "input.csv.gz").write_bytes(request.dataset_csv_gz)
        job = {
            "analysis_id": analysis_id,
            "group_column": request.group_column,
            "control_value": request.control_value,
            "treatment_value": request.treatment_value,
            "environment_column": request.environment_column or "source_dataset",
            "models": list(request.models),
            "max_features": request.max_features or 20,
            "seed": request.seed,
            "test_fraction": request.test_fraction or 0.2,
            "val_fraction": request.val_fraction or 0.1,
            "ranking_method": request.ranking_method or "consensus",
            "top_k": request.top_k,
            "model_options_json": request.model_options_json,
        }
        (workdir / "job.json").write_text(json.dumps(job, indent=2))
        timeout = request.timeout_seconds or DEFAULT_TIMEOUT
        log_path = workdir / "crisp.log"

        logger.info(f"CRISP {analysis_id}: waiting for a slot")
        with self._slots:
            logger.info(f"CRISP {analysis_id}: running (models {job['models'] or 'default'}, "
                        f"max_features {job['max_features']}, ranking {job['ranking_method']})")
            t0 = time.time()
            env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
            with log_path.open("w") as log:
                proc = subprocess.Popen([sys.executable, "-m", "src.crisp_runner", str(workdir / "job.json")],
                                        cwd=str(APP_DIR), stdout=log, stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True)
                try:
                    code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    return self._fail(f"CRISP did not finish within {timeout} s (raise timeout_seconds, "
                                      f"lower max_features, or run fewer models)", log_path, t0)

        out_path = workdir / "output.json"
        if not out_path.exists():
            if code == -signal.SIGKILL:
                msg = ("CRISP was killed (most likely out of memory). Lower max_features, or leave NLICP out: "
                       "its defining-set step grows quickly when many gene subsets pass its invariance test.")
            else:
                msg = f"CRISP exited with code {code} without writing results"
            return self._fail(msg, log_path, t0)
        out = json.loads(out_path.read_text())
        if not out.get("success"):
            return self._fail(out.get("error_message", "CRISP failed"), log_path, t0)

        logger.info(f"CRISP {analysis_id}: done in {out['runtime_seconds']:.0f}s, {len(out['ranking'])} ranked genes")
        return causal_service_pb2.CRISPResponse(
            success=True,
            ranking_method=out["ranking_method"],
            methods=[causal_service_pb2.MethodResult(
                method=m["method"], code=m["code"], causal=m["causal"], features=m["features"],
                coefficients=m["coefficients"], pvals=m["pvals"], pvalue=_nan(m["pvalue"]),
                test_acc=_nan(m["test_acc"]), val_acc=_nan(m["val_acc"]), note=m["note"],
            ) for m in out["methods"]],
            ranking=[causal_service_pb2.RankedGene(
                gene=str(r["gene"]), rank=int(r["rank"]), score=float(r["score"]), votes=int(r["votes"]),
                n_methods=int(r["n_methods"]), direction=int(r["direction"]), consensus=bool(r["consensus"]),
                log2fc=_nan(r.get("log2fc")), consistent=bool(r.get("consistent")),
            ) for r in out["ranking"]],
            environments=[causal_service_pb2.EnvironmentInfo(**e) for e in out["environments"]],
            n_samples=out["n_samples"], n_genes=out["n_genes"], normalization=out["normalization"],
            runtime_seconds=out["runtime_seconds"], notes=out["notes"], log_tail=_tail(log_path, 15),
        )

    @staticmethod
    def _fail(message, log_path, t0):
        logger.error(f"CRISP failed: {message}")
        return causal_service_pb2.CRISPResponse(success=False, error_message=message,
                                                log_tail=_tail(log_path), runtime_seconds=time.time() - t0)
