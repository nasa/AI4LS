from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import hashlib
import logging
import os
import threading
import time
import uuid
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import grpc
import pandas as pd

from src.config import Settings, get_settings
from src.models import (
    ValidationResponse,
    TransformationRequest,
    TransformationResponse,
    PipelineConfig,
    PipelineRequest,
    PipelineResponse,
    PipelineStatus,
    HealthResponse,
    DatasetInfo,
    ColumnInfo
)
from src.clients.data_client import DataServiceClient
from src.clients.ml_client import MLServiceClient
from src.clients.feature_importance_client import FeatureImportanceClient
from src.clients.bioinformatics_client import BioinformaticsClient
from src.clients.causal_client import CausalServiceClient
from src.enrichment_models import EnrichmentRunRequest, detect_organism

from src.generated import data_service_pb2, data_service_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global clients
data_client: DataServiceClient = None
ml_client: MLServiceClient = None
fi_client: FeatureImportanceClient = None
bio_client: BioinformaticsClient = None  # ← ADD
causal_client: Optional[CausalServiceClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    global data_client, ml_client, fi_client, bio_client, causal_client
    
    settings = get_settings()
    
    logger.info("Initializing gRPC clients...")
    try:
        data_client = DataServiceClient(settings.data_service_url)
        logger.info("✓ Data Service client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Data Service client: {e}")
    
    try:
        ml_client = MLServiceClient(settings.ml_service_url)
        logger.info("✓ ML Service client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ML Service client: {e}")

    try:
        fi_client = FeatureImportanceClient(settings.feature_importance_service_url)
        logger.info("✓ Feature Importance Service client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Feature Importance Service client: {e}")

    try:
        bio_client = BioinformaticsClient(settings.bioinformatics_service_url)
        logger.info("✓ Bioinformatics Service client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Bioinformatics Service client: {e}")

    try:
        causal_url = getattr(settings, "causal_service_url", None) or os.environ.get(
            "CAUSAL_SERVICE_URL", "causal_service:50055")
        causal_client = CausalServiceClient(causal_url)
        logger.info(f"✓ Causal Service (CRISP) client initialized ({causal_url})")
    except Exception as e:
        logger.error(f"Failed to initialize Causal Service client: {e}")
    
    yield
    
    logger.info("Shutting down...")
    if data_client:
        data_client.close()
    if ml_client:
        ml_client.close()
    if fi_client:
        fi_client.close()
    if bio_client:
        bio_client.close()
    if causal_client:
        causal_client.close()

app = FastAPI(
    title="ML Pipeline Orchestration Service",
    description="REST API for managing ML pipeline workflows",
    version="1.0.0",
    lifespan=lifespan
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────

class UploadRequest(BaseModel):
    exclude_columns: Optional[List[str]] = []
    cv_step: float = 0.25 

class DownloadRequest(BaseModel):
    osd_id: str
    patterns: List[str] = ["Unnormalized", "RSEM"]
    dataset_id: Optional[str] = ""
    factor_name: str
    factor_values: List[str]
    min_features: int
    exclude_columns: List[str]
    cv_step: float


# ── Health ────────────────────────────────────────────────────────────────────

HEALTH_TIMEOUT_SECONDS = float(os.environ.get("HEALTH_TIMEOUT_SECONDS", "2"))


def _client_channel(client) -> Optional[grpc.Channel]:
    """The client's gRPC channel, whatever the attribute is called."""
    for attr in ("channel", "_channel"):
        ch = getattr(client, attr, None)
        if isinstance(ch, grpc.Channel):
            return ch
    return None


def _channel_ready(name: str, client) -> bool:
    """True if the service's gRPC channel connects within HEALTH_TIMEOUT_SECONDS.
    Readiness means the server is up and accepting connections; it doesn't call any RPC,
    so a client without a health_check() method no longer counts as down."""
    if client is None:
        logger.warning(f"Health: {name} client was not initialized")
        return False
    channel = _client_channel(client)
    if channel is None:
        # Fallback for a client that doesn't expose its channel
        if hasattr(client, "health_check"):
            try:
                return bool(client.health_check())
            except Exception as e:
                logger.warning(f"Health: {name} health_check() failed: {e}")
                return False
        logger.warning(f"Health: {name} client has no channel attribute or health_check()")
        return False
    try:
        grpc.channel_ready_future(channel).result(timeout=HEALTH_TIMEOUT_SECONDS)
        return True
    except grpc.FutureTimeoutError:
        logger.warning(f"Health: {name} not reachable within {HEALTH_TIMEOUT_SECONDS}s")
        return False
    except Exception as e:
        logger.warning(f"Health: {name} check failed: {e}")
        return False


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check health of orchestration service and downstream services.
    Plain def (runs in a worker thread): the readiness checks block, and they run in
    parallel so a down service costs one timeout, not one per service."""
    clients = {
        "data_service": data_client,
        "ml_service": ml_client,
        "feature_importance_service": fi_client,
        "bioinformatics_service": bio_client,
        "causal_service": causal_client,      # optional: reported, but not counted in "healthy"
    }
    optional = {"causal_service"}
    # Own short-lived pool: _executor runs pipelines and may be busy
    with ThreadPoolExecutor(max_workers=len(clients), thread_name_prefix="health") as pool:
        futures = {name: pool.submit(_channel_ready, name, c) for name, c in clients.items()}
        services_status = {name: f.result() for name, f in futures.items()}

    overall_status = "healthy" if all(v for k, v in services_status.items() if k not in optional) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        services=services_status
    )


# ── Dataset endpoints ─────────────────────────────────────────────────────────

@app.post("/api/datasets/upload", response_model=ValidationResponse)
@app.post("/api/datasets/validate", response_model=ValidationResponse)  # kept for backwards compatibility
async def upload_dataset(
    file: UploadFile = File(...),
    exclude_columns: str = "",  # ADD THIS - query parameter
    settings: Settings = Depends(get_settings)
):
    """Upload a dataset to the data service for storage"""
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size / 1024 / 1024}MB"
        )
    
    format_type = "csv" if file.filename.endswith(".csv") else "json"
    
    # Parse exclude_columns from comma-separated string
    exclude_cols_list = [col.strip() for col in exclude_columns.split(",") if col.strip()]
    
    try:
        result = data_client.upload_dataset(content, format_type, exclude_cols_list)

        response = ValidationResponse(
            is_valid=result["is_valid"],
            errors=result["errors"],
            warnings=result["warnings"]
        )
        
        # Add dataset_info if present
        if result["dataset_info"]:
            info = result["dataset_info"]
            response.dataset_info = DatasetInfo(
                dataset_id=info["dataset_id"],
                num_rows=info["num_rows"],
                num_columns=info["num_columns"],
                size_bytes=info["size_bytes"],
                columns=[ColumnInfo(**col) for col in info["columns"]]
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/api/datasets/download", response_model=ValidationResponse)
async def download_dataset(request: DownloadRequest):
    """
    Download a dataset from NASA OSDR by dataset number.

    - **osd_id**: NASA OSDR dataset number, e.g. "379" for OSD-379
    - **patterns**: File name patterns to match (default: Unnormalized, RSEM)
    - **dataset_id**: Optional ID to assign; auto-generated if omitted
    - **factor_name**: name of column in metadata to get factor values from 
    - **factor_values**: list of column values in metadata for target values 
    - **min_features**: minimum number of features to keep after dim reduction 
    - **exclude_columns**: list of columns to exclude from df 
    """
    try:
        logger.info(f"osd_id = {request.osd_id}") 
        logger.info(f"patterns = {request.patterns}") 
        logger.info(f"dataset_id = {request.dataset_id}") 
        logger.info(f"factor_name = {request.factor_name}") 
        logger.info(f"factor_values = {request.factor_values}") 
        logger.info(f"min_features = {request.min_features}") 
        logger.info(f"exclude_columns = {request.exclude_columns}") 
        logger.info(f"cv_step = {request.cv_step}") 
        result = data_client.download_dataset(
            osd_id=request.osd_id,
            patterns=request.patterns,
            dataset_id=request.dataset_id or "",
            factor_name=request.factor_name,
            factor_values=request.factor_values,
            min_features=request.min_features,
            exclude_columns=request.exclude_columns or [],
            cv_step=request.cv_step or 0.25
        )

        #logger.info(f"Received result from data_client: {result}")

        if "dataset_id" in result:
            dataset_id=result["dataset_id"]
        else:
            dataset_id=""
        response = ValidationResponse(
            is_valid=result["is_valid"],
            dataset_id=dataset_id,
            errors=result["errors"],
            warnings=result["warnings"],
            dataset_info=result["dataset_info"]
            
        )

        return response

    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.post("/api/datasets/{dataset_id}/transform", response_model=TransformationResponse)
async def transform_dataset(
    dataset_id: str,
    request: TransformationRequest
):
    """Apply transformations to a dataset"""
    try:
        transformations = [
            {
                "type": t.type.value,
                "columns": t.columns,
                "params": t.params
            }
            for t in request.transformations
        ]
        
        logger.info(f"applying transformations: {request.transformations}")
        result = data_client.apply_transformations(dataset_id, transformations)
        
        response = TransformationResponse(
            success=result["success"],
            transformed_dataset_id=result.get("transformed_dataset_id"),
            error_message=result.get("error_message")
        )
        
        if result.get("transformed_info"):
            info = result["transformed_info"]
            response.transformed_info = DatasetInfo(
                dataset_id=info["dataset_id"],
                num_rows=info["num_rows"],
                num_columns=info["num_columns"],
                size_bytes=info["size_bytes"],
                columns=[ColumnInfo(**col) for col in info["columns"]]
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Transformation error: {e}")
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")


@app.get("/api/datasets/{dataset_id}", response_model=DatasetInfo)
async def get_dataset_info(dataset_id: str):
    """Get information about a dataset"""
    try:
        result = data_client.get_dataset_info(dataset_id)
        
        return DatasetInfo(
            dataset_id=result["dataset_id"],
            num_rows=result["num_rows"],
            num_columns=result["num_columns"],
            size_bytes=result["size_bytes"],
            columns=[ColumnInfo(**col) for col in result["columns"]]
        )
        
    except Exception as e:
        logger.error(f"Get dataset info error: {e}")
        raise HTTPException(status_code=404, detail=f"Dataset not found: {str(e)}")

@app.get("/api/datasets/{dataset_id}/gene_stats")
def dataset_gene_stats(dataset_id: str):
    """Each gene's mean and variance across samples (for mean-variance plots).
    Plain def: the gRPC fetch blocks, so FastAPI runs it in a worker thread."""
    import numpy as np
    import pandas as pd

    try:
        df = data_client.get_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    except grpc.RpcError as e:
        code = e.code()
        raise HTTPException(status_code=404 if code == grpc.StatusCode.NOT_FOUND else 502,
                            detail=f"Data service: {code.name}: {e.details()}")

    X = df.select_dtypes(include="number")
    # Drop sample metadata columns (factor values, characteristics) that happen to be numeric
    X = X.loc[:, [c for c in X.columns if not str(c).startswith(("Factor Value", "Characteristics", "Comment"))]]
    stats = pd.DataFrame({"gene": X.columns.astype(str), "mean": X.mean().values, "variance": X.var().values})
    stats = stats.replace([np.inf, -np.inf], np.nan).dropna()
    return {"dataset_id": dataset_id, "n_samples": int(len(X)), "genes": stats.to_dict(orient="list")}

_METADATA_PREFIXES = ("Factor Value", "Characteristics", "Comment", "Parameter Value")
_NON_GENE_COLUMNS = {"source_dataset"}


class ExpressionRequest(BaseModel):
    genes: List[str]
    group_column: str = "Factor Value[Spaceflight]"
    normalization: str = "log2_cpm"   # "log2_cpm" = log2(CPM + 1) over all genes in the dataset; "raw" = counts as stored


@app.post("/api/datasets/{dataset_id}/expression")
def dataset_expression(dataset_id: str, req: ExpressionRequest):
    """Per-sample expression for selected genes, with each sample's group (and study, if combined).
    For box plots. Plain def: the gRPC fetch blocks, so FastAPI runs it in a worker thread."""
    import numpy as np

    if req.normalization not in ("log2_cpm", "raw"):
        raise HTTPException(status_code=400, detail="normalization must be 'log2_cpm' or 'raw'")
    if not req.genes:
        raise HTTPException(status_code=400, detail="genes is empty")
    try:
        df = data_client.get_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    except grpc.RpcError as e:
        code = e.code()
        raise HTTPException(status_code=404 if code == grpc.StatusCode.NOT_FOUND else 502,
                            detail=f"Data service: {code.name}: {e.details()}")
    if req.group_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.group_column}' not in dataset {dataset_id}")

    gene_cols = [c for c in df.select_dtypes(include="number").columns
                 if c != req.group_column and c not in _NON_GENE_COLUMNS
                 and not str(c).startswith(_METADATA_PREFIXES)]
    X = df[gene_cols].astype(float)
    if req.normalization == "log2_cpm":
        # Library size over every gene in this dataset (after the pipeline's biotype/CV filters)
        lib = X.sum(axis=1).replace(0, np.nan)
        X = np.log2(X.div(lib, axis=0) * 1e6 + 1)

    present = [g for g in dict.fromkeys(req.genes) if g in X.columns]
    missing = [g for g in dict.fromkeys(req.genes) if g not in X.columns]
    return {
        "dataset_id": dataset_id,
        "n_samples": int(len(df)),
        "normalization": req.normalization,
        "group_column": req.group_column,
        "samples": [str(s) for s in df.index],
        "groups": df[req.group_column].astype(str).tolist(),
        "studies": df["source_dataset"].astype(str).tolist() if "source_dataset" in df.columns else None,
        "values": {g: [None if not np.isfinite(v) else float(v) for v in X[g].to_numpy()] for g in present},
        "missing": missing,
    }


# ── Pipeline jobs ─────────────────────────────────────────────────────────────
#
# Each pipeline run is a background job in a worker thread, so:
#   * a dropped connection (Wi-Fi, laptop sleep, timeout) no longer loses the run;
#     the client reconnects and picks up where it left off by pipeline_id
#   * the server stays responsive while a run is going (the gRPC calls are blocking,
#     and used to freeze the event loop when called from the async generator)
#
#   POST /api/pipeline/jobs                  start a run -> {"pipeline_id", "status"}
#   GET  /api/pipeline/jobs/{id}?since=N     status, events after N, result when done
#   GET  /api/pipeline/jobs                  recent runs
#   POST /api/pipeline/run                   same run, streamed as NDJSON (old clients);
#                                            sends keep-alive newlines during long steps

VALID_ALGORITHMS = [
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'svm',
    'neural_network',
    'logistic_regression',
    'naive_bayes'
]

PIPELINE_WORKERS = int(os.environ.get("PIPELINE_WORKERS", "1"))   # concurrent runs; each is CPU/RAM heavy
JOB_TTL_SECONDS = int(os.environ.get("PIPELINE_JOB_TTL_SECONDS", str(24 * 3600)))
MAX_FINISHED_JOBS = int(os.environ.get("PIPELINE_MAX_FINISHED_JOBS", "50"))
KEEPALIVE_SECONDS = 15

_executor = ThreadPoolExecutor(max_workers=PIPELINE_WORKERS, thread_name_prefix="pipeline")


class PipelineJob:
    """In-memory record of one run. Jobs are lost if the orchestration container restarts."""

    def __init__(self, pipeline_id: str, request: PipelineRequest):
        self.pipeline_id = pipeline_id
        self.request = request
        self.status = "queued"
        self.message = "Waiting for a free worker..."
        self.progress_percent = 0
        self.events: List[dict] = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.updated_at = self.created_at
        self._lock = threading.Lock()

    def emit(self, event: dict):
        event = {"pipeline_id": self.pipeline_id, **event}
        with self._lock:
            self.events.append(event)
            self.status = event.get("status", self.status)
            self.message = event.get("message", self.message)
            if event.get("progress_percent") is not None:
                self.progress_percent = event["progress_percent"]
            if self.status == "completed":
                self.result = event
            elif self.status == "failed":
                self.error = event.get("error") or event.get("message")
            self.updated_at = time.time()

    @property
    def finished(self) -> bool:
        return self.status in ("completed", "failed")

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            return {
                "pipeline_id": self.pipeline_id,
                "status": self.status,
                "message": self.message,
                "progress_percent": self.progress_percent,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "elapsed_seconds": round(self.updated_at - self.created_at),
                "next_since": len(self.events),
                "events": self.events[since:],
                "error": self.error,
                "result": self.result,
            }


_jobs: Dict[str, PipelineJob] = {}
_jobs_lock = threading.Lock()


def _prune_jobs():
    now = time.time()
    with _jobs_lock:
        finished = sorted((j for j in _jobs.values() if j.finished), key=lambda j: j.updated_at)
        for j in finished:
            if now - j.updated_at > JOB_TTL_SECONDS:
                _jobs.pop(j.pipeline_id, None)
        finished = [j for j in finished if j.pipeline_id in _jobs]
        for j in finished[:max(0, len(finished) - MAX_FINISHED_JOBS)]:
            _jobs.pop(j.pipeline_id, None)


def _validate_pipeline_request(request: PipelineRequest):
    if not request.dataset_id and not request.osd_ids:
        raise HTTPException(status_code=400, detail="Either 'dataset_id' or 'osd_ids' must be provided")
    for algo in request.config.ensemble_algorithms:
        if algo not in VALID_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm: '{algo}'. Valid options are: {', '.join(VALID_ALGORITHMS)}"
            )


def _start_job(request: PipelineRequest) -> PipelineJob:
    _validate_pipeline_request(request)
    _prune_jobs()
    job = PipelineJob(str(uuid.uuid4()), request)
    with _jobs_lock:
        _jobs[job.pipeline_id] = job
    _executor.submit(_execute_pipeline, job)
    logger.info(f"Pipeline {job.pipeline_id}: queued")
    return job


# ── DESeq2 result cache ───────────────────────────────────────────────────────
# Keyed on every setting that determines the counts DESeq2 sees and how it's called. In memory
# only: cleared when the orchestration container restarts. Bump DESEQ2_CACHE_VERSION after changing
# the data service's filtering/conversion or the DESeq2 code, so old results aren't reused.
DESEQ2_CACHE_VERSION = 1
DESEQ2_CACHE_MAX = int(os.environ.get("DESEQ2_CACHE_MAX", "20"))
DESEQ2_PADJ_THRESHOLD = 0.05
DESEQ2_LOG2FC_THRESHOLD = 0.0
DOWNLOAD_PATTERNS = ['unnormalized']
DOWNLOAD_CV_STEP = 0.25

_deseq2_cache: "OrderedDict[str, dict]" = OrderedDict()
_deseq2_cache_lock = threading.Lock()
_deseq2_key_locks: Dict[str, threading.Lock] = {}


def _data_settings(request: PipelineRequest) -> dict:
    """Every setting that determines the final dataset (the counts DESeq2 and CRISP are given)."""
    cfg = request.config
    osd_ids = sorted({i.strip() for i in (request.osd_ids or "").split(",") if i.strip()},
                     key=lambda s: (len(s), s))
    coding_only = bool(getattr(cfg, "coding_only", False))
    return {
        # the pipeline uses dataset_id when given, otherwise downloads osd_ids
        "source": {"dataset_id": request.dataset_id} if request.dataset_id else {"osd_ids": osd_ids},
        "factor_name": cfg.factor_name or "Factor Value[Spaceflight]",
        "download_factor_values": list(cfg.factor_values or ["Ground Control", "Space Flight"]),
        "download_patterns": DOWNLOAD_PATTERNS,
        "download_cv_step": DOWNLOAD_CV_STEP,
        "coding_only": coding_only,
        "keep_biotypes": sorted(getattr(cfg, "keep_biotypes", None) or ["protein_coding"]) if coding_only else None,
        "target_column": cfg.target_column,
        "min_features": cfg.min_features or 0,
    }


def _deseq2_cache_key(request: PipelineRequest, condition_col: str, factor_values: List[str]) -> str:
    key = {
        "version": DESEQ2_CACHE_VERSION,
        **_data_settings(request),
        "condition_column": condition_col,
        "control": factor_values[0],
        "treatment": factor_values[1],
        "padj": DESEQ2_PADJ_THRESHOLD,
        "log2fc": DESEQ2_LOG2FC_THRESHOLD,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return f"deseq2_{digest}"


def _deseq2_key_lock(key: str) -> threading.Lock:
    with _deseq2_cache_lock:
        return _deseq2_key_locks.setdefault(key, threading.Lock())


def _deseq2_cache_get(key: str) -> Optional[dict]:
    with _deseq2_cache_lock:
        hit = _deseq2_cache.get(key)
        if hit is None:
            return None
        _deseq2_cache.move_to_end(key)
        return {**hit, "reused": True}


def _deseq2_cache_put(key: str, result: dict):
    with _deseq2_cache_lock:
        _deseq2_cache[key] = result
        _deseq2_cache.move_to_end(key)
        while len(_deseq2_cache) > DESEQ2_CACHE_MAX:
            _deseq2_cache.popitem(last=False)


@app.delete("/api/deseq2/cache")
async def clear_deseq2_cache():
    """Forget all cached DESeq2 results (they are also cleared when the service restarts)."""
    with _deseq2_cache_lock:
        n = len(_deseq2_cache)
        _deseq2_cache.clear()
    return {"cleared": n}


# ── CRISP (causal_service) result cache ───────────────────────────────────────
# Same idea as the DESeq2 cache: keyed on the data settings plus every CRISP setting, in memory only.
# CRISP can take minutes to hours, so runs that differ only in ML settings share one result.
CRISP_CACHE_VERSION = 1


class _ResultCache:
    """Small LRU cache with one lock per key, so identical runs in parallel compute once."""

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._items: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: Dict[str, threading.Lock] = {}

    def key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            self._items.move_to_end(key)
            return {**hit, "reused": True}

    def put(self, key: str, value: dict):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> int:
        with self._lock:
            n = len(self._items)
            self._items.clear()
            return n

    def __len__(self):
        return len(self._items)


_crisp_cache = _ResultCache(int(os.environ.get("CRISP_CACHE_MAX", "20")))
_CRISP_KEY_EXCLUDE = {"enabled", "reuse", "timeout_minutes", "min_votes", "gene_set"}


def _crisp_cache_key(request: PipelineRequest, group_column: str, factor_values: List[str]) -> str:
    causal = request.config.causal
    key = {"version": CRISP_CACHE_VERSION, **_data_settings(request), "group_column": group_column,
           "control": factor_values[0], "treatment": factor_values[1],
           # settings that don't change CRISP's output are left out of the key
           "crisp": causal.model_dump(exclude=_CRISP_KEY_EXCLUDE)
           if hasattr(causal, "model_dump") else causal.dict(exclude=_CRISP_KEY_EXCLUDE)}
    digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return f"crisp_{digest}"


@app.delete("/api/causal/cache")
async def clear_crisp_cache():
    """Forget all cached CRISP results (they are also cleared when the service restarts)."""
    return {"cleared": _crisp_cache.clear()}


def _study_label(osd_id: str) -> str:
    osd_id = str(osd_id).strip()
    return osd_id if osd_id.upper().startswith(("OSD", "GLDS")) else f"OSD-{osd_id}"


def _add_environment_column(df, column: str, dataset_chain: List[str], pipeline_id: str,
                            study_datasets: Optional[Dict[str, str]] = None):
    """CRISP needs each sample's study. If the final dataset lacks `column`:
      1. copy it from the latest earlier dataset of this run that has it (matched by sample id), else
      2. label each sample with the study whose downloaded dataset contains it (matched by sample id).
    Returns (df, note); raises ValueError if neither works."""
    if column in df.columns:
        return df, ""
    for did in reversed(dataset_chain[:-1]):
        try:
            earlier = data_client.get_dataset(did)
        except Exception as e:
            logger.warning(f"Pipeline {pipeline_id}: couldn't fetch {did} for '{column}': {e}")
            continue
        if column not in earlier.columns:
            continue
        labels = earlier[column].reindex(df.index)
        missing = int(labels.isna().sum())
        if missing == len(df):
            logger.warning(f"Pipeline {pipeline_id}: {did} has '{column}' but no matching sample ids")
            continue
        df = df.copy()
        df[column] = labels
        note = f"'{column}' taken from {did}" + (f" ({missing} samples without a label)" if missing else "")
        logger.info(f"Pipeline {pipeline_id}: {note}")
        return df, note

    if study_datasets and len(study_datasets) >= 2:
        owner, dup = {}, set()
        example_ids, fetch_errors = {}, {}
        logger.info(f"Pipeline {pipeline_id}: labelling samples from study datasets {study_datasets}")
        for osd, did in study_datasets.items():
            try:
                ids = [str(i) for i in data_client.get_dataset(did).index]
            except Exception as e:
                fetch_errors[f"{osd}->{did}"] = f"{type(e).__name__}: {str(e)[:120]}"
                logger.warning(f"Pipeline {pipeline_id}: couldn't fetch study dataset {did} (OSD {osd}): {e}")
                continue
            example_ids[osd] = ids[:2]
            for i in ids:
                if i in owner and owner[i] != osd:
                    dup.add(i)
                owner[i] = osd
        labels = [(_study_label(owner[str(i)]) if str(i) in owner and str(i) not in dup else None) for i in df.index]
        n_ok = sum(l is not None for l in labels)
        if n_ok:
            df = df.copy()
            df[column] = labels
            n_bad = len(df) - n_ok
            note = (f"'{column}' rebuilt from the {len(study_datasets)} downloaded study datasets "
                    f"({n_ok}/{len(df)} samples matched" + (f"; {len(dup)} sample ids in several studies" if dup else "")
                    + ")")
            if n_bad:
                logger.warning(f"Pipeline {pipeline_id}: {note}; unmatched samples are left out of CRISP")
            logger.info(f"Pipeline {pipeline_id}: {note}")
            return df, note
        if fetch_errors and not example_ids:
            raise ValueError(f"Couldn't fetch any of the downloaded study datasets to label samples by study: "
                             f"{fetch_errors}")
        raise ValueError(f"Couldn't match the final dataset's sample ids (e.g. {[str(i) for i in df.index[:2]]}) to "
                         f"the downloaded study datasets (e.g. {example_ids})"
                         + (f"; fetch errors: {fetch_errors}" if fetch_errors else "")
                         + "; CRISP needs each sample's study")

    other = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    raise ValueError(f"Column '{column}' is not in the final dataset or in any earlier dataset of this run "
                     f"({', '.join(dataset_chain) or 'none recorded'}), and the per-study datasets aren't available "
                     f"(the run started from a dataset_id). Non-numeric columns in the final dataset: {other[:10]}. "
                     f"Set causal.environment_column to the column that identifies the study.")


def _decode_condition(df, column: str, factor_values: List[str], pipeline_id: str):
    """The data service stores the condition as numbers: 1 = factor_values[0], 0 = factor_values[1]
    (DownloadDataset). CRISP matches the text labels, so map the codes back."""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df
    codes = set(pd.unique(df[column].dropna()))
    if not codes <= {0, 1}:
        return df
    df = df.copy()
    df[column] = df[column].map({1: factor_values[0], 0: factor_values[1]})
    logger.info(f"Pipeline {pipeline_id}: '{column}' decoded 1 -> {factor_values[0]}, 0 -> {factor_values[1]}")
    return df


def _run_crisp_step(job: "PipelineJob", request: PipelineRequest, dataset_id: str,
                    dataset_chain: Optional[List[str]] = None,
                    study_datasets: Optional[Dict[str, str]] = None) -> Optional[dict]:
    """CRISP on the final dataset, OSD study as the environment. Returns the result dict or None."""
    causal = request.config.causal
    pipeline_id = job.pipeline_id
    factor_values = request.config.factor_values or []
    group_column = request.config.factor_name or request.config.target_column
    if len(factor_values) < 2:
        job.emit({"status": "causal", "message": "CRISP skipped: needs two factor_values", "progress_percent": 91})
        return None
    if causal_client is None:
        job.emit({"status": "causal", "message": "CRISP skipped: causal service not configured",
                  "progress_percent": 91})
        return {"success": False, "error": "causal service client not initialized (is causal_service running?)"}

    cache_key = _crisp_cache_key(request, group_column, factor_values)
    with _crisp_cache.key_lock(cache_key):
        cached = _crisp_cache.get(cache_key) if causal.reuse else None
        if cached:
            logger.info(f"Pipeline {pipeline_id}: reusing CRISP result {cache_key}")
            result = cached
        else:
            models = ", ".join(causal.models)
            job.emit({"status": "causal",
                      "message": f"Running CRISP ({models}; environments = {causal.environment_column}). "
                                 f"This can take a while...", "progress_percent": 91})
            df = data_client.get_dataset(dataset_id)
            try:
                df, env_note = _add_environment_column(df, causal.environment_column,
                                                       dataset_chain or [dataset_id], pipeline_id, study_datasets)
            except ValueError as e:
                return {"success": False, "error": str(e)}
            df = _decode_condition(df, group_column, factor_values, pipeline_id)
            result = causal_client.run_crisp(
                df, analysis_id=f"{pipeline_id[:8]}_{cache_key[-6:]}", group_column=group_column,
                control_value=factor_values[0], treatment_value=factor_values[1],
                environment_column=causal.environment_column, models=causal.models,
                max_features=causal.max_features, seed=causal.seed, test_fraction=causal.test_fraction,
                val_fraction=causal.val_fraction, ranking_method=causal.ranking_method, top_k=causal.top_k,
                model_options=causal.model_options, timeout_minutes=causal.timeout_minutes)
            result = {**result, "cache_key": cache_key, "source_pipeline_id": pipeline_id, "reused": False}
            if env_note and result.get("success"):
                result["notes"] = [env_note] + list(result.get("notes", []))
            if result.get("success"):
                _crisp_cache.put(cache_key, result)

    if result.get("success"):
        n_cons = sum(1 for g in result["ranking"] if g["consensus"])
        how = (f" (reused from run {result['source_pipeline_id'][:8]}: same data and CRISP settings)"
               if result.get("reused") else f" in {result.get('runtime_seconds', 0) / 60:.1f} min")
        n_used = len([m for m in result["methods"] if m["causal"] and m["coefficients"]])
        msg = (f"CRISP: {result.get('n_significant', 0)} significant (ICP/NLICP), {len(result['ranking'])} ranked, "
               f"{n_cons} chosen by a majority of {n_used} method{'s' if n_used != 1 else ''}{how}")
        for note in result.get("notes", []):
            logger.info(f"Pipeline {pipeline_id}: CRISP note: {note}")
    else:
        msg = f"CRISP failed: {result.get('error')}"
        logger.warning(f"Pipeline {pipeline_id}: {msg}")
    logger.info(f"Pipeline {pipeline_id}: {msg}")
    job.emit({"status": "causal", "message": msg, "progress_percent": 95})
    return result


def _execute_pipeline(job: PipelineJob):
    """Runs in a worker thread. Same steps as before; progress goes to job.emit()."""
    pipeline_id = job.pipeline_id
    request = job.request
    try:
        job.emit({"status": "starting", "message": "Pipeline started", "progress_percent": 1})

        # ── Download and combine OSD studies (was done before the response started) ──
        transformed_id = request.dataset_id
        dataset_chain = [transformed_id] if transformed_id else []   # every dataset this run produced, in order
        study_datasets: Dict[str, str] = {}                           # OSD id -> that study's downloaded dataset
        if request.osd_ids and not request.dataset_id:
            osd_ids_list = [i.strip() for i in request.osd_ids.split(',') if i.strip()]
            logger.info(f"Pipeline {pipeline_id}: Downloading and combining OSD datasets {osd_ids_list}...")
            job.emit({"status": "downloading",
                      "message": f"Downloading {len(osd_ids_list)} OSDR stud{'y' if len(osd_ids_list) == 1 else 'ies'}: "
                                 f"{', '.join('OSD-' + i for i in osd_ids_list)} (first time can take a while)...",
                      "progress_percent": 3})

            data_channel = grpc.insecure_channel(
                get_settings().data_service_url,
                options=[('grpc.max_send_message_length', 500 * 1024 * 1024),
                         ('grpc.max_receive_message_length', 500 * 1024 * 1024)],
            )
            try:
                data_stub = data_service_pb2_grpc.MultiDatasetServiceStub(data_channel)
                download_request = data_service_pb2.DownloadMultipleDatasetsRequest(
                    osd_ids=osd_ids_list,
                    patterns=DOWNLOAD_PATTERNS,
                    factor_name=request.config.factor_name or 'Factor Value[Spaceflight]',
                    factor_values=request.config.factor_values or ['Ground Control', 'Space Flight'],
                    min_features=request.config.min_features or 1000,
                    cv_step=DOWNLOAD_CV_STEP
                )
                download_response = data_stub.DownloadMultipleDatasets(download_request)
                if not download_response.success:
                    raise RuntimeError(f"Failed to download: {download_response.error_message}")
                study_datasets = {str(k): str(v) for k, v in download_response.dataset_ids.items()}

                job.emit({"status": "downloading",
                          "message": f"Downloaded {len(download_response.dataset_ids)} "
                                     f"stud{'y' if len(download_response.dataset_ids) == 1 else 'ies'}; combining...",
                          "progress_percent": 15})

                combine_request = data_service_pb2.CombineDatasetsRequest(
                    dataset_ids=list(download_response.dataset_ids.values()),
                    common_genes=[],  # Let service compute
                    output_name=f"combined_{pipeline_id[:8]}"
                )
                combine_response = data_stub.CombineDatasets(combine_request)
                if not combine_response.success:
                    raise RuntimeError(f"Failed to combine: {combine_response.error_message}")
            finally:
                data_channel.close()

            transformed_id = combine_response.combined_dataset_id
            dataset_chain.append(transformed_id)
            logger.info(f"✓ Datasets combined: {transformed_id}")
            job.emit({"status": "downloading", "message": "Studies combined", "progress_percent": 20})

        # ── Keep only coding genes (before CV, so min_features counts coding genes) ──
        if getattr(request.config, "coding_only", False):
            keep_biotypes = getattr(request.config, "keep_biotypes", None) or ["protein_coding"]
            logger.info(f"Pipeline {pipeline_id}: Keeping biotypes {keep_biotypes}...")
            job.emit({"status": "filtering",
                      "message": f"Keeping only {', '.join(keep_biotypes)} genes...", "progress_percent": 22})
            bt = data_client.filter_by_biotype(
                dataset_id=transformed_id,
                keep_biotypes=keep_biotypes,
                protected_columns=[c for c in {request.config.target_column, request.config.factor_name} if c],
            )
            if not bt["success"]:
                raise RuntimeError(f"Biotype filtering failed: {bt['error_message']}")
            transformed_id = bt["filtered_dataset_id"]
            dataset_chain.append(transformed_id)
            removed = ", ".join(f"{n} {k}" for k, n in sorted(bt.get("removed_by_biotype", {}).items(),
                                                              key=lambda kv: -kv[1])[:4])
            job.emit({"status": "filtering",
                      "message": f"Kept {bt['genes_after']:,} of {bt['genes_before']:,} genes"
                                 + (f" (removed {removed}{', …' if len(bt.get('removed_by_biotype', {})) > 4 else ''})"
                                    if removed else ""),
                      "progress_percent": 24})

        # ── CV-based feature filtering ──
        if request.config.min_features and request.config.min_features > 0:
            logger.info(f"Pipeline {pipeline_id}: Filtering to {request.config.min_features} features by CV...")
            job.emit({"status": "filtering", "message": "Filtering features by CV...", "progress_percent": 25})

            filter_response_dict = data_client.filter_by_cv(
                dataset_id=transformed_id,
                min_features=request.config.min_features,
                target_column=request.config.target_column
            )
            if not filter_response_dict['success']:
                raise RuntimeError(f"Feature filtering failed: {filter_response_dict['error_message']}")
            transformed_id = filter_response_dict['filtered_dataset_id']
            dataset_chain.append(transformed_id)
            logger.info(f"✓ Features filtered: {filter_response_dict['original_features']} → "
                        f"{filter_response_dict['filtered_features']}")
            job.emit({"status": "filtering",
                      "message": f"Filtered to {filter_response_dict['filtered_features']} features",
                      "progress_percent": 30})

        # ── Ensembl IDs -> gene symbols (non-fatal) ──
        logger.info(f"Pipeline {pipeline_id}: Converting Ensembl IDs to gene symbols...")
        job.emit({"status": "converting", "message": "Converting feature names...", "progress_percent": 32})
        try:
            convert_result = data_client.convert_feature_names(transformed_id)
            if convert_result["success"]:
                transformed_id = convert_result["converted_dataset_id"]
                dataset_chain.append(transformed_id)
                logger.info(f"✓ Converted {convert_result['converted_count']} features "
                            f"({convert_result['conversion_rate']:.1%})")
                job.emit({"status": "converting",
                          "message": f"Converted {convert_result['converted_count']} Ensembl IDs to gene symbols",
                          "progress_percent": 33})
            else:
                logger.warning(f"Feature conversion failed: {convert_result['error_message']}")
        except Exception as e:
            logger.warning(f"Feature conversion error: {e}")

        # ── Train the ensemble (log/standardize happen inside each model, fit on the training split) ──
        n_algos = len(request.config.ensemble_algorithms)
        cv_desc = (f"{request.config.cv_folds}-fold CV"
                   + (f" x{request.config.cv_repeats}" if request.config.cv_repeats > 1 else "")
                   + (f", GridSearchCV tuning on {request.config.cv_scoring}"
                      if request.config.tune_hyperparameters else ", no tuning"))
        logger.info(f"Pipeline {pipeline_id}: Training ensemble models ({cv_desc})...")
        job.emit({"status": "training",
                  "message": f"Training {n_algos} models ({cv_desc} on the training split): "
                             f"{', '.join(request.config.ensemble_algorithms)}...",
                  "progress_percent": 35})

        ensemble_result = None
        for progress in ml_client.train_ensemble(
            tune_hyperparameters=request.config.tune_hyperparameters,
            cv_folds=request.config.cv_folds,
            cv_repeats=request.config.cv_repeats,
            cv_scoring=request.config.cv_scoring,
            param_grids=request.config.param_grids,
            dataset_id=transformed_id,
            algorithms=request.config.ensemble_algorithms,
            target_column=request.config.target_column,
            task_type=request.config.task_type,
            feature_columns=request.config.feature_columns or [],
            hyperparameters={k: str(v) for k, v in request.config.hyperparameters.items()},
            test_size=request.config.test_size,
            random_state=request.config.random_state,
            fi_methods=request.config.fi_methods or [],
            trans_list=request.config.trans_list or "",
        ):
            ensemble_result = progress
            if progress.get("error_message"):
                logger.warning(f"Ensemble training reported: {progress['error_message']}")

        models = ((ensemble_result or {}).get("test_metrics") or {}).get("models") or []
        if not models:
            raise RuntimeError("No models were trained: "
                               + ((ensemble_result or {}).get("error_message") or "see ML service logs"))
        cv_summary = "; ".join(
            f"{m.get('algorithm')} CV F1 {m['cv_metrics']['f1_weighted_mean']:.2f}"
            f"±{m['cv_metrics'].get('f1_weighted_std', 0):.2f}"
            for m in models if (m.get("cv_metrics") or {}).get("f1_weighted_mean") is not None)
        job.emit({"status": "training",
                  "message": f"Trained {len(models)} of {n_algos} models"
                             + (f" ({cv_summary})" if cv_summary else " (no CV metrics returned)"),
                  "progress_percent": 55,
                  "test_metrics": ensemble_result.get("test_metrics")})

        # ── Feature importance, one model at a time (now with progress) ──
        logger.info(f"Pipeline {pipeline_id}: Computing feature importance...")
        fi_methods = request.config.fi_methods or ["permutation"]
        feature_importance_results = {}
        for i, model in enumerate(models, start=1):
            model_id = model["model_id"]
            job.emit({"status": "feature_importance",
                      "message": f"Feature importance {i}/{len(models)}: {model.get('algorithm', model_id)} "
                                 f"({', '.join(fi_methods)}"
                                 + (f"; significance test: {request.config.fi_n_permutations} label shuffles"
                                    if request.config.fi_n_permutations and "permutation" in fi_methods else "")
                                 + ")...",
                      "progress_percent": 55 + int(25 * (i - 1) / len(models))})
            fi_result = fi_client.compute_importance(
                model_id=model_id, dataset_id=transformed_id, methods=fi_methods,
                params={"n_permutations": request.config.fi_n_permutations, "n_repeats": request.config.fi_n_repeats,
                        "alpha": request.config.fi_alpha, "scoring": request.config.fi_scoring,
                        "random_state": request.config.random_state or 42})
            if fi_result.get("success"):
                feature_importance_results[model_id] = fi_result.get("importances", {})
            else:
                logger.warning(f"Failed to compute importance for {model_id}: {fi_result.get('error')}")

        # ── DESeq2 on the same filtered raw counts ──
        # Its input depends only on the data settings, so runs that share them (e.g. configs that
        # differ only in ML settings) reuse the first run's result instead of recomputing it.
        deseq2_result = None
        try:
            condition_col = request.config.factor_name or request.config.target_column
            factor_values = request.config.factor_values or []
            if len(factor_values) >= 2:
                cache_key = _deseq2_cache_key(request, condition_col, factor_values)
                reuse = getattr(request.config, "reuse_deseq2", True)
                with _deseq2_key_lock(cache_key):     # an identical run in parallel waits, then reuses
                    cached = _deseq2_cache_get(cache_key) if reuse else None
                    if cached:
                        deseq2_result = cached
                        logger.info(f"Pipeline {pipeline_id}: reusing DESeq2 result {cache_key} "
                                    f"from pipeline {cached.get('source_pipeline_id')}")
                    else:
                        logger.info(f"Pipeline {pipeline_id}: Running DESeq2 analysis ({cache_key})...")
                        job.emit({"status": "deseq2", "message": "Running DESeq2 differential expression analysis...",
                                  "progress_percent": 85})
                        deseq2_result = bio_client.run_deseq2(
                            dataset_id=transformed_id,
                            condition_column=condition_col,
                            control_group=factor_values[0],
                            treatment_group=factor_values[1],
                            padj_threshold=DESEQ2_PADJ_THRESHOLD,
                            log2fc_threshold=DESEQ2_LOG2FC_THRESHOLD
                        )
                        if deseq2_result.get("success"):
                            deseq2_result = {**deseq2_result, "cache_key": cache_key,
                                             "source_pipeline_id": pipeline_id, "reused": False}
                            _deseq2_cache_put(cache_key, deseq2_result)
                if deseq2_result.get("success"):
                    sig = deseq2_result.get("num_significant", 0)
                    up = deseq2_result.get("num_upregulated", 0)
                    down = deseq2_result.get("num_downregulated", 0)
                    how = (f" (reused from run {deseq2_result['source_pipeline_id'][:8]}: same data settings)"
                           if deseq2_result.get("reused") else "")
                    logger.info(f"✓ DESeq2 complete: {sig} significant genes ({up} up, {down} down){how}")
                    job.emit({"status": "deseq2",
                              "message": f"DESeq2: {sig} significant genes ({up} up, {down} down){how}",
                              "progress_percent": 90})
                else:
                    logger.warning(f"DESeq2 failed: {deseq2_result.get('error')}")
                    job.emit({"status": "deseq2", "message": f"DESeq2 failed: {deseq2_result.get('error')}",
                              "progress_percent": 90})
            else:
                logger.warning("Insufficient factor values for DESeq2 analysis")
        except Exception as e:
            logger.error(f"Error running DESeq2: {e}", exc_info=True)
            job.emit({"status": "deseq2", "message": f"DESeq2 error: {e}", "progress_percent": 90})

        # ── CRISP causal ensemble (optional): genes whose link to the factor holds in every study ──
        causal_result = None
        if request.config.causal.enabled:
            try:
                causal_result = _run_crisp_step(job, request, transformed_id, dataset_chain, study_datasets)
            except Exception as e:
                logger.error(f"Error running CRISP: {e}", exc_info=True)
                job.emit({"status": "causal", "message": f"CRISP error: {e}", "progress_percent": 95})
                causal_result = {"success": False, "error": str(e)}

        # ── Final result (same shape as before, so the notebook's result cells are unchanged) ──
        job.emit({
            "status": "completed",
            "message": "Pipeline completed successfully",
            "progress_percent": 100,
            "config": {
                "osd_ids": request.osd_ids,
                "algorithms": request.config.ensemble_algorithms,
                "target_column": request.config.target_column,
                "test_size": request.config.test_size,
                "min_features": request.config.min_features,
                "random_state": request.config.random_state,
                "trans_list": request.config.trans_list,
                "tune_hyperparameters": request.config.tune_hyperparameters,
                "cv_folds": request.config.cv_folds,
                "cv_repeats": request.config.cv_repeats,
                "cv_scoring": request.config.cv_scoring,
                "param_grids": request.config.param_grids,
                "fi_methods": fi_methods,
                "fi_n_permutations": request.config.fi_n_permutations, "fi_n_repeats": request.config.fi_n_repeats,
                "fi_alpha": request.config.fi_alpha, "fi_scoring": request.config.fi_scoring,
                "causal": (request.config.causal.model_dump() if hasattr(request.config.causal, "model_dump")
                           else request.config.causal.dict()),
            },
            "training_results": ensemble_result or {},
            "feature_importance": feature_importance_results,
            "deseq2_results": deseq2_result or {},
            "causal_results": causal_result or {},
            "transformed_dataset_id": transformed_id
        })
        logger.info(f"Pipeline {pipeline_id}: completed")

    except Exception as e:
        logger.error(f"Pipeline {pipeline_id} error: {e}", exc_info=True)
        job.emit({"status": "failed", "message": "Pipeline execution failed", "error": str(e)})


def _get_job(pipeline_id: str) -> PipelineJob:
    with _jobs_lock:
        job = _jobs.get(pipeline_id)
    if job is None:
        raise HTTPException(status_code=404,
                            detail=f"Unknown pipeline {pipeline_id} (finished runs are kept "
                                   f"{JOB_TTL_SECONDS // 3600} h, and none survive a service restart)")
    return job


@app.post("/api/pipeline/jobs")
async def start_pipeline_job(request: PipelineRequest):
    """Start a pipeline run in the background. Poll GET /api/pipeline/jobs/{pipeline_id} for progress."""
    job = _start_job(request)
    return {"pipeline_id": job.pipeline_id, "status": job.status}


@app.get("/api/pipeline/jobs/{pipeline_id}")
async def get_pipeline_job(pipeline_id: str, since: int = 0):
    """Status of a run. Pass `since` = the previous response's `next_since` to get only new events."""
    return _get_job(pipeline_id).snapshot(since=max(0, since))


@app.get("/api/pipeline/jobs")
async def list_pipeline_jobs(limit: int = 20):
    """Recent runs, newest first (without their results)."""
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
    out = []
    for j in jobs:
        s = j.snapshot(since=len(j.events))
        s.pop("result"); s.pop("events")
        s["osd_ids"] = j.request.osd_ids
        s["algorithms"] = j.request.config.ensemble_algorithms
        out.append(s)
    return {"jobs": out}


@app.post("/api/pipeline/run")
async def run_pipeline(request: PipelineRequest):
    """Run a pipeline and stream its progress as NDJSON (kept for older notebooks).

    The run itself is a background job: if the connection drops, it keeps going and can be
    fetched with GET /api/pipeline/jobs/{pipeline_id} (the id is in every streamed line).
    """
    job = _start_job(request)

    async def stream():
        sent = 0
        last_sent = time.monotonic()
        while True:
            snap = job.snapshot(since=sent)
            for event in snap["events"]:
                yield json.dumps(event) + "\n"
            if snap["events"]:
                sent = snap["next_since"]
                last_sent = time.monotonic()
            if job.finished and sent >= len(job.events):
                return
            if time.monotonic() - last_sent > KEEPALIVE_SECONDS:
                yield "\n"   # blank line: keeps proxies/tunnels/client read timeouts happy; clients skip it
                last_sent = time.monotonic()
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ── GO enrichment ─────────────────────────────────────────────────────────────

@app.post("/api/enrichment")
def run_enrichment(req: EnrichmentRunRequest):
    """GO enrichment on gene lists (ORA) and rankings (GSEA), as selected in req.config.

    A plain (non-async) endpoint, so FastAPI runs it in a worker thread and the R call
    doesn't block other requests.
    """
    cfg = req.config
    if cfg.min_gs_size > cfg.max_gs_size:
        raise HTTPException(400, f"min_gs_size ({cfg.min_gs_size}) is larger than max_gs_size ({cfg.max_gs_size})")
    if not req.universe:
        raise HTTPException(400, "universe is empty: pass every gene the pipeline analysed")

    gene_lists = {}
    if "ora" in cfg.methods:
        missing = [n for n in cfg.ora_lists if n not in req.gene_lists]
        if missing:
            raise HTTPException(400, f"Config asks for ORA on {missing}, but those lists weren't sent "
                                     f"(sent: {sorted(req.gene_lists)})")
        gene_lists = {n: req.gene_lists[n] for n in cfg.ora_lists}

    rankings = {}
    if "gsea" in cfg.methods:
        missing = [n for n in cfg.gsea_rankings if n not in req.rankings]
        if missing:
            raise HTTPException(400, f"Config asks for GSEA on {missing}, but those rankings weren't sent "
                                     f"(sent: {sorted(req.rankings)})")
        for n in cfg.gsea_rankings:
            r = req.rankings[n]
            genes = list(r.scores)
            rankings[n] = {"genes": genes, "scores": [r.scores[g] for g in genes], "score_type": r.score_type}

    if not gene_lists and not rankings:
        raise HTTPException(400, "Nothing to run: config.methods selects no lists or rankings")

    organism = detect_organism(req.universe) if cfg.organism == "auto" else cfg.organism
    analysis_id = req.analysis_id or uuid.uuid4().hex[:12]
    logger.info(f"GO enrichment {analysis_id}: organism={organism} ora={list(gene_lists)} "
                f"gsea={list(rankings)} ontologies={cfg.ontologies}")

    result = bio_client.run_go_enrichment(
        gene_lists=gene_lists, rankings=rankings, universe=req.universe, organism=organism,
        ontologies=cfg.ontologies, p_adjust_method=cfg.p_adjust_method,
        padj_cutoff=cfg.padj_cutoff, qvalue_cutoff=cfg.qvalue_cutoff,
        min_gs_size=cfg.min_gs_size, max_gs_size=cfg.max_gs_size,
        simplify=cfg.simplify, simplify_cutoff=cfg.simplify_cutoff, analysis_id=analysis_id,
    )
    if not result.get("success"):
        raise HTTPException(502, f"GO enrichment failed in the bioinformatics service: {result.get('error')}")
    return {"analysis_id": analysis_id, "organism": organism, **result}


# ── Model endpoints ───────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models(
    algorithm: str = None,
    task_type: str = None,
    limit: int = 10
):
    """List all trained models"""
    try:
        result = ml_client.list_models(
            algorithm=algorithm,
            task_type=task_type,
            limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    try:
        result = ml_client.get_model_info(model_id)
        return result
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")

@app.get("/api/config/defaults")
async def config_defaults():
    """Every pipeline setting with its default value (including the enrichment block)."""
    cfg = PipelineConfig()
    return cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "ML Pipeline Orchestration Service",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
