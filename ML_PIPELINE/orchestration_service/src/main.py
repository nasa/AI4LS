# orchestration-service/src/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import logging
import os
import threading
import time
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import grpc

from src.config import Settings, get_settings
from src.models import (
    ValidationResponse,
    TransformationRequest,
    TransformationResponse,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    global data_client, ml_client, fi_client, bio_client
    
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of orchestration service and downstream services"""
    services_status = {
        "data_service": False,
        "ml_service": False,
        "feature_importance_service": False
    }
    
    try:
        if data_client:
            services_status["data_service"] = data_client.health_check()
    except Exception as e:
        logger.error(f"Data Service health check failed: {e}")
    
    try:
        if ml_client:
            services_status["ml_service"] = ml_client.health_check()
    except Exception as e:
        logger.error(f"ML Service health check failed: {e}")
    
    overall_status = "healthy" if all([
        services_status["data_service"],
        services_status["ml_service"]
    ]) else "degraded"
    
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


def _execute_pipeline(job: PipelineJob):
    """Runs in a worker thread. Same steps as before; progress goes to job.emit()."""
    pipeline_id = job.pipeline_id
    request = job.request
    try:
        job.emit({"status": "starting", "message": "Pipeline started", "progress_percent": 1})

        # ── Download and combine OSD studies (was done before the response started) ──
        transformed_id = request.dataset_id
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
                    patterns=['unnormalized'],
                    factor_name=request.config.factor_name or 'Factor Value[Spaceflight]',
                    factor_values=request.config.factor_values or ['Ground Control', 'Space Flight'],
                    min_features=request.config.min_features or 1000,
                    cv_step=0.25
                )
                download_response = data_stub.DownloadMultipleDatasets(download_request)
                if not download_response.success:
                    raise RuntimeError(f"Failed to download: {download_response.error_message}")

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
            logger.info(f"✓ Datasets combined: {transformed_id}")
            job.emit({"status": "downloading", "message": "Studies combined", "progress_percent": 20})

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
        logger.info(f"Pipeline {pipeline_id}: Training ensemble models...")
        job.emit({"status": "training",
                  "message": f"Training {n_algos} models: {', '.join(request.config.ensemble_algorithms)}...",
                  "progress_percent": 35})

        ensemble_result = None
        for progress in ml_client.train_ensemble(
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
        job.emit({"status": "training",
                  "message": f"Trained {len(models)} of {n_algos} models",
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
                                 f"({', '.join(fi_methods)})...",
                      "progress_percent": 55 + int(25 * (i - 1) / len(models))})
            fi_result = fi_client.compute_importance(model_id=model_id, dataset_id=transformed_id, methods=fi_methods)
            if fi_result.get("success"):
                feature_importance_results[model_id] = fi_result.get("importances", {})
            else:
                logger.warning(f"Failed to compute importance for {model_id}: {fi_result.get('error')}")

        # ── DESeq2 on the same filtered raw counts ──
        logger.info(f"Pipeline {pipeline_id}: Running DESeq2 analysis...")
        job.emit({"status": "deseq2", "message": "Running DESeq2 differential expression analysis...",
                  "progress_percent": 85})
        deseq2_result = None
        try:
            condition_col = request.config.factor_name or request.config.target_column
            factor_values = request.config.factor_values or []
            if len(factor_values) >= 2:
                deseq2_result = bio_client.run_deseq2(
                    dataset_id=transformed_id,
                    condition_column=condition_col,
                    control_group=factor_values[0],
                    treatment_group=factor_values[1],
                    padj_threshold=0.05,
                    log2fc_threshold=0.0
                )
                if deseq2_result.get("success"):
                    sig = deseq2_result.get("num_significant", 0)
                    up = deseq2_result.get("num_upregulated", 0)
                    down = deseq2_result.get("num_downregulated", 0)
                    logger.info(f"✓ DESeq2 complete: {sig} significant genes ({up} up, {down} down)")
                    job.emit({"status": "deseq2",
                              "message": f"DESeq2: {sig} significant genes ({up} up, {down} down)",
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
                "min_features": request.config.min_features
            },
            "training_results": ensemble_result or {},
            "feature_importance": feature_importance_results,
            "deseq2_results": deseq2_result or {},
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
