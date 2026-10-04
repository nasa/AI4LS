# orchestration-service/src/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import uuid
import json
from contextlib import asynccontextmanager
from typing import List, Optional
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    global data_client, ml_client, fi_client
    
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
    
    yield
    
    logger.info("Shutting down...")
    if data_client:
        data_client.close()
    if ml_client:
        ml_client.close()
    if fi_client:
        fi_client.close()

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


# ── Pipeline endpoint ─────────────────────────────────────────────────────────

@app.post("/api/pipeline/run")
async def run_pipeline(request: PipelineRequest):
    """Run a complete ML pipeline with streaming progress"""
    pipeline_id = str(uuid.uuid4())

    from fastapi import HTTPException

    # Validate that either dataset_id or osd_ids is provided
    if not request.dataset_id and not request.osd_ids:
        raise HTTPException(
            status_code=400,
            detail="Either 'dataset_id' or 'osd_ids' must be provided"
        )
    
    # If osd_ids provided, download and combine datasets first
    dataset_id = request.dataset_id
    if request.osd_ids and not request.dataset_id:
        logger.info(f"Pipeline {pipeline_id}: Downloading and combining OSD datasets...")
        try:
            # Initialize data client
            data_channel = grpc.insecure_channel('data_service:50051')
            #data_stub = data_service_pb2_grpc.DataServiceStub(data_channel)
            data_stub = data_service_pb2_grpc.MultiDatasetServiceStub(data_channel)
            
            # Download and combine
            osd_ids_list = [id.strip() for id in request.osd_ids.split(',')]
            
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
                raise HTTPException(status_code=400, detail=f"Failed to download: {download_response.error_message}")
        
            # Step 2: Combine datasets
            logger.info(f"Combining {len(download_response.dataset_ids)} datasets...")
        
            combine_request = data_service_pb2.CombineDatasetsRequest(
                dataset_ids=list(download_response.dataset_ids.values()),
                common_genes=[],  # Let service compute
                output_name=f"combined_{pipeline_id[:8]}"
            )
        
            combine_response = data_stub.CombineDatasets(combine_request)
        
            if not combine_response.success:
                raise HTTPException(status_code=400, detail=f"Failed to combine: {combine_response.error_message}")
        
            dataset_id = combine_response.combined_dataset_id
            transformed_id = dataset_id
            logger.info(f"✓ Datasets combined: {dataset_id}")
            
        except Exception as e:
            logger.error(f"Error downloading datasets: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    else:
        transformed_id = request.dataset_id
    
    # Validate ensemble algorithms
    VALID_ALGORITHMS = [
        'random_forest',
        'gradient_boosting',
        'xgboost',
        'svm',
        'neural_network',
        'logistic_regression',
        'naive_bayes'
    ]
    
    for algo in request.config.ensemble_algorithms:
        if algo not in VALID_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm: '{algo}'. Valid options are: {', '.join(VALID_ALGORITHMS)}"
            ) 
    
    async def generate_progress():
        nonlocal transformed_id
        try:
            #transformed_id = request.dataset_id
            factor_name = request.config.factor_name
            factor_values = request.config.factor_values
            min_features = request.config.min_features
            #fi_methods = request.fi_methods
            
            # Handle transformations if provided
            trans_list = request.config.trans_list
            if trans_list:
                logger.info(f"Pipeline {pipeline_id}: Applying transformations: {trans_list}")
                
                yield json.dumps({
                    "pipeline_id": pipeline_id,
                    "status": "transforming",
                    "message": f"Applying transformations: {trans_list}...",
                    "progress_percent": 10
                }) + "\n"
                
                # trans_list is comma-separated string like "s,l"
                # For now, just log it - actual transformation happens in data service
                logger.info(f"✓ Transformations: {trans_list}")
                
                yield json.dumps({
                    "pipeline_id": pipeline_id,
                    "status": "transforming",
                    "message": "Transformations completed",
                    "progress_percent": 30,
                    "transformed_dataset_id": transformed_id
                }) + "\n"

            # Apply CV-based feature filtering
            if request.config.min_features and request.config.min_features > 0:
                logger.info(f"Pipeline {pipeline_id}: Filtering to {request.config.min_features} features by CV...")
    
                yield json.dumps({
                    "pipeline_id": pipeline_id,
                    "status": "filtering",
                    "message": f"Filtering features by CV...",
                    "progress_percent": 25
                }) + "\n"
    
                try:
                    filter_response_dict = data_client.filter_by_cv( 
                        dataset_id=transformed_id,
                        min_features=request.config.min_features,
                        target_column=request.config.target_column
                    )
        
                    if filter_response_dict['success']:
                        transformed_id = filter_response_dict['filtered_dataset_id']
                        logger.info(f"✓ Features filtered: {filter_response_dict['original_features']} → {filter_response_dict['filtered_features']}")
            
                        yield json.dumps({
                            "pipeline_id": pipeline_id,
                            "status": "filtering",
                            "message": f"Filtered to {filter_response_dict['filtered_features']} features",
                            "progress_percent": 30
                        }) + "\n"
                    else:
                        logger.error(f"Feature filtering failed: {filter_response_dict['error_message']}")
                        yield json.dumps({
                            "pipeline_id": pipeline_id,
                            "status": "failed",
                            "message": f"Feature filtering failed: {filter_response_dict['error_message']}",
                            "error": filter_response_dict['error_message']
                        }) + "\n"
                        return
                except Exception as e:
                    logger.error(f"Error calling FilterByCV: {e}", exc_info=True)
                    yield json.dumps({
                        "pipeline_id": pipeline_id,
                        "status": "failed",
                        "message": "Feature filtering error",
                        "error": str(e)
                    }) + "\n"
                    return

            logger.info(f"Pipeline {pipeline_id}: Training ensemble models...")
            
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
                fi_methods=request.config.fi_methods or []
            ):
                ensemble_result = progress
    
                yield json.dumps({
                    "pipeline_id": pipeline_id,
                    "status": progress["status"],
                    "message": progress["message"],
                    "progress_percent": 30 + int(progress["progress_percent"] * 0.7),
                    "training_metrics": progress.get("training_metrics"),
                    "test_metrics": progress.get("test_metrics"),
                    "error": progress.get("error_message")
                }) + "\n"

            # Compute feature importance for each model
            logger.info(f"Pipeline {pipeline_id}: Computing feature importance...")
            feature_importance_results = {}

            if ensemble_result and ensemble_result.get("test_metrics") and ensemble_result["test_metrics"].get("models"):
                for model in ensemble_result["test_metrics"]["models"]:
                    model_id = model["model_id"]
                    logger.info(f"Computing importance for model {model_id}...")
        
                    fi_result = fi_client.compute_importance(
                        model_id=model_id,
                        dataset_id=transformed_id,
                        methods=request.config.fi_methods or ["permutation"]
                    )
        
                    if fi_result.get("success"):
                        feature_importance_results[model_id] = fi_result.get("importances", {})
                    else:
                        logger.warning(f"Failed to compute importance for {model_id}: {fi_result.get('error')}")


            # Final response with complete results
            yield json.dumps({
                "pipeline_id": pipeline_id,
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
                "transformed_dataset_id": transformed_id
            }) + "\n"
            
            
        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} error: {e}", exc_info=True)
            yield json.dumps({
                "pipeline_id": pipeline_id,
                "status": "failed",
                "message": "Pipeline execution failed",
                "error": str(e)
            }) + "\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="application/x-ndjson"
    )


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
