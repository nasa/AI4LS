# orchestration-service/src/main.py (update the relevant sections)
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
import uuid
import json
from contextlib import asynccontextmanager

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global clients
data_client: DataServiceClient = None
ml_client: MLServiceClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    global data_client, ml_client
    
    settings = get_settings()
    
    # Initialize gRPC clients
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
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if data_client:
        data_client.close()
    if ml_client:
        ml_client.close()

# Create FastAPI app
app = FastAPI(
    title="ML Pipeline Orchestration Service",
    description="REST API for managing ML pipeline workflows",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of orchestration service and downstream services"""
    services_status = {
        "data_service": False,
        "ml_service": False,
        "metrics_service": False
    }
    
    # Check Data Service
    try:
        if data_client:
            services_status["data_service"] = data_client.health_check()
    except Exception as e:
        logger.error(f"Data Service health check failed: {e}")
    
    # Check ML Service
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

# ... (keep all the dataset endpoints from before) ...

@app.post("/api/pipeline/run")
async def run_pipeline(request: PipelineRequest):
    """
    Run a complete ML pipeline with streaming progress
    
    This endpoint orchestrates the full workflow:
    1. Apply transformations (if any)
    2. Train the model (with streaming progress)
    3. Return final results
    """
    pipeline_id = str(uuid.uuid4())
    
    async def generate_progress():
        """Generator function for streaming progress"""
        try:
            # Step 1: Apply transformations
            transformed_id = request.dataset_id
            
            if request.config.transformations:
                logger.info(f"Pipeline {pipeline_id}: Applying transformations...")
                
                yield json.dumps({
                    "pipeline_id": pipeline_id,
                    "status": "transforming",
                    "message": "Applying data transformations...",
                    "progress_percent": 10
                }) + "\n"
                
                transformations = [
                    {
                        "type": t.type.value,
                        "columns": t.columns,
                        "params": t.params
                    }
                    for t in request.config.transformations
                ]
                
                transform_result = data_client.apply_transformations(
                    request.dataset_id, 
                    transformations
                )
                
                if not transform_result["success"]:
                    yield json.dumps({
                        "pipeline_id": pipeline_id,
                        "status": "failed",
                        "message": "Transformation failed",
                        "error": transform_result["error_message"]
                    }) + "\n"
                    return
                
                transformed_id = transform_result["transformed_dataset_id"]
                logger.info(f"Pipeline {pipeline_id}: Transformations complete")
                
                yield json.dumps({
                    "pipeline_id": pipeline_id,
                    "status": "transforming",
                    "message": "Transformations completed",
                    "progress_percent": 30,
                    "transformed_dataset_id": transformed_id
                }) + "\n"
            
            # Step 2: Train model with streaming progress
            logger.info(f"Pipeline {pipeline_id}: Training model...")
            
            model_id = None
            final_metrics = None
            
            for progress in ml_client.train_model(
                dataset_id=transformed_id,
                algorithm=request.config.algorithm.value,
                target_column=request.config.target_column,
                task_type=request.config.task_type,
                feature_columns=request.config.feature_columns or [],
                hyperparameters={k: str(v) for k, v in request.config.hyperparameters.items()},
                test_size=request.config.test_size,
                random_state=request.config.random_state
            ):
                # Map ML service progress (0-100) to pipeline progress (30-100)
                pipeline_progress = 30 + int(progress["progress_percent"] * 0.7)
                
                yield json.dumps({
                    "pipeline_id": pipeline_id,
                    "status": progress["status"],
                    "message": progress["message"],
                    "progress_percent": pipeline_progress,
                    "model_id": progress["model_id"],
                    "training_metrics": progress.get("training_metrics"),
                    "test_metrics": progress.get("test_metrics"),
                    "error": progress.get("error_message")
                }) + "\n"
                
                if progress["status"] == "completed":
                    model_id = progress["model_id"]
                    final_metrics = progress.get("test_metrics", {})
                elif progress["status"] == "failed":
                    return
            
            # Final response
            yield json.dumps({
                "pipeline_id": pipeline_id,
                "status": "completed",
                "message": "Pipeline completed successfully",
                "progress_percent": 100,
                "dataset_id": request.dataset_id,
                "transformed_dataset_id": transformed_id,
                "model_id": model_id,
                "metrics": final_metrics
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

# New endpoint: List trained models
@app.get("/api/models")
async def list_models(
    algorithm: str = None,
    task_type: str = None,
    limit: int = 10
):
    """
    List all trained models
    
    - **algorithm**: Filter by algorithm (optional)
    - **task_type**: Filter by task type (optional)
    - **limit**: Maximum number of models to return
    """
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

# New endpoint: Get model info
@app.get("/api/models/{model_id}")
async def get_model_info(model_id: str):
    """
    Get information about a specific model
    
    - **model_id**: ID of the trained model
    """
    try:
        result = ml_client.get_model_info(model_id)
        return result
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")

# ... (keep the root endpoint from before) ...

'''# src/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uuid
from contextlib import asynccontextmanager

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global clients
data_client: DataServiceClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for the application"""
    global data_client
    
    settings = get_settings()
    
    # Initialize gRPC clients
    logger.info("Initializing gRPC clients...")
    try:
        data_client = DataServiceClient(settings.data_service_url)
        logger.info("✓ Data Service client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Data Service client: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if data_client:
        data_client.close()

# Create FastAPI app
app = FastAPI(
    title="ML Pipeline Orchestration Service",
    description="REST API for managing ML pipeline workflows",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of orchestration service and downstream services"""
    services_status = {
        "data_service": False,
        "ml_service": False,  # Will implement later
        "metrics_service": False  # Will implement later
    }
    
    # Check Data Service
    try:
        if data_client:
            services_status["data_service"] = data_client.health_check()
    except Exception as e:
        logger.error(f"Data Service health check failed: {e}")
    
    overall_status = "healthy" if services_status["data_service"] else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        services=services_status
    )

# Dataset endpoints
@app.post("/api/datasets/validate", response_model=ValidationResponse)
async def validate_dataset(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    """
    Validate an uploaded dataset
    
    - **file**: CSV or JSON file to validate
    """
    # Check file size
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size / 1024 / 1024}MB"
        )
    
    # Determine format from filename
    format_type = "csv" if file.filename.endswith(".csv") else "json"
    
    try:
        result = data_client.validate_dataset(content, format_type)
        
        # Convert to response model
        response = ValidationResponse(
            is_valid=result["is_valid"],
            errors=result["errors"],
            warnings=result["warnings"]
        )
        
        if result["dataset_info"]:
            info = result["dataset_info"]
            response.dataset_info = DatasetInfo(
                dataset_id=info["dataset_id"],
                num_rows=info["num_rows"],
                num_columns=info["num_columns"],
                size_bytes=info["size_bytes"],
                columns=[
                    ColumnInfo(**col) for col in info["columns"]
                ]
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/api/datasets/{dataset_id}/transform", response_model=TransformationResponse)
async def transform_dataset(
    dataset_id: str,
    request: TransformationRequest
):
    """
    Apply transformations to a dataset
    
    - **dataset_id**: ID of the dataset to transform
    - **transformations**: List of transformations to apply
    """
    try:
        # Convert request to dict format for gRPC client
        transformations = [
            {
                "type": t.type.value,
                "columns": t.columns,
                "params": t.params
            }
            for t in request.transformations
        ]
        
        result = data_client.apply_transformations(dataset_id, transformations)
        
        # Convert to response model
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
                columns=[
                    ColumnInfo(**col) for col in info["columns"]
                ]
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Transformation error: {e}")
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")

@app.get("/api/datasets/{dataset_id}", response_model=DatasetInfo)
async def get_dataset_info(dataset_id: str):
    """
    Get information about a dataset
    
    - **dataset_id**: ID of the dataset
    """
    try:
        result = data_client.get_dataset_info(dataset_id)
        
        return DatasetInfo(
            dataset_id=result["dataset_id"],
            num_rows=result["num_rows"],
            num_columns=result["num_columns"],
            size_bytes=result["size_bytes"],
            columns=[
                ColumnInfo(**col) for col in result["columns"]
            ]
        )
        
    except Exception as e:
        logger.error(f"Get dataset info error: {e}")
        raise HTTPException(status_code=404, detail=f"Dataset not found: {str(e)}")

@app.post("/api/pipeline/run", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest):
    """
    Run a complete ML pipeline
    
    This endpoint orchestrates the full workflow:
    1. Apply transformations (if any)
    2. Train the model
    3. Evaluate with metrics
    
    - **dataset_id**: ID of the validated dataset
    - **config**: Pipeline configuration
    """
    pipeline_id = str(uuid.uuid4())
    
    try:
        # Step 1: Apply transformations
        transformed_id = request.dataset_id
        
        if request.config.transformations:
            logger.info(f"Pipeline {pipeline_id}: Applying transformations...")
            
            transformations = [
                {
                    "type": t.type.value,
                    "columns": t.columns,
                    "params": t.params
                }
                for t in request.config.transformations
            ]
            
            transform_result = data_client.apply_transformations(
                request.dataset_id, 
                transformations
            )
            
            if not transform_result["success"]:
                return PipelineResponse(
                    pipeline_id=pipeline_id,
                    status=PipelineStatus.FAILED,
                    message="Transformation failed",
                    error=transform_result["error_message"]
                )
            
            transformed_id = transform_result["transformed_dataset_id"]
            logger.info(f"Pipeline {pipeline_id}: Transformations complete")
        
        # Step 2: Train model (placeholder - will implement with ML Service)
        logger.info(f"Pipeline {pipeline_id}: Training model...")
        # TODO: Call ML Service to train model
        model_id = f"model_{uuid.uuid4().hex[:8]}"
        
        # Step 3: Evaluate model (placeholder - will implement with Metrics Service)
        logger.info(f"Pipeline {pipeline_id}: Evaluating model...")
        # TODO: Call Metrics Service to evaluate
        metrics = {
            "accuracy": 0.85,  # Placeholder
            "f1_score": 0.82   # Placeholder
        }
        
        return PipelineResponse(
            pipeline_id=pipeline_id,
            status=PipelineStatus.COMPLETED,
            message="Pipeline completed successfully",
            dataset_id=request.dataset_id,
            transformed_dataset_id=transformed_id,
            model_id=model_id,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Pipeline {pipeline_id} error: {e}")
        return PipelineResponse(
            pipeline_id=pipeline_id,
            status=PipelineStatus.FAILED,
            message="Pipeline execution failed",
            error=str(e)
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
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
    )'''
