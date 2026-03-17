# orchestration-service/src/main.py (FastAPI example)
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from .clients.data_client import DataServiceClient

app = FastAPI()
data_client = DataServiceClient()

class TransformConfig(BaseModel):
    type: str
    columns: List[str]
    params: dict = {}

class PipelineRequest(BaseModel):
    transformations: List[TransformConfig]
    algorithm: str
    metrics: List[str]

@app.post("/api/validate-dataset")
async def validate_dataset(file: UploadFile = File(...)):
    """Validate uploaded dataset"""
    content = await file.read()
    
    result = await data_client.validate_dataset(
        dataset_content=content,
        format="csv"
    )
    
    return result

@app.post("/api/pipeline/{dataset_id}")
async def run_pipeline(dataset_id: str, config: PipelineRequest):
    """Run full ML pipeline"""
    
    # Step 1: Apply transformations
    transform_result = await data_client.apply_transformations(
        dataset_id=dataset_id,
        transformations=[t.dict() for t in config.transformations]
    )
    
    if not transform_result["success"]:
        return {"error": transform_result["error_message"]}
    
    transformed_id = transform_result["transformed_dataset_id"]
    
    # Step 2: Train model (call ML service - similar pattern)
    # ml_result = await ml_client.train_model(transformed_id, config.algorithm)
    
    # Step 3: Evaluate (call metrics service)
    # metrics_result = await metrics_client.evaluate(model_id, config.metrics)
    
    return {
        "status": "success",
        "transformed_dataset_id": transformed_id,
        "message": "Pipeline completed"
    }

@app.on_event("shutdown")
async def shutdown():
    data_client.close()
