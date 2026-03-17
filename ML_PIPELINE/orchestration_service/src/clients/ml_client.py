# orchestration-service/src/clients/ml_client.py
import grpc
from typing import Dict, List, Iterator
import sys
from pathlib import Path
import logging

# Add path to ml-service generated code
#ml_service_path = Path(__file__).parent.parent.parent.parent / "ml-service" / "generated"
#sys.path.insert(0, str(ml_service_path))



from generated.ml_service_pb2 import (
    TrainRequest, ModelInfoRequest, PredictRequest, 
    ListModelsRequest, TrainingProgress
)
from generated.ml_service_pb2_grpc import MLServiceStub

logger = logging.getLogger(__name__)

class MLServiceClient:
    """Client for ML Service gRPC communication"""
    
    def __init__(self, service_url: str):
        self.service_url = service_url
        #self.channel = grpc.insecure_channel(service_url)
        self.channel = grpc.insecure_channel(
           self.service_url,
              options=[
                 ('grpc.max_send_message_length', 50 * 1024 * 1024),
                 ('grpc.max_receive_message_length', 50 * 1024 * 1024),
              ]
        )
        self.stub = MLServiceStub(self.channel)
        logger.info(f"Connected to ML Service at {service_url}")
    
    def train_model(
        self,
        dataset_id: str,
        algorithm: str,
        target_column: str,
        task_type: str,
        feature_columns: List[str] = None,
        hyperparameters: Dict[str, str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Iterator[Dict]:
        """Train a model with streaming progress"""
        try:
            request = TrainRequest(
                dataset_id=dataset_id,
                algorithm=algorithm,
                hyperparameters=hyperparameters or {},
                target_column=target_column,
                feature_columns=feature_columns or [],
                test_size=test_size,
                random_state=random_state,
                task_type=task_type
            )
            
            for progress in self.stub.TrainModel(request):
                yield {
                    "model_id": progress.model_id,
                    "status": progress.status,
                    "message": progress.message,
                    "progress_percent": progress.progress_percent,
                    "training_metrics": dict(progress.training_metrics) if progress.training_metrics else None,
                    "test_metrics": dict(progress.test_metrics) if progress.test_metrics else None,
                    "error_message": progress.error_message if progress.error_message else None
                }
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error in train_model: {e.code()} - {e.details()}")
            raise
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a trained model"""
        try:
            request = ModelInfoRequest(model_id=model_id)
            response = self.stub.GetModelInfo(request)
            
            return {
                "model_id": response.model_id,
                "algorithm": response.algorithm,
                "task_type": response.task_type,
                "dataset_id": response.dataset_id,
                "target_column": response.target_column,
                "feature_columns": list(response.feature_columns),
                "num_samples": response.num_samples,
                "num_features": response.num_features,
                "hyperparameters": dict(response.hyperparameters),
                "created_at": response.created_at,
                "training_metrics": dict(response.training_metrics),
                "test_metrics": dict(response.test_metrics)
            }
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error in get_model_info: {e.code()} - {e.details()}")
            raise
    
    def list_models(
        self,
        algorithm: str = None,
        task_type: str = None,
        limit: int = 10
    ) -> Dict:
        """List trained models"""
        try:
            request = ListModelsRequest(
                algorithm=algorithm or "",
                task_type=task_type or "",
                limit=limit
            )
            
            response = self.stub.ListModels(request)
            
            models = []
            for model in response.models:
                models.append({
                    "model_id": model.model_id,
                    "algorithm": model.algorithm,
                    "task_type": model.task_type,
                    "dataset_id": model.dataset_id,
                    "target_column": model.target_column,
                    "num_samples": model.num_samples,
                    "num_features": model.num_features,
                    "created_at": model.created_at,
                    "test_metrics": dict(model.test_metrics)
                })
            
            return {
                "models": models,
                "total_count": response.total_count
            }
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error in list_models: {e.code()} - {e.details()}")
            raise
    
    def health_check(self) -> bool:
        """Check if ML Service is healthy"""
        try:
            # Try to list models as a health check
            request = ListModelsRequest(limit=1)
            self.stub.ListModels(request, timeout=2)
            return True
        except:
            return False
    
    def close(self):
        """Close the gRPC channel"""
        self.channel.close()
        logger.info("ML Service connection closed")
