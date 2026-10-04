import grpc
import logging
from typing import Dict, List
from src.generated.feature_importance_service_pb2 import ImportanceRequest
from src.generated.feature_importance_service_pb2_grpc import FeatureImportanceServiceStub

logger = logging.getLogger(__name__)

class FeatureImportanceClient:
    """Client for feature importance service"""
    
    def __init__(self, service_url: str):
        self.service_url = service_url
        self.channel = grpc.insecure_channel(service_url)
        self.stub = FeatureImportanceServiceStub(self.channel)
        logger.info(f"Feature Importance client initialized: {service_url}")
    
    def compute_importance(
        self,
        model_id: str,
        dataset_id: str,
        methods: List[str] = None
    ) -> Dict:
        """Compute feature importance for a model"""
        try:
            request = ImportanceRequest(
                model_id=model_id,
                dataset_id=dataset_id,
                methods=methods or ["permutation"]
            )
            
            response = self.stub.ComputeImportance(request)
            
            if response.success:
                # Convert protobuf to dict
                importances = {}
                for method, importance_data in response.importances.items():
                    features = []
                    for score in importance_data.scores:
                        features.append({
                            "feature_name": score.feature_name,
                            "importance": float(score.importance),
                            "rank": int(score.rank)
                        })
                    importances[method] = {"features": features}
                
                return {
                    "success": True,
                    "importances": importances,
                    "model_id": model_id
                }
            else:
                return {
                    "success": False,
                    "error": response.error_message,
                    "model_id": model_id
                }
        
        except Exception as e:
            logger.error(f"Error computing importance: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_id": model_id
            }
    
    def close(self):
        self.channel.close()
