# feature_importance_service/src/service.py
import grpc
import sys
from pathlib import Path
import logging
import json
from typing import Dict
import time

# Add ml_service path to import model_store
ml_service_path = Path(__file__).parent.parent.parent / "ml_service"
sys.path.insert(0, str(ml_service_path))

from src.model_store import ModelStore
from src.data_client import DataServiceClient

from generated import feature_importance_service_pb2, feature_importance_service_pb2_grpc
from src.importance_methods import FeatureImportanceMethods

logger = logging.getLogger(__name__)

class FeatureImportanceServiceImpl(feature_importance_service_pb2_grpc.FeatureImportanceServiceServicer):
    """gRPC service for computing feature importance"""
    
    def __init__(self, data_service_url: str = "data_service:50051"):
        # Access ML service's model store (shared volume in Docker)
        self.model_store = ModelStore(base_path="/app/models")
        self.data_client = DataServiceClient(data_service_url)
        self.importance_methods = FeatureImportanceMethods()
        
        # Cache computed importances
        self.importance_cache: Dict[str, Dict] = {}
        
        logger.info("FeatureImportanceService initialized")
    
    def ComputeImportance(self, request, context):
        """Compute feature importance for a trained model"""
        try:
            model_id = request.model_id
            dataset_id = request.dataset_id
            methods = list(request.methods) if request.methods else ["built_in"]
            params = dict(request.params)
            
            logger.info(f"Computing importance for model {model_id} using methods: {methods}")
            
            # Load the trained model
            model = self.model_store.load_model(model_id)
            if model is None:
                return feature_importance_service_pb2.ImportanceResponse(
                    success=False,
                    model_id=model_id,
                    error_message=f"Model {model_id} not found"
                )
            
            # Get model metadata
            model_info = self.model_store.get_model_info(model_id)
            feature_names = model_info["feature_columns"]
            
            # Get dataset from data service
            df = self.data_client.get_dataset(dataset_id)
            if df is None:
                return feature_importance_service_pb2.ImportanceResponse(
                    success=False,
                    model_id=model_id,
                    error_message=f"Dataset {dataset_id} not found"
                )
            
            # Prepare features and target
            X = df[feature_names]
            y = df[model_info["target_column"]]
            
            # Compute importance for each requested method
            all_importances = {}
            
            for method in methods:
                start_time = time.time()
                
                if method == "built_in":
                    scores = self.importance_methods.built_in_importance(model, feature_names)
                    metadata = {"execution_time": f"{time.time() - start_time:.2f}s"}
                
                elif method == "recursive":
                    n_features = int(params.get("n_features_to_select", len(feature_names) // 2))
                    step = int(params.get("step", 1))
                    scores = self.importance_methods.recursive_feature_elimination(
                        model, X, y, n_features_to_select=n_features, step=step
                    )
                    metadata = {
                        "execution_time": f"{time.time() - start_time:.2f}s",
                        "n_features_selected": str(n_features)
                    }
                
                elif method == "permutation":
                    n_repeats = int(params.get("n_repeats", 10))
                    scores = self.importance_methods.permutation_feature_importance(
                        model, X, y, n_repeats=n_repeats, random_state=int(params.get("random_state"))
                    )
                    metadata = {
                        "execution_time": f"{time.time() - start_time:.2f}s",
                        "n_repeats": str(n_repeats)
                    }
                
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue
                
                # Convert to protobuf format
                feature_scores = []
                for score in scores:
                    feature_scores.append(
                        feature_importance_service_pb2.FeatureScore(
                            feature_name=score["feature_name"],
                            importance=score["importance"],
                            rank=score["rank"]
                        )
                    )
                
                all_importances[method] = feature_importance_service_pb2.FeatureImportances(
                    scores=feature_scores,
                    metadata=metadata
                )
            
            # Cache the results
            self.importance_cache[model_id] = {
                "importances": all_importances,
                "computed_at": time.time()
            }
            
            return feature_importance_service_pb2.ImportanceResponse(
                success=True,
                model_id=model_id,
                importances=all_importances
            )
            
        except Exception as e:
            logger.error(f"Error computing importance: {e}", exc_info=True)
            return feature_importance_service_pb2.ImportanceResponse(
                success=False,
                model_id=request.model_id,
                error_message=str(e)
            )
    
    def GetImportance(self, request, context):
        """Get previously computed importance results"""
        try:
            model_id = request.model_id
            
            if model_id not in self.importance_cache:
                return feature_importance_service_pb2.ImportanceResponse(
                    success=False,
                    model_id=model_id,
                    error_message="No cached importance results for this model"
                )
            
            cached = self.importance_cache[model_id]
            
            return feature_importance_service_pb2.ImportanceResponse(
                success=True,
                model_id=model_id,
                importances=cached["importances"]
            )
            
        except Exception as e:
            logger.error(f"Error retrieving importance: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
