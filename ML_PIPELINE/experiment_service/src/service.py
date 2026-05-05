# experiment_service/src/service.py

import grpc
import logging
import uuid

from generated import experiment_service_pb2, experiment_service_pb2_grpc
from src.experiment_store import ExperimentStore

logger = logging.getLogger(__name__)

class ExperimentServiceImpl(experiment_service_pb2_grpc.ExperimentServiceServicer):
    """gRPC service for experiment management"""
    
    def __init__(self, store_path: str = "/app/experiments"):
        self.store = ExperimentStore(store_path)
        logger.info("ExperimentService initialized")
    
    def CreateExperiment(self, request, context):
        """Create a new experiment"""
        try:
            experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
            
            experiment = self.store.create_experiment(
                experiment_id=experiment_id,
                name=request.name,
                description=request.description,
                metadata=dict(request.metadata)
            )
            
            return experiment_service_pb2.CreateExperimentResponse(
                success=True,
                experiment_id=experiment_id
            )
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}", exc_info=True)
            return experiment_service_pb2.CreateExperimentResponse(
                success=False,
                error_message=str(e)
            )
    
    def UpdateExperiment(self, request, context):
        """Update an experiment with pipeline component IDs"""
        try:
            updates = {}
            
            if request.dataset_id:
                updates["dataset_id"] = request.dataset_id
            if request.model_id:
                updates["model_id"] = request.model_id
            if request.feature_importance_id:
                updates["feature_importance_id"] = request.feature_importance_id
            if request.kegg_analysis_id:
                updates["kegg_analysis_id"] = request.kegg_analysis_id
            if request.status:
                updates["status"] = request.status
            
            success = self.store.update_experiment(request.experiment_id, **updates)
            
            if not success:
                return experiment_service_pb2.UpdateExperimentResponse(
                    success=False,
                    error_message=f"Experiment {request.experiment_id} not found"
                )
            
            return experiment_service_pb2.UpdateExperimentResponse(success=True)
            
        except Exception as e:
            logger.error(f"Error updating experiment: {e}", exc_info=True)
            return experiment_service_pb2.UpdateExperimentResponse(
                success=False,
                error_message=str(e)
            )
    
    def GetExperiment(self, request, context):
        """Get an experiment by ID"""
        try:
            experiment_data = self.store.get_experiment(request.experiment_id)
            
            if not experiment_data:
                return experiment_service_pb2.GetExperimentResponse(
                    success=False,
                    error_message=f"Experiment {request.experiment_id} not found"
                )
            
            # Convert to protobuf
            experiment = self._dict_to_proto(experiment_data)
            
            return experiment_service_pb2.GetExperimentResponse(
                success=True,
                experiment=experiment
            )
            
        except Exception as e:
            logger.error(f"Error getting experiment: {e}", exc_info=True)
            return experiment_service_pb2.GetExperimentResponse(
                success=False,
                error_message=str(e)
            )
    
    def ListExperiments(self, request, context):
        """List all experiments"""
        try:
            limit = request.limit or 100
            offset = request.offset or 0
            filter_status = request.filter_status or None
            
            experiments_data = self.store.list_experiments(
                limit=limit,
                offset=offset,
                filter_status=filter_status
            )
            
            # Convert to protobuf
            experiments = [self._dict_to_proto(exp) for exp in experiments_data]
            
            total_count = self.store.get_total_count(filter_status)
            
            return experiment_service_pb2.ListExperimentsResponse(
                success=True,
                experiments=experiments,
                total_count=total_count
            )
            
        except Exception as e:
            logger.error(f"Error listing experiments: {e}", exc_info=True)
            return experiment_service_pb2.ListExperimentsResponse(
                success=False,
                error_message=str(e)
            )
    
    def DeleteExperiment(self, request, context):
        """Delete an experiment"""
        try:
            success = self.store.delete_experiment(request.experiment_id)
            
            if not success:
                return experiment_service_pb2.DeleteExperimentResponse(
                    success=False,
                    error_message=f"Experiment {request.experiment_id} not found"
                )
            
            return experiment_service_pb2.DeleteExperimentResponse(success=True)
            
        except Exception as e:
            logger.error(f"Error deleting experiment: {e}", exc_info=True)
            return experiment_service_pb2.DeleteExperimentResponse(
                success=False,
                error_message=str(e)
            )
    
    def _dict_to_proto(self, data: dict) -> experiment_service_pb2.Experiment:
        """Convert dict to protobuf Experiment"""
        results = experiment_service_pb2.ExperimentResults(
            top_features=data.get("results", {}).get("top_features", []),
            top_pathways=data.get("results", {}).get("top_pathways", [])
        )
        
        return experiment_service_pb2.Experiment(
            experiment_id=data.get("experiment_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
            dataset_id=data.get("dataset_id", ""),
            model_id=data.get("model_id", ""),
            feature_importance_id=data.get("feature_importance_id", ""),
            kegg_analysis_id=data.get("kegg_analysis_id", ""),
            metadata=data.get("metadata", {}),
            status=data.get("status", ""),
            results=results
        )
