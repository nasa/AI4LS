# experiment_service/src/experiment_client.py

import grpc
import logging
from typing import Dict, List, Optional

from generated import experiment_service_pb2, experiment_service_pb2_grpc

logger = logging.getLogger(__name__)

class ExperimentClient:
    """Client for ExperimentService"""
    
    def __init__(self, service_url: str = "localhost:50055"):
        self.channel = grpc.insecure_channel(service_url)
        self.stub = experiment_service_pb2_grpc.ExperimentServiceStub(self.channel)
        logger.info(f"ExperimentClient connected to {service_url}")
    
    def create_experiment(self, name: str, description: str = "", 
                         metadata: Dict[str, str] = None) -> Optional[str]:
        """Create a new experiment and return experiment_id"""
        try:
            request = experiment_service_pb2.CreateExperimentRequest(
                name=name,
                description=description,
                metadata=metadata or {}
            )
            
            response = self.stub.CreateExperiment(request)
            
            if response.success:
                logger.info(f"Created experiment: {response.experiment_id}")
                return response.experiment_id
            else:
                logger.error(f"Failed to create experiment: {response.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            return None
    
    def update_experiment(self, experiment_id: str, dataset_id: str = None,
                         model_id: str = None, feature_importance_id: str = None,
                         kegg_analysis_id: str = None, status: str = None) -> bool:
        """Update an experiment with pipeline component IDs"""
        try:
            request = experiment_service_pb2.UpdateExperimentRequest(
                experiment_id=experiment_id,
                dataset_id=dataset_id or "",
                model_id=model_id or "",
                feature_importance_id=feature_importance_id or "",
                kegg_analysis_id=kegg_analysis_id or "",
                status=status or ""
            )
            
            response = self.stub.UpdateExperiment(request)
            
            if response.success:
                logger.info(f"Updated experiment: {experiment_id}")
                return True
            else:
                logger.error(f"Failed to update experiment: {response.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating experiment: {e}")
            return False
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment details"""
        try:
            request = experiment_service_pb2.GetExperimentRequest(
                experiment_id=experiment_id
            )
            
            response = self.stub.GetExperiment(request)
            
            if response.success:
                exp = response.experiment
                return {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "description": exp.description,
                    "created_at": exp.created_at,
                    "updated_at": exp.updated_at,
                    "dataset_id": exp.dataset_id,
                    "model_id": exp.model_id,
                    "feature_importance_id": exp.feature_importance_id,
                    "kegg_analysis_id": exp.kegg_analysis_id,
                    "metadata": dict(exp.metadata),
                    "status": exp.status,
                    "results": {
                        "top_features": list(exp.results.top_features),
                        "top_pathways": list(exp.results.top_pathways)
                    }
                }
            else:
                logger.error(f"Failed to get experiment: {response.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting experiment: {e}")
            return None
    
    def list_experiments(self, limit: int = 100, offset: int = 0,
                        filter_status: str = None) -> List[Dict]:
        """List all experiments"""
        try:
            request = experiment_service_pb2.ListExperimentsRequest(
                limit=limit,
                offset=offset,
                filter_status=filter_status or ""
            )
            
            response = self.stub.ListExperiments(request)
            
            if response.success:
                experiments = []
                for exp in response.experiments:
                    experiments.append({
                        "experiment_id": exp.experiment_id,
                        "name": exp.name,
                        "description": exp.description,
                        "created_at": exp.created_at,
                        "updated_at": exp.updated_at,
                        "dataset_id": exp.dataset_id,
                        "model_id": exp.model_id,
                        "status": exp.status
                    })
                return experiments
            else:
                logger.error(f"Failed to list experiments: {response.error_message}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment"""
        try:
            request = experiment_service_pb2.DeleteExperimentRequest(
                experiment_id=experiment_id
            )
            
            response = self.stub.DeleteExperiment(request)
            
            if response.success:
                logger.info(f"Deleted experiment: {experiment_id}")
                return True
            else:
                logger.error(f"Failed to delete experiment: {response.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting experiment: {e}")
            return False
