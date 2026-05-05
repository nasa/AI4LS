# experiment_service/src/experiment_store.py

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class ExperimentStore:
    """Persistent storage for experiments"""
    
    def __init__(self, store_path: str = "/app/experiments"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.store_path / "experiments.json"
        
        # Load existing experiments
        self.experiments: Dict[str, Dict] = {}
        self._load_experiments()
        
        logger.info(f"ExperimentStore initialized at {self.store_path}")
        logger.info(f"Loaded {len(self.experiments)} experiments")
    
    def _load_experiments(self):
        """Load experiments from disk"""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, 'r') as f:
                    self.experiments = json.load(f)
                logger.info(f"Loaded {len(self.experiments)} experiments from disk")
            except Exception as e:
                logger.error(f"Error loading experiments: {e}")
                self.experiments = {}
        else:
            self.experiments = {}
    
    def _save_experiments(self):
        """Save experiments to disk"""
        try:
            with open(self.experiments_file, 'w') as f:
                json.dump(self.experiments, f, indent=2)
            logger.info(f"Saved {len(self.experiments)} experiments to disk")
        except Exception as e:
            logger.error(f"Error saving experiments: {e}")
    
    def create_experiment(self, experiment_id: str, name: str, description: str, 
                         metadata: Dict[str, str]) -> Dict:
        """Create a new experiment"""
        experiment = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "dataset_id": "",
            "model_id": "",
            "feature_importance_id": "",
            "kegg_analysis_id": "",
            "metadata": metadata,
            "status": "created",
            "results": {}
        }
        
        self.experiments[experiment_id] = experiment
        self._save_experiments()
        
        logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment
    
    def update_experiment(self, experiment_id: str, **updates) -> bool:
        """Update an experiment"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        # Update fields
        for key, value in updates.items():
            if value:  # Only update non-empty values
                self.experiments[experiment_id][key] = value
        
        # Update timestamp
        self.experiments[experiment_id]["updated_at"] = int(time.time())
        
        self._save_experiments()
        logger.info(f"Updated experiment {experiment_id}")
        return True
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get an experiment by ID"""
        return self.experiments.get(experiment_id)
    
    def list_experiments(self, limit: int = 100, offset: int = 0, 
                        filter_status: str = None) -> List[Dict]:
        """List all experiments with optional filtering"""
        experiments = list(self.experiments.values())
        
        # Filter by status if provided
        if filter_status:
            experiments = [e for e in experiments if e.get("status") == filter_status]
        
        # Sort by created_at (newest first)
        experiments.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        # Apply pagination
        return experiments[offset:offset + limit]
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment"""
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            self._save_experiments()
            logger.info(f"Deleted experiment {experiment_id}")
            return True
        return False
    
    def get_total_count(self, filter_status: str = None) -> int:
        """Get total count of experiments"""
        if filter_status:
            return len([e for e in self.experiments.values() 
                       if e.get("status") == filter_status])
        return len(self.experiments)
