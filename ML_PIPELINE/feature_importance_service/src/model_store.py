# src/model_store.py
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ModelStore:
    """Persistent storage for trained models"""
    
    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.metadata_file = self.base_path / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.debug(f"Loaded metadata: {len(self.metadata)} models")
            else:
                logger.warning(f"Metadata file not found: {self.metadata_file}")
                self.metadata = {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.metadata = {}

    
    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(self, model_id: str, model: Any, model_info: Dict[str, Any]):
        """Save a trained model and its metadata"""
        try:
            # Save model artifact
            model_path = self.base_path / f"{model_id}.joblib"
            joblib.dump(model, model_path)
            
            # Save metadata
            self.metadata[model_id] = {
                **model_info,
                "model_path": str(model_path),
                "created_at": datetime.utcnow().isoformat()
            }
            self._save_metadata()
            
            logger.info(f"Model {model_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a trained model"""
        try:
            # Reload metadata from disk to get latest models
            self._load_metadata()  # ADD THIS LINE

            # Remove .joblib extension if present
            model_id = model_id.replace('.joblib', '')

            if model_id not in self.metadata:
                logger.error(f"Model {model_id} not found in metadata")
                return None
            
            model_path = Path(self.metadata[model_id]["model_path"])
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            model = joblib.load(model_path)
            logger.info(f"Model {model_id} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        try:
            # Reload metadata from disk to get latest models
            self._load_metadata()  # ADD THIS LINE
        
            # Remove .joblib extension if present
            model_id = model_id.replace('.joblib', '')
        
            if model_id in self.metadata:
                return self.metadata[model_id]
            else:
                logger.warning(f"No metadata found for model {model_id}")
                return {}
            
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
            return {}

    
    def list_models(
        self, 
        algorithm: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List all models with optional filtering"""
        models = list(self.metadata.values())
        
        # Apply filters
        if algorithm:
            models = [m for m in models if m.get("algorithm") == algorithm]
        
        if task_type:
            models = [m for m in models if m.get("task_type") == task_type]
        
        # Sort by created_at (newest first)
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Apply limit
        if limit:
            models = models[:limit]
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its metadata"""
        try:
            if model_id not in self.metadata:
                return False
            
            # Delete model file
            model_path = Path(self.metadata[model_id]["model_path"])
            if model_path.exists():
                model_path.unlink()
            
            # Remove from metadata
            del self.metadata[model_id]
            self._save_metadata()
            
            logger.info(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
