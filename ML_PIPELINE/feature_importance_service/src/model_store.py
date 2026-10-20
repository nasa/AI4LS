# src/model_store.py  (same file for ml_service and feature_importance_service)
"""
Persistent storage for trained models, safe for concurrent pipeline runs.

Layout under base_path (shared between the ML and feature importance services):
    <model_id>.joblib          the fitted model (or Pipeline)
    meta/<model_id>.json       that model's metadata
    metadata.json              legacy single-file metadata (read-only fallback for old models)

Why: one shared metadata.json, rewritten in full on every save, broke under parallel runs.
Threads changed the dict while it was being written ("dictionary changed size during
iteration"), and the feature importance service read half-written files ("Expecting ','
delimiter"). Now every model has its own small file, every file is written to a temp name
and renamed into place (readers see the old file or the complete new one, never a partial
one), the metadata is written only after the model file is complete, and in-memory state
is guarded by a lock.
"""
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)


def _json_default(o):
    """Let numpy scalars/arrays and other odd values through json.dump."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def _atomic_write_bytes(path: Path, write_fn):
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        write_fn(tmp)
        os.replace(tmp, path)          # atomic on the same filesystem
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


class ModelStore:
    """Persistent storage for trained models"""

    def __init__(self, base_path: str = "./models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.base_path / "meta"
        self.meta_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_path / "metadata.json"   # legacy

        self._lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}             # model_id -> metadata
        self._legacy: Optional[Dict[str, Dict[str, Any]]] = None

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_id(model_id: str) -> str:
        return str(model_id).replace(".joblib", "")

    def _meta_path(self, model_id: str) -> Path:
        return self.meta_dir / f"{model_id}.json"

    def _legacy_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Old shared metadata.json, read once. A corrupt file is reported, not fatal."""
        if self._legacy is None:
            self._legacy = {}
            if self.metadata_file.exists():
                try:
                    with open(self.metadata_file) as f:
                        self._legacy = json.load(f)
                    logger.info(f"Loaded {len(self._legacy)} models from legacy {self.metadata_file}")
                except Exception as e:
                    logger.warning(f"Legacy metadata file {self.metadata_file} is unreadable ({e}); "
                                   "models saved before the per-model format may be unavailable")
        return self._legacy

    def _read_meta(self, model_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if model_id in self._cache:
                return self._cache[model_id]
        path = self._meta_path(model_id)
        info = None
        if path.exists():
            try:
                with open(path) as f:
                    info = json.load(f)
            except Exception as e:
                logger.error(f"Unreadable metadata for {model_id} at {path}: {e}")
                return None
        else:
            info = self._legacy_metadata().get(model_id)
        if info is not None:
            with self._lock:
                self._cache[model_id] = info
        return info

    def _model_path(self, model_id: str, info: Dict[str, Any]) -> Path:
        recorded = Path(info.get("model_path", ""))
        if recorded.name and recorded.exists():
            return recorded
        return self.base_path / f"{model_id}.joblib"     # e.g. different mount path per service

    # ── public API (unchanged signatures) ───────────────────────────────────

    def save_model(self, model_id: str, model: Any, model_info: Dict[str, Any]):
        """Save a trained model and its metadata"""
        model_id = self._clean_id(model_id)
        try:
            model_path = self.base_path / f"{model_id}.joblib"
            # Model first, metadata second: if metadata exists, the model file is complete
            _atomic_write_bytes(model_path, lambda tmp: joblib.dump(model, tmp))

            info = {
                "model_id": model_id,
                **model_info,
                "model_path": str(model_path),
                "created_at": datetime.utcnow().isoformat(),
            }
            payload = json.dumps(info, indent=2, default=_json_default)
            _atomic_write_bytes(self._meta_path(model_id), lambda tmp: tmp.write_text(payload))

            with self._lock:
                self._cache[model_id] = json.loads(payload)
            logger.info(f"Model {model_id} saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}", exc_info=True)
            return False

    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a trained model"""
        model_id = self._clean_id(model_id)
        try:
            info = self._read_meta(model_id)
            if info is None:
                logger.error(f"Model {model_id} not found in metadata")
                return None
            model_path = self._model_path(model_id, info)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            model = joblib.load(model_path)
            logger.info(f"Model {model_id} loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            return None

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Metadata for a model, or {} if unknown (falsy, so `if not info:` checks still work)"""
        model_id = self._clean_id(model_id)
        info = self._read_meta(model_id)
        if info is None:
            logger.warning(f"No metadata found for model {model_id}")
            return {}
        return info

    def list_models(
        self,
        algorithm: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List all models with optional filtering"""
        models: Dict[str, Dict[str, Any]] = dict(self._legacy_metadata())
        for path in self.meta_dir.glob("*.json"):
            info = self._read_meta(path.stem)
            if info is not None:
                models[path.stem] = info
        models = list(models.values())

        if algorithm:
            models = [m for m in models if m.get("algorithm") == algorithm]
        if task_type:
            models = [m for m in models if m.get("task_type") == task_type]
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        if limit:
            models = models[:limit]
        return models

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its metadata"""
        model_id = self._clean_id(model_id)
        try:
            info = self._read_meta(model_id)
            if info is None:
                return False
            self._model_path(model_id, info).unlink(missing_ok=True)
            self._meta_path(model_id).unlink(missing_ok=True)
            with self._lock:
                self._cache.pop(model_id, None)
            if self._legacy and model_id in self._legacy:
                self._legacy.pop(model_id, None)   # in memory only; the legacy file is no longer written
            logger.info(f"Model {model_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
