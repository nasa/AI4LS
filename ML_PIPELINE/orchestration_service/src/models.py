# src/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class TransformationType(str, Enum):
    """Available transformation types"""
    LOG = "log"
    STANDARDIZE = "standardize"
    NORMALIZE = "normalize"
    ONE_HOT_ENCODE = "one_hot_encode"

class ColumnInfo(BaseModel):
    """Information about a dataset column"""
    name: str
    dtype: str
    null_count: int
    sample_values: List[str] = []

class DatasetInfo(BaseModel):
    """Dataset metadata"""
    dataset_id: str
    num_rows: int
    num_columns: int
    columns: List[ColumnInfo]
    size_bytes: int

class ValidationResponse(BaseModel):
    """Response from dataset validation"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    dataset_info: Optional[DatasetInfo] = None

class TransformationConfig(BaseModel):
    """Configuration for a single transformation"""
    type: TransformationType
    columns: List[str]
    params: Dict[str, Any] = {}

class TransformationRequest(BaseModel):
    """Request to transform a dataset"""
    #dataset_id: str
    transformations: List[TransformationConfig]

class TransformationResponse(BaseModel):
    """Response from transformation"""
    success: bool
    transformed_dataset_id: Optional[str] = None
    error_message: Optional[str] = None
    transformed_info: Optional[DatasetInfo] = None

class MLAlgorithm(str, Enum):
    """Available ML algorithms"""
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    RIDGE = "Ridge"
    LASSO = "Lasso"


class MetricType(str, Enum):
    """Available evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    RMSE = "rmse"
    MAE = "mae"
    R2_SCORE = "r2_score"

class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""
    target_column: str  # ADD THIS
    task_type: str = "classification"  # ADD THIS: "classification" or "regression"
    feature_columns: List[str] = []  # ADD THIS: empty means use all except target
    transformations: List[TransformationConfig] = []
    algorithm: MLAlgorithm
    hyperparameters: Dict[str, Any] = {}
    metrics: List[MetricType]
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: Optional[int] = 42

class PipelineRequest(BaseModel):
    """Request to run full ML pipeline"""
    dataset_id: str
    config: PipelineConfig

class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    VALIDATING = "validating"
    TRANSFORMING = "transforming"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"

class PipelineResponse(BaseModel):
    """Response from pipeline execution"""
    pipeline_id: str
    status: PipelineStatus
    message: str
    dataset_id: Optional[str] = None
    transformed_dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, bool]
