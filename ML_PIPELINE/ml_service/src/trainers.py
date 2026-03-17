# src/trainers.py
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Factory for creating and training ML models"""
    
    CLASSIFICATION_MODELS = {
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "logistic_regression": LogisticRegression,
        "gradient_boosting": GradientBoostingClassifier,
        "xgboost": xgb.XGBClassifier,
        "neural_network": MLPClassifier,
    }
    
    REGRESSION_MODELS = {
        "random_forest": RandomForestRegressor,
        "svm": SVR,
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "gradient_boosting": GradientBoostingRegressor,
        "xgboost": xgb.XGBRegressor,
        "neural_network": MLPRegressor,
    }
    
    @staticmethod
    def create_model(algorithm: str, task_type: str, hyperparameters: Dict[str, Any]):
        """Create a model instance"""
        if task_type == "classification":
            model_class = ModelTrainer.CLASSIFICATION_MODELS.get(algorithm)
        else:
            model_class = ModelTrainer.REGRESSION_MODELS.get(algorithm)
        
        if model_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm} for task: {task_type}")
        
        # Convert hyperparameters from strings to appropriate types
        parsed_params = ModelTrainer._parse_hyperparameters(hyperparameters, algorithm)
        
        return model_class(**parsed_params)
    
    @staticmethod
    def _parse_hyperparameters(params: Dict[str, str], algorithm: str) -> Dict[str, Any]:
        """Parse string hyperparameters to appropriate types"""
        parsed = {}
        
        for key, value in params.items():
            # Try to parse as int
            try:
                parsed[key] = int(value)
                continue
            except ValueError:
                pass
            
            # Try to parse as float
            try:
                parsed[key] = float(value)
                continue
            except ValueError:
                pass
            
            # Try to parse as bool
            if value.lower() in ('true', 'false'):
                parsed[key] = value.lower() == 'true'
                continue
            
            # Keep as string
            parsed[key] = value
        
        return parsed
    
    @staticmethod
    def train_model(
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str
    ) -> Tuple[Any, Dict[str, float], Dict[str, float]]:
        """Train model and calculate metrics"""
        
        # Train the model
        logger.info(f"Training {type(model).__name__}...")
        model.fit(X_train, y_train)
        
        # Get predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        if task_type == "classification":
            training_metrics = {
                "accuracy": float(accuracy_score(y_train, y_train_pred)),
                "precision": float(precision_score(y_train, y_train_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_train, y_train_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0)),
            }
            
            test_metrics = {
                "accuracy": float(accuracy_score(y_test, y_test_pred)),
                "precision": float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
            }
            
            # Add ROC AUC if binary classification and model supports predict_proba
            if hasattr(model, 'predict_proba') and len(np.unique(y_train)) == 2:
                try:
                    y_train_proba = model.predict_proba(X_train)[:, 1]
                    y_test_proba = model.predict_proba(X_test)[:, 1]
                    training_metrics["roc_auc"] = float(roc_auc_score(y_train, y_train_proba))
                    test_metrics["roc_auc"] = float(roc_auc_score(y_test, y_test_proba))
                except:
                    pass
        
        else:  # regression
            training_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                "mae": float(mean_absolute_error(y_train, y_train_pred)),
                "r2_score": float(r2_score(y_train, y_train_pred)),
            }
            
            test_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                "mae": float(mean_absolute_error(y_test, y_test_pred)),
                "r2_score": float(r2_score(y_test, y_test_pred)),
            }
        
        return model, training_metrics, test_metrics
    
    @staticmethod
    def prepare_data(
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        test_size: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        
        # Select features
        if not feature_columns:
            # Use all columns except target
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, y_train, X_test, y_test
