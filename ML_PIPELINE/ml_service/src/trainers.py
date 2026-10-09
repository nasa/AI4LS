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
        "Ridge": Ridge,
        "Lasso": Lasso,
        "gradient_boosting": GradientBoostingRegressor,
        "xgboost": xgb.XGBRegressor,
        "neural_network": MLPRegressor,
    }
    
    @staticmethod
    def create_model(algorithm: str, task_type: str, hyperparameters: Dict[str, Any]):
        """Create a model instance"""
        if task_type == "classification":
            model_class = ModelTrainer.CLASSIFICATION_MODELS.get(algorithm)
        elif task_type == "regression":
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
            if not isinstance(value, str):
                parsed[key] = value
                continue
            if key == "hidden_layer_sizes":
                parsed[key] = tuple(int(v) for v in value.replace("(", "").replace(")", "").split(",") if v.strip())
                continue
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
        task_type: str,
        min_features: int = None
    ) -> Tuple[Any, Dict[str, float], Dict[str, float], List[str]]:
        """Train model and calculate metrics"""

        # init selected_features to all cols
        selected_features = list(X_train.columns)  # ← Add this line early

        # Remove non-numeric columns that cause issues with some models
        #cols_to_drop = ['source_dataset', 'Factor Value[Spaceflight]']
        cols_to_drop = ['source_dataset']
        X_train = X_train.drop(columns=[col for col in cols_to_drop if col in X_train.columns])
        X_test = X_test.drop(columns=[col for col in cols_to_drop if col in X_test.columns])

        # pull out just the values from the y_train and y_test series
        #y_train_array = y_train.to_numpy()
        #y_test_array = y_test.to_numpy()
        
        # Keep only numeric columns
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])

        # update selected_features after dropping cols
        selected_features = list(X_train.columns)

        logger.info(f"columns of X_train: {X_train.columns}")
        logger.info(f"y_train values: {y_train}")

        #X_train_array = X_train.to_numpy()
        #X_test_array = X_test.to_numpy()


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
            
            from sklearn.metrics import confusion_matrix   # better at the top of the file

            labels = list(model.classes_) if hasattr(model, "classes_") else sorted(set(y_test) | set(y_test_pred))
            test_metrics = {
                "accuracy": float(accuracy_score(y_test, y_test_pred)),
                "precision": float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_test_pred, labels=labels).tolist(), "labels": [str(l) for l in labels],
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
        
        return model, training_metrics, test_metrics, selected_features
    
    @staticmethod
    def prepare_data(
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        test_size: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for training"""

        # Remove source_dataset and other non-numeric columns
        df = df.drop(columns=['source_dataset'], errors='ignore')
        
        # Keep only numeric columns (plus target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Add target column if it's not numeric
        if target_column not in numeric_cols and target_column in df.columns:
            numeric_cols.append(target_column)
        
        df = df[numeric_cols] 

        # Select features
        if not feature_columns:
            # Use all columns except target
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, y_train, X_test, y_test


    @staticmethod
    def compute_loocv_metrics(
        model,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> Dict[str, float]:
        """
        Compute Leave-One-Out Cross-Validation metrics on full dataset
    
        LOOCV is excellent for small datasets - it uses every sample as test set once.
        This provides a robust estimate of model performance.
    
        Args:
            model: Trained sklearn model
            X: Full feature matrix
            y: Full target vector
            task_type: "classification" or "regression"
    
        Returns:
            Dict with LOOCV metrics
        """
        from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_validate
    
        logger.info(f"Computing Leave-One-Out Cross-Validation on {len(X)} samples...")
    
        loo = LeaveOneOut()
    
        try:
            if task_type == "classification":
                # Compute multiple metrics for classification
                scoring = {
                    'accuracy': 'accuracy',
                    'precision': 'precision_weighted',
                    'recall': 'recall_weighted',
                    'f1': 'f1_weighted'
                }
            
                cv_results = cross_validate(
                    model, X, y, cv=loo, scoring=scoring, n_jobs=-1
                )
            
                loocv_metrics = {
                    "accuracy": float(cv_results['test_accuracy'].mean()),
                    "precision": float(cv_results['test_precision'].mean()),
                    "recall": float(cv_results['test_recall'].mean()),
                    "f1_score": float(cv_results['test_f1'].mean()),
                    "n_folds": len(cv_results['test_accuracy']),
                    "std_accuracy": float(cv_results['test_accuracy'].std()),
                }
            
                # Try to add ROC-AUC if binary classification
                if hasattr(model, 'predict_proba'):
                    try:
                        roc_scores = cross_val_score(
                            model, X, y, cv=loo, scoring='roc_auc_weighted'
                        )
                        loocv_metrics['roc_auc'] = float(roc_scores.mean())
                    except:
                        pass
        
            else:  # regression
                scoring = {
                    'rmse': 'neg_mean_squared_error',
                    'mae': 'neg_mean_absolute_error',
                    'r2': 'r2'
                }
            
                cv_results = cross_validate(
                    model, X, y, cv=loo, scoring=scoring, n_jobs=-1
                )
            
                loocv_metrics = {
                    "rmse": float(np.sqrt(-cv_results['test_rmse'].mean())),
                    "mae": float(-cv_results['test_mae'].mean()),
                    "r2_score": float(cv_results['test_r2'].mean()),
                    "n_folds": len(cv_results['test_rmse']),
                    "std_rmse": float(np.sqrt(-cv_results['test_rmse'].std())),
                }
        
            logger.info(f"✓ LOOCV completed: {loocv_metrics}")
            return loocv_metrics
    
        except Exception as e:
            logger.error(f"Error computing LOOCV: {e}")
            return {}
