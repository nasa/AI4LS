# ml-service/src/service.py (update the __init__ and TrainModel methods)
import grpc
import pandas as pd
from io import BytesIO
import uuid
from typing import Dict, Optional
import logging
from datetime import datetime

import random

from generated import ml_service_pb2, ml_service_pb2_grpc
from src.trainers import ModelTrainer
from src.model_store import ModelStore
from src.data_client import DataServiceClient
from src.preprocessing import build_pipeline, check_input_for_log, parse_trans_list, final_estimator
from src.cross_validation import cross_validate_and_tune, DEFAULT_SCORING

logger = logging.getLogger(__name__)


def _numeric_metrics(metrics: Optional[Dict]) -> Dict[str, float]:
    """Keep only scalar numeric metrics for proto map<string, double> fields
    (drops confusion_matrix / labels, which made ListModels fail)."""
    out = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _str_params(params: Optional[Dict]) -> Dict[str, str]:
    """Proto map<string, string> needs string values."""
    return {str(k): str(v) for k, v in (params or {}).items()}


def _non_default_params(estimator) -> Dict[str, str]:
    """The estimator's parameters that differ from the class defaults, as strings."""
    try:
        defaults = type(estimator)().get_params()
    except Exception:
        defaults = {}
    out = {}
    for k, v in estimator.get_params(deep=False).items():
        if k not in defaults or str(defaults[k]) != str(v):
            out[k] = str(v)
    return out

class MLServiceImpl(ml_service_pb2_grpc.MLServiceServicer):
    """gRPC service implementation for ML operations"""
    
    def __init__(self, data_service_url: str = "data-service:50051"):
        self.model_store = ModelStore()
        self.data_client = DataServiceClient(data_service_url)
        # Cache datasets in memory to avoid repeated fetches
        self.dataset_cache: Dict[str, pd.DataFrame] = {}
        # Import the ModelTrainer class
        from src.trainers import ModelTrainer
        self.model_trainer = ModelTrainer()
    
        logger.info("MLServiceImpl initialized")

    @staticmethod
    def _base_hyperparameters(algorithm: str, random_state: int) -> Dict:
        """Fixed hyperparameters used when not tuning (and as the starting point when tuning).
        Models use the same seed as the train/test split."""
        if algorithm == 'logistic_regression':
            return {'max_iter': 1000, 'solver': 'lbfgs', 'random_state': random_state}
        if algorithm == 'random_forest':
            return {'n_estimators': 50, 'random_state': random_state}
        if algorithm == 'xgboost':
            return {'random_state': random_state}
        if algorithm == 'svm':
            return {'probability': True, 'random_state': random_state}   # predict_proba -> ROC-AUC
        if algorithm == 'neural_network':
            return {
                'hidden_layer_sizes': (50,),     # one hidden layer of 50
                'max_iter': 1000,
                'early_stopping': False,         # so validation_fraction is unused
                'n_iter_no_change': 10,          # stop after 10 epochs without training-loss improvement
                'learning_rate_init': 0.001,
                'solver': 'adam',
                'alpha': 0.001,                  # L2 regularization
                'batch_size': 32,
                'random_state': random_state,
            }
        return {}


    def _get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Get dataset from cache or fetch from Data Service"""
        if dataset_id in self.dataset_cache:
            logger.info(f"Using cached dataset {dataset_id}")
            return self.dataset_cache[dataset_id]
        
        df = self.data_client.get_dataset(dataset_id)
        if df is not None:
            self.dataset_cache[dataset_id] = df
        return df
    
    def TrainModel(self, request, context):
        """Train a model with streaming progress updates"""
        model_id = f"model_{uuid.uuid4().hex[:12]}"
        
        try:
            # Yield starting status
            yield ml_service_pb2.TrainingProgress(
                model_id=model_id,
                status="starting",
                message="Initializing training...",
                progress_percent=0
            )
            
            # Get dataset from Data Service
            df = self._get_dataset(request.dataset_id)

            if df is None:
                yield ml_service_pb2.TrainingProgress(
                    model_id=model_id,
                    status="failed",
                    message=f"Failed to fetch dataset: {request.dataset_id}",
                    error_message=f"Dataset {request.dataset_id} could not be retrieved from Data Service",
                    progress_percent=0
                )
                return
            
            # Validate target column
            if request.target_column not in df.columns:
                yield ml_service_pb2.TrainingProgress(
                    model_id=model_id,
                    status="failed",
                    message=f"Target column '{request.target_column}' not found",
                    error_message=f"Column '{request.target_column}' does not exist in dataset.",
                    progress_percent=0
                )
                return
            
            yield ml_service_pb2.TrainingProgress(
                model_id=model_id,
                status="training",
                message="Preparing data...",
                progress_percent=20
            )
            
            # Prepare data
            X_train, y_train, X_test, y_test = ModelTrainer.prepare_data(
                df,
                request.target_column,
                list(request.feature_columns) if request.feature_columns else [],
                request.test_size or 0.2,
                request.random_state or random.randint(0, 100)
            )
            
            yield ml_service_pb2.TrainingProgress(
                model_id=model_id,
                status="training",
                message=f"Training {request.algorithm} model...",
                progress_percent=40
            )
            
            # Create and train model
            model = ModelTrainer.create_model(
                request.algorithm,
                request.task_type,
                dict(request.hyperparameters)
            )
            
            trained_model, training_metrics, test_metrics, _ = ModelTrainer.train_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                request.task_type
            )

            yield ml_service_pb2.TrainingProgress(
                model_id=model_id,
                status="evaluating",
                message="Evaluating model...",
                progress_percent=80,
                training_metrics=_numeric_metrics(training_metrics)
            )

            # Save model
            model_info = {
                "model_id": model_id,
                "algorithm": request.algorithm,
                "task_type": request.task_type,
                "dataset_id": request.dataset_id,
                "target_column": request.target_column,
                "feature_columns": list(X_train.columns),
                "num_samples": len(df),
                "num_features": len(X_train.columns),
                "hyperparameters": dict(request.hyperparameters),
                "training_metrics": training_metrics,
                "test_metrics": test_metrics,
            }
            self.model_store.save_model(model_id, trained_model, model_info)
            
            # Build final ModelInfo message
            model_info_pb = ml_service_pb2.ModelInfo(
                model_id=model_id,
                algorithm=request.algorithm,
                task_type=request.task_type,
                dataset_id=request.dataset_id,
                target_column=request.target_column,
                feature_columns=list(X_train.columns),
                num_samples=len(df),
                num_features=len(X_train.columns),
                hyperparameters=request.hyperparameters,
                created_at=datetime.utcnow().isoformat(),
                training_metrics=_numeric_metrics(training_metrics),
                test_metrics=_numeric_metrics(test_metrics)
            )

            # Yield completed status
            yield ml_service_pb2.TrainingProgress(
                model_id=model_id,
                status="completed",
                message="Training completed successfully",
                progress_percent=100,
                training_metrics=_numeric_metrics(training_metrics),
                test_metrics=_numeric_metrics(test_metrics),
                model_info=model_info_pb
            )
            
        except Exception as e:
            logger.error(f"Training error for model {model_id}: {e}", exc_info=True)
            yield ml_service_pb2.TrainingProgress(
                model_id=model_id,
                status="failed",
                message=f"Training failed: {str(e)}",
                error_message=str(e),
                progress_percent=0
            )
    
    # GetModelInfo, Predict, and ListModels methods stay the same...
    def GetModelInfo(self, request, context):
        """Get information about a trained model"""
        try:
            model_info = self.model_store.get_model_info(request.model_id)
            
            if not model_info:
                context.abort(grpc.StatusCode.NOT_FOUND, f"Model {request.model_id} not found")
            
            return ml_service_pb2.ModelInfo(
                model_id=model_info["model_id"],
                algorithm=model_info["algorithm"],
                task_type=model_info["task_type"],
                dataset_id=model_info["dataset_id"],
                target_column=model_info["target_column"],
                feature_columns=model_info["feature_columns"],
                num_samples=model_info["num_samples"],
                num_features=model_info["num_features"],
                hyperparameters=_str_params(model_info.get("hyperparameters")),
                created_at=model_info.get("created_at", ""),
                training_metrics=_numeric_metrics(model_info.get("training_metrics")),
                test_metrics=_numeric_metrics(model_info.get("test_metrics"))
            )
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    def Predict(self, request, context):
        """Make predictions with a trained model"""
        try:
            # Load model
            model = self.model_store.load_model(request.model_id)
            if model is None:
                return ml_service_pb2.PredictResponse(
                    success=False,
                    error_message=f"Model {request.model_id} not found"
                )
            
            # Load model info to get feature columns
            model_info = self.model_store.get_model_info(request.model_id)
            
            # Parse input data
            if request.format == "csv":
                df = pd.read_csv(BytesIO(request.input_data))
            else:
                df = pd.read_json(BytesIO(request.input_data))
            
            # Select features
            X = df[model_info["feature_columns"]]
            
            # Make predictions
            predictions = model.predict(X)
            
            # Get probabilities if available (classification)
            probabilities = []
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                # Flatten probabilities for all classes
                probabilities = proba.flatten().tolist()
            
            return ml_service_pb2.PredictResponse(
                success=True,
                predictions=predictions.tolist(),
                probabilities=probabilities
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return ml_service_pb2.PredictResponse(
                success=False,
                error_message=str(e)
            )

    def TrainEnsemble(self, request, context):
        """Train multiple models (ensemble) on the same dataset."""
        try:
            dataset_id = request.dataset_id
            target_column = request.target_column
            algorithms = list(request.algorithms) if request.algorithms else [
                "random_forest", "xgboost", "svm", "logistic_regression", "neural_network"
            ]
        
            logger.info(f"Training ensemble with {len(algorithms)} algorithms")
            logger.info(f"Training ensemble with algorithms: {algorithms}")
        
            # Fetch dataset
            df = self.data_client.get_dataset(dataset_id)
            if df is None:
                return ml_service_pb2.EnsembleResponse(
                    success=False,
                    error_message=f"Failed to fetch dataset {dataset_id}"
                )
        
            logger.info(f"Dataset shape: {df.shape[0]} samples × {df.shape[1]} features")
        
            # Prepare data
            X = df.drop(columns=[target_column])
            y = df[target_column]

            test_size = round(request.test_size, 4) or 0.2   # proto float32 -> 0.20000000298...
            random_state = request.random_state   # proto3 default 0 is a valid, reproducible seed
            trans_list = request.trans_list
            logger.info(f"test_size={test_size}, random_state={random_state}, "
                        f"transformations={parse_trans_list(trans_list) or 'none'} (fit on train split only)")

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # log1p expects raw counts; fail clearly if the data were already transformed
            check_input_for_log(X_train, trans_list)

            logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
            # Cross-validation settings (training split only)
            tune = bool(request.tune_hyperparameters)
            cv_folds = request.cv_folds or 5
            cv_repeats = request.cv_repeats or 1
            cv_scoring = request.cv_scoring or DEFAULT_SCORING
            logger.info(f"CV: {cv_folds}-fold x {cv_repeats} on the training split, "
                        f"tune_hyperparameters={tune}, scoring={cv_scoring}")

            # Train each model
            model_results = []

            for algorithm in algorithms:
                try:
                    logger.info(f"Training {algorithm}...")
                    # Base hyperparameters (used as-is when not tuning; the grid overrides
                    # the parameters it covers when tuning)
                    hyperparams = self._base_hyperparameters(algorithm, random_state)

                    # Create model using ModelTrainer factory
                    model = self.model_trainer.create_model(
                        algorithm=algorithm,
                        task_type="classification",
                        hyperparameters=hyperparams
                    )

                    # Wrap as Pipeline([log, scale, model]); transforms are fit on X_train only
                    model = build_pipeline(model, trans_list)

                    # K-fold CV (and optional GridSearchCV) on X_train only; returns an
                    # unfitted pipeline with the chosen parameters
                    cv_result = cross_validate_and_tune(
                        model, algorithm, X_train, y_train,
                        tune=tune, cv_folds=cv_folds, cv_repeats=cv_repeats,
                        random_state=random_state, scoring=cv_scoring,
                        param_grids_json=request.param_grids_json,
                    )
                    model = cv_result["estimator"]

                    # Refit on all of X_train, score once on the held-out test set
                    trained_model, train_metrics, test_metrics, selected_features = self.model_trainer.train_model(
                        model, X_train, y_train, X_test, y_test, task_type="classification"
                    )
                    final_params = _non_default_params(final_estimator(trained_model))

                    # Generate model ID
                    model_id = f"model_{uuid.uuid4().hex[:12]}"

                    # Store model
                    self.model_store.save_model(model_id, trained_model, {
                        'algorithm': algorithm,
                        'task_type': 'classification',
                        'dataset_id': dataset_id,
                        'target_column': target_column,
                        'feature_columns': selected_features,
                        'num_samples': len(X_train) + len(X_test),
                        'num_features': len(selected_features),
                        'hyperparameters': final_params,
                        'trans_list': ','.join(parse_trans_list(trans_list)),
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'training_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'cv_metrics': cv_result["cv_metrics"],
                        'best_params': cv_result["best_params"],
                        'tuned': cv_result["tuned"],
                        'feature_names': selected_features
                    })
                    logger.info(f"✓ {algorithm} trained: {model_id}")
                    logger.info(f"  Test accuracy: {test_metrics.get('accuracy', 0):.4f}")

                    # Add to results
                    model_results.append(
                        ml_service_pb2.ModelResult(
                            model_id=model_id,
                            algorithm=algorithm,
                            accuracy=test_metrics.get('accuracy', 0),
                            precision=test_metrics.get('precision', 0),
                            recall=test_metrics.get('recall', 0),
                            f1_score=test_metrics.get('f1_score', 0),
                            confusion_matrix_flat=[int(v) for row in test_metrics["confusion_matrix"] for v in row],
                            labels=test_metrics["labels"],
                            roc_auc=test_metrics.get('roc_auc', 0.0),
                            cv_metrics=cv_result["cv_metrics"],
                            best_params=cv_result["best_params"],
                            tuned=cv_result["tuned"],
                            hyperparameters=final_params,
                        )
                    )
                
                except Exception as e:
                    logger.error(f"Failed to train {algorithm}: {e}", exc_info=True)
                    continue
        
            if not model_results:
                return ml_service_pb2.EnsembleResponse(
                    success=False,
                    error_message="No models were successfully trained"
                )
        
            logger.info(f"✓ Ensemble training complete: {len(model_results)} models")
        
            return ml_service_pb2.EnsembleResponse(
                success=True,
                models=model_results,
                num_models=len(model_results)
            )
        
        except Exception as e:
            logger.error(f"Ensemble training error: {e}", exc_info=True)
            return ml_service_pb2.EnsembleResponse(
                success=False,
                error_message=str(e)
            )
    
    def ListModels(self, request, context):
        """List all trained models"""
        try:
            models = self.model_store.list_models(
                algorithm=request.algorithm if request.algorithm else None,
                task_type=request.task_type if request.task_type else None,
                limit=request.limit if request.limit > 0 else None
            )
            
            model_infos = []
            for model_info in models:
                model_infos.append(
                    ml_service_pb2.ModelInfo(
                        model_id=model_info["model_id"],
                        algorithm=model_info["algorithm"],
                        task_type=model_info["task_type"],
                        dataset_id=model_info["dataset_id"],
                        target_column=model_info["target_column"],
                        feature_columns=model_info["feature_columns"],
                        num_samples=model_info["num_samples"],
                        num_features=model_info["num_features"],
                        hyperparameters=_str_params(model_info.get("hyperparameters")),
                        created_at=model_info.get("created_at", ""),
                        training_metrics=_numeric_metrics(model_info.get("training_metrics")),
                        test_metrics=_numeric_metrics(model_info.get("test_metrics"))
                    )
                )
            
            return ml_service_pb2.ModelList(
                models=model_infos,
                total_count=len(model_infos)
            )
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
