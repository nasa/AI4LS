# ml-service/src/service.py (update the __init__ and TrainModel methods)
import grpc
import pandas as pd
from io import BytesIO
import uuid
from typing import Dict, Optional
import logging
from datetime import datetime

from generated import ml_service_pb2, ml_service_pb2_grpc
from src.trainers import ModelTrainer
from src.model_store import ModelStore
from src.data_client import DataServiceClient

logger = logging.getLogger(__name__)

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
            logger.info(f"shape: {df.shape[0]} by {df.shape[1]} ")
            
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
                request.random_state or 42
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
            
            trained_model, training_metrics, test_metrics = ModelTrainer.train_model(
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
                training_metrics=training_metrics
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
                training_metrics=training_metrics,
                test_metrics=test_metrics
            )
            
            # Yield completed status
            yield ml_service_pb2.TrainingProgress(
                model_id=model_id,
                status="completed",
                message="Training completed successfully",
                progress_percent=100,
                training_metrics=training_metrics,
                test_metrics=test_metrics,
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
                hyperparameters=model_info["hyperparameters"],
                created_at=model_info["created_at"],
                training_metrics=model_info["training_metrics"],
                test_metrics=model_info["test_metrics"]
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
        
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
            logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
            # Train each model
            model_results = []
            
            for algorithm in algorithms:
                try:
                    logger.info(f"Training {algorithm}...")
                
                    # Create model using ModelTrainer factory
                    model = self.model_trainer.create_model(
                        algorithm=algorithm,
                        task_type="classification",
                        hyperparameters={}  # Use defaults
                    )
                
                    # Train model
                    trained_model, train_metrics, test_metrics = self.model_trainer.train_model(
                        model, X_train, y_train, X_test, y_test, task_type="classification"
                    )
                
                    # Generate model ID
                    model_id = f"model_{uuid.uuid4().hex[:12]}"
                    
                    # Store model
                    self.model_store.save_model(model_id, trained_model, {
                        'algorithm': algorithm,
                        'task_type': 'classification',  # Add this
                        'dataset_id': dataset_id,
                        'target_column': target_column,
                        'feature_columns': list(X.columns),  # Add this
                        'num_samples': len(X_train) + len(X_test),  # Add this
                        'num_features': len(X.columns),  # Add this
                        'hyperparameters': {},  # Add this
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'training_metrics': train_metrics,  # Change from train_metrics
                        'test_metrics': test_metrics,
                        'feature_names': list(X.columns)
                    }) 
                    logger.info(f"✓ {algorithm} trained: {model_id}")
                    logger.info(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                
                    # Add to results
                    model_results.append(
                        ml_service_pb2.ModelResult(
                            model_id=model_id,
                            algorithm=algorithm,
                            accuracy=test_metrics.get('accuracy', 0),
                            precision=test_metrics.get('precision', 0),
                            recall=test_metrics.get('recall', 0),
                            f1_score=test_metrics.get('f1_score', 0)
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
                        hyperparameters=model_info["hyperparameters"],
                        created_at=model_info["created_at"],
                        training_metrics=model_info["training_metrics"],
                        test_metrics=model_info["test_metrics"]
                    )
                )
            
            return ml_service_pb2.ModelList(
                models=model_infos,
                total_count=len(model_infos)
            )
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
