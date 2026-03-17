# orchestration-service/test_ml_pipeline.py
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("STEP 1: Health Check")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    if not result['services']['data_service']:
        print("\n⚠️  Data Service is not healthy!")
        return False
    
    if not result['services']['ml_service']:
        print("\n⚠️  ML Service is not healthy!")
        return False
    
    print("\n✓ All services healthy")
    return True

def test_upload_and_validate(file_path):
    """Upload and validate dataset"""
    print("\n" + "=" * 60)
    print("STEP 2: Upload and Validate Dataset")
    print("=" * 60)
    
    with open(file_path, 'rb') as f:
        files = {'file': ('data.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/api/datasets/validate", files=files)
    
    result = response.json()
    print(f"Valid: {result['is_valid']}")
    
    if result['dataset_info']:
        dataset_id = result['dataset_info']['dataset_id']
        print(f"Dataset ID: {dataset_id}")
        print(f"Shape: {result['dataset_info']['num_rows']} x {result['dataset_info']['num_columns']}")
        print(f"\nColumns:")
        for col in result['dataset_info']['columns']:
            print(f"  - {col['name']} ({col['dtype']})")
        
        return dataset_id
    
    return None

def test_run_pipeline_with_streaming(dataset_id):
    """Run full ML pipeline with streaming progress"""
    print("\n" + "=" * 60)
    print("STEP 3: Run ML Pipeline (Streaming)")
    print("=" * 60)
    
    payload = {
        "dataset_id": dataset_id,
        "config": {
            "target_column": "income_category",
            "task_type": "classification",
            "feature_columns": [],  # Use all columns except target
            "transformations": [
                {
                    "type": "standardize",
                    "columns": ["age", "income", "education_years", "hours_per_week"],
                    "params": {}
                }
            ],
            "algorithm": "random_forest",
            "hyperparameters": {
                "n_estimators": "100",
                "max_depth": "10",
                "random_state": "42"
            },
            "metrics": ["accuracy", "f1_score"],
            "test_size": 0.2,
            "random_state": 42
        }
    }
    
    print("\nConfiguration:")
    print(f"  Target: {payload['config']['target_column']}")
    print(f"  Task: {payload['config']['task_type']}")
    print(f"  Algorithm: {payload['config']['algorithm']}")
    print(f"  Transformations: {len(payload['config']['transformations'])}")
    print(f"  Test size: {payload['config']['test_size']}")
    
    print("\nStreaming progress:")
    print("-" * 60)
    
    response = requests.post(
        f"{BASE_URL}/api/pipeline/run",
        json=payload,
        stream=True
    )
    
    model_id = None
    final_metrics = None
    
    for line in response.iter_lines():
        if line:
            progress = json.loads(line)
            
            status = progress.get('status', 'unknown')
            message = progress.get('message', '')
            percent = progress.get('progress_percent', 0)
            
            # Progress bar
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"[{bar}] {percent:3d}% | {status:12s} | {message}")
            
            # Show metrics when available
            if progress.get('training_metrics'):
                print(f"  Training Metrics: {progress['training_metrics']}")
            
            if progress.get('test_metrics'):
                print(f"  Test Metrics: {progress['test_metrics']}")
                final_metrics = progress['test_metrics']
            
            if progress.get('model_id'):
                model_id = progress['model_id']
            
            if progress.get('error'):
                print(f"  ❌ Error: {progress['error']}")
                return None, None
    
    print("-" * 60)
    return model_id, final_metrics

def test_get_model_info(model_id):
    """Get information about trained model"""
    print("\n" + "=" * 60)
    print("STEP 4: Get Model Information")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/models/{model_id}")
    result = response.json()
    
    print(f"Model ID: {result['model_id']}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Task Type: {result['task_type']}")
    print(f"Target Column: {result['target_column']}")
    print(f"Features: {len(result['feature_columns'])} columns")
    print(f"Training Samples: {result['num_samples']}")
    print(f"Created: {result['created_at']}")
    
    print("\nTraining Metrics:")
    for metric, value in result['training_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in result['test_metrics'].items():
        print(f"  {metric}: {value:.4f}")

def test_list_models():
    """List all trained models"""
    print("\n" + "=" * 60)
    print("STEP 5: List All Models")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/models?limit=5")
    result = response.json()
    
    print(f"Total models: {result['total_count']}")
    
    if result['models']:
        print("\nRecent models:")
        for i, model in enumerate(result['models'], 1):
            print(f"\n{i}. {model['model_id']}")
            print(f"   Algorithm: {model['algorithm']}")
            print(f"   Task: {model['task_type']}")
            print(f"   Samples: {model['num_samples']}")
            print(f"   Test Accuracy: {model['test_metrics'].get('accuracy', 'N/A')}")

def test_different_algorithms(dataset_id):
    """Test multiple algorithms"""
    print("\n" + "=" * 60)
    print("BONUS: Testing Multiple Algorithms")
    print("=" * 60)
    
    algorithms = ["logistic_regression", "svm", "gradient_boosting"]
    results = []
    
    for algo in algorithms:
        print(f"\n--- Testing {algo} ---")
        
        payload = {
            "dataset_id": dataset_id,
            "config": {
                "target_column": "income_category",
                "task_type": "classification",
                "feature_columns": [],
                "transformations": [
                    {
                        "type": "standardize",
                        "columns": ["age", "income", "education_years", "hours_per_week"],
                        "params": {}
                    }
                ],
                "algorithm": algo,
                "hyperparameters": {},
                "metrics": ["accuracy", "f1_score"],
                "test_size": 0.2,
                "random_state": 42
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/api/pipeline/run",
            json=payload,
            stream=True
        )
        
        model_id = None
        test_metrics = None
        
        for line in response.iter_lines():
            if line:
                progress = json.loads(line)
                if progress.get('status') == 'completed':
                    model_id = progress.get('model_id')
                    test_metrics = progress.get('metrics')
        
        if test_metrics:
            results.append({
                "algorithm": algo,
                "model_id": model_id,
                "metrics": test_metrics
            })
            print(f"✓ {algo}: Accuracy = {test_metrics.get('accuracy', 0):.4f}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    
    for result in sorted(results, key=lambda x: x['metrics'].get('accuracy', 0), reverse=True):
        print(f"{result['algorithm']:20s} | Accuracy: {result['metrics'].get('accuracy', 0):.4f} | F1: {result['metrics'].get('f1_score', 0):.4f}")

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ML PIPELINE END-TO-END TEST" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Step 1: Health check
    if not test_health():
        print("\n❌ Services not healthy. Exiting.")
        return
    
    # Step 2: Upload dataset
    dataset_id = test_upload_and_validate("../test_classification.csv")
    if not dataset_id:
        print("\n❌ Failed to upload dataset. Exiting.")
        return
    
    print(f"\n✓ Dataset ready: {dataset_id}")
    
    # Step 3: Run pipeline
    model_id, metrics = test_run_pipeline_with_streaming(dataset_id)
    if not model_id:
        print("\n❌ Pipeline failed. Exiting.")
        return
    
    print(f"\n✓ Model trained: {model_id}")
    print(f"✓ Test Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    # Step 4: Get model info
    test_get_model_info(model_id)
    
    # Step 5: List models
    test_list_models()
    
    # Bonus: Test multiple algorithms
    user_input = input("\nWould you like to test multiple algorithms? (y/n): ")
    if user_input.lower() == 'y':
        test_different_algorithms(dataset_id)
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
