# orchestration-service/test_orchestration.py
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("=" * 50)
    print("Testing Health Check")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_validate(file_path):
    """Test dataset validation"""
    print("\n" + "=" * 50)
    print("Testing Dataset Validation")
    print("=" * 50)
    
    with open(file_path, 'rb') as f:
        files = {'file': ('data.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/api/datasets/validate", files=files)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    if result.get('dataset_info'):
        return result['dataset_info']['dataset_id']
    return None

def test_transform(dataset_id):
    """Test dataset transformation"""
    print("\n" + "=" * 50)
    print("Testing Dataset Transformation")
    print("=" * 50)
    
    payload = {
        "transformations": [
            {
                "type": "log",
                "columns": ["income"],
                "params": {}
            },
            {
                "type": "standardize",
                "columns": ["income", "age"],
                "params": {}
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/datasets/{dataset_id}/transform",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    if result.get('transformed_dataset_id'):
        return result['transformed_dataset_id']
    return None

def test_get_dataset_info(dataset_id):
    """Test getting dataset info"""
    print("\n" + "=" * 50)
    print("Testing Get Dataset Info")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/api/datasets/{dataset_id}")
    
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_pipeline(dataset_id):
    """Test full pipeline"""
    print("\n" + "=" * 50)
    print("Testing Full Pipeline")
    print("=" * 50)
    
    payload = {
        "dataset_id": dataset_id,
        "config": {
            "transformations": [
                {
                    "type": "standardize",
                    "columns": ["age", "income"],
                    "params": {}
                }
            ],
            "algorithm": "random_forest",
            "hyperparameters": {},
            "metrics": ["accuracy", "f1_score"],
            "test_size": 0.2
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/pipeline/run", json=payload)
    
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Test health
    health = test_health()
    
    if not health['services']['data_service']:
        print("\n⚠️  Data Service is not healthy. Make sure it's running.")
        print("   Run: cd data-service && python -m src.server")
        exit(1)
    
    # Test with your sample data
    dataset_id = test_validate("../data-service/sample_data.csv")
    
    if dataset_id:
        print(f"\n✓ Dataset validated: {dataset_id}")
        
        # Test transformation
        transformed_id = test_transform(dataset_id)
        
        if transformed_id:
            print(f"\n✓ Dataset transformed: {transformed_id}")
            
            # Get info about transformed dataset
            test_get_dataset_info(transformed_id)
        
        # Test full pipeline
        test_pipeline(dataset_id)
    
    print("\n" + "=" * 50)
    print("✓ All tests complete!")
    print("=" * 50)
