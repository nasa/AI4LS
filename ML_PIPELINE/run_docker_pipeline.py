# run_docker_pipeline.py
import requests
import json
import time
import sys
import argparse
import grpc
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def get_feature_importance(model_id, dataset_id, fi_methods, random_state):
    fi_service_path = Path(__file__).parent / "feature_importance_service"
    sys.path.insert(0, str(fi_service_path))

    from generated import feature_importance_service_pb2, feature_importance_service_pb2_grpc

    # Use model ID WITHOUT .joblib extension

    channel = grpc.insecure_channel('localhost:50053')
    stub = feature_importance_service_pb2_grpc.FeatureImportanceServiceStub(channel)

    request = feature_importance_service_pb2.ImportanceRequest(
        model_id=model_id,
        dataset_id=dataset_id,
        methods=fi_methods,
        params={"n_repeats": "5", "random_state": str(random_state)}
    )

    print(f"Computing feature importance for model: {model_id}")

    try:
        response = stub.ComputeImportance(request)
        
        print(f"Success: {response.success}")
    
        if response.success:
            for method, importances in response.importances.items():
                print(f"\n{method.upper()} Feature Importance:")
                print(f"  Metadata: {dict(importances.metadata)}")
                print(f"  Top 10 Features:")
                for i, score in enumerate(importances.scores[:10]):
                    print(f"    {i+1}. {score.feature_name}: {score.importance:.6f}")
        else:
            print(f"Error: {response.error_message}")
        
    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"Error: {e}")



def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("STEP 1: Health Check")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    result = response.json()
    print(json.dumps(result, indent=2))

    if result['status'] != 'healthy':
        print("\n⚠️  Services not healthy!")
        return False

    print("\n✓ All services healthy")
    return True


def download_dataset(args):
    """Download dataset from NASA OSDR via the data-service"""
    print("\n" + "=" * 60)
    print("STEP 2: Download Dataset from NASA OSDR")
    print("=" * 60)

    print(f"  OSD ID   : OSD-{args.osd_id}")
    print(f"  Patterns : {args.patterns}")
    print(f"  Factor Name : {args.factor_name}")
    print(f"  Factor Values : {args.factor_values}")

    payload = {
        "osd_id": args.osd_id,
        "patterns": args.patterns,
        "exclude_columns": args.exclude_columns,
        "factor_name": args.factor_name,
        "factor_values": args.factor_values,
        "min_features": args.min_features,
        "cv_step": args.cv_step,
    }

    response = requests.post(f"{BASE_URL}/api/datasets/download", json=payload)

    if response.status_code != 200:
        print(f"\n❌ Download request failed: HTTP {response.status_code}")
        print(response.text)
        return None, None

    result = response.json()
    print(f"Valid: {result['is_valid']}")

    if result.get('errors'):
        print(f"Errors: {result['errors']}")

    if result.get('warnings'):
        print(f"Warnings: {result['warnings']}")

    columns = []
    if result.get('dataset_info'):
        dataset_id = result['dataset_info']['dataset_id']
        print(f"Dataset ID: {dataset_id}")
        print(f"Shape: {result['dataset_info']['num_rows']} x {result['dataset_info']['num_columns']}")
        #print(f"\nColumns:")
        for col in result['dataset_info']['columns']:
            #print(f"  - {col['name']} ({col['dtype']})")
            columns.append(col['name'])

        return dataset_id, columns

    return None, None


def upload_dataset(file_name, exclude_columns, cv_step):
    """Upload and validate dataset from local file"""
    print("\n" + "=" * 60)
    print("STEP 2: Upload and Validate Dataset")
    print("=" * 60)

    with open(file_name, 'rb') as f:
        #files = {'file': ('data.csv', f, 'text/csv')}
        files = {'file': ('data.csv', f, 'csv')}
        response = requests.post(f"{BASE_URL}/api/datasets/validate", files=files)

    result = response.json()
    print(f"Valid: {result['is_valid']}")

    columns = []
    if result['dataset_info']:
        dataset_id = result['dataset_info']['dataset_id']
        print(f"Dataset ID: {dataset_id}")
        print(f"Shape: {result['dataset_info']['num_rows']} x {result['dataset_info']['num_columns']}")
        #print(f"\nColumns:")
        for col in result['dataset_info']['columns']:
            #print(f"  - {col['name']} ({col['dtype']})")
            columns.append(col['name'])

        return dataset_id, columns

    return None, None

def run_pipeline(dataset_id, target_column, sample_column, columns, task_type, algorithm, test_size, trans_list, factor_name, factor_values, min_features, fi_methods, exclude_columns, cv_step):
    """Run full ML pipeline"""
    print("\n" + "=" * 60)
    print("STEP 3: Run ML Pipeline")
    print("=" * 60)

    # remove sample and target from features
    if target_column in columns:
        columns.remove(target_column)

    transformations = []
    if 'l' in trans_list:
        transformations.append({"type": "log", "columns": columns, "params": {}})
    if 'n' in trans_list:
        transformations.append({"type": "normalize", "columns": columns, "params": {}})
    if 's' in trans_list:
        transformations.append({"type": "standardize", "columns": columns, "params": {}})
    if 't' in trans_list:
        transformations.append({"type": "tpm", "columns": columns, "params": {}})

    payload = {
        "dataset_id": dataset_id,
        "config": {
            "target_column": target_column,
            "task_type": task_type,
            "feature_columns": [],
            "transformations": transformations,
            "algorithm": algorithm,
            "hyperparameters": {},
            "metrics": ["accuracy", "f1_score"],
            "test_size": test_size,
            "random_state": 42,
            "factor_name": factor_name,
            "factor_values": factor_values,
            "min_features": min_features,
            "fi_methods": fi_methods,
            "exclude_columns": exclude_columns,
            "cv_step": cv_step,
        }
    }

    print("\nConfiguration:")
    print(f"  Algorithm: {payload['config']['algorithm']}")
    print(f"  Target: {payload['config']['target_column']}")
    print(f"  Features: All except target")
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

            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"[{bar}] {percent:3d}% | {status:12s} | {message}")

            if progress.get('test_metrics'):
                final_metrics = progress['test_metrics']

            if progress.get('model_id'):
                model_id = progress['model_id']

            if progress.get('error'):
                print(f"  ❌ Error: {progress['error']}")
                return None, None

    print("-" * 60)
    return model_id, final_metrics


def get_model_info(model_id):
    """Get model information"""
    print("\n" + "=" * 60)
    print("STEP 4: Get Model Information")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/api/models/{model_id}")
    result = response.json()

    print(f"Model ID: {result['model_id']}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Task Type: {result['task_type']}")
    print(f"Target: {result['target_column']}")
    print(f"Features: {len(result['feature_columns'])} columns")
    print(f"Samples: {result['num_samples']}")

    print("\nTest Metrics:")
    for metric, value in result['test_metrics'].items():
        print(f"  {metric}: {value:.4f}")


def list_models():
    """List all models"""
    print("\n" + "=" * 60)
    print("STEP 5: List All Models")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/api/models?limit=10")
    result = response.json()

    print(f"Total models: {result['total_count']}")

    if result['models']:
        print("\nModels:")
        for i, model in enumerate(result['models'], 1):
            acc = model['test_metrics'].get('accuracy', 'N/A')
            print(f"{i}. {model['model_id']} | {model['algorithm']} | Acc: {acc}")

def list_of_strings(arg):
    return arg.split(",")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', '--operation', help='download|upload', default=None, required=True)
    parser.add_argument('-tt', '--task_type', help='classification|regression', default=None, required=True)
    parser.add_argument('-al', '--algorithm', help='name of ML algorithm', default=None, required=True)
    parser.add_argument('-ts', '--test_size', help='decimal amount of data for testing', default=0.2, required=False)
    parser.add_argument('-if', '--input_file', help='custom file to provide', default=None, required=False)
    parser.add_argument('-oi', '--osd_id', help='OSDR dataset ID', default=None, required=False)
    parser.add_argument('-tl', '--trans_list', help='list of transformations', default=[], type=list_of_strings, required=False)
    parser.add_argument('-ec', '--exclude_columns', help='list of columns to exclude', default=[], type=list_of_strings, required=False)
    parser.add_argument('-cs', '--cv_step', help='float value to start coef of variation dim reduction', default=0.25, type=float, required=False)
    parser.add_argument('-pa', '--patterns', help='string patterns to identify RNA-seq files', default=['unnormalized', 'RSEM'], type=list_of_strings, required=False)
    parser.add_argument('-sc', '--sample_column', help='sample column name', default=None, required=False)
    parser.add_argument('-tc', '--target_column', help='name of target column', default=None, required=False)
    parser.add_argument('-fl', '--factor_name', help='metadata factor name', default='Factor Value[Spaceflight]', required=False)
    parser.add_argument('-fv', '--factor_values', help='metadata factor values', type=list_of_strings, default=['Ground Control', 'Space Flight'], required=False)
    parser.add_argument('-mf', '--min_features', help='minimum number of features to keep after CVS dimensionality reduction', default=1000, required=False)
    parser.add_argument('-fi', '--fi_methods', help='list of feature importance methods to use', type=list_of_strings, default=['recursive', 'permutation', 'built_in'] , required=False)
    parser.add_argument('-rs', '--random_state', help='random state(seed)', type=int, default=42, required=False)

    
    args = parser.parse_args()

    operation = args.operation
    osd_id        = args.osd_id 
    target_column = args.target_column
    sample_column = args.sample_column 
    task_type     = args.task_type 
    algorithm     = args.algorithm 
    test_size     = args.test_size 
    trans_list    = list(args.trans_list)
    exclude_columns    = list(args.exclude_columns)
    cv_step    = float(args.cv_step)
    patterns      = list(args.patterns) 
    input_file = args.input_file
    factor_name = args.factor_name
    factor_values = args.factor_values
    min_features = args.min_features
    fi_methods = list(args.fi_methods)
    random_state = int(args.random_state)

    if operation == 'upload':
        if input_file is None:
            print("must provide input_file if using upload operation")
            sys.exit(1)
        if target_column is None:
            print("must provide target_column if using upload operation")
            sys.exit(1)
        if sample_column is None:
            print("must provide sample_column if using upload operation")
            sys.exit(1)
    elif operation == 'download':
        if osd_id is None:
            print("must provide osd_id if using download operation")
            sys.exit(1)
        if target_column is None:
            target_column = factor_name
            print(f"  ℹ️  Using factor_name as target_column: {target_column}")
    # Add after argument parsing
    print(f"DEBUG - Arguments:")
    print(f"  operation: {operation}")
    print(f"  osd_id: {osd_id}")
    print(f"  target_column: {target_column}")
    print(f"  task_type: {task_type}")
    print(f"  algorithm: {algorithm}")
    print(f"  transformations: {trans_list}")
    print(f"  factor_name: {factor_name}")
    print(f"  factor_values: {factor_values}")
    print(f"  min_features: {min_features}")
    print()

    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 8 + "DOCKER ML PIPELINE TEST" + " " * 27 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Step 1: Health check
    if not test_health():
        print("\n❌ Services not healthy. Check logs:")
        print("   docker-compose logs")
        return

    time.sleep(2)

    # Step 2: Get dataset (download or upload)
    if operation == 'download':
        dataset_id, columns = download_dataset(args)
    elif operation == 'upload':
        dataset_id, columns = upload_dataset(input_file, exclude_columns, cv_step)
    else:
        print("unknown operation: ", operation)
        sys.exit(1)

    if not dataset_id:
        print("\n❌ Failed to load dataset")
        return

    print(f"\n✓ Dataset ready: {dataset_id}")

    # Step 3: Run pipeline
    model_id, metrics = run_pipeline(
        dataset_id, target_column, sample_column, 
        columns, task_type, algorithm, test_size, trans_list,
        factor_name, factor_values,min_features, fi_methods,
        exclude_columns, cv_step
    )
    if not model_id:
        print("\n❌ Pipeline failed")
        return

    print(f"\n✓ Model trained: {model_id}")
    if metrics:
        if task_type == 'classification':
            print(f"✓ Test Accuracy: {metrics.get('accuracy', 0):.4f}")
        elif task_type == 'regression':
            print(f"✓ Test RMSE: {metrics.get('rmse', 0):.4f}")

    # Step 4: Get model info
    get_model_info(model_id)

    # Step 5: Get feature importance
    get_feature_importance(model_id, dataset_id, fi_methods, random_state) 

    print("\n" + "=" * 60)
    print("  - Access Swagger UI: http://localhost:8000/docs")
    print("  - View logs: docker-compose logs -f")
    print("  - Stop services: docker-compose down")
    print()


if __name__ == "__main__":
    main()
