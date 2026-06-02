# view_feature_importance.py

import sys
import grpc
from pathlib import Path

sys.path.insert(0, 'feature_importance_service')
from generated import feature_importance_service_pb2, feature_importance_service_pb2_grpc

sys.path.insert(0, 'ml_service')
from src.model_store import ModelStore

def main():
    # List all models that might have feature importance
    model_store = ModelStore(base_path="./models")
    models_data = model_store.list_models()
    
    if not models_data:
        print("No models found")
        return
    
    # Extract model IDs (handle both list of strings and list of dicts)
    if isinstance(models_data[0], dict):
        models = [m['model_id'] for m in models_data]
    else:
        models = models_data
    
    # Connect to feature importance service
    channel = grpc.insecure_channel('localhost:50053')
    stub = feature_importance_service_pb2_grpc.FeatureImportanceServiceStub(channel)
    
    if len(sys.argv) > 1:
        # Show detailed feature importance for specific model
        model_id = sys.argv[1]
        method = sys.argv[2] if len(sys.argv) > 2 else "built_in"
        
        print("\n" + "=" * 100)
        print(f"FEATURE IMPORTANCE: {model_id}")
        print("=" * 100)
        
        # Get cached importance if available
        request = feature_importance_service_pb2.GetImportanceRequest(model_id=model_id)
        
        try:
            response = stub.GetImportance(request)
            
            if response.success and method in response.importances:
                importances = response.importances[method]
                scores = list(importances.scores)
                
                print(f"\nMethod: {method}")
                print(f"Total Features: {len(scores)}")
                
                if importances.metadata:
                    print(f"\nMetadata:")
                    for key, value in importances.metadata.items():
                        print(f"  {key}: {value}")
                
                print(f"\nTop 50 Features by Importance:")
                print(f"{'Rank':<6} {'Feature Name':<60} {'Importance':<15}")
                print("-" * 100)
                
                for score in scores[:50]:
                    print(f"{score.rank:<6} {score.feature_name:<60} {score.importance:<15.8f}")
                
                if len(scores) > 50:
                    print(f"\n... and {len(scores) - 50} more features")
                
                # Statistics
                importances_values = [s.importance for s in scores]
                print(f"\nImportance Statistics:")
                print(f"  Min: {min(importances_values):.8f}")
                print(f"  Max: {max(importances_values):.8f}")
                print(f"  Mean: {sum(importances_values) / len(importances_values):.8f}")
                
                # Top 10 features with high importance
                top_10 = [s for s in scores[:10]]
                print(f"\nTop 10 Most Important Features:")
                for i, score in enumerate(top_10, 1):
                    print(f"  {i:2d}. {score.feature_name}: {score.importance:.8f}")
                
            else:
                if response.success:
                    available_methods = list(response.importances.keys())
                    print(f"Method '{method}' not found for model {model_id}")
                    print(f"Available methods: {available_methods}")
                else:
                    print(f"No cached feature importance found for model {model_id}")
                    print("Run feature importance computation first")
        
        except grpc.RpcError as e:
            print(f"Error: {e.details()}")
    
    else:
        # List all models with summary
        print("\n" + "=" * 100)
        print("MODELS WITH FEATURE IMPORTANCE")
        print("=" * 100)
        print(f"{'Model ID':<25} {'Algorithm':<20} {'Features':<10} {'Status'}")
        print("-" * 100)
        
        for model_id in sorted(models, reverse=True):
            info = model_store.get_model_info(model_id)
            
            if not info:
                continue
                
            algorithm = info.get('algorithm', 'unknown')
            num_features = len(info.get('feature_columns', []))
            
            # Check if importance is cached
            request = feature_importance_service_pb2.GetImportanceRequest(model_id=model_id)
            try:
                response = stub.GetImportance(request)
                if response.success:
                    methods = list(response.importances.keys())
                    status = f"Cached ({', '.join(methods)})"
                else:
                    status = "Not computed"
            except:
                status = "Not computed"
            
            print(f"{model_id:<25} {algorithm:<20} {num_features:<10} {status}")
        
        print("-" * 100)
        print(f"Total: {len(models)} models")
        print("\nTo view feature importance for a specific model, run:")
        print("  python view_feature_importance.py <model_id> [method]")
        print("\nAvailable methods: built_in, permutation, recursive")

if __name__ == "__main__":
    main()
