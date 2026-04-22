# test_feature_importance.py
import grpc
import sys
from pathlib import Path

fi_service_path = Path(__file__).parent / "feature_importance_service"
sys.path.insert(0, str(fi_service_path))

from generated import feature_importance_service_pb2, feature_importance_service_pb2_grpc

# Use model ID WITHOUT .joblib extension
MODEL_ID = "model_ba99bfb6affa"
DATASET_ID = "fc5bf3ef-480a-41f5-856f-52237315fc16"

channel = grpc.insecure_channel('localhost:50053')
stub = feature_importance_service_pb2_grpc.FeatureImportanceServiceStub(channel)

request = feature_importance_service_pb2.ImportanceRequest(
    model_id=MODEL_ID,
    dataset_id=DATASET_ID,
    methods=["built_in", "permutation"],
    params={"n_repeats": "5"}
)

print(f"Computing feature importance for model: {MODEL_ID}")

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
