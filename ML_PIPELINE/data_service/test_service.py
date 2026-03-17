# test_service.py
import sys
sys.path.insert(0, './generated')

from generated import data_service_pb2, data_service_pb2_grpc
import grpc
from concurrent import futures
import pandas as pd
from io import BytesIO

# Import your service implementation
from src.service import DataServiceImpl

def test_local():
    """Test the service locally without gRPC"""
    print("Testing service implementation...")
    
    service = DataServiceImpl()
    
    # Create a simple test dataset
    df = pd.DataFrame({
        'age': [25, 30, 35, 40],
        'income': [50000, 60000, 70000, 80000],
        'category': ['A', 'B', 'A', 'C']
    })
    
    # Convert to CSV bytes
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()
    
    # Create validation request
    request = data_service_pb2.ValidateRequest(
        dataset_content=csv_bytes,
        format="csv"
    )
    
    # Call the service
    response = service.ValidateDataset(request, None)
    
    print(f"✓ Validation successful: {response.is_valid}")
    print(f"  Dataset ID: {response.info.dataset_id}")
    print(f"  Rows: {response.info.num_rows}")
    print(f"  Columns: {response.info.num_columns}")
    
    if response.is_valid:
        # Test transformation
        transform_request = data_service_pb2.TransformRequest(
            dataset_id=response.info.dataset_id,
            transformations=[
                data_service_pb2.Transformation(
                    type="log",
                    columns=["income", "age"]
                ),
                data_service_pb2.Transformation(
                    type="standardize",
                    columns=["income", "age"]
                )
            ]
        )
        
        transform_response = service.ApplyTransformation(transform_request, None)
        print(f"✓ Transformation successful: {transform_response.success}")
        print(f"  New dataset ID: {transform_response.transformed_dataset_id}")
        print(f"  New columns: {transform_response.transformed_info.num_columns}")

if __name__ == "__main__":
    test_local()
