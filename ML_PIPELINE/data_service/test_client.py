# test_client.py
import grpc
from generated import data_service_pb2, data_service_pb2_grpc
import pandas as pd
from io import BytesIO

def test_grpc_client():
    """Test the gRPC service with a client"""
    
    # Create a connection to the server
    channel = grpc.insecure_channel('localhost:50051')
    stub = data_service_pb2_grpc.DataServiceStub(channel)
    
    # Create test data
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 70000, 80000, 90000],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()
    
    print("1. Validating dataset...")
    request = data_service_pb2.ValidateRequest(
        dataset_content=csv_bytes,
        format="csv"
    )
    
    response = stub.ValidateDataset(request)
    
    print(f"   Valid: {response.is_valid}")
    print(f"   Dataset ID: {response.info.dataset_id}")
    print(f"   Shape: {response.info.num_rows} x {response.info.num_columns}")
    print(f"   Columns: {[col.name for col in response.info.columns]}")
    
    if response.is_valid:
        dataset_id = response.info.dataset_id
        
        print("\n2. Applying transformations...")
        transform_request = data_service_pb2.TransformRequest(
            dataset_id=dataset_id,
            transformations=[
                data_service_pb2.Transformation(
                    type="log",
                    columns=["income"]
                ),
                data_service_pb2.Transformation(
                    type="standardize",
                    columns=["income", "age"]
                ),
                data_service_pb2.Transformation(
                    type="one_hot_encode",
                    columns=["category"]
                )
            ]
        )
        
        transform_response = stub.ApplyTransformation(transform_request)
        
        print(f"   Success: {transform_response.success}")
        print(f"   New dataset ID: {transform_response.transformed_dataset_id}")
        print(f"   New shape: {transform_response.transformed_info.num_rows} x {transform_response.transformed_info.num_columns}")
        
        print("\n3. Testing streaming...")
        stream_request = data_service_pb2.StreamRequest(
            dataset_id=transform_response.transformed_dataset_id,
            chunk_size=2
        )
        
        chunk_count = 0
        for chunk in stub.StreamDataset(stream_request):
            chunk_count += 1
            print(f"   Received chunk {chunk.chunk_number} ({len(chunk.data)} bytes)")
            if chunk.is_final:
                print(f"   Total chunks: {chunk_count}")
    
    channel.close()
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_grpc_client()
