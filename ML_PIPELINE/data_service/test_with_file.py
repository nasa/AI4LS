# test_with_file.py
import grpc
from generated import data_service_pb2, data_service_pb2_grpc

def test_with_csv_file(filepath):
    """Test the gRPC service with a real CSV file"""
    
    # Connect to server
    channel = grpc.insecure_channel('localhost:50051')
    stub = data_service_pb2_grpc.DataServiceStub(channel)
    
    # Read the CSV file
    with open(filepath, 'rb') as f:
        csv_bytes = f.read()
    
    print(f"Testing with file: {filepath}")
    print(f"File size: {len(csv_bytes)} bytes\n")
    
    # 1. Validate the dataset
    print("=" * 50)
    print("STEP 1: Validating Dataset")
    print("=" * 50)
    
    request = data_service_pb2.ValidateRequest(
        dataset_content=csv_bytes,
        format="csv"
    )
    
    response = stub.ValidateDataset(request)
    
    print(f"Valid: {response.is_valid}")
    
    if response.errors:
        print("Errors:")
        for error in response.errors:
            print(f"  - {error}")
    
    if response.warnings:
        print("Warnings:")
        for warning in response.warnings:
            print(f"  - {warning}")
    
    print(f"\nDataset ID: {response.info.dataset_id}")
    print(f"Rows: {response.info.num_rows}")
    print(f"Columns: {response.info.num_columns}")
    print(f"Size: {response.info.size_bytes:,} bytes")
    
    print("\nColumn Details:")
    for col in response.info.columns:
        print(f"  {col.name}:")
        print(f"    Type: {col.dtype}")
        print(f"    Nulls: {col.null_count}")
        print(f"    Sample: {col.sample_values[:3]}")
    
    if not response.is_valid:
        channel.close()
        return
    
    dataset_id = response.info.dataset_id
    
    # 2. Apply transformations
    print("\n" + "=" * 50)
    print("STEP 2: Applying Transformations")
    print("=" * 50)
    
    transformations = [
        {"type": "log", "columns": ["income", "score"]},
        {"type": "standardize", "columns": ["income", "score", "age"]},
        {"type": "one_hot_encode", "columns": ["category"]}
    ]
    
    print("Transformations to apply:")
    for i, t in enumerate(transformations, 1):
        print(f"  {i}. {t['type']} on {t['columns']}")
    
    transform_request = data_service_pb2.TransformRequest(
        dataset_id=dataset_id,
        transformations=[
            data_service_pb2.Transformation(
                type=t["type"],
                columns=t["columns"]
            )
            for t in transformations
        ]
    )
    
    transform_response = stub.ApplyTransformation(transform_request)
    
    print(f"\nSuccess: {transform_response.success}")
    
    if transform_response.success:
        print(f"Transformed Dataset ID: {transform_response.transformed_dataset_id}")
        print(f"New shape: {transform_response.transformed_info.num_rows} rows x {transform_response.transformed_info.num_columns} columns")
        
        print("\nNew columns after transformation:")
        for col in transform_response.transformed_info.columns:
            print(f"  - {col.name} ({col.dtype})")
        
        # 3. Stream the transformed dataset
        print("\n" + "=" * 50)
        print("STEP 3: Streaming Transformed Dataset")
        print("=" * 50)
        
        stream_request = data_service_pb2.StreamRequest(
            dataset_id=transform_response.transformed_dataset_id,
            chunk_size=3  # 3 rows per chunk
        )
        
        print(f"Requesting chunks of 3 rows each...\n")
        
        total_bytes = 0
        for chunk in stub.StreamDataset(stream_request):
            total_bytes += len(chunk.data)
            print(f"Chunk {chunk.chunk_number}: {len(chunk.data)} bytes")
            
            # Show first chunk's data as preview
            if chunk.chunk_number == 0:
                print("  Preview of first chunk:")
                preview = chunk.data.decode('utf-8')[:200]
                print(f"  {preview}...")
            
            if chunk.is_final:
                print(f"\n✓ Streaming complete")
                print(f"  Total chunks: {chunk.chunk_number + 1}")
                print(f"  Total bytes: {total_bytes:,}")
    else:
        print(f"Error: {transform_response.error_message}")
    
    # 4. Get dataset info
    print("\n" + "=" * 50)
    print("STEP 4: Fetching Dataset Info")
    print("=" * 50)
    
    info_request = data_service_pb2.DatasetInfoRequest(
        dataset_id=dataset_id
    )
    
    info_response = stub.GetDatasetInfo(info_request)
    
    print(f"Dataset: {info_response.dataset_id}")
    print(f"Dimensions: {info_response.num_rows} x {info_response.num_columns}")
    print(f"Memory usage: {info_response.size_bytes:,} bytes")
    
    channel.close()
    print("\n" + "=" * 50)
    print("✓ All tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "sample_data.csv"
    
    try:
        test_with_csv_file(filepath)
    except grpc.RpcError as e:
        print(f"\n✗ gRPC Error: {e.code()}")
        print(f"  Details: {e.details()}")
        print("\nMake sure the server is running: python -m src.server")
    except FileNotFoundError:
        print(f"\n✗ File not found: {filepath}")
        print("  Create a sample file or specify a valid path")
