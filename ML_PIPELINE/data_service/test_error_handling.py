# test_error_handling.py
import grpc
from generated import data_service_pb2, data_service_pb2_grpc

def test_invalid_data():
    """Test error handling with invalid data"""
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = data_service_pb2_grpc.DataServiceStub(channel)
    
    print("Testing with invalid JSON data as CSV...")
    invalid_data = b'{"this": "is not", "valid": "csv"}'
    
    request = data_service_pb2.ValidateRequest(
        dataset_content=invalid_data,
        format="csv"
    )
    
    response = stub.ValidateDataset(request)
    
    print(f"Valid: {response.is_valid}")
    print(f"Errors: {list(response.errors)}")
    
    print("\n" + "-" * 50)
    print("Testing with empty dataset...")
    
    empty_data = b''
    request = data_service_pb2.ValidateRequest(
        dataset_content=empty_data,
        format="csv"
    )
    
    response = stub.ValidateDataset(request)
    print(f"Valid: {response.is_valid}")
    print(f"Errors: {list(response.errors)}")
    
    channel.close()
    print("\n✓ Error handling tests complete")

if __name__ == "__main__":
    test_invalid_data()
