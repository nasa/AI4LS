# orchestration-service/src/clients/data_client.py
import grpc
from generated import data_service_pb2, data_service_pb2_grpc
from typing import List, Dict

class DataServiceClient:
    """Client for Data Service gRPC calls"""
    
    def __init__(self, service_url: str = "data-service:50051"):
        self.channel = grpc.insecure_channel(service_url)
        self.stub = data_service_pb2_grpc.DataServiceStub(self.channel)
    
    async def validate_dataset(self, dataset_content: bytes, format: str = "csv") -> Dict:
        """Validate a dataset"""
        request = data_service_pb2.ValidateRequest(
            dataset_content=dataset_content,
            format=format
        )
        
        response = self.stub.ValidateDataset(request)
        
        return {
            "is_valid": response.is_valid,
            "errors": list(response.errors),
            "warnings": list(response.warnings),
            "dataset_id": response.info.dataset_id,
            "num_rows": response.info.num_rows,
            "num_columns": response.info.num_columns,
            "columns": [
                {
                    "name": col.name,
                    "dtype": col.dtype,
                    "null_count": col.null_count
                }
                for col in response.info.columns
            ]
        }
    
    async def apply_transformations(
        self, 
        dataset_id: str, 
        transformations: List[Dict]
    ) -> Dict:
        """Apply transformations to dataset"""
        transform_messages = []
        for t in transformations:
            transform_messages.append(
                data_service_pb2.Transformation(
                    type=t["type"],
                    columns=t["columns"],
                    params=t.get("params", {})
                )
            )
        
        request = data_service_pb2.TransformRequest(
            dataset_id=dataset_id,
            transformations=transform_messages
        )
        
        response = self.stub.ApplyTransformation(request)
        
        return {
            "success": response.success,
            "transformed_dataset_id": response.transformed_dataset_id,
            "error_message": response.error_message,
            "num_rows": response.transformed_info.num_rows,
            "num_columns": response.transformed_info.num_columns
        }
    
    def close(self):
        """Close the gRPC channel"""
        self.channel.close()
