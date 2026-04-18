# orchestration-service/src/clients/data_client.py
import grpc
from typing import List, Dict, Optional
from generated import data_service_pb2, data_service_pb2_grpc
import logging

logger = logging.getLogger(__name__)

class DataServiceClient:
    """Client for Data Service gRPC communication"""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[data_service_pb2_grpc.DataServiceStub] = None
        self._connect()

    def _connect(self):
        """Establish connection to Data Service"""
        try:
            self.channel = grpc.insecure_channel(
                self.service_url,
                options=[
                    ('grpc.max_send_message_length',    50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ]
            )
            self.stub = data_service_pb2_grpc.DataServiceStub(self.channel)
            logger.info(f"Connected to Data Service at {self.service_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Data Service: {e}")
            raise

    def _validation_result_to_dict(self, response) -> Dict:
        """Convert a ValidationResult gRPC response to a dict."""
        return {
            "is_valid": response.is_valid,
            "errors": list(response.errors),
            "warnings": list(response.warnings),
            "dataset_info": {
                "dataset_id": response.dataset_id,
                "num_rows": response.dataset_info.num_rows,
                "num_columns": response.dataset_info.num_columns,
                "size_bytes": response.dataset_info.size_bytes,
                "columns": [
                    {
                        "name": col.name,
                        "dtype": col.dtype,
                        "null_count": col.null_count,
                        "sample_values": list(col.sample_values)
                    }
                    for col in response.dataset_info.columns
                ]
            } if response.is_valid else None
        }

    def upload_dataset(
        self, 
        dataset_content: bytes, 
        format: str = "csv",
        dataset_id: str = "",
        exclude_columns: List[str] = []
    ) -> Dict:
        """Upload and validate a dataset"""
        try:
            request = data_service_pb2.UploadRequest(
                file_content=dataset_content,  # Note: file_content, not dataset_content
                format=format,
                dataset_id="",  # Empty = auto-generate
                exclude_columns=exclude_columns
            )

            response = self.stub.UploadDataset(request)  # Note: UploadDataset, not ValidateDataset
            return self._validation_result_to_dict(response)
            
        
        except grpc.RpcError as e:
            logger.error(f"gRPC error in upload_dataset: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Error in upload_dataset: {e}", exc_info=True)
            raise

    def validate_dataset(self, dataset_content: bytes, format: str = "csv", exclude_columns: List[str] = []) -> Dict:
        """Validate a dataset (does not store — use upload_dataset to store)"""
        try:
            request = data_service_pb2.ValidateRequest(
                dataset_content=dataset_content,
                format=format,
                exclude_columns=exclude_columns
            )
            response = self.stub.ValidateDataset(request)
            return self._validation_result_to_dict(response)
        except grpc.RpcError as e:
            logger.error(f"gRPC error in validate_dataset: {e.code()} - {e.details()}")
            raise

    def download_dataset(self, osd_id: str, patterns: List[str],
                         dataset_id: str, factor_name: str, factor_values: List[str], min_features: int, exclude_columns: List[str]) -> Dict:
        """Download a dataset from NASA OSDR"""
        try:
            request = data_service_pb2.DownloadRequest(
                osd_id=osd_id,
                patterns=patterns,
                dataset_id=dataset_id,
                factor_name=factor_name,
                factor_values=factor_values,
                min_features=min_features,
                exclude_columns=exclude_columns
            )
            response = self.stub.DownloadDataset(request)
            return self._validation_result_to_dict(response)
        except grpc.RpcError as e:
            logger.error(f"gRPC error in download_dataset: {e.code()} - {e.details()}")
            raise

    def apply_transformations(self, dataset_id: str,
                               transformations: List[Dict]) -> Dict:
        """Apply transformations to a dataset"""
        try:
            transform_messages = [
                data_service_pb2.Transformation(
                    type=t["type"],
                    columns=t["columns"],
                    params=t.get("params", {})
                )
                for t in transformations
            ]
            request = data_service_pb2.TransformRequest(
                dataset_id=dataset_id,
                transformations=transform_messages
            )
            response = self.stub.ApplyTransformation(request)
            return {
                "success": response.success,
                "transformed_dataset_id": response.transformed_dataset_id if response.success else None,
                "error_message": response.error_message if not response.success else None,
                "transformed_info": {
                    "dataset_id": response.transformed_info.dataset_id,
                    "num_rows": response.transformed_info.num_rows,
                    "num_columns": response.transformed_info.num_columns,
                    "size_bytes": response.transformed_info.size_bytes,
                    "columns": [
                        {
                            "name": col.name,
                            "dtype": col.dtype,
                            "null_count": col.null_count,
                            "sample_values": list(col.sample_values)
                        }
                        for col in response.transformed_info.columns
                    ]
                } if response.success else None
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error in apply_transformations: {e.code()} - {e.details()}")
            raise

    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get information about a dataset"""
        try:
            request = data_service_pb2.DatasetInfoRequest(dataset_id=dataset_id)
            response = self.stub.GetDatasetInfo(request)
            return {
                "dataset_id": response.dataset_id,
                "num_rows": response.num_rows,
                "num_columns": response.num_columns,
                "size_bytes": response.size_bytes,
                "columns": [
                    {
                        "name": col.name,
                        "dtype": col.dtype,
                        "null_count": col.null_count,
                        "sample_values": list(col.sample_values)
                    }
                    for col in response.columns
                ]
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC error in get_dataset_info: {e.code()} - {e.details()}")
            raise

    def health_check(self) -> bool:
        """Check if Data Service is healthy"""
        try:
            import pandas as pd
            from io import BytesIO
            test_df = pd.DataFrame({'test': [1, 2, 3]})
            csv_buffer = BytesIO()
            test_df.to_csv(csv_buffer, index=False)
            request = data_service_pb2.UploadRequest(
                file_content=csv_buffer.getvalue(),
                format="csv"
            )
            response = self.stub.UploadDataset(request, timeout=300000000)
            return response.is_valid
        except grpc.RpcError as e:
            logger.error(f"Health check failed: {e.code()} - {e.details()}")
            return False
        except Exception as e:
            logger.error(f"Health check exception: {e}")
            return False

    def close(self):
        """Close the gRPC channel"""
        if self.channel:
            self.channel.close()
            logger.info("Data Service connection closed")
