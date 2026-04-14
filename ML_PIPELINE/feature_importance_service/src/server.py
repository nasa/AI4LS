# feature_importance_service/src/server.py
import grpc
from concurrent import futures
import logging
import os

from generated import feature_importance_service_pb2_grpc
from src.service import FeatureImportanceServiceImpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve():
    """Start the gRPC server"""
    # Get data service URL from environment
    data_service_url = os.environ.get('DATA_SERVICE_URL', 'data_service:50051')
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    feature_importance_service_pb2_grpc.add_FeatureImportanceServiceServicer_to_server(
        FeatureImportanceServiceImpl(data_service_url=data_service_url), 
        server
    )
    
    port = "50053"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    logger.info(f"Feature Importance Service started on port {port}")
    logger.info(f"Connecting to Data Service at: {data_service_url}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
