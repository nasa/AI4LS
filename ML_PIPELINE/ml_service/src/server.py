# ml-service/src/server.py
import grpc
from concurrent import futures
import logging
import os

from generated import ml_service_pb2_grpc
from src.service import MLServiceImpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve():
    """Start the gRPC server"""
    # Get data service URL from environment or use Docker service name
    data_service_url = os.environ.get('DATA_SERVICE_URL', 'data-service:50051')
    
    #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
    )
    
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(
        MLServiceImpl(data_service_url=data_service_url), server
    )
    
    port = "50052"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    logger.info(f"ML Service started on port {port}")
    logger.info(f"Connecting to Data Service at: {data_service_url}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
