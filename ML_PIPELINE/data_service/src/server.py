# data-service/src/server.py
import grpc
from concurrent import futures
import logging

from generated import data_service_pb2_grpc
from src.service import DataServiceImpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global reference to service instance so other services can access datasets
service_instance = None

def get_service_instance():
    """Get the global service instance"""
    return service_instance

def serve():
    """Start the gRPC server"""
    global service_instance
    
    #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),    # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
        ]
    )
    
    service_instance = DataServiceImpl()
    
    data_service_pb2_grpc.add_DataServiceServicer_to_server(
        service_instance, server
    )
    
    port = "50051"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    logger.info(f"Data Service started on port {port}")
    logger.info(f"Datasets available: {len(service_instance.datasets)}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
