# experiment_service/src/server.py

import grpc
from concurrent import futures
import logging
import os

from generated import experiment_service_pb2_grpc
from src.service import ExperimentServiceImpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve():
    """Start the gRPC server"""
    store_path = os.environ.get('EXPERIMENT_STORE_PATH', '/app/experiments')
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    experiment_service_pb2_grpc.add_ExperimentServiceServicer_to_server(
        ExperimentServiceImpl(store_path=store_path),
        server
    )
    
    port = "50055"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    logger.info(f"Experiment Service started on port {port}")
    logger.info(f"Experiment store path: {store_path}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
