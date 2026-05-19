# src/server.py

import grpc
from concurrent import futures
import logging
import os

from generated import bioinformatics_service_pb2_grpc
from src.service import BioinformaticsServiceImpl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve():
    """Start the gRPC server"""
    results_path = os.environ.get('RESULTS_PATH', '/app/results')
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    bioinformatics_service_pb2_grpc.add_BioinformaticsServiceServicer_to_server(
        BioinformaticsServiceImpl(results_path=results_path),
        server
    )
    
    port = "50054"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    
    logger.info(f"Bioinformatics Service started on port {port}")
    logger.info(f"Results path: {results_path}")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
