"""causal_service gRPC server:  python -m src.server"""
import logging
import os
from concurrent import futures

import grpc

from src import causal_service_pb2_grpc
from src.service import CausalServiceImpl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_MESSAGE = 500 * 1024 * 1024


def serve():
    port = os.environ.get("CAUSAL_PORT", "50055")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.max_send_message_length", MAX_MESSAGE),
                                  ("grpc.max_receive_message_length", MAX_MESSAGE)])
    causal_service_pb2_grpc.add_CausalServiceServicer_to_server(CausalServiceImpl(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Causal service (CRISP) listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
