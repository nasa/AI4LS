# experiment_service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto files
COPY proto/ ./proto/

# Generate gRPC code
RUN mkdir -p generated && \
    touch generated/__init__.py && \
    python -m grpc_tools.protoc \
        -I./proto \
        --python_out=./generated \
        --grpc_python_out=./generated \
        ./proto/experiment_service.proto && \
    sed -i 's/^import experiment_service_pb2/from . import experiment_service_pb2/' generated/experiment_service_pb2_grpc.py

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Copy source code
COPY src/ ./src/

# Expose gRPC port
EXPOSE 50055

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50055'); channel.close()" || exit 1

# Run server
CMD ["python", "-m", "src.server"]
