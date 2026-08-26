# feature_importance_service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proto files
COPY proto/ ./proto/

# Generate gRPC code for feature importance service
RUN mkdir -p generated && \
    touch generated/__init__.py && \
    python -m grpc_tools.protoc \
        -I./proto \
        --python_out=./generated \
        --grpc_python_out=./generated \
        ./proto/feature_importance_service.proto && \
    sed -i 's/^import feature_importance_service_pb2/from . import feature_importance_service_pb2/' generated/feature_importance_service_pb2_grpc.py

COPY generated/ ./generated/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Copy source code
COPY src/ ./src/

# Expose gRPC port
EXPOSE 50053

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50053'); channel.close()" || exit 1

# Run server
CMD ["python", "-m", "src.server"]
