# ml-service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including OpenMP for XGBoost
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir grpcio-tools==1.62.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy proto files (includes both ml_service.proto and data_service.proto)
COPY proto/ ./proto/

# Generate gRPC code for both services
RUN mkdir -p generated && \
    touch generated/__init__.py && \
    python -m grpc_tools.protoc \
        -I./proto \
        --python_out=./generated \
        --grpc_python_out=./generated \
        ./proto/ml_service.proto && \
    sed -i 's/^import ml_service_pb2/from . import ml_service_pb2/' generated/ml_service_pb2_grpc.py && \
    python -m grpc_tools.protoc \
        -I./proto \
        --python_out=./generated \
        --grpc_python_out=./generated \
        ./proto/data_service.proto && \
    sed -i 's/^import data_service_pb2/from . import data_service_pb2/' generated/data_service_pb2_grpc.py

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Copy source code
COPY src/ ./src/

# Create models directory
RUN mkdir -p models

# Expose gRPC port
EXPOSE 50052

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50052'); channel.close()" || exit 1

# Run server
CMD ["python", "-m", "src.server"]
