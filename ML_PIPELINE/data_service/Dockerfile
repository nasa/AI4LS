# data-service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

COPY Mus_musculus.GRCm39.115.gtf.gz . 

# Install grpcio-tools explicitly first, then the rest
RUN pip install --no-cache-dir grpcio-tools==1.62.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy proto files
COPY proto/ ./proto/

# Verify grpc_tools is installed and generate code
RUN python -c "import grpc_tools.protoc; print('grpc_tools found')" && \
    mkdir -p generated && \
    touch generated/__init__.py && \
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

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); channel.close()" || exit 1

# Run server
CMD ["python", "-m", "src.server"]
