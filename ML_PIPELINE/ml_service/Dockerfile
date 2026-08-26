FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#libomp-dev 

COPY requirements.txt .
#RUN pip install --no-cache-dir grpcio-tools==1.62.0 && \
RUN  pip install --no-cache-dir -r requirements.txt

# Copy proto files (relative to ml_service/ directory)
COPY proto/ ./proto/

# Create generated directory
#RUN mkdir -p generated && \
#    touch generated/__init__.py

# Generate ml_service protobuf files from proto
RUN mkdir -p generated && \
    touch generated/__init__.py && \
    python -m grpc_tools.protoc \
    	-I./proto \
    	--python_out=./generated \
    	--grpc_python_out=./generated \
    	./proto/ml_service.proto && \
    sed -i 's/^import ml_service_pb2/from . import ml_service_pb2/' generated/ml_service_pb2_grpc.py

# Copy pre-generated data_service protobuf files from host
# Paths are relative to Dockerfile 
COPY generated/ ./generated/

ENV PYTHONPATH=/app

# Copy source code (relative to ml_service/ directory)
COPY src/ ./src/

RUN mkdir -p models

EXPOSE 50052

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50052'); channel.close()" || exit 1

CMD ["python", "-m", "src.server"]
