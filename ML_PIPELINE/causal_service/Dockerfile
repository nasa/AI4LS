# causal_service: CRISP v1.1 (NASA FDL 2020 Astronaut Health) as a gRPC service.
# CRISP needs Python 3.7 and 2021-era libraries, hence its own image.
FROM python:3.7-slim

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade "pip<24.1" && \
    pip install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY crisp/ crisp/
COPY proto/ proto/
COPY src/ src/

# gRPC stubs are generated here, with this image's grpcio-tools/protobuf, so they always match.
# (orchestration still needs its own copy in src/generated/, made the usual way.)
RUN python -m grpc_tools.protoc -I proto --python_out=src --grpc_python_out=src proto/causal_service.proto && \
    sed -i 's/^import causal_service_pb2 as/from src import causal_service_pb2 as/' src/causal_service_pb2_grpc.py && \
    touch src/__init__.py

RUN mkdir -p /app/results
EXPOSE 50055
CMD ["python", "-m", "src.server"]
