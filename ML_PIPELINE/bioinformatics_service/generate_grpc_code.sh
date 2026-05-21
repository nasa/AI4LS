#!/bin/bash

echo "Cleaning generated directory..."
rm -rf generated/*
mkdir -p generated
touch generated/__init__.py

echo "Generating gRPC code with protoc..."
python3 -m grpc_tools.protoc \
  -I./proto \
  --python_out=./generated \
  --grpc_python_out=./generated \
  ./proto/bioinformatics_service.proto

if [ $? -ne 0 ]; then
    echo "Code generation failed!"
    exit 1
fi

echo "Fixing imports..."
sed -i.bak 's/^import bioinformatics_service_pb2/from . import bioinformatics_service_pb2/' generated/bioinformatics_service_pb2_grpc.py 2>/dev/null || \
sed -i 's/^import bioinformatics_service_pb2/from . import bioinformatics_service_pb2/' generated/bioinformatics_service_pb2_grpc.py

echo "gRPC code generation complete!"
echo ""
echo "Generated files:"
ls -la generated/
