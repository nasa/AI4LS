# generate_grpc_code.sh
#!/bin/bash

echo "Cleaning generated directory..."
rm -rf generated/*
mkdir -p generated
touch generated/__init__.py

echo "Generating gRPC code with protoc..."
protoc -I./proto \
  --python_out=./generated \
  --grpc_out=./generated \
  --plugin=protoc-gen-grpc=$(which grpc_python_plugin) \
  ./proto/data_service.proto

if [ $? -ne 0 ]; then
    echo "✗ Code generation failed!"
    exit 1
fi

echo "Fixing imports..."
# Fix the import in the gRPC file to use relative imports
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  sed -i '' 's/^import data_service_pb2/from . import data_service_pb2/' generated/data_service_pb2_grpc.py
else
  # Linux
  sed -i 's/^import data_service_pb2/from . import data_service_pb2/' generated/data_service_pb2_grpc.py
fi

echo "✓ gRPC code generation complete!"
echo ""
echo "Generated files:"
ls -la generated/
