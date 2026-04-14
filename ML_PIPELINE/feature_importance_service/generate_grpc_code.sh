# feature_importance_service/generate_grpc_code.sh
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
  ./proto/feature_importance_service.proto

if [ $? -ne 0 ]; then
    echo "✗ Code generation failed!"
    exit 1
fi

echo "Fixing imports..."
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' 's/^import feature_importance_service_pb2/from . import feature_importance_service_pb2/' generated/feature_importance_service_pb2_grpc.py
else
  sed -i 's/^import feature_importance_service_pb2/from . import feature_importance_service_pb2/' generated/feature_importance_service_pb2_grpc.py
fi

echo "✓ gRPC code generation complete!"
echo ""
echo "Generated files:"
ls -la generated/
