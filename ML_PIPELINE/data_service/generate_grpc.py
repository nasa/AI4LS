# generate_grpc.py
import subprocess
import sys
import os

# Create output directory
os.makedirs("generated", exist_ok=True)

# Create __init__.py
open("generated/__init__.py", "w").close()

# Run protoc directly
result = subprocess.run([
    sys.executable, "-m", "grpc_tools.protoc",
    "-I./proto",
    "--python_out=./generated",
    "--grpc_python_out=./generated",
    "./proto/data_service.proto"
], capture_output=True, text=True)

if result.returncode == 0:
    print("✓ gRPC code generation complete!")
    print("\nGenerated files:")
    for f in os.listdir("generated"):
        print(f"  - {f}")
else:
    print("✗ Error generating gRPC code:")
    print(result.stderr)
    sys.exit(1)
