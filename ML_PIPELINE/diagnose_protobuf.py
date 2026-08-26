#!/usr/bin/env python3
"""
Diagnose protobuf file issues in local directories
"""

from pathlib import Path

def check_file(file_path, name):
    """Check if file has MultiDatasetServiceStub"""
    if not file_path.exists():
        print(f"  ✗ {name} NOT FOUND")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    has_stub = "class MultiDatasetServiceStub" in content
    status = "✓" if has_stub else "✗"
    print(f"  {status} {name} - {'has MultiDatasetServiceStub' if has_stub else 'MISSING MultiDatasetServiceStub'}")
    
    return has_stub

print("="*60)
print("Checking Protobuf Files")
print("="*60)

files_to_check = [
    ("data_service/generated/data_service_pb2_grpc.py", "data_service (source)"),
    ("orchestration_service/generated/data_service_pb2_grpc.py", "orchestration_service"),
    ("ml_service/generated/data_service_pb2_grpc.py", "ml_service"),
    ("feature_importance_service/generated/data_service_pb2_grpc.py", "feature_importance_service"),
    ("bioinformatics_service/generated/data_service_pb2_grpc.py", "bioinformatics_service"),
]

all_good = True
for file_path, name in files_to_check:
    if not check_file(Path(file_path), name):
        all_good = False

print("\n" + "="*60)

if all_good:
    print("✓ All local files are correct!")
    print("\nThe issue is Docker using cached files.")
    print("\nFix:")
    print("  1. docker-compose down -v  (remove volumes)")
    print("  2. docker system prune -a  (remove unused images)")
    print("  3. docker-compose up --build -d  (rebuild)")
else:
    print("✗ Some local files are missing MultiDatasetServiceStub")
    print("\nFix:")
    print("  1. python regenerate_data_service_proto.py")
    print("  2. python complete_service_fix.py")
    print("  3. docker-compose down -v")
    print("  4. docker-compose up --build -d")
