#!/usr/bin/env python3
"""
Fix the import statement in data_service_pb2_grpc.py
The issue: when copied to orchestration_service/generated/, it tries:
  import data_service_pb2
But data_service_pb2 is in the same directory, so it needs:
  from . import data_service_pb2
"""

from pathlib import Path

def fix_grpc_imports():
    print("Fixing protobuf import statements...")
    print("=" * 60)
    
    grpc_file = Path("/Users/jcasalet/Desktop/CODES/NASA/AI4LS/ML_PIPELINE/orchestration_service/generated/data_service_pb2_grpc.py")
    
    if not grpc_file.exists():
        print(f"✗ File not found: {grpc_file}")
        return False
    
    print(f"✓ Found: {grpc_file}")
    
    with open(grpc_file, 'r') as f:
        content = f.read()
    
    # Fix the problematic import
    # Change: import data_service_pb2 as data__service__pb2
    # To:     from . import data_service_pb2 as data__service__pb2
    
    if 'import data_service_pb2 as data__service__pb2' in content:
        print("\n✓ Found problematic import statement")
        content = content.replace(
            'import data_service_pb2 as data__service__pb2',
            'from . import data_service_pb2 as data__service__pb2'
        )
        print("✓ Fixed to use relative import")
    else:
        print("\nℹ️  Import statement already uses relative import or different format")
        print("   Checking for other patterns...")
        
        # Check for other import patterns
        if 'import data_service_pb2' in content:
            print("✓ Found: import data_service_pb2")
            content = content.replace(
                'import data_service_pb2',
                'from . import data_service_pb2'
            )
            print("✓ Fixed to use relative import")
    
    # Write back
    with open(grpc_file, 'w') as f:
        f.write(content)
    
    print(f"\n✓ File updated successfully!")
    print("=" * 60)
    print("\nNow try running your pipeline:")
    print("  python new_multi_pipeline_updated.py --tissue liver -tc 'Factor Value[Spaceflight]'")
    
    return True

if __name__ == "__main__":
    import sys
    success = fix_grpc_imports()
    sys.exit(0 if success else 1)
