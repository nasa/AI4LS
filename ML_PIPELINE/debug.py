#!/usr/bin/env python3
"""
Debug script to see what DownloadMultipleDatasets actually returns
"""

import sys
sys.path.insert(0, '/Users/jcasalet/Desktop/CODES/NASA/AI4LS/ML_PIPELINE/orchestration_service')

from src.clients.data_client import DataServiceClient

print("="*60)
print("Testing DownloadMultipleDatasets")
print("="*60)

client = DataServiceClient(service_url="localhost:50051")

osd_ids = ['47', '48']
print(f"\nRequesting: {osd_ids}")

try:
    dataset_map = client.download_multiple_datasets(
        osd_ids=osd_ids,
        patterns=['unnormalized', 'RSEM'],
        factor_name='Factor Value[Spaceflight]',
        factor_values=['Ground Control', 'Space Flight'],
        min_features=1000,
        cv_step=0.25
    )
    
    print(f"\nDownloaded {len(dataset_map)} datasets:")
    for osd_id, dataset_id in dataset_map.items():
        print(f"  {osd_id} -> {dataset_id}")
    
    print("\nNow checking if they're on disk...")
    from pathlib import Path
    for osd_id, dataset_id in dataset_map.items():
        path = Path("./datasets") / f"{dataset_id}.parquet"
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {path}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

client.close()
