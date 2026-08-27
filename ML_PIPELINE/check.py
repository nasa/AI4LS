#!/usr/bin/env python3
"""
Check what datasets are actually on disk
"""

from pathlib import Path
import os

datasets_dir = Path("./datasets")

print("="*60)
print("Checking Datasets Directory")
print("="*60)

if not datasets_dir.exists():
    print(f"✗ Directory not found: {datasets_dir}")
    print("  Creating it...")
    datasets_dir.mkdir(parents=True)
    print("✓ Created")
else:
    print(f"✓ Found: {datasets_dir}")
    
    # List all parquet files
    parquet_files = list(datasets_dir.glob("*.parquet"))
    
    print(f"\nParquet files ({len(parquet_files)}):")
    for f in parquet_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    if not parquet_files:
        print("  (none found)")
    
    # List all files
    all_files = list(datasets_dir.glob("*"))
    print(f"\nAll files ({len(all_files)}):")
    for f in all_files:
        if f.is_file():
            print(f"  - {f.name}")

print("\n" + "="*60)
print("What This Means")
print("="*60)

if not parquet_files:
    print("✗ No datasets found on disk!")
    print("\nPossible causes:")
    print("1. DownloadMultipleDatasets didn't save datasets to disk")
    print("2. Datasets were saved somewhere else")
    print("3. Only 1 of 2 OSD IDs was successfully downloaded")
    print("\nCheck the data service logs for download errors")
else:
    print(f"✓ Found {len(parquet_files)} dataset(s)")
    print("\nIf you're getting 'No datasets found' error:")
    print("- The dataset_ids returned may not match these file names")
    print("- Or find_common_genes is looking in the wrong directory")
