# view_datasets.py

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def format_size(bytes_size):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def main():
    datasets_dir = Path("datasets")
    cache_file = datasets_dir / "download_cache.json"
    
    # Load cache
    cache = {}
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    
    # Get all parquet files
    parquet_files = list(datasets_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("No datasets found")
        return
    
    print("\n" + "=" * 100)
    print("CACHED DATASETS")
    print("=" * 100)
    print(f"{'Dataset ID':<40} {'Shape':<15} {'Size':<10} {'Columns':<10} {'Created'}")
    print("-" * 100)
    
    datasets_info = []
    
    for parquet_file in sorted(parquet_files):
        dataset_id = parquet_file.stem
        
        # Load dataset to get info
        try:
            df = pd.read_parquet(parquet_file)
            shape = f"{df.shape[0]} × {df.shape[1]}"
            size = format_size(parquet_file.stat().st_size)
            num_cols = len(df.columns)
            created = datetime.fromtimestamp(parquet_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            
            datasets_info.append({
                'dataset_id': dataset_id,
                'shape': shape,
                'size': size,
                'num_cols': num_cols,
                'created': created,
                'file': parquet_file
            })
            
            print(f"{dataset_id:<40} {shape:<15} {size:<10} {num_cols:<10} {created}")
            
        except Exception as e:
            print(f"{dataset_id:<40} ERROR: {e}")
    
    print("-" * 100)
    print(f"Total: {len(datasets_info)} datasets")
    
    # Find cache entries
    print("\n" + "=" * 100)
    print("CACHE ENTRIES (Download Parameters → Dataset ID)")
    print("=" * 100)
    
    if cache:
        for cache_key, dataset_id in cache.items():
            print(f"Cache key: {cache_key[:60]}...")
            print(f"  → Dataset ID: {dataset_id}")
            print()
    else:
        print("No cache entries found")
    
    # Show detailed info if dataset_id provided
    if len(datasets_info) > 0:
        print("\n" + "=" * 100)
        print("DETAILED VIEW")
        print("=" * 100)
        print("\nTo view detailed info for a dataset, run:")
        print(f"  python view_datasets.py <dataset_id>")
        
        # If argument provided, show details
        import sys
        if len(sys.argv) > 1:
            dataset_id = sys.argv[1]
            dataset_file = datasets_dir / f"{dataset_id}.parquet"
            
            if dataset_file.exists():
                print(f"\n{'=' * 100}")
                print(f"DATASET: {dataset_id}")
                print('=' * 100)
                
                df = pd.read_parquet(dataset_file)
                
                print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
                print(f"Size: {format_size(dataset_file.stat().st_size)}")
                print(f"File: {dataset_file}")
                
                print(f"\nColumns ({len(df.columns)}):")
                for i, col in enumerate(df.columns, 1):
                    dtype = df[col].dtype
                    nunique = df[col].nunique()
                    null_count = df[col].isnull().sum()
                    print(f"  {i:3d}. {col:<50} {str(dtype):<15} {nunique:>8} unique  {null_count:>6} nulls")
                
                # Check for condition/factor columns
                condition_cols = [col for col in df.columns if any(x in col.lower() 
                                  for x in ['factor', 'condition', 'treatment', 'group'])]
                
                if condition_cols:
                    print(f"\nPotential Condition Columns:")
                    for col in condition_cols:
                        unique_vals = df[col].unique()
                        print(f"  {col}:")
                        for val in unique_vals[:10]:
                            count = (df[col] == val).sum()
                            print(f"    - {val}: {count} samples")
                        if len(unique_vals) > 10:
                            print(f"    ... and {len(unique_vals) - 10} more")
                
                print(f"\nFirst 5 rows:")
                print(df.head())
                
                print(f"\nData types distribution:")
                print(df.dtypes.value_counts())
                
                print(f"\nMemory usage:")
                print(df.memory_usage(deep=True).sum() / 1024 / 1024, "MB")
            else:
                print(f"\nDataset {dataset_id} not found")

if __name__ == "__main__":
    main()
