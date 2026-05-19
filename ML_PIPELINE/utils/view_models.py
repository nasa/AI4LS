# view_models.py

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'ml_service')
from src.model_store import ModelStore

def format_timestamp(ts):
    """Format Unix timestamp to readable date"""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return ts

def main():
    model_store = ModelStore(base_path="./models")
    
    # List all models
    models = model_store.list_models()
    
    if not models:
        print("No models found")
        return
    
    print("\n" + "=" * 120)
    print("TRAINED MODELS")
    print("=" * 120)
    print(f"{'Model ID':<25} {'Algorithm':<20} {'Task':<15} {'Accuracy':<10} {'Features':<10} {'Trained'}")
    print("-" * 120)
    
    for model_id in sorted(models, reverse=True):
        info = model_store.get_model_info(model_id)
        
        if info:
            algorithm = info.get('algorithm', 'unknown')
            task_type = info.get('task_type', 'unknown')
            accuracy = info.get('metrics', {}).get('accuracy', 0.0)
            num_features = len(info.get('feature_columns', []))
            trained_at = format_timestamp(info.get('trained_at', 0))
            
            print(f"{model_id:<25} {algorithm:<20} {task_type:<15} {accuracy:<10.4f} {num_features:<10} {trained_at}")
    
    print("-" * 120)
    print(f"Total: {len(models)} models")
    
    # Detailed view if model_id provided
    if len(sys.argv) > 1:
        model_id = sys.argv[1]
        info = model_store.get_model_info(model_id)
        
        if info:
            print("\n" + "=" * 120)
            print(f"MODEL: {model_id}")
            print("=" * 120)
            
            print(f"\nAlgorithm: {info.get('algorithm', 'unknown')}")
            print(f"Task Type: {info.get('task_type', 'unknown')}")
            print(f"Dataset ID: {info.get('dataset_id', 'unknown')}")
            print(f"Target Column: {info.get('target_column', 'unknown')}")
            print(f"Trained: {format_timestamp(info.get('trained_at', 0))}")
            
            print(f"\nFeatures ({len(info.get('feature_columns', []))}):")
            for i, feature in enumerate(info.get('feature_columns', [])[:20], 1):
                print(f"  {i:3d}. {feature}")
            if len(info.get('feature_columns', [])) > 20:
                print(f"  ... and {len(info.get('feature_columns', [])) - 20} more")
            
            print(f"\nHyperparameters:")
            for key, value in info.get('hyperparameters', {}).items():
                print(f"  {key}: {value}")
            
            print(f"\nMetrics:")
            for key, value in info.get('metrics', {}).items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            print(f"\nClass Distribution:")
            for label, count in info.get('class_distribution', {}).items():
                print(f"  {label}: {count}")
            
            print(f"\nModel File:")
            model_file = Path("models") / f"{model_id}.joblib"
            if model_file.exists():
                size_mb = model_file.stat().st_size / 1024 / 1024
                print(f"  {model_file} ({size_mb:.2f} MB)")
            else:
                print(f"  Model file not found")
        else:
            print(f"\nModel {model_id} not found")
    else:
        print("\nTo view detailed info for a model, run:")
        print("  python view_models.py <model_id>")

if __name__ == "__main__":
    main()
