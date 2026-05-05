# view_experiments.py

import sys
sys.path.insert(0, 'experiment_service')
from src.experiment_client import ExperimentClient
from datetime import datetime

def format_timestamp(ts):
    """Format Unix timestamp to readable date"""
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def main():
    client = ExperimentClient('localhost:50055')
    
    if len(sys.argv) > 1:
        # Get specific experiment
        experiment_id = sys.argv[1]
        exp = client.get_experiment(experiment_id)
        
        if exp:
            print("\n" + "=" * 80)
            print(f"EXPERIMENT: {exp['name']}")
            print("=" * 80)
            print(f"ID: {exp['experiment_id']}")
            print(f"Description: {exp['description']}")
            print(f"Status: {exp['status']}")
            print(f"Created: {format_timestamp(exp['created_at'])}")
            print(f"Updated: {format_timestamp(exp['updated_at'])}")
            print()
            print("Pipeline Components:")
            print(f"  Dataset ID: {exp['dataset_id']}")
            print(f"  Model ID: {exp['model_id']}")
            print(f"  Feature Importance ID: {exp['feature_importance_id']}")
            print(f"  KEGG Analysis ID: {exp['kegg_analysis_id']}")
            print()
            print("Metadata:")
            for key, value in exp['metadata'].items():
                print(f"  {key}: {value}")
        else:
            print(f"Experiment {experiment_id} not found")
    else:
        # List all experiments
        experiments = client.list_experiments(limit=50)
        
        if not experiments:
            print("No experiments found")
            return
        
        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS")
        print("=" * 80)
        print(f"{'ID':<20} {'Name':<40} {'Status':<12} {'Created'}")
        print("-" * 80)
        
        for exp in experiments:
            created = format_timestamp(exp['created_at'])
            print(f"{exp['experiment_id']:<20} {exp['name']:<40} {exp['status']:<12} {created}")
        
        print()
        print(f"Total: {len(experiments)} experiments")
        print()
        print("Usage: python view_experiments.py <experiment_id>  # View details")

if __name__ == "__main__":
    main()
