#!/usr/bin/env python3
"""
Cleanup utility to delete experiments, datasets, models, feature importances, and KEGG analyses
Works on host machine with proper path detection
"""

import sys
import os
from pathlib import Path
import shutil
import json
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineCleanup:
    """Manage deletion of pipeline artifacts"""
    
    def __init__(self, ml_models_path="./models", datasets_path="./datasets", 
                 experiments_path="./experiments", results_path="./bioinformatics_service/results"):
        # Set paths
        self.ml_service_path = Path(ml_models_path)
        self.data_service_path = Path(datasets_path)
        self.experiment_service_path = Path(experiments_path)
        self.bioinformatics_results_path = Path(results_path)
        
        logger.info(f"Models path: {self.ml_service_path}")
        logger.info(f"Datasets path: {self.data_service_path}")
        logger.info(f"Experiments path: {self.experiment_service_path}")
        logger.info(f"Results path: {self.bioinformatics_results_path}")
    
    def delete_experiment(self, experiment_id: str, dry_run: bool = False) -> bool:
        """Delete an experiment from experiments.json and all associated data"""
        try:
            logger.info(f"Deleting experiment: {experiment_id}")
            
            # Load experiments from experiments.json
            exp_file = self.experiment_service_path / "experiments.json"
            
            if not exp_file.exists():
                logger.error(f"Experiments file not found: {exp_file}")
                return False
            
            with open(exp_file, 'r') as f:
                experiments = json.load(f)
            
            # Find the experiment
            if experiment_id not in experiments:
                logger.error(f"Experiment not found: {experiment_id}")
                logger.info(f"Available experiments: {list(experiments.keys())}")
                return False
            
            exp_data = experiments[experiment_id]
            
            logger.info(f"Found experiment: {exp_data.get('description', 'N/A')}")
            
            # Delete associated datasets
            if 'datasets' in exp_data:
                for dataset_type, dataset_id in exp_data['datasets'].items():
                    if dataset_id:
                        logger.info(f"Deleting associated dataset: {dataset_id}")
                        self.delete_dataset(dataset_id, dry_run=dry_run)
            
            # Delete associated models
            if 'models' in exp_data:
                for model_info in exp_data['models']:
                    if 'model_id' in model_info:
                        logger.info(f"Deleting associated model: {model_info['model_id']}")
                        self.delete_model(model_info['model_id'], dry_run=dry_run)
            
            # Remove experiment from experiments.json
            if not dry_run:
                del experiments[experiment_id]
                with open(exp_file, 'w') as f:
                    json.dump(experiments, f, indent=2)
                logger.info(f"✓ Deleted experiment {experiment_id} from {exp_file}")
            else:
                logger.info(f"[DRY RUN] Would delete experiment {experiment_id} from {exp_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}", exc_info=True)
            return False
    
    def delete_dataset(self, dataset_id: str, dry_run: bool = False) -> bool:
        """Delete a dataset"""
        try:
            logger.info(f"Deleting dataset: {dataset_id}")
            
            # Find dataset files (raw, filtered, transformed variants)
            dataset_dir = self.data_service_path
            dataset_files = list(dataset_dir.glob(f"{dataset_id}*.parquet"))
            
            if not dataset_files:
                logger.warning(f"No dataset files found for: {dataset_id}")
                return False
            
            for dataset_file in dataset_files:
                if not dry_run:
                    dataset_file.unlink()
                    logger.info(f"✓ Deleted dataset: {dataset_file}")
                else:
                    logger.info(f"[DRY RUN] Would delete dataset: {dataset_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            return False
    
    def delete_model(self, model_id: str, dry_run: bool = False) -> bool:
        """Delete a trained model"""
        try:
            logger.info(f"Deleting model: {model_id}")
            
            model_dir = self.ml_service_path
            
            # Delete model file
            model_file = model_dir / f"{model_id}.joblib"
            if model_file.exists():
                if not dry_run:
                    model_file.unlink()
                    logger.info(f"✓ Deleted model file: {model_file}")
                else:
                    logger.info(f"[DRY RUN] Would delete model file: {model_file}")
            
            # Delete model metadata from metadata.json
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if model_id in metadata:
                    if not dry_run:
                        del metadata[model_id]
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        logger.info(f"✓ Removed {model_id} from metadata.json")
                    else:
                        logger.info(f"[DRY RUN] Would remove {model_id} from metadata.json")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def delete_feature_importance(self, model_id: str, dry_run: bool = False) -> bool:
        """Delete feature importance results for a model"""
        try:
            logger.info(f"Deleting feature importance results for: {model_id}")
            
            # Feature importance results are stored in ML service
            results_dir = self.ml_service_path.parent / "feature_importance"
            
            if not results_dir.exists():
                logger.warning(f"Feature importance directory not found: {results_dir}")
                return False
            
            # Look for files related to this model
            model_result_files = list(results_dir.glob(f"*{model_id}*"))
            
            if not model_result_files:
                logger.warning(f"No feature importance files found for: {model_id}")
                return False
            
            for result_file in model_result_files:
                if not dry_run:
                    if result_file.is_file():
                        result_file.unlink()
                    elif result_file.is_dir():
                        shutil.rmtree(result_file)
                    logger.info(f"✓ Deleted feature importance results: {result_file}")
                else:
                    logger.info(f"[DRY RUN] Would delete feature importance results: {result_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete feature importance for {model_id}: {e}")
            return False
    
    def delete_kegg_analysis(self, analysis_id: str = None, organism: str = "mmu", dry_run: bool = False) -> bool:
        """Delete KEGG enrichment analysis results"""
        try:
            logger.info(f"Deleting KEGG analysis for: {analysis_id or organism}")
            
            results_dir = self.bioinformatics_results_path
            
            # Find KEGG result directories
            if analysis_id:
                kegg_dirs = list(results_dir.glob(f"kegg_*_{analysis_id}"))
            else:
                kegg_dirs = list(results_dir.glob(f"kegg_{organism}_*"))
            
            if not kegg_dirs:
                logger.warning(f"No KEGG analysis directories found")
                return False
            
            for kegg_dir in kegg_dirs:
                if not dry_run:
                    shutil.rmtree(kegg_dir)
                    logger.info(f"✓ Deleted KEGG analysis directory: {kegg_dir}")
                else:
                    logger.info(f"[DRY RUN] Would delete KEGG analysis directory: {kegg_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete KEGG analysis: {e}")
            return False
    
    def list_experiments(self) -> list:
        """List all experiments"""
        try:
            exp_file = self.experiment_service_path / "experiments.json"
            
            if not exp_file.exists():
                logger.warning("No experiments file found")
                return []
            
            with open(exp_file, 'r') as f:
                exp_data = json.load(f)
            
            experiments = []
            for exp_id, exp_info in exp_data.items():
                experiments.append({
                    'experiment_id': exp_id,
                    'status': exp_info.get('status'),
                    'created_at': exp_info.get('created_at'),
                    'description': exp_info.get('description', ''),
                    'num_models': len(exp_info.get('models', []))
                })
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def list_models(self) -> list:
        """List all models"""
        try:
            metadata_file = self.ml_service_path / "metadata.json"
            
            if not metadata_file.exists():
                logger.warning("No models metadata file found")
                return []
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            models = []
            for model_id, model_info in metadata.items():
                models.append({
                    'model_id': model_id,
                    'algorithm': model_info.get('algorithm'),
                    'task_type': model_info.get('task_type'),
                    'created_at': model_info.get('created_at')
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def list_datasets(self) -> list:
        """List all datasets"""
        try:
            dataset_dir = self.data_service_path
            dataset_files = list(dataset_dir.glob("*.parquet"))
            
            # Group by dataset ID
            datasets = {}
            for dataset_file in dataset_files:
                # Extract base UUID (before _filtered, _transformed, etc)
                name = dataset_file.stem
                dataset_id = name.split('_')[0]
                
                if dataset_id not in datasets:
                    datasets[dataset_id] = []
                datasets[dataset_id].append(dataset_file.name)
            
            return [
                {
                    'dataset_id': dataset_id,
                    'files': files,
                    'num_files': len(files)
                }
                for dataset_id, files in sorted(datasets.items())
            ]
            
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(
        description='Clean up ML pipeline artifacts (experiments, datasets, models, analyses)'
    )
    
    # Path arguments (defaults to your local structure)
    parser.add_argument('--ml-models-path', default='./models', help='Path to ML service models directory (default: ./models)')
    parser.add_argument('--datasets-path', default='./datasets', help='Path to data service datasets directory (default: ./datasets)')
    parser.add_argument('--experiments-path', default='./experiments', help='Path to experiment service directory (default: ./experiments)')
    parser.add_argument('--results-path', default='./bioinformatics_service/results', help='Path to bioinformatics results directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Delete experiment
    exp_parser = subparsers.add_parser('delete-experiment', help='Delete an experiment')
    exp_parser.add_argument('experiment_id', help='Experiment ID to delete')
    exp_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    
    # Delete dataset
    data_parser = subparsers.add_parser('delete-dataset', help='Delete a dataset')
    data_parser.add_argument('dataset_id', help='Dataset ID to delete')
    data_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    
    # Delete model
    model_parser = subparsers.add_parser('delete-model', help='Delete a model')
    model_parser.add_argument('model_id', help='Model ID to delete')
    model_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    
    # Delete feature importance
    fi_parser = subparsers.add_parser('delete-importance', help='Delete feature importance results')
    fi_parser.add_argument('model_id', help='Model ID to delete importance for')
    fi_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    
    # Delete KEGG analysis
    kegg_parser = subparsers.add_parser('delete-kegg', help='Delete KEGG enrichment results')
    kegg_parser.add_argument('--analysis-id', help='Analysis ID to delete')
    kegg_parser.add_argument('--organism', default='mmu', help='Organism to delete (default: mmu)')
    kegg_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    
    # List commands
    list_parser = subparsers.add_parser('list', help='List artifacts')
    list_parser.add_argument('type', choices=['experiments', 'models', 'datasets'],
                            help='Type of artifact to list')
    
    args = parser.parse_args()
    
    cleanup = PipelineCleanup(
        ml_models_path=args.ml_models_path,
        datasets_path=args.datasets_path,
        experiments_path=args.experiments_path,
        results_path=args.results_path
    )
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'delete-experiment':
        if cleanup.delete_experiment(args.experiment_id, dry_run=args.dry_run):
            logger.info("✓ Experiment deleted successfully")
        else:
            logger.error("✗ Failed to delete experiment")
            sys.exit(1)
    
    elif args.command == 'delete-dataset':
        if cleanup.delete_dataset(args.dataset_id, dry_run=args.dry_run):
            logger.info("✓ Dataset deleted successfully")
        else:
            logger.error("✗ Failed to delete dataset")
            sys.exit(1)
    
    elif args.command == 'delete-model':
        if cleanup.delete_model(args.model_id, dry_run=args.dry_run):
            logger.info("✓ Model deleted successfully")
        else:
            logger.error("✗ Failed to delete model")
            sys.exit(1)
    
    elif args.command == 'delete-importance':
        if cleanup.delete_feature_importance(args.model_id, dry_run=args.dry_run):
            logger.info("✓ Feature importance deleted successfully")
        else:
            logger.error("✗ Failed to delete feature importance")
            sys.exit(1)
    
    elif args.command == 'delete-kegg':
        if cleanup.delete_kegg_analysis(args.analysis_id, args.organism, dry_run=args.dry_run):
            logger.info("✓ KEGG analysis deleted successfully")
        else:
            logger.error("✗ Failed to delete KEGG analysis")
            sys.exit(1)
    
    elif args.command == 'list':
        if args.type == 'experiments':
            experiments = cleanup.list_experiments()
            if experiments:
                print("\n" + "=" * 100)
                print("EXPERIMENTS")
                print("=" * 100)
                print(f"{'ID':<36} {'Status':<12} {'Description':<35} {'Models':<8}")
                print("-" * 100)
                for exp in experiments:
                    desc = exp['description'][:32] if exp['description'] else ''
                    print(f"{exp['experiment_id']:<36} {exp['status']:<12} {desc:<35} {exp['num_models']:<8}")
            else:
                print("No experiments found")
        
        elif args.type == 'models':
            models = cleanup.list_models()
            if models:
                print("\n" + "=" * 80)
                print("MODELS")
                print("=" * 80)
                print(f"{'ID':<36} {'Algorithm':<20} {'Task':<15}")
                print("-" * 80)
                for model in models:
                    print(f"{model['model_id']:<36} {model['algorithm']:<20} {model['task_type']:<15}")
            else:
                print("No models found")
        
        elif args.type == 'datasets':
            datasets = cleanup.list_datasets()
            if datasets:
                print("\n" + "=" * 80)
                print("DATASETS")
                print("=" * 80)
                print(f"{'Dataset ID':<40} {'Files':<8}")
                print("-" * 80)
                for dataset in datasets:
                    print(f"{dataset['dataset_id']:<40} {dataset['num_files']:<8}")
            else:
                print("No datasets found")


if __name__ == '__main__':
    main()
