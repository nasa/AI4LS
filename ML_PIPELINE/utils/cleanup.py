#!/usr/bin/env python3
"""
Cleanup utility to delete experiments, datasets, models, feature importances, and KEGG analyses
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
    
    def __init__(self):
        # Base paths
        self.ml_service_path = Path("/app/models")  # Inside Docker
        self.data_service_path = Path("/app/datasets")  # Inside Docker
        self.experiment_service_path = Path("/app/experiments")  # Inside Docker
        self.bioinformatics_results_path = Path("/app/results")  # Inside Docker
    
    def delete_experiment(self, experiment_id: str, dry_run: bool = False) -> bool:
        """Delete an experiment and all associated data"""
        try:
            logger.info(f"Deleting experiment: {experiment_id}")
            
            # Load experiment to find associated resources
            exp_file = self.experiment_service_path / f"{experiment_id}.json"
            
            if not exp_file.exists():
                logger.error(f"Experiment not found: {experiment_id}")
                return False
            
            with open(exp_file, 'r') as f:
                exp_data = json.load(f)
            
            # Delete associated datasets
            if 'datasets' in exp_data:
                for dataset_type, dataset_id in exp_data['datasets'].items():
                    if dataset_id:
                        self.delete_dataset(dataset_id, dry_run=dry_run)
            
            # Delete associated models
            if 'models' in exp_data:
                for model_info in exp_data['models']:
                    if 'model_id' in model_info:
                        self.delete_model(model_info['model_id'], dry_run=dry_run)
            
            # Delete experiment file
            if not dry_run:
                exp_file.unlink()
                logger.info(f"✓ Deleted experiment file: {exp_file}")
            else:
                logger.info(f"[DRY RUN] Would delete experiment file: {exp_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}")
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
            exp_files = list(self.experiment_service_path.glob("exp_*.json"))
            experiments = []
            
            for exp_file in exp_files:
                try:
                    with open(exp_file, 'r') as f:
                        exp_data = json.load(f)
                    experiments.append({
                        'experiment_id': exp_data.get('experiment_id'),
                        'status': exp_data.get('status'),
                        'created_at': exp_data.get('created_at'),
                        'num_models': len(exp_data.get('models', [])),
                        'file': str(exp_file)
                    })
                except Exception as e:
                    logger.warning(f"Could not parse {exp_file}: {e}")
            
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
                for dataset_id, files in datasets.items()
            ]
            
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(
        description='Clean up ML pipeline artifacts (experiments, datasets, models, analyses)'
    )
    
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
    
    cleanup = PipelineCleanup()
    
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
                print("\n" + "=" * 80)
                print("EXPERIMENTS")
                print("=" * 80)
                print(f"{'ID':<30} {'Status':<12} {'Models':<8} {'Created'}")
                print("-" * 80)
                for exp in experiments:
                    print(f"{exp['experiment_id']:<30} {exp['status']:<12} {exp['num_models']:<8} {exp['created_at']}")
            else:
                print("No experiments found")
        
        elif args.type == 'models':
            models = cleanup.list_models()
            if models:
                print("\n" + "=" * 80)
                print("MODELS")
                print("=" * 80)
                print(f"{'ID':<30} {'Algorithm':<20} {'Task':<15} {'Created'}")
                print("-" * 80)
                for model in models:
                    print(f"{model['model_id']:<30} {model['algorithm']:<20} {model['task_type']:<15} {model['created_at']}")
            else:
                print("No models found")
        
        elif args.type == 'datasets':
            datasets = cleanup.list_datasets()
            if datasets:
                print("\n" + "=" * 80)
                print("DATASETS")
                print("=" * 80)
                print(f"{'Dataset ID':<40} {'Files':<8} {'Variants'}")
                print("-" * 80)
                for dataset in datasets:
                    print(f"{dataset['dataset_id']:<40} {dataset['num_files']:<8} {', '.join(dataset['files'])}")
            else:
                print("No datasets found")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
