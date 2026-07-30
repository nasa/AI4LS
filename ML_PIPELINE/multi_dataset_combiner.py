#!/usr/bin/env python3
"""
Multi-Dataset Combiner
Combine multiple OSD datasets by tissue type or explicit OSD IDs
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tissue registry: maps tissue names to OSD IDs (without "OSD-" prefix)
TISSUE_REGISTRY = {
    "muscle": ["48", "51", "71", "97"],
    "bone": ["179", "180", "181"],
    "liver": ["47", "48", "137", "168", "463", "379", "245", "173"],
    "kidney": ["123", "124"],
    "heart": ["142", "143"],
    "brain": ["165", "166"],
    "blood": ["200", "201"],
    "skin": ["220", "221"],
    "intestine": ["250", "251"],
    "lung": ["275", "276"],
}

class MultiDatasetCombiner:
    """Combine multiple OSD datasets"""
    
    def __init__(self, data_client):
        self.data_client = data_client
        self.combined_datasets = {}
    
    def get_osd_ids_for_tissue(self, tissue_name: str) -> List[str]:
        """Get OSD IDs for a specific tissue"""
        tissue_lower = tissue_name.lower()
        
        if tissue_lower not in TISSUE_REGISTRY:
            available = ", ".join(TISSUE_REGISTRY.keys())
            raise ValueError(f"Tissue '{tissue_name}' not found. Available tissues: {available}")
        
        osd_ids = TISSUE_REGISTRY[tissue_lower]
        logger.info(f"Found {len(osd_ids)} datasets for tissue '{tissue_name}': {osd_ids}")
        
        return osd_ids
    
    def download_multiple_datasets(
        self,
        osd_ids: List[str],
        patterns: List[str],
        factor_name: str,
        factor_values: List[str],
        exclude_columns: List[str] = None,
        min_features: int = 1000,
        cv_step: float = 0.25
    ) -> Dict[str, pd.DataFrame]:
        """Download multiple OSD datasets"""
        
        if exclude_columns is None:
            exclude_columns = []
        
        datasets = {}
        
        for osd_id in osd_ids:
            # Ensure osd_id is just the number (no "OSD-" prefix)
            osd_id = str(osd_id).replace("OSD-", "").strip()
            
            logger.info(f"Downloading OSD-{osd_id}...")
            
            try:
                # Call download_dataset - returns a dictionary
                response = self.data_client.download_dataset(
                    osd_id=osd_id,
                    dataset_id=osd_id,
                    patterns=patterns,
                    factor_name=factor_name,
                    factor_values=factor_values,
                    exclude_columns=exclude_columns,
                    min_features=min_features,
                    cv_step=cv_step
                )
                
                # Response structure: {'is_valid': bool, 'dataset_info': {'dataset_id': ...}}
                if not response.get('is_valid'):
                    logger.warning(f"Failed to download OSD-{osd_id}: {response.get('errors', [])}")
                    continue
                
                # Extract dataset_id from nested structure
                dataset_id = response.get('dataset_info', {}).get('dataset_id')
                if not dataset_id:
                    logger.warning(f"No dataset_id in response for OSD-{osd_id}")
                    continue
                
                # Load the downloaded dataset from disk (stored as parquet)
                dataset_path = Path("./datasets") / f"{dataset_id}.parquet"
                
                if not dataset_path.exists():
                    logger.warning(f"Dataset file not found at {dataset_path}")
                    continue
                
                # Read the parquet file
                df = pd.read_parquet(dataset_path)
                datasets[f"OSD-{osd_id}"] = df
                
                logger.info(f"✓ Downloaded OSD-{osd_id}: {df.shape[0]} samples × {df.shape[1]} genes")
                
            except Exception as e:
                logger.error(f"Error downloading OSD-{osd_id}: {e}", exc_info=True)
                continue
        
        if not datasets:
            raise ValueError("No datasets were successfully downloaded")
        
        logger.info(f"✓ Downloaded {len(datasets)} datasets")
        
        return datasets
    
    def find_common_genes(self, datasets: Dict[str, pd.DataFrame]) -> pd.Index:
        """Find genes common to all datasets"""
        
        logger.info("Finding common genes across datasets...")
        
        # Get all gene sets
        gene_sets = []
        for osd_id, df in datasets.items():
            # Exclude the condition column and metadata columns
            condition_cols = [col for col in df.columns if 'Factor' in col or 'Condition' in col or col == 'source_dataset']
            genes = [col for col in df.columns if col not in condition_cols]
            gene_sets.append(set(genes))
            logger.info(f"  {osd_id}: {len(genes)} genes")
        
        if not gene_sets:
            raise ValueError("No genes found in datasets")
        
        # Find intersection (genes common to all)
        common_genes = gene_sets[0]
        for gene_set in gene_sets[1:]:
            common_genes = common_genes.intersection(gene_set)
        
        logger.info(f"✓ Common genes: {len(common_genes)}")
        
        return pd.Index(sorted(list(common_genes)))
    
    def combine_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        common_genes: pd.Index = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Combine multiple datasets into one
        
        Returns:
            combined_df: DataFrame with all samples
            dataset_map: Dict mapping sample index to OSD ID
        """
        
        logger.info("Combining datasets...")
        
        # Find common genes if not provided
        if common_genes is None:
            common_genes = self.find_common_genes(datasets)
        
        combined_data = []
        dataset_map = {}
        condition_column = None
        condition_values = {}
        
        for osd_id, df in datasets.items():
            logger.info(f"Processing {osd_id}...")
            
            # Find condition column (Factor or Condition)
            condition_cols = [col for col in df.columns if 'Factor' in col or 'Condition' in col]
            
            if condition_cols:
                cond_col = condition_cols[0]
                if condition_column is None:
                    condition_column = cond_col
                
                cond_values = df[cond_col].copy()
            else:
                cond_values = None
            
            # Select only common genes
            df_subset = df[list(common_genes)].copy()
            
            # Track which dataset each sample came from
            for sample_idx, sample_name in enumerate(df_subset.index):
                dataset_map[sample_name] = osd_id
                if cond_values is not None:
                    condition_values[sample_name] = cond_values.iloc[sample_idx]
            
            combined_data.append(df_subset)
            logger.info(f"  Added {len(df_subset)} samples")
        
        # Concatenate all datasets
        combined_df = pd.concat(combined_data, axis=0)
        
        # Add condition column back
        if condition_column is not None:
            combined_df[condition_column] = pd.Series(condition_values)
        
        # Add source dataset as metadata column
        combined_df['source_dataset'] = pd.Series(dataset_map)
        
        logger.info(f"\n✓ Combined dataset:")
        logger.info(f"  Samples: {len(combined_df)}")
        logger.info(f"  Genes: {len(common_genes)}")
        logger.info(f"  Condition column: {condition_column}")
        logger.info(f"  Source datasets: {', '.join(sorted(set(dataset_map.values())))}")
        
        return combined_df, dataset_map
    
    def save_combined_dataset(self, combined_df: pd.DataFrame, output_path: str = None) -> str:
        """Save combined dataset to parquet"""
        
        if output_path is None:
            output_path = "./combined_dataset.parquet"
        
        combined_df.to_parquet(output_path)
        logger.info(f"✓ Saved combined dataset to {output_path}")
        
        return output_path
    
    def print_dataset_summary(self, combined_df: pd.DataFrame, dataset_map: Dict[str, str]):
        """Print summary of combined dataset"""
        
        print("\n" + "=" * 80)
        print("COMBINED DATASET SUMMARY")
        print("=" * 80)
        
        # Count samples per dataset
        print("\nSamples per dataset:")
        for osd_id in sorted(set(dataset_map.values())):
            count = sum(1 for d in dataset_map.values() if d == osd_id)
            print(f"  {osd_id}: {count} samples")
        
        # Condition distribution
        condition_cols = [col for col in combined_df.columns if 'Factor' in col or 'Condition' in col]
        if condition_cols:
            cond_col = condition_cols[0]
            print(f"\nCondition distribution ({cond_col}):")
            cond_counts = combined_df[cond_col].value_counts()
            for cond, count in cond_counts.items():
                print(f"  {cond}: {count} samples")
        
        gene_cols = [col for col in combined_df.columns if col not in ['source_dataset'] + condition_cols]
        print(f"\nTotal samples: {len(combined_df)}")
        print(f"Total genes: {len(gene_cols)}")
        print("=" * 80 + "\n")
