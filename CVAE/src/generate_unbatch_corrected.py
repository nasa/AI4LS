import utils
import anndata
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import scanpy.external as sce
import scanorama
import os
import anndata
from typing import List, Optional

path = ""

processed_612 = sc.read_h5ad(path+'data/processed_612_data.h5ad')
processed_613 = sc.read_h5ad(path+'data/processed_613_data.h5ad')
processed_352 = sc.read_h5ad(path+'data/processed_352_data.h5ad')

processed_data = [processed_352, processed_612, processed_613]

def gene_overlap_check(adata_list: list[sc.AnnData]):
    """Quick check of gene overlap between datasets"""
    gene_sets = [set(adata.var.index) for adata in adata_list]
    common_genes = set.intersection(*gene_sets)
    all_genes = set.union(*gene_sets)

    print(f"Gene overlap summary:")
    print(f"  Genes common to ALL datasets: {len(common_genes)}")
    print(f"  Total unique genes: {len(all_genes)}")
    print(f"  Overlap percentage: {len(common_genes) / len(all_genes) * 100:.1f}%")
    return common_genes

def merge_anndata_on_shared_vars(
        adata_list: List[sc.AnnData],
        output_path: Optional[str] = None
) -> sc.AnnData:
    """
    Merge multiple AnnData objects on shared var indices (genes).
    Assumes each dataset already has a 'dataset' column in .obs.

    Parameters:
    -----------
    adata_list : List[sc.AnnData]
        List of AnnData objects to merge
    output_path : Optional[str]
        Path to save the merged object as h5ad file

    Returns:
    --------
    sc.AnnData
        Merged AnnData object with only shared genes
    """

    # Validate inputs
    if len(adata_list) < 2:
        raise ValueError("Need at least 2 AnnData objects to merge")

    print(f"Merging {len(adata_list)} AnnData objects on shared variables...")

    # Print basic info about each dataset
    for i, adata in enumerate(adata_list):
        dataset_info = adata.obs['dataset'].value_counts()
        print(f"Dataset {i}: {adata.n_obs} cells × {adata.n_vars} genes")
        print(f"  Dataset labels: {list(dataset_info.index)}")

    # Merge datasets using inner join (only shared genes)
    merged_adata = sc.concat(
        adata_list,
        join='inner',  # Only keep genes present in ALL datasets
        index_unique=None
    )

    print(f"\nMerged dataset: {merged_adata.n_obs} cells × {merged_adata.n_vars} genes")
    print(f"Genes kept: {merged_adata.n_vars} (shared across all datasets)")

    # Show dataset distribution in merged object
    print(f"\nDataset distribution in merged object:")
    dataset_counts = merged_adata.obs['dataset'].value_counts()
    print(dataset_counts)

    # Save if output path provided
    if output_path:
        merged_adata.write(output_path)
        print(f"\nSaved merged dataset to: {output_path}")

    return merged_adata

merge_anndata_on_shared_vars(processed_data, path+'data/unbatch_corrected_data.h5ad')

