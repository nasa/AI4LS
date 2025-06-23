import os
from typing import List, Optional, Dict, Any, Union
import scanpy as sc
import anndata
import h5py
import numpy as np


def merge_h5_files(
    file_paths: List[str],
    output_path: Optional[str] = None,
    batch_key: str = 'batch',
    batch_categories: Optional[List[str]] = None,
    add_file_metadata: bool = True,
    metadata_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    export_as_h5: bool = True
) -> anndata.AnnData:
    """
    Merge multiple 10x h5 files into a single AnnData object.
    
    Parameters:
    -----------
    file_paths : List[str]
        List of paths to h5 files.
    output_path : Optional[str], default=None
        Path to save the merged h5ad file. If None, the file will not be saved.
    batch_key : str, default='batch'
        Key in obs where the batch ID will be stored.
    batch_categories : Optional[List[str]], default=None
        List of batch categories. If None, will use the basenames of file_paths.
    add_file_metadata : bool, default=True
        Whether to add file name as metadata in the AnnData object.
    metadata_dict : Optional[Dict[str, Dict[str, Any]]], default=None
        Dictionary of metadata for each file. Keys are batch identifiers and 
        values are dictionaries of metadata to be added to obs.
    
    Returns:
    --------
    adata_combined : anndata.AnnData
        Combined AnnData object containing all cells from the input files.
    """
    if len(file_paths) == 0:
        raise ValueError('No files provided')
    
    # Create batch categories from filenames if not provided
    if batch_categories is None:
        batch_categories = [os.path.basename(path).split('_')[0] for path in file_paths]
    
    if len(batch_categories) != len(file_paths):
        raise ValueError('Length of batch_categories must match length of file_paths')
    
    adata_list = []
    
    # Read and process each file
    for i, (file_path, batch_id) in enumerate(zip(file_paths, batch_categories)):
        print(f'Processing file {i+1}/{len(file_paths)}: {file_path}')
        
        # Read the h5 file
        try:
            adata = sc.read_10x_h5(file_path)
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
            continue
        
        # Add batch information
        adata.obs[batch_key] = batch_id
        
        # Add filename as metadata if requested
        if add_file_metadata:
            adata.obs['filename'] = os.path.basename(file_path)
        
        # Add additional metadata if provided
        if metadata_dict and batch_id in metadata_dict:
            for key, value in metadata_dict[batch_id].items():
                adata.obs[key] = value
        
        adata_list.append(adata)
    
    if len(adata_list) == 0:
        raise ValueError('Failed to read any files')
    
    # Concatenate all AnnData objects
    print('Merging datasets...')
    adata_combined = anndata.concat(
        adata_list, 
        join='outer',  # Use outer join to keep all genes
        batch_key=batch_key,
        batch_categories=batch_categories,
        index_unique='-'  # Add suffix to ensure unique indices
    )
    
    print(f'Combined dataset shape: {adata_combined.shape}')
    
    # Save the combined dataset if output path is provided
    if output_path:
        if export_as_h5:
            # Export as h5 file
            print(f'Saving combined dataset to {output_path} as h5 file')
            export_to_h5(adata_combined, output_path)
        else:
            # Export as h5ad file (AnnData native format)
            print(f'Saving combined dataset to {output_path} as h5ad file')
            adata_combined.write(output_path)
    
    return adata_combined


def export_to_h5(adata: anndata.AnnData, output_path: str):
    """
    Export AnnData object to h5 file in 10x format.
    
    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object to export
    output_path : str
        Path to save the h5 file
    """
    # Ensure output file has .h5 extension
    if not output_path.endswith('.h5'):
        output_path = output_path + '.h5'
    
    with h5py.File(output_path, 'w') as f:
        # Create the 10x-like structure
        matrix_group = f.create_group('matrix')
        
        # Extract gene information
        features_group = matrix_group.create_group('features')
        gene_ids = adata.var_names.values
        gene_names = adata.var_names.values if 'gene_symbols' not in adata.var else adata.var['gene_symbols'].values
        
        # Dataset for gene IDs
        features_group.create_dataset('id', data=np.array(gene_ids, dtype='S'))
        
        # Dataset for gene names
        features_group.create_dataset('name', data=np.array(gene_names, dtype='S'))
        
        # Feature type (usually 'Gene Expression')
        feature_types = np.array(['Gene Expression'] * len(gene_ids), dtype='S')
        features_group.create_dataset('feature_type', data=feature_types)
        
        # Cell barcodes
        barcodes = adata.obs_names.values
        matrix_group.create_dataset('barcodes', data=np.array(barcodes, dtype='S'))

        # Get counts matrix as sparse coordinate format
        from scipy import sparse
        if sparse.issparse(adata.X):
            X = adata.X
            if not isinstance(X, sparse.csr_matrix):
                X = X.tocsr()
        else:
            X = sparse.csr_matrix(adata.X)
            
        # Convert to column-major sparse matrix (10x format uses CSC)
        X = X.tocsc()  
        
        # Create data, indices, indptr datasets
        data_group = matrix_group.create_group('data')
        data_group.create_dataset('data', data=X.data)
        data_group.create_dataset('indices', data=X.indices)
        data_group.create_dataset('indptr', data=X.indptr)
        
        # Store shape information
        data_group.create_dataset('shape', data=np.array(X.shape))
        
        # Add observation metadata as attributes
        obs_group = f.create_group('obs')
        for col in adata.obs.columns:
            try:
                if adata.obs[col].dtype.kind in ['i', 'u', 'f']:  # numeric data
                    obs_group.create_dataset(col, data=adata.obs[col].values)
                else:  # categorical or string data
                    obs_group.create_dataset(col, data=np.array(adata.obs[col].astype(str).values, dtype='S'))
            except Exception as e:
                print(f"Warning: Could not store obs column '{col}': {e}")


if __name__ == "__main__":
    # Example usage
    # file_paths = [
    #     'data/612_filtered_feature_bc_matrix.h5', 
    #     'data/613_filtered_feature_bc_matrix.h5', 
    #     'data/352_filtered_feature_bc_matrix.h5'
    # ]
    # merged_adata = merge_h5_files(file_paths, 'data/merged_data.h5')
    pass
