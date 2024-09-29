import os
import re
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# gmt fxns if needed: dict_to_gmt, read_gmt
# mask fxns if needed: create pathway mask, shuffle_mask, filter_pathways

# ADATA PREPROCESSING FXNS

def preprocess_adata(adata, n_top_genes = 1000):
    """ Default sc preprocessing """
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes = 300)
    sc.pp.filter_genes(adata, min_cells = 5)

    sc.pp.normalize_total(adata, target_sum = 1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes = n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]

    return adata

# METADATA FXNS

def extract_batch(obs_name):
    # Regex pattern to match samples with the exact batch number at the end
    match = re.search(r'-(\d+)$', obs_name)
    if match:
        return match.group(1)
    else:
        return None

def parse_age_at_launch(age_range):
    return int(age_range.split('-')[0].strip())

def parse_age_at_launch_2(age_range):
    return int(age_range.split()[0])

def parse_duration(duration):
    return int(duration.split()[0])

def parse_flight(flight):
    return int(flight == 'Space Flight')

# TRAINING SET UP FXNS

def split_adata(adata, condition_column, val_split=0.2, random_state=42):
    # Split indices
    indices = list(range(adata.shape[0]))
    train_indices, val_indices = train_test_split(
        indices, test_size=val_split, random_state=random_state
    )
    # Split AnnData object
    train_adata = adata[train_indices].copy()
    val_adata = adata[val_indices].copy()
    
    return train_adata, val_adata

class CVAEDataset(Dataset):
    def __init__(self, adata, condition_cols):
        if isinstance(adata.X, np.ndarray):
            self.expression_data = adata.X.copy()
        else:
            self.expression_data = adata.X.toarray().copy()

        self.conditions = np.zeros((adata.shape[0], len(condition_cols)), dtype=np.float32)
        for i, col in enumerate(condition_cols):
            if adata.obs[col].dtype == 'object' or isinstance(adata.obs[col][0], str):
                # If the column is categorical, convert to integers using LabelEncoder
                le = LabelEncoder()
                self.conditions[:, i] = le.fit_transform(adata.obs[col].values)
            else:
                # Directly assign the numerical values
                self.conditions[:, i] = adata.obs[col].values.astype(np.float32)
        
        # print(self.conditions)
        # nan_rows_conditions = np.isnan(self.conditions).any(axis=1)
        # print("Rows with NaN in conditions data:")
        # print(self.conditions[nan_rows_conditions, :])

        self.condition_cols = condition_cols

    def __len__(self):
        return self.expression_data.shape[0]

    def __getitem__(self, idx):
        expression = torch.tensor(self.expression_data[idx, :], dtype=torch.float32)
        condition = torch.tensor(self.conditions[idx, :], dtype=torch.float32)
        return expression, condition

def create_dataloader(adata, condition_cols, batch_size=32, shuffle=True):
    dataset = CVAEDataset(adata, condition_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# TRAINING FUNCTIONS

def early_stopping(new_loss, curr_count_to_patience, global_min_loss):
    """Calculate new patience and validation loss.

    Increment curr_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_patience and global_min_loss
    """
    if new_loss < global_min_loss:
        global_min_loss = new_loss
        curr_count_to_patience = 0

    else:
        curr_count_to_patience = curr_count_to_patience + 1

    return curr_count_to_patience, global_min_loss

# GENE MODULE ANNOTATION SPECIFIC FUNCTIONS FROM VEGA

def dict_to_gmt(dict_obj, path_gmt, sep='\t', second_col=True):
    """ Write dictionary to gmt format """
    with open(path_gmt, 'w') as f:
        for k,v in dict_obj.items():
            if second_col:
                to_write = sep.join([k,'SECOND_COL'] + v)+'\n'
            else:
                to_write = sep.join([k] + v) + '\n'
            f.write(to_write)
    return

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of pathway:genes.
    min_g and max_g are optional gene set size filters.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, n_labels, add_missing=True, fully_connected=True, to_tensor=False):
    """ Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.
    Expects a list of genes and pathway dict.
    If add_missing is True or an int, input genes that are not part of pathways are all connected to a "placeholder" pathway.
    If fully_connected is True, all input genes are connected to the placeholder units.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted."""
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), (len(dict_pathway) + n_labels)))
    for j, k in enumerate(dict_pathway.keys()):
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask

def filter_pathways(pathway_list, pathway_mask, top_k=1000):
    """ Filter pathway by size """
    print('Retaining top ',top_k,' pathways')
    idx_sorted = np.argsort(np.sum(pathway_mask, axis=0))[::-1][:top_k]
    pathway_mask_filtered = pathway_mask[:,idx_sorted]
    pathway_list_filtered = list(np.array(pathway_list)[idx_sorted])
    return pathway_list_filtered, pathway_mask_filtered


