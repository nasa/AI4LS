import torch
import numpy as np
import src.utils as utils
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import src.vanilla_cvae as vanilla_cvae
import src.gma_cvae as gma_cvae

# To train the Vanilla CVAE, leave as 'vanilla', otherwise, change to gma
model_to_train = 'vanilla'

# Read in data
#adata = sc.read_h5ad('data/corrected_data.h5ad')
adata = sc.read_h5ad('data/unbatch_corrected_data.h5ad')
adata.obs = adata.obs.drop(columns=['dataset'])
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

# Create the data loaders
condition_cols = ['Strain', 'Sex', 'Age at Launch', 'Duration', 'Flight']
train_adata, val_adata = utils.split_adata(adata, condition_cols, val_split=0.3, random_state=42)
train_dataloader = utils.create_dataloader(train_adata, condition_cols, batch_size=128)
val_dataloader = utils.create_dataloader(val_adata, condition_cols, batch_size=128)

# GPU enable or disable
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train/test vanilla cvae
if model_to_train == 'vanilla':
    van_cvae = vanilla_cvae.Vanilla_CVAE(n_genes=2000, n_labels=5, latent_size=32, lr=0.01, device=device)
    van_cvae.train_net(train_loader=train_dataloader, test_loader=val_dataloader, n_epochs=30)

# Train/test gma cvae
else:
    # Read in .gmt/create pathway mask
    gmt_dict = utils.read_gmt('data/GL-DPPD-7111_Mmus_Brain_CellType_GeneMarkers.gmt', min_g=0, max_g=1000)
    gm_mask = utils.create_pathway_mask(train_adata.var.index.tolist(), gmt_dict, n_labels=5, add_missing=5, fully_connected=True)
    print(gm_mask.shape)
    print(gm_mask)
    gma_cvae = gma_cvae.GMA_CVAE(n_labels=5, pathway_mask=gm_mask, lr=0.01)
    gma_cvae.train_net(train_loader = train_dataloader, test_loader = val_dataloader, n_epochs = 30)