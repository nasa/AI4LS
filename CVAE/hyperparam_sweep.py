import torch
import itertools
import numpy as np
import src.utils as utils
import scanpy as sc
from scipy import sparse
import pandas as pd
import src.vanilla_cvae as vanilla_cvae
import src.gma_cvae as gma_cvae
from sklearn.model_selection import KFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rates = [0.001, 0.01, 0.1]
weight_decay = [0.001, 0.01, 0.1]
betas = [0.01, 0.05, 0.1] # go higher?
latent_sizes = [16, 32, 64] # maybe 128?

# K-fold cross-validation setup
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Load data
adata = sc.read_h5ad('data/corrected_data.h5ad')
adata.obs = adata.obs.drop(columns=['dataset'])
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

# Create dataloaders
condition_cols = ['Strain', 'Sex', 'Age at Launch', 'Duration', 'Flight']
train_adata, val_adata = utils.split_adata(adata, condition_cols, val_split=0.3, random_state=42)

print("------------VANILLA CVAE------------")

# Store results for each hyperparameter combination
vanilla_results = []

# Perform hyperparameter sweep
for lr, wd, beta, latent_size in itertools.product(learning_rates, weight_decay, betas, latent_sizes):
    fold_results = []
    for fold, (train_index, val_index) in enumerate(kf.split(train_adata)):
        print(f"Fold {fold+1}")
        
        # Split data for fold
        fold_train_data = train_adata[train_index]
        fold_val_data = train_adata[val_index]
        
        # Create dataloaders
        train_loader = utils.create_dataloader(fold_train_data, condition_cols, batch_size=128)
        val_loader = utils.create_dataloader(fold_val_data, condition_cols, batch_size=128)
        
        # Initialize model
        model = vanilla_cvae.Vanilla_CVAE(n_genes=2000, n_labels=5, latent_size=latent_size, lr=lr, beta=beta, wd=wd,
                                          verbose=False, device=device, save_model=False)
        
        # Train model
        model.train_net(train_loader=train_loader, n_epochs=1, save_model=False)
        print("done")
        # Evaluate model on validation set
        val_dict = model.test_model(val_loader)
        val_loss = val_dict['loss']
        fold_results.append(val_loss)
        print(fold_results)
    
    # Calculate average validation loss for this hyperparameter combo
    avg_val_loss = sum(fold_results) / len(fold_results)
    vanilla_results.append((lr, beta, latent_size, avg_val_loss))
    print(f"Hyperparameters: lr={lr}, beta={beta}, latent_size={latent_size} -> Avg Val Loss: {avg_val_loss}")

# Find the best hyperparameters based on validation loss
best_van_hyperparams = min(vanilla_results, key=lambda x: x[3])
print(f"Best hyperparameters: Learning Rate: {best_van_hyperparams[0]}, Beta: {best_van_hyperparams[1]}, Latent Size: {best_van_hyperparams[2]}")

with open('best_hyperparameters.txt', 'w') as file:
    file.write(f"Best vanilla hyperparameters: Learning Rate: {best_van_hyperparams[0]}, Beta: {best_van_hyperparams[1]}, Latent Size: {best_van_hyperparams[2]}")

print("------------GENE MODULE ANNOTATED CVAE------------")

# Create pathway mask
gmt_dict = utils.read_gmt('data/GL-DPPD-7111_Mmus_Brain_CellType_GeneMarkers.gmt', min_g=0, max_g=1000)
gm_mask = utils.create_pathway_mask(train_adata.var.index.tolist(), gmt_dict, n_labels=5, add_missing=5, fully_connected=True)

gma_results = []

for lr, wd, beta in itertools.product(learning_rates, weight_decay, betas):
    fold_results = []
    for fold, (train_index, val_index) in enumerate(kf.split(train_adata)):
        print(f"Fold {fold+1}")
        
        # Split data for fold
        fold_train_data = train_adata[train_index]
        fold_val_data = train_adata[val_index]
        
        # Create dataloaders
        train_loader = utils.create_dataloader(fold_train_data, condition_cols, batch_size=128)
        val_loader = utils.create_dataloader(fold_val_data, condition_cols, batch_size=128)
        
        # Initialize model
        model = gma_cvae.GMA_CVAE(n_labels=5, pathway_mask=gm_mask, lr=lr, beta=beta,
                                  device=device)
        model.train_net(train_loader=train_loader, test_loader=val_loader, n_epochs=20, save_model=False)
        
        # Train model
        model.train_net(train_loader=train_loader, n_epochs=30)
        
        # Evaluate model on validation set
        val_dict = model.test_model(val_loader)
        val_loss = val_dict['loss']
        fold_results.append(val_loss)
    
    # Calculate average validation loss for this hyperparameter combo
    avg_val_loss = sum(fold_results) / len(fold_results)
    gma_results.append((lr, wd, beta, avg_val_loss))
    print(f"Hyperparameters: lr={lr}, wd={wd}, beta={beta} -> Avg Val Loss: {avg_val_loss}")

# Find the best hyperparameters based on validation loss
best_gma_hyperparams = min(gma_results, key=lambda x: x[3])
print(f"Best hyperparameters: Learning Rate: {best_gma_hyperparams[0]}, Beta: {best_gma_hyperparams[1]}, Latent Size: {best_gma_hyperparams[2]}")

with open('best_hyperparameters.txt', 'w') as file:
    file.write(f"Best GMA hyperparameters: Learning Rate: {best_gma_hyperparams[0]}, Beta: {best_gma_hyperparams[1]}, Latent Size: {best_gma_hyperparams[2]}")