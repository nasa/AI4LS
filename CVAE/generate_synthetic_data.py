import torch
import requests
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import src.utils as utils
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
import pandas as pd
import src.vanilla_cvae as vanilla_cvae
import src.gma_cvae as gma_cvae

# Read in integrated data
adata = sc.read_h5ad('data/corrected_data.h5ad')
adata.obs = adata.obs.drop(columns=['dataset'])
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]

# GPU enable or disable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Decide which model to use: 'vanilla' for vanilla CVAE
model_for_gen = 'vanilla'

# Vanilla CVAE
if model_for_gen is 'vanilla':
    # model = vanilla_cvae.Vanilla_CVAE(n_genes = 1000, n_labels = 5, latent_size = 32, lr = 0.01)
    # van_cvae.train_net(train_loader = train_dataloader, test_loader = val_dataloader, n_epochs = 30)
    # model.load_state_dict(torch.load('trained_models/trained_gma_cvae.pt', map_location=device))
    model = vanilla_cvae.Vanilla_CVAE(n_genes=2000, n_labels=5, latent_size=64, beta=0.01, lr=0.001, wd=0.1, device=device)
    model.load_state_dict(torch.load('trained_models/trained_vanilla_cvae.pt', map_location=device))
# GMA CVAE
else:
    gmt_dict = utils.read_gmt('data/GL-DPPD-7111_Mmus_Brain_CellType_GeneMarkers.gmt', min_g=0, max_g=1000)
    gm_mask = utils.create_pathway_mask(adata.var.index.tolist(), gmt_dict, n_labels=5, add_missing=5, fully_connected=True)
    model = gma_cvae.GMA_CVAE(n_labels=5, pathway_mask=gm_mask, lr=0.01)
    # gma_cvae.train_net(train_loader = train_dataloader, test_loader = val_dataloader, n_epochs = 30)
    model.load_state_dict(torch.load('trained_models/trained_gma_cvae.pt', map_location=device))

condition_cols = ['Strain', 'Sex', 'Age at Launch', 'Duration', 'Flight']

# generate data
existing_data = adata.X.toarray()
# modify conditions to desired values ['Strain' = 0 - 2, 'Sex' = 0 (female), 'Age at Launch' = int of weeks, 
# 'Duration' = int of days, 'Flight' = 0 for ground, 1 for flight]
conditions = torch.tensor([0, 0, 20, 30, 1], dtype=torch.float32)
generated_data = model.generate(conditions, num_samples = 1000).numpy()

# unique_elements, counts = np.unique(generated_data, return_counts=True)

# # Print the unique elements and their counts
# for element, count in zip(unique_elements, counts):
#     print(f"Element {element} occurs {count} times")

combined_data = np.vstack([existing_data, generated_data])
labels = np.array(['Existing'] * existing_data.shape[0] + ['Generated'] * generated_data.shape[0])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_data)

# Create a DataFrame for Plotly
pca_df = pd.DataFrame(pca_result, columns=['PCA Component 1', 'PCA Component 2'])

# Plotting with Plotly
fig = px.scatter(
    pca_df, x='PCA Component 1', y='PCA Component 2',
    color=labels,
    labels={'PCA Component 1': 'PCA Component 1', 'PCA Component 2': 'PCA Component 2'},
    title='PCA: Existing v. Generated'
)
fig.update_traces(marker=dict(size=2.5))
fig.show()

pca = PCA(n_components=3)
pca_result = pca.fit_transform(combined_data)

# Create a DataFrame for Plotly
pca_df = pd.DataFrame(pca_result, columns=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])

# Plotting with Plotly
fig = px.scatter_3d(
    pca_df, x='PCA Component 1', y='PCA Component 2', z = 'PCA Component 3',
    color=labels,
    labels={'PCA Component 1': 'PCA Component 1', 'PCA Component 2': 'PCA Component 2'},
    title='PCA: Existing v. Generated'
)
fig.update_traces(marker=dict(size=2))
fig.show()

umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
umap_result = umap_model.fit_transform(combined_data)

# Perform KMeans clustering
num_clusters = 6  # Define the number of clusters you want
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(umap_result)

umap_df = pd.DataFrame(umap_result, columns=['UMAP Component 1', 'UMAP Component 2'])
umap_df['Cluster'] = cluster_labels
umap_df['Data Type'] = ['Existing'] * existing_data.shape[0] + ['Generated'] * generated_data.shape[0]

# Plotting with Plotly
fig = px.scatter(
    umap_df, x='UMAP Component 1', y='UMAP Component 2',
    color=labels,
    symbol='Data Type',
    labels={'Cluster': 'Cluster', 'Data Type': 'Data Type'},
    title='UMAP: Color by Type'
)
fig.update_traces(marker=dict(size=3.5))
# Show the plot
fig.show()