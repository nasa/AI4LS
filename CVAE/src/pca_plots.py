import utils
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
import plotly.express as px

print("Loading data...")
batch_corrected  = sc.read_h5ad( 'data/corrected_data.h5ad')
unbatch_corrected = sc.read_h5ad( 'data/unbatch_corrected_data.h5ad')
batch_corrected.obs = batch_corrected.obs.drop(columns=['dataset'])

print("Calculating highly variable genes...")
sc.pp.highly_variable_genes(batch_corrected, n_top_genes=2000)
adata = batch_corrected[:, batch_corrected.var['highly_variable']]
unbatch_corrected.obs = unbatch_corrected.obs.drop(columns=['dataset'])

print("Calculating highly variable genes for unbatch corrected data...")
sc.pp.highly_variable_genes(unbatch_corrected, n_top_genes=2000)
unbatch_corrected = unbatch_corrected[:, unbatch_corrected.var['highly_variable']]

print("Running vstack")
combined_data = np.vstack([batch_corrected, unbatch_corrected])
labels = np.array(['Batch Corrected'] * batch_corrected.shape[0] + ['UnBatch Corrrected'] * unbatch_corrected.shape[0])

print("Running PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_data)

# Create a DataFrame for Plotly
pca_df = pd.DataFrame(pca_result, columns=['PCA Component 1', 'PCA Component 2'])

print("Plotting")
# Plotting with Plotly
fig = px.scatter(
    pca_df, x='PCA Component 1', y='PCA Component 2',
    color=labels,
    labels={'PCA Component 1': 'PCA Component 1', 'PCA Component 2': 'PCA Component 2'},
    title='PCA: Existing v. Generated')
fig.update_traces(marker=dict(size=2.5))
fig.write_html('a_plot.html')
fig.show()
