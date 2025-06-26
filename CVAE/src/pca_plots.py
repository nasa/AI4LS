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
unbatch_corrected.obs = unbatch_corrected.obs.drop(columns=['dataset'])


print("Running vstack")
batch_corrected = batch_corrected.X.toarray()
unbatch_corrected = unbatch_corrected.X.toarray()
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
fig.write_html('pca_plot_nogenedrop.html')
fig.write_image('pca_plot.png', scale=2)
fig.show()
