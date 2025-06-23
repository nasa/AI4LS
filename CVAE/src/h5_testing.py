import utils
import anndata
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import scanpy.external as sce
import scanorama
import os


adata = sc.read_h5ad('data/corrected_data.h5ad')
adata.obs = adata.obs.drop(columns=['dataset'])
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']]
print(adata.obs)
print(adata.var)
