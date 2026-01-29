#!/usr/bin/env python
# coding: utf-8

# In[25]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from model.bulkformer import BulkFormer, model_params

device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[26]:


ckpt_path = "bulkformer_checkpoints/best_model.pt"   # or whichever epoch you want
state_dict = torch.load(ckpt_path, map_location="cpu")

import json

with open("bulkformer_checkpoints/config.json", "r") as f:
    cfg = json.load(f)

cfg

#esm2_data = torch.load("./data/embeddings/esm2_t6_8M_UR50D_gene_embeddings.pt")
#gene_emb = esm2_data["embeddings"].float()
esm3_data = torch.load("./data/embeddings/esm3_sm_open_v1_gene_embeddings.pt")
gene_emb = esm3_data["embeddings"].float()


edge_index = torch.load("./graph/edge_index_top20.pt").long()


# In[27]:


model = BulkFormer(
    dim=cfg["dim"],
    graph=edge_index,                # You need to load this (see below)
    gene_emb=gene_emb,             # And this
    gene_length=cfg["num_genes"],
    bin_head=cfg["bin_head"],
    full_head=cfg["full_head"],
    bins=cfg["bins"],
    gb_repeat=cfg["gb_repeat"],
    p_repeat=cfg["p_repeat"],
)



# In[28]:


model.load_state_dict(state_dict)
model = model.to("cuda")



# In[29]:


with torch.no_grad():
    # gene identity embedding only
    gene_embed = model.gene_emb_proj(model.gene_emb)  # shape (19357, 320)

gene_embed = gene_embed.cpu().numpy()
gene_embed.shape



# In[30]:


tsne = TSNE(
    n_components=2,
    learning_rate="auto",
    init="random",
    perplexity=30,
    random_state=42
)

tsne_emb = tsne.fit_transform(gene_embed)
tsne_emb.shape


# In[31]:


plt.figure(figsize=(10,10))
plt.scatter(tsne_emb[:,0], tsne_emb[:,1], s=3, alpha=0.6)
plt.title("t-SNE of Gene Embeddings")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")
plt.savefig("figures/tsne-gene-embeddings.png")

# In[32]:


k = 8  # number of clusters
km = KMeans(n_clusters=k, random_state=42)
clusters = km.fit_predict(gene_embed)

clusters[:20]

plt.figure(figsize=(10,10))

# Scatter points
plt.scatter(
    tsne_emb[:,0], tsne_emb[:,1],
    c=clusters,
    cmap="tab10",
    s=4,
    alpha=0.8
)

# Add cluster labels at the cluster centroid
for c in range(k):
    cx = tsne_emb[clusters == c, 0].mean()
    cy = tsne_emb[clusters == c, 1].mean()
    plt.text(
        cx, cy, str(c),
        fontsize=20, fontweight="bold",
        color="black",
        ha="center", va="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor='none')
    )

plt.title("K-means Clusters (k=10) with Labels")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")
plt.savefig("figures/kmeans_clusters_k=10_with_labels.png")


# In[33]:


import hdbscan

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='leaf',   
)
clusters = clusterer.fit_predict(gene_embed)
clusters[:20]

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,10))

# Mask noise points (-1)
is_noise = clusters == -1
is_cluster = clusters != -1

# Plot noise in light gray
plt.scatter(
    tsne_emb[is_noise, 0],
    tsne_emb[is_noise, 1],
    c='lightgray',
    s=4,
    alpha=0.4,
    label='Noise'
)

# Plot clusters
plt.scatter(
    tsne_emb[is_cluster, 0],
    tsne_emb[is_cluster, 1],
    c=clusters[is_cluster],
    cmap='tab20',
    s=4,
    alpha=0.9
)

# Get number of clusters (excluding noise)
unique_clusters = [c for c in np.unique(clusters) if c != -1]

# Add cluster labels at centroid of each cluster
for c in unique_clusters:
    cx = tsne_emb[clusters == c, 0].mean()
    cy = tsne_emb[clusters == c, 1].mean()

    plt.text(
        cx, cy, str(c),
        fontsize=20, fontweight="bold",
        color="black",
        ha="center", va="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor='none')
    )

plt.title("HDBSCAN Clusters (variable k) with Labels")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")
plt.savefig("figures/hdbscan_clusters_variable_k_with_labels.png")


# In[34]:


from sklearn.cluster import KMeans
import numpy as np

k = 25  # number of clusters

km = KMeans(
    n_clusters=k,
    init='k-means++',
    random_state=42,
    max_iter=300,
    n_init=10,                    
)
clusters = km.fit_predict(tsne_emb)

plt.figure(figsize=(10,10))

# Scatter points
plt.scatter(
    tsne_emb[:,0], tsne_emb[:,1],
    c=clusters,
    cmap="tab10",
    s=4,
    alpha=0.8
)

# Add cluster labels at the cluster centroid
for c in range(k):
    cx = tsne_emb[clusters == c, 0].mean()
    cy = tsne_emb[clusters == c, 1].mean()
    plt.text(
        cx, cy, str(c),
        fontsize=20, fontweight="bold",
        color="black",
        ha="center", va="center",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor='none')
    )

plt.title("K-means Clusters (k=10) with Labels")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")
plt.savefig("figures/kmeans_clusters_k=10_with_labels_2.png")


# In[35]:


# load the gene order file from training if you have one
gene_order = pd.read_csv("./data/archs4/processed_short_proteins/test_gene_order_short.csv")

print("Columns:", gene_order.columns.tolist())
print(gene_order.head(10))

gene_names = gene_order["gene_symbol"].tolist()

len(gene_names)


# In[36]:


enrichment_results = {}

for c in range(k):
    gene_list = [gene_names[i] for i in range(len(gene_names)) if clusters[i] == c]

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=["GO_Biological_Process_2021"],
        organism="Human",
        cutoff=0.05
    )

    enrichment_results[c] = enr.results
    print(f"Cluster {c}: {len(enr.results)} significant GO terms")


# In[37]:


kegg_results = {}

for c in range(k):
    gene_list = [gene_names[i] for i in range(len(gene_names)) if clusters[i] == c]

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=["KEGG_2021_Human"],
        organism="Human",
        cutoff=0.05
    )

    kegg_results[c] = enr.results
    print(f"Cluster {c}: {len(enr.results)} KEGG pathways")


# In[38]:


go_counts = [len(enrichment_results[c]) for c in range(k)]

plt.figure(figsize=(10,5))
plt.bar(range(k), go_counts)
plt.xlabel("Cluster")
plt.ylabel("Number of enriched GO terms")
plt.title("GO Enrichment Count per Cluster")
plt.savefig("figures/go_enrichment_count_per_cluster.png")

# In[39]:


def compute_rich_factor(df):
    rf = []
    for s in df["Overlap"]:
        a, b = s.split("/")
        rf.append(int(a) / int(b))
    return np.array(rf)


# In[40]:


cell_cycle_terms = [
    "cell cycle",
    "mitotic",
    "chromosome",
    "DNA replication",
    "splicing",
    "RNA splicing",
    "DNA repair",
    "cell division",
    "chromatin organization"
]


immune_terms = [
    "immune",
    "defense",
    "antigen",
    "interferon",
    "T cell",
    "B cell",
    "inflammatory",
    "cytokine",
    "pathogen",
    "response to bacterium",
]

import re

def count_matches(df, keywords):
    if df is None or len(df) == 0:
        return 0
    pattern = "|".join([re.escape(k) for k in keywords])
    return df["Term"].str.contains(pattern, case=False, regex=True).sum()

cluster_theme_scores = {}

for c in range(k):
    df = enrichment_results[c]

    score_cellcycle = count_matches(df, cell_cycle_terms)
    score_immune    = count_matches(df, immune_terms)

    cluster_theme_scores[c] = {
        "cell_cycle_score": score_cellcycle,
        "immune_score": score_immune,
        "n_terms": len(df)
    }

cluster_theme_scores



# In[41]:


c1, c2 = 15, 23

df1 = enrichment_results[c1].copy()
df2 = enrichment_results[c2].copy()

df1["rich_factor"] = compute_rich_factor(df1)
df2["rich_factor"] = compute_rich_factor(df2)

# Top 12 for clean plot
df1_top = df1.nlargest(12, "rich_factor")
df2_top = df2.nlargest(12, "rich_factor")

plt.figure(figsize=(12,7))
plt.barh(df1_top["Term"], df1_top["rich_factor"], alpha=0.6, label=f"Cluster {c1}")
plt.barh(df2_top["Term"], df2_top["rich_factor"], alpha=0.6, label=f"Cluster {c2}")
plt.xlabel("Rich factor")
plt.title(f"GO Enrichment Comparison: Cluster {c1} vs Cluster {c2}")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("figures/go_enrichment_comparison_cluster_" + str(c1) + "_vs_cluster_" + str(c2) + ".png")

print(f"\nTop 5 GO terms for Cluster {c1}:")
for i, (term, rf) in enumerate(zip(df1_top["Term"].head(5),
                                   df1_top["rich_factor"].head(5)), 1):
    print(f"{i}. {term} (rich factor = {rf:.3f})")

print(f"\nTop 5 GO terms for Cluster {c2}:")
for i, (term, rf) in enumerate(zip(df2_top["Term"].head(5),
                                   df2_top["rich_factor"].head(5)), 1):
    print(f"{i}. {term} (rich factor = {rf:.3f})")


# In[42]:


c1, c2 = 9, 21

df1 = enrichment_results[c1].copy()
df2 = enrichment_results[c2].copy()

df1["rich_factor"] = compute_rich_factor(df1)
df2["rich_factor"] = compute_rich_factor(df2)

# Top 12 for clean plot
df1_top = df1.nlargest(12, "rich_factor")
df2_top = df2.nlargest(12, "rich_factor")

plt.figure(figsize=(12,7))
plt.barh(df1_top["Term"], df1_top["rich_factor"], alpha=0.6, label=f"Cluster {c1}")
plt.barh(df2_top["Term"], df2_top["rich_factor"], alpha=0.6, label=f"Cluster {c2}")
plt.xlabel("Rich factor")
plt.title(f"GO Enrichment Comparison: Cluster {c1} vs Cluster {c2}")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("figures/go_enrichment_comparison_cluster_" + str(c1) + "_vs_cluster_" + str(c2) + ".png")

print(f"\nTop 5 GO terms for Cluster {c1}:")
for i, (term, rf) in enumerate(zip(df1_top["Term"].head(5),
                                   df1_top["rich_factor"].head(5)), 1):
    print(f"{i}. {term} (rich factor = {rf:.3f})")

print(f"\nTop 5 GO terms for Cluster {c2}:")
for i, (term, rf) in enumerate(zip(df2_top["Term"].head(5),
                                   df2_top["rich_factor"].head(5)), 1):
    print(f"{i}. {term} (rich factor = {rf:.3f})")



# In[43]:


import matplotlib.pyplot as plt

# Prepare cluster → df mapping
cluster_top = {}

for c in range(k):
    df = enrichment_results[c].copy()
    df["rich_factor"] = compute_rich_factor(df)

    # keep top N
    df_top = df.nlargest(10, "rich_factor")
    cluster_top[c] = df_top

# Plot
rows = (k + 2) // 2
cols = 2
fig, axes = plt.subplots(rows, cols, figsize=(18, 22), sharex=True)

for idx, c in enumerate(range(k)):
    ax = axes[idx // cols][idx % cols]
    df = cluster_top[c]

    ax.barh(df["Term"], df["rich_factor"], color="steelblue")
    ax.set_title(f"Cluster {c} — Top GO terms", fontsize=10)
    ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=7)

fig.suptitle("GO Enrichment (Top Terms per Cluster)", fontsize=16)
plt.tight_layout()
plt.savefig("figures/go_enrichment_top_terms_per_cluster.png")

# In[44]:


kegg_counts = [len(kegg_results[c]) for c in range(k)]

plt.figure(figsize=(10,5))
plt.bar(range(k), kegg_counts)
plt.xlabel("Cluster")
plt.ylabel("Number of enriched KEGG pathways")
plt.title("KEGG Pathway Count per Cluster")
plt.savefig("figures/kegg_pathway_count_per_cluster.png")

# In[45]:


def compute_kegg_rich_factor(df):
    rf = []
    for s in df["Overlap"]:
        a, b = s.split("/")
        rf.append(int(a) / int(b))
    return np.array(rf)


c1, c2 = 7, 10

df1 = kegg_results[c1].copy()
df2 = kegg_results[c2].copy()

df1["rich_factor"] = compute_kegg_rich_factor(df1)
df2["rich_factor"] = compute_kegg_rich_factor(df2)

df1_top = df1.nlargest(12, "rich_factor")
df2_top = df2.nlargest(12, "rich_factor")

plt.figure(figsize=(12,7))
plt.barh(df1_top["Term"], df1_top["rich_factor"], alpha=0.6, label=f"Cluster {c1}")
plt.barh(df2_top["Term"], df2_top["rich_factor"], alpha=0.6, label=f"Cluster {c2}")
plt.xlabel("Rich factor")
plt.title(f"KEGG Enrichment Comparison: Cluster {c1} vs Cluster {c2}")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("figures/kegg_enrichment_comparison_cluster_" + str(c1) + "_vs_cluster_" + str(c2) + ".png")

print(f"\nTop 5 KEGG terms for Cluster {c1}:")
for i, (term, rf) in enumerate(zip(df1_top["Term"].head(5),
                                   df1_top["rich_factor"].head(5)), 1):
    print(f"{i}. {term} (rich factor = {rf:.3f})")

print(f"\nTop 5 KEGG terms for Cluster {c2}:")
for i, (term, rf) in enumerate(zip(df2_top["Term"].head(5),
                                   df2_top["rich_factor"].head(5)), 1):
    print(f"{i}. {term} (rich factor = {rf:.3f})")


# In[46]:


def compute_kegg_rich_factor(df):
    rf = []
    for s in df["Overlap"]:
        a, b = s.split("/")
        rf.append(int(a) / int(b))
    return np.array(rf)


c1, c2 = 14, 21

df1 = kegg_results[c1].copy()
df2 = kegg_results[c2].copy()

df1["rich_factor"] = compute_kegg_rich_factor(df1)
df2["rich_factor"] = compute_kegg_rich_factor(df2)

df1_top = df1.nlargest(12, "rich_factor")
df2_top = df2.nlargest(12, "rich_factor")

plt.figure(figsize=(12,7))
plt.barh(df1_top["Term"], df1_top["rich_factor"], alpha=0.6, label=f"Cluster {c1}")
plt.barh(df2_top["Term"], df2_top["rich_factor"], alpha=0.6, label=f"Cluster {c2}")
plt.xlabel("Rich factor")
plt.title(f"KEGG Enrichment Comparison: Cluster {c1} vs Cluster {c2}")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("figures/kegg_enrichment_comparison_cluster_" + str(c1) + "_vs_cluster_" + str(c2) + ".png")

print(f"\nTop 5 KEGG terms for Cluster {c1}:")
for i, (term, rf) in enumerate(zip(df1_top["Term"].head(5),
                                   df1_top["rich_factor"].head(5)), 1):
    print(f"{i}. {term} (rich factor = {rf:.3f})")

print(f"\nTop 5 KEGG terms for Cluster {c2}:")
for i, (term, rf) in enumerate(zip(df2_top["Term"].head(5),
                                   df2_top["rich_factor"].head(5)), 1):
    print(f"{i}. {term} (rich factor = {rf:.3f})")


# In[47]:


import matplotlib.pyplot as plt

# Compute top KEGG pathways per cluster
cluster_top_kegg = {}

for c in range(k):
    df = kegg_results[c].copy()
    df["rich_factor"] = compute_rich_factor(df)

    # keep top 10 KEGG terms
    df_top = df.nlargest(10, "rich_factor")
    cluster_top_kegg[c] = df_top

# Plot: 5 rows × 2 columns = 10 clusters
rows, cols = (k + 2) // 2, 2
fig, axes = plt.subplots(rows, cols, figsize=(18, 22), sharex=True)

for idx, c in enumerate(range(k)):
    ax = axes[idx // cols][idx % cols]
    df = cluster_top_kegg[c]

    ax.barh(df["Term"], df["rich_factor"], color="darkorange", alpha=0.7)
    ax.set_title(f"Cluster {c} — Top KEGG pathways", fontsize=10)
    ax.invert_yaxis()  # largest factor at top
    ax.tick_params(axis='y', labelsize=7)

fig.suptitle("KEGG Enrichment (Top pathways per Cluster)", fontsize=16)
plt.tight_layout()
plt.savefig("figures/kegg_enrichment_top_pathways_per_cluster_2.png")


# In[48]:


# Canonical squamous epithelial program markers
squamous_markers = [
    "KRT5","KRT6A","KRT6B","KRT6C","KRT14","KRT16","KRT17","KRT19",
    "DSC1","DSC2","DSG1","DSG2","DSG3",
    "JUP","PKP1","PKP3",
    "IVL","SPRR1A","SPRR1B","SPRR2A","SPRR2B",
    "LCE1A","LCE1B","LCE1C","LCE2A","LCE2B",
    "TP63","SOX2","GRHL3","ELF3"
]

# Keep only genes that exist in your expression dataset
squamous_present = [g for g in squamous_markers if g in gene_names]
print("Squamous markers present:", squamous_present)

squamous_gene_mask = np.array([1 if g in squamous_present else 0 for g in gene_names])


# In[49]:


import pandas as pd
import seaborn as sns

cluster_squamous_counts = []

for c in range(k):
    cluster_indices = np.where(clusters == c)[0]
    genes_in_cluster = [gene_names[i] for i in cluster_indices]
    count = sum(1 for g in squamous_present if g in genes_in_cluster)
    cluster_squamous_counts.append(count)

df_sq = pd.DataFrame({
    "cluster": list(range(k)),
    "squamous_markers": cluster_squamous_counts
})

plt.figure(figsize=(8,5))
sns.barplot(data=df_sq, x="cluster", y="squamous_markers", palette="viridis")
plt.title("Squamous epithelial program markers per cluster")
plt.xlabel("Cluster")
plt.ylabel("# squamous markers")
plt.savefig("figures/squamous_epithelial_program_markers_per_cluster.png")

