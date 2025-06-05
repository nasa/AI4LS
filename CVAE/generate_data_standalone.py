import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
import pandas as pd
import src.vanilla_cvae as vanilla_cvae
import src.gma_cvae as gma_cvae
import src.utils as utils
import seaborn as sns

# Define random gene names for visualization purposes
np.random.seed(20)
gene_names = [f"gene_{i}" for i in range(2000)]
random_indices = np.random.choice(len(gene_names), size=10, replace=False)
genes_to_plot = [gene_names[i] for i in random_indices]
indices = random_indices  # These will be used for plotting

# GPU enable or disable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Decide which model to use: 'vanilla' for vanilla CVAE
model_for_gen = 'vanilla'  # Change to anything else for GMA CVAE

# Load the appropriate pre-trained model
if model_for_gen == 'vanilla':
    model = vanilla_cvae.Vanilla_CVAE(n_genes=2000, n_labels=5, latent_size=64, beta=0.01, lr=0.001, wd=0.1, device=device)
    model.load_state_dict(torch.load('trained_models/trained_vanilla_cvae.pt', map_location=device))
    print("Vanilla CVAE model loaded successfully")
else:
    # For GMA CVAE
    try:
        gmt_dict = utils.read_gmt('data/BrainGMTv2_MouseOrthologs.gmt', min_g=0, max_g=2000)
        gm_mask = utils.create_pathway_mask(gene_names, gmt_dict, n_labels=5, add_missing=5, fully_connected=True)
        model = gma_cvae.GMA_CVAE(n_labels=5, pathway_mask=gm_mask, lr=0.01)
        model.load_state_dict(torch.load('trained_models/trained_gma_cvae.pt', map_location=device))
        print("GMA CVAE model loaded successfully")
    except Exception as e:
        print(f"Error loading GMA CVAE model: {e}")
        print("Falling back to Vanilla CVAE")
        model = vanilla_cvae.Vanilla_CVAE(n_genes=2000, n_labels=5, latent_size=64, beta=0.01, lr=0.001, wd=0.1, device=device)
        model.load_state_dict(torch.load('trained_models/trained_vanilla_cvae.pt', map_location=device))

# Generate data for each dataset
print("Generating data...")

# Label mapping: {'B6129SF2/J': 0 (612), 'BALB/c': 1 (352), 'C57BL/6NTac': 2 (613)}
conditions = torch.tensor([0, 0, 14, 28, 1], dtype=torch.float32)
generated_data_612 = model.generate(conditions, num_samples=5000).numpy()
print("Generated 612 dataset (B6129SF2/J) - 5000 samples")

conditions = torch.tensor([2, 0, 29, 53, 1], dtype=torch.float32)
generated_data_613 = model.generate(conditions, num_samples=5000).numpy()
print("Generated 613 dataset (C57BL/6NTac) - 5000 samples")

conditions = torch.tensor([1, 0, 12, 41, 1], dtype=torch.float32)
generated_data_352 = model.generate(conditions, num_samples=5000).numpy()
print("Generated 352 dataset (BALB/c) - 5000 samples")

# Combine all generated data
generated_data = np.concatenate((generated_data_612, generated_data_613, generated_data_352))
dataset_labels = np.array(['612'] * 5000 + ['613'] * 5000 + ['352'] * 5000)

# Create a DataFrame for the generated data
df_generated = pd.DataFrame(generated_data)
df_generated['dataset'] = dataset_labels
df_generated.to_csv('generated_data.csv', index=False)
print("Data saved to generated_data.csv")

# Calculate statistics for the generated data
generated_means_612 = np.mean(generated_data_612, axis=0)
generated_std_612 = np.std(generated_data_612, axis=0)
generated_vars_612 = np.var(generated_data_612, axis=0)

# Define visualization functions
def plot_pca_2d():
    """Generate 2D PCA plot of the data"""
    print("Generating 2D PCA plot...")
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(generated_data)
    
    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(pca_result, columns=['PCA Component 1', 'PCA Component 2'])
    pca_df['Dataset'] = dataset_labels
    
    # Plotting with Plotly
    fig = px.scatter(
        pca_df, x='PCA Component 1', y='PCA Component 2',
        color='Dataset',
        labels={'PCA Component 1': 'PCA Component 1', 'PCA Component 2': 'PCA Component 2'},
        title='PCA: Generated Data'
    )
    fig.update_traces(marker=dict(size=2.5))
    fig.write_html('pca_2d_plot.html')
    fig.show()

def plot_pca_3d():
    """Generate 3D PCA plot of the data"""
    print("Generating 3D PCA plot...")
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(generated_data)
    
    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(pca_result, columns=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])
    pca_df['Dataset'] = dataset_labels
    
    # Plotting with Plotly
    fig = px.scatter_3d(
        pca_df, x='PCA Component 1', y='PCA Component 2', z='PCA Component 3',
        color='Dataset',
        labels={'PCA Component 1': 'PCA Component 1', 'PCA Component 2': 'PCA Component 2', 'PCA Component 3': 'PCA Component 3'},
        title='PCA: Generated Data'
    )
    fig.update_traces(marker=dict(size=2))
    fig.write_html('pca_3d_plot.html')
    fig.show()

def plot_umap():
    """Generate UMAP plot of the data with clustering"""
    print("Generating UMAP plot (this may take a while)...")
    
    # Perform UMAP dimensionality reduction
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_result = umap_model.fit_transform(generated_data)
    
    # Perform KMeans clustering
    num_clusters = 6  # Define the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(umap_result)
    
    # Create a DataFrame for plotting
    umap_df = pd.DataFrame(umap_result, columns=['UMAP Component 1', 'UMAP Component 2'])
    umap_df['Cluster'] = cluster_labels
    umap_df['Dataset'] = dataset_labels
    
    # Plotting with Plotly - Color by Dataset
    fig = px.scatter(
        umap_df, x='UMAP Component 1', y='UMAP Component 2',
        color='Dataset',
        labels={'Dataset': 'Dataset'},
        title='UMAP: Color by Dataset'
    )
    fig.update_traces(marker=dict(size=3.5))
    fig.write_html('umap_dataset_plot.html')
    fig.show()
    
    # Plotting with Plotly - Color by Cluster
    fig = px.scatter(
        umap_df, x='UMAP Component 1', y='UMAP Component 2',
        color='Cluster',
        labels={'Cluster': 'Cluster'},
        title='UMAP: Color by Cluster'
    )
    fig.update_traces(marker=dict(size=3.5))
    fig.write_html('umap_cluster_plot.html')
    fig.show()

# Additional visualization functions that can be called if needed
def plot_heatmap():
    """Generate heatmap of a subset of the generated data"""
    print("Creating heatmap of a subset of the generated data...")
    # Take a subset of the data for better visualization
    sample_indices = np.random.choice(generated_data.shape[0], size=50, replace=False)
    gene_indices = np.random.choice(generated_data.shape[1], size=50, replace=False)
    subset_data = generated_data[sample_indices][:, gene_indices]

    plt.figure(figsize=(12, 10))
    sns.heatmap(subset_data, cmap='viridis')
    plt.title('Heatmap of Generated Data Subset')
    plt.xlabel('Gene Index')
    plt.ylabel('Sample Index')
    plt.savefig('heatmap.png')
    plt.show()

def plot_histograms():
    """Generate histograms of gene expression values for each dataset"""
    print("Creating histograms of gene expression values...")
    plt.figure(figsize=(12, 6))
    
    # Plot histogram for each dataset
    datasets = ['612', '613', '352']
    dataset_data = [generated_data_612, generated_data_613, generated_data_352]
    
    for i, (dataset, data) in enumerate(zip(datasets, dataset_data)):
        # Flatten the data to get all expression values
        all_values = data.flatten()
        plt.hist(all_values, bins=100, alpha=0.5, label=dataset)

    plt.title('Distribution of Gene Expression Values by Dataset')
    plt.xlabel('Expression Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('expression_distribution.png')
    plt.show()

def plot_boxplots():
    """Generate box plots for selected genes across datasets"""
    print("Creating box plots for selected genes...")
    # Select a few random genes for visualization
    selected_genes = np.random.choice(generated_data.shape[1], size=10, replace=False)
    fig, axes = plt.subplots(5, 2, figsize=(15, 20), sharex=True)
    axes = axes.flatten()

    # Prepare data for each dataset
    datasets = ['612', '613', '352']
    dataset_data = [generated_data_612, generated_data_613, generated_data_352]
    
    for i, gene_idx in enumerate(selected_genes):
        gene_data = []
        for data in dataset_data:
            gene_data.append(data[:, gene_idx])
        
        axes[i].boxplot(gene_data, labels=datasets)
        axes[i].set_title(f'Gene {gene_idx}')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('gene_boxplots.png')
    plt.show()

# Run the main visualizations
print("\nGenerating visualizations...")
plot_pca_2d()
plot_pca_3d()
plot_umap()

# To run additional visualizations, uncomment the following lines:
# plot_heatmap()
# plot_histograms()
# plot_boxplots()

print("\nData generation and visualization complete.")
print("Additional visualization functions are available but not automatically run:")
print(" - plot_heatmap(): Creates a heatmap of a subset of generated data")
print(" - plot_histograms(): Shows distribution of gene expression values by dataset")
print(" - plot_boxplots(): Creates box plots for selected genes across datasets")
