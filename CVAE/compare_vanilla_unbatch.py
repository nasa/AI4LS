import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import src.vanilla_cvae as vanilla_cvae

# 2. Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Load both models
model_vanilla = vanilla_cvae.Vanilla_CVAE(n_genes=2000, n_labels=5, latent_size=64, beta=0.01, lr=0.001, wd=0.1, device=device)
model_vanilla.load_state_dict(torch.load('trained_models/trained_vanilla_cvae.pt', map_location=device))
model_unbatch = vanilla_cvae.Vanilla_CVAE(n_genes=2000, n_labels=5, latent_size=64, beta=0.01, lr=0.001, wd=0.1, device=device)
model_unbatch.load_state_dict(torch.load('trained_models/trained_unbatch_vanilla_cvae.pt', map_location=device))

# 4. Generate 1000 samples each (using same condition vector)
conditions = torch.tensor([0, 0, 20, 30, 1], dtype=torch.float32)
samples_vanilla = model_vanilla.generate(conditions, num_samples=1000).cpu().numpy()
samples_unbatch = model_unbatch.generate(conditions, num_samples=1000).cpu().numpy()

# 5. Save samples
np.save('generated_vanilla_samples.npy', samples_vanilla)
np.save('generated_unbatch_samples.npy', samples_unbatch)

# 6. Run PCA (2D) and plot
pca = PCA(n_components=2)
all_samples = np.concatenate([
    samples_vanilla,
    samples_unbatch
], axis=0)
labels = np.array(['vanilla'] * 1000 + ['unbatch'] * 1000)

pca_result = pca.fit_transform(all_samples)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[labels == 'vanilla', 0], pca_result[labels == 'vanilla', 1], alpha=0.5, label='Vanilla CVAE')
plt.scatter(pca_result[labels == 'unbatch', 0], pca_result[labels == 'unbatch', 1], alpha=0.5, label='Unbatch Vanilla CVAE')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Comparison: Vanilla vs. Unbatch Vanilla CVAE (1000 samples each)')
plt.legend()
plt.tight_layout()
plt.savefig('pca_comparison_vanilla_unbatch.png')
plt.show()
