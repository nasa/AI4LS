import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SpaceflightDataset
from model import SpaceflightCVAE

dataset = SpaceflightDataset("/home/jcasalet/nobackup/CVAE/DATA/subset_final.h6")
ckpt    = torch.load("/home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt", map_location="cpu", weights_only=False)

model = SpaceflightCVAE(
    n_genes=dataset.n_genes, n_strains=dataset.n_strains,
    n_sexes=dataset.n_sexes, n_studies=dataset.n_studies,
    n_tissues=dataset.n_tissues, n_euths=dataset.n_euths,
    latent_dim=64, hidden_dims=[256, 128], dropout=0.0, grl_alpha=0.0
)
model.load_state_dict(ckpt["model_state"])
model.eval()

loader = DataLoader(dataset, batch_size=64, shuffle=False)
all_mu = []
with torch.no_grad():
    for batch in loader:
        mu = model.encode(
            batch["x"], batch["strain"], batch["sex"],
            batch["study"], batch["tissue"], batch["euth"], batch["flight"]
        )
        all_mu.append(mu.numpy())

mu_all = np.concatenate(all_mu)   # (2080, 32)
dim_var = mu_all.var(axis=0)      # variance of each latent dimension

print("Active dimensions (var > 0.1):", (dim_var > 0.1).sum())
print("Collapsed dimensions (var < 0.01):", (dim_var < 0.01).sum())
print("Per-dimension variance:")
for i, v in enumerate(sorted(dim_var, reverse=True)):
    print(f"  z{i:02d}: {v:.4f}")


# print a few raw mu values to see what's actually coming out
with torch.no_grad():
    batch = next(iter(loader))
    mu = model.encode(
        batch["x"], batch["strain"], batch["sex"],
        batch["study"], batch["tissue"], batch["euth"], batch["flight"]
    )
    print("mu shape:", mu.shape)
    print("mu sample (first 5 samples, first 8 dims):")
    print(mu[:5, :8])
    print("mu mean:", mu.mean().item())
    print("mu std:",  mu.std().item())
