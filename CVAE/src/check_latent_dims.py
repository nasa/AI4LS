import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SpaceflightDataset
from model import SpaceflightCVAE
from pathlib import Path

parser = argparse.ArgumentParser(description="Latent dimension variance diagnostic")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--output_dir",  type=str, required=True, help="Path to directory to store output")
parser.add_argument("--data",       type=str, required=True, help="Path to subset_final.h5")
parser.add_argument("--hidden_dims",  type=int, nargs="+",
                    default=[512, 256], help="dimensions of encoder layers")

args = parser.parse_args()

dataset = SpaceflightDataset(args.data)
output_dir = args.output_dir

def _arch_from_state_dict(ckpt):
    """
    Read true model architecture from state dict weights.
    More reliable than ckpt["args"] which may be stale or incorrect.
    """
    sd   = ckpt["model_state"]

    # which condition embeddings actually exist in the weights
    conditions = [
        c for c in ["tissue", "strain", "sex", "study", "euth"]
        if f"embedder.embeddings.{c}.weight" in sd
    ]
    if not conditions:
        conditions = ckpt.get("args", {}).get(
            "conditions", ["tissue","strain","sex","study","euth"])

    # latent dim from mu head output size
    latent_dim = sd["encoder.mu.weight"].shape[0]

    # embedding dims from weight shapes
    def emb_dim(name, default):
        key = f"embedder.embeddings.{name}.weight"
        return int(sd[key].shape[1]) if key in sd else default

    return dict(
        conditions=conditions,
        latent_dim=latent_dim,
        tissue_emb_dim=emb_dim("tissue", 32),
        strain_emb_dim=emb_dim("strain", 16),
        sex_emb_dim=emb_dim("sex",     4),
        study_emb_dim=emb_dim("study",  16),
        euth_emb_dim=emb_dim("euth",    8),
    )


ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
arch  = _arch_from_state_dict(ckpt)
model = SpaceflightCVAE(
    n_genes=dataset.n_genes,
    n_strains=dataset.n_strains,
    n_sexes=dataset.n_sexes,
    n_studies=dataset.n_studies,
    n_tissues=dataset.n_tissues,
    n_euths=dataset.n_euths,
    **arch,
    hidden_dims=args.hidden_dims,
    dropout=0.0,
    grl_alpha=0.0,
)


out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

model.load_state_dict(ckpt['model_state'])
model.to('cpu').eval()
print(f'Loaded {args.checkpoint} | conditions={arch["conditions"]} latent_dim={arch["latent_dim"]}')
loader = DataLoader(dataset, batch_size=64, shuffle=False)
all_mu = []
with torch.no_grad():
    for batch in loader:
        mu = model.encode(
            batch["x"], batch["strain"], batch["sex"],
            batch["study"], batch["tissue"], batch["euth"], batch["flight"]
        )
        all_mu.append(mu.numpy())

mu_all  = np.concatenate(all_mu)
dim_var = mu_all.var(axis=0)

with open(out_dir / "latent_dims.txt", "w") as f:
    f.write(f"Active dimensions (var > 0.1):    {(dim_var > 0.1).sum()}\n")
    f.write(f"Collapsed dimensions (var < 0.01): {(dim_var < 0.01).sum()}\n")
    f.write("Per-dimension variance:\n")
    for i, v in enumerate(sorted(dim_var, reverse=True)):
        f.write(f"  z{i:02d}: {v:.4f}\n")
print("  Saved: " + str(out_dir / "latent_dims.txt"))
f.close()

'''print(f"Active dimensions (var > 0.1):    {(dim_var > 0.1).sum()}")
print(f"Collapsed dimensions (var < 0.01): {(dim_var < 0.01).sum()}")
print("Per-dimension variance:")
for i, v in enumerate(sorted(dim_var, reverse=True)):
    print(f"  z{i:02d}: {v:.4f}")'''

with torch.no_grad():
    batch = next(iter(loader))
    mu = model.encode(
        batch["x"], batch["strain"], batch["sex"],
        batch["study"], batch["tissue"], batch["euth"], batch["flight"]
    )
    with open(out_dir / "latent_distribution.txt", "w") as f:
        f.write(f"mu shape: {mu.shape}\n")
        f.write("mu sample (first 5 samples, first 8 dims):\n")
        f.write(f" {mu[:5, :8]}\n")
        f.write("\n")
        f.write(f"mu mean: {mu.mean().item()}\n")
        f.write(f"mu std:  {mu.std().item()}\n")
        f.close()
    '''print("mu shape:", mu.shape)
    print("mu sample (first 5 samples, first 8 dims):")
    print(mu[:5, :8])
    print("mu mean:", mu.mean().item())
    print("mu std:",  mu.std().item())'''

