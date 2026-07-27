import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset import SpaceflightDataset
from model import SpaceflightCVAE
from pathlib import Path

parser = argparse.ArgumentParser(description="Within-study flight AUROC in latent space")
parser.add_argument("--checkpoint",  type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--output_dir",  type=str, required=True, help="Path to directory to store output")
parser.add_argument("--data",        type=str, required=True, help="Path to subset_final.h5")
parser.add_argument("--conditions",  type=str, nargs="+", default=None,
                    choices=["tissue","strain","sex","study","euth"],
                    help="Override conditions at inference (default: use checkpoint conditions)")
parser.add_argument("--hidden_dims",  type=int, nargs="+",
                    default=[512, 256], help="dimensions of encoder layers")

args = parser.parse_args()

dataset = SpaceflightDataset(args.data)

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

# --conditions flag overrides checkpoint conditions
conditions = args.conditions if args.conditions else arch["conditions"]

model = SpaceflightCVAE(
    n_genes=dataset.n_genes,
    n_strains=dataset.n_strains,
    n_sexes=dataset.n_sexes,
    n_studies=dataset.n_studies,
    n_tissues=dataset.n_tissues,
    n_euths=dataset.n_euths,
    conditions=conditions,
    latent_dim=arch["latent_dim"],
    tissue_emb_dim=arch["tissue_emb_dim"],
    strain_emb_dim=arch["strain_emb_dim"],
    sex_emb_dim=arch["sex_emb_dim"],
    study_emb_dim=arch["study_emb_dim"],
    euth_emb_dim=arch["euth_emb_dim"],
    hidden_dims=args.hidden_dims,
    dropout=0.0,
    grl_alpha=0.0,
)
missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
if unexpected:
    print(f"  Skipped {len(unexpected)} keys not used by conditions={conditions}")
model.to('cpu').eval()
print(f'Loaded {args.checkpoint} | conditions={conditions} latent_dim={arch["latent_dim"]}')
loader = DataLoader(dataset, batch_size=64, shuffle=False)
zs, flights, studies, tissues = [], [], [], []
with torch.no_grad():
    for batch in loader:
        mu = model.encode(
            batch["x"], batch["strain"], batch["sex"],
            batch["study"], batch["tissue"], batch["euth"], batch["flight"]
        )
        zs.append(mu.numpy())
        flights.append(batch["flight"].numpy())
        studies.append(batch["study"].numpy())
        tissues.append(batch["tissue"].numpy())

zs      = np.concatenate(zs)
flights = np.concatenate(flights)
studies = np.concatenate(studies)
tissues = np.concatenate(tissues)

scaler = StandardScaler()
zs_sc  = scaler.fit_transform(zs)

rows = []
for s in np.unique(studies):
    mask = studies == s
    if len(np.unique(flights[mask])) < 2:
        continue
    clf = LogisticRegression(max_iter=200, C=1.0)
    clf.fit(zs_sc[mask], flights[mask])
    probs = clf.predict_proba(zs_sc[mask])[:, 1]
    auroc = roc_auc_score(flights[mask], probs)
    tissue_name = dataset.tissue_enc.classes_[
        np.bincount(tissues[mask]).argmax()
    ]
    rows.append({
        "study":    dataset.study_enc.classes_[s],
        "tissue":   tissue_name,
        "n_flight": int((flights[mask] == 1).sum()),
        "n_ground": int((flights[mask] == 0).sum()),
        "auroc":    float(auroc),
    })

df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir /  "auroc_data.txt", "w") as f:
    f.write(f"Studies with both conditions: {len(df)}\n")
    f.write(f"Mean within-study AUROC:      {df['auroc'].mean():.3f}\n")
    f.write(f"Median within-study AUROC:    {df['auroc'].median():.3f}\n")
    f.write(f"Studies AUROC > 0.8:          {(df['auroc'] > 0.8).sum()}\n")
    f.write(f"Studies AUROC < 0.5:          {(df['auroc'] < 0.5).sum()}\n")
    f.write(df.to_string(index=False))

f.close()
