"""
Within-Tissue Study Predictability
====================================
Tests whether study ID can be predicted from z WITHIN each tissue.

High within-tissue study predictability = batch effects in z
Low within-tissue study predictability = tissue encoding driving
  the overall study predictability (good)

Usage:
    python check_within_tissue_study_pred.py \\
        --checkpoint checkpoints/finetune/tissue_grl/best_model.pt \\
        --data DATA/osdr_mouse.h5
"""

import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from dataset import SpaceflightDataset
from model import SpaceflightCVAE


def _arch_from_state_dict(ckpt):
    sd = ckpt["model_state"]
    conditions = [
        c for c in ["tissue", "strain", "sex", "study", "euth"]
        if f"embedder.embeddings.{c}.weight" in sd
    ]
    if not conditions:
        conditions = ckpt.get("args", {}).get(
            "conditions", ["tissue","strain","sex","study","euth"])
    latent_dim = sd["encoder.mu.weight"].shape[0]
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


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--data",       type=str, required=True)
parser.add_argument("--min_studies", type=int, default=2,
                    help="Min number of studies in tissue to include (default 2)")
args = parser.parse_args()

# load
dataset = SpaceflightDataset(args.data)
ckpt    = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
arch    = _arch_from_state_dict(ckpt)

model = SpaceflightCVAE(
    n_genes=dataset.n_genes,
    n_strains=dataset.n_strains,
    n_sexes=dataset.n_sexes,
    n_studies=dataset.n_studies,
    n_tissues=dataset.n_tissues,
    n_euths=dataset.n_euths,
    **arch,
    hidden_dims=[256, 128],
    dropout=0.0,
    grl_alpha=0.0,
)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded | conditions={arch['conditions']} latent_dim={arch['latent_dim']}")

# encode all samples
loader = DataLoader(dataset, batch_size=64, shuffle=False)
zs, tissues, studies = [], [], []
with torch.no_grad():
    for batch in loader:
        mu = model.encode(
            batch["x"], batch["strain"], batch["sex"],
            batch["study"], batch["tissue"], batch["euth"], batch["flight"]
        )
        zs.append(mu.numpy())
        tissues.append(batch["tissue"].numpy())
        studies.append(batch["study"].numpy())

zs      = np.concatenate(zs)
tissues = np.concatenate(tissues)
studies = np.concatenate(studies)

# overall study predictability
scaler  = StandardScaler()
zs_sc   = scaler.fit_transform(zs)
clf     = LogisticRegression(max_iter=1000, C=1.0, solver="saga")
clf.fit(zs_sc, studies)
overall_acc = accuracy_score(studies, clf.predict(zs_sc))
print(f"\nOverall study predictability from z: {overall_acc:.3f}")

# within-tissue study predictability
print(f"\nWithin-tissue study predictability (tissues with >= {args.min_studies} studies):")
print(f"{'tissue':<20} {'n_samples':>9} {'n_studies':>9} {'study_acc':>9}")
print("-" * 52)

rows = []
for tissue_idx in np.unique(tissues):
    tissue_name = dataset.tissue_enc.classes_[tissue_idx]
    mask        = tissues == tissue_idx
    study_t     = studies[mask]
    n_studies   = len(np.unique(study_t))

    if n_studies < args.min_studies:
        continue

    z_t  = zs[mask]
    z_sc = StandardScaler().fit_transform(z_t)

    clf_t = LogisticRegression(max_iter=1000, C=1.0, solver="saga")
    clf_t.fit(z_sc, study_t)
    acc = accuracy_score(study_t, clf_t.predict(z_sc))

    rows.append({
        "tissue":    tissue_name,
        "n_samples": int(mask.sum()),
        "n_studies": n_studies,
        "study_acc": float(acc),
    })
    print(f"{tissue_name:<20} {mask.sum():>9,} {n_studies:>9} {acc:>9.3f}")

df = pd.DataFrame(rows).sort_values("study_acc", ascending=False)
print(f"\nMean within-tissue study accuracy:   {df['study_acc'].mean():.3f}")
print(f"Median within-tissue study accuracy: {df['study_acc'].median():.3f}")
print(f"\nHigh batch effect (acc > 0.7): {(df['study_acc'] > 0.7).sum()} tissues")
print(f"Low batch effect  (acc < 0.4): {(df['study_acc'] < 0.4).sum()} tissues")
