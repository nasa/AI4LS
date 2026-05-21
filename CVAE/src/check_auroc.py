import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset import SpaceflightDataset
from model import SpaceflightCVAE

# load
dataset = SpaceflightDataset("/home/jcasalet/nobackup/CVAE/DATA/subset_final.h6")
ckpt    = torch.load("/home/jcasalet/nobackup/CVAE/checkpoints/v9/best_model.pt", map_location="cpu",
                     weights_only=False)
model   = SpaceflightCVAE(
    n_genes=dataset.n_genes, n_strains=dataset.n_strains,
    n_sexes=dataset.n_sexes, n_studies=dataset.n_studies,
    n_tissues=dataset.n_tissues, n_euths=dataset.n_euths,
    latent_dim=64, hidden_dims=[256, 128], dropout=0.0, grl_alpha=0.0
)
model.load_state_dict(ckpt["model_state"])
model.eval()

# extract latents
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

# within-study AUROC
scaler = StandardScaler()
zs_sc  = scaler.fit_transform(zs)

rows = []
for s in np.unique(studies):
    mask = studies == s
    if len(np.unique(flights[mask])) < 2:
        continue   # skip single-condition studies
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

import pandas as pd
df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
print(f"Studies with both conditions: {len(df)}")
print(f"Mean within-study AUROC:      {df['auroc'].mean():.3f}")
print(f"Median within-study AUROC:    {df['auroc'].median():.3f}")
print(f"Studies AUROC > 0.8:          {(df['auroc'] > 0.8).sum()}")
print(f"Studies AUROC < 0.5:          {(df['auroc'] < 0.5).sum()}")
print()
print(df.to_string(index=False))
