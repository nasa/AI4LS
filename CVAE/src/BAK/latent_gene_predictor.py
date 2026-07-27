"""
Latent Gene Predictor
=====================
Identifies genes that most strongly predict the flight-correlated
latent dimensions of the cVAE — i.e. the genes whose expression
drives the latent space organization separating flight from ground.

This is distinct from standard DE analysis:
  - DE measures observed expression differences between conditions
  - This measures which genes the MODEL uses to encode spaceflight
    in its internal representation

Three levels of analysis:
  1. Global     — all samples, find genes predicting flight dims in z
  2. Per-tissue — within each tissue, find tissue-specific latent predictors
  3. Comparison — which genes are shared vs tissue-specific predictors

Method:
  Step 1: Encode all samples → z (n_samples, latent_dim)
  Step 2: Find latent dims with strong Spearman correlation to flight label
  Step 3: For each flight-predictive dim, fit LassoCV(expression → z_dim)
  Step 4: Aggregate gene scores weighted by |flight correlation|
  Step 5: Report top genes with their scores and direction

Usage:
    # global analysis
    python latent_gene_predictor.py \\
        --checkpoint checkpoints/finetune/tissue/best_model.pt \\
        --data DATA/osdr_mouse.h5 \\
        --output_dir latent_gene_pred/global/

    # per-tissue analysis
    python latent_gene_predictor.py \\
        --checkpoint checkpoints/finetune/tissue/best_model.pt \\
        --data DATA/osdr_mouse.h5 \\
        --output_dir latent_gene_pred/tissue/ \\
        --by_tissue

    # specific tissue
    python latent_gene_predictor.py \\
        --checkpoint checkpoints/finetune/tissue/best_model.pt \\
        --data DATA/osdr_mouse.h5 \\
        --output_dir latent_gene_pred/liver/ \\
        --tissue Liver
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader

from dataset import SpaceflightDataset
from model import SpaceflightCVAE


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


def load_model(checkpoint_path, dataset, device="cpu"):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch  = _arch_from_state_dict(ckpt)
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
    model.to(device).eval()
    print(f"Loaded {checkpoint_path}")
    print(f"  conditions={arch['conditions']}  latent_dim={arch['latent_dim']}")
    return model, arch["latent_dim"]


# ---------------------------------------------------------------------------
# Encode all samples
# ---------------------------------------------------------------------------

def encode_dataset(model, dataset, device, batch_size=256):
    """Encode all samples → μ vectors (n_samples, latent_dim)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_mu = []
    with torch.no_grad():
        for batch in loader:
            mu = model.encode(
                batch["x"].to(device),
                batch["strain"].to(device),
                batch["sex"].to(device),
                batch["study"].to(device),
                batch["tissue"].to(device),
                batch["euth"].to(device),
                batch["flight"].to(device),
            )
            all_mu.append(mu.cpu().numpy())
    return np.concatenate(all_mu, axis=0)   # (n_samples, latent_dim)


# ---------------------------------------------------------------------------
# Step 2: Find flight-predictive latent dimensions
# ---------------------------------------------------------------------------

def find_flight_dims(z, flight, corr_threshold=0.20):
    """
    Find latent dimensions with |Spearman r| > corr_threshold to flight label.
    Returns dims sorted by |correlation|, with their correlation values.
    """
    n_dims = z.shape[1]
    corrs  = np.array([spearmanr(z[:, d], flight).correlation
                       for d in range(n_dims)])
    flight_mask = np.abs(corrs) > corr_threshold
    flight_dims = np.where(flight_mask)[0]
    flight_dims = flight_dims[np.argsort(np.abs(corrs[flight_dims]))[::-1]]

    print(f"\nFlight-predictive latent dimensions (|r| > {corr_threshold}):")
    for d in flight_dims:
        direction = "↑ in flight" if corrs[d] > 0 else "↓ in flight"
        print(f"  z{d:02d}: r={corrs[d]:+.3f}  {direction}")
    print(f"  Total: {len(flight_dims)} / {n_dims} dims")

    return flight_dims, corrs


# ---------------------------------------------------------------------------
# Step 3+4: Lasso regression + gene score aggregation
# ---------------------------------------------------------------------------

def compute_gene_scores(X, z, flight_dims, corrs, method="lasso",
                        min_samples=20):
    """
    For each flight-predictive latent dim, fit a regularized regression
    of expression → z_dim. Aggregate |coefficients| weighted by |flight r|.

    Args:
        X:           (n_samples, n_genes) expression matrix
        z:           (n_samples, latent_dim) latent vectors
        flight_dims: indices of flight-predictive dims
        corrs:       (latent_dim,) Spearman correlations to flight
        method:      'lasso' (sparse) or 'ridge' (dense)
        min_samples: minimum samples needed to fit

    Returns:
        gene_scores:    (n_genes,) aggregated importance scores
        gene_direction: (n_genes,) +1 = predictive of flight, -1 = ground
    """
    n_samples, n_genes = X.shape

    if n_samples < min_samples:
        print(f"  Skipping — too few samples ({n_samples} < {min_samples})")
        return None, None

    if len(flight_dims) == 0:
        print("  No flight-predictive dims found — try lowering --corr_threshold")
        return None, None

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    gene_scores    = np.zeros(n_genes)
    gene_direction = np.zeros(n_genes)

    for d in flight_dims:
        z_d    = z[:, d]
        weight = np.abs(corrs[d])   # weight by flight correlation strength

        if method == "lasso":
            reg = LassoCV(
                cv=5, max_iter=5000, n_alphas=30,
                random_state=42, n_jobs=-1
            )
        else:
            alphas = np.logspace(-3, 3, 30)
            reg = RidgeCV(alphas=alphas, cv=5)

        reg.fit(X_sc, z_d)
        coef = reg.coef_   # (n_genes,)

        # direction: positive coef means gene ↑ → z_d ↑
        # if z_d correlates positively with flight, gene ↑ → flight signal
        flight_sign = np.sign(corrs[d])
        gene_scores    += np.abs(coef) * weight
        gene_direction += np.sign(coef) * flight_sign * weight

    # normalize direction to [-1, +1]
    nonzero = gene_scores > 0
    gene_direction[nonzero] = gene_direction[nonzero] / gene_scores[nonzero]

    return gene_scores, gene_direction


# ---------------------------------------------------------------------------
# Step 5: Build results DataFrame
# ---------------------------------------------------------------------------

def build_results(gene_scores, gene_direction, dataset, n_top=200,
                  tissue_label=None):
    """Build ranked gene DataFrame from scores."""
    if gene_scores is None:
        return None

    top_idx = np.argsort(gene_scores)[::-1][:n_top]

    df = pd.DataFrame({
        "rank":          range(1, n_top + 1),
        "ensembl_id":    dataset.ensembl_ids[top_idx],
        "symbol":        dataset.gene_symbols[top_idx],
        "latent_score":  gene_scores[top_idx],
        "direction":     np.where(gene_direction[top_idx] > 0,
                                  "flight", "ground"),
        "direction_val": gene_direction[top_idx],
    })

    if tissue_label is not None:
        df.insert(0, "tissue", tissue_label)

    return df


# ---------------------------------------------------------------------------
# Cross-validation: how well do top genes predict flight?
# ---------------------------------------------------------------------------

def validate_top_genes(X, flight, gene_scores, n_top_list=(20, 50, 100, 200)):
    """
    Quick validation: can the top N latent-predictive genes classify flight
    in a held-out CV? Compares against random gene sets.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    scaler = StandardScaler()
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf    = LogisticRegression(max_iter=500, C=1.0, random_state=42)

    print("\nValidation — flight classification AUROC using top N latent genes:")
    sorted_idx = np.argsort(gene_scores)[::-1]

    for n_top in n_top_list:
        if n_top > X.shape[1]:
            continue
        top_idx  = sorted_idx[:n_top]
        X_top    = scaler.fit_transform(X[:, top_idx])
        probs    = cross_val_predict(clf, X_top, flight,
                                     cv=cv, method="predict_proba")[:, 1]
        auroc    = roc_auc_score(flight, probs)

        # random baseline
        rand_idx = np.random.choice(X.shape[1], n_top, replace=False)
        X_rand   = scaler.fit_transform(X[:, rand_idx])
        p_rand   = cross_val_predict(clf, X_rand, flight,
                                     cv=cv, method="predict_proba")[:, 1]
        auroc_rand = roc_auc_score(flight, p_rand)

        print(f"  Top {n_top:4d} genes: AUROC={auroc:.3f}  "
              f"(random baseline={auroc_rand:.3f})")


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_analysis(dataset, model, latent_dim, device, args, out_dir,
                 sample_mask=None, tissue_label=None):
    """
    Run the full latent gene predictor pipeline on a subset of samples.

    Args:
        sample_mask:  boolean array selecting samples (None = all)
        tissue_label: label for output files (None = 'global')
    """
    label = tissue_label or "global"
    print(f"\n{'='*60}")
    print(f"Analysis: {label}")
    print(f"{'='*60}")

    # select samples
    if sample_mask is not None:
        indices = np.where(sample_mask)[0]
    else:
        indices = np.arange(dataset.n_samples)

    n_flight = int((dataset.flight[indices] == 1).sum())
    n_ground = int((dataset.flight[indices] == 0).sum())
    print(f"Samples: {len(indices)} ({n_flight} flight / {n_ground} ground)")

    if n_flight < 5 or n_ground < 5:
        print("Skipping — insufficient flight or ground samples.")
        return None

    # encode to latent space
    from torch.utils.data import Subset
    sub_dataset = Subset(dataset, indices)
    loader      = DataLoader(sub_dataset, batch_size=256, shuffle=False)

    all_mu = []
    with torch.no_grad():
        for batch in loader:
            mu = model.encode(
                batch["x"].to(device),
                batch["strain"].to(device),
                batch["sex"].to(device),
                batch["study"].to(device),
                batch["tissue"].to(device),
                batch["euth"].to(device),
                batch["flight"].to(device),
            )
            all_mu.append(mu.cpu().numpy())
    z      = np.concatenate(all_mu, axis=0)
    X      = dataset.x[indices]         # (n, n_genes) log1p CPM
    flight = dataset.flight[indices]    # (n,)

    # find flight-predictive dims
    flight_dims, corrs = find_flight_dims(z, flight, args.corr_threshold)

    if len(flight_dims) == 0:
        print(f"No flight-predictive dims found for {label}. "
              f"Try --corr_threshold lower than {args.corr_threshold}.")
        return None

    # compute gene scores
    print(f"\nFitting {args.method} regression for {len(flight_dims)} dims...")
    gene_scores, gene_direction = compute_gene_scores(
        X, z, flight_dims, corrs, method=args.method
    )

    if gene_scores is None:
        return None

    # build results
    df = build_results(gene_scores, gene_direction, dataset,
                       n_top=args.n_top, tissue_label=tissue_label)

    # print top 20
    print(f"\nTop 20 latent-predictive genes ({label}):")
    display_cols = ["rank","symbol","latent_score","direction"]
    if tissue_label:
        display_cols = ["tissue"] + display_cols
    print(df[display_cols].head(20).to_string(index=False))

    # validate
    if args.validate:
        validate_top_genes(X, flight, gene_scores)

    # save
    fname = label.lower().replace(" ", "_") + "_latent_genes.csv"
    df.to_csv(out_dir / fname, index=False)
    print(f"\nSaved: {out_dir / fname}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find genes predictive of flight-correlated latent dims"
    )
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--data",            type=str, required=True)
    parser.add_argument("--output_dir",      type=str, default="latent_gene_pred")
    parser.add_argument("--by_tissue",       action="store_true",
                        help="Run per-tissue analysis")
    parser.add_argument("--tissue",          type=str, default=None,
                        help="Run for a specific tissue only")
    parser.add_argument("--corr_threshold",  type=float, default=0.20,
                        help="Min |Spearman r| to flight for a dim to be "
                             "included (default 0.20)")
    parser.add_argument("--method",          type=str, default="lasso",
                        choices=["lasso", "ridge"],
                        help="Regression method (default: lasso)")
    parser.add_argument("--n_top",           type=int, default=200,
                        help="Number of top genes to report (default 200)")
    parser.add_argument("--min_samples",     type=int, default=10,
                        help="Min flight+ground samples per tissue (default 10)")
    parser.add_argument("--validate",        action="store_true",
                        help="Run CV validation of top gene sets")
    parser.add_argument("--device",          type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load
    print("Loading dataset...")
    dataset = SpaceflightDataset(args.data)

    print("\nLoading model...")
    model, latent_dim = load_model(args.checkpoint, dataset, args.device)

    all_dfs = []

    if args.tissue:
        # single tissue
        if args.tissue not in dataset.tissue_enc.classes_:
            raise ValueError(f"Unknown tissue: {args.tissue}. "
                             f"Available: {list(dataset.tissue_enc.classes_)}")
        t_idx = dataset.tissue_enc.transform([args.tissue])[0]
        mask  = dataset.tissue_ids == t_idx
        df    = run_analysis(dataset, model, latent_dim, args.device,
                             args, out_dir, sample_mask=mask,
                             tissue_label=args.tissue)
        if df is not None:
            all_dfs.append(df)

    elif args.by_tissue:
        # all tissues
        for tissue_name in dataset.tissue_enc.classes_:
            t_idx = dataset.tissue_enc.transform([tissue_name])[0]
            mask  = dataset.tissue_ids == t_idx
            n_f   = int((dataset.flight[mask] == 1).sum())
            n_g   = int((dataset.flight[mask] == 0).sum())
            if n_f < args.min_samples or n_g < args.min_samples:
                print(f"\nSkipping {tissue_name} "
                      f"({n_f}f/{n_g}g < {args.min_samples})")
                continue
            df = run_analysis(dataset, model, latent_dim, args.device,
                              args, out_dir, sample_mask=mask,
                              tissue_label=tissue_name)
            if df is not None:
                all_dfs.append(df)

        # save combined
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(out_dir / "all_tissues_latent_genes.csv", index=False)
            print(f"\nSaved combined: {out_dir / 'all_tissues_latent_genes.csv'}")

    else:
        # global analysis
        df = run_analysis(dataset, model, latent_dim, args.device,
                          args, out_dir, sample_mask=None,
                          tissue_label=None)
        if df is not None:
            all_dfs.append(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
