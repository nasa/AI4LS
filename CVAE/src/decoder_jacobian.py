"""
Decoder Jacobian Gene Importance
==================================
Computes the Jacobian of the encoder with respect to input expression,
projected onto the flight-predictive latent dimensions.

This answers: "which genes, when their expression changes, most strongly
move a sample's position in the flight-relevant region of latent space?"

Unlike standard gene attribution (gradient of classifier output w.r.t. input),
the Jacobian approach:
  1. Identifies which latent dimensions encode flight (via correlation)
  2. Computes how much each gene influences those specific dimensions
  3. Is more interpretable — each gene's score is tied to specific latent axes

Unlike DE analysis:
  - Measures model sensitivity, not observed expression differences
  - Captures non-linear interactions through the encoder
  - Is tissue-specific when computed on tissue-stratified samples

Method:
  J = ∂μ / ∂x   (Jacobian of encoder mean w.r.t. input expression)
  Shape: (n_samples, latent_dim, n_genes)

  For each sample, J[s, d, g] = how much gene g influences latent dim d.

  Gene importance = mean over samples of:
    sum over flight dims d of: |J[s, d, g]| * |flight_corr[d]|

Usage:
    # global analysis
    python decoder_jacobian.py \\
        --checkpoint checkpoints/finetune/tissue/best_model.pt \\
        --data DATA/osdr_mouse.h5 \\
        --output_dir jacobian_results/global/

    # specific tissue
    python decoder_jacobian.py \\
        --checkpoint checkpoints/finetune/tissue/best_model.pt \\
        --data DATA/osdr_mouse.h5 \\
        --output_dir jacobian_results/bone_marrow/ \\
        --tissue "Bone Marrow"

    # all tissues
    python decoder_jacobian.py \\
        --checkpoint checkpoints/finetune/tissue/best_model.pt \\
        --data DATA/osdr_mouse.h5 \\
        --output_dir jacobian_results/by_tissue/ \\
        --by_tissue
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

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
# Encode all samples (no grad)
# ---------------------------------------------------------------------------

def encode_all(model, dataset, indices, device, batch_size=256):
    """Encode samples at given indices → μ (n, latent_dim)."""
    sub     = Subset(dataset, indices)
    loader  = DataLoader(sub, batch_size=batch_size, shuffle=False)
    all_mu  = []
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
    return np.concatenate(all_mu, axis=0)


# ---------------------------------------------------------------------------
# Find flight-correlated latent dimensions
# ---------------------------------------------------------------------------

def find_flight_dims(z, flight, corr_threshold=0.20):
    """Return dims with |Spearman r| > threshold, sorted by |r|."""
    corrs = np.array([spearmanr(z[:, d], flight).correlation
                      for d in range(z.shape[1])])
    mask  = np.abs(corrs) > corr_threshold
    dims  = np.where(mask)[0]
    dims  = dims[np.argsort(np.abs(corrs[dims]))[::-1]]

    print(f"\nFlight-predictive dims (|r| > {corr_threshold}):")
    for d in dims:
        print(f"  z{d:02d}: r={corrs[d]:+.4f}  "
              f"({'↑ flight' if corrs[d] > 0 else '↓ flight'})")
    print(f"  Total: {len(dims)} / {z.shape[1]} dims")
    return dims, corrs


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian_batch(model, batch, flight_dims, corrs, device):
    """
    Compute Jacobian ∂μ/∂x for one batch, projected onto flight dims.

    For each sample s and gene g:
      score[s, g] = sum_d |J[s, d, g]| * |corrs[d]|

    where the sum is only over flight-predictive dims d.

    Returns:
        scores:    (batch_size, n_genes) weighted Jacobian scores
        direction: (batch_size, n_genes) sign of flight effect
    """
    x      = batch["x"].to(device).float()
    strain = batch["strain"].to(device)
    sex    = batch["sex"].to(device)
    study  = batch["study"].to(device)
    tissue = batch["tissue"].to(device)
    euth   = batch["euth"].to(device)
    flight = batch["flight"].to(device)

    x.requires_grad_(True)
    batch_size = x.shape[0]
    n_genes    = x.shape[1]

    # get condition embedding (no grad needed for cond)
    with torch.no_grad():
        cond = model.embedder(strain, sex, study, tissue, euth)

    # forward through encoder to get μ
    # (need grad through x, so can't use torch.no_grad here)
    h  = model.encoder.net(torch.cat([x, cond.detach()], dim=-1))
    mu = model.encoder.mu(h)   # (batch_size, latent_dim)

    scores    = np.zeros((batch_size, n_genes), dtype=np.float32)
    direction = np.zeros((batch_size, n_genes), dtype=np.float32)

    for d in flight_dims:
        weight = float(np.abs(corrs[d]))
        flight_sign = float(np.sign(corrs[d]))

        # gradient of μ_d w.r.t. x for all samples simultaneously
        # sum over batch to get one backward pass, then divide
        grad_outputs = torch.zeros_like(mu)
        grad_outputs[:, d] = 1.0

        if x.grad is not None:
            x.grad.zero_()

        mu.backward(gradient=grad_outputs, retain_graph=(d != flight_dims[-1]))

        grad = x.grad.detach().cpu().numpy()   # (batch_size, n_genes)

        scores    += np.abs(grad) * weight
        direction += np.sign(grad) * flight_sign * weight

        # zero grad for next dim
        x.grad.zero_()

    # re-run forward to reset computation graph for next batch
    x.requires_grad_(False)

    return scores, direction


def compute_jacobian(model, dataset, indices, flight_dims, corrs,
                     device, batch_size=32):
    """
    Compute Jacobian scores for all samples at given indices.
    Uses smaller batch_size than encoding because we need grad computation.

    Returns:
        mean_scores:    (n_genes,) mean absolute Jacobian score per gene
        mean_direction: (n_genes,) mean direction of flight effect
    """
    sub    = Subset(dataset, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False)

    all_scores    = []
    all_directions = []

    print(f"\nComputing Jacobian for {len(indices)} samples "
          f"({len(flight_dims)} flight dims)...")
    print("  (using batch_size=32 — this may take a few minutes)")

    for i, batch in enumerate(loader):
        scores, direction = compute_jacobian_batch(
            model, batch, flight_dims, corrs, device
        )
        all_scores.append(scores)
        all_directions.append(direction)

        if (i + 1) % 10 == 0:
            done = min((i + 1) * batch_size, len(indices))
            print(f"  {done}/{len(indices)} samples processed")

    all_scores    = np.concatenate(all_scores,    axis=0)   # (n, n_genes)
    all_directions = np.concatenate(all_directions, axis=0)  # (n, n_genes)

    mean_scores    = all_scores.mean(axis=0)      # (n_genes,)

    # direction: weighted average sign
    nonzero        = mean_scores > 0
    mean_direction = np.zeros_like(mean_scores)
    mean_direction[nonzero] = (
        all_directions[:, nonzero].mean(axis=0) /
        (mean_scores[nonzero] + 1e-8)
    )
    mean_direction = np.clip(mean_direction, -1, 1)

    return mean_scores, mean_direction


# ---------------------------------------------------------------------------
# Build and save results
# ---------------------------------------------------------------------------

def build_results(gene_scores, gene_direction, dataset,
                  n_top=200, tissue_label=None):
    if gene_scores is None:
        return None

    top_idx = np.argsort(gene_scores)[::-1][:n_top]
    df = pd.DataFrame({
        "rank":          range(1, n_top + 1),
        "ensembl_id":    dataset.ensembl_ids[top_idx],
        "symbol":        dataset.gene_symbols[top_idx],
        "jacobian_score": gene_scores[top_idx],
        "direction":     np.where(gene_direction[top_idx] > 0,
                                  "flight", "ground"),
        "direction_val": gene_direction[top_idx],
    })
    if tissue_label is not None:
        df.insert(0, "tissue", tissue_label)
    return df


# ---------------------------------------------------------------------------
# Per-analysis runner
# ---------------------------------------------------------------------------

def run_analysis(dataset, model, latent_dim, device, args, out_dir,
                 sample_mask=None, tissue_label=None):
    label = tissue_label or "global"
    print(f"\n{'='*60}")
    print(f"Jacobian Analysis: {label}")
    print(f"{'='*60}")

    if sample_mask is not None:
        indices = np.where(sample_mask)[0]
    else:
        indices = np.arange(dataset.n_samples)

    n_flight = int((dataset.flight[indices] == 1).sum())
    n_ground = int((dataset.flight[indices] == 0).sum())
    print(f"Samples: {len(indices)} ({n_flight} flight / {n_ground} ground)")

    if n_flight < 5 or n_ground < 5:
        print("Skipping — insufficient samples.")
        return None

    # encode to find flight-predictive dims
    print("\nEncoding samples...")
    z      = encode_all(model, dataset, indices, device)
    flight = dataset.flight[indices]

    flight_dims, corrs = find_flight_dims(z, flight, args.corr_threshold)

    if len(flight_dims) == 0:
        print(f"No flight-predictive dims for {label}. "
              f"Try --corr_threshold lower than {args.corr_threshold}.")
        return None

    # compute Jacobian
    gene_scores, gene_direction = compute_jacobian(
        model, dataset, indices, flight_dims, corrs,
        device, batch_size=args.batch_size
    )

    # build results
    df = build_results(gene_scores, gene_direction, dataset,
                       n_top=args.n_top, tissue_label=tissue_label)

    print(f"\nTop 20 Jacobian genes ({label}):")
    display_cols = ["rank", "symbol", "jacobian_score", "direction"]
    if tissue_label:
        display_cols = ["tissue"] + display_cols
    print(df[display_cols].head(20).to_string(index=False))

    fname = label.lower().replace(" ", "_") + "_jacobian_genes.csv"
    df.to_csv(out_dir / fname, index=False)
    print(f"Saved: {out_dir / fname}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decoder Jacobian gene importance from latent space"
    )
    parser.add_argument("--checkpoint",     type=str, required=True)
    parser.add_argument("--data",           type=str, required=True)
    parser.add_argument("--output_dir",     type=str,
                        default="jacobian_results")
    parser.add_argument("--tissue",         type=str, default=None,
                        help="Specific tissue to analyze")
    parser.add_argument("--by_tissue",      action="store_true",
                        help="Run for all tissues")
    parser.add_argument("--corr_threshold", type=float, default=0.20,
                        help="Min |Spearman r| to flight (default 0.20)")
    parser.add_argument("--n_top",          type=int, default=200,
                        help="Top N genes to report (default 200)")
    parser.add_argument("--min_samples",    type=int, default=10,
                        help="Min flight+ground per tissue (default 10)")
    parser.add_argument("--batch_size",     type=int, default=32,
                        help="Batch size for Jacobian computation (default 32)")
    parser.add_argument("--device",         type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = SpaceflightDataset(args.data)

    print("\nLoading model...")
    model, latent_dim = load_model(args.checkpoint, dataset, args.device)

    all_dfs = []

    if args.tissue:
        if args.tissue not in dataset.tissue_enc.classes_:
            raise ValueError(
                f"Unknown tissue '{args.tissue}'. "
                f"Available: {list(dataset.tissue_enc.classes_)}"
            )
        t_idx = dataset.tissue_enc.transform([args.tissue])[0]
        mask  = dataset.tissue_ids == t_idx
        df    = run_analysis(dataset, model, latent_dim, args.device,
                             args, out_dir,
                             sample_mask=mask, tissue_label=args.tissue)
        if df is not None:
            all_dfs.append(df)

    elif args.by_tissue:
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
                              args, out_dir,
                              sample_mask=mask, tissue_label=tissue_name)
            if df is not None:
                all_dfs.append(df)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(out_dir / "all_tissues_jacobian_genes.csv",
                            index=False)
            print(f"\nSaved: {out_dir / 'all_tissues_jacobian_genes.csv'}")

    else:
        df = run_analysis(dataset, model, latent_dim, args.device,
                          args, out_dir,
                          sample_mask=None, tissue_label=None)
        if df is not None:
            all_dfs.append(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
