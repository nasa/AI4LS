"""
Latent Space Visualization for Spaceflight cVAE
=================================================
Four visualization types:

  1. LATENT DIMENSION HEATMAP
     Correlates each of the 32 latent dimensions with metadata
     (tissue, strain, sex, flight, euthanasia) to show what each
     dimension encodes.

  2. PCA OF LATENT SPACE
     Projects z into PCA space, shows variance explained per component,
     and colors the first two PCs by each metadata variable.

  3. PER-TISSUE LATENT DISTRIBUTIONS
     Violin plots of each latent dimension grouped by tissue,
     showing how tissues separate in latent space.

  4. FLIGHT VS GROUND CENTROID DISTANCE PER TISSUE
     For each tissue, computes the Euclidean distance between the
     mean flight latent vector and mean ground latent vector.
     Shows which tissues have the strongest spaceflight response.

Usage:
    python visualize_latent.py \\
        --checkpoint checkpoints_v6/best_model.pt \\
        --data subset_final.h5 \\
        --output_dir latent_viz/
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from dataset import SpaceflightDataset
from model import SpaceflightCVAE


# ---------------------------------------------------------------------------
# Model loading and latent extraction (same pattern as inference.py)
# ---------------------------------------------------------------------------


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

    # latent dim from μ head output size
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
        hidden_dims=args.hidden_dims,
        dropout=0.0,
        grl_alpha=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded {checkpoint_path} | epoch={ckpt['epoch']} "
          f"val_loss={ckpt['val_loss']:.4f} | "
          f"conditions={arch['conditions']} latent_dim={arch['latent_dim']}")
    return model


def extract_latents(model, dataset, device="cpu"):
    """
    Encode all samples, return z matrix and metadata arrays.

    Returns:
        z:       (N, latent_dim) numpy array of latent means
        meta:    dict of metadata arrays (flight, tissue, strain, sex, euth, study)
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    zs     = []
    meta   = {k: [] for k in ["flight", "tissue", "strain", "sex", "euth", "study"]}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x      = batch["x"].to(device)
            strain = batch["strain"].to(device)
            sex    = batch["sex"].to(device)
            study  = batch["study"].to(device)
            tissue = batch["tissue"].to(device)
            euth   = batch["euth"].to(device)
            flight = batch["flight"].to(device)
            mu     = model.encode(x, strain, sex, study, tissue, euth, flight)
            zs.append(mu.cpu().numpy())
            meta["flight"].append(flight.cpu().numpy())
            meta["tissue"].append(tissue.cpu().numpy())
            meta["strain"].append(strain.cpu().numpy())
            meta["sex"].append(sex.cpu().numpy())
            meta["euth"].append(euth.cpu().numpy())
            meta["study"].append(study.cpu().numpy())
    z    = np.concatenate(zs)
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    print("Extracted latents: " + str(z.shape))
    return z, meta


# ---------------------------------------------------------------------------
# 1. Latent dimension heatmap
# ---------------------------------------------------------------------------

def plot_dimension_heatmap(z, meta, dataset, out_dir):
    """
    Compute Spearman correlation between each latent dimension and
    each metadata variable. Plot as a heatmap (dims x variables).

    High correlation = that dimension encodes that variable.
    """
    print("Computing latent dimension correlations...")

    variables = {
        "Flight":     meta["flight"],
        "Tissue":     meta["tissue"],
        "Strain":     meta["strain"],
        "Sex":        meta["sex"],
        "Euthanasia": meta["euth"],
        "Study":      meta["study"],
    }

    n_dims = z.shape[1]
    n_vars = len(variables)
    corr_matrix = np.zeros((n_dims, n_vars))

    for j, (var_name, var_vals) in enumerate(variables.items()):
        for i in range(n_dims):
            r, _ = spearmanr(z[:, i], var_vals)
            corr_matrix[i, j] = r

    fig, ax = plt.subplots(figsize=(max(8, n_vars * 1.2), max(10, n_dims * 0.35)))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman r")

    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(list(variables.keys()), fontsize=10)
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels(["z" + str(i) for i in range(n_dims)], fontsize=7)
    ax.set_xlabel("Metadata variable")
    ax.set_ylabel("Latent dimension")
    ax.set_title("Latent Dimension Correlations with Metadata\n(Spearman r)")

    plt.tight_layout()
    path = out_dir / "latent_dimension_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + str(path))

    # also save as CSV
    df = pd.DataFrame(corr_matrix,
                      index=["z" + str(i) for i in range(n_dims)],
                      columns=list(variables.keys()))
    df.to_csv(out_dir / "latent_dimension_correlations.csv")
    print("  Saved: " + str(out_dir / "latent_dimension_correlations.csv"))

    # print top dimensions per variable
    print("\nTop 3 latent dimensions per metadata variable:")
    for j, var_name in enumerate(variables.keys()):
        top = np.argsort(np.abs(corr_matrix[:, j]))[::-1][:3]
        vals = [f"z{d}(r={corr_matrix[d,j]:.2f})" for d in top]
        print("  " + var_name + ": " + ", ".join(vals))


# ---------------------------------------------------------------------------
# 2. PCA of latent space
# ---------------------------------------------------------------------------

def plot_pca(z, meta, dataset, out_dir):
    """
    PCA of the latent space.
    - Scree plot: variance explained per component
    - Scatter plots of PC1 vs PC2 colored by each metadata variable
    """
    print("Computing PCA of latent space...")
    pca       = PCA(n_components=min(z.shape[1], 10))
    z_pca     = pca.fit_transform(z)
    var_exp   = pca.explained_variance_ratio_ * 100

    # --- scree plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(var_exp) + 1), var_exp, color="steelblue", edgecolor="white")
    ax.plot(range(1, len(var_exp) + 1), np.cumsum(var_exp),
            color="red", marker="o", markersize=4, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Scree Plot — Latent Space")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + str(out_dir / "pca_scree.png"))

    # --- PC1 vs PC2 colored by metadata ---
    plot_vars = [
        ("flight",  None,                "Spaceflight",      True),
        ("tissue",  dataset.tissue_enc,  "Tissue Type",      True),
        ("strain",  dataset.strain_enc,  "Strain",           True),
        ("sex",     dataset.sex_enc,     "Sex",              True),
        ("euth",    dataset.euth_enc,    "Euthanasia",       True),
    ]

    for key, enc, label, categorical in plot_vars:
        values   = meta[key]
        n_unique = len(np.unique(values))
        fig, ax  = plt.subplots(figsize=(8, 6))

        if categorical:
            unique_vals = np.unique(values)
            if n_unique <= 20:
                palette = [cm.get_cmap("tab20")(i / 20) for i in range(n_unique)]
            else:
                palette = [cm.get_cmap("hsv")(i / n_unique) for i in range(n_unique)]
            val_to_color = {v: palette[i] for i, v in enumerate(unique_vals)}
            colors = [val_to_color[v] for v in values]
            ax.scatter(z_pca[:, 0], z_pca[:, 1], c=colors, s=10, alpha=0.7)
            if n_unique <= 20:
                from matplotlib.patches import Patch
                labels = enc.classes_ if enc is not None else [str(v) for v in unique_vals]
                handles = [Patch(color=palette[i], label=labels[i])
                           for i in range(n_unique)]
                ax.legend(handles=handles, fontsize=7, title=label,
                          title_fontsize=8, loc="best", framealpha=0.6)
        else:
            sc = ax.scatter(z_pca[:, 0], z_pca[:, 1],
                            c=values, cmap="viridis", s=10, alpha=0.7)
            plt.colorbar(sc, ax=ax, label=label)

        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")
        ax.set_title("Latent Space PCA - " + label)
        plt.tight_layout()
        path = out_dir / ("pca_" + key + ".png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: " + str(path))

    # save PCA coordinates
    pca_df = pd.DataFrame(
        z_pca,
        columns=["PC" + str(i+1) for i in range(z_pca.shape[1])]
    )
    for key in ["flight", "tissue", "strain", "sex", "euth"]:
        enc = getattr(dataset, key + "_enc", None)
        if enc is not None:
            pca_df[key] = enc.inverse_transform(meta[key])
        else:
            pca_df[key] = meta[key]
    pca_df.to_csv(out_dir / "pca_coordinates.csv", index=False)
    print("  Saved: " + str(out_dir / "pca_coordinates.csv"))


# ---------------------------------------------------------------------------
# 3. Per-tissue latent distributions
# ---------------------------------------------------------------------------

def plot_tissue_distributions(z, meta, dataset, out_dir, max_dims=8):
    """
    Violin plots of the top max_dims most variable latent dimensions,
    grouped by tissue type. Shows how tissues separate in latent space.
    """
    print("Plotting per-tissue latent distributions...")

    # select the most variable dimensions
    dim_vars  = z.var(axis=0)
    top_dims  = np.argsort(dim_vars)[::-1][:max_dims]

    tissues     = meta["tissue"]
    tissue_names = dataset.tissue_enc.classes_
    n_tissues   = len(tissue_names)

    n_cols = 2
    n_rows = (max_dims + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    palette = [cm.get_cmap("tab20")(i / n_tissues) for i in range(n_tissues)]

    for plot_idx, dim in enumerate(top_dims):
        ax = axes[plot_idx]
        data_by_tissue = [z[tissues == t_idx, dim]
                          for t_idx in range(n_tissues)]
        # filter empty
        non_empty = [(tissue_names[i], data_by_tissue[i])
                     for i in range(n_tissues) if len(data_by_tissue[i]) > 0]
        labels   = [x[0] for x in non_empty]
        data     = [x[1] for x in non_empty]
        colors   = [palette[list(tissue_names).index(l)] for l in labels]

        parts = ax.violinplot(data, positions=range(len(data)),
                              showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_title("z" + str(dim) +
                     " (var=" + f"{dim_vars[dim]:.3f})", fontsize=9)
        ax.set_ylabel("Activation")

    # hide unused subplots
    for i in range(len(top_dims), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Per-Tissue Latent Distributions\n(top " +
                 str(max_dims) + " most variable dimensions)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "tissue_latent_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + str(path))


# ---------------------------------------------------------------------------
# 4. Flight vs ground centroid distance per tissue
# ---------------------------------------------------------------------------

def plot_centroid_distances(z, meta, dataset, out_dir):
    """
    For each tissue, compute Euclidean distance between:
        mean(z | flight=1, tissue=t)
    and
        mean(z | flight=0, tissue=t)

    Ranked bar chart showing which tissues have the largest
    spaceflight-induced shift in latent space.
    """
    print("Computing flight vs ground centroid distances per tissue...")

    tissues      = meta["tissue"]
    flights      = meta["flight"]
    tissue_names = dataset.tissue_enc.classes_

    rows = []
    for t_idx, t_name in enumerate(tissue_names):
        mask        = tissues == t_idx
        flight_mask = mask & (flights == 1)
        ground_mask = mask & (flights == 0)

        n_flight = flight_mask.sum()
        n_ground = ground_mask.sum()

        if n_flight < 3 or n_ground < 3:
            continue

        centroid_flight = z[flight_mask].mean(axis=0)
        centroid_ground = z[ground_mask].mean(axis=0)
        distance        = np.linalg.norm(centroid_flight - centroid_ground)

        # also compute cosine similarity
        norm_f   = centroid_flight / (np.linalg.norm(centroid_flight) + 1e-8)
        norm_g   = centroid_ground / (np.linalg.norm(centroid_ground) + 1e-8)
        cosine   = float(np.dot(norm_f, norm_g))

        rows.append({
            "tissue":           t_name,
            "n_flight":         int(n_flight),
            "n_ground":         int(n_ground),
            "centroid_distance": float(distance),
            "cosine_similarity": cosine,
        })

    df = pd.DataFrame(rows).sort_values("centroid_distance", ascending=False)
    df.to_csv(out_dir / "centroid_distances.csv", index=False)
    print("  Saved: " + str(out_dir / "centroid_distances.csv"))

    # bar chart
    n    = len(df)
    cmap = cm.get_cmap("RdYlGn_r", n)
    colors = [cmap(i / n) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.35)))

    # left: Euclidean distance
    ax = axes[0]
    ax.barh(df["tissue"][::-1], df["centroid_distance"][::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Euclidean Distance in Latent Space")
    ax.set_title("Spaceflight vs Ground Control\nLatent Centroid Distance by Tissue")
    for i, (_, row) in enumerate(df[::-1].iterrows()):
        ax.text(row["centroid_distance"] * 0.02,
                i, f"n={row['n_flight']}f/{row['n_ground']}g",
                va="center", fontsize=6, color="black")

    # right: cosine similarity (1 = same direction, -1 = opposite)
    ax2 = axes[1]
    cos_colors = ["#d73027" if v < 0.9 else "#4575b4" for v in df["cosine_similarity"][::-1]]
    ax2.barh(df["tissue"][::-1], df["cosine_similarity"][::-1],
             color=cos_colors, edgecolor="white")
    ax2.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Cosine Similarity (1 = same direction)")
    ax2.set_title("Latent Centroid Cosine Similarity\n(low = large directional shift)")
    ax2.set_xlim(min(df["cosine_similarity"].min() - 0.05, 0.7), 1.05)

    plt.tight_layout()
    path = out_dir / "centroid_distances.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: " + str(path))

    print("\nTop 10 tissues by spaceflight latent shift:")
    print(df[["tissue", "centroid_distance", "cosine_similarity",
              "n_flight", "n_ground"]].head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = SpaceflightDataset(args.data)
    model   = load_model(args.checkpoint, dataset, device)

    print("\nExtracting latent representations...")
    z, meta = extract_latents(model, dataset, device)

    print("\n--- 1. Latent Dimension Heatmap ---")
    plot_dimension_heatmap(z, meta, dataset, out_dir)

    print("\n--- 2. PCA of Latent Space ---")
    plot_pca(z, meta, dataset, out_dir)

    print("\n--- 3. Per-Tissue Latent Distributions ---")
    plot_tissue_distributions(z, meta, dataset, out_dir,
                              max_dims=args.max_dims)

    print("\n--- 4. Flight vs Ground Centroid Distances ---")
    plot_centroid_distances(z, meta, dataset, out_dir)

    print("\nDone. All visualizations saved to: " + str(out_dir))



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latent space visualization for Spaceflight cVAE"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data",       type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="latent_viz")
    parser.add_argument("--max_dims",   type=int, default=8,
                        help="Number of latent dims to show in violin plots")
    parser.add_argument("--hidden_dims",  type=int, nargs="+",
                    default=[512, 256], help="dimensions of encoder layers")

    args = parser.parse_args()
    run(args)
