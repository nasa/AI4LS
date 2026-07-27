"""
Visualize a CVAE's latent space as Gaussian confidence ellipses.

Two modes:
  1. Per-datapoint ellipses: one ellipse per (mu, sigma) pair from the encoder,
     colored by condition. Good for seeing how individual posteriors vary in
     shape/size/overlap.
  2. Per-class-aggregate ellipses: one ellipse per class, fit either by
     averaging mu/sigma within a class or by fitting a covariance to the
     scattered mu points. Good for seeing class-level separation.

Assumes a 2D latent space (latent_dim == 2). If your latent_dim > 2, reduce
with PCA first (see `reduce_to_2d` below) — note that reducing mu is
straightforward, but reducing per-point sigma requires either using the PCA
projection matrix to transform the covariance, or just using the aggregate
spread of projected mu points as a proxy (this script handles both).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm


def encode_dataset(model, dataloader, device="cpu"):
    """
    Run the encoder over a dataset and collect mu, log_var, and condition
    labels. Adjust the unpacking below to match your dataloader's batch
    format and your model's encoder signature.

    Returns:
        mus: (N, latent_dim) array
        sigmas: (N, latent_dim) array  (standard deviations, not variances)
        labels: (N,) array of condition labels (for coloring)
    """
    model.eval()
    all_mu, all_sigma, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x, c = batch[0].to(device), batch[1].to(device)
            mu, log_var = model.encode(x, c)  # adjust to your model's API
            sigma = torch.exp(0.5 * log_var)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_labels.append(c.cpu().numpy())

    return (
        np.concatenate(all_mu, axis=0),
        np.concatenate(all_sigma, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def reduce_to_2d(mus, sigmas=None):
    """
    PCA-reduce mu vectors to 2D for visualization when latent_dim > 2.
    If sigmas is provided, approximates 2D sigma by projecting the diagonal
    covariance through the PCA components (a reasonable approximation when
    the encoder's covariance is diagonal, which it almost always is).
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    mus_2d = pca.fit_transform(mus)

    sigmas_2d = None
    if sigmas is not None:
        # Project per-dimension variances through PCA components.
        # var_2d_j = sum_i (component[j, i]^2 * var_i)
        variances = sigmas ** 2
        components = pca.components_  # shape (2, latent_dim)
        var_2d = variances @ (components ** 2).T  # (N, 2)
        sigmas_2d = np.sqrt(var_2d)

    return mus_2d, sigmas_2d, pca


def confidence_ellipse_from_diag(mu, sigma, ax, n_std=2.0, **kwargs):
    """
    Draw an axis-aligned ellipse from a diagonal-covariance Gaussian
    (i.e. independent mu_x, mu_y and sigma_x, sigma_y — the standard
    case for a VAE encoder with diagonal covariance output).

    n_std=1 draws the 1-std ellipse, n_std=2 draws ~95% confidence region.
    """
    width = 2 * n_std * sigma[0]
    height = 2 * n_std * sigma[1]
    ellipse = Ellipse(
        (mu[0], mu[1]), width=width, height=height,
        fill=False, **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def confidence_ellipse_from_cov(mu, cov, ax, n_std=2.0, **kwargs):
    """
    Draw a (possibly rotated) ellipse from a full 2x2 covariance matrix,
    e.g. when fitting an aggregate ellipse to a scatter of class mu points
    rather than relying on the encoder's per-point diagonal sigma.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        (mu[0], mu[1]), width=width, height=height, angle=angle,
        fill=False, **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def plot_per_point_ellipses(mus, sigmas, labels, n_std=1.0, max_points=150,
                             title="Per-Datapoint Latent Posteriors",
                             save_path=None):
    """
    Plot one ellipse per datapoint, colored by condition/class label.
    Subsamples to `max_points` per class if the dataset is large, since
    plotting thousands of overlapping ellipses gets unreadable.
    """
    unique_labels = np.unique(labels)
    colors = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))
    #colors = ['blue', 'red']

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, lbl in enumerate(unique_labels):
        if not lbl in ['Brain', 'EDL']:
            print('label: ', lbl)
            continue
        idx = np.where(labels == lbl)[0]
        if len(idx) > max_points:
            idx = np.random.choice(idx, max_points, replace=False)

        if lbl == 'Brain':
            color='red'
        elif lbl == 'EDL':
            color='blue'
        else:
            color = colors(i)

        for j in idx:
            confidence_ellipse_from_diag(
                mus[j], sigmas[j], ax, n_std=n_std,
                edgecolor=color, alpha=0.3, linewidth=0.8
            )
        # Plot the mu centers too, slightly bolder, with a legend entry
        ax.scatter(mus[idx, 0], mus[idx, 1], s=8, color=color,
                   label=f"class {lbl}", alpha=0.8)

    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_title(f"{title} ({n_std}-std contours)")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    plt.tight_layout()
    
    # mkdir 
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def plot_class_aggregate_ellipses(mus, sigmas, labels, n_std=2.0,
                                   method="mean_sigma",
                                   title="Per-Class Aggregate Latent Posteriors",
                                   save_path=None):
    """
    Plot one ellipse per class.

    method="mean_sigma": ellipse centered at the mean mu of the class,
        sized by the mean sigma of the class (treats the class as one
        "average" Gaussian — simple, but ignores spread of mu within
        the class).

    method="fit_cov": ellipse centered at the mean mu of the class, sized
        and oriented by the empirical covariance of the mu points
        themselves (captures how spread out / correlated the class's
        encodings are in latent space — usually more informative).
    """
    unique_labels = np.unique(labels)
    colors = plt.colormaps.get_cmap("tab10").resampled(len(unique_labels))
    #colors = ['blue', 'red']

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, lbl in enumerate(unique_labels):
        if not lbl in ['Brain', 'EDL']:
            print('label: ', lbl)
            continue
        idx = np.where(labels == lbl)[0]
        class_mus = mus[idx]
        mean_mu = class_mus.mean(axis=0)

        if lbl == 'Brain':
            color='red'
        elif lbl == 'EDL':
            color='blue'
        else:
            color = colors(i)


        if method == "mean_sigma":
            mean_sigma = sigmas[idx].mean(axis=0)
            confidence_ellipse_from_diag(
                mean_mu, mean_sigma, ax, n_std=n_std,
                edgecolor=color, linewidth=2.5,
                label=f"class {lbl}"
            )
        elif method == "fit_cov":
            cov = np.cov(class_mus.T)
            confidence_ellipse_from_cov(
                mean_mu, cov, ax, n_std=n_std,
                edgecolor=color, linewidth=2.5,
                label=f"class {lbl}"
            )
        else:
            raise ValueError("method must be 'mean_sigma' or 'fit_cov'")

        ax.scatter(class_mus[:, 0], class_mus[:, 1], s=10, color=color,
                   alpha=0.4)
        ax.scatter(*mean_mu, marker="x", s=120, color=color, linewidths=2.5)

    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_title(f"{title} ({method}, {n_std}-std)")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig, ax


def encode_dataset_h5(model, dataloader, dataset, device="cpu",
                       label_field="tissue"):
    """
    Run the SpaceflightCVAE encoder over a SpaceflightDataset-backed
    DataLoader.

    IMPORTANT — matches the real model interface found in pretrain.py:
    SpaceflightCVAE does NOT take a single pre-concatenated condition
    vector. It embeds each condition field internally (tissue_emb_dim,
    strain_emb_dim, sex_emb_dim, study_emb_dim, euth_emb_dim are separate
    learned embedding tables), so you pass raw label-encoded id tensors
    directly into the forward call:

        out = model(x, strain, sex, study, tissue, euth, flight)
        mu, log_var = out["mu"], out["log_var"]

    Only the condition fields actually in model.conditions are used
    internally by the model (e.g. if the model was trained with
    --conditions tissue, only the tissue embedding is active — the other
    id tensors are still passed positionally but ignored/zeroed
    internally, matching the run_epoch() pattern in pretrain.py). We still
    pass all of them since that's what the forward signature expects.

    `label_field` is pulled separately purely for plot coloring — it does
    not need to be one of the model's active conditions.

    Returns:
        mus: (N, latent_dim) array
        sigmas: (N, latent_dim) array (std devs)
        labels: (N,) array of string class labels (decoded, for legend)
    """
    model.eval()
    all_mu, all_sigma, all_label_ids = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x      = batch["x"].to(device)
            strain = batch["strain"].to(device)
            sex    = batch["sex"].to(device)
            study  = batch["study"].to(device)
            tissue = batch["tissue"].to(device)
            euth   = batch["euth"].to(device)
            flight = batch["flight"].to(device)

            out = model(x, strain, sex, study, tissue, euth, flight)
            mu, log_var = out["mu"], out["log_var"]
            sigma = torch.exp(0.5 * log_var)

            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_label_ids.append(batch[label_field].cpu().numpy())

    mus = np.concatenate(all_mu, axis=0)
    sigmas = np.concatenate(all_sigma, axis=0)
    label_ids = np.concatenate(all_label_ids, axis=0)

    # Decode integer label ids back to readable strings (e.g. "Liver", "Muscle")
    label_encoder = getattr(dataset, f"{label_field}_enc")
    labels = label_encoder.inverse_transform(label_ids)

    return mus, sigmas, labels


def main():
    """
    Wire SpaceflightDataset -> trained SpaceflightCVAE -> latent ellipse
    plots, colored by tissue.
    """
    import torch
    from dataset import SpaceflightDataset, make_dataloaders  # adjust import path if needed
    from model import SpaceflightCVAE  # adjust import path if needed
    import argparse
    parser = argparse.ArgumentParser(
        description="Pretrain cVAE on ARCHS4 (reconstruction + KL only)"
    )
    parser.add_argument("--data",  type=str,   required=True,
                        help="Path to ARCHS4 pretrain H5")
    parser.add_argument("--ckpt",  type=str,   required=True,
                        help="Path to ckpt file")
    parser.add_argument("--output",  type=str,   required=True,
                        help="Path to output dir")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = args.ckpt 
    H5_PATH = args.data 
    OUTPUT_DIR = args.output

    # make OUTPUT_DIR if doesn't exist
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load checkpoint first — it carries the exact architecture args
    #         used at training time (n_genes, latent_dim, conditions, etc.),
    #         so we don't have to guess or duplicate them by hand. ---
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    train_args = checkpoint["args"]  # dict of every CLI arg from pretrain.py/train.py

    # --- 2. Load dataset ---
    dataset = SpaceflightDataset(H5_PATH)
    _, val_loader, _ = make_dataloaders(
        dataset, batch_size=64, num_workers=0
    )
    # If this is the ARCHS4 pretrain H5 (no train/val/test split helper),
    # use a plain DataLoader instead, e.g.:
    # from torch.utils.data import DataLoader
    # val_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # --- 3. Rebuild model with the exact args used at training time ---
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_strains=dataset.n_strains,
        n_sexes=dataset.n_sexes,
        n_studies=dataset.n_studies,
        n_tissues=dataset.n_tissues,
        n_euths=dataset.n_euths,
        conditions=train_args["conditions"],
        latent_dim=train_args["latent_dim"],
        tissue_emb_dim=train_args["tissue_emb_dim"],
        strain_emb_dim=train_args["strain_emb_dim"],
        sex_emb_dim=train_args["sex_emb_dim"],
        study_emb_dim=train_args["study_emb_dim"],
        euth_emb_dim=train_args["euth_emb_dim"],
        hidden_dims=train_args["hidden_dims"],
        dropout=train_args["dropout"],
        grl_alpha=0.0,  # no domain-adversarial gradient needed for inference
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")
    print(f"Active conditions: {checkpoint['conditions']}")
    print(f"Latent dim: {checkpoint['latent_dim']}")

    # --- 4. Encode validation set, colored by tissue ---
    mus, sigmas, labels = encode_dataset_h5(
        model, val_loader, dataset, device=device,
        label_field="tissue",
    )

    # --- 5. Reduce to 2D if latent_dim > 2 (very likely here — default is 64) ---
    if mus.shape[1] > 2:
        mus, sigmas, _ = reduce_to_2d(mus, sigmas)

    # --- 6. Plot ---
    plot_per_point_ellipses(
        mus, sigmas, labels, n_std=1.0,
        title="Per-Datapoint Latent Posteriors by Tissue",
        save_path=OUTPUT_DIR + "/per_point_ellipses_tissue.png",
    )
    plot_class_aggregate_ellipses(
        mus, sigmas, labels, n_std=2.0, method="fit_cov",
        title="Per-Tissue Aggregate Latent Posteriors",
        save_path=OUTPUT_DIR + "/class_aggregate_ellipses_tissue.png",
    )
    #plt.show()


if __name__ == "__main__":
    import torch  # only needed if running encode_dataset / encode_dataset_h5

    # By default this runs a dummy-data demo so you can sanity-check the
    # plotting code works before wiring up your real model + H5 file.
    # Once your checkpoint + model class are ready, switch to calling main().

    main()
