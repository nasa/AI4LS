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

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, lbl in enumerate(unique_labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) > max_points:
            idx = np.random.choice(idx, max_points, replace=False)

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

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, lbl in enumerate(unique_labels):
        idx = np.where(labels == lbl)[0]
        class_mus = mus[idx]
        mean_mu = class_mus.mean(axis=0)
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


def build_condition_vector(batch, dataset, fields=("flight", "strain", "sex", "study", "tissue", "euth")):
    """
    Build the single concatenated one-hot condition vector your CVAE expects,
    from a batch dict produced by SpaceflightDataset.__getitem__.

    `dataset` must be the original SpaceflightDataset instance (not a Subset)
    so we can read n_strains, n_sexes, etc. for one-hot widths. If you used
    Subset(dataset, idx) for train/val/test splits, pass `subset.dataset`.

    IMPORTANT: this must exactly match however condition vectors were built
    during training (same field order, same one-hot widths) or the encoder
    will silently produce garbage. Adjust `fields` and the n_classes lookups
    below if your training script concatenated things differently (e.g. used
    embeddings instead of one-hot, or omitted some fields).
    """
    import torch.nn.functional as F

    n_classes_lookup = {
        "flight": 2,
        "strain": dataset.n_strains,
        "sex":    dataset.n_sexes,
        "study":  dataset.n_studies,
        "tissue": dataset.n_tissues,
        "euth":   dataset.n_euths,
    }

    one_hots = []
    for field in fields:
        ids = batch[field]  # (batch_size,) long tensor
        n_classes = n_classes_lookup[field]
        one_hots.append(F.one_hot(ids, num_classes=n_classes).float())

    return torch.cat(one_hots, dim=1)  # (batch_size, total_condition_dim)


def encode_dataset_h5(model, dataloader, dataset, device="cpu",
                       condition_fields=("flight", "strain", "sex", "study", "tissue", "euth"),
                       label_field="tissue"):
    """
    Run the CVAE encoder over a SpaceflightDataset-backed DataLoader.

    Builds the condition vector via `build_condition_vector` (single
    concatenated one-hot, matching your "single concatenated condition
    vector" training setup), and pulls `label_field` out of each batch
    separately to use purely for plot coloring (it doesn't have to be part
    of the condition vector, though by default it is, since "tissue" is
    in condition_fields too).

    Returns:
        mus: (N, latent_dim) array
        sigmas: (N, latent_dim) array (std devs)
        labels: (N,) array of string class labels (decoded, for legend)
    """
    model.eval()
    all_mu, all_sigma, all_label_ids = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            c = build_condition_vector(batch, dataset, fields=condition_fields).to(device)

            mu, log_var = model.encode(x, c)  # adjust if your model returns extra outputs
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
    Wire SpaceflightDataset -> trained CVAE -> latent ellipse plots,
    colored by tissue.
    """
    import torch
    from dataset import SpaceflightDataset, make_dataloaders  # adjust import path
    import argparse



    parser = argparse.ArgumentParser(
        description="Pretrain cVAE on ARCHS4 (reconstruction + KL only)"
    )
    parser.add_argument("--data",  type=str,   required=True,
                        help="Path to ARCHS4 pretrain H5")
    parser.add_argument("--ckpt",  type=str,   required=True,
                        help="Path to ckpt file")
    args = parser.parse_args()

    # from my_model import CVAE  # import your actual model class here

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Load dataset ---
    dataset = SpaceflightDataset(args.data, normalize=True)
    train_loader, val_loader, test_loader = make_dataloaders(
        dataset, batch_size=64, num_workers=0  # num_workers=0 is simpler for one-off viz scripts
    )

    # --- 2. Load trained model ---
    model = SpaceflightCVAE(
        input_dim=dataset.n_genes,
        condition_dim=2 + dataset.n_strains + dataset.n_sexes +
                       dataset.n_studies + dataset.n_tissues + dataset.n_euths,
        latent_dim=128,           # set to whatever you trained with
        hidden_dims=[512, 256],
    ).to(device)
    state_dict = torch.load(args.ckpt + "/pretrain_best.pt", map_location=device)
    model.load_state_dict(state_dict.get("model_state_dict", state_dict))
    #model = None  # <-- replace with the lines above once your checkpoint path is set

    if model is None:
        raise RuntimeError(
            "Set up the model loading block above before running main() for real. "
            "Until then, run this script's __main__ block for a dummy-data demo."
        )

    # --- 3. Encode validation set, colored by tissue ---
    mus, sigmas, labels = encode_dataset_h5(
        model, val_loader, dataset, device=device,
        label_field="tissue",
    )

    # --- 4. Reduce to 2D if latent_dim > 2 ---
    if mus.shape[1] > 2:
        mus, sigmas, _ = reduce_to_2d(mus, sigmas)

    # --- 5. Plot ---
    plot_per_point_ellipses(
        mus, sigmas, labels, n_std=1.0,
        title="Per-Datapoint Latent Posteriors by Tissue",
        save_path="./per_point_ellipses_tissue.png",
    )
    plot_class_aggregate_ellipses(
        mus, sigmas, labels, n_std=2.0, method="fit_cov",
        title="Per-Tissue Aggregate Latent Posteriors",
        save_path="./class_aggregate_ellipses_tissue.png",
    )
    plt.show()


if __name__ == "__main__":
    import torch  # only needed if running encode_dataset / encode_dataset_h5

    # By default this runs a dummy-data demo so you can sanity-check the
    # plotting code works before wiring up your real model + H5 file.
    # Once your checkpoint + model class are ready, switch to calling main().
    RUN_REAL_PIPELINE = True

    if RUN_REAL_PIPELINE:
        main()
    else:
        np.random.seed(0)
        n_per_class = 80
        mus = np.concatenate([
            np.random.randn(n_per_class, 2) * 0.8 + np.array([-2, 0]),
            np.random.randn(n_per_class, 2) * 0.8 + np.array([2, 0]),
        ])
        sigmas = np.abs(np.random.randn(2 * n_per_class, 2) * 0.3 + 0.5)
        labels = np.array(["tissue_A"] * n_per_class + ["tissue_B"] * n_per_class)

        plot_per_point_ellipses(
            mus, sigmas, labels, n_std=1.0,
            save_path="./per_point_ellipses.png"
        )
        plot_class_aggregate_ellipses(
            mus, sigmas, labels, n_std=2.0, method="fit_cov",
            save_path="./class_aggregate_ellipses.png"
        )
        plt.show()
