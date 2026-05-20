"""
Inference & Analysis Utilities for Spaceflight cVAE
=====================================================
Tools for:
  1. Loading a trained model
  2. Extracting latent representations
  3. UMAP visualization of the latent space
  4. Latent space arithmetic (e.g. spaceflight perturbation vectors)
  5. Gene importance via decoder gradient attribution
  6. Generating synthetic spaceflight/ground samples
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from model import SpaceflightCVAE


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    """
    Load a trained SpaceflightCVAE from a checkpoint.

    Returns:
        model:          SpaceflightCVAE in eval mode
        label_encoders: dict of {tissue, strain, study} LabelEncoders
        args:           training args dict
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]

    model = SpaceflightCVAE(
        n_genes=args["n_genes"],          # you may need to save this in train.py
        n_tissues=args["n_tissues"],
        n_strains=args["n_strains"],
        n_studies=args["n_studies"],
        latent_dim=args["latent_dim"],
        hidden_dims=[1024, 512, 256],
        dropout=0.0,                       # disable dropout at inference
        grl_alpha=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    return model, ckpt["label_encoders"], args


# ---------------------------------------------------------------------------
# Latent space extraction
# ---------------------------------------------------------------------------

def get_latent_representations(
    model: SpaceflightCVAE,
    dataloader,
    device: str = "cpu",
) -> dict:
    """
    Run encoder over a dataloader and collect latent means (μ).
    Uses μ (not sampled z) for deterministic downstream analysis.

    Returns dict with:
        z:        (N, latent_dim) numpy array of latent means
        flight:   (N,) array of spaceflight labels
        study:    (N,) array of study IDs
        tissue:   (N,) array of tissue IDs
        duration: (N,) array of mission durations
    """
    model.eval()
    zs, flights, studies, tissues, durations = [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            mu = model.encode(
                batch["x"].to(device),
                batch["tissue"].to(device),
                batch["strain"].to(device),
                batch["study"].to(device),
                batch["flight"].to(device),
                batch["duration"].to(device),
            )
            zs.append(mu.cpu().numpy())
            flights.append(batch["flight"].numpy())
            studies.append(batch["study"].numpy())
            tissues.append(batch["tissue"].numpy())
            durations.append(batch["duration"].numpy())

    return {
        "z":        np.concatenate(zs),
        "flight":   np.concatenate(flights),
        "study":    np.concatenate(studies),
        "tissue":   np.concatenate(tissues),
        "duration": np.concatenate(durations),
    }


# ---------------------------------------------------------------------------
# UMAP visualization
# ---------------------------------------------------------------------------

def plot_latent_umap(latent_dict: dict, color_by: str = "flight", save_path: str = None):
    """
    UMAP of latent space colored by a metadata variable.

    Args:
        latent_dict: output from get_latent_representations()
        color_by:    one of 'flight', 'study', 'tissue', 'duration'
        save_path:   if provided, save figure to this path
    """
    try:
        import umap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install umap-learn and matplotlib: pip install umap-learn matplotlib")

    print("Fitting UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent_dict["z"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=latent_dict[color_by],
        cmap="tab10" if color_by != "duration" else "viridis",
        s=12,
        alpha=0.7,
    )
    plt.colorbar(sc, ax=ax, label=color_by)
    ax.set_title(f"Latent Space (colored by {color_by})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved UMAP to {save_path}")
    else:
        plt.show()

    return embedding


# ---------------------------------------------------------------------------
# Latent space arithmetic
# ---------------------------------------------------------------------------

def compute_flight_vector(latent_dict: dict) -> np.ndarray:
    """
    Compute a 'spaceflight perturbation vector' in latent space:
        v_flight = mean(z | flight=1) - mean(z | flight=0)

    This vector captures the average direction of spaceflight effect
    and can be used for:
      - Counterfactual generation (what would this sample look like in space?)
      - Projecting new samples onto the spaceflight axis

    Returns:
        v_flight: (latent_dim,) numpy array
    """
    z_flight = latent_dict["z"][latent_dict["flight"] == 1]
    z_ground = latent_dict["z"][latent_dict["flight"] == 0]
    return z_flight.mean(axis=0) - z_ground.mean(axis=0)


def generate_counterfactual(
    model: SpaceflightCVAE,
    z: torch.Tensor,
    flight_vector: np.ndarray,
    tissue, strain, study, flight, duration,
    device: str = "cpu",
    scale: float = 1.0,
) -> tuple:
    """
    Generate a counterfactual expression profile by moving z along
    the spaceflight vector.

    For a ground control sample, this predicts what its expression
    would look like under spaceflight (and vice versa).

    Args:
        z:              (1, latent_dim) latent vector of the sample
        flight_vector:  (latent_dim,) spaceflight perturbation vector
        scale:          how far to move (1.0 = full perturbation)

    Returns:
        log_r, p:  NB reconstruction params for counterfactual
    """
    v = torch.tensor(flight_vector * scale, dtype=torch.float32).to(device)
    z_cf = z + v
    log_r, p = model.generate(z_cf, tissue, strain, study, flight, duration)
    return log_r, p


# ---------------------------------------------------------------------------
# Gene attribution via decoder gradients
# ---------------------------------------------------------------------------

def gene_attribution(
    model: SpaceflightCVAE,
    z: torch.Tensor,
    tissue, strain, study, flight, duration,
    target: str = "flight",
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute gene importance scores as the gradient of a target output
    with respect to the NB mean reconstruction.

    This answers: "which genes most influence the model's spaceflight
    prediction for this sample?"

    Args:
        z:      (1, latent_dim) latent vector
        target: 'flight' or 'duration'

    Returns:
        attribution: (n_genes,) importance scores (absolute gradient values)
    """
    model.eval()
    cond = model.embedder(tissue, strain, study, flight, duration)
    z = z.clone().requires_grad_(True)

    log_r, p, flight_logit, duration_hat = model.decoder(z, cond)

    # NB mean = r * (1-p) / p
    r = torch.exp(log_r)
    nb_mean = r * (1 - p) / p.clamp(min=1e-6)   # (1, n_genes)

    if target == "flight":
        scalar = flight_logit.squeeze()
    elif target == "duration":
        scalar = duration_hat.squeeze()
    else:
        raise ValueError(f"Unknown target: {target}")

    # Gradient of target output w.r.t. NB mean
    grads = torch.autograd.grad(scalar, nb_mean, retain_graph=False)[0]
    attribution = grads.abs().squeeze().detach().cpu().numpy()
    return attribution


# ---------------------------------------------------------------------------
# Quick analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(checkpoint_path: str, dataloader, gene_names: list, device: str = "cpu"):
    """
    End-to-end analysis: load model, extract latents, compute flight vector,
    and return a ranked gene attribution DataFrame.

    Args:
        checkpoint_path: path to .pt checkpoint
        dataloader:      DataLoader over your dataset
        gene_names:      list of gene names/IDs in the same order as model input

    Returns:
        latent_dict:     dict from get_latent_representations()
        flight_vector:   (latent_dim,) spaceflight perturbation vector
    """
    model, label_encoders, args = load_model(checkpoint_path, device=device)

    print("Extracting latent representations...")
    latent_dict = get_latent_representations(model, dataloader, device=device)

    print("Computing spaceflight perturbation vector...")
    flight_vector = compute_flight_vector(latent_dict)

    print(f"Latent space shape: {latent_dict['z'].shape}")
    print(f"Spaceflight samples: {(latent_dict['flight'] == 1).sum()}")
    print(f"Ground control samples: {(latent_dict['flight'] == 0).sum()}")

    return latent_dict, flight_vector
