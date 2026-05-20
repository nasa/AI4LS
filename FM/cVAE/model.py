"""
Conditional Multi-Task VAE for Spaceflight Transcriptomics
==========================================================
Architecture:
  - Conditional encoder (expression + metadata → μ, σ)
  - NB decoder with reconstruction, classification, and regression heads
  - Adversarial batch correction via gradient reversal
  - β-VAE with KL annealing

Designed for bulk RNA-seq data from NASA GeneLab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for adversarial batch correction)
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Reverses gradients during backprop to adversarially remove batch effects."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ---------------------------------------------------------------------------
# Condition Embedding Block
# ---------------------------------------------------------------------------

class ConditionEmbedder(nn.Module):
    """
    Embeds all metadata covariates into a single condition vector.

    Args:
        n_tissues:        number of unique tissue types
        n_strains:        number of unique mouse strains
        n_studies:        number of unique GeneLab study IDs
        tissue_emb_dim:   embedding dimension for tissue
        strain_emb_dim:   embedding dimension for strain
        study_emb_dim:    embedding dimension for study ID
        flight_emb_dim:   embedding dimension for spaceflight condition
        duration_dim:     output dim for mission duration projection
    """
    def __init__(
        self,
        n_tissues: int,
        n_strains: int,
        n_studies: int,
        tissue_emb_dim: int = 32,
        strain_emb_dim: int = 16,
        study_emb_dim: int = 32,
        flight_emb_dim: int = 8,
        duration_dim: int = 8,
    ):
        super().__init__()

        self.tissue_emb  = nn.Embedding(n_tissues, tissue_emb_dim)
        self.strain_emb  = nn.Embedding(n_strains, strain_emb_dim)
        self.study_emb   = nn.Embedding(n_studies, study_emb_dim)
        self.flight_emb  = nn.Embedding(2, flight_emb_dim)          # 0=ground, 1=flight
        self.duration_proj = nn.Linear(1, duration_dim)

        self.output_dim = tissue_emb_dim + strain_emb_dim + study_emb_dim + flight_emb_dim + duration_dim

    def forward(self, tissue, strain, study, flight, duration):
        """
        Args:
            tissue:   (B,) LongTensor
            strain:   (B,) LongTensor
            study:    (B,) LongTensor
            flight:   (B,) LongTensor  — 0 or 1
            duration: (B,) FloatTensor — mission duration in days
        Returns:
            cond: (B, output_dim)
        """
        dur = self.duration_proj(duration.unsqueeze(-1))
        cond = torch.cat([
            self.tissue_emb(tissue),
            self.strain_emb(strain),
            self.study_emb(study),
            self.flight_emb(flight),
            dur,
        ], dim=-1)
        return cond


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Maps (expression, condition) → (μ, log_var).

    Args:
        n_genes:     number of input genes
        cond_dim:    dimension of condition vector from ConditionEmbedder
        hidden_dims: list of hidden layer sizes
        latent_dim:  dimension of latent space z
        dropout:     dropout rate
    """
    def __init__(
        self,
        n_genes: int,
        cond_dim: int,
        hidden_dims: list = [1024, 512, 256],
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = n_genes + cond_dim
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x, cond):
        """
        Args:
            x:    (B, n_genes)   log1p-transformed counts
            cond: (B, cond_dim)
        Returns:
            mu, log_var: each (B, latent_dim)
        """
        h = self.net(torch.cat([x, cond], dim=-1))
        return self.mu(h), self.logvar(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Maps (z, condition) → reconstruction params + task outputs.

    Heads:
      - NB reconstruction: log_r (dispersion), p (success prob) per gene
      - Classification:    P(spaceflight) scalar
      - Regression:        predicted mission duration scalar

    Args:
        latent_dim:  dimension of z
        cond_dim:    dimension of condition vector
        n_genes:     number of output genes
        hidden_dims: list of hidden layer sizes (reversed from encoder)
        dropout:     dropout rate
    """
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        n_genes: int,
        hidden_dims: list = [256, 512, 1024],
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = latent_dim + cond_dim
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.net = nn.Sequential(*layers)

        # NB reconstruction heads
        self.log_r_head = nn.Linear(in_dim, n_genes)   # log dispersion
        self.p_head     = nn.Linear(in_dim, n_genes)   # logit of success prob

        # Task heads
        self.cls_head   = nn.Linear(in_dim, 1)         # spaceflight binary
        self.reg_head   = nn.Linear(in_dim, 1)         # mission duration

    def forward(self, z, cond):
        """
        Args:
            z:    (B, latent_dim)
            cond: (B, cond_dim)
        Returns:
            log_r:        (B, n_genes)   log NB dispersion
            p:            (B, n_genes)   NB success probability ∈ (0,1)
            flight_logit: (B, 1)         raw logit for spaceflight classification
            duration_hat: (B, 1)         predicted mission duration
        """
        h = self.net(torch.cat([z, cond], dim=-1))
        log_r        = self.log_r_head(h)
        p            = torch.sigmoid(self.p_head(h))
        flight_logit = self.cls_head(h)
        duration_hat = self.reg_head(h)
        return log_r, p, flight_logit, duration_hat


# ---------------------------------------------------------------------------
# Batch Corrector (adversarial)
# ---------------------------------------------------------------------------

class BatchDiscriminator(nn.Module):
    """
    Predicts study ID from z. Used adversarially to force z to be
    study-agnostic. Receives reversed gradients from GradientReversal.

    Args:
        latent_dim: dimension of z
        n_studies:  number of unique study IDs to classify
        alpha:      gradient reversal strength
    """
    def __init__(self, latent_dim: int, n_studies: int, alpha: float = 1.0):
        super().__init__()
        self.reversal = GradientReversal(alpha=alpha)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_studies),
        )

    def forward(self, z):
        return self.net(self.reversal(z))


# ---------------------------------------------------------------------------
# Full cVAE Model
# ---------------------------------------------------------------------------

class SpaceflightCVAE(nn.Module):
    """
    Conditional Multi-Task VAE for spaceflight transcriptomics.

    Combines:
      - ConditionEmbedder  (metadata → condition vector)
      - Encoder            (expression + condition → μ, log_var)
      - Decoder            (z + condition → NB params + task outputs)
      - BatchDiscriminator (z → study ID, adversarial)

    Args:
        n_genes:       number of genes (after filtering)
        n_tissues:     number of unique tissue types
        n_strains:     number of unique mouse strains
        n_studies:     number of unique GeneLab study IDs
        latent_dim:    size of latent space z
        hidden_dims:   encoder hidden layer sizes
        dropout:       dropout rate
        grl_alpha:     gradient reversal layer strength
        **emb_kwargs:  passed to ConditionEmbedder for embedding dims
    """
    def __init__(
        self,
        n_genes: int,
        n_tissues: int,
        n_strains: int,
        n_studies: int,
        latent_dim: int = 64,
        hidden_dims: list = [1024, 512, 256],
        dropout: float = 0.1,
        grl_alpha: float = 1.0,
        **emb_kwargs,
    ):
        super().__init__()

        self.embedder = ConditionEmbedder(
            n_tissues=n_tissues,
            n_strains=n_strains,
            n_studies=n_studies,
            **emb_kwargs,
        )
        cond_dim = self.embedder.output_dim

        self.encoder = Encoder(
            n_genes=n_genes,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            n_genes=n_genes,
            hidden_dims=list(reversed(hidden_dims)),
            dropout=dropout,
        )
        self.batch_disc = BatchDiscriminator(
            latent_dim=latent_dim,
            n_studies=n_studies,
            alpha=grl_alpha,
        )

        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """Sample z via reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, tissue, strain, study, flight, duration):
        """
        Full forward pass.

        Args:
            x:        (B, n_genes)  log1p raw counts
            tissue:   (B,) LongTensor
            strain:   (B,) LongTensor
            study:    (B,) LongTensor
            flight:   (B,) LongTensor
            duration: (B,) FloatTensor

        Returns dict with:
            mu, log_var:     latent distribution params
            z:               sampled latent vector
            log_r, p:        NB reconstruction params
            flight_logit:    classification logit
            duration_hat:    regression output
            study_logit:     batch discriminator output (for adversarial loss)
        """
        cond = self.embedder(tissue, strain, study, flight, duration)
        mu, log_var = self.encoder(x, cond)
        z = self.reparameterize(mu, log_var)
        log_r, p, flight_logit, duration_hat = self.decoder(z, cond)
        study_logit = self.batch_disc(z)

        return {
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "log_r": log_r,
            "p": p,
            "flight_logit": flight_logit,
            "duration_hat": duration_hat,
            "study_logit": study_logit,
        }

    @torch.no_grad()
    def encode(self, x, tissue, strain, study, flight, duration):
        """Inference-only: return μ (no sampling)."""
        cond = self.embedder(tissue, strain, study, flight, duration)
        mu, _ = self.encoder(x, cond)
        return mu

    @torch.no_grad()
    def generate(self, z, tissue, strain, study, flight, duration):
        """Decode a given z vector back to expression space."""
        cond = self.embedder(tissue, strain, study, flight, duration)
        log_r, p, _, _ = self.decoder(z, cond)
        return log_r, p
