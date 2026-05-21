"""
Conditional Multi-Task VAE for Spaceflight Transcriptomics
==========================================================
Architecture:
  - ConditionEmbedder: strain, sex, study, tissue, euth → condition vector
                       NOTE: flight is intentionally excluded here so the
                       decoder cannot trivially classify from the condition.
  - Encoder:           expression + condition → μ, log_var
  - Decoder:           z + condition → NB reconstruction params only
  - cls_head:          z → flight logit (forces encoder to encode spaceflight
                       signal into z, preventing posterior collapse)
  - BatchDiscriminator (optional, disabled by default)

Key design decision:
  Flight is removed from the condition embedder and the decoder classification
  head is moved to operate on z only. This forces the encoder to encode the
  spaceflight signal into the latent space — if it doesn't, the classification
  loss cannot be minimized. This prevents the KL collapse seen when the decoder
  could classify directly from the flight condition embedding.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal (for optional adversarial batch correction)
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
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ---------------------------------------------------------------------------
# Condition Embedder
# ---------------------------------------------------------------------------

class ConditionEmbedder(nn.Module):
    """
    Embeds metadata covariates into a single condition vector.

    Covariates (flight intentionally excluded — see module docstring):
        strain:  mouse strain        (categorical)
        sex:     sex                 (categorical)
        study:   GeneLab study ID    (categorical — batch variable)
        tissue:  tissue type         (categorical — biological)
        euth:    euthanasia method   (categorical — potential confounder)

    Args:
        n_strains:      number of unique strains
        n_sexes:        number of unique sex categories
        n_studies:      number of unique study IDs
        n_tissues:      number of unique tissue types
        n_euths:        number of unique euthanasia methods
        strain_emb_dim: embedding dim for strain
        sex_emb_dim:    embedding dim for sex
        study_emb_dim:  embedding dim for study
        tissue_emb_dim: embedding dim for tissue
        euth_emb_dim:   embedding dim for euthanasia method
    """
    def __init__(
        self,
        n_strains: int,
        n_sexes: int,
        n_studies: int,
        n_tissues: int,
        n_euths: int,
        strain_emb_dim: int = 16,
        sex_emb_dim: int = 4,
        study_emb_dim: int = 16,
        tissue_emb_dim: int = 32,
        euth_emb_dim: int = 8,
    ):
        super().__init__()

        self.strain_emb = nn.Embedding(n_strains, strain_emb_dim)
        self.sex_emb    = nn.Embedding(n_sexes,   sex_emb_dim)
        self.study_emb  = nn.Embedding(n_studies, study_emb_dim)
        self.tissue_emb = nn.Embedding(n_tissues, tissue_emb_dim)
        self.euth_emb   = nn.Embedding(n_euths,   euth_emb_dim)

        self.output_dim = (strain_emb_dim + sex_emb_dim + study_emb_dim +
                           tissue_emb_dim + euth_emb_dim)

    def forward(self, strain, sex, study, tissue, euth):
        """
        Args:
            strain: (B,) LongTensor
            sex:    (B,) LongTensor
            study:  (B,) LongTensor
            tissue: (B,) LongTensor
            euth:   (B,) LongTensor
        Returns:
            cond: (B, output_dim)  — does NOT include flight
        """
        return torch.cat([
            self.strain_emb(strain),
            self.sex_emb(sex),
            self.study_emb(study),
            self.tissue_emb(tissue),
            self.euth_emb(euth),
        ], dim=-1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Maps (expression, condition) → (μ, log_var).

    Args:
        n_genes:     number of input genes
        cond_dim:    dimension of condition vector (no flight)
        hidden_dims: list of hidden layer sizes
        latent_dim:  latent space dimension
        dropout:     dropout rate
    """
    def __init__(
        self,
        n_genes: int,
        cond_dim: int,
        hidden_dims: list = [256, 128],
        latent_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = n_genes + cond_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x, cond):
        h = self.net(torch.cat([x, cond], dim=-1))
        return self.mu(h), self.logvar(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Maps (z, condition) → NB reconstruction params only.
    Classification has been moved to a standalone head on z.

    Args:
        latent_dim:  dimension of z
        cond_dim:    dimension of condition vector (no flight)
        n_genes:     number of output genes
        hidden_dims: list of hidden layer sizes
        dropout:     dropout rate
    """
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        n_genes: int,
        hidden_dims: list = [128, 256],
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = latent_dim + cond_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim

        self.net        = nn.Sequential(*layers)
        self.log_r_head = nn.Linear(in_dim, n_genes)   # NB dispersion
        self.p_head     = nn.Linear(in_dim, n_genes)   # NB success prob

    def forward(self, z, cond):
        h     = self.net(torch.cat([z, cond], dim=-1))
        log_r = self.log_r_head(h)
        p     = torch.sigmoid(self.p_head(h))
        return log_r, p


# ---------------------------------------------------------------------------
# Batch Discriminator (optional adversarial batch correction)
# ---------------------------------------------------------------------------

class BatchDiscriminator(nn.Module):
    """
    Predicts study ID from z via gradient reversal.
    Disabled by default (grl_alpha=0.0).
    """
    def __init__(self, latent_dim: int, n_studies: int, alpha: float = 0.0):
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
# Full cVAE
# ---------------------------------------------------------------------------

class SpaceflightCVAE(nn.Module):
    """
    Conditional Multi-Task VAE for spaceflight transcriptomics.

    Flight is excluded from the condition embedder. Instead a standalone
    classification head reads directly from z, forcing the encoder to
    encode the spaceflight signal into the latent space.

    Args:
        n_genes:     number of protein-coding genes
        n_strains:   number of mouse strains
        n_sexes:     number of sex categories
        n_studies:   number of GeneLab study IDs
        n_tissues:   number of tissue types
        n_euths:     number of euthanasia method categories
        latent_dim:  latent space size
        hidden_dims: encoder hidden layer sizes
        dropout:     dropout rate
        grl_alpha:   gradient reversal strength (0.0 = disabled)
    """
    def __init__(
        self,
        n_genes: int,
        n_strains: int,
        n_sexes: int,
        n_studies: int,
        n_tissues: int,
        n_euths: int,
        latent_dim: int = 32,
        hidden_dims: list = [256, 128],
        dropout: float = 0.2,
        grl_alpha: float = 0.0,
    ):
        super().__init__()

        self.embedder = ConditionEmbedder(
            n_strains=n_strains,
            n_sexes=n_sexes,
            n_studies=n_studies,
            n_tissues=n_tissues,
            n_euths=n_euths,
        )
        cond_dim = self.embedder.output_dim   # now 76 (no flight embedding)

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

        # Classification head reads from z only — forces encoder to encode
        # spaceflight signal into the latent space
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.batch_disc = BatchDiscriminator(
            latent_dim=latent_dim,
            n_studies=n_studies,
            alpha=grl_alpha,
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, strain, sex, study, tissue, euth, flight):
        """
        Full forward pass.
        Note: flight is accepted for API compatibility but is NOT passed
        to the condition embedder — it is only used as the classification
        target in the loss function.

        Args:
            x:       (B, n_genes)
            strain:  (B,) LongTensor
            sex:     (B,) LongTensor
            study:   (B,) LongTensor
            tissue:  (B,) LongTensor
            euth:    (B,) LongTensor
            flight:  (B,) LongTensor  — target label, NOT fed to embedder

        Returns dict:
            mu, log_var, z, log_r, p, flight_logit, study_logit
        """
        cond = self.embedder(strain, sex, study, tissue, euth)
        mu, log_var = self.encoder(x, cond)
        z = self.reparameterize(mu, log_var)
        log_r, p = self.decoder(z, cond)
        flight_logit = self.cls_head(z)          # classify from z only
        study_logit  = self.batch_disc(z)

        return {
            "mu": mu, "log_var": log_var, "z": z,
            "log_r": log_r, "p": p,
            "flight_logit": flight_logit,
            "study_logit":  study_logit,
        }

    @torch.no_grad()
    def encode(self, x, strain, sex, study, tissue, euth, flight):
        """Return μ deterministically (no sampling). flight ignored."""
        cond = self.embedder(strain, sex, study, tissue, euth)
        mu, _ = self.encoder(x, cond)
        return mu

    @torch.no_grad()
    def generate(self, z, strain, sex, study, tissue, euth, flight):
        """Decode z back to expression space. flight ignored."""
        cond = self.embedder(strain, sex, study, tissue, euth)
        log_r, p = self.decoder(z, cond)
        return log_r, p
