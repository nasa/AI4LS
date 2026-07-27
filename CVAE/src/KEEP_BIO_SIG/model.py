"""
Conditional Multi-Task VAE for Spaceflight Transcriptomics
==========================================================
Architecture:
  - ConditionEmbedder: configurable subset of covariates → condition vector
                       Flight intentionally excluded — classifier reads z only.
  - Encoder:           [expression ‖ condition] → μ, log_var
  - Decoder:           [z ‖ condition] → NB reconstruction params
  - cls_head:          z → flight logit (forces spaceflight encoding into z)
  - BatchDiscriminator (optional, disabled by default)

Supported conditions (any subset via --conditions CLI arg):
  tissue    — 25 GeneLab tissue categories (emb dim configurable, default 32)
  strain    — 5 mouse strains             (emb dim configurable, default 16)
  sex       — Female/Male/Unknown         (emb dim configurable, default 4)
  study     — 72 GeneLab study IDs        (emb dim configurable, default 16)
  euth      — 6 euthanasia methods        (emb dim configurable, default 8)

Example — tissue only:
  python finetune.py --conditions tissue

Example — tissue + strain:
  python finetune.py --conditions tissue strain

Example — all conditions (default):
  python finetune.py --conditions tissue strain sex study euth
"""

import torch
import torch.nn as nn
from torch.autograd import Function


VALID_CONDITIONS = ["tissue", "strain", "sex", "study", "euth"]

DEFAULT_EMB_DIMS = {
    "tissue": 32,
    "strain": 16,
    "sex":    4,
    "study":  16,
    "euth":   8,
}


# ---------------------------------------------------------------------------
# Gradient Reversal
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
# Condition Embedder — configurable condition subset
# ---------------------------------------------------------------------------

class ConditionEmbedder(nn.Module):
    """
    Embeds a configurable subset of metadata covariates into a condition vector.
    Flight is always excluded — it is the classification target, not a condition.

    Args:
        conditions:     list of condition names to include, e.g. ["tissue", "strain"]
        n_tissues:      number of tissue categories
        n_strains:      number of strain categories
        n_sexes:        number of sex categories
        n_studies:      number of study IDs
        n_euths:        number of euthanasia methods
        tissue_emb_dim: embedding dim for tissue  (default 32)
        strain_emb_dim: embedding dim for strain  (default 16)
        sex_emb_dim:    embedding dim for sex     (default 4)
        study_emb_dim:  embedding dim for study   (default 16)
        euth_emb_dim:   embedding dim for euth    (default 8)
    """
    def __init__(
        self,
        conditions,
        n_tissues:  int = 25,
        n_strains:  int = 5,
        n_sexes:    int = 3,
        n_studies:  int = 72,
        n_euths:    int = 6,
        tissue_emb_dim: int = 32,
        strain_emb_dim: int = 16,
        sex_emb_dim:    int = 4,
        study_emb_dim:  int = 16,
        euth_emb_dim:   int = 8,
    ):
        super().__init__()

        self.conditions = [c for c in conditions if c in VALID_CONDITIONS]
        if not self.conditions:
            raise ValueError(
                f"No valid conditions in {conditions}. "
                f"Choose from: {VALID_CONDITIONS}"
            )

        self.output_dim = 0
        emb_map = {}

        if "tissue" in self.conditions:
            emb_map["tissue"] = nn.Embedding(n_tissues, tissue_emb_dim)
            self.output_dim += tissue_emb_dim
        if "strain" in self.conditions:
            emb_map["strain"] = nn.Embedding(n_strains, strain_emb_dim)
            self.output_dim += strain_emb_dim
        if "sex" in self.conditions:
            emb_map["sex"]    = nn.Embedding(n_sexes, sex_emb_dim)
            self.output_dim += sex_emb_dim
        if "study" in self.conditions:
            emb_map["study"]  = nn.Embedding(n_studies, study_emb_dim)
            self.output_dim += study_emb_dim
        if "euth" in self.conditions:
            emb_map["euth"]   = nn.Embedding(n_euths, euth_emb_dim)
            self.output_dim += euth_emb_dim

        # register as ModuleDict so parameters are tracked
        self.embeddings = nn.ModuleDict(emb_map)

        # store dims for inspection and checkpoint saving
        self.tissue_emb_dim = tissue_emb_dim if "tissue" in self.conditions else 0
        self.strain_emb_dim = strain_emb_dim if "strain" in self.conditions else 0
        self.sex_emb_dim    = sex_emb_dim    if "sex"    in self.conditions else 0
        self.study_emb_dim  = study_emb_dim  if "study"  in self.conditions else 0
        self.euth_emb_dim   = euth_emb_dim   if "euth"   in self.conditions else 0

    def forward(self, strain, sex, study, tissue, euth):
        """
        Accepts all five covariate tensors for API compatibility
        but only embeds the configured subset.

        Args: all (B,) LongTensors
        Returns: cond (B, output_dim)
        """
        parts = []
        if "tissue" in self.conditions:
            parts.append(self.embeddings["tissue"](tissue))
        if "strain" in self.conditions:
            parts.append(self.embeddings["strain"](strain))
        if "sex" in self.conditions:
            parts.append(self.embeddings["sex"](sex))
        if "study" in self.conditions:
            parts.append(self.embeddings["study"](study))
        if "euth" in self.conditions:
            parts.append(self.embeddings["euth"](euth))
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Maps [expression ‖ condition] → (μ, log_var)."""

    def __init__(
        self,
        n_genes: int,
        cond_dim: int,
        hidden_dims: list = [256, 128],
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = n_genes + cond_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

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
    """Maps [z ‖ condition] → NB reconstruction params."""

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
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h

        self.net        = nn.Sequential(*layers)
        self.log_r_head = nn.Linear(in_dim, n_genes)
        self.p_head     = nn.Linear(in_dim, n_genes)

    def forward(self, z, cond):
        h     = self.net(torch.cat([z, cond], dim=-1))
        log_r = self.log_r_head(h)
        p     = torch.sigmoid(self.p_head(h))
        return log_r, p


# ---------------------------------------------------------------------------
# Batch Discriminator
# ---------------------------------------------------------------------------

class BatchDiscriminator(nn.Module):
    """Predicts study ID from z via gradient reversal. Disabled by default."""

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

    The condition set is configurable via the `conditions` argument,
    allowing experiments with any subset of covariates.
    Flight is always excluded from conditions — it is the classification target.

    Args:
        n_genes:        number of protein-coding genes
        n_strains:      number of mouse strains
        n_sexes:        number of sex categories
        n_studies:      number of GeneLab study IDs
        n_tissues:      number of tissue types
        n_euths:        number of euthanasia method categories
        conditions:     list of conditions to use, e.g. ["tissue", "strain"]
                        default: all five conditions
        latent_dim:     latent space dimension (default 64)
        hidden_dims:    encoder hidden layer sizes (default [256, 128])
        dropout:        dropout rate (default 0.2)
        grl_alpha:      gradient reversal strength (0.0 = disabled)
        tissue_emb_dim: tissue embedding dim (default 32)
        strain_emb_dim: strain embedding dim (default 16)
        sex_emb_dim:    sex embedding dim (default 4)
        study_emb_dim:  study embedding dim (default 16)
        euth_emb_dim:   euthanasia embedding dim (default 8)
    """
    def __init__(
        self,
        n_genes: int,
        n_strains: int,
        n_sexes: int,
        n_studies: int,
        n_tissues: int,
        n_euths: int,
        conditions: list = None,
        latent_dim: int = 64,
        hidden_dims: list = [256, 128],
        dropout: float = 0.2,
        grl_alpha: float = 0.0,
        detach_cls_head: bool = False,
        tissue_emb_dim: int = 32,
        strain_emb_dim: int = 16,
        sex_emb_dim:    int = 4,
        study_emb_dim:  int = 16,
        euth_emb_dim:   int = 8,
    ):
        super().__init__()

        if conditions is None:
            conditions = VALID_CONDITIONS   # all five by default

        self.embedder = ConditionEmbedder(
            conditions=conditions,
            n_tissues=n_tissues,
            n_strains=n_strains,
            n_sexes=n_sexes,
            n_studies=n_studies,
            n_euths=n_euths,
            tissue_emb_dim=tissue_emb_dim,
            strain_emb_dim=strain_emb_dim,
            sex_emb_dim=sex_emb_dim,
            study_emb_dim=study_emb_dim,
            euth_emb_dim=euth_emb_dim,
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

        # Classification head reads z only — forces spaceflight encoding
        '''nn.Linear(latent_dim, 16),
        nn.GELU(),
        nn.Linear(16, 1),'''
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim, 1)
        )

        self.batch_disc = BatchDiscriminator(
            latent_dim=latent_dim,
            n_studies=n_studies,
            alpha=grl_alpha,
        )

        self.latent_dim  = latent_dim
        self.cond_dim    = cond_dim
        self.conditions  = self.embedder.conditions
        self.detach_cls_head = detach_cls_head

        # print architecture summary
        dims_str = " + ".join(
            f"{c}={getattr(self.embedder, c + '_emb_dim')}"
            for c in self.conditions
        )
        print(f"SpaceflightCVAE:")
        print(f"  Conditions:     {self.conditions}")
        print(f"  Cond dims:      {dims_str} = {cond_dim} total")
        print(f"  Encoder input:  {n_genes} + {cond_dim} = {n_genes + cond_dim}")
        print(f"  Latent dim:     {latent_dim}")
        print(f"  Decoder input:  {latent_dim} + {cond_dim} = {latent_dim + cond_dim}")
        print(f"  cls_head grad -> encoder: {'DISCONNECTED (detached)' if detach_cls_head else 'connected'}")

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, strain, sex, study, tissue, euth, flight):
        """
        flight accepted for API compatibility but NOT passed to embedder.
        All five covariate tensors accepted; only configured ones are embedded.
        """
        cond = self.embedder(strain, sex, study, tissue, euth)
        mu, log_var = self.encoder(x, cond)
        z = self.reparameterize(mu, log_var)
        log_r, p = self.decoder(z, cond)

        # If detach_cls_head is set, cls_loss still trains cls_head normally,
        # but gradients stop here and never reach the encoder — cls_head
        # becomes a passive probe on z rather than a force reshaping it.
        # batch_disc (adversarial study-invariance) intentionally stays
        # connected regardless — that term is meant to reshape the encoder.
        z_for_cls = z.detach() if self.detach_cls_head else z
        flight_logit = self.cls_head(z_for_cls)
        study_logit  = self.batch_disc(z)

        return {
            "mu": mu, "log_var": log_var, "z": z,
            "log_r": log_r, "p": p,
            "flight_logit": flight_logit,
            "study_logit":  study_logit,
        }

    @torch.no_grad()
    def encode(self, x, strain, sex, study, tissue, euth, flight):
        """Return μ deterministically. flight ignored."""
        cond = self.embedder(strain, sex, study, tissue, euth)
        mu, _ = self.encoder(x, cond)
        return mu

    @torch.no_grad()
    def generate(self, z, strain, sex, study, tissue, euth, flight):
        """Decode z to expression space. flight ignored."""
        cond = self.embedder(strain, sex, study, tissue, euth)
        log_r, p = self.decoder(z, cond)
        return log_r, p
