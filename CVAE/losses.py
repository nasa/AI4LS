"""
Loss Functions for Spaceflight cVAE
=====================================
L_total = L_recon + β·KL + λ_cls·L_cls + λ_reg·L_reg + λ_adv·L_adv

Components:
  L_recon  — Negative Binomial NLL on raw counts
  KL       — KL divergence from N(0,I), with annealing
  L_cls    — Binary cross-entropy for spaceflight classification
  L_reg    — Huber loss for mission duration regression
  L_adv    — Cross-entropy for adversarial batch correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def nb_nll_loss(x_raw: torch.Tensor, log_r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Negative Binomial negative log-likelihood (summed over genes, meaned over batch).

    NB parameterization:
        r = exp(log_r)     dispersion (number of successes)
        p                  probability of success ∈ (0, 1)
        mean = r * (1-p)/p

    Args:
        x_raw:  (B, G) raw integer counts
        log_r:  (B, G) log dispersion from decoder
        p:      (B, G) success probability from decoder ∈ (0,1)

    Returns:
        Scalar mean NLL loss.
    """
    r = torch.exp(log_r).clamp(min=1e-4, max=1e4)
    p = p.clamp(min=1e-6, max=1 - 1e-6)

    # NB log-likelihood per element
    log_nll = (
        torch.lgamma(x_raw + r)
        - torch.lgamma(r)
        - torch.lgamma(x_raw + 1)
        + r * torch.log(p)
        + x_raw * torch.log(1 - p)
    )
    return -log_nll.mean()


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL divergence: KL(N(μ, σ²) ‖ N(0, I))

    Args:
        mu:      (B, latent_dim)
        log_var: (B, latent_dim)

    Returns:
        Scalar mean KL loss.
    """
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl.sum(dim=-1).mean()


def classification_loss(flight_logit: torch.Tensor, flight: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy for spaceflight vs. ground control.

    Args:
        flight_logit: (B, 1) raw logit
        flight:       (B,) float labels — 0.0 or 1.0

    Returns:
        Scalar BCE loss.
    """
    return F.binary_cross_entropy_with_logits(
        flight_logit.squeeze(-1),
        flight.float(),
    )


def regression_loss(duration_hat: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
    """
    Huber loss for mission duration regression.
    Huber is more robust than MSE to occasional outlier missions.

    Args:
        duration_hat: (B, 1) predicted duration
        duration:     (B,) true duration in days

    Returns:
        Scalar Huber loss.
    """
    return F.huber_loss(duration_hat.squeeze(-1), duration.float())


def adversarial_batch_loss(study_logit: torch.Tensor, study: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy for batch discriminator.
    The gradient reversal layer inside BatchDiscriminator handles the
    adversarial flip — this loss is computed normally.

    Args:
        study_logit: (B, n_studies) raw logits
        study:       (B,) LongTensor of study IDs

    Returns:
        Scalar CE loss.
    """
    return F.cross_entropy(study_logit, study)


class CVAELoss(nn.Module):
    """
    Combined loss for the spaceflight cVAE.

    Args:
        beta:       KL weight (β-VAE). Use KL annealing to ramp this up.
        lambda_cls: weight for classification loss
        lambda_adv: weight for adversarial batch correction loss
    """
    def __init__(
        self,
        beta: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_adv: float = 0.1,
    ):
        super().__init__()
        self.beta       = beta
        self.lambda_cls = lambda_cls
        self.lambda_adv = lambda_adv

    def forward(
        self,
        outputs: dict,
        x_raw: torch.Tensor,
        flight: torch.Tensor,
        study: torch.Tensor,
        kl_weight: float = None,
    ) -> dict:
        """
        Args:
            outputs:   dict returned by SpaceflightCVAE.forward()
            x_raw:     (B, G) raw integer counts
            flight:    (B,) LongTensor spaceflight labels
            study:     (B,) LongTensor study IDs
            kl_weight: if provided, overrides self.beta (used during annealing)

        Returns:
            dict with individual loss components and total loss.
        """
        beta = kl_weight if kl_weight is not None else self.beta

        l_recon = nb_nll_loss(x_raw, outputs["log_r"], outputs["p"])
        l_kl    = kl_divergence(outputs["mu"], outputs["log_var"])
        l_cls   = classification_loss(outputs["flight_logit"], flight)
        l_adv   = adversarial_batch_loss(outputs["study_logit"], study)

        total = (
            l_recon
            + beta * l_kl
            + self.lambda_cls * l_cls
            + self.lambda_adv * l_adv
        )

        return {
            "loss":  total,
            "recon": l_recon.item(),
            "kl":    l_kl.item(),
            "cls":   l_cls.item(),
            "adv":   l_adv.item(),
        }


class KLAnnealer:
    """
    Linearly anneals KL weight from 0 → beta over `n_epochs` epochs.
    Prevents posterior collapse in early training.

    Usage:
        annealer = KLAnnealer(beta=1.0, n_epochs=50)
        for epoch in range(total_epochs):
            kl_w = annealer.get(epoch)
            loss_dict = criterion(outputs, ..., kl_weight=kl_w)
    """
    def __init__(self, beta: float = 1.0, n_epochs: int = 50):
        self.beta     = beta
        self.n_epochs = n_epochs

    def get(self, epoch: int) -> float:
        return min(self.beta, self.beta * epoch / self.n_epochs)
