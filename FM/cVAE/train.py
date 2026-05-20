"""
Training Loop for Spaceflight cVAE
====================================
Features:
  - KL annealing over first 50 epochs
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Early stopping on val loss
  - Checkpointing (best model saved)
  - wandb logging (optional)

Usage:
    python train.py --data genelab_samples.csv --gene_prefix ENSM
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import SpaceflightDataset, make_dataloaders
from losses import CVAELoss, KLAnnealer
from model import SpaceflightCVAE

# Optional wandb logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits.squeeze(-1) > 0).long()
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, kl_weight, training=True):
    model.train(training)
    totals = {"loss": 0, "recon": 0, "kl": 0, "cls": 0, "reg": 0, "adv": 0}
    accs = []
    n_batches = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            x        = batch["x"].to(device)
            x_raw    = batch["x_raw"].to(device)
            tissue   = batch["tissue"].to(device)
            strain   = batch["strain"].to(device)
            study    = batch["study"].to(device)
            flight   = batch["flight"].to(device)
            duration = batch["duration"].to(device)

            outputs = model(x, tissue, strain, study, flight, duration)
            loss_dict = criterion(
                outputs, x_raw, flight, duration, study,
                kl_weight=kl_weight,
            )

            if training:
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k in totals:
                totals[k] += loss_dict[k] if k == "loss" else loss_dict[k]
            accs.append(binary_accuracy(outputs["flight_logit"], flight))
            n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}, np.mean(accs)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Data ---
    print("Loading data...")
    df = pd.read_csv(args.data)
    gene_cols = [c for c in df.columns if c.startswith(args.gene_prefix)]
    print(f"  Samples: {len(df)}  |  Genes: {len(gene_cols)}")

    dataset = SpaceflightDataset(df, gene_cols)
    train_loader, val_loader, _ = make_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --- Model ---
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_tissues=dataset.n_tissues,
        n_strains=dataset.n_strains,
        n_studies=dataset.n_studies,
        latent_dim=args.latent_dim,
        hidden_dims=[1024, 512, 256],
        dropout=args.dropout,
        grl_alpha=args.grl_alpha,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # --- Optimizer & scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Loss & annealing ---
    criterion = CVAELoss(
        beta=args.beta,
        lambda_cls=args.lambda_cls,
        lambda_reg=args.lambda_reg,
        lambda_adv=args.lambda_adv,
    )
    annealer = KLAnnealer(beta=args.beta, n_epochs=args.kl_anneal_epochs)

    # --- wandb ---
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(project="spaceflight-cvae", config=vars(args))

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = Path(args.output_dir) / "best_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        kl_w = annealer.get(epoch)

        train_metrics, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, kl_w, training=True
        )
        val_metrics, val_acc = run_epoch(
            model, val_loader, criterion, None, device, kl_w, training=False
        )
        scheduler.step()

        # Logging
        log = {
            "epoch": epoch,
            "kl_weight": kl_w,
            "lr": scheduler.get_last_lr()[0],
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "train/flight_acc": train_acc,
            "val/flight_acc": val_acc,
        }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.3f}  "
                f"val_loss={val_metrics['loss']:.3f}  "
                f"val_acc={val_acc:.3f}  "
                f"kl_w={kl_w:.3f}"
            )

        if WANDB_AVAILABLE and not args.no_wandb:
            wandb.log(log)

        # Checkpointing
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
                "label_encoders": {
                    "tissue": dataset.tissue_enc,
                    "strain": dataset.strain_enc,
                    "study":  dataset.study_enc,
                },
            }, ckpt_path)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spaceflight cVAE")

    # Data
    parser.add_argument("--data",        type=str, required=True,  help="Path to CSV data file")
    parser.add_argument("--gene_prefix", type=str, default="ENSM", help="Prefix for gene count columns")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--latent_dim",  type=int,   default=64)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--grl_alpha",   type=float, default=1.0,  help="Gradient reversal strength")

    # Training
    parser.add_argument("--epochs",           type=int,   default=300)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--patience",         type=int,   default=30,  help="Early stopping patience")
    parser.add_argument("--kl_anneal_epochs", type=int,   default=50)

    # Loss weights
    parser.add_argument("--beta",       type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--lambda_adv", type=float, default=0.1)

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--no_wandb",   action="store_true")

    args = parser.parse_args()
    train(args)
