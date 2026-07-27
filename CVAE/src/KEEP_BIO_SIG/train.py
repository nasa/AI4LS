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
  - Leave-one-study-out CV (see loso_cv.py)

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
    totals = {"loss": 0, "recon": 0, "kl": 0, "cls": 0, "adv": 0}
    all_logits, all_labels = [], []
    n_batches = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            x      = batch["x"].to(device)
            x_raw  = batch["x_raw"].to(device)
            strain = batch["strain"].to(device)
            sex    = batch["sex"].to(device)
            study  = batch["study"].to(device)
            tissue = batch["tissue"].to(device)
            euth   = batch["euth"].to(device)
            flight = batch["flight"].to(device)

            outputs = model(x, strain, sex, study, tissue, euth, flight)
            loss_dict = criterion(
                outputs, x_raw, flight, study,
                kl_weight=kl_weight,
            )

            if training:
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k in totals:
                totals[k] += loss_dict[k] if k == "loss" else loss_dict[k]
            all_logits.append(outputs["flight_logit"].squeeze(-1).detach().cpu().numpy())
            all_labels.append(flight.cpu().numpy())
            n_batches += 1

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    all_probs  = 1 / (1 + np.exp(-all_logits))
    preds      = (all_probs > 0.5).astype(int)
    accuracy   = (preds == all_labels).mean()

    from sklearn.metrics import roc_auc_score
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auroc = float("nan")

    return {k: v / n_batches for k, v in totals.items()}, accuracy, auroc


# ---------------------------------------------------------------------------
# Model + optimizer construction (shared by single-run training and LOSO CV)
# ---------------------------------------------------------------------------

def build_model(args, dataset, device):
    """Construct SpaceflightCVAE and optionally transfer pretrained weights.

    Called fresh for every LOSO fold so that each fold starts from the
    same pretrained checkpoint rather than leaking weights between folds.
    """
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_strains=dataset.n_strains,
        n_sexes=dataset.n_sexes,
        n_tissues=dataset.n_tissues,
        n_euths=dataset.n_euths,
        n_studies=dataset.n_studies,
        conditions=args.conditions,
        latent_dim=args.latent_dim,
        tissue_emb_dim=args.tissue_emb_dim,
        strain_emb_dim=args.strain_emb_dim,
        sex_emb_dim=args.sex_emb_dim,
        study_emb_dim=args.study_emb_dim,
        euth_emb_dim=args.euth_emb_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        grl_alpha=args.grl_alpha,
        detach_cls_head=args.detach_cls_head,
    ).to(device)

    if args.pretrain_checkpoint:
        print("Loading pretrained weights from: " + args.pretrain_checkpoint)
        ckpt = torch.load(args.pretrain_checkpoint, map_location=device,
                          weights_only=False)
        transfer_pretrained_weights(model, args.pretrain_checkpoint, device,
                                     ckpt=ckpt, verbose=True)

        if args.reinit_latent_heads:
            for layer in [model.encoder.mu, model.encoder.logvar]:
                layer.reset_parameters()
            print("  Re-initialized encoder mu and logvar heads (fresh latent projection)")

        if args.freeze_decoder_epochs > 0:
            print(f"  Freezing decoder for first {args.freeze_decoder_epochs} epochs")
            for param in model.decoder.parameters():
                param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    return model


def transfer_pretrained_weights(model, checkpoint_path, device, ckpt=None, verbose=True):
    """
    Load whatever weights from `checkpoint_path` match `model`'s architecture
    by name (after key-remapping) and shape, leaving anything that doesn't
    match at its current (randomly initialized) value.

    This was originally written for pretrain -> finetune transfer (handles
    PretrainCVAE's different submodule names: encoder_net/mu_head/etc.), but
    the shape-checked skip-on-mismatch behavior also makes it safe to reuse
    for finetune -> eval-probe transfer, where condition vocab sizes (and
    therefore the very first encoder/decoder layer) may legitimately differ
    between the checkpoint's original training data and a new probe dataset.

    Returns (transferred_keys, skipped_entries).
    """
    if ckpt is None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrain_state = ckpt["model_state"]
    model_state    = model.state_dict()

    # PretrainCVAE key    → SpaceflightCVAE key
    #   encoder_net.*     → encoder.net.*
    #   mu_head.*         → encoder.mu.*
    #   logvar_head.*     → encoder.logvar.*
    #   decoder_net.*     → decoder.net.*
    #   log_r_head.*      → decoder.log_r_head.*
    #   p_head.*          → decoder.p_head.*
    #
    # Condition embeddings and classifier/discriminator heads are skipped
    # (except tissue, which is assumed to share vocab with GeneLab) — these
    # are either not present in a pretrain checkpoint, or represent
    # task/vocab-specific weights that shouldn't transfer to a differently
    # -scoped model.
    transferred = []
    skipped     = []
    for key, val in pretrain_state.items():
        if any(key.startswith(p) for p in ["cls_head.", "batch_disc."]):
            skipped.append(key)
            continue
        if key.startswith("embedder.") and "tissue" not in key:
            skipped.append(key)
            continue

        if key.startswith("encoder_net."):
            mapped_key = key.replace("encoder_net.", "encoder.net.")
        elif key.startswith("mu_head."):
            mapped_key = key.replace("mu_head.", "encoder.mu.")
        elif key.startswith("logvar_head."):
            mapped_key = key.replace("logvar_head.", "encoder.logvar.")
        elif key.startswith("decoder_net."):
            mapped_key = key.replace("decoder_net.", "decoder.net.")
        elif key.startswith("log_r_head."):
            mapped_key = key.replace("log_r_head.", "decoder.log_r_head.")
        elif key.startswith("p_head."):
            mapped_key = key.replace("p_head.", "decoder.p_head.")
        else:
            mapped_key = key

        if mapped_key in model_state and model_state[mapped_key].shape == val.shape:
            model_state[mapped_key] = val
            transferred.append(mapped_key)
        else:
            skipped.append(key + " (shape mismatch: checkpoint "
                           + str(val.shape) + " vs model "
                           + str(model_state.get(mapped_key, torch.tensor([])).shape) + ")")

    if verbose:
        print(f"  Transferring from: {checkpoint_path}")
        print('  Skipping the following weights at xfer:')
        for skip in skipped:
            print(f"    - {skip}")
    model.load_state_dict(model_state)
    if verbose:
        print(f"  Transferred: {len(transferred)} parameter tensors")
        print(f"  Skipped:     {len(skipped)} (condition embeddings + mismatches)")

    return transferred, skipped




def build_optimizer(args, model):
    if args.pretrain_checkpoint:
        # Differential learning rates:
        # - pretrained encoder/decoder weights: slow lr (already optimized)
        # - new random components (embedder, cls_head, batch_disc): fast lr
        #
        # NOTE: if --reinit_latent_heads was used, encoder.mu / encoder.logvar
        # were just randomly reinitialized. They're submodules of
        # model.encoder, so without special-casing them here they'd get
        # bundled into the SLOW pretrained LR group despite being untrained —
        # risking them barely moving off their random init within a short
        # fine-tune budget. Route them to the fast group in that case.
        decoder_params = list(model.decoder.parameters())

        if args.reinit_latent_heads:
            head_param_ids = {
                id(p) for p in list(model.encoder.mu.parameters())
                            + list(model.encoder.logvar.parameters())
            }
            encoder_body_params = [
                p for p in model.encoder.parameters()
                if id(p) not in head_param_ids
            ]
            encoder_head_params = (
                list(model.encoder.mu.parameters()) +
                list(model.encoder.logvar.parameters())
            )
            pretrained_params = encoder_body_params + decoder_params
            new_params = (
                list(model.embedder.parameters()) +
                list(model.cls_head.parameters()) +
                list(model.batch_disc.parameters()) +
                encoder_head_params
            )
        else:
            pretrained_params = list(model.encoder.parameters()) + decoder_params
            new_params = (
                list(model.embedder.parameters()) +
                list(model.cls_head.parameters()) +
                list(model.batch_disc.parameters())
            )

        lr_pretrained = args.lr
        lr_new        = args.lr * args.new_lr_mult
        optimizer = AdamW([
            {"params": pretrained_params, "lr": lr_pretrained},
            {"params": new_params,        "lr": lr_new},
        ], weight_decay=1e-4)
        print(f"  Differential LR: pretrained={lr_pretrained:.2e}  "
              f"new components={lr_new:.2e} ({args.new_lr_mult}x)")
        if args.reinit_latent_heads:
            print("  (encoder.mu / encoder.logvar routed to fast LR group "
                  "— reinitialized, not pretrained)")
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer


# ---------------------------------------------------------------------------
# Core training loop — shared by single-run CLI training and LOSO CV.
# Everything here is a function of (args, dataset, train_loader, val_loader,
# ckpt_path, run_label) so a caller can point it at any split.
# ---------------------------------------------------------------------------

def run_training(args, dataset, train_loader, val_loader, ckpt_path, run_label=""):
    """
    Runs the full training loop (model build -> training -> early stopping)
    for one train/val split, and returns the best-checkpoint metrics.

    run_label is just a string used in print statements and wandb run
    naming, e.g. "" for a normal run or "loso_OSD-123" for a LOSO fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args, dataset, device)
    optimizer = build_optimizer(args, model)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = CVAELoss(
        beta=args.beta,
        lambda_cls=args.lambda_cls,
        lambda_adv=args.lambda_adv,
    )
    annealer = KLAnnealer(beta=args.beta, n_epochs=args.kl_anneal_epochs)

    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(project="spaceflight-cvae", config=vars(args),
                   name=run_label or None, reinit=True)

    best_val_loss = float("inf")
    best_val_acc  = float("nan")
    best_val_auroc = float("nan")
    patience_counter = 0
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = f"[{run_label}] " if run_label else ""

    for epoch in range(1, args.epochs + 1):
        kl_w = annealer.get(epoch)

        if (args.pretrain_checkpoint and
                args.freeze_decoder_epochs > 0 and
                epoch == args.freeze_decoder_epochs + 1):
            print(f"{prefix}Unfreezing decoder at epoch {epoch}")
            for param in model.decoder.parameters():
                param.requires_grad = True
            # Rebuild via build_optimizer() rather than duplicating the
            # param-group logic here, so the reinit_latent_heads LR routing
            # (mu/logvar -> fast group) stays consistent in both places.
            optimizer = build_optimizer(args, model)

        train_metrics, train_acc, train_auroc = run_epoch(
            model, train_loader, criterion, optimizer, device, kl_w, training=True
        )
        val_metrics, val_acc, val_auroc = run_epoch(
            model, val_loader, criterion, None, device, kl_w, training=False
        )
        scheduler.step()

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
                f"{prefix}Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.3f}  "
                f"val_loss={val_metrics['loss']:.3f}  "
                f"val_acc={val_acc:.3f}  "
                f"val_auroc={val_auroc:.3f}  "
                f"kl_w={kl_w:.3f}"
            )

        if epoch % 20 == 0:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            model.eval()
            zs, studies = [], []
            with torch.no_grad():
                for batch in val_loader:
                    z = model.encode(
                        batch["x"].to(device),
                        batch["strain"].to(device),
                        batch["sex"].to(device),
                        batch["study"].to(device),
                        batch["tissue"].to(device),
                        batch["euth"].to(device),
                        batch["flight"].to(device),
                    )
                    zs.append(z.cpu().numpy())
                    studies.append(batch["study"].numpy())

            zs      = np.concatenate(zs)
            studies = np.concatenate(studies)

            # In a LOSO fold, the held-out val set is a SINGLE study, so
            # this probe can't run (only one class present) — skip it there.
            if len(np.unique(studies)) > 1:
                clf       = LogisticRegression(max_iter=200, C=0.1)
                scaler    = StandardScaler()
                zs_scaled = scaler.fit_transform(zs)
                clf.fit(zs_scaled, studies)
                study_acc = clf.score(zs_scaled, studies)
                print(f"{prefix}  Study predictability from z: {study_acc:.3f}")
            else:
                print(f"{prefix}  Study predictability probe skipped "
                      f"(val set is a single held-out study)")

        if WANDB_AVAILABLE and not args.no_wandb:
            wandb.log(log)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_acc  = val_acc
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "val_auroc": best_val_auroc,
                "args": vars(args),
                "label_encoders": {
                    "strain": dataset.strain_enc,
                    "sex":    dataset.sex_enc,
                    "study":  dataset.study_enc,
                    "tissue": dataset.tissue_enc,
                    "euth":   dataset.euth_enc,
                },
            }, ckpt_path)
            print(f"{prefix}  \u2713 Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"{prefix}Early stopping at epoch {epoch}")
                break

    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.finish()

    print(f"{prefix}Training complete. Best val loss: {best_val_loss:.4f}  "
          f"acc={best_val_acc:.3f}  auroc={best_val_auroc:.3f}")
    print(f"{prefix}Model saved to: {ckpt_path}")

    return {
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "val_auroc": best_val_auroc,
        "ckpt_path": str(ckpt_path),
    }


# ---------------------------------------------------------------------------
# Training entry point (single train/val split, as before)
# ---------------------------------------------------------------------------

def train(args):
    print("Loading data...")
    dataset = SpaceflightDataset(args.data)
    train_loader, val_loader, _ = make_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ckpt_path = Path(args.output_dir) / "best_model.pt"
    run_training(args, dataset, train_loader, val_loader, ckpt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train Spaceflight cVAE")

    # Data
    parser.add_argument("--data",        type=str, required=True, help="Path to subset_final.h5")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--latent_dim", type=int,   default=32)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--grl_alpha",  type=float, default=1.0, help="Gradient reversal strength")
    parser.add_argument("--detach_cls_head", action="store_true",
                        help="Detach z before cls_head so classification "
                             "gradients don't reach the encoder — cls_head "
                             "becomes a passive probe rather than reshaping "
                             "the latent space. Reconstruction/KL/adversarial "
                             "study-invariance are unaffected.")
    parser.add_argument("--hidden_dims",  type=int, nargs="+",
                        default=[512, 256], help="dimensions of encoder layers")

    # Training
    parser.add_argument("--epochs",           type=int,   default=300)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--patience",         type=int,   default=30)
    parser.add_argument("--kl_anneal_epochs", type=int,   default=100)

    # Loss weights
    parser.add_argument("--beta",       type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.1)

    # Output
    parser.add_argument("--output_dir",            type=str, default="./checkpoints")
    parser.add_argument("--no_wandb",              action="store_true")
    # which conditions to include in the embedder
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["tissue", "strain", "sex", "study", "euth"],
                        choices=["tissue", "strain", "sex", "study", "euth"],
                        help="Conditions to embed (default: all five). "
                             "Example: --conditions tissue strain")

    # condition embedding dimensions
    parser.add_argument("--tissue_emb_dim", type=int, default=32,
                        help="Tissue embedding dim (default 32)")
    parser.add_argument("--strain_emb_dim", type=int, default=16,
                        help="Strain embedding dim (default 16)")
    parser.add_argument("--sex_emb_dim",    type=int, default=4,
                        help="Sex embedding dim (default 4)")
    parser.add_argument("--study_emb_dim",  type=int, default=16,
                        help="Study embedding dim (default 16)")
    parser.add_argument("--euth_emb_dim",   type=int, default=8,
                        help="Euthanasia embedding dim (default 8)")
    parser.add_argument("--new_lr_mult",           type=float, default=10.0,
                        help="LR multiplier for new (non-pretrained) params "
                             "during fine-tuning (default: 10.0)")
    parser.add_argument("--pretrain_checkpoint",   type=str, default=None,
                        help="Path to pretrain_best.pt for transfer learning")
    parser.add_argument("--reinit_latent_heads",   action="store_true",
                        help="Re-initialize encoder mu and logvar heads after "
                             "weight transfer. Forces fresh latent organization "
                             "while preserving pretrained encoder body weights.")
    parser.add_argument("--freeze_decoder_epochs", type=int, default=0,
                        help="Freeze decoder for N epochs after loading pretrain")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
