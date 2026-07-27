import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_v2 import SpaceflightDataset, make_dataloaders
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
            tissue = batch["tissue"].to(device)
            study  = batch["study"].to(device)
            flight = batch["flight"].to(device)
            euth = batch["euth"].to(device)

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
# Training entry point
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Data ---
    '''print("Loading data...")
    dataset = SpaceflightDataset(args.data)
    train_loader, val_loader, _ = make_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )'''
    # --- Data ---
    print("Loading data...")
    # Try to read target metadata categories from the pretrain checkpoint (if provided)
    pre_target_cats = None
    pre_label_encs = None
    if args.pretrain_checkpoint:
        if os.path.exists(args.pretrain_checkpoint):
            print(f"Reading pretrain checkpoint metadata from {args.pretrain_checkpoint}")
            try:
                ckpt_meta = torch.load(args.pretrain_checkpoint, map_location="cpu")
                pre_target_cats = ckpt_meta.get("target_metadata_categories")
                pre_label_encs = ckpt_meta.get("label_encoders")
                if pre_target_cats:
                    print("  Found target_metadata_categories in pretrain checkpoint.")
                if pre_label_encs:
                    print("  Found label_encoders in pretrain checkpoint.")
            except Exception as e:
                print(f"  Warning: failed to read pretrain checkpoint metadata: {e}")
        else:
            print(f"  Warning: pretrain checkpoint {args.pretrain_checkpoint} not found.")

    # Construct dataset using pretrain's canonical categories when available
    dataset = SpaceflightDataset(args.data, target_metadata_categories=pre_target_cats)

    # Sanity print: compare classes from pretrain checkpoint (if any) and finetune dataset
    if pre_label_encs:
        print("Pretrain checkpoint label_encoders (sample):")
        for k in ["strain", "sex", "study", "tissue", "euth"]:
            v = pre_label_encs.get(k)
            if v is not None:
                print(f"  {k}: {len(v)} classes, sample: {v[:10]}")
    print("Finetune dataset classes (SpaceflightDataset):")
    for k in ["strain", "sex", "study", "tissue", "euth"]:
        enc = getattr(dataset, f"{k}_enc", None)
        classes = list(getattr(enc, "classes_", [])) if enc is not None else []
        print(f"  {k}: {len(classes)} classes, sample: {classes[:10]}")

    train_loader, val_loader, _ = make_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --- Model ---
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
        hidden_dims=[256, 128],
        dropout=args.dropout,
        grl_alpha=args.grl_alpha,
    ).to(device)

    # add to train.py temporarily
    print(f"Model conditions: {model.conditions}")
    print(f"Model latent_dim: {model.latent_dim}")
    print(f"Model cond_dim:   {model.cond_dim}")

    # --- Load pretrained weights (optional) ---
    if args.pretrain_checkpoint:
        print("Loading pretrained weights from: " + args.pretrain_checkpoint)
        ckpt = torch.load(args.pretrain_checkpoint, map_location=device,
                          weights_only=False)
        pretrain_state = ckpt["model_state"]
        model_state    = model.state_dict()

        pretrained_keys = set(pretrain_state.keys())
        current_keys = set(model_state.keys())
 
        # see if the xfer is good
        print("Keys in pretrained but NOT in current model:")
        for k in sorted(pretrained_keys - current_keys)[:10]:
            print(f"  {k}  {pretrain_state[k].shape}")

        print("\nKeys in current model but NOT in pretrained:")
        for k in sorted(current_keys - pretrained_keys)[:10]:
            print(f"  {k}  {model_state[k].shape}")

        print("\nKeys in both but shape mismatch:")
        for k in pretrained_keys & current_keys:
            if pretrain_state[k].shape != model_state[k].shape:
                print(f"  {k}: pretrained={pretrain_state[k].shape}  current={model_state[k].shape}")

        # Selectively transfer weights: copy all tensors where key name and
        # shape both match. Skip mismatches (e.g. batch_disc output layer whose
        # size differs between pretrain and fine-tune datasets).
        new_state = model_state.copy()
        transferred, skipped = [], []

        for name, param in pretrain_state.items():
            if name not in model_state:
                skipped.append((name, "not in current model"))
            elif param.shape != model_state[name].shape:
                skipped.append((name, f"shape mismatch {param.shape} vs {model_state[name].shape}"))
            else:
                new_state[name] = param.to(device)  # ensure correct device
                transferred.append(name)

        model.load_state_dict(new_state)
        print(f"  Transferred: {len(transferred)} parameter tensors")
        print(f"  Skipped:     {len(skipped)}")
        for name, reason in skipped:
            print(f"    - {name}: {reason}")

        # optionally freeze decoder for first N epochs
        if args.freeze_decoder_epochs > 0:
            print(f"  Freezing decoder for first {args.freeze_decoder_epochs} epochs")
            for param in model.decoder.parameters():
                param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")


    # --- Optimizer & scheduler ---
    if args.pretrain_checkpoint:
        # Differential learning rates:
        # - pretrained encoder/decoder weights: slow lr (already optimized)
        # - new random components (embedder, cls_head, batch_disc): fast lr
        pretrained_params = (
            list(model.encoder.parameters()) +
            list(model.decoder.parameters())
        )
        new_params = (
            list(model.embedder.parameters()) +
            list(model.cls_head.parameters()) +
            list(model.batch_disc.parameters())
        )
        lr_pretrained = args.lr                    # e.g. 5e-5
        lr_new        = args.lr * args.new_lr_mult # e.g. 5e-4 (10x)
        optimizer = AdamW([
            {"params": pretrained_params, "lr": lr_pretrained},
            {"params": new_params,        "lr": lr_new},
        ], weight_decay=1e-4)
        print(f"  Differential LR: pretrained={lr_pretrained:.2e}  "
              f"new components={lr_new:.2e} ({args.new_lr_mult}x)")
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Loss & annealing ---
    criterion = CVAELoss(
        beta=args.beta,
        lambda_cls=args.lambda_cls,
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

        # unfreeze decoder after freeze period
        if (args.pretrain_checkpoint and
                args.freeze_decoder_epochs > 0 and
                epoch == args.freeze_decoder_epochs + 1):
            print("Unfreezing decoder at epoch " + str(epoch))
            for param in model.decoder.parameters():
                param.requires_grad = True
            # rebuild optimizer preserving differential LR
            if args.pretrain_checkpoint:
                pretrained_params = (
                    list(model.encoder.parameters()) +
                    list(model.decoder.parameters())
                )
                new_params = (
                    list(model.embedder.parameters()) +
                    list(model.cls_head.parameters()) +
                    list(model.batch_disc.parameters())
                )
                optimizer = AdamW([
                    {"params": pretrained_params, "lr": args.lr},
                    {"params": new_params,        "lr": args.lr * args.new_lr_mult},
                ], weight_decay=1e-4)
            else:
                optimizer = AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)

        train_metrics, train_acc, train_auroc = run_epoch(
            model, train_loader, criterion, optimizer, device, kl_w, training=True
        )
        val_metrics, val_acc, val_auroc = run_epoch(
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
                f"val_auroc={val_auroc:.3f}  "
                f"kl_w={kl_w:.3f}"
                f"train_metrics={train_metrics}"
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

            #clf       = LogisticRegression(max_iter=200, C=0.1)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='saga')

            scaler    = StandardScaler()
            zs_scaled = scaler.fit_transform(zs)
            clf.fit(zs_scaled, studies)
            study_acc = clf.score(zs_scaled, studies)
            print(f"  Study predictability from z: {study_acc:.3f} (high expected — studies differ biologically)")

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
                    "strain": list(getattr(dataset.strain_enc, "classes_", [])),
                    "sex":    list(getattr(dataset.sex_enc,    "classes_", [])),
                    "study":  list(getattr(dataset.study_enc,  "classes_", [])),
                    "tissue": list(getattr(dataset.tissue_enc, "classes_", [])),
                    "euth":   list(getattr(dataset.euth_enc,   "classes_", [])),
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
    parser.add_argument("--data",        type=str, required=True, help="Path to subset_final.h5")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--latent_dim", type=int,   default=32)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--grl_alpha",  type=float, default=1.0, help="Gradient reversal strength")

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
    parser.add_argument("--tissue_emb_dim", type=int, default=8,
                        help="Tissue embedding dim (default 8)")
    parser.add_argument("--strain_emb_dim", type=int, default=8,
                        help="Strain embedding dim (default 8)")
    parser.add_argument("--sex_emb_dim",    type=int, default=4,
                        help="Sex embedding dim (default 4)")
    parser.add_argument("--study_emb_dim",  type=int, default=8,
                        help="Study embedding dim (default 8)")
    parser.add_argument("--euth_emb_dim",   type=int, default=4,
                        help="Euthanasia embedding dim (default 4)")
    parser.add_argument("--new_lr_mult",           type=float, default=10.0,
                        help="LR multiplier for new (non-pretrained) params "
                             "during fine-tuning (default: 10.0)")
    parser.add_argument("--pretrain_checkpoint",   type=str, default=None,
                        help="Path to pretrain_best.pt for transfer learning")
    parser.add_argument("--freeze_decoder_epochs", type=int, default=0,
                        help="Freeze decoder for N epochs after loading pretrain")

    args = parser.parse_args()
    train(args)
