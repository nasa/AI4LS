"""
Pretrain cVAE on ARCHS4 Mouse Bulk RNA-seq
==========================================
Stage 1 of 2-stage transfer learning.

Uses the same SpaceflightCVAE architecture and SpaceflightDataset
as train.py — the only difference is:
  - Loss: reconstruction + KL only (no flight classification head)
  - Dataset: ARCHS4 pretrain H5 (no flight labels)
  - No train/val/test split from the GeneLab dataset

The ARCHS4 H5 must have been prepared by prepare_archs4.py and have:
  data/expression          (n_genes, n_samples)
  meta/genes/ensembl_id    (n_genes,)
  meta/genes/symbol        (n_genes,)
  meta/samples/tissue      (n_samples,)   — optional
  meta/samples/spaceflight (n_samples,)   — all -1 (ignored)

Usage:
    # tissue-only conditioning (recommended)
    python pretrain_archs4.py \\
        --data archs4_pretrain.h5 \\
        --output_dir checkpoints/pretrain/tissue_v3/ \\
        --conditions tissue \\
        --epochs 500 --batch_size 128 --patience 40

    # zero-pad all conditions (unconditional)
    python pretrain_archs4.py \\
        --data archs4_pretrain.h5 \\
        --output_dir checkpoints/pretrain/uncond/ \\
        --conditions tissue strain sex study euth \\
        --epochs 500

Fine-tune after pretraining:
    python train.py \\
        --data subset_final.h5 \\
        --pretrain_checkpoint checkpoints/pretrain/tissue_v3/pretrain_best.pt \\
        --output_dir checkpoints/finetune/tissue_v3/ \\
        --conditions tissue \\
        --lr 5e-5 --new_lr_mult 10.0 \\
        --beta 0.005 --lambda_cls 2.0 \\
        --patience 40 --dropout 0.3
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from dataset import SpaceflightDataset
from model import SpaceflightCVAE, VALID_CONDITIONS
from losses import kl_divergence, nb_nll_loss, KLAnnealer


# ---------------------------------------------------------------------------
# Training loop — reconstruction + KL only, no flight classification
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, kl_weight, training=True):
    model.train(training)
    total_loss = total_recon = total_kl = 0.0
    n = 0

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

            # forward — use model.forward() but only use recon + KL
            out     = model(x, strain, sex, study, tissue, euth, flight)
            l_recon = nb_nll_loss(x_raw, out["log_r"], out["p"])
            l_kl    = kl_divergence(out["mu"], out["log_var"])
            loss    = l_recon + kl_weight * l_kl

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss  += loss.item()
            total_recon += l_recon.item()
            total_kl    += l_kl.item()
            n += 1

    return total_loss / n, total_recon / n, total_kl / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pretrain(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading ARCHS4 dataset...")
    dataset = SpaceflightDataset(args.data)

    # train/val split — 95/5
    n_val   = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # build model with same architecture as SpaceflightCVAE
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_strains=dataset.n_strains,
        n_sexes=dataset.n_sexes,
        n_studies=dataset.n_studies,
        n_tissues=dataset.n_tissues,
        n_euths=dataset.n_euths,
        conditions=args.conditions,
        latent_dim=args.latent_dim,
        tissue_emb_dim=args.tissue_emb_dim,
        strain_emb_dim=args.strain_emb_dim,
        sex_emb_dim=args.sex_emb_dim,
        study_emb_dim=args.study_emb_dim,
        euth_emb_dim=args.euth_emb_dim,
        hidden_dims=[256, 128],
        dropout=args.dropout,
        grl_alpha=0.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Conditions:       {args.conditions}")
    print(f"Latent dim:       {args.latent_dim}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    annealer  = KLAnnealer(beta=args.beta, n_epochs=args.kl_anneal_epochs)

    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "pretrain_best.pt"

    best_val         = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        kl_w = annealer.get(epoch)

        tr_loss, tr_recon, tr_kl = run_epoch(
            model, train_loader, optimizer, device, kl_w, training=True
        )
        va_loss, va_recon, va_kl = run_epoch(
            model, val_loader, None, device, kl_w, training=False
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train={tr_loss:.3f} (recon={tr_recon:.3f} kl={tr_kl:.2f}) | "
                f"val={va_loss:.3f} (recon={va_recon:.3f} kl={va_kl:.2f}) | "
                f"kl_w={kl_w:.4f}"
            )

        if va_loss < best_val:
            best_val         = va_loss
            patience_counter = 0
            torch.save({
                "epoch":          epoch,
                "val_loss":       va_loss,
                "model_state":    model.state_dict(),
                "args":           vars(args),
                "n_genes":        dataset.n_genes,
                "latent_dim":     args.latent_dim,
                "cond_dim":       model.cond_dim,
                "conditions":     model.conditions,
            }, ckpt_path)
            print(f"  ✓ Saved (val={best_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nPretraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"\nNext step — fine-tune on GeneLab:")
    print(f"  python train.py \\")
    print(f"      --data subset_final.h5 \\")
    print(f"      --pretrain_checkpoint {ckpt_path} \\")
    print(f"      --output_dir checkpoints/finetune/ \\")
    print(f"      --conditions {' '.join(args.conditions)} \\")
    print(f"      --lr 5e-5 --new_lr_mult 10.0 \\")
    print(f"      --beta 0.005 --lambda_cls 2.0 \\")
    print(f"      --patience 40 --dropout 0.3")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain cVAE on ARCHS4 (reconstruction + KL only)"
    )
    parser.add_argument("--data",             type=str,   required=True,
                        help="Path to ARCHS4 pretrain H5")
    parser.add_argument("--output_dir",       type=str,   required=True,
                        help="Output directory for checkpoint")
    parser.add_argument("--conditions",       type=str,   nargs="+",
                        default=["tissue"],
                        choices=VALID_CONDITIONS,
                        help="Conditions to embed (default: tissue)")
    parser.add_argument("--latent_dim",       type=int,   default=64)
    parser.add_argument("--tissue_emb_dim",   type=int,   default=32)
    parser.add_argument("--strain_emb_dim",   type=int,   default=16)
    parser.add_argument("--sex_emb_dim",      type=int,   default=4)
    parser.add_argument("--study_emb_dim",    type=int,   default=16)
    parser.add_argument("--euth_emb_dim",     type=int,   default=8)
    parser.add_argument("--dropout",          type=float, default=0.2)
    parser.add_argument("--epochs",           type=int,   default=500)
    parser.add_argument("--batch_size",       type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--beta",             type=float, default=0.01)
    parser.add_argument("--kl_anneal_epochs", type=int,   default=100)
    parser.add_argument("--patience",         type=int,   default=40)
    args = parser.parse_args()
    pretrain(args)
