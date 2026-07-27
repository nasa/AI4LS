"""
Pretrain cVAE on ARCHS4 Mouse Bulk RNA-seq (v5 — zero-pad conditions)
======================================================================
Stage 1 of 2-stage transfer learning.

Key design: condition vector is zero-padded during pretraining.
The encoder receives [x ‖ 0...0] of shape (B, 18983) — the same
input dimension as SpaceflightCVAE which receives [x ‖ cond].

This means ALL encoder and decoder layer weights transfer perfectly
to SpaceflightCVAE during fine-tuning, including the critical first
encoder layer Linear(18983 → 256). Fine-tuning then activates the
condition portion with real GeneLab metadata.

Loss: NB reconstruction + β·KL  (no flight classification)

Architecture mirrors SpaceflightCVAE exactly:
  encoder: Linear(18983→256) → LayerNorm → GELU → Dropout
           Linear(256→128)   → LayerNorm → GELU → Dropout
           → μ(64), log_var(64)
  decoder: Linear(140→128)   → LayerNorm → GELU → Dropout
           Linear(128→256)   → LayerNorm → GELU → Dropout
           → log_r(18907), p(18907)

  where 18983 = 18907 genes + 76 condition dims (all zeros during pretrain)
  and   140   = 64 latent dims + 76 condition dims (all zeros during pretrain)

Usage:
    python pretrain_archs4.py \\
        --data archs4_pretrain.h5 \\
        --output_dir checkpoints_pretrain_v5/

Fine-tune:
    python finetune.py \\
        --data subset_final.h5 \\
        --pretrain_checkpoint checkpoints_pretrain_v5/pretrain_best.pt \\
        --output_dir checkpoints_finetune_v5/ \\
        --lr 5e-5 --beta 0.005 --lambda_cls 2.0 \\
        --freeze_decoder_epochs 0 --patience 40 --dropout 0.3
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split

import h5py

from losses import kl_divergence, nb_nll_loss, KLAnnealer


# These must match SpaceflightCVAE exactly so weights transfer cleanly
N_GENES    = 18907
#COND_DIM   = 76    # total condition embedding dim in SpaceflightCVAE
                    # = tissue(32) + strain(16) + sex(4) + study(16) + euth(8)
COND_DIM   = 28     # tissue (16) + strain(4) + sex(2) + study(4) + euth(2)
                    # --tissue_emb_dim 16 
                    # --strain_emb_dim 4 
                    # --sex_emb_dim 2 
                    # --study_emb_dim 4 
                    # --euth_emb_dim 2
LATENT_DIM = 64
ENC_INPUT  = N_GENES + COND_DIM   # 18983 — same as SpaceflightCVAE
DEC_INPUT  = LATENT_DIM + COND_DIM  # 140   — same as SpaceflightCVAE


# ---------------------------------------------------------------------------
# ARCHS4 Dataset — no metadata needed, just expression
# ---------------------------------------------------------------------------

class ARCHS4Dataset(Dataset):
    """
    Loads ARCHS4 pretrain H5.
    Only needs data/expression and meta/genes — no sample metadata.
    """

    def __init__(self, h5_path, normalize=True):
        super().__init__()

        def decode(val):
            if isinstance(val, (bytes, np.bytes_)):
                return val.decode("utf-8", errors="replace").strip()
            return str(val).strip()

        with h5py.File(h5_path, "r") as f:
            print("  Reading expression matrix...")
            expr = f["data/expression"][:].T.astype(np.float32)
            # (n_samples, n_genes)

            self.ensembl_ids  = np.array([decode(v)
                                          for v in f["meta/genes/ensembl_id"][:]])
            self.gene_symbols = np.array([decode(v)
                                          for v in f["meta/genes/symbol"][:]])

        assert expr.shape[1] == N_GENES, (
            f"Gene count mismatch: expected {N_GENES}, got {expr.shape[1]}. "
            f"Regenerate pretrain H5 with the correct gene set."
        )

        self.raw_counts = expr

        if normalize:
            lib       = np.maximum(expr.sum(axis=1, keepdims=True), 1.0)
            self.x    = np.log1p(expr / lib * 1e4)
        else:
            self.x    = np.log1p(expr)

        self.n_samples = len(expr)
        self.n_genes   = expr.shape[1]

        print("=== ARCHS4Dataset ===")
        print(f"  Samples:  {self.n_samples:,}")
        print(f"  Genes:    {self.n_genes:,}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "x":     torch.from_numpy(self.x[idx]),
            "x_raw": torch.from_numpy(self.raw_counts[idx]),
        }


# ---------------------------------------------------------------------------
# PretrainCVAE — mirrors SpaceflightCVAE architecture exactly
# ---------------------------------------------------------------------------

class PretrainCVAE(nn.Module):
    """
    VAE whose encoder and decoder have identical layer shapes to
    SpaceflightCVAE. Condition slots are zero-padded during pretraining
    so all Linear layer weights transfer without any shape mismatch.

    Weight transfer map (pretrain → SpaceflightCVAE):
      encoder_net.* → encoder.net.*
      mu_head.*     → encoder.mu.*
      logvar_head.* → encoder.logvar.*
      decoder_net.* → decoder.net.*
      log_r_head.*  → decoder.log_r_head.*
      p_head.*      → decoder.p_head.*

    Condition embeddings (embedder.*) are NOT pretrained —
    they are randomly initialized during fine-tuning and learned
    from the GeneLab data.
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # encoder: input dim = 18907 genes + 76 condition zeros = 18983
        enc_layers = []
        in_dim = ENC_INPUT
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.encoder_net = nn.Sequential(*enc_layers)
        self.mu_head     = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)

        # decoder: input dim = 64 latent + 76 condition zeros = 140
        dec_layers = []
        in_dim = DEC_INPUT
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.decoder_net = nn.Sequential(*dec_layers)
        self.log_r_head  = nn.Linear(in_dim, N_GENES)
        self.p_head      = nn.Linear(in_dim, N_GENES)

        self.latent_dim = latent_dim
        self.cond_dim   = COND_DIM

    def encode(self, x):
        # pad condition slots with zeros — same input shape as SpaceflightCVAE
        zeros = torch.zeros(x.shape[0], self.cond_dim, device=x.device)
        h     = self.encoder_net(torch.cat([x, zeros], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        # pad condition slots with zeros — same input shape as SpaceflightCVAE
        zeros = torch.zeros(z.shape[0], self.cond_dim, device=z.device)
        h     = self.decoder_net(torch.cat([z, zeros], dim=-1))
        log_r = self.log_r_head(h)
        p     = torch.sigmoid(self.p_head(h))
        return log_r, p

    def forward(self, x):
        mu, log_var = self.encode(x)
        std  = torch.exp(0.5 * log_var)
        z    = mu + std * torch.randn_like(std)
        log_r, p = self.decode(z)
        return {"mu": mu, "log_var": log_var, "z": z,
                "log_r": log_r, "p": p}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, kl_weight, training=True):
    model.train(training)
    total_loss = total_recon = total_kl = 0.0
    n = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            x     = batch["x"].to(device)
            x_raw = batch["x_raw"].to(device)

            out     = model(x)
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
    dataset = ARCHS4Dataset(args.data)

    n_val   = int(len(dataset) * 0.05)
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

    model = PretrainCVAE(
        latent_dim=args.latent_dim,
        hidden_dims=[256, 128],
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Encoder input:    {ENC_INPUT} "
          f"({N_GENES} genes + {COND_DIM} condition zeros)")
    print(f"Decoder input:    {DEC_INPUT} "
          f"({args.latent_dim} latent + {COND_DIM} condition zeros)")
    print(f"Latent dim:       {args.latent_dim}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    annealer  = KLAnnealer(beta=args.beta, n_epochs=args.kl_anneal_epochs)

    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "pretrain_best.pt"

    best_val  = float("inf")
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
            best_val = va_loss
            patience_counter = 0
            torch.save({
                "epoch":      epoch,
                "val_loss":   va_loss,
                "model_state": model.state_dict(),
                "args":       vars(args),
                "n_genes":    N_GENES,
                "cond_dim":   COND_DIM,
                "latent_dim": args.latent_dim,
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
    print(f"  python finetune.py \\")
    print(f"      --data subset_final.h5 \\")
    print(f"      --pretrain_checkpoint {ckpt_path} \\")
    print(f"      --output_dir checkpoints_finetune_v5/ \\")
    print(f"      --lr 5e-5 --beta 0.005 --lambda_cls 2.0 \\")
    print(f"      --freeze_decoder_epochs 0 --patience 40 --dropout 0.3")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain cVAE on ARCHS4 with zero-padded conditions"
    )
    parser.add_argument("--data",             type=str,   required=True,
                        help="Path to archs4_pretrain.h5")
    parser.add_argument("--output_dir",       type=str,
                        default="checkpoints_pretrain_v5")
    parser.add_argument("--latent_dim",       type=int,   default=LATENT_DIM)
    parser.add_argument("--dropout",          type=float, default=0.2)
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--beta",             type=float, default=0.01)
    parser.add_argument("--kl_anneal_epochs", type=int,   default=30)
    parser.add_argument("--patience",         type=int,   default=10)
    args = parser.parse_args()
    pretrain(args)
