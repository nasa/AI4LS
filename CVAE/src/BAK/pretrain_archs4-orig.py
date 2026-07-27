"""
Pretrain cVAE on ARCHS4 Mouse Bulk RNA-seq
============================================
Stage 1 of 2-stage transfer learning:
  - Pretrain on large ARCHS4 corpus (reconstruction + KL only)
  - No spaceflight labels — learns general mouse transcriptome prior
  - Saves pretrained weights for fine-tuning on GeneLab data

Stage 2 (fine-tuning) is done with train.py using --pretrain_checkpoint.

Usage:
    python pretrain_archs4.py \
        --data archs4_pretrain.h5 \
        --genelab subset_final.h5 \
        --output_dir checkpoints_pretrain/

Then fine-tune:
    python train.py \
        --data subset_final.h5 \
        --pretrain_checkpoint checkpoints_pretrain/best_model.pt \
        --output_dir checkpoints_finetune/ \
        --lr 1e-4 \
        --freeze_decoder_epochs 20
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
from sklearn.preprocessing import LabelEncoder

from losses import kl_divergence, nb_nll_loss, KLAnnealer
from model import SpaceflightCVAE
from dataset import SpaceflightDataset


# ---------------------------------------------------------------------------
# ARCHS4 Dataset
# ---------------------------------------------------------------------------

class ARCHS4Dataset(Dataset):
    """
    Loads archs4_pretrain.h5 produced by prepare_archs4.py.

    Matches the interface of SpaceflightDataset so DataLoaders
    are interchangeable. flight is always -1 (unknown) —
    the pretraining loss ignores classification.
    """

    def __init__(self, h5_path, normalize=True):
        super().__init__()

        def decode(val):
            if isinstance(val, (bytes, np.bytes_)):
                return val.decode("utf-8", errors="replace").strip()
            return str(val).strip()

        with h5py.File(h5_path, "r") as f:
            # expression: genes x samples -> samples x genes
            expr = f["data/expression"][:].T.astype(np.float32)

            self.ensembl_ids  = np.array([decode(v) for v in f["meta/genes/ensembl_id"][:]])
            self.gene_symbols = np.array([decode(v) for v in f["meta/genes/symbol"][:]])

        # raw counts for NB loss
        self.raw_counts = expr

        # normalized input: log1p(CPM)
        if normalize:
            lib_sizes  = expr.sum(axis=1, keepdims=True)
            lib_sizes  = np.maximum(lib_sizes, 1.0)
            self.x     = np.log1p(expr / lib_sizes * 1e4)
        else:
            self.x = np.log1p(expr)

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
# Pretrain model — reconstruction + KL only, no classification
# ---------------------------------------------------------------------------

class PretrainCVAE(nn.Module):
    """
    Unconditional VAE for pretraining on ARCHS4.
    No condition embeddings — pure expression input.
    Learns general mouse transcriptome structure (gene co-expression,
    count distributions) from 500K+ bulk RNA-seq samples.

    Encoder and decoder weights are transferred to SpaceflightCVAE
    for fine-tuning on GeneLab data.
    """

    def __init__(self, n_genes, latent_dim=64, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # encoder: expression → μ, log_var
        enc_layers = []
        in_dim = n_genes
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)
            ]
            in_dim = h
        self.encoder_net = nn.Sequential(*enc_layers)
        self.mu_head     = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)

        # decoder: z → NB params
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)
            ]
            in_dim = h
        self.decoder_net = nn.Sequential(*dec_layers)
        self.log_r_head  = nn.Linear(in_dim, n_genes)
        self.p_head      = nn.Linear(in_dim, n_genes)

        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder_net(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        h     = self.decoder_net(z)
        log_r = self.log_r_head(h)
        p     = torch.sigmoid(self.p_head(h))
        return log_r, p

    def forward(self, x):
        mu, log_var = self.encode(x)
        std  = torch.exp(0.5 * log_var)
        z    = mu + std * torch.randn_like(std)
        log_r, p = self.decode(z)
        return {"mu": mu, "log_var": log_var, "z": z, "log_r": log_r, "p": p}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_pretrain_epoch(model, loader, optimizer, device, kl_weight, training=True):
    model.train(training)
    total_loss = total_recon = total_kl = 0.0
    n = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            x     = batch["x"].to(device)
            x_raw = batch["x_raw"].to(device)

            outputs = model(x)
            l_recon = nb_nll_loss(x_raw, outputs["log_r"], outputs["p"])
            l_kl    = kl_divergence(outputs["mu"], outputs["log_var"])
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

    # --- Load ARCHS4 data ---
    print("\nLoading ARCHS4 dataset...")
    dataset = ARCHS4Dataset(args.data)

    # train/val split
    n_val   = int(len(dataset) * 0.05)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # --- Build pretrain model ---
    model = PretrainCVAE(
        n_genes=dataset.n_genes,
        latent_dim=args.latent_dim,
        hidden_dims=[256, 128],
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    annealer  = KLAnnealer(beta=args.beta, n_epochs=args.kl_anneal_epochs)

    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "pretrain_best.pt"

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        kl_w = annealer.get(epoch)

        train_loss, train_recon, train_kl = run_pretrain_epoch(
            model, train_loader, optimizer, device, kl_w, training=True
        )
        val_loss, val_recon, val_kl = run_pretrain_epoch(
            model, val_loader, None, device, kl_w, training=False
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.3f} (recon={train_recon:.3f} kl={train_kl:.2f}) | "
                f"val={val_loss:.3f} (recon={val_recon:.3f} kl={val_kl:.2f}) | "
                f"kl_w={kl_w:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "val_loss":    val_loss,
                "model_state": model.state_dict(),
                "args":        vars(args),
                "n_genes":     dataset.n_genes,
                "latent_dim":  args.latent_dim,
            }, ckpt_path)
            print(f"  ✓ Saved (val={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print("\nNext step — fine-tune on GeneLab:")
    print(f"  python train.py \\")
    print(f"      --data subset_final.h5 \\")
    print(f"      --pretrain_checkpoint {ckpt_path} \\")
    print(f"      --output_dir checkpoints_finetune/ \\")
    print(f"      --lr 1e-4")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain cVAE on ARCHS4 mouse bulk RNA-seq"
    )
    parser.add_argument("--data",             type=str, required=True,
                        help="Path to archs4_pretrain.h5")
    parser.add_argument("--genelab",          type=str, default=None,
                        help="Path to subset_final.h5 (optional, for gene set reference)")
    parser.add_argument("--output_dir",       type=str, default="checkpoints_pretrain")
    parser.add_argument("--latent_dim",       type=int,   default=64)
    parser.add_argument("--dropout",          type=float, default=0.2)
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--beta",             type=float, default=0.01)
    parser.add_argument("--kl_anneal_epochs", type=int,   default=30)
    parser.add_argument("--patience",         type=int,   default=10)
    args = parser.parse_args()
    pretrain(args)
