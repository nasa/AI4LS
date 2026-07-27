"""
Pretrain cVAE on ARCHS4 Mouse Bulk RNA-seq (updated to save label encoders)
See original header in your script for more usage notes.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from dataset_v2 import SpaceflightDataset
from model import SpaceflightCVAE, VALID_CONDITIONS
from losses import kl_divergence, nb_nll_loss, KLAnnealer

# small helpers to load metadata from h5 and build target category lists
import pandas as pd
import h5py
import os
from typing import Dict, List, Optional


def read_metadata_h5(path: str, key: Optional[str] = None) -> pd.DataFrame:
    """
    Try to read sample metadata from a variety of H5 layouts.
    Returns a pandas DataFrame with shape (n_samples, n_columns).
    """
    if key is None:
        candidate_keys = ['/meta/samples', 'meta/samples', '/meta', 'meta', '/samples', 'samples', '/obs', 'obs', '/metadata', 'metadata']
    else:
        candidate_keys = [key]

    # 1) Try pandas HDFStore style keys (pandas.read_hdf)
    for k in candidate_keys:
        try:
            df = pd.read_hdf(path, key=k)
            if isinstance(df, pd.DataFrame):
                return df
        except Exception:
            pass

    # 2) Heuristic read via h5py
    try:
        with h5py.File(path, 'r') as f:
            # check for nested groups where each child is a column
            for name in ['meta', 'metadata', 'samples', 'obs', 'annotations']:
                if name in f:
                    grp = f[name]
                    # If there's a child 'samples' or 'samples/tissue', try deeper search
                    # Build dict of arrays from group's children, if any
                    if isinstance(grp, h5py.Group):
                        data = {}
                        # prefer 'samples' or 'samples/*'
                        for k2 in grp.keys():
                            try:
                                arr = grp[k2][()]
                                data[k2] = arr
                            except Exception:
                                pass
                        if data:
                            # cast bytes to str for display-friendly df
                            for k2, v in data.items():
                                if isinstance(v, np.ndarray) and v.dtype.kind in ('S', 'O'):
                                    try:
                                        data[k2] = v.astype(str)
                                    except Exception:
                                        pass
                            return pd.DataFrame(data)

            # fallback: try root-level datasets that are 1D arrays with length matching samples
            candidates = {}
            for name in f.keys():
                node = f[name]
                if isinstance(node, h5py.Dataset):
                    try:
                        arr = node[()]
                        if isinstance(arr, (list, tuple, np.ndarray)):
                            # 1D or structured
                            if getattr(arr, 'ndim', 1) == 1 or (hasattr(arr, 'dtype') and arr.dtype.names):
                                candidates[name] = arr
                    except Exception:
                        pass
            # If any candidates are structured dtypes, convert to df
            for name, arr in candidates.items():
                if hasattr(arr, 'dtype') and arr.dtype.names:
                    data = {n: arr[n].astype('O') for n in arr.dtype.names}
                    return pd.DataFrame(data)
            # as a last resort, build dataframe with each candidate as a column (best-effort)
            if candidates:
                data = {}
                for k2, arr in candidates.items():
                    try:
                        if isinstance(arr, np.ndarray) and arr.dtype.kind in ('S', 'O'):
                            data[k2] = arr.astype(str)
                        else:
                            data[k2] = arr
                    except Exception:
                        pass
                return pd.DataFrame(data)
    except Exception:
        pass

    raise ValueError(f"Could not extract metadata DataFrame from {path} — pass an explicit key or examine file layout.")


def build_target_category_lists(special_meta: pd.DataFrame, conditions: List[str]) -> Dict[str, List[str]]:
    """
    For each condition name in conditions, build a list of categories from special_meta.
    These lists are used as the canonical category lists. Mapping will set unknown->index 0.
    """
    target = {}
    for cond in conditions:
        if cond in special_meta.columns:
            # take unique non-null strings
            vals = special_meta[cond].dropna().unique().tolist()
            # ensure stable ordering (sort) for deterministic behaviour
            vals = sorted([str(v) for v in vals])
            target[cond] = vals
        else:
            # if the special metadata lacks this column, treat as empty list
            target[cond] = []
    return target


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

    # If a special/fine-tune H5 is provided, read its metadata and build canonical category lists
    target_category_lists = None
    if args.special_data:
        if not os.path.exists(args.special_data):
            raise FileNotFoundError(args.special_data)
        print("Reading special/fine-tune metadata to build target categories...")
        special_meta = read_metadata_h5(args.special_data)
        target_category_lists = build_target_category_lists(special_meta, args.conditions)
        print("Target categories (sample):")
        for k, v in target_category_lists.items():
            print(f"  {k}: {len(v)} categories")

    # Pass the target_category_lists into SpaceflightDataset so public metadata is mapped to the same indices.
    dataset = SpaceflightDataset(args.data, target_metadata_categories=target_category_lists)

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

            # Build label encoders/classes dict for saving so finetune can remap embeddings by name
            label_encoders = {
                "strain": list(getattr(dataset.strain_enc, "classes_", [])),
                "sex":    list(getattr(dataset.sex_enc,    "classes_", [])),
                "study":  list(getattr(dataset.study_enc,  "classes_", [])),
                "tissue": list(getattr(dataset.tissue_enc, "classes_", [])),
                "euth":   list(getattr(dataset.euth_enc,   "classes_", [])),
            }

            torch.save({
                "epoch":          epoch,
                "val_loss":       va_loss,
                "model_state":    model.state_dict(),
                "args":           vars(args),
                "n_genes":        dataset.n_genes,
                "latent_dim":     args.latent_dim,
                "cond_dim":       model.cond_dim,
                "conditions":     model.conditions,
                "label_encoders": label_encoders,
                "target_metadata_categories": target_category_lists,
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
    parser.add_argument("--special_data",     type=str,   default=None,
                        help="Path to special/fine-tune H5 to extract canonical metadata categories (optional)")
    parser.add_argument("--output_dir",       type=str,   required=True,
                        help="Output directory for checkpoint")
    parser.add_argument("--conditions",       type=str,   nargs="+",
                        default=['tissue', 'strain', 'sex', 'study', 'euth'],
                        choices=VALID_CONDITIONS,
                        help="Conditions to embed (default: [])")
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
    parser.add_argument("--latent_dim",       type=int,   default=64)
    parser.add_argument("--dropout",          type=float, default=0.2)
    parser.add_argument("--epochs",           type=int,   default=500)
    parser.add_argument("--batch_size",       type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--beta",             type=float, default=0.01)
    parser.add_argument("--kl_anneal_epochs", type=int,   default=100)
    parser.add_argument("--patience",         type=int,   default=40)
    args = parser.parse_args()
    pretrain(args)
