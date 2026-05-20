"""
Dataset & DataLoader for GeneLab Spaceflight Bulk RNA-seq
=========================================================
Expects a pandas DataFrame with:
  - Gene columns:        raw integer count columns (one per gene)
  - 'tissue':            tissue type string
  - 'strain':            mouse strain string
  - 'study_id':          GeneLab study accession (e.g. 'GLDS-4')
  - 'spaceflight':       0 (ground control) or 1 (spaceflight)
  - 'duration_days':     float, mission duration in days (0.0 for ground)

Example usage:
    df = pd.read_csv("genelab_samples.csv")
    gene_cols = [c for c in df.columns if c.startswith("ENSM")]

    dataset = SpaceflightDataset(df, gene_cols)
    train_ds, val_ds, test_ds = dataset.split(val_frac=0.15, test_frac=0.15)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


class SpaceflightDataset(Dataset):
    """
    PyTorch Dataset for GeneLab bulk RNA-seq samples.

    Handles:
      - Label encoding of categorical metadata
      - log1p transformation of counts (stored separately from raw)
      - Train/val/test splitting stratified by spaceflight + study

    Args:
        df:           DataFrame with samples as rows
        gene_cols:    list of column names corresponding to gene counts
        normalize:    if True, also compute library-size-normalized log1p counts
                      for the encoder input (raw counts always kept for NB loss)
    """
    def __init__(self, df: pd.DataFrame, gene_cols: list, normalize: bool = True):
        self.gene_cols = gene_cols
        self.df        = df.reset_index(drop=True)

        # --- Encode categoricals ---
        self.tissue_enc = LabelEncoder().fit(df["tissue"])
        self.strain_enc = LabelEncoder().fit(df["strain"])
        self.study_enc  = LabelEncoder().fit(df["study_id"])

        self.tissue_ids  = self.tissue_enc.transform(df["tissue"])
        self.strain_ids  = self.strain_enc.transform(df["strain"])
        self.study_ids   = self.study_enc.transform(df["study_id"])
        self.flight      = df["spaceflight"].values.astype(np.int64)
        self.duration    = df["duration_days"].values.astype(np.float32)

        # --- Raw counts (integer, for NB loss) ---
        self.raw_counts = df[gene_cols].values.astype(np.float32)

        # --- Encoder input: log1p of library-size-normalized counts ---
        if normalize:
            lib_sizes = self.raw_counts.sum(axis=1, keepdims=True)
            lib_sizes = np.maximum(lib_sizes, 1.0)
            normalized = self.raw_counts / lib_sizes * 1e4         # CPM-like
            self.x = np.log1p(normalized).astype(np.float32)
        else:
            self.x = np.log1p(self.raw_counts).astype(np.float32)

        # Useful metadata for downstream analysis
        self.n_tissues  = len(self.tissue_enc.classes_)
        self.n_strains  = len(self.strain_enc.classes_)
        self.n_studies  = len(self.study_enc.classes_)
        self.n_genes    = len(gene_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x":         torch.from_numpy(self.x[idx]),
            "x_raw":     torch.from_numpy(self.raw_counts[idx]),
            "tissue":    torch.tensor(self.tissue_ids[idx],  dtype=torch.long),
            "strain":    torch.tensor(self.strain_ids[idx],  dtype=torch.long),
            "study":     torch.tensor(self.study_ids[idx],   dtype=torch.long),
            "flight":    torch.tensor(self.flight[idx],      dtype=torch.long),
            "duration":  torch.tensor(self.duration[idx],    dtype=torch.float),
        }

    def split(
        self,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        random_state: int = 42,
    ):
        """
        Stratified train/val/test split.
        Stratification key = spaceflight × study_id to ensure
        all conditions and studies appear in every split.

        Returns:
            train_ds, val_ds, test_ds  (Subset objects)
        """
        # Combine flight + study for stratification
        strat_key = [
            f"{f}_{s}" for f, s in zip(self.flight, self.study_ids)
        ]
        indices = np.arange(len(self))

        # First: carve out test set
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=random_state
        )
        train_val_idx, test_idx = next(splitter.split(indices, strat_key))

        # Second: split remaining into train/val
        strat_key_tv = [strat_key[i] for i in train_val_idx]
        val_frac_adjusted = val_frac / (1 - test_frac)
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac_adjusted, random_state=random_state
        )
        train_rel_idx, val_rel_idx = next(
            splitter2.split(train_val_idx, strat_key_tv)
        )
        train_idx = train_val_idx[train_rel_idx]
        val_idx   = train_val_idx[val_rel_idx]

        return (
            Subset(self, train_idx),
            Subset(self, val_idx),
            Subset(self, test_idx),
        )


def make_dataloaders(
    dataset: SpaceflightDataset,
    batch_size: int = 64,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_workers: int = 4,
    random_state: int = 42,
):
    """
    Convenience function: split dataset and return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds, val_ds, test_ds = dataset.split(
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,      # keeps BatchNorm stable
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
