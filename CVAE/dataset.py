"""
Dataset & DataLoader for Spaceflight cVAE
==========================================
Loads from subset_final.h5 with structure:

    data/expression          (21970, 2080)  float  — genes x samples
    meta/genes/ensembl_id    (21970,)       bytes
    meta/genes/symbol        (21970,)       bytes
    meta/samples/spaceflight (2080,)        int8
    meta/samples/strain      (2080,)        bytes
    meta/samples/sex         (2080,)        bytes
    meta/samples/study_id    (2080,)        bytes
    meta/samples/tissue      (2080,)        bytes
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def _decode(arr: np.ndarray) -> np.ndarray:
    """Decode byte-string numpy array to UTF-8 strings."""
    return np.array([
        v.decode("utf-8").strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()
        for v in arr
    ])


class SpaceflightDataset(Dataset):
    """
    PyTorch Dataset for GeneLab bulk RNA-seq from subset_final.h5.

    Expression matrix stored genes x samples in H5; transposed to
    samples x genes internally.

    Args:
        h5_path:   path to subset_final.h5
        normalize: library-size normalize before log1p if True
    """

    def __init__(self, h5_path: str, normalize: bool = True):
        super().__init__()

        with h5py.File(h5_path, "r") as f:
            # Expression: (genes, samples) -> (samples, genes)
            expr = f["data/expression"][:].T.astype(np.float32)

            # Gene metadata
            self.ensembl_ids  = _decode(f["meta/genes/ensembl_id"][:])
            self.gene_symbols = _decode(f["meta/genes/symbol"][:])

            # Sample metadata
            self.flight = f["meta/samples/spaceflight"][:].astype(np.int64)
            strain_raw  = _decode(f["meta/samples/strain"][:])
            sex_raw     = _decode(f["meta/samples/sex"][:])
            study_raw   = _decode(f["meta/samples/study_id"][:])
            tissue_raw  = _decode(f["meta/samples/tissue"][:])

        # Label encode all categoricals
        self.strain_enc = LabelEncoder().fit(strain_raw)
        self.sex_enc    = LabelEncoder().fit(sex_raw)
        self.study_enc  = LabelEncoder().fit(study_raw)
        self.tissue_enc = LabelEncoder().fit(tissue_raw)

        self.strain_ids = self.strain_enc.transform(strain_raw).astype(np.int64)
        self.sex_ids    = self.sex_enc.transform(sex_raw).astype(np.int64)
        self.study_ids  = self.study_enc.transform(study_raw).astype(np.int64)
        self.tissue_ids = self.tissue_enc.transform(tissue_raw).astype(np.int64)

        # Raw counts kept for NB loss
        self.raw_counts = expr

        # Encoder input: log1p of library-size-normalized counts
        if normalize:
            lib_sizes  = expr.sum(axis=1, keepdims=True)
            lib_sizes  = np.maximum(lib_sizes, 1.0)
            normalized = expr / lib_sizes * 1e4
            self.x     = np.log1p(normalized)
        else:
            self.x = np.log1p(expr)

        # Dimensions
        self.n_samples = len(self.flight)
        self.n_genes   = expr.shape[1]
        self.n_strains = len(self.strain_enc.classes_)
        self.n_sexes   = len(self.sex_enc.classes_)
        self.n_studies = len(self.study_enc.classes_)
        self.n_tissues = len(self.tissue_enc.classes_)

        self._print_summary()

    def _print_summary(self):
        print("=== SpaceflightDataset ===")
        print(f"  Samples:     {self.n_samples}")
        print(f"  Genes:       {self.n_genes}")
        print(f"  Tissues:     {self.n_tissues} — {list(self.tissue_enc.classes_)}")
        print(f"  Strains:     {self.n_strains} — {list(self.strain_enc.classes_)}")
        print(f"  Sexes:       {self.n_sexes}   — {list(self.sex_enc.classes_)}")
        print(f"  Studies:     {self.n_studies}")
        print(f"  Spaceflight: {(self.flight==1).sum()} flight / {(self.flight==0).sum()} ground")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "x":       torch.from_numpy(self.x[idx]),
            "x_raw":   torch.from_numpy(self.raw_counts[idx]),
            "strain":  torch.tensor(self.strain_ids[idx], dtype=torch.long),
            "sex":     torch.tensor(self.sex_ids[idx],    dtype=torch.long),
            "study":   torch.tensor(self.study_ids[idx],  dtype=torch.long),
            "tissue":  torch.tensor(self.tissue_ids[idx], dtype=torch.long),
            "flight":  torch.tensor(self.flight[idx],     dtype=torch.long),
        }

    def split(self, val_frac=0.15, test_frac=0.15, random_state=42):
        """
        Stratified train/val/test split on spaceflight x study_id.

        Returns:
            train_ds, val_ds, test_ds
        """
        strat_key = [f"{f}_{s}" for f, s in zip(self.flight, self.study_ids)]
        indices   = np.arange(self.n_samples)

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=random_state
        )
        train_val_idx, test_idx = next(splitter.split(indices, strat_key))

        strat_key_tv      = [strat_key[i] for i in train_val_idx]
        val_frac_adjusted = val_frac / (1 - test_frac)
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac_adjusted, random_state=random_state
        )
        train_rel_idx, val_rel_idx = next(
            splitter2.split(train_val_idx, strat_key_tv)
        )
        train_idx = train_val_idx[train_rel_idx]
        val_idx   = train_val_idx[val_rel_idx]

        print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)

    def kfold(self, n_splits=5, random_state=42):
        """
        Stratified k-fold cross-validation.

        Yields:
            fold_idx, train_ds, val_ds
        """
        strat_key = [f"{f}_{s}" for f, s in zip(self.flight, self.study_ids)]
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold_idx, (train_idx, val_idx) in enumerate(
            kf.split(np.arange(self.n_samples), strat_key)
        ):
            print(f"Fold {fold_idx+1}: {len(train_idx)} train / {len(val_idx)} val")
            yield fold_idx, Subset(self, train_idx), Subset(self, val_idx)


def make_dataloaders(
    dataset: SpaceflightDataset,
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    num_workers: int = 4,
    random_state: int = 42,
):
    """
    Split dataset and return train/val/test DataLoaders.
    """
    train_ds, val_ds, test_ds = dataset.split(
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
