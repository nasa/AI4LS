"""
Dataset & DataLoader for Spaceflight cVAE (updated to support canonical metadata categories)
See original header in your file for dataset layout notes.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from typing import Optional, Dict, List


def _decode(arr: np.ndarray) -> np.ndarray:
    """Decode byte-string numpy array to UTF-8 strings."""
    return np.array([
        v.decode("utf-8").strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()
        for v in arr
    ])


def _load_optional(group, key, n_samples, default="Unknown"):
    """
    Load a metadata field from H5 if it exists, otherwise return
    an array of `default` values with length n_samples.
    """
    if key in group:
        return _decode(group[key][:])
    return np.array([default] * n_samples)


class _SimpleEncoder:
    """Lightweight encoder that exposes a classes_ attribute for compatibility."""
    def __init__(self, classes: List[str]):
        self.classes_ = np.array(classes, dtype=object)


class SpaceflightDataset(Dataset):
    """
    PyTorch Dataset for GeneLab bulk RNA-seq.

    New parameter:
        target_metadata_categories: optional dict mapping
            condition name -> list of canonical categories (strings).
        If provided, dataset will map its metadata into these canonical categories
        and reserve index 0 for 'Unknown' (missing / unseen values).
    """

    def __init__(
        self,
        h5_path: str,
        normalize: bool = True,
        target_metadata_categories: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()

        with h5py.File(h5_path, "r") as f:
            # Expression: (genes, samples) -> (samples, genes)
            expr = f["data/expression"][:].T.astype(np.float32)
            n = expr.shape[0]

            # Gene metadata
            self.ensembl_ids = _decode(f["meta/genes/ensembl_id"][:])
            self.gene_symbols = _decode(f["meta/genes/symbol"][:])

            # Sample metadata — all optional
            self.flight = f["meta/samples/spaceflight"][:].astype(np.int64) \
                if "meta/samples/spaceflight" in f \
                else np.zeros(n, dtype=np.int64)

            # handle missing meta/samples group robustly
            if "meta/samples" in f:
                s = f["meta/samples"]
            else:
                s = {}

            # try common key variants for study
            if isinstance(s, dict) or "study_id" not in s and "study" not in s:
                # s may be an empty dict if meta/samples missing
                study_raw = np.array(["Unknown"] * n)
            else:
                if "study_id" in s:
                    study_raw = _decode(s["study_id"][:])
                elif "study" in s:
                    study_raw = _decode(s["study"][:])
                else:
                    study_raw = np.array(["Unknown"] * n)

            # try common key variants for euthanasia
            if isinstance(s, dict) or ("euthanasia" not in s and "euth" not in s):
                euth_raw = np.array(["Unknown"] * n)
            else:
                if "euthanasia" in s:
                    euth_raw = _decode(s["euthanasia"][:])
                elif "euth" in s:
                    euth_raw = _decode(s["euth"][:])
                else:
                    euth_raw = np.array(["Unknown"] * n)

            # optional fields (strain/sex/tissue)
            if isinstance(s, dict):
                # meta/samples missing entirely -> default arrays
                strain_raw = np.array(["Unknown"] * n)
                sex_raw = np.array(["Unknown"] * n)
                tissue_raw = np.array(["Unknown"] * n)
            else:
                strain_raw = _load_optional(s, "strain", n)
                sex_raw = _load_optional(s, "sex", n)
                tissue_raw = _load_optional(s, "tissue", n)

        # Helper to build canonical classes + mapping
        def _build_classes_and_map(raw_arr, target_list: Optional[List[str]]):
            """
            Returns (classes_list, mapping_dict)
            classes_list always begins with 'Unknown' at index 0.
            """
            if target_list is not None and len(target_list) > 0:
                # canonical list provided by special/fine-tune dataset
                classes = ["Unknown"] + [str(v) for v in target_list]
            else:
                # infer from this file's raw values (drop NA-like entries)
                vals = sorted({str(v) for v in raw_arr if v is not None and str(v).strip().lower() != "nan"})
                classes = ["Unknown"] + vals
            mapping = {c: i for i, c in enumerate(classes)}
            return classes, mapping

        tmc = target_metadata_categories or {}

        # Build classes + mappings for each condition
        strain_classes, strain_map = _build_classes_and_map(strain_raw, tmc.get("strain"))
        sex_classes, sex_map = _build_classes_and_map(sex_raw, tmc.get("sex"))
        study_classes, study_map = _build_classes_and_map(study_raw, tmc.get("study") or tmc.get("study_id"))
        tissue_classes, tissue_map = _build_classes_and_map(tissue_raw, tmc.get("tissue"))
        euth_classes, euth_map = _build_classes_and_map(euth_raw, tmc.get("euth") or tmc.get("euthanasia"))

        # Encoders (compatibility — expose classes_ attr)
        self.strain_enc = _SimpleEncoder(strain_classes)
        self.sex_enc = _SimpleEncoder(sex_classes)
        self.study_enc = _SimpleEncoder(study_classes)
        self.tissue_enc = _SimpleEncoder(tissue_classes)
        self.euth_enc = _SimpleEncoder(euth_classes)

        # Transform raw values -> integer ids (unknown/missing -> 0)
        def _encode_array(raw_arr, mapping):
            out = np.zeros(len(raw_arr), dtype=np.int64)
            for i, v in enumerate(raw_arr):
                if v is None:
                    out[i] = 0
                else:
                    s = str(v)
                    out[i] = mapping.get(s, 0)
            return out

        self.strain_ids = _encode_array(strain_raw, strain_map)
        self.sex_ids = _encode_array(sex_raw, sex_map)
        self.study_ids = _encode_array(study_raw, study_map)
        self.tissue_ids = _encode_array(tissue_raw, tissue_map)
        self.euth_ids = _encode_array(euth_raw, euth_map)

        # Raw counts kept for NB loss
        self.raw_counts = expr

        # Encoder input: log1p of library-size-normalized counts
        if normalize:
            lib_sizes = expr.sum(axis=1, keepdims=True)
            lib_sizes = np.maximum(lib_sizes, 1.0)
            self.x = np.log1p(expr / lib_sizes * 1e4)
        else:
            self.x = np.log1p(expr)

        # Dimensions
        self.n_samples = len(self.flight)
        self.n_genes = expr.shape[1]
        self.n_strains = len(self.strain_enc.classes_)
        self.n_sexes = len(self.sex_enc.classes_)
        self.n_studies = len(self.study_enc.classes_)
        self.n_tissues = len(self.tissue_enc.classes_)
        self.n_euths = len(self.euth_enc.classes_)

        self._print_summary()

    def _print_summary(self):
        print("=== SpaceflightDataset ===")
        print(f"  Samples:     {self.n_samples}")
        print(f"  Genes:       {self.n_genes}")
        print(f"  Tissues:     {self.n_tissues} — {list(self.tissue_enc.classes_)}")
        print(f"  Strains:     {self.n_strains} — {list(self.strain_enc.classes_)}")
        print(f"  Sexes:       {self.n_sexes}   — {list(self.sex_enc.classes_)}")
        print(f"  Euthanasia:  {self.n_euths}   — {list(self.euth_enc.classes_)}")
        print(f"  Studies:     {self.n_studies}")
        print(f"  Spaceflight: {(self.flight==1).sum()} flight / "
              f"{(self.flight==0).sum()} ground")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "x":      torch.from_numpy(self.x[idx]),
            "x_raw":  torch.from_numpy(self.raw_counts[idx]),
            "strain": torch.tensor(self.strain_ids[idx], dtype=torch.long),
            "sex":    torch.tensor(self.sex_ids[idx],    dtype=torch.long),
            "study":  torch.tensor(self.study_ids[idx],  dtype=torch.long),
            "tissue": torch.tensor(self.tissue_ids[idx], dtype=torch.long),
            "euth":   torch.tensor(self.euth_ids[idx],   dtype=torch.long),
            "flight": torch.tensor(self.flight[idx],     dtype=torch.long),
        }

    def split(self, val_frac=0.15, test_frac=0.15, random_state=42):
        """Stratified train/val/test split on spaceflight x study_id."""
        strat_key = [f"{f}_{s}" for f, s in zip(self.flight, self.study_ids)]
        indices = np.arange(self.n_samples)

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=random_state
        )
        train_val_idx, test_idx = next(splitter.split(indices, strat_key))

        strat_key_tv = [strat_key[i] for i in train_val_idx]
        val_frac_adjusted = val_frac / (1 - test_frac)
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac_adjusted, random_state=random_state
        )
        train_rel_idx, val_rel_idx = next(
            splitter2.split(train_val_idx, strat_key_tv)
        )
        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]

        print(f"Split: {len(train_idx)} train / {len(val_idx)} val / "
              f"{len(test_idx)} test")
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)

    def kfold(self, n_splits=5, random_state=42):
        """Stratified k-fold cross-validation."""
        strat_key = [f"{f}_{s}" for f, s in zip(self.flight, self.study_ids)]
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
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
    """Split dataset and return train/val/test DataLoaders."""
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
