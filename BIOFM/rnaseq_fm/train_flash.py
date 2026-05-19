# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ExpressionBERT: Masked gene expression prediction using Transformer.

Adapted from Google's SLiMPerformer for continuous gene expression data.
Architecture:
  - Gene identity embedding (learned, like BERT token IDs)
  - Rotary Expression Embedding (REE) for value-based positional encoding
  - N Transformer layers (multi-head attention + FFN + LayerNorm)
  - Output projection for per-gene expression reconstruction

Training objective: MLM-style masking (mask 15% of genes, predict their expression)

Usage:
  torchrun --nproc_per_node=2 train.py
"""

import os
import sys

# Force unbuffered output for DDP visibility (safe for all ranks)
try:
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
except Exception as e:
    print(f"[WARN] Could not set unbuffered output: {e}", file=sys.stderr)

import time
import json
import math
from pathlib import Path
from collections import OrderedDict
import bisect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler, get_worker_info
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
import pandas as pd
import pyarrow.parquet as pq

from slim_performer_model import SLiMPerformerLayer

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    'hidden_dim': 512,
    'ffn_dim': 2048,
    'num_heads': 8,
    'num_layers': 12,
    'ree_base': 100.0,
    'feature_type': 'flash',       # Attention type: 'relu', 'elu+1', 'sqr', 'favor+', 'flash'
    'compute_type': 'iter',      # Prefix sum method: 'iter', 'ps', 'parallel_ps'
    'normalization': 'log1p_tpm',      # 'tpm' or 'log1p_tpm' applied before REE/model input
    'mask_ratio': 0.15,
    'mask_token': -10,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    # Linear warmup fraction of total training steps.
    'warmup_ratio': 0.05,
    # Global gradient norm clipping (None or <=0 disables clipping).
    'grad_clip_norm': 1.0,
    # Mixed precision training: can be toggled off for older/unsupported GPUs.
    'use_amp': True,
    # 'auto' picks bf16 on supported GPUs, else fp16.
    # Valid: 'auto', 'bf16', 'fp16'
    'amp_dtype': 'bf16',
    'batch_size': 4,
    'epochs': 20,
    'early_stopping': True,
    'patience': 5,
    'seed': 42,
    # Data loading mode: 'preload' (load arrays into RAM) or 'streaming' (on-the-fly parquet reads)
    'data_mode': 'preload',
    # Streaming DDP sharding mode:
    # False = sample-level DistributedSampler (max data usage, no min-batch truncation)
    # True  = file-level split by rank (better locality, may require min-batch truncation)
    'stream_ddp_file_split': True,
    'stream_cache_size': 4,  # Number of row groups to cache in memory per worker
    'num_workers': 4,
    'prefetch_factor': 2,
    'persistent_workers': True,
    # Data subset sizes (set to None for all available)
    'train_subset': None,
    'val_subset': None,
    'balanced_sampling': True,
    # If True, only keep shards where human/mouse counts are exactly equal
    # (or within balanced_shard_tolerance) before train/val sampling.
    'balanced_shards_only': True,
    'balanced_shard_tolerance': 0,
    # Optional cap on balanced shards retained after filtering.
    # Example: 40 keeps ~800k mixed samples (at 20k/shard) and is divisible by 4 GPUs.
    'max_balanced_shards': 40,
    'include_species_embedding': False,
    'data_dir': './data/archs4/train_orthologs_sharded',
    'checkpoint_dir': './checkpoints_performer',
    'progress_log_interval_sec': 60,
    'timing_profile': False,
    # If True, collect GPU kernel timings with CUDA events for fwd/bwd/step
    # This adds synchronization overhead but improves timing fidelity.
    'timing_cuda_events': True,
    # Resume policy: 'off' | 'path' | 'run_id'
    'resume_mode': 'off',
    'resume': False,
    'resume_path': None,
    'resume_run_id': None,
}


# ============================================================
# ROTARY EXPRESSION EMBEDDING (REE)
# ============================================================
class RotaryExpressionEmbedding(nn.Module):
    """
    Rotary Expression Embedding (REE): Converts continuous gene expression
    values into sinusoidal rotation features.

    Modulates rotary positional encodings using expression magnitude.
    Includes masking support for special tokens (e.g., masked expression = -10).
    Original base=100 (from Google SLiMPerformer research).
    """

    def __init__(self, dim, base=100.0, mask_token_id=-10):
        super().__init__()
        self.dim = dim
        self.mask_token_id = mask_token_id

        # inv_freq for sinusoidal encoding
        # base=100 (from original code) vs 10000 (standard Transformer)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_genes] expression values

        Returns:
            [batch_size, num_genes, dim] sinusoidal encodings
        """
        # Identify masked positions
        x_mask_idx = (x == self.mask_token_id).nonzero(as_tuple=False)

        # Multiply expression values by frequencies: [B, G] x [D/2] → [B, G, D/2]
        freqs = torch.einsum("bi,j->bij", x, self.inv_freq)

        # Apply sin and cos, then concatenate: [B, G, D/2] → [B, G, D]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)

        # Mask out special token positions (set to 0)
        if len(x_mask_idx) > 0:
            emb[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = 0

        return emb


class FlashTransformerLayer(nn.Module):
    """Transformer block using PyTorch scaled-dot-product attention (flash path when available)."""

    def __init__(self, hidden_dim, ffn_dim, n_heads, dropout=0.0):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, int(ffn_dim)),
            nn.GELU(),
            nn.Linear(int(ffn_dim), self.hidden_dim),
        )

    def forward(self, x):
        # Pre-norm attention block.
        h = self.norm1(x)
        B, G, _ = h.shape

        q = self.q_proj(h).view(B, G, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, G, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, G, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, G, self.hidden_dim)
        x = x + self.out_proj(attn)

        # Pre-norm FFN block.
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# EXPRESSION PERFORMER MODEL
# ============================================================
class ExpressionPerformer(nn.Module):
    """
    ExpressionBERT: Transformer for continuous gene expression data.
    Uses SLiMPerformer's linear attention (O(n) memory) from Google Research.

    Input:  [batch, num_genes] expression values (with masked positions = -10)
    Output: [batch, num_genes] predicted expression values

    Embeddings (summed, like BERT):
      1. Gene identity embedding — learned per-gene vector (like BERT token IDs)
      2. REE — sinusoidal encoding driven by expression magnitude
    """

    def __init__(self, num_genes, hidden_dim=256, n_heads=8, n_layers=4,
                 ffn_dim=1024, ree_base=100.0, mask_token_id=-10,
                 feature_type='sqr', compute_type='iter',
                 include_species_embedding=False, num_species=1):
        super().__init__()
        self.num_genes = num_genes
        self._hidden_dim = hidden_dim
        self.include_species_embedding = bool(include_species_embedding)

        # Gene identity embedding (like BERT's token embedding)
        self.gene_embedding = nn.Embedding(num_genes, hidden_dim)

        # Rotary Expression Embedding
        self.ree = RotaryExpressionEmbedding(hidden_dim, base=ree_base,
                                              mask_token_id=mask_token_id)

        if self.include_species_embedding:
            self.species_embedding = nn.Embedding(int(num_species), hidden_dim)

        self.feature_type = str(feature_type).strip().lower()
        self.use_flash = self.feature_type == 'flash'

        if self.use_flash:
            # FlashAttention-backed dense attention layers.
            self.layers = nn.ModuleList([
                FlashTransformerLayer(hidden_dim, ffn_dim, n_heads, dropout=0.0)
                for _ in range(n_layers)
            ])
        else:
            # SLiMPerformer layers (linear O(n) attention via prefix sums)
            self.layers = nn.ModuleList([
                SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads,
                                   feature_type, compute_type, on_gptln=True)
                for _ in range(n_layers)
            ])

        # Output: predict single expression value per gene
        self.output_map = nn.Linear(hidden_dim, 1)

    def forward(self, x, species_ids=None):
        """
        Args:
            x: [batch, num_genes] expression values
        Returns:
            [batch, num_genes] predicted expression
        """
        B, G = x.shape
        device = x.device

        # Gene identity embeddings: [G, hidden_dim] → broadcast to [B, G, hidden_dim]
        gene_ids = torch.arange(G, device=device)
        gene_emb = self.gene_embedding(gene_ids)

        # REE from expression values: [B, G, hidden_dim]
        ree_emb = self.ree(x)

        # Sum embeddings (like BERT: token + position)
        h = gene_emb.unsqueeze(0) + ree_emb

        if self.include_species_embedding:
            if species_ids is None:
                species_ids = torch.zeros(B, dtype=torch.long, device=device)
            species_emb = self.species_embedding(species_ids.long())
            h = h + species_emb.unsqueeze(1)

        if self.use_flash:
            for layer in self.layers:
                h = layer(h)
        else:
            # Pass through SLiMPerformer layers (linear attention)
            for layer in self.layers:
                rfs = layer.attention.sample_rfs(device)
                h = layer.full_forward(h, rfs)

        # Project to scalar per gene
        out = self.output_map(h).squeeze(-1)  # [B, G]

        return out


# ============================================================
# DATASET
# ============================================================
class ExpressionMLMDataset(Dataset):
    """Expression dataset with MLM-style masking."""

    def __init__(self, expr_array, species_ids=None, mask_ratio=0.15, mask_token=-10):
        self.X = expr_array.astype(np.float32)
        if species_ids is None:
            self.species_ids = np.zeros(len(self.X), dtype=np.int64)
        else:
            self.species_ids = np.asarray(species_ids, dtype=np.int64)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        num_genes = x.shape[0]

        num_mask = max(1, int(num_genes * self.mask_ratio))
        mask_indices = np.random.choice(num_genes, num_mask, replace=False)

        x_masked = x.copy()
        x_masked[mask_indices] = self.mask_token

        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(mask_indices, dtype=torch.long),
            torch.tensor(self.species_ids[idx], dtype=torch.long),
        )


class StreamingParquetMLMDataset(Dataset):
    """
    High-throughput parquet streaming dataset using PyArrow row-group reads.

    Performance features:
      - Avoids pandas/DataFrame conversion in the hot path.
      - Reads row groups instead of full files for sample access.
      - LRU cache stores decoded row-group arrays to reduce repeated I/O.
      - DDP-aware file sharding (files[rank::world_size]) to reduce contention.
    """

    def __init__(self, batch_dir, sample_indices, normalization='tpm', mask_ratio=0.15,
                 mask_token=-10, cache_size=16, rank=0, world_size=1,
                 ddp_file_split=True):
        self.batch_dir = Path(batch_dir)
        self.batch_files = sorted(self.batch_dir.glob("*.parquet"))
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.normalization = normalization
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
        self._base_cache_size = max(1, int(cache_size))
        self.cache_size = self._base_cache_size
        self._base_max_open_files = 8
        self._max_open_files = self._base_max_open_files
        self._cache = OrderedDict()  # (batch_idx, row_group_idx) -> pyarrow.Table
        self._process_pid = None
        self._worker_id = None
        self._worker_count = 1
        self._parquet_files = {}
        self._parquet_file_lru = OrderedDict()  # batch_idx -> None

        # Build row-group cumulative row starts for fast row lookup.
        self._row_group_starts = []
        first_pf = None
        for file_idx, batch_file in enumerate(self.batch_files):
            pf = pq.ParquetFile(str(batch_file))
            if file_idx == 0:
                first_pf = pf
            starts = [0]
            for rg in range(pf.metadata.num_row_groups):
                starts.append(starts[-1] + pf.metadata.row_group(rg).num_rows)
            self._row_group_starts.append(starts)

        sample_indices = list(sample_indices)
        self.ddp_file_split_active = False

        # DDP-aware file ownership: each rank gets a disjoint subset of files.
        if ddp_file_split and self.world_size > 1:
            if len(self.batch_files) >= self.world_size:
                my_files = list(range(len(self.batch_files)))[self.rank::self.world_size]
                my_file_set = set(my_files)
                self.sample_indices = [s for s in sample_indices if s[0] in my_file_set]
                self.ddp_file_split_active = True
            else:
                # Not enough parquet files for file-based rank partitioning.
                # Fall back to sample-level sharding in the DataLoader.
                if self.rank == 0:
                    print(
                        f"[DATA] Streaming file-shard fallback: {len(self.batch_files)} parquet files for world_size={self.world_size}. "
                        "Using sample-level DistributedSampler sharding.",
                        flush=True,
                    )
                self.sample_indices = sample_indices
        else:
            self.sample_indices = sample_indices

        # Precompute per-sample row-group metadata once to avoid repeated bisect work.
        # record = (batch_idx, row_group_idx, row_offset, species_id)
        self.records = []
        self.group_to_indices = {}
        for i, rec in enumerate(self.sample_indices):
            if len(rec) >= 3:
                batch_idx, sample_row, species_id = rec[:3]
            else:
                batch_idx, sample_row = rec
                species_id = 0
            rg_idx, rg_offset = self._locate_row_group(batch_idx, sample_row)
            self.records.append((batch_idx, rg_idx, rg_offset, int(species_id)))
            key = (batch_idx, rg_idx)
            self.group_to_indices.setdefault(key, []).append(i)

        # Keep only gene columns in the streaming hot path.
        first_schema_cols = first_pf.schema_arrow.names
        self._gene_columns = [c for c in first_schema_cols if c not in ('geo_accession', '__index_level_0__')]
        self.num_genes = len(self._gene_columns)
        self.num_mask = max(1, int(self.num_genes * self.mask_ratio))

    def __len__(self):
        return len(self.sample_indices)

    def _table_to_numpy(self, table):
        # Convert Arrow table [rows, genes] to float32 NumPy for the selected rows only.
        cols = [table.column(i).combine_chunks().to_numpy(zero_copy_only=False)
                for i in range(table.num_columns)]
        return np.column_stack(cols).astype(np.float32, copy=False)

    def _locate_row_group(self, batch_idx, sample_row):
        starts = self._row_group_starts[batch_idx]
        rg_idx = bisect.bisect_right(starts, sample_row) - 1
        rg_offset = sample_row - starts[rg_idx]
        return rg_idx, rg_offset

    def _ensure_process_state(self):
        current_pid = os.getpid()
        worker_info = get_worker_info()
        worker_id = int(worker_info.id) if worker_info is not None else -1
        worker_count = int(worker_info.num_workers) if worker_info is not None else 1

        if self._process_pid == current_pid and self._worker_id == worker_id:
            return

        self._process_pid = current_pid
        self._worker_id = worker_id
        self._worker_count = max(1, worker_count)
        self._cache = OrderedDict()
        self._parquet_files = {}
        self._parquet_file_lru = OrderedDict()

        # Keep aggregate cache memory bounded when using multiple workers.
        self.cache_size = max(1, self._base_cache_size // self._worker_count)
        self._max_open_files = max(
            2,
            min(len(self.batch_files), max(2, self._base_max_open_files // self._worker_count)),
        )

    def _get_parquet_file(self, batch_idx):
        self._ensure_process_state()
        pf = self._parquet_files.get(batch_idx)
        if pf is None:
            pf = pq.ParquetFile(str(self.batch_files[batch_idx]))
            self._parquet_files[batch_idx] = pf
        self._parquet_file_lru[batch_idx] = None
        self._parquet_file_lru.move_to_end(batch_idx)

        if len(self._parquet_file_lru) > self._max_open_files:
            evict_idx, _ = self._parquet_file_lru.popitem(last=False)
            self._parquet_files.pop(evict_idx, None)

        return pf

    def _get_row_group_table(self, batch_idx, row_group_idx):
        self._ensure_process_state()
        key = (batch_idx, row_group_idx)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        table = self._get_parquet_file(batch_idx).read_row_group(
            row_group_idx, columns=self._gene_columns, use_threads=True
        )
        self._cache[key] = table
        self._cache.move_to_end(key)

        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return table

    def __getitem__(self, idx):
        # Return lightweight record index; real data is loaded in collate_batch.
        return int(idx)

    def collate_batch(self, batch_record_indices):
        """
        Batch-level row-group loading.

        This reduces Python overhead and avoids repeatedly materializing row-group
        arrays per individual sample.
        """
        B = len(batch_record_indices)
        x_true = np.empty((B, self.num_genes), dtype=np.float32)
        species_ids = np.zeros((B,), dtype=np.int64)

        # Group requests by (file, row_group) to maximize locality.
        grouped = {}
        for out_i, rec_i in enumerate(batch_record_indices):
            batch_idx, rg_idx, rg_off, species_id = self.records[rec_i]
            grouped.setdefault((batch_idx, rg_idx), []).append((out_i, rg_off))
            species_ids[out_i] = species_id

        for (batch_idx, rg_idx), reqs in grouped.items():
            table = self._get_row_group_table(batch_idx, rg_idx)
            local_rows = [r for _, r in reqs]

            # Read only requested rows from this row-group.
            sub = table.take(np.array(local_rows, dtype=np.int64))
            sub_np = self._table_to_numpy(sub)
            for j, (out_i, _) in enumerate(reqs):
                x_true[out_i] = sub_np[j]

        if self.normalization == 'log1p_tpm':
            x_true = np.log1p(np.maximum(x_true, 0.0)).astype(np.float32, copy=False)

        mask_indices = np.empty((B, self.num_mask), dtype=np.int64)
        for i in range(B):
            mask_indices[i] = np.random.choice(self.num_genes, self.num_mask, replace=False)

        x_masked = x_true.copy()
        x_masked[np.arange(B)[:, None], mask_indices] = self.mask_token

        return (
            torch.from_numpy(x_masked),
            torch.from_numpy(x_true),
            torch.from_numpy(mask_indices),
            torch.from_numpy(species_ids),
        )


class RowGroupBatchSampler(Sampler):
    """Batch sampler that keeps batches within the same (file, row_group)."""

    def __init__(
        self,
        group_to_indices,
        batch_size,
        shuffle=True,
        seed=42,
        drop_last=False,
        target_num_batches=None,
    ):
        self.group_to_indices = group_to_indices
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.target_num_batches = (
            None if target_num_batches is None else max(0, int(target_num_batches))
        )
        self.epoch = 0

        total = 0
        for idxs in self.group_to_indices.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += (len(idxs) + self.batch_size - 1) // self.batch_size
        self._len = total

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        batches = self._build_batches()
        if self.target_num_batches is not None:
            batches = batches[: self.target_num_batches]
        for batch in batches:
            yield batch

    def _build_batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = list(self.group_to_indices.keys())
        if self.shuffle:
            rng.shuffle(keys)

        batches = []
        for k in keys:
            idxs = np.array(self.group_to_indices[k], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(idxs)

            n = len(idxs)
            stop = (n // self.batch_size) * self.batch_size if self.drop_last else n
            for start in range(0, stop, self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch.tolist())

        return batches

    def __len__(self):
        if self.target_num_batches is not None:
            return min(self._len, self.target_num_batches)
        return self._len


# ============================================================
# DATA LOADING
# ============================================================
def get_sample_indices(
    batch_dir,
    train_subset=None,
    val_subset=None,
    balanced_sampling=True,
    seed=42,
    verbose=True,
    balanced_shards_only=False,
    balanced_shard_tolerance=0,
    max_balanced_shards=None,
):
    """
    Build sample index lists for train/val without loading all data.
    
    Args:
        batch_dir: Path to batch parquet files
        train_subset: Exact number of train samples to select (None = use all available)
        val_subset: Exact number of val samples to select (None = use remaining after train)
        balanced_sampling: If True, balance human/mouse to min count
        seed: Random seed
        verbose: Print diagnostics
    
    Returns:
        (train_sample_indices, val_sample_indices, species_to_id)
    """
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.parquet"))
    
    if not batch_files:
        # Final debug before error
        print(f"[ERROR] No parquet files found in {batch_dir}")
        print(f"[DEBUG] Directory exists: {batch_dir.exists()}")
        print(f"[DEBUG] Is directory: {batch_dir.is_dir()}")
        if batch_dir.exists():
            print(f"[DEBUG] Files in directory: {list(batch_dir.iterdir())[:10]}")
        raise FileNotFoundError(f"No parquet files found in {batch_dir}")
    
    if verbose:
        print(f"[DEBUG] Found {len(batch_files)} batch files")
    
    # Load metadata to track species per sample.
    # Check metadata.csv first (geo_accession, species columns), then fall back
    # to samples.json with {"id", "species"} entries.  Search batch_dir and its
    # parent so we work regardless of whether batch files live in a sub-folder.
    sample_to_species = {}
    for search_dir in (batch_dir, batch_dir.parent):
        csv_path = search_dir / "metadata.csv"
        if csv_path.exists():
            import csv as _csv
            with open(csv_path, newline='') as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    sid = row.get("geo_accession") or row.get("id") or ""
                    sp  = row.get("species", "")
                    if sid and sp:
                        sample_to_species[sid] = sp
            break
        json_path = search_dir / "samples.json"
        if json_path.exists():
            with open(json_path) as f:
                samples_meta = json.load(f)
            if samples_meta and isinstance(samples_meta[0], dict):
                sample_to_species = {s["id"]: s["species"] for s in samples_meta if "species" in s}
            break
    
    rng = np.random.default_rng(seed)

    def _norm_species(sp):
        s = str(sp).strip().lower()
        if s.startswith('human'):
            return 'human'
        if s.startswith('mouse'):
            return 'mouse'
        return s

    preferred_species_order = ['human', 'mouse']
    species_to_id = {}
    for sp in preferred_species_order:
        if any(v == sp for v in sample_to_species.values()):
            species_to_id[sp] = len(species_to_id)
    
    # Build master list of all (batch_idx, sample_in_batch, species) tuples.
    # New preprocessing saves sample-major batch files, so sample IDs are parquet index.
    all_samples = []  # [(batch_idx, sample_in_batch, species), ...]

    manifest_path = batch_dir.parent / "batch_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            batch_manifest = json.load(f)

        # Prefer direct filename lookup; fallback to positional lists for legacy
        # manifests that use synthetic keys like batch_0001.parquet.
        ordered_manifest_lists = [batch_manifest[k] for k in sorted(batch_manifest.keys())]

        for batch_idx, batch_file in enumerate(batch_files):
            sample_ids = batch_manifest.get(batch_file.name)
            if sample_ids is None and batch_idx < len(ordered_manifest_lists):
                sample_ids = ordered_manifest_lists[batch_idx]
            if sample_ids is None:
                sample_ids = []
            for sample_idx, sample_id in enumerate(sample_ids):
                species = _norm_species(sample_to_species.get(sample_id, "unknown"))
                if species not in species_to_id:
                    species_to_id[species] = len(species_to_id)
                all_samples.append((batch_idx, sample_idx, species))

        # If manifest exists but produced no sample rows, fallback to parquet index.
        if not all_samples:
            for batch_idx, batch_file in enumerate(batch_files):
                pf = pq.ParquetFile(str(batch_file))
                cols = pf.schema_arrow.names
                idx_col = 'geo_accession' if 'geo_accession' in cols else (
                    '__index_level_0__' if '__index_level_0__' in cols else None
                )
                if idx_col is not None:
                    table = pf.read(columns=[idx_col], use_threads=True)
                    sample_ids = table.column(0).to_pylist()
                else:
                    # Gene-major parquet: column names are the sample IDs
                    non_meta = [c for c in cols if c != 'gene_symbol']
                    if non_meta and (non_meta[0] in sample_to_species or non_meta[0].startswith(('GSM', 'GSE'))):
                        sample_ids = non_meta
                    else:
                        sample_ids = [str(i) for i in range(pf.metadata.num_rows)]
                for sample_idx, sample_id in enumerate(sample_ids):
                    species = _norm_species(sample_to_species.get(sample_id, "unknown"))
                    if species not in species_to_id:
                        species_to_id[species] = len(species_to_id)
                    all_samples.append((batch_idx, sample_idx, species))
    else:
        # Fallback for legacy data without manifest: read index column via PyArrow.
        for batch_idx, batch_file in enumerate(batch_files):
            pf = pq.ParquetFile(str(batch_file))
            cols = pf.schema_arrow.names
            idx_col = 'geo_accession' if 'geo_accession' in cols else (
                '__index_level_0__' if '__index_level_0__' in cols else None
            )
            if idx_col is not None:
                table = pf.read(columns=[idx_col], use_threads=True)
                sample_ids = table.column(0).to_pylist()
            else:
                # Gene-major parquet: column names are the sample IDs
                non_meta = [c for c in cols if c != 'gene_symbol']
                if non_meta and (non_meta[0] in sample_to_species or non_meta[0].startswith(('GSM', 'GSE'))):
                    sample_ids = non_meta
                else:
                    sample_ids = [str(i) for i in range(pf.metadata.num_rows)]
            for sample_idx, sample_id in enumerate(sample_ids):
                species = _norm_species(sample_to_species.get(sample_id, "unknown"))
                if species not in species_to_id:
                    species_to_id[species] = len(species_to_id)
                all_samples.append((batch_idx, sample_idx, species))

    if balanced_shards_only and all_samples:
        tol = max(0, int(balanced_shard_tolerance))
        shard_species_counts = {}
        for batch_idx, _sample_idx, species in all_samples:
            if batch_idx not in shard_species_counts:
                shard_species_counts[batch_idx] = {'human': 0, 'mouse': 0}
            if species == 'human':
                shard_species_counts[batch_idx]['human'] += 1
            elif species == 'mouse':
                shard_species_counts[batch_idx]['mouse'] += 1

        keep_batch_idx = {
            bidx
            for bidx, counts in shard_species_counts.items()
            if counts['human'] > 0 and counts['mouse'] > 0 and abs(counts['human'] - counts['mouse']) <= tol
        }

        if max_balanced_shards is not None:
            max_keep = max(0, int(max_balanced_shards))
            if len(keep_batch_idx) > max_keep:
                keep_sorted = sorted(keep_batch_idx)
                keep_batch_idx = set(keep_sorted[:max_keep])

        before = len(all_samples)
        all_samples = [rec for rec in all_samples if rec[0] in keep_batch_idx]

        if verbose:
            print(
                f"[DATA] balanced_shards_only=True kept {len(keep_batch_idx):,}/{len(shard_species_counts):,} shards "
                f"(tolerance={tol})",
                flush=True,
            )
            if max_balanced_shards is not None:
                print(
                    f"[DATA] max_balanced_shards={int(max_balanced_shards):,}",
                    flush=True,
                )
            print(
                f"[DATA] balanced_shards_only samples kept: {len(all_samples):,}/{before:,}",
                flush=True,
            )
    
    if verbose:
        print(f"[DATA] Total samples available: {len(all_samples):,}", flush=True)
    
    # Separate by species
    samples_by_species = {}
    for batch_idx, sample_idx, species in all_samples:
        if species not in samples_by_species:
            samples_by_species[species] = []
        samples_by_species[species].append((batch_idx, sample_idx, species_to_id.get(species, 0)))

    # Convert species labels to integer IDs for optional species embedding lookup.
    all_samples = [(b, s, species_to_id.get(sp, 0)) for b, s, sp in all_samples]
    
    if verbose:
        for sp, samples in samples_by_species.items():
            print(f"       {sp}: {len(samples):,} samples", flush=True)
    
    # Apply balanced sampling and subsetting
    if balanced_sampling and len(samples_by_species) > 1:
        # Determine total requested before split.
        requested_total = None
        if train_subset is not None and val_subset is not None:
            requested_total = train_subset + val_subset
        elif train_subset is not None:
            requested_total = train_subset

        # Determine per-species limit.
        if requested_total is not None:
            per_species = requested_total // len(samples_by_species)
        else:
            # Use max fully balanced pool based on minority species.
            per_species = min(len(samples) for samples in samples_by_species.values())
        
        if verbose:
            print(f"       Balanced to {per_species:,} per species", flush=True)
        
        all_samples_balanced = []
        for species, samples in samples_by_species.items():
            if len(samples) > per_species:
                selected = rng.choice(len(samples), per_species, replace=False)
                all_samples_balanced.extend([samples[i] for i in selected])
            else:
                all_samples_balanced.extend(samples)
        all_samples = all_samples_balanced
    
    elif train_subset is not None:
        # No species balancing, just subsample
        if val_subset is not None:
            requested_total = train_subset + val_subset
        else:
            requested_total = train_subset
        requested_total = min(requested_total, len(all_samples))
        selected = rng.choice(len(all_samples), requested_total, replace=False)
        all_samples = [all_samples[i] for i in selected]

    # Final shuffled pool from which we take exact train/val counts.
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)
    shuffled = [all_samples[i] for i in indices]

    if train_subset is None:
        # Backward-compatible default when no explicit train size is provided.
        train_count = int(0.8 * len(shuffled))
    else:
        train_count = min(train_subset, len(shuffled))

    remaining = max(0, len(shuffled) - train_count)
    if val_subset is None:
        val_count = remaining
    else:
        val_count = min(val_subset, remaining)

    train_indices = shuffled[:train_count]
    val_indices = shuffled[train_count:train_count + val_count]
    
    if verbose:
        print(f"       Train: {len(train_indices):,} samples", flush=True)
        print(f"       Val:   {len(val_indices):,} samples", flush=True)
    
    return train_indices, val_indices, species_to_id


def load_batch_data(batch_dir, sample_indices, normalization='tpm', verbose=True):
    """
    Load selected samples from batch parquet files into a single numpy array.
    
    Args:
        batch_dir: Path to directory with batch parquet files
        sample_indices: List of (batch_idx, sample_idx_in_batch) tuples
        normalization: 'tpm' or 'log1p_tpm'
        verbose: Print progress
    
    Returns:
        numpy array of shape [num_samples, num_genes]
    """
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.parquet"))
    
    # Group samples by batch file for efficient loading
    from collections import defaultdict
    batch_to_samples = defaultdict(list)
    for idx, rec in enumerate(sample_indices):
        batch_idx, sample_in_batch = rec[:2]
        batch_to_samples[batch_idx].append((idx, sample_in_batch))
    
    # Detect parquet orientation: gene-major has sample IDs as column names.
    first_pf = pq.ParquetFile(str(batch_files[0]))
    first_cols = first_pf.schema_arrow.names
    _non_meta = [c for c in first_cols if c not in ('geo_accession', '__index_level_0__', 'gene_symbol')]
    _gene_major = bool(_non_meta and (_non_meta[0].startswith(('GSM', 'GSE'))))

    if _gene_major:
        # Gene-major: rows=genes, cols=samples. num_genes = num_rows.
        num_genes = first_pf.metadata.num_rows
        # Only need the row data — read all numeric columns (all except gene_symbol)
        _read_cols = [c for c in first_cols if c != 'gene_symbol']
    else:
        # Sample-major: rows=samples, cols=genes.
        _read_cols = [c for c in first_cols if c not in ('geo_accession', '__index_level_0__')]
        num_genes = len(_read_cols)

    result = np.empty((len(sample_indices), num_genes), dtype=np.float32)

    # Load batch-by-batch and gather selected samples.
    total_batches = len(batch_to_samples)
    for i, (batch_idx, idx_pairs) in enumerate(batch_to_samples.items(), start=1):
        pf_b = pq.ParquetFile(str(batch_files[batch_idx]))
        b_cols = pf_b.schema_arrow.names
        if _gene_major:
            read_cols = [c for c in b_cols if c != 'gene_symbol']
        else:
            read_cols = [c for c in b_cols if c not in ('geo_accession', '__index_level_0__')]
        table = pq.read_table(batch_files[batch_idx], columns=read_cols, use_threads=True)
        cols_np = [table.column(j).combine_chunks().to_numpy(zero_copy_only=False)
                   for j in range(table.num_columns)]
        data = np.stack(cols_np, axis=1).astype(np.float32, copy=False)
        if _gene_major:
            # data shape: [num_genes, num_samples_in_batch] — transpose to [samples, genes]
            data = data.T
        for out_idx, sample_in_batch in idx_pairs:
            result[out_idx] = data[sample_in_batch]

        if verbose and (i % 25 == 0 or i == total_batches):
            print(f"  ...loaded {i}/{total_batches} batch files", flush=True)
    
    # Apply normalization
    if normalization == 'log1p_tpm':
        result = np.log1p(np.maximum(result, 0.0)).astype(np.float32)
    
    if verbose:
        print(f"  ✓ Loaded {result.shape[0]:,} samples × {result.shape[1]:,} genes")
    
    return result


def get_num_genes_from_batches(batch_dir):
    """Infer number of genes from sample-major batch parquet shape."""
    batch_files = sorted(Path(batch_dir).glob("*.parquet"))
    if not batch_files:
        raise FileNotFoundError(f"No parquet files found in {batch_dir}")
    pf = pq.ParquetFile(str(batch_files[0]))
    cols = [c for c in pf.schema_arrow.names if c not in ('geo_accession', '__index_level_0__')]
    return len(cols)


def _format_bytes(num_bytes):
    value = float(num_bytes)
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for u in units:
        if value < 1024.0 or u == units[-1]:
            return f"{value:.1f}{u}"
        value /= 1024.0


def _get_mem_available_bytes():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _estimate_streaming_cache_bytes(batch_dir, num_genes, cache_size, total_workers):
    batch_files = sorted(Path(batch_dir).glob("*.parquet"))
    if not batch_files:
        return None, None
    try:
        pf = pq.ParquetFile(str(batch_files[0]))
        if pf.metadata.num_row_groups <= 0:
            return None, None
        avg_rows_per_rg = max(1, int(round(pf.metadata.num_rows / pf.metadata.num_row_groups)))
        # Approximate float32 dense tensor footprint per cached row-group table.
        bytes_per_cached_rg = avg_rows_per_rg * int(num_genes) * 4
        total_bytes = bytes_per_cached_rg * int(cache_size) * int(total_workers)
        return total_bytes, avg_rows_per_rg
    except Exception:
        return None, None


def format_float_for_tag(v: float) -> str:
    s = f"{v:.2e}" if (abs(v) < 1e-3 or abs(v) >= 1e3) else f"{v:.6f}"
    return s.replace('.', 'p').replace('+', '').replace('-', 'm')


def build_run_tag(cfg: dict) -> str:
    return (
        f"norm-{cfg['normalization']}"
        f"_lr-{format_float_for_tag(cfg['learning_rate'])}"
        f"_wd-{format_float_for_tag(cfg['weight_decay'])}"
        f"_mask-{format_float_for_tag(cfg['mask_ratio'])}"
        f"_ree-{format_float_for_tag(cfg['ree_base'])}"
    )


# ============================================================
# TRAINING (DDP)
# ============================================================
def main():
    print("\n[STARTUP] train.py started - initializing DDP...", flush=True)
    script_start = time.time()

    # Initialize DDP
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = rank == 0
    wandb_active = False

    if is_main:
        print("\n" + "=" * 70)
        print(f"ExpressionPerformer Training — DDP ({world_size} processes)")
        print("=" * 70)
        print(f"\n[SETUP] Rank: {rank}, Device: {device}")

    # Optional runtime overrides from launcher env (applied on rank 0, then broadcast).
    if is_main:
        env_resume_mode = os.environ.get('BRIDGE_RESUME_MODE')
        env_resume_path = os.environ.get('BRIDGE_RESUME_PATH')
        env_resume_run_id = os.environ.get('BRIDGE_RESUME_RUN_ID')
        env_target_epochs = os.environ.get('BRIDGE_TARGET_EPOCHS')
        env_use_amp = os.environ.get('BRIDGE_USE_AMP')
        env_amp_dtype = os.environ.get('BRIDGE_AMP_DTYPE')
        env_warmup_ratio = os.environ.get('BRIDGE_WARMUP_RATIO')
        env_grad_clip_norm = os.environ.get('BRIDGE_GRAD_CLIP_NORM')
        env_timing_cuda_events = os.environ.get('BRIDGE_TIMING_CUDA_EVENTS')

        if env_resume_mode:
            CONFIG['resume_mode'] = env_resume_mode.strip().lower()
        if env_resume_path:
            CONFIG['resume_path'] = env_resume_path.strip()
        if env_resume_run_id:
            CONFIG['resume_run_id'] = env_resume_run_id.strip()
        if env_target_epochs:
            CONFIG['epochs'] = int(env_target_epochs)
        if env_use_amp is not None:
            CONFIG['use_amp'] = str(env_use_amp).strip().lower() in ('1', 'true', 'yes', 'on')
        if env_amp_dtype:
            CONFIG['amp_dtype'] = str(env_amp_dtype).strip().lower()
        if env_warmup_ratio is not None:
            CONFIG['warmup_ratio'] = float(env_warmup_ratio)
        if env_grad_clip_norm is not None:
            CONFIG['grad_clip_norm'] = float(env_grad_clip_norm)
        if env_timing_cuda_events is not None:
            CONFIG['timing_cuda_events'] = str(env_timing_cuda_events).strip().lower() in (
                '1', 'true', 'yes', 'on'
            )

    # ─────────────────────────────────────────────────────────
    # WANDB (init early so sweep can override CONFIG)
    # ─────────────────────────────────────────────────────────
    if is_main and HAS_WANDB:
        wandb_init_timeout = int(os.environ.get('WANDB_INIT_TIMEOUT', '300'))
        wandb_strict_init = str(os.environ.get('WANDB_STRICT_INIT', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
        if is_main:
            print(f"[WANDB] init_timeout={wandb_init_timeout}s strict_init={wandb_strict_init}", flush=True)
        try:
            wandb.init(
                project="expression-performer",
                config=CONFIG,
                settings=wandb.Settings(init_timeout=wandb_init_timeout),
            )
            wandb_active = True
            # When running a sweep, wandb.config overrides CONFIG values
            for key in CONFIG:
                if key in wandb.config:
                    CONFIG[key] = wandb.config[key]
            # Always derive ffn_dim from hidden_dim (4x multiplier)
            CONFIG['ffn_dim'] = CONFIG['hidden_dim'] * 4
            wandb.config.update(CONFIG, allow_val_change=True)
        except Exception as e:
            wandb_active = False
            msg = f"[WANDB] init failed ({type(e).__name__}): {e}. Continuing without W&B."
            print(msg, flush=True)
            if wandb_strict_init:
                raise

    # Broadcast CONFIG from rank 0 so all ranks use the same hyperparams
    config_list = [CONFIG if is_main else None]
    dist.broadcast_object_list(config_list, src=0)
    CONFIG.update(config_list[0])

    # ─────────────────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────────────────
    data_dir = Path(CONFIG['data_dir'])
    batch_dir = data_dir / "batch_files"
    if not batch_dir.exists():
        batch_dir = data_dir

    if is_main:
        print("\n[DATA] Building sample indices...", flush=True)

    t0 = time.time()
    train_indices = None
    val_indices = None
    species_to_id = None
    if is_main:
        train_indices, val_indices, species_to_id = get_sample_indices(
            batch_dir,
            train_subset=CONFIG.get('train_subset', None),
            val_subset=CONFIG.get('val_subset', None),
            balanced_sampling=CONFIG.get('balanced_sampling', True),
            seed=CONFIG['seed'],
            verbose=True,
            balanced_shards_only=CONFIG.get('balanced_shards_only', False),
            balanced_shard_tolerance=CONFIG.get('balanced_shard_tolerance', 0),
            max_balanced_shards=CONFIG.get('max_balanced_shards', None),
        )

    train_indices_list = [train_indices if is_main else None]
    val_indices_list = [val_indices if is_main else None]
    species_to_id_list = [species_to_id if is_main else None]
    dist.broadcast_object_list(train_indices_list, src=0)
    dist.broadcast_object_list(val_indices_list, src=0)
    dist.broadcast_object_list(species_to_id_list, src=0)
    train_indices = train_indices_list[0]
    val_indices = val_indices_list[0]
    species_to_id = species_to_id_list[0]
    num_species = max(1, len(species_to_id))
    if is_main:
        print(f"[DATA] Species IDs: {species_to_id}", flush=True)
        print(f"  ✓ Index time: {time.time()-t0:.1f}s", flush=True)

    data_mode = CONFIG.get('data_mode', 'preload')
    if data_mode == 'streaming':
        if is_main:
            print("\n[DATA] Using streaming mode (on-the-fly parquet reads)", flush=True)

        train_ds = StreamingParquetMLMDataset(
            batch_dir,
            train_indices,
            normalization=CONFIG['normalization'],
            mask_ratio=CONFIG['mask_ratio'],
            mask_token=CONFIG['mask_token'],
            cache_size=CONFIG.get('stream_cache_size', 2),
            rank=rank,
            world_size=world_size,
            ddp_file_split=bool(CONFIG.get('stream_ddp_file_split', False)),
        )
        val_ds = StreamingParquetMLMDataset(
            batch_dir,
            val_indices,
            normalization=CONFIG['normalization'],
            mask_ratio=CONFIG['mask_ratio'],
            mask_token=CONFIG['mask_token'],
            cache_size=CONFIG.get('stream_cache_size', 2),
            rank=rank,
            world_size=world_size,
            ddp_file_split=bool(CONFIG.get('stream_ddp_file_split', False)),
        )
        num_genes = get_num_genes_from_batches(batch_dir)
    else:
        if is_main:
            print("\n[DATA] Loading training data into memory...", flush=True)
        X_train = load_batch_data(batch_dir, train_indices,
                                  normalization=CONFIG['normalization'],
                                  verbose=is_main)
        if is_main:
            print("[DATA] Loading validation data into memory...", flush=True)
        X_val = load_batch_data(batch_dir, val_indices,
                                normalization=CONFIG['normalization'],
                                verbose=is_main)

        train_species = np.asarray([int(rec[2]) if len(rec) >= 3 else 0 for rec in train_indices], dtype=np.int64)
        val_species = np.asarray([int(rec[2]) if len(rec) >= 3 else 0 for rec in val_indices], dtype=np.int64)

        num_genes = X_train.shape[1]

        # Data stored fully in host memory; faster per-step but higher RAM.
        train_ds = ExpressionMLMDataset(X_train, species_ids=train_species,
                        mask_ratio=CONFIG['mask_ratio'],
                        mask_token=CONFIG['mask_token'])
        val_ds = ExpressionMLMDataset(X_val, species_ids=val_species,
                          mask_ratio=CONFIG['mask_ratio'],
                          mask_token=CONFIG['mask_token'])

    if is_main:
        print(f"\n[CHECK] num_genes={num_genes}")
        assert num_genes > 10000, f"Expected ~16K genes, got {num_genes}"

    # ─────────────────────────────────────────────────────────
    # DATASETS & DATALOADERS
    # ─────────────────────────────────────────────────────────
    stream_use_file_shard = False
    if data_mode == 'streaming':
        stream_use_file_shard = bool(getattr(train_ds, 'ddp_file_split_active', False))
        if stream_use_file_shard:
            train_batch_sampler = RowGroupBatchSampler(
                train_ds.group_to_indices,
                batch_size=CONFIG['batch_size'],
                shuffle=True,
                seed=CONFIG.get('seed', 42),
                drop_last=False,
            )
            val_batch_sampler = RowGroupBatchSampler(
                val_ds.group_to_indices,
                batch_size=CONFIG['batch_size'],
                shuffle=False,
                seed=CONFIG.get('seed', 42),
                drop_last=False,
            )

            # Keep DDP ranks in lockstep by forcing identical per-rank batch counts.
            local_batch_counts = torch.tensor(
                [len(train_batch_sampler), len(val_batch_sampler)],
                dtype=torch.long,
                device=device,
            )
            gathered_batch_counts = [torch.zeros_like(local_batch_counts) for _ in range(world_size)]
            dist.all_gather(gathered_batch_counts, local_batch_counts)
            batch_counts_matrix = torch.stack(gathered_batch_counts, dim=0).cpu().numpy()

            synced_train_batches = int(batch_counts_matrix[:, 0].min())
            synced_val_batches = int(batch_counts_matrix[:, 1].min())

            train_batch_sampler.target_num_batches = synced_train_batches
            val_batch_sampler.target_num_batches = synced_val_batches

            if is_main:
                train_counts = [int(x) for x in batch_counts_matrix[:, 0].tolist()]
                val_counts = [int(x) for x in batch_counts_matrix[:, 1].tolist()]
                batch_size = int(CONFIG['batch_size'])

                # Local dataset sizes after rank-specific file filtering.
                local_train_samples = [c * batch_size for c in train_counts]
                local_val_samples = [c * batch_size for c in val_counts]

                # Samples actually consumed in synchronized DDP steps.
                eff_train_global = synced_train_batches * batch_size * world_size
                eff_val_global = synced_val_batches * batch_size * world_size

                # Samples skipped because ranks are truncated to min batch count.
                dropped_train = max(0, sum(local_train_samples) - eff_train_global)
                dropped_val = max(0, sum(local_val_samples) - eff_val_global)

                print(
                    f"[DATA] Streaming file-shard local batch counts (train): {train_counts} -> synced={synced_train_batches}",
                    flush=True,
                )
                print(
                    f"[DATA] Streaming file-shard local batch counts (val): {val_counts} -> synced={synced_val_batches}",
                    flush=True,
                )
                print(
                    f"[DATA] Streaming file-shard effective train samples/epoch (global): {eff_train_global:,} | dropped by sync: {dropped_train:,}",
                    flush=True,
                )
                print(
                    f"[DATA] Streaming file-shard effective val samples/epoch (global): {eff_val_global:,} | dropped by sync: {dropped_val:,}",
                    flush=True,
                )
        else:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=42
            )
            val_sampler = DistributedSampler(
                val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=42
            )
            if is_main:
                print("[DATA] Streaming using sample-level DistributedSampler sharding.", flush=True)
    else:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                            rank=rank, shuffle=True, seed=42)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                          rank=rank, shuffle=False, seed=42)

    num_workers = int(CONFIG.get('num_workers', 0))
    if data_mode == 'streaming' and num_workers > 0 and is_main:
        total_workers = num_workers * world_size
        avail_cores = os.cpu_count() or 0
        if avail_cores > 0 and total_workers > avail_cores:
            print(
                f"[WARN] num_workers*world_size={total_workers} exceeds visible CPU cores={avail_cores}. "
                "This may increase context-switching and reduce throughput.",
                flush=True,
            )

        cache_size = int(CONFIG.get('stream_cache_size', 2))
        cache_bytes, avg_rows_per_rg = _estimate_streaming_cache_bytes(
            batch_dir=batch_dir,
            num_genes=num_genes,
            cache_size=cache_size,
            total_workers=total_workers,
        )
        mem_available = _get_mem_available_bytes()
        if cache_bytes is not None:
            msg = (
                f"[DATA] Estimated streaming cache upper bound: {_format_bytes(cache_bytes)} "
                f"(cache_size={cache_size}, workers={total_workers}, avg_row_group_rows~{avg_rows_per_rg})."
            )
            if mem_available is not None:
                msg += f" MemAvailable~{_format_bytes(mem_available)}."
            print(msg, flush=True)

            if mem_available is not None and cache_bytes >= int(0.7 * mem_available):
                print(
                    "[WARN] cache_size*num_workers*world_size may exceed practical memory headroom. "
                    "Reduce stream_cache_size and/or num_workers if OOM occurs.",
                    flush=True,
                )

    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = int(CONFIG.get('prefetch_factor', 2))
        loader_kwargs['persistent_workers'] = bool(CONFIG.get('persistent_workers', False))

    if data_mode == 'streaming':
        if stream_use_file_shard:
            train_loader = DataLoader(
                train_ds,
                batch_sampler=train_batch_sampler,
                collate_fn=train_ds.collate_batch,
                **loader_kwargs,
            )
            val_loader = DataLoader(
                val_ds,
                batch_sampler=val_batch_sampler,
                collate_fn=val_ds.collate_batch,
                **loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=CONFIG['batch_size'],
                sampler=train_sampler,
                collate_fn=train_ds.collate_batch,
                **loader_kwargs,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=CONFIG['batch_size'],
                sampler=val_sampler,
                collate_fn=val_ds.collate_batch,
                **loader_kwargs,
            )
    else:
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                                  sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                                sampler=val_sampler, **loader_kwargs)

    if is_main:
        local_train_samples = len(train_ds)
        local_val_samples = len(val_ds)
        local_train_batches = len(train_loader)
        local_val_batches = len(val_loader)
        batch_size = int(CONFIG['batch_size'])

        # These are rank-local views (rank 0 only in logs).
        print(f"\n[DATA] Local (rank 0) train: {local_train_samples:,} samples, {local_train_batches} batches")
        print(f"[DATA] Local (rank 0) val:   {local_val_samples:,} samples, {local_val_batches} batches")

        # Effective global samples consumed per epoch under synchronized DDP steps.
        eff_train_global = local_train_batches * batch_size * world_size
        eff_val_global = local_val_batches * batch_size * world_size
        print(f"[DATA] Effective global train samples/epoch from loaders: {eff_train_global:,}")
        print(f"[DATA] Effective global val samples/epoch from loaders:   {eff_val_global:,}")

    # Synchronize after data loading
    dist.barrier(device_ids=[local_rank])

    # ─────────────────────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n[MODEL] Building ExpressionPerformer...")

    model = ExpressionPerformer(
        num_genes=num_genes,
        hidden_dim=CONFIG['hidden_dim'],
        n_heads=CONFIG['num_heads'],
        n_layers=CONFIG['num_layers'],
        ffn_dim=CONFIG['ffn_dim'],
        ree_base=CONFIG['ree_base'],
        mask_token_id=CONFIG['mask_token'],
        feature_type=CONFIG['feature_type'],
        compute_type=CONFIG['compute_type'],
        include_species_embedding=CONFIG.get('include_species_embedding', False),
        num_species=num_species,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  ✓ Parameters: {total_params:,}")

    # ─────────────────────────────────────────────────────────
    # OPTIMIZER & SCHEDULER
    # ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])
    warmup_ratio = float(CONFIG.get('warmup_ratio', 0.0))
    warmup_ratio = max(0.0, min(1.0, warmup_ratio))
    total_train_steps = max(1, len(train_loader) * int(CONFIG['epochs']))
    warmup_steps = int(total_train_steps * warmup_ratio)

    def _lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        if total_train_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    amp_enabled = bool(CONFIG.get('use_amp', False)) and device.type == 'cuda'
    amp_dtype_cfg = str(CONFIG.get('amp_dtype', 'auto')).strip().lower()
    if amp_dtype_cfg not in ('auto', 'bf16', 'fp16'):
        if is_main:
            print(f"[WARN] Invalid amp_dtype='{amp_dtype_cfg}', defaulting to 'auto'.", flush=True)
        amp_dtype_cfg = 'auto'

    if amp_enabled:
        if amp_dtype_cfg == 'bf16':
            if torch.cuda.is_bf16_supported():
                amp_dtype_torch = torch.bfloat16
            else:
                if is_main:
                    print("[WARN] bf16 requested but not supported on this GPU. Falling back to fp16.", flush=True)
                amp_dtype_torch = torch.float16
        elif amp_dtype_cfg == 'fp16':
            amp_dtype_torch = torch.float16
        else:
            amp_dtype_torch = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype_torch = torch.float16

    # GradScaler is only needed for fp16 autocast.
    amp_scaler_enabled = amp_enabled and amp_dtype_torch == torch.float16
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            amp_scaler = torch.amp.GradScaler('cuda', enabled=amp_scaler_enabled)
        except TypeError:
            amp_scaler = torch.amp.GradScaler(enabled=amp_scaler_enabled)
    else:
        amp_scaler = torch.cuda.amp.GradScaler(enabled=amp_scaler_enabled)

    if is_main:
        print(f"  ✓ AdamW (lr={CONFIG['learning_rate']})")
        print(f"  ✓ LR schedule: linear warmup {warmup_ratio:.1%} ({warmup_steps}/{total_train_steps} steps) + cosine decay")
        grad_clip_norm_cfg = float(CONFIG.get('grad_clip_norm', 0.0) or 0.0)
        if grad_clip_norm_cfg > 0:
            print(f"  ✓ Grad clipping: max_norm={grad_clip_norm_cfg}")
        else:
            print("  ✓ Grad clipping: disabled")
        if amp_enabled:
            amp_name = 'bf16' if amp_dtype_torch == torch.bfloat16 else 'fp16'
            scaler_state = 'on' if amp_scaler.is_enabled() else 'off'
            print(f"  ✓ AMP enabled ({amp_name}, GradScaler={scaler_state})")
        else:
            print("  ✓ AMP disabled")

    # ─────────────────────────────────────────────────────────
    # CHECKPOINT RESUME
    # ─────────────────────────────────────────────────────────
    start_epoch = 0
    latest_ckpt = None

    resume_mode = str(CONFIG.get('resume_mode', 'off')).strip().lower()
    resume_path = CONFIG.get('resume_path')
    resume_run_id = CONFIG.get('resume_run_id')

    # Backward compatibility with legacy boolean flag; explicit mode wins.
    if resume_mode == 'off' and CONFIG.get('resume', False):
        if resume_path:
            resume_mode = 'path'
        elif resume_run_id:
            resume_mode = 'run_id'

    if resume_mode not in ('off', 'path', 'run_id'):
        raise ValueError(f"Invalid resume_mode='{resume_mode}'. Use one of: off, path, run_id")

    if resume_mode == 'path':
        if not resume_path:
            raise ValueError("resume_mode='path' requires CONFIG['resume_path']")
        latest_ckpt = Path(resume_path)
    elif resume_mode == 'run_id':
        if not resume_run_id:
            raise ValueError("resume_mode='run_id' requires CONFIG['resume_run_id']")
        latest_ckpt = Path(CONFIG['checkpoint_dir']) / str(resume_run_id) / 'latest.pt'

    ckpt = None
    if latest_ckpt is not None:
        if not latest_ckpt.exists():
            raise FileNotFoundError(f"Requested resume checkpoint not found: {latest_ckpt}")
        if is_main:
            print(f"[RESUME] Loading checkpoint: {latest_ckpt}", flush=True)
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if amp_scaler.is_enabled() and ckpt.get('scaler_state_dict') is not None:
            amp_scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch']
        if is_main:
            print(f"[RESUME] Restarting from epoch {start_epoch}", flush=True)
    elif is_main:
        print("[RESUME] Fresh start (resume_mode=off)", flush=True)

    # ─────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────
    if is_main:
        print("\n" + "=" * 70, flush=True)
        print("[TRAIN] Starting training...", flush=True)
        print("=" * 70 + "\n", flush=True)

    best_val_loss = float('inf')
    if latest_ckpt is not None and latest_ckpt.exists():
        best_val_loss = float(ckpt.get('val_loss', best_val_loss))
    patience_counter = 0
    train_losses, val_losses = [], []
    final_epoch = start_epoch

    ckpt_base = Path(CONFIG['checkpoint_dir'])
    # Per-run subdir (wandb run ID if available, else timestamp)
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    if resume_mode == 'run_id' and resume_run_id:
        run_id = str(resume_run_id)
    elif is_main and wandb_active and wandb.run is not None:
        run_id = wandb.run.id
    else:
        run_id = run_timestamp
    run_tag = build_run_tag(CONFIG)
    ckpt_dir = ckpt_base / run_id
    if is_main:
        ckpt_base.mkdir(exist_ok=True, parents=True)
        ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Load global best val loss (across all runs)
    global_best_path = ckpt_base / 'global_best_val_loss.json'
    if global_best_path.exists():
        with open(global_best_path) as f:
            global_best_val_loss = json.load(f)['val_loss']
    else:
        global_best_val_loss = float('inf')

    run_metadata = {
        'run_id': run_id,
        'timestamp': run_timestamp,
        'run_tag': run_tag,
        'resume_mode': resume_mode,
        'resume_checkpoint': str(latest_ckpt) if latest_ckpt is not None else None,
        'normalization': CONFIG['normalization'],
        'sweep_parameters': {
            'learning_rate': CONFIG['learning_rate'],
            'weight_decay': CONFIG['weight_decay'],
            'mask_ratio': CONFIG['mask_ratio'],
            'ree_base': CONFIG['ree_base'],
            'early_stopping': CONFIG['early_stopping'],
        },
        'architecture': {
            'hidden_dim': CONFIG['hidden_dim'],
            'ffn_dim': CONFIG['ffn_dim'],
            'num_heads': CONFIG['num_heads'],
            'num_layers': CONFIG['num_layers'],
            'include_species_embedding': CONFIG.get('include_species_embedding', False),
            'num_species': int(num_species),
        },
        'dataset': {
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'num_genes': int(num_genes),
            'train_used_counts': None,  # Not computed for lazy-loaded data
            'val_used_counts': None,
            'train_raw_counts': None,
            'val_raw_counts': None,
            'balanced_sampling': CONFIG['balanced_sampling'],
            'train_subset': CONFIG['train_subset'],
            'val_subset': CONFIG['val_subset'],
        },
    }

    if is_main and start_epoch >= CONFIG['epochs']:
        print(
            f"[RESUME] start_epoch ({start_epoch}) >= epochs ({CONFIG['epochs']}); no training iterations to run.",
            flush=True,
        )

    for epoch in range(start_epoch, CONFIG['epochs']):
        final_epoch = epoch + 1
        epoch_start = time.time()
        if data_mode == 'streaming':
            if stream_use_file_shard:
                train_batch_sampler.set_epoch(epoch)
            else:
                train_sampler.set_epoch(epoch)
        else:
            train_sampler.set_epoch(epoch)

        # --- Train ---
        model.train()
        running_loss = 0.0
        num_batches = 0
        progress_log_interval_sec = float(CONFIG.get('progress_log_interval_sec', 60))
        timing_profile = bool(CONFIG.get('timing_profile', True))
        timing_cuda_events = bool(CONFIG.get('timing_cuda_events', False)) and device.type == 'cuda'
        grad_clip_norm = float(CONFIG.get('grad_clip_norm', 0.0) or 0.0)

        # Per-epoch timing breakdown
        train_data_wait_s = 0.0
        train_h2d_s = 0.0
        train_forward_s = 0.0
        train_loss_s = 0.0
        train_zero_grad_s = 0.0
        train_backward_s = 0.0
        train_optim_s = 0.0
        train_step_s = 0.0
        train_cuda_forward_s = 0.0
        train_cuda_backward_s = 0.0
        train_cuda_step_s = 0.0

        # Time-based heartbeat (useful when batches are slow and 25% cadence is too sparse)
        last_progress_log_t = time.perf_counter()
        prev_step_end_t = time.perf_counter()

        # Windowed timing accumulators (reset on each heartbeat print)
        win_batches = 0
        win_wait_s = 0.0
        win_h2d_s = 0.0
        win_forward_s = 0.0
        win_loss_s = 0.0
        win_zero_s = 0.0
        win_backward_s = 0.0
        win_optim_s = 0.0
        win_step_s = 0.0
        win_cuda_forward_s = 0.0
        win_cuda_backward_s = 0.0
        win_cuda_step_s = 0.0

        if is_main and epoch == start_epoch and timing_profile and timing_cuda_events:
            print("[TIMING] CUDA-event timing enabled for train fwd/bwd/step.", flush=True)

        for batch_idx, (x_masked, x_true, mask_idx, species_id) in enumerate(train_loader):
            if timing_cuda_events:
                ev_step_start = torch.cuda.Event(enable_timing=True)
                ev_step_end = torch.cuda.Event(enable_timing=True)
                ev_fwd_start = torch.cuda.Event(enable_timing=True)
                ev_fwd_end = torch.cuda.Event(enable_timing=True)
                ev_bwd_start = torch.cuda.Event(enable_timing=True)
                ev_bwd_end = torch.cuda.Event(enable_timing=True)
                ev_opt_start = torch.cuda.Event(enable_timing=True)
                ev_opt_end = torch.cuda.Event(enable_timing=True)

            step_start_t = time.perf_counter()
            step_wait_s = max(0.0, step_start_t - prev_step_end_t)
            train_data_wait_s += step_wait_s
            win_wait_s += step_wait_s

            t0 = time.perf_counter()
            x_masked = x_masked.to(device)
            x_true = x_true.to(device)
            species_id = species_id.to(device)
            dt = time.perf_counter() - t0
            train_h2d_s += dt
            win_h2d_s += dt

            if timing_cuda_events:
                ev_step_start.record()

            t0 = time.perf_counter()
            if timing_cuda_events:
                ev_fwd_start.record()
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype_torch, enabled=amp_enabled):
                pred = model(x_masked, species_id)  # [B, G]
            if timing_cuda_events:
                ev_fwd_end.record()
            dt = time.perf_counter() - t0
            train_forward_s += dt
            win_forward_s += dt

            t0 = time.perf_counter()
            # MSE loss on masked positions only
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype_torch, enabled=amp_enabled):
                loss_parts = []
                for i in range(len(x_masked)):
                    idxs = mask_idx[i]
                    if len(idxs) > 0:
                        loss_parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))

                loss = torch.stack(loss_parts).mean() if loss_parts else torch.tensor(0.0, device=device)
            dt = time.perf_counter() - t0
            train_loss_s += dt
            win_loss_s += dt

            t0 = time.perf_counter()
            optimizer.zero_grad()
            dt = time.perf_counter() - t0
            train_zero_grad_s += dt
            win_zero_s += dt

            t0 = time.perf_counter()
            if timing_cuda_events:
                ev_bwd_start.record()
            if amp_scaler.is_enabled():
                amp_scaler.scale(loss).backward()
            else:
                loss.backward()
            if timing_cuda_events:
                ev_bwd_end.record()
            dt = time.perf_counter() - t0
            train_backward_s += dt
            win_backward_s += dt

            t0 = time.perf_counter()
            if timing_cuda_events:
                ev_opt_start.record()
            if grad_clip_norm > 0.0:
                if amp_scaler.is_enabled():
                    amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            if amp_scaler.is_enabled():
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if timing_cuda_events:
                ev_opt_end.record()
                ev_step_end.record()
                torch.cuda.synchronize(device)
                cuda_fwd_s = ev_fwd_start.elapsed_time(ev_fwd_end) * 1e-3
                cuda_bwd_s = ev_bwd_start.elapsed_time(ev_bwd_end) * 1e-3
                cuda_step_s = ev_step_start.elapsed_time(ev_step_end) * 1e-3
                train_cuda_forward_s += cuda_fwd_s
                train_cuda_backward_s += cuda_bwd_s
                train_cuda_step_s += cuda_step_s
                win_cuda_forward_s += cuda_fwd_s
                win_cuda_backward_s += cuda_bwd_s
                win_cuda_step_s += cuda_step_s
            dt = time.perf_counter() - t0
            train_optim_s += dt
            win_optim_s += dt

            loss_item = float(loss.item())
            running_loss += loss_item
            num_batches += 1
            step_end_t = time.perf_counter()
            step_dur_s = step_end_t - step_start_t
            train_step_s += step_dur_s
            win_step_s += step_dur_s
            win_batches += 1
            prev_step_end_t = step_end_t

            # Progress every 25% or on heartbeat interval, whichever comes first.
            should_log_batch = (batch_idx + 1) % max(1, len(train_loader) // 4) == 0
            should_log_time = (step_end_t - last_progress_log_t) >= progress_log_interval_sec
            should_log_local = bool(should_log_batch or should_log_time)
            should_log_global = should_log_local
            if world_size > 1:
                log_flag = torch.tensor(
                    1.0 if should_log_local else 0.0,
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(log_flag, op=dist.ReduceOp.MAX)
                should_log_global = bool(log_flag.item() > 0.0)

            if should_log_global:
                elapsed = step_end_t - last_progress_log_t
                if world_size > 1:
                    train_metric_buf = torch.tensor(
                        [loss_item, running_loss, float(num_batches), train_step_s],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(train_metric_buf, op=dist.ReduceOp.SUM)
                    global_loss = (train_metric_buf[0] / world_size).item()
                    global_avg = (train_metric_buf[1] / train_metric_buf[2].clamp(min=1.0)).item()
                    global_avg_step = (train_metric_buf[3] / train_metric_buf[2].clamp(min=1.0)).item()
                else:
                    global_loss = loss_item
                    global_avg = running_loss / max(1, num_batches)
                    global_avg_step = train_step_s / max(1, num_batches)

                if is_main:
                    print(f"  Epoch {epoch+1}/{CONFIG['epochs']} | "
                          f"Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {global_loss:.6f} | Avg: {global_avg:.6f} | "
                          f"Avg step: {global_avg_step:.3f}s | Window: {elapsed:.1f}s")
                    if timing_profile and win_batches > 0:
                        print(
                            "    Timing(window avg/batch): "
                            f"wait={win_wait_s/win_batches:.3f}s, "
                            f"h2d={win_h2d_s/win_batches:.3f}s, "
                            f"fwd={win_forward_s/win_batches:.3f}s, "
                            f"loss={win_loss_s/win_batches:.3f}s, "
                            f"zero={win_zero_s/win_batches:.3f}s, "
                            f"bwd={win_backward_s/win_batches:.3f}s, "
                            f"opt={win_optim_s/win_batches:.3f}s"
                        )
                        if timing_cuda_events:
                            print(
                                "    Timing(window CUDA avg/batch): "
                                f"fwd={win_cuda_forward_s/win_batches:.3f}s, "
                                f"bwd={win_cuda_backward_s/win_batches:.3f}s, "
                                f"step={win_cuda_step_s/win_batches:.3f}s"
                            )

                win_batches = 0
                win_wait_s = 0.0
                win_h2d_s = 0.0
                win_forward_s = 0.0
                win_loss_s = 0.0
                win_zero_s = 0.0
                win_backward_s = 0.0
                win_optim_s = 0.0
                win_step_s = 0.0
                win_cuda_forward_s = 0.0
                win_cuda_backward_s = 0.0
                win_cuda_step_s = 0.0
                last_progress_log_t = step_end_t

        epoch_train_loss = running_loss / max(1, num_batches)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_start_t = time.perf_counter()
        val_data_wait_s = 0.0
        val_h2d_s = 0.0
        val_forward_s = 0.0
        val_loss_s = 0.0
        val_step_s = 0.0
        prev_val_step_end_t = time.perf_counter()

        with torch.no_grad():
            for x_masked, x_true, mask_idx, species_id in val_loader:
                val_step_start_t = time.perf_counter()
                val_data_wait_s += max(0.0, val_step_start_t - prev_val_step_end_t)

                t0 = time.perf_counter()
                x_masked = x_masked.to(device)
                x_true = x_true.to(device)
                species_id = species_id.to(device)
                val_h2d_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype_torch, enabled=amp_enabled):
                    pred = model(x_masked, species_id)
                val_forward_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype_torch, enabled=amp_enabled):
                    loss_parts = []
                    for i in range(len(x_masked)):
                        idxs = mask_idx[i]
                        if len(idxs) > 0:
                            loss_parts.append(F.mse_loss(pred[i, idxs], x_true[i, idxs]))

                if loss_parts:
                    val_loss += torch.stack(loss_parts).mean().item()
                    val_batches += 1
                val_loss_s += time.perf_counter() - t0

                val_step_end_t = time.perf_counter()
                val_step_s += val_step_end_t - val_step_start_t
                prev_val_step_end_t = val_step_end_t
            val_time_s = time.perf_counter() - val_start_t

        # Sync validation across ranks
        val_sync_t0 = time.perf_counter()
        vl = torch.tensor(val_loss, device=device)
        vb = torch.tensor(float(val_batches), device=device)
        dist.all_reduce(vl, op=dist.ReduceOp.SUM)
        dist.all_reduce(vb, op=dist.ReduceOp.SUM)
        val_sync_s = time.perf_counter() - val_sync_t0
        epoch_val_loss = (vl / vb.clamp(min=1)).item()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Log to wandb
        if is_main and wandb_active:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'lr': scheduler.get_last_lr()[0],
            })

        epoch_time = time.time() - epoch_start

        # --- Checkpoint ---
        if is_main:
            model_sd = model.module.state_dict()

            print(f"\n  ╔════════════════════════════════════════════╗")
            print(f"  ║ Epoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  ║ Train Loss: {epoch_train_loss:.6f}")
            print(f"  ║ Val Loss:   {epoch_val_loss:.6f}")
            print(f"  ║ Time: {epoch_time:.1f}s")
            if timing_profile:
                denom = max(1, num_batches)
                global_train_samples = num_batches * int(CONFIG['batch_size']) * world_size
                train_samples_per_s = global_train_samples / max(train_step_s, 1e-9)
                val_denom = max(1, val_batches)
                print(
                    f"  ║ Timing/train avg per batch (s): "
                    f"wait={train_data_wait_s/denom:.3f}, h2d={train_h2d_s/denom:.3f}, "
                    f"fwd={train_forward_s/denom:.3f}, loss={train_loss_s/denom:.3f}, "
                    f"zero={train_zero_grad_s/denom:.3f}, bwd={train_backward_s/denom:.3f}, "
                    f"opt={train_optim_s/denom:.3f}"
                )
                print(
                    f"  ║ Timing/train throughput: {train_samples_per_s:.1f} samples/s "
                    f"(global, this rank group)"
                )
                if timing_cuda_events:
                    print(
                        f"  ║ Timing/train CUDA avg per batch (s): "
                        f"fwd={train_cuda_forward_s/denom:.3f}, "
                        f"bwd={train_cuda_backward_s/denom:.3f}, "
                        f"step={train_cuda_step_s/denom:.3f}"
                    )
                print(
                    f"  ║ Timing/validate avg per batch (s): "
                    f"wait={val_data_wait_s/val_denom:.3f}, h2d={val_h2d_s/val_denom:.3f}, "
                    f"fwd={val_forward_s/val_denom:.3f}, loss={val_loss_s/val_denom:.3f}"
                )
                print(
                    f"  ║ Timing/validate totals (s): loop={val_time_s:.1f}, ddp_sync={val_sync_s:.3f}"
                )

            checkpoint_payload = {
                'model_state_dict': model_sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': amp_scaler.state_dict() if amp_scaler.is_enabled() else None,
                'epoch': epoch + 1,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'config': dict(CONFIG),
                'run_metadata': run_metadata,
                'total_params': total_params,
            }

            torch.save(checkpoint_payload, ckpt_dir / f"epoch_{epoch:02d}.pt")
            torch.save(checkpoint_payload, ckpt_dir / "latest.pt")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(checkpoint_payload, ckpt_dir / "best_model.pt")
                best_named = ckpt_dir / f"best_{run_tag}_run-{run_id}.pt"
                torch.save(checkpoint_payload, best_named)
                print(f"  ║ ✓ New best (run)! Saved best_model.pt")

                # Update global best across all runs
                if epoch_val_loss < global_best_val_loss:
                    global_best_val_loss = epoch_val_loss
                    torch.save(checkpoint_payload, ckpt_base / "best_model.pt")
                    with open(global_best_path, 'w') as f:
                        json.dump({'val_loss': global_best_val_loss,
                                   'run_id': run_id,
                                   'epoch': epoch + 1,
                                   'run_tag': run_tag,
                                   'normalization': CONFIG['normalization']}, f, indent=2)
                    print(f"  ║ ★ New global best! {epoch_val_loss:.6f}")
            else:
                if CONFIG['early_stopping']:
                    patience_counter += 1
                    print(f"  ║ ✗ No improvement ({patience_counter}/{CONFIG['patience']})")
                    if patience_counter >= CONFIG['patience']:
                        print(f"  ║ ⚠ Early stopping!")
                        print(f"  ╚════════════════════════════════════════════╝\n")
                        break
                else:
                    print("  ║ ✗ No improvement (early_stopping=False; continuing)")

            print(f"  ╚════════════════════════════════════════════╝\n")

    # ─────────────────────────────────────────────────────────
    # SAVE ARTIFACTS
    # ─────────────────────────────────────────────────────────
    if is_main:
        # Config
        cfg = {
            **CONFIG,
            'num_genes': num_genes,
            'total_params': total_params,
            'best_val_loss': best_val_loss,
            'final_epoch': final_epoch,
            'run_id': run_id,
            'timestamp': run_timestamp,
            'run_tag': run_tag,
            'dataset': run_metadata['dataset'],
            'architecture': run_metadata['architecture'],
            'sweep_parameters': run_metadata['sweep_parameters'],
        }
        with open(ckpt_dir / "config.json", 'w') as f:
            json.dump(cfg, f, indent=2)

        with open(ckpt_dir / "run_metadata.json", 'w') as f:
            json.dump(run_metadata, f, indent=2)

        # Loss CSV
        pd.DataFrame({'epoch': range(len(train_losses)),
                       'train_loss': train_losses,
                       'val_loss': val_losses}).to_csv(
            ckpt_dir / "loss_history.csv", index=False)

        # Loss plot
        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, marker='o', label='Train Loss', linewidth=2)
            plt.plot(val_losses, marker='s', label='Val Loss', linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("ExpressionPerformer Training")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(ckpt_dir / "loss_plot.png", dpi=150)
            plt.close()

        total_time = time.time() - script_start
        print("=" * 70)
        print(f"Training complete! {total_time:.0f}s ({total_time/60:.1f}m)")
        print(f"  Run best val loss:    {best_val_loss:.6f}")
        print(f"  Global best val loss: {global_best_val_loss:.6f}")
        print(f"  Run checkpoints:      {ckpt_dir}/")
        print(f"  Global best model:    {ckpt_base / 'best_model.pt'}")
        print("=" * 70 + "\n")

    if is_main and wandb_active:
        wandb.finish()

    # Ensure all ranks finish before cleanup
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        rank = os.environ.get("RANK", "unknown")
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        ts = time.strftime('%Y%m%d_%H%M%S')
        crash_dir = Path("./logs")
        crash_dir.mkdir(exist_ok=True, parents=True)
        crash_log = crash_dir / f"train_single_crash_rank-{rank}_local-{local_rank}_{ts}.log"
        err_msg = f"\n[ERROR] Exception in train_single.py (rank={rank}, local_rank={local_rank}): {e}"
        print(err_msg, flush=True, file=sys.stderr)
        tb = traceback.format_exc()
        try:
            with open(crash_log, "w") as f:
                f.write(err_msg + "\n")
                f.write(tb)
            print(f"[ERROR] Crash log saved: {crash_log}", flush=True, file=sys.stderr)
        except Exception as log_err:
            print(f"[WARN] Failed to write crash log: {log_err}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Try to cleanup DDP even on error
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass
        sys.exit(1)
