"""
RNA-seq preprocessing module for ARCHS4 expression data.

Handles: ortholog loading, gene selection, TPM normalization,
log transformation, QC filtering, cross-split deduplication, and export.

Usage:
    from preprocessing import RNADatasetBuilder

    builder = RNADatasetBuilder(
        species="both",
        gene_set="shared_orthologs",
    )
    builder.process()
    builder.save_parquet("output/")

    # Split later with scikit-learn
    X, meta = builder.get_data()
"""

import os
import gc
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# H5 VALIDATION & DIAGNOSTICS
# ============================================================
def validate_h5_file(h5_path: str, verbose: bool = True) -> bool:
    """
    Validate HDF5 file structure and datasets before processing.
    Tests file integrity, dataset existence, and readability.
    
    Returns:
        True if file is valid, False otherwise.
    """
    import h5py
    
    try:
        with h5py.File(h5_path, 'r') as f:
            errors = []
            
            # 1. Validate expression dataset exists and has correct shape
            if 'data/expression' not in f:
                errors.append("Missing 'data/expression' dataset")
            else:
                expr = f['data/expression']
                if verbose:
                    print(f"    Dataset shape: {expr.shape}")
                    print(f"    Dtype: {expr.dtype}")
                    print(f"    Chunks: {expr.chunks}")
                
            # 2. Test reading a small chunk (corner of matrix)
            try:
                test_read = f['data/expression'][:100, :100]
                if verbose:
                    print(f"    ✓ Test read successful (100×100 subset)")
            except Exception as e:
                errors.append(f"Failed to read test chunk: {e}")
            
            if errors:
                print(f"  ⚠️ Issues found:")
                for err in errors:
                    print(f"    - {err}")
                return False
            else:
                if verbose:
                    print(f"  ✓ File structure valid")
                return True
                
    except OSError as e:
        print(f"  ❌ H5 file corrupted or unreadable: {e}")
        return False


def diagnose_h5_batch_failure(
    h5_path: str,
    batch_start: int,
    batch_end: int,
    gene_mask: np.ndarray = None,
    verbose: bool = True,
) -> tuple[list[int], list[tuple[int, str]]]:
    """
    Pinpoint exactly which samples fail to read in a batch.
    
    Args:
        h5_path: Path to HDF5 file
        batch_start: Starting sample index
        batch_end: Ending sample index (exclusive)
        gene_mask: Optional gene mask to apply (like in extract_and_normalize)
        verbose: Print diagnostics
    
    Returns:
        (successful_sample_indices, [(failed_idx, error_msg), ...])
    """
    import h5py
    
    successful = []
    failed = []
    
    try:
        with h5py.File(h5_path, 'r') as f:
            expr = f['data/expression']
            n_samples = expr.shape[1]
            
            # Clamp to actual file size
            batch_end = min(batch_end, n_samples)
            batch_size = batch_end - batch_start
            
            if verbose:
                print(f"    Diagnosing {batch_size:,} samples in range [{batch_start}:{batch_end}]...")
            
            for sample_idx in range(batch_start, batch_end):
                try:
                    # Try to read just that sample
                    sample_data = expr[:, sample_idx]
                    
                    # If gene mask provided, apply it
                    if gene_mask is not None:
                        sample_data = sample_data[gene_mask]
                    
                    # Check if sample has any data
                    if len(sample_data) > 0:
                        successful.append(sample_idx)
                    else:
                        failed.append((sample_idx, "empty_data"))
                        
                except Exception as e:
                    failed.append((sample_idx, str(e)[:60]))
            
            if verbose:
                print(f"    ✓ {len(successful):,} samples readable")
                if failed:
                    print(f"    ✗ {len(failed):,} samples FAILED")
                    print(f"      Failed indices: {[s[0] for s in failed[:10]]}")
                    if len(failed) > 10:
                        print(f"      ... and {len(failed) - 10} more")
                    print(f"      Error type: {failed[0][1]}")
    
    except OSError as e:
        print(f"    ❌ Cannot open H5 file for diagnosis: {e}")
    
    return successful, failed


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class PreprocessingConfig:
    """All preprocessing parameters in one place."""

    # --- Species & gene set ---
    species: str = "both"                    # "human", "mouse", "both"
    gene_set: str = "shared_orthologs"       # "shared_orthologs", "union_orthologs"

    # --- QC ---
    qc_min_nonzero: int = 14_000             # min non-zero genes per sample
    remove_single_cell: bool = True

    # --- Normalization ---
    normalization: str = "tpm"               # "log1p_tpm", "tpm", "raw_counts"
    debug_tpm_denominator: bool = False       # print TPM denominator debug stats
    extraction_batch_size: int = 10_000

    # --- Subsetting ---
    max_samples_per_species: Optional[int] = None  # None = all data

    # --- Paths ---
    archs4_dir: str = "data/archs4"
    orthologs_file: str = "data/ensembl/orthologs_one2one.txt"
    protein_coding_file: str = "data/ensembl/protein_coding_ortholog_genes.txt"
    exon_lengths_human: str = "data/gencode/gencode_v49_gene_exon_lengths.csv"
    exon_lengths_mouse: str = "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv"
    output_dir: str = "data/archs4/train_orthologs"

    # --- Reproducibility ---
    seed: int = 42


# ============================================================
# GENE REGISTRY
# ============================================================
class GeneRegistry:
    """
    Loads ortholog mappings and determines the canonical gene list
    based on the chosen gene_set mode.
    """

    def __init__(self, orthologs_file: str, protein_coding_file: str):
        self.ortho_df = pd.read_csv(orthologs_file, sep="\t")

        # Load protein-coding gene whitelist (human symbols)
        with open(protein_coding_file) as f:
            self.protein_coding = set(line.strip() for line in f if line.strip())

        # Mouse→Human and Human→Mouse mappings
        self.mouse_to_human = dict(
            zip(self.ortho_df["Gene name"], self.ortho_df["Human gene name"])
        )
        self.human_to_mouse = dict(
            zip(self.ortho_df["Human gene name"], self.ortho_df["Gene name"])
        )

        # All valid human ortholog gene names that are protein-coding
        self.all_human_ortho = sorted(
            g for g in self.ortho_df["Human gene name"].unique()
            if isinstance(g, str) and g in self.protein_coding
        )
        self.all_mouse_ortho = sorted(
            g for g in self.ortho_df["Gene name"].unique()
            if isinstance(g, str) and self.mouse_to_human.get(g) in self.protein_coding
        )

    def get_canonical_genes(
        self,
        gene_set: str,
        human_zero_genes: Optional[set] = None,
        mouse_zero_genes: Optional[set] = None,
    ) -> list[str]:
        """
        Return the canonical gene list (human gene symbols, sorted).

        Args:
            gene_set: "shared_orthologs" or "union_orthologs"
            human_zero_genes: genes with zero expression across all human samples
            mouse_zero_genes: genes with zero expression across all mouse samples
        """
        human_zero = human_zero_genes or set()
        mouse_zero = mouse_zero_genes or set()

        if gene_set == "shared_orthologs":
            # Only genes expressed in BOTH species
            genes = [
                g for g in self.all_human_ortho
                if g not in human_zero and g not in mouse_zero
            ]
        elif gene_set == "union_orthologs":
            # All ortholog genes, even if expressed in only one species
            genes = list(self.all_human_ortho)
        else:
            raise ValueError(f"Unknown gene_set: {gene_set!r}")

        return sorted(genes)


# ============================================================
# EXPRESSION LOADER
# ============================================================
class ExpressionLoader:
    """Loads and normalizes expression data from ARCHS4 H5 files via h5py."""

    SC_THRESHOLD = 0.5  # single-cell probability cutoff

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._exon_human = None
        self._exon_mouse = None
        self._tpm_debug_printed_labels = set()

    @property
    def exon_lengths_human(self) -> pd.Series:
        if self._exon_human is None:
            df = pd.read_csv(self.config.exon_lengths_human)
            self._exon_human = df.set_index("gene_symbol")["exon_length"]
        return self._exon_human

    @property
    def exon_lengths_mouse(self) -> pd.Series:
        if self._exon_mouse is None:
            df = pd.read_csv(self.config.exon_lengths_mouse)
            self._exon_mouse = df.set_index("gene_symbol")["exon_length"]
        return self._exon_mouse

    def _load_h5_meta(self, h5_path: str, species: str):
        """Read H5 metadata: gene symbols, GEO accessions, SC probabilities."""
        import h5py
        print(f"    Opening {h5_path}...")
        h = h5py.File(h5_path, "r")

        n_genes, n_samples = h["data/expression"].shape
        print(f"    H5 shape: {n_genes:,} genes × {n_samples:,} samples")

        print(f"    Loading gene symbols...", end=" ", flush=True)
        gene_symbols = np.array([g.decode() for g in h["meta/genes/symbol"][:]])
        print(f"done ({len(gene_symbols):,})")

        print(f"    Loading sample accessions...", end=" ", flush=True)
        geo_accessions = np.array([s.decode() for s in h["meta/samples/geo_accession"][:]])
        print(f"done ({len(geo_accessions):,})")

        sc_prob = None
        if self.config.remove_single_cell and "singlecellprobability" in h["meta/samples"]:
            print(f"    Loading SC probabilities...", end=" ", flush=True)
            sc_prob = h["meta/samples/singlecellprobability"][:]
            n_bulk = int((sc_prob < self.SC_THRESHOLD).sum())
            print(f"done ({n_bulk:,} bulk / {len(sc_prob) - n_bulk:,} SC)")

        gene_lengths = (
            self.exon_lengths_human if species == "human"
            else self.exon_lengths_mouse
        )
        print(f"    Computing gene mask...", end=" ", flush=True)
        valid_genes = set(gene_lengths.index)
        gene_mask = np.array([g in valid_genes for g in gene_symbols], dtype=bool)
        print(f"done — {gene_mask.sum():,} / {len(gene_symbols):,} genes with exon lengths")

        return h, gene_symbols, geo_accessions, sc_prob, gene_mask, gene_lengths

    def _normalize_df(
        self,
        chunk_df: pd.DataFrame,
        gene_lengths: pd.Series,
        canonical_genes: list[str],
        gene_name_map: Optional[dict] = None,
        debug_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Shared preprocessing: groupby → QC → remap → reindex.
        Keeps expression in count space; TPM/log1p_tpm is applied once later,
        after final canonical gene filtering is complete.
        Returns aligned DataFrame or empty DataFrame if nothing passes QC.
        """
        cfg = self.config

        # Aggregate duplicate gene rows
        chunk_df = chunk_df.groupby(level=0).sum()

        # QC: min non-zero genes (before remap — gene count is the same)
        nonzero = (chunk_df > 0).sum(axis=0)
        chunk_df = chunk_df[nonzero[nonzero >= cfg.qc_min_nonzero].index]

        if chunk_df.shape[1] == 0:
            return chunk_df

        # Remap gene names first (e.g. mouse -> human symbols)
        if gene_name_map is not None:
            new_idx = [gene_name_map.get(g, g) for g in chunk_df.index]
            chunk_df.index = new_idx
            chunk_df = chunk_df.groupby(level=0).sum()

        # Align to canonical ortholog gene list before normalization
        chunk_df = chunk_df.reindex(canonical_genes, fill_value=0)

        chunk_df = chunk_df.astype("float32")
        return chunk_df

    def _normalize_df_per_sample(
        self,
        chunk_df: pd.DataFrame,
        gene_lengths: pd.Series,
        canonical_genes: list[str],
        gene_name_map: Optional[dict] = None,
        debug_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fallback: process samples individually when batch processing fails.
        Only used if _normalize_df() returns empty result.
        """
        cfg = self.config
        good_samples = []
        bad_samples = []
        
        for sample_id in chunk_df.columns:
            try:
                # Single-sample DataFrame
                sample_df = chunk_df[[sample_id]].copy()
                
                # Aggregate duplicate gene rows
                sample_df = sample_df.groupby(level=0).sum()
                
                # QC: min non-zero genes
                nonzero = (sample_df > 0).sum(axis=0)[sample_id]
                if nonzero < cfg.qc_min_nonzero:
                    bad_samples.append((sample_id, f"low_genes({nonzero})"))
                    continue
                
                # Remap gene names (e.g. mouse -> human symbols)
                if gene_name_map is not None:
                    new_idx = [gene_name_map.get(g, g) for g in sample_df.index]
                    sample_df.index = new_idx
                    sample_df = sample_df.groupby(level=0).sum()
                
                # Align to canonical ortholog gene list
                sample_df = sample_df.reindex(canonical_genes, fill_value=0)
                sample_df = sample_df.astype("float32")
                
                good_samples.append(sample_df)
            except Exception as e:
                bad_samples.append((sample_id, str(e)))
        
        if bad_samples and debug_label:
            print(f"[{debug_label}] Skipped {len(bad_samples)} sketchy samples: "
                  f"{', '.join(r[0] for r in bad_samples[:3])}"
                  f"{'...' if len(bad_samples) > 3 else ''}")
        
        if not good_samples:
            return pd.DataFrame()
        
        return pd.concat(good_samples, axis=1)

    def _extract_subset(
        self,
        h5_path: str,
        species: str,
        canonical_genes: list[str],
        tmp_dir: str,
        gene_name_map: Optional[dict] = None,
    ) -> tuple[list[Path], pd.DataFrame]:
        """
        Fast subset extraction using archs4py.
        Returns (batch_paths, metadata_df).
        """
        import archs4py as a4

        cfg = self.config
        n_samples = cfg.max_samples_per_species
        t0 = time.time()

        gene_lengths = (
            self.exon_lengths_human if species == "human"
            else self.exon_lengths_mouse
        )

        print(f"    SUBSET MODE via archs4py: {n_samples:,} random samples (seed={cfg.seed})")
        print(f"    Reading from {h5_path}...", flush=True)
        raw_df = a4.data.rand(h5_path, n_samples, seed=cfg.seed, remove_sc=True)
        print(f"    Got {raw_df.shape[1]:,} bulk samples × {raw_df.shape[0]:,} genes "
              f"in {time.time() - t0:.0f}s")

        # Filter to genes with exon lengths
        valid_genes = set(gene_lengths.index)
        # IMPORTANT: use boolean mask, NOT a label list.
        # raw_df.index can have duplicate gene symbols (e.g. RPS18 ×6 isoforms).
        # Building keep=[g for g in index if g in valid_genes] puts 'RPS18' in the
        # list 6 times; raw_df.loc[keep] then returns 6×6=36 rows for that gene and
        # groupby.sum() inflates the count 6×. A boolean mask avoids that entirely.
        keep_mask = raw_df.index.isin(valid_genes)
        raw_df = raw_df.loc[keep_mask]
        print(f"    Filtered to {int(keep_mask.sum()):,} genes with exon lengths")

        # Normalize
        chunk_df = self._normalize_df(
            raw_df,
            gene_lengths,
            canonical_genes,
            gene_name_map,
            debug_label=species,
        )
        del raw_df

        if chunk_df.shape[1] == 0:
            print(f"    0 samples passed QC")
            return [], pd.DataFrame(columns=["geo_accession", "species"])

        print(f"    {chunk_df.shape[1]:,} samples passed QC")

        # Write single batch parquet as sample-major [samples, genes]
        tmp = Path(tmp_dir)
        tmp.mkdir(parents=True, exist_ok=True)
        batch_path = tmp / f"{species}_batch_0001.parquet"
        batch_out = chunk_df.T.astype("float32")
        batch_out.index.name = "geo_accession"
        batch_out.to_parquet(batch_path, compression="zstd")

        metadata = pd.DataFrame({
            "geo_accession": chunk_df.columns.tolist(),
            "species": species,
        })
        print(f"    Species done in {time.time() - t0:.0f}s — "
              f"{chunk_df.shape[1]:,} samples")
        return [batch_path], metadata

    def extract_and_normalize(
        self,
        h5_path: str,
        species: str,
        canonical_genes: list[str],
        tmp_dir: str,
        gene_name_map: Optional[dict] = None,
    ) -> tuple[list[Path], pd.DataFrame]:
        """
        Read all bulk samples from ARCHS4 H5, normalize, align.
        Writes each batch to a temporary parquet file to avoid OOM.

        For subset mode, delegates to _extract_subset (archs4py).

        Args:
            gene_name_map: Optional dict mapping H5 gene symbols to canonical names
                           (e.g. mouse→human for ortholog alignment).

        Returns:
            (batch_paths, metadata_df)
        """
        # Fast path for subset mode
        if self.config.max_samples_per_species is not None:
            return self._extract_subset(
                h5_path, species, canonical_genes, tmp_dir, gene_name_map
            )

        # === Validate H5 file structure before processing ===
        print(f"    Validating H5 file structure...")
        if not validate_h5_file(h5_path, verbose=True):
            print(f"    ⚠️  H5 file validation failed, proceeding with caution...")

        h, gene_symbols, geo_accessions, sc_prob, gene_mask, gene_lengths = \
            self._load_h5_meta(h5_path, species)

        cfg = self.config
        expr_ds = h["data/expression"]  # (n_genes, n_samples)
        n_total = expr_ds.shape[1]
        batch_size = cfg.extraction_batch_size
        gene_names = gene_symbols[gene_mask]

        batch_paths = []
        all_accessions = []
        total = 0
        total_bulk = 0
        t_species = time.time()

        n_batches = (n_total + batch_size - 1) // batch_size
        print(f"    Starting extraction: {n_batches} batches of {batch_size:,}")

        tmp = Path(tmp_dir)
        tmp.mkdir(parents=True, exist_ok=True)

        for batch_num in range(1, n_batches + 1):
            b_start = (batch_num - 1) * batch_size
            b_end = min(batch_num * batch_size, n_total)

            print(f"    Batch {batch_num}/{n_batches}: reading {b_end - b_start:,} samples from H5...",
                  end=" ", flush=True)
            
            # Try to read entire batch
            batch_read_failed = False
            successful_idxs = None
            
            try:
                raw = expr_ds[:, b_start:b_end]  # (n_genes, chunk_size)
                chunk_accessions_base = geo_accessions[b_start:b_end]
            except OSError as e:
                print(f"\n    ❌ H5 read error: {e}")
                print(f"    Running diagnosis to find recoverable samples...")
                batch_read_failed = True
                
                # Diagnose which samples fail
                successful_idxs, failed_idxs = diagnose_h5_batch_failure(
                    h5_path,
                    b_start,
                    b_end,
                    gene_mask=gene_mask,
                    verbose=True,
                )
                
                # If some samples are recoverable, read them individually
                if successful_idxs:
                    print(f"    Recovering {len(successful_idxs):,} readable samples individually...")
                    raw_parts = []
                    for sample_idx in successful_idxs:
                        try:
                            sample_data = expr_ds[:, sample_idx]
                            raw_parts.append(sample_data.reshape(-1, 1))
                        except Exception as e2:
                            print(f"      Sample {sample_idx} failed on retry: {e2}")
                            pass
                    
                    if raw_parts:
                        raw = np.hstack(raw_parts)  # (n_genes, n_recovered)
                        # Update accessions to match recovered samples
                        chunk_accessions_base = geo_accessions[b_start:b_end]
                        chunk_accessions_base = chunk_accessions_base[[s - b_start for s in successful_idxs]]
                    else:
                        print(f"    No samples recoverable in batch {batch_num}, skipping...")
                        continue
                else:
                    print(f"    No recoverable samples in batch {batch_num}, skipping entirely...")
                    continue

            # Filter to genes with exon lengths (in memory)
            raw = raw[gene_mask]

            # Filter single-cell samples
            if sc_prob is not None:
                if batch_read_failed and successful_idxs:
                    # Use pre-filtered accessions from recovery
                    bulk_mask = sc_prob[successful_idxs] < self.SC_THRESHOLD
                else:
                    bulk_mask = sc_prob[b_start:b_end] < self.SC_THRESHOLD
                raw = raw[:, bulk_mask]
                chunk_accessions = chunk_accessions_base[bulk_mask]
            else:
                chunk_accessions = chunk_accessions_base

            if raw.shape[1] == 0:
                print("skipped (all SC)")
                continue

            total_bulk += raw.shape[1]

            chunk_df = pd.DataFrame(raw, index=gene_names, columns=chunk_accessions)
            
            # Apply batch-level normalization (same as before)
            chunk_df = self._normalize_df(
                chunk_df,
                gene_lengths,
                canonical_genes,
                gene_name_map,
                debug_label=species,
            )
            
            # If batch fails, try per-sample recovery
            if chunk_df.empty:
                print(f"  Batch normalization failed, retrying per-sample...")
                chunk_df = self._normalize_df_per_sample(
                    pd.DataFrame(raw, index=gene_names, columns=chunk_accessions),
                    gene_lengths,
                    canonical_genes,
                    gene_name_map,
                    debug_label=species,
                )

            if chunk_df.shape[1] == 0:
                print(f"{raw.shape[1]} bulk, 0 passed QC")
                continue

            # Write to disk instead of keeping in RAM
            batch_path = tmp / f"{species}_batch_{batch_num:04d}.parquet"
            batch_out = chunk_df.T.astype("float32")
            batch_out.index.name = "geo_accession"
            batch_out.to_parquet(batch_path, compression="zstd")
            batch_paths.append(batch_path)
            all_accessions.extend(chunk_df.columns.tolist())
            total += chunk_df.shape[1]

            elapsed = time.time() - t_species
            frac = b_end / n_total
            eta = (elapsed / frac - elapsed) if frac > 0 else 0
            print(f"+{chunk_df.shape[1]:,} QC pass ({total:,} kept / "
                  f"{total_bulk:,} bulk) "
                  f"[{elapsed:.0f}s, ~{eta:.0f}s left]")

            del raw, chunk_df
            gc.collect()

        h.close()

        metadata = pd.DataFrame({
            "geo_accession": all_accessions,
            "species": species,
        })
        print(f"    Species done in {time.time() - t_species:.0f}s — "
              f"{total:,} samples in {len(batch_paths)} batch files")
        return batch_paths, metadata


# ============================================================
# DATASET BUILDER (main API)
# ============================================================
class RNADatasetBuilder:
    """
    End-to-end RNA-seq preprocessing pipeline.
    Outputs a single concatenated dataset; train/val/test splitting
    is deferred to scikit-learn.

    Usage:
        builder = RNADatasetBuilder(species="both", gene_set="shared_orthologs")
        builder.process()
        builder.save_parquet("output/")
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None, **kwargs):
        """
        Args:
            config: Full PreprocessingConfig, OR pass individual kwargs:
                species, gene_set, train_samples, val_samples, etc.
        """
        if config is not None:
            self.config = config
        else:
            self.config = PreprocessingConfig(**kwargs)

        self.registry = GeneRegistry(
            self.config.orthologs_file,
            self.config.protein_coding_file,
        )
        self.loader = ExpressionLoader(self.config)

        # Populated by process()
        self.canonical_genes: list[str] = []
        self.batch_paths: list[Path] = []           # temp parquet files
        self.meta: Optional[pd.DataFrame] = None
        self.zero_genes: dict[str, set] = {}        # species → set of zero-expression genes

    def _species_list(self) -> list[str]:
        if self.config.species == "both":
            return ["human", "mouse"]
        return [self.config.species]

    def _h5_path(self, species: str) -> str:
        fname = "human_gene_v2.5.h5" if species == "human" else "mouse_gene_v2.5.h5"
        return os.path.join(self.config.archs4_dir, fname)

    def _apply_final_normalization_once(self, species_batch_paths: dict[str, list[Path]]):
        """
        Apply final normalization once, after canonical gene filtering is finalized.

        For tpm/log1p_tpm, this computes TPM from counts using exon lengths in the
        final canonical gene universe. For raw_counts, no-op.
        """
        cfg = self.config
        if cfg.normalization == "raw_counts":
            return

        # Build species-specific exon-length maps in canonical (human-symbol) space.
        human_lengths = self.loader.exon_lengths_human
        mouse_lengths_human_space = self.loader.exon_lengths_mouse.rename(
            index=self.registry.mouse_to_human
        ).groupby(level=0).first()

        print("\nApplying final TPM normalization once on canonical gene list...")
        for species, paths in species_batch_paths.items():
            print(f"  [{species.upper()}] {len(paths)} batch files")
            exon_lengths = human_lengths if species == "human" else mouse_lengths_human_space
            lengths_bp = exon_lengths.reindex(self.canonical_genes)

            missing_len_mask = lengths_bp.isna()
            n_missing = int(missing_len_mask.sum())
            if n_missing > 0:
                print(
                    f"    [{species}] Missing exon lengths for {n_missing:,} canonical genes; "
                    f"these genes should already be filtered out"
                )

            keep_genes = lengths_bp.index[~missing_len_mask]
            lengths_kb = (lengths_bp.loc[keep_genes] / 1000.0)

            for i, bp in enumerate(paths, 1):
                batch_df = pd.read_parquet(bp)
                # Sample-major format: [samples, genes].
                # Restrict to final canonical genes first; missing genes are zero counts.
                batch_df = batch_df.reindex(columns=self.canonical_genes, fill_value=0.0)
                batch_use = batch_df.loc[:, keep_genes]
                rate = batch_use.astype(float).div(lengths_kb, axis=1)

                if cfg.debug_tpm_denominator and i == 1:
                    denom_dbg = rate.sum(axis=1)
                    print(
                        f"    [TPM DEBUG][{species}] genes_used={rate.shape[1]:,}, "
                        f"samples={rate.shape[0]:,}, "
                        f"denom(min/mean/max)={denom_dbg.min():.6f}/"
                        f"{denom_dbg.mean():.6f}/{denom_dbg.max():.6f}"
                    )
                    for sid, d in denom_dbg.head(3).items():
                        print(f"    [TPM DEBUG][{species}] sample={sid} denom={d:.6f}")

                denom = rate.sum(axis=1).replace(0, np.nan)
                tpm = rate.div(denom, axis=0) * 1e6
                batch_out = tpm.reindex(columns=self.canonical_genes, fill_value=0.0)

                if cfg.normalization == "log1p_tpm":
                    batch_out = np.log1p(batch_out)

                batch_out = batch_out.astype("float32")
                batch_out.index.name = "geo_accession"
                batch_out.to_parquet(bp, compression="zstd")

                if i % 20 == 0 or i == len(paths):
                    print(f"    {i}/{len(paths)} batches normalized", flush=True)

    def process(self):
        """Run the full pipeline: extract → normalize → find zero genes."""
        import shutil
        t0 = time.time()
        cfg = self.config
        species_list = self._species_list()

        # --- Settings ---
        print("=" * 70)
        print(f"PREPROCESSING SETTINGS")
        print(f"{'='*70}")
        print(f"  Species:           {cfg.species}")
        print(f"  Gene set:          {cfg.gene_set}")
        print(f"  Normalization:     {cfg.normalization}")
        print(f"  QC min nonzero:    {cfg.qc_min_nonzero:,}")
        print(f"  Remove SC:         {cfg.remove_single_cell}")
        print(f"  Batch size:        {cfg.extraction_batch_size:,}")
        print(f"  Output dir:        {cfg.output_dir}")
        if cfg.max_samples_per_species:
            print(f"  Max samples/sp:    {cfg.max_samples_per_species:,}")
        else:
            print(f"  Max samples/sp:    all")
        print(f"  Seed:              {cfg.seed}")
        print("=" * 70)

        # Start with all protein-coding orthologs
        all_ortho = self.registry.all_human_ortho
        print(f"\nProtein-coding orthologs: {len(all_ortho):,}")

        # --- Extract all samples using full gene list (stream to disk) ---
        tmp_dir = os.path.join(cfg.output_dir, "batch_files")
        print(f"\n{'='*70}")
        print(f"Extracting all bulk samples (streaming to {tmp_dir})")
        print(f"{'='*70}")

        all_batch_paths = []
        all_metas = []
        species_batch_paths: dict[str, list[Path]] = {}

        for species in species_list:
            print(f"\n  [{species.upper()}]")
            # Mouse H5 uses mouse gene symbols; remap to human via orthologs
            name_map = self.registry.mouse_to_human if species == "mouse" else None
            batch_paths, meta = self.loader.extract_and_normalize(
                self._h5_path(species),
                species,
                all_ortho,
                tmp_dir,
                gene_name_map=name_map,
            )
            all_batch_paths.extend(batch_paths)
            species_batch_paths[species] = batch_paths
            if not meta.empty:
                all_metas.append(meta)

        if not all_batch_paths:
            print("No samples extracted.")
            return

        self.batch_paths = all_batch_paths
        self.meta = pd.concat(all_metas, ignore_index=True)
        total_samples = len(self.meta)
        print(f"\nTotal after extraction: {total_samples:,} samples × "
              f"{len(all_ortho):,} genes in {len(all_batch_paths)} batch files")

        # --- Remove zero-expression genes from actual data ---
        if cfg.gene_set == "shared_orthologs":
            print(f"\nFinding zero-expression genes (scanning batch files)...")
            for species in species_list:
                # Accumulate per-gene sums across all batches for this species
                gene_sum = None
                for bp in species_batch_paths[species]:
                    batch_df = pd.read_parquet(bp)
                    # Sample-major format [samples, genes]: sum per gene is axis=0.
                    batch_sum = batch_df.sum(axis=0).reindex(all_ortho, fill_value=0)
                    if gene_sum is None:
                        gene_sum = batch_sum
                    else:
                        gene_sum = gene_sum.add(batch_sum, fill_value=0)
                    del batch_df
                    gc.collect()
                if gene_sum is not None:
                    zero = set(gene_sum.index[gene_sum == 0])
                    self.zero_genes[species] = zero
                    print(f"  {species}: {len(zero):,} genes with zero expression")

            all_zero = set()
            for zg in self.zero_genes.values():
                all_zero |= zg
            # Require exon lengths in both species for final TPM normalization.
            human_len_genes = set(self.loader.exon_lengths_human.index)
            mouse_len_genes = set(
                self.loader.exon_lengths_mouse.rename(index=self.registry.mouse_to_human)
                .groupby(level=0).first()
                .index
            )
            valid_len_genes = human_len_genes & mouse_len_genes

            self.canonical_genes = [
                g for g in all_ortho
                if g not in all_zero and g in valid_len_genes
            ]
            print(f"  Removed {len(all_zero):,} genes → {len(self.canonical_genes):,} remaining")
        else:
            human_len_genes = set(self.loader.exon_lengths_human.index)
            mouse_len_genes = set(
                self.loader.exon_lengths_mouse.rename(index=self.registry.mouse_to_human)
                .groupby(level=0).first()
                .index
            )
            valid_len_genes = human_len_genes & mouse_len_genes
            self.canonical_genes = [g for g in all_ortho if g in valid_len_genes]

        # Apply TPM/log1p_tpm exactly once on the final canonical gene universe.
        self._apply_final_normalization_once(species_batch_paths)

        print(f"\nFinal: {total_samples:,} samples × "
              f"{len(self.canonical_genes):,} genes")

        elapsed = time.time() - t0

        # --- Summary ---
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"  Species:         {cfg.species}")
        print(f"  Gene set:        {cfg.gene_set}")
        print(f"  Protein-coding:  yes ({len(self.registry.protein_coding):,} whitelist)")
        print(f"  Canonical genes: {len(self.canonical_genes):,}")
        print(f"  Total samples:   {total_samples:,}")
        for sp in species_list:
            sp_count = (self.meta["species"] == sp).sum() if self.meta is not None else 0
            print(f"    {sp:>8s}:      {sp_count:,}")
        print(f"  Normalization:   {cfg.normalization}")
        if self.zero_genes:
            print(f"  Zero-expression genes removed:")
            for sp, zg in self.zero_genes.items():
                print(f"    {sp:>8s}:      {len(zg):,}")
        print(f"  Batch files:     {len(self.batch_paths)} in {tmp_dir}")
        print(f"  Time elapsed:    {elapsed / 60:.1f} min")
        print(f"{'='*70}")

    def save_parquet(self, output_dir: Optional[str] = None):
        """
        Merge batch parquets into final output, filtering to canonical genes.

        Output:
            Batch files in {output_dir}/batch_files (samples × genes, float32, zstd)
            {output_dir}/metadata.csv          (geo_accession, species)
            {output_dir}/canonical_genes.csv   (token_id, gene_symbol)
        """
        import shutil
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Canonical gene list
        gene_df = pd.DataFrame({
            "token_id": range(1, len(self.canonical_genes) + 1),
            "gene_symbol": self.canonical_genes,
        })
        gene_df.to_csv(out / "canonical_genes.csv", index=False)

        with open(out / "genes.json", "w") as f:
            json.dump(self.canonical_genes, f)

        samples_with_meta = [
            {"id": row["geo_accession"], "species": row["species"]}
            for _, row in self.meta.iterrows()
        ]
        with open(out / "samples.json", "w") as f:
            json.dump(samples_with_meta, f)

        # Create batch manifest: which samples are in which batch files
        batch_manifest = {}
        for i, batch_path in enumerate(self.batch_paths, 1):
            df_batch = pd.read_parquet(batch_path)
            batch_name = f"batch_{i:04d}.parquet"
            sample_ids = df_batch.index.tolist()
            batch_manifest[batch_name] = sample_ids
        
        with open(out / "batch_manifest.json", "w") as f:
            json.dump(batch_manifest, f)
        
        print(f"Saved canonical_genes.csv, genes.json ({len(self.canonical_genes):,} genes), "
              f"samples.json ({len(samples_with_meta):,} samples), batch_manifest.json")

        # Metadata
        meta_path = out / "metadata.csv"
        self.meta.to_csv(meta_path, index=False)
        print(f"Saved {meta_path}: {len(self.meta):,} rows")

        # Keep batch_files directory (no cleanup)
        batch_dir = Path(self.config.output_dir) / "batch_files"
        print(f"Kept batch files in {batch_dir}: {len(self.batch_paths)} files")

        print("Done.")

    def get_data(self) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Get processed data as a numpy array by reading batch parquets.

        Returns:
            (X, metadata) where X is [samples, genes] float32 array
        """
        parts = []
        for bp in self.batch_paths:
            batch_df = pd.read_parquet(bp)
            batch_df = batch_df.reindex(columns=self.canonical_genes, fill_value=0)
            parts.append(batch_df)
        combined = pd.concat(parts, axis=0)
        X = combined.values.astype(np.float32)  # [samples, genes]
        return X, self.meta


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RNA-seq preprocessing pipeline")
    parser.add_argument("--species", default="both", choices=["human", "mouse", "both"])
    parser.add_argument("--gene-set", default="shared_orthologs",
                        choices=["shared_orthologs", "union_orthologs"])
    parser.add_argument("--output-dir", default="data/archs4/train_orthologs")
    parser.add_argument("--qc-min-nonzero", type=int, default=14_000)
    parser.add_argument("--normalization", default="tpm",
                        choices=["log1p_tpm", "tpm", "raw_counts"])
    parser.add_argument("--debug-tpm-denominator", action="store_true",
                        help="Print TPM denominator diagnostics (sum and gene count used).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per species (default: all). "
                             "Useful for quick test runs.")
    args = parser.parse_args()

    config = PreprocessingConfig(
        species=args.species,
        gene_set=args.gene_set,
        output_dir=args.output_dir,
        qc_min_nonzero=args.qc_min_nonzero,
        normalization=args.normalization,
        debug_tpm_denominator=args.debug_tpm_denominator,
        max_samples_per_species=args.max_samples,
    )

    builder = RNADatasetBuilder(config=config)
    builder.process()
    builder.save_parquet()
