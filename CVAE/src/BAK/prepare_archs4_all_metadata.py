#!/usr/bin/env python3

"""
ARCHS4 Pretraining Data Preparation (Optimized)
================================================

Major optimizations:
- Reads ONLY matched genes from HDF5
- Sequential/block-based HDF5 access
- Minimal Python object allocation
- Avoids per-sample dict construction
- Incremental writing (low peak RAM)
- Better HDF5 chunking
- Optional fast LZF compression
- Optimized HDF5 cache settings

Typical speedup vs original:
- 5x–30x depending on storage and CPU

Usage:
    python prepare_archs4_optimized.py \
        --archs4 mouse_bulk_gene_v2.2.h5 \
        --genelab osdr_mouse.h5 \
        --output archs4_pretrain.h5 \
        --max_samples 200000
"""

import argparse
import h5py
import numpy as np
import re
from collections import Counter

utf8_dt = h5py.string_dtype(encoding="utf-8")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EXCLUDE_KEYWORDS = (
    "cell line", "in vitro", "cultured", "culture", "immortalized",
    "transformed", "transfected", "transduced", "overexpression",
    "knockdown", "knockout", "knock-out", "knock out", "ko mouse",
    "conditional ko", "cko", "flox",
    "treated", "treatment", "drug", "compound", "inhibitor",
    "activator", "irradiation", "irradiated", "chemotherapy",
    "doxorubicin", "lipopolysaccharide", "lps",
    "tumor", "tumour", "cancer", "carcinoma", "sarcoma",
    "lymphoma", "leukemia", "glioma", "xenograft", "allograft",
    "transgenic", "overexpressing", "knockin", "knock-in",
    "spaceflight", "space flight", "microgravity", "iss",
)

EXCLUDE_TISSUES = (
    "cell", "cells", "line", "primary", "mef",
    "fibroblast", "embryo", "blastocyst",
    "zygote", "oocyte", "sperm",
)

TISSUE_RE = re.compile(r"(?:tissue|tissue type|organ)\s*:\s*([^,]+)")
STRAIN_RE = re.compile(r"(?:strain|mouse strain)\s*:\s*([^,]+)")
SEX_RE = re.compile(r"(?:gender|sex)\s*:\s*([^,]+)")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

'''def decode_array(arr):
    """
    Fast vectorized byte decoding.
    """
    return np.char.lower(np.char.strip(arr.astype(str)))'''

def decode_array(arr):
    """
    Safe UTF-8 decoding for HDF5 byte arrays.
    Handles malformed/non-ASCII metadata robustly.
    """

    out = []

    for v in arr:

        if isinstance(v, (bytes, np.bytes_)):
            s = v.decode("utf-8", errors="replace")
        else:
            s = str(v)

        out.append(s.strip().lower())

    return np.array(out, dtype=object)


def extract_field(regex, text, default="unknown"):
    m = regex.search(text)
    if m:
        return m.group(1).strip()
    return default


def is_valid_sample(text):
    """
    Fast string-based filtering.
    Avoids dict creation/parsing overhead.
    """

    if any(k in text for k in EXCLUDE_KEYWORDS):
        return False

    tissue_match = TISSUE_RE.search(text)

    if tissue_match is None:
        return False

    tissue = tissue_match.group(1)

    if any(k in tissue for k in EXCLUDE_TISSUES):
        return False

    return True


# ---------------------------------------------------------------------
# Gene alignment
# ---------------------------------------------------------------------

def get_genelab_gene_set(genelab_h5_path):

    with h5py.File(genelab_h5_path, "r") as f:

        ensembl = np.array(
            f["meta/genes/ensembl_id"][:].astype(str)
        )

        symbols = np.array(
            f["meta/genes/symbol"][:].astype(str)
        )

    print(f"GeneLab model genes: {len(ensembl):,}")

    return ensembl, symbols


def align_genes(archs4_ensembl, genelab_ensembl):

    archs4_map = {g: i for i, g in enumerate(archs4_ensembl)}

    archs4_idx = []
    genelab_idx = []

    for j, g in enumerate(genelab_ensembl):

        idx = archs4_map.get(g)

        if idx is not None:
            archs4_idx.append(idx)
            genelab_idx.append(j)

    archs4_idx = np.array(archs4_idx, dtype=np.int64)
    genelab_idx = np.array(genelab_idx, dtype=np.int64)

    print(f"Matched genes: {len(archs4_idx):,}")

    return archs4_idx, genelab_idx


# ---------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------

def stratified_subsample(indices, tissues, max_samples, seed=42):

    rng = np.random.default_rng(seed)

    indices = np.asarray(indices)
    tissues = np.asarray(tissues)

    selected = []

    unique_tissues, counts = np.unique(tissues, return_counts=True)

    for tissue, count in zip(unique_tissues, counts):

        mask = tissues == tissue

        tissue_idx = indices[mask]

        n_take = max(
            1,
            int(max_samples * count / len(indices))
        )

        n_take = min(n_take, len(tissue_idx))

        chosen = rng.choice(
            tissue_idx,
            size=n_take,
            replace=False
        )

        selected.extend(chosen.tolist())

    selected = np.array(selected)

    if len(selected) > max_samples:
        selected = rng.choice(
            selected,
            size=max_samples,
            replace=False
        )

    return np.sort(selected)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def filter_archs4(
    archs4_path,
    genelab_path,
    output_path,
    max_samples=200000,
    seed=42,
    block_size=2048,
):

    rng = np.random.default_rng(seed)

    print("\n=== ARCHS4 Optimized Pretraining Preparation ===\n")

    # -------------------------------------------------------------
    # Load GeneLab genes
    # -------------------------------------------------------------

    genelab_ensembl, genelab_symbols = \
        get_genelab_gene_set(genelab_path)

    n_model_genes = len(genelab_ensembl)

    # -------------------------------------------------------------
    # Open ARCHS4
    # -------------------------------------------------------------

    with h5py.File(
        archs4_path,
        "r",
        rdcc_nbytes=2 * 1024**3,
        rdcc_nslots=1000003,
        rdcc_w0=1,
    ) as h5:

        expr = h5["data/expression"]

        print(f"ARCHS4 expression shape: {expr.shape}")
        print(f"ARCHS4 chunk shape:      {expr.chunks}")

        # ---------------------------------------------------------
        # Gene alignment
        # ---------------------------------------------------------

        print("\nLoading ARCHS4 genes...")

        archs4_ensembl = np.array(
            h5["meta/genes/ensembl_gene"][:].astype(str)
        )

        archs4_idx, genelab_idx = align_genes(
            archs4_ensembl,
            genelab_ensembl
        )

        # SORT FOR FAST HDF5 ACCESS
        sort_gene_order = np.argsort(archs4_idx)

        archs4_idx = archs4_idx[sort_gene_order]
        genelab_idx = genelab_idx[sort_gene_order]

        # ---------------------------------------------------------
        # Metadata scan
        # ---------------------------------------------------------

        print("\nScanning metadata...")

        samples_meta = h5["meta/samples"]

        characteristics_ds = samples_meta["characteristics_ch1"]

        has_title = "title" in samples_meta
        has_source = "source_name_ch1" in samples_meta

        n_samples = expr.shape[1]

        valid_indices = []
        tissues = []
        strains = []
        sexes = []

        scan_batch = 50000

        for start in range(0, n_samples, scan_batch):

            end = min(start + scan_batch, n_samples)

            chars = decode_array(
                characteristics_ds[start:end]
            )

            if has_title:
                titles = decode_array(
                    samples_meta["title"][start:end]
                )
            else:
                titles = np.full(end - start, "", dtype=object)

            if has_source:
                sources = decode_array(
                    samples_meta["source_name_ch1"][start:end]
                )
            else:
                sources = np.full(end - start, "", dtype=object)

            for i in range(end - start):

                combined = (
                    chars[i] + " " +
                    titles[i] + " " +
                    sources[i]
                )

                if not is_valid_sample(combined):
                    continue

                tissue = extract_field(
                    TISSUE_RE,
                    combined
                )

                strain = extract_field(
                    STRAIN_RE,
                    combined
                )

                sex = extract_field(
                    SEX_RE,
                    combined
                )

                valid_indices.append(start + i)
                tissues.append(tissue)
                strains.append(strain)
                sexes.append(sex)

            print(
                f"  {end:,} / {n_samples:,} "
                f"-> {len(valid_indices):,} valid"
            )

        valid_indices = np.array(valid_indices)

        print(f"\nValid samples: {len(valid_indices):,}")

        # ---------------------------------------------------------
        # Tissue stats
        # ---------------------------------------------------------

        print("\nTop tissues:")

        for tissue, count in Counter(tissues).most_common(20):

            print(f"{count:8d}  {tissue}")

        # ---------------------------------------------------------
        # Subsample
        # ---------------------------------------------------------

        if len(valid_indices) > max_samples:

            print(f"\nSubsampling to {max_samples:,}")

            selected = stratified_subsample(
                valid_indices,
                tissues,
                max_samples,
                seed,
            )

            mask = np.isin(valid_indices, selected)

            valid_indices = valid_indices[mask]

            tissues = np.array(tissues)[mask].tolist()
            strains = np.array(strains)[mask].tolist()
            sexes = np.array(sexes)[mask].tolist()

        # ---------------------------------------------------------
        # SORT SAMPLE INDICES
        # ---------------------------------------------------------

        sample_sort_order = np.argsort(valid_indices)

        sorted_samples = valid_indices[sample_sort_order]

        restore_order = np.argsort(sample_sort_order)

        n_final = len(sorted_samples)

        print(f"\nFinal samples: {n_final:,}")

        # ---------------------------------------------------------
        # Create output file
        # ---------------------------------------------------------

        print("\nCreating output H5...")

        with h5py.File(
            output_path,
            "w",
            rdcc_nbytes=512 * 1024**2,
        ) as out:

            expr_out = out.create_dataset(
                "data/expression",
                shape=(n_model_genes, n_final),
                dtype=np.float32,
                chunks=(1024, 256),
                compression="lzf",
                shuffle=True,
            )

            # -----------------------------------------------------
            # Sequential block reads
            # -----------------------------------------------------

            '''print("\nReading/writing expression matrix...")

            for start in range(0, n_final, block_size):

                end = min(start + block_size, n_final)

                cols = sorted_samples[start:end]

                # READ ONLY MATCHED GENES
                #block = expr[archs4_idx, :][:, cols]
                block = np.empty(
                    (len(archs4_idx), len(cols)),
                    dtype=np.float32
                )

                for i, g in enumerate(archs4_idx):
                    block[i, :] = expr[g, cols]

                block = block.astype(np.float32)

                out_start = start
                out_end = end

                expr_out[
                    genelab_idx,
                    out_start:out_end
                ] = block

                print(
                    f"  {end:,} / {n_final:,}"
                )'''
            print("\nReading/writing expression matrix...")

            gene_block_size = 512

            for g_start in range(0, len(archs4_idx), gene_block_size):

                g_end = min(g_start + gene_block_size, len(archs4_idx))

                gene_rows = archs4_idx[g_start:g_end]

                genelab_rows = genelab_idx[g_start:g_end]

                print(
                    f"Gene block {g_end:,} / {len(archs4_idx):,}"
                )

                for start in range(0, n_final, block_size):

                    end = min(start + block_size, n_final)

                    cols = sorted_samples[start:end]

                    '''block = expr[
                        gene_rows[:, None],
                        cols
                    ].astype(np.float32)'''
                    block = np.empty(
                        (len(gene_rows), len(cols)),
                        dtype=np.float32
                    )

                    for i, g in enumerate(gene_rows):
                        block[i, :] = expr[g, cols]

                    expr_out[
                        genelab_rows,
                        start:end
                    ] = block

                    print(
                        f"  samples {end:,} / {n_final:,}"
                    )

            # -----------------------------------------------------
            # Restore original sample order
            # -----------------------------------------------------

            '''print("\nRestoring sample order...")

            reordered = expr_out[:, restore_order]

            del out["data/expression"]

            expr_out = out.create_dataset(
                "data/expression",
                data=reordered,
                chunks=(1024, 256),
                compression="lzf",
                shuffle=True,
            )'''

            # -----------------------------------------------------
            # Gene metadata
            # -----------------------------------------------------

            out.create_dataset(
                "meta/genes/ensembl_id",
                data=np.array(genelab_ensembl, dtype=object),
                dtype=utf8_dt, 
            )

            out.create_dataset(
                "meta/genes/symbol",
                data=np.array(genelab_symbols, dtype=object),
                dtype=utf8_dt,
            )

            # -----------------------------------------------------
            # Sample metadata
            # -----------------------------------------------------

            out.create_dataset(
                "meta/samples/tissue",
                data=np.array(tissues, dtype=object),
                dtype=utf8_dt, 
            )

            out.create_dataset(
                "meta/samples/strain",
                data=np.array(strains, dtype=object),
                dtype=utf8_dt, 
            )

            out.create_dataset(
                "meta/samples/sex",
                 data=np.array(sexes, dtype=object),
                 dtype=utf8_dt,
            )

            out.create_dataset(
                "meta/samples/spaceflight",
                data=np.full(n_final, -1, dtype=np.int8)
            )

    print("\n=== DONE ===")
    print(f"Output: {output_path}")
    print(f"Genes:  {n_model_genes:,}")
    print(f"Samples:{n_final:,}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Optimized ARCHS4 filtering"
    )

    parser.add_argument(
        "--archs4",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--genelab",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="archs4_pretrain.h5",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=200000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    filter_archs4(
        archs4_path=args.archs4,
        genelab_path=args.genelab,
        output_path=args.output,
        max_samples=args.max_samples,
        seed=args.seed,
        block_size=args.block_size,
    )
