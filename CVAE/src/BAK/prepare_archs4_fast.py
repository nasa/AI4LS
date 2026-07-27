#!/usr/bin/env python3

"""
ARCHS4 Bulk RNA-seq Pretraining Preparation
===========================================

Optimized for ARCHS4 HDF5 layout:
    expression.chunks == (2000, 1)

Key Features
------------
- Keeps only likely BULK RNA-seq samples
- Removes likely single-cell samples
- Aligns genes to GeneLab model gene set
- Streams directly from ARCHS4 -> output H5
- No giant in-memory expression matrix
- Stable memory usage
- Efficient sequential HDF5 access

Filters
-------
1. library_source == "transcriptomic"
2. singlecellprobability < threshold

Output
------
Creates:
    data/expression
    meta/genes/*
    meta/samples/*

Usage
-----
python prepare_archs4_bulk.py \
    --archs4 mouse_gene_v2.5.h5 \
    --genelab subset_final.h5 \
    --output archs4_pretrain.h5
"""

import argparse
from collections import Counter

import h5py
import numpy as np

import gc


# ---------------------------------------------------------------------
# UTF8 HDF5 string type
# ---------------------------------------------------------------------

UTF8 = h5py.string_dtype(encoding="utf-8")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def decode(val):

    if isinstance(val, (bytes, np.bytes_)):
        return val.decode(
            "utf-8",
            errors="replace"
        ).strip()

    return str(val).strip()


def decode_array(arr):

    out = []

    for v in arr:

        if isinstance(v, (bytes, np.bytes_)):
            s = v.decode(
                "utf-8",
                errors="replace"
            )
        else:
            s = str(v)

        out.append(s.strip())

    return np.array(
        out,
        dtype=object
    )


# ---------------------------------------------------------------------
# Gene alignment
# ---------------------------------------------------------------------

def get_genelab_gene_set(genelab_h5_path):

    with h5py.File(genelab_h5_path, "r") as f:

        ensembl = decode_array(
            f["meta/genes/ensembl_id"][:]
        )

        symbols = decode_array(
            f["meta/genes/symbol"][:]
        )

    print(
        f"GeneLab gene set: "
        f"{len(ensembl):,} genes"
    )

    return ensembl, symbols


def align_genes(
    archs4_ensembl,
    genelab_ensembl
):

    archs4_map = {
        g: i
        for i, g in enumerate(archs4_ensembl)
    }

    archs4_idx = []
    genelab_idx = []

    for j, g in enumerate(genelab_ensembl):

        idx = archs4_map.get(g)

        if idx is not None:

            archs4_idx.append(idx)
            genelab_idx.append(j)

    archs4_idx = np.array(
        archs4_idx,
        dtype=np.int64
    )

    genelab_idx = np.array(
        genelab_idx,
        dtype=np.int64
    )

    print(
        f"Genes matched: "
        f"{len(archs4_idx):,} / "
        f"{len(genelab_ensembl):,}"
    )

    return archs4_idx, genelab_idx


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def filter_archs4(
    archs4_path,
    genelab_path,
    output_path,
    sc_prob_threshold=0.5,
    sample_block_size=8192,
):

    print(
        "\n=== ARCHS4 Bulk RNA-seq Preparation ===\n"
    )

    # -------------------------------------------------------------
    # GeneLab gene set
    # -------------------------------------------------------------

    (
        genelab_ensembl,
        genelab_symbols
    ) = get_genelab_gene_set(
        genelab_path
    )

    n_model_genes = len(
        genelab_ensembl
    )

    # -------------------------------------------------------------
    # Open ARCHS4
    # -------------------------------------------------------------

    with h5py.File(
        archs4_path,
        "r",
        rdcc_nbytes=512 * 1024**2,
        rdcc_nslots=1000003,
        rdcc_w0=1,
    ) as h5:

        expr = h5["data/expression"]

        n_archs4_genes = expr.shape[0]
        n_archs4_samples = expr.shape[1]

        gene_chunk_size = expr.chunks[0]

        print(
            f"ARCHS4 expression shape: "
            f"{expr.shape}"
        )

        print(
            f"ARCHS4 chunk layout: "
            f"{expr.chunks}"
        )

        print(
            f"ARCHS4: "
            f"{n_archs4_genes:,} genes x "
            f"{n_archs4_samples:,} samples"
        )

        # ---------------------------------------------------------
        # Gene alignment
        # ---------------------------------------------------------

        print("\nLoading ARCHS4 genes...")

        archs4_ensembl = decode_array(
            h5["meta/genes/ensembl_gene"][:]
        )

        (
            archs4_idx,
            genelab_idx
        ) = align_genes(
            archs4_ensembl,
            genelab_ensembl
        )

        # Sort genes for monotonic access
        gene_order = np.argsort(
            archs4_idx
        )

        archs4_idx = archs4_idx[
            gene_order
        ]

        genelab_idx = genelab_idx[
            gene_order
        ]

        # ---------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------

        print("\nLoading metadata...")

        meta = h5["meta/samples"]

        sc_prob = meta[
            "singlecellprobability"
        ][:]

        lib_src = decode_array(
            meta["library_source"][:]
        )

        print(
            f"Loaded metadata for "
            f"{len(lib_src):,} samples"
        )

        print(
            f"Library sources: "
            f"{Counter(lib_src).most_common(10)}"
        )

        # ---------------------------------------------------------
        # Filters
        # ---------------------------------------------------------

        print("\nApplying filters...")

        mask_src = (
            lib_src == "transcriptomic"
        )

        mask_sc = (
            sc_prob < sc_prob_threshold
        )

        mask = mask_src & mask_sc

        valid_indices = np.where(mask)[0]

        n_final = len(valid_indices)

        print(
            f"\nBulk-like samples retained: "
            f"{n_final:,} / "
            f"{n_archs4_samples:,} "
            f"({n_final / n_archs4_samples:.1%})"
        )

        # IMPORTANT:
        # sort for monotonic HDF5 access
        sorted_samples = np.sort(
            valid_indices
        )

        # ---------------------------------------------------------
        # Create output H5
        # ---------------------------------------------------------

        print("\nCreating output H5...")

        with h5py.File(
            output_path,
            "w",
            rdcc_nbytes=512 * 1024**2,
        ) as out:

            expr_out = out.create_dataset(
                "data/expression",
                shape=(
                    n_model_genes,
                    n_final
                ),
                dtype=np.float32,
                chunks=(2048, 4096),
                compression="lzf",
                shuffle=True,
            )

            # -----------------------------------------------------
            # Stream expression matrix
            # -----------------------------------------------------

            print(
                "\nStreaming expression matrix..."
            )

            n_sample_blocks = int(
                np.ceil(
                    n_final /
                    sample_block_size
                )
            )

            n_gene_chunks = int(
                np.ceil(
                    n_archs4_genes /
                    gene_chunk_size
                )
            )

            for s_i, s_start in enumerate(
                range(
                    0,
                    n_final,
                    sample_block_size
                )
            ):

                s_end = min(
                    s_start +
                    sample_block_size,
                    n_final
                )

                cols = sorted_samples[
                    s_start:s_end
                ]

                print(
                    f"\nSample block "
                    f"{s_i+1}/{n_sample_blocks} "
                    f"({s_end:,} / {n_final:,})"
                )

                for chunk_i, gene_start in enumerate(
                    range(
                        0,
                        n_archs4_genes,
                        gene_chunk_size
                    )
                ):

                    gene_end = min(
                        gene_start +
                        gene_chunk_size,
                        n_archs4_genes
                    )

                    # model genes in this chunk
                    mask = (
                        (archs4_idx >= gene_start)
                        &
                        (archs4_idx < gene_end)
                    )

                    if not np.any(mask):
                        continue

                    local_archs4 = (
                        archs4_idx[mask]
                        - gene_start
                    )

                    out_rows = genelab_idx[
                        mask
                    ]

                    # -------------------------------------------------
                    # Bounded chunk-aligned read
                    # -------------------------------------------------

                    chunk = expr[
                        gene_start:gene_end,
                        cols
                    ]

                    block = chunk[
                        local_archs4,
                        :
                    ]

                    expr_out[
                        out_rows,
                        s_start:s_end
                    ] = block.astype(
                        np.float32
                    )

                    print(
                        f"  Chunk "
                        f"{chunk_i+1}/{n_gene_chunks} "
                        f"genes "
                        f"{gene_start:,}-"
                        f"{gene_end:,}"
                    )
                    del chunk
                    del block
                gc.collect()


            # -----------------------------------------------------
            # Gene metadata
            # -----------------------------------------------------

            out.create_dataset(
                "meta/genes/ensembl_id",
                data=np.array(
                    genelab_ensembl,
                    dtype=object
                ),
                dtype=UTF8,
            )

            out.create_dataset(
                "meta/genes/symbol",
                data=np.array(
                    genelab_symbols,
                    dtype=object
                ),
                dtype=UTF8,
            )

            # -----------------------------------------------------
            # Sample metadata
            # -----------------------------------------------------

            out.create_dataset(
                "meta/samples/archs4_sample_index",
                data=sorted_samples.astype(
                    np.int64
                )
            )

            out.create_dataset(
                "meta/samples/spaceflight",
                data=np.full(
                    n_final,
                    -1,
                    dtype=np.int8
                )
            )

    print("\n=== DONE ===")

    print(f"Output: {output_path}")

    print(
        f"Expression: "
        f"({n_model_genes:,} genes x "
        f"{n_final:,} samples)"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Prepare ARCHS4 bulk RNA-seq "
            "for pretraining"
        )
    )

    parser.add_argument(
        "--archs4",
        type=str,
        required=True,
        help="ARCHS4 mouse H5"
    )

    parser.add_argument(
        "--genelab",
        type=str,
        required=True,
        help="GeneLab subset_final.h5"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="archs4_pretrain.h5",
        help="Output H5"
    )

    parser.add_argument(
        "--sc_threshold",
        type=float,
        default=0.5,
        help=(
            "Max singlecellprobability "
            "to retain"
        )
    )

    parser.add_argument(
        "--sample_block_size",
        type=int,
        default=8192,
        help="Sample block size"
    )

    args = parser.parse_args()

    filter_archs4(
        archs4_path=args.archs4,
        genelab_path=args.genelab,
        output_path=args.output,
        sc_prob_threshold=args.sc_threshold,
        sample_block_size=args.sample_block_size,
    )
