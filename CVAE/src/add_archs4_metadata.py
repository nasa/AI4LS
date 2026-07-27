"""
Add Metadata to Existing ARCHS4 Pretrain H5
=============================================
The expression data is already in archs4_pretrain_nometadata.h5.
This script:
  1. Re-runs the metadata filtering on the original ARCHS4 file
     (fast — no expression reads)
  2. Verifies the filtered sample count matches the pretrain file
  3. Writes meta/genes and meta/samples into the existing pretrain file

Usage:
    python add_archs4_metadata.py \
        --archs4 mouse_bulk_gene_v2.2.h5 \
        --genelab subset_final.h5 \
        --pretrain archs4_pretrain_nometadata.h5 \
        --output archs4_pretrain.h5
"""

import argparse
import h5py
import numpy as np
import shutil
from pathlib import Path


def decode(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace").strip()
    return str(val).strip()


def run(args):
    print("=== Adding Metadata to ARCHS4 Pretrain H5 ===\n")

    # --- check pretrain file ---
    with h5py.File(args.pretrain, "r") as f:
        n_pretrain = f["data/expression"].shape[1]
        n_genes    = f["data/expression"].shape[0]
    print(f"Pretrain file: {n_pretrain:,} samples x {n_genes:,} genes")

    # --- load genelab gene metadata ---
    print("\nLoading GeneLab gene metadata...")
    with h5py.File(args.genelab, "r") as f:
        ensembl_ids = np.array([decode(v) for v in f["meta/genes/ensembl_id"][:]])
        symbols     = np.array([decode(v) for v in f["meta/genes/symbol"][:]])
    print(f"  {len(ensembl_ids):,} genes")

    # --- re-run metadata filtering on ARCHS4 ---
    print("\nLoading ARCHS4 filter fields into RAM...")
    with h5py.File(args.archs4, "r") as h5:
        n_archs4 = h5["data/expression"].shape[1]
        print(f"  ARCHS4 total samples: {n_archs4:,}")

        sc_prob  = h5["meta"]["samples"]["singlecellprobability"][:]
        lib_src  = np.array([decode(v)
                             for v in h5["meta"]["samples"]["library_source"][:]])
        print(f"  singlecellprobability: loaded")
        print(f"  library_source: loaded")

        # apply same filters as prepare_archs4.py
        mask_src = lib_src == "transcriptomic"
        mask_sc  = sc_prob < args.sc_threshold
        mask_all = mask_src & mask_sc
        valid_indices = np.where(mask_all)[0]
        n_valid = len(valid_indices)
        print(f"\n  Valid after filtering: {n_valid:,} / {n_archs4:,}")

        # --- verify count matches pretrain file ---
        if n_valid != n_pretrain:
            print(f"\nWARNING: filter gives {n_valid:,} samples but pretrain "
                  f"file has {n_pretrain:,} samples.")
            print("This likely means --sc_threshold or subsampling differed.")
            print("Attempting to match by checking if pretrain is a subsample...")

            if n_pretrain <= n_valid:
                print(f"Pretrain ({n_pretrain:,}) <= valid ({n_valid:,}) — "
                      f"assuming pretrain is a random subsample.")
                print("Using first n_pretrain valid indices as approximation.")
                print("Note: sample-level metadata may not perfectly match "
                      "if subsampling used a different seed.")
                valid_indices = valid_indices[:n_pretrain]
            else:
                raise ValueError(
                    f"Pretrain file has MORE samples ({n_pretrain:,}) than "
                    f"filtering produces ({n_valid:,}). "
                    f"Cannot reconstruct metadata. Try a lower --sc_threshold."
                )
        else:
            print(f"  ✓ Sample count matches pretrain file ({n_pretrain:,})")

        # --- read metadata for valid samples ---
        print("\nReading metadata for valid samples...")

        # library_source already loaded — just select valid rows
        # (no per-sample seeks needed)
        valid_lib_src = lib_src[valid_indices]

        # read singlecellprobability for QC
        valid_sc_prob = sc_prob[valid_indices]

        # source_name_ch1 — tissue type 
        has_source = "source_name_ch1" in h5["meta"]["samples"]
        if has_source:
            print("  Reading source_ch1...")
            source = np.array([decode(v)
                            for v in h5["meta"]["samples"]["source_name_ch1"][:]])
            valid_source = source[valid_indices]
        else:
            valid_source = np.array(["Unknown"] * n_pretrain)

        # characteristics_ch1 — contains lots of free-form metadata 
        has_ch1 = "characteristics_ch1" in h5["meta"]["samples"]
        if has_ch1:
            print("  Reading characteristics_ch1...")
            ch1 = np.array([decode(v)
                            for v in h5["meta"]["samples"]["characteristics_ch1"][:]])
            valid_ch1 = ch1[valid_indices]
        else:
            valid_ch1 = np.array(["Unknown"] * n_pretrain)

        # geo_accession — useful identifier
        has_geo = "geo_accession" in h5["meta"]["samples"]
        if has_geo:
            print("  Reading geo_accession...")
            geo = np.array([decode(v)
                            for v in h5["meta"]["samples"]["geo_accession"][:]])
            valid_geo = geo[valid_indices]
        else:
            valid_geo = np.array(["Unknown"] * n_pretrain)

        # series_id — batch identifier
        has_series = "series_id" in h5["meta"]["samples"]
        if has_series:
            print("  Reading series_id...")
            series = np.array([decode(v)
                               for v in h5["meta"]["samples"]["series_id"][:]])
            valid_series = series[valid_indices]
        else:
            valid_series = np.array(["Unknown"] * n_pretrain)

        # instrument_model — sequencing platform
        has_instrument = "instrument_model" in h5["meta"]["samples"]
        if has_instrument:
            print("  Reading instrument_model...")
            instruments = np.array([decode(v)
                                    for v in h5["meta"]["samples"]["instrument_model"][:]])
            valid_instruments = instruments[valid_indices]
        else:
            valid_instruments = np.array(["Unknown"] * n_pretrain)

    # --- copy pretrain file to output (if different path) ---
    '''if args.output != args.pretrain:
        print(f"\nCopying {args.pretrain} → {args.output}...")
        shutil.copy2(args.pretrain, args.output)
        print("  Copy complete.")
    else:
        print(f"\nWriting metadata in-place to {args.output}...")'''

    import os
    if args.output != args.pretrain:
        if os.path.exists(args.output):
            print(f"\nOutput file already exists: {args.output}")
            print("  Skipping copy — writing metadata directly into existing file.")
        else:
            print(f"\nCopying {args.pretrain} → {args.output}...")
            shutil.copy2(args.pretrain, args.output)
            print("  Copy complete.")
    else:
        print(f"\nWriting metadata in-place to {args.output}...")

    # --- write metadata into output file ---
    print("\nWriting metadata...")
    with h5py.File(args.output, "a") as out:

        # remove existing meta group if present
        if "meta" in out:
            del out["meta"]
            print("  Removed existing meta group")

        # gene metadata — same order as GeneLab model
        out.create_dataset("meta/genes/ensembl_id",
                           data=ensembl_ids.astype("S32"))
        out.create_dataset("meta/genes/symbol",
                           data=symbols.astype("S32"))
        print(f"  meta/genes: {len(ensembl_ids):,} genes written")

        # sample metadata
        # spaceflight: -1 sentinel (unknown — pretrain data)
        out.create_dataset("meta/samples/spaceflight",
                           data=np.full(n_pretrain, -1, dtype=np.int8))

        # characteristics_ch1
        out.create_dataset("meta/samples/characteristics_ch1",
                           data=np.array([s.encode('ascii', errors='ignore') for s in valid_ch1], dtype="S1024"))
                           #data=valid_ch1.astype("S32"))

        # source_name_ch1
        out.create_dataset("meta/samples/source_name_ch1",
                           data=np.array([s.encode('ascii', errors='ignore') for s in valid_source], dtype="S256"))
                           #data=valid_source.astype("S32"))

        # geo_accession
        out.create_dataset("meta/samples/geo_accession",
                           data=valid_geo.astype("S32"))

        # series_id (batch identifier — useful for QC)
        out.create_dataset("meta/samples/series_id",
                           data=valid_series.astype("S64"))

        # instrument_model
        out.create_dataset("meta/samples/instrument_model",
                           data=valid_instruments.astype("S64"))

        # singlecellprobability (keep for QC)
        out.create_dataset("meta/samples/singlecellprobability",
                           data=valid_sc_prob.astype(np.float32))

        print(f"  meta/samples: {n_pretrain:,} samples written")
        print(f"  Fields: source_name_ch1, characteristics_ch1, spaceflight, geo_accession, series_id, "
              f"instrument_model, singlecellprobability")

    # --- verify output ---
    print("\nVerifying output...")
    with h5py.File(args.output, "r") as f:
        print(f"  data/expression: {f['data/expression'].shape}")
        print(f"  meta/genes keys:   {list(f['meta/genes'].keys())}")
        print(f"  meta/samples keys: {list(f['meta/samples'].keys())}")
        print(f"  Spaceflight values: "
              f"{np.unique(f['meta/samples/spaceflight'][:])}")

    print(f"\n=== Done ===")
    print(f"Output: {args.output}")
    print(f"\nNext step — pretrain:")
    print(f"  python pretrain_archs4.py \\")
    print(f"      --data {args.output} \\")
    print(f"      --output_dir checkpoints_pretrain/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add metadata to existing ARCHS4 pretrain H5 file"
    )
    parser.add_argument("--archs4",       type=str, required=True,
                        help="Path to original ARCHS4 mouse bulk H5")
    parser.add_argument("--genelab",      type=str, required=True,
                        help="Path to subset_final.h5 (for gene metadata)")
    parser.add_argument("--pretrain",     type=str, required=True,
                        help="Path to existing pretrain H5 (expression only)")
    parser.add_argument("--output",       type=str, default=None,
                        help="Output path (default: overwrite pretrain file)")
    parser.add_argument("--sc_threshold", type=float, default=0.5,
                        help="singlecellprobability threshold used in prepare_archs4.py")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.pretrain

    run(args)
