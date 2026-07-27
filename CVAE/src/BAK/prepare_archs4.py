"""
ARCHS4 Pretraining Data Preparation (v4)
=========================================
Filters ARCHS4 mouse bulk RNA-seq H5 to high-quality bulk samples,
harmonizes tissue labels to the GeneLab 25-tissue vocabulary,
aligns to the GeneLab model gene set, and writes a pretrain-ready H5.

Filters:
  1. library_source == "transcriptomic"
  2. singlecellprobability < 0.5

Tissue harmonization:
  - Parses tissue from characteristics_ch1
  - Maps to GeneLab 25-tissue vocabulary using keyword matching
  - Samples with no tissue field get label "Unknown" (still included)
  - Samples whose tissue maps to an excluded category are still included
    as "Unknown" (not dropped — we want all 539K samples)

Usage:
    python prepare_archs4.py \\
        --archs4 mouse_bulk_gene_v2.2.h5 \\
        --genelab subset_final.h5 \\
        --output archs4_pretrain_v4.h5
"""

import argparse
import h5py
import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode(val):
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace").strip()
    return str(val).strip()


# ---------------------------------------------------------------------------
# Tissue harmonization to GeneLab vocabulary
# ---------------------------------------------------------------------------

# Maps ARCHS4 tissue strings (lowercase) to GeneLab tissue categories.
# GeneLab tissues: Adipose, Adrenal Gland, Bone, Bone Marrow, Brain,
#   Cecum, Cerebellum, Colon, EDL, Gastrocnemius, Heart, Hippocampus,
#   Kidney, Liver, Lung, Mammary Gland, Optic Nerve, Other, Quadriceps,
#   Retina, Skin, Soleus, Spleen, Thymus, Tibialis
# Plus "Unknown" for samples with no parseable tissue

TISSUE_MAP = {
    # Adipose
    "adipose":                              "Adipose",
    "epididymal white adipose tissue":      "Adipose",
    "white adipose tissue":                 "Adipose",
    "white adipose tissue around gonad":    "Adipose",
    "brown adipose tissue":                 "Adipose",
    "inguinal adipose tissue":              "Adipose",
    "visceral adipose":                     "Adipose",
    "fat":                                  "Adipose",
    # Adrenal Gland
    "adrenal gland":                        "Adrenal Gland",
    "adrenal":                              "Adrenal Gland",
    # Bone
    "bone":                                 "Bone",
    "femur":                                "Bone",
    "tibia":                                "Bone",
    "calvaria":                             "Bone",
    # Bone Marrow
    "bone marrow":                          "Bone Marrow",
    "whole bone marrow":                    "Bone Marrow",
    # Brain (cortex / whole brain / non-specific regions)
    "brain":                                "Brain",
    "whole brain":                          "Brain",
    "cortex":                               "Brain",
    "prefrontal cortex":                    "Brain",
    "visual cortex":                        "Brain",
    "frontal cortex":                       "Brain",
    "motor cortex":                         "Brain",
    "striatum":                             "Brain",
    "subventricular zone":                  "Brain",
    "medial ganglionic eminence":           "Brain",
    "olfactory bulb":                       "Brain",
    "thalamus":                             "Brain",
    "hypothalamus":                         "Brain",
    "brainstem":                            "Brain",
    "midbrain":                             "Brain",
    "forebrain":                            "Brain",
    "hindbrain":                            "Brain",
    "amygdala":                             "Brain",
    "basal ganglia":                        "Brain",
    "substantia nigra":                     "Brain",
    # Cecum
    "cecum":                                "Cecum",
    "caecum":                               "Cecum",
    # Cerebellum
    "cerebellum":                           "Cerebellum",
    "brain - cerebellum":                   "Cerebellum",
    # Colon
    "colon":                                "Colon",
    "large intestine":                      "Colon",
    "proximal colon":                       "Colon",
    "distal colon":                         "Colon",
    # EDL
    "edl":                                  "EDL",
    "extensor digitorum longus":            "EDL",
    # Gastrocnemius
    "gastrocnemius":                        "Gastrocnemius",
    "muscle":                               "Gastrocnemius",
    "skeletal muscle":                      "Gastrocnemius",
    "tibialis anterior":                    "Gastrocnemius",
    # Heart
    "heart":                                "Heart",
    "cardiac ventricles":                   "Heart",
    "whole-heart tissue":                   "Heart",
    "left ventricle":                       "Heart",
    "right ventricle":                      "Heart",
    "cardiac muscle":                       "Heart",
    "myocardium":                           "Heart",
    # Hippocampus
    "hippocampus":                          "Hippocampus",
    "brain - hippocampus":                  "Hippocampus",
    # Kidney
    "kidney":                               "Kidney",
    # Liver
    "liver":                                "Liver",
    # Lung
    "lung":                                 "Lung",
    "whole lung":                           "Lung",
    # Mammary Gland
    "mammary gland":                        "Mammary Gland",
    "mammary":                              "Mammary Gland",
    # Optic Nerve
    "optic nerve":                          "Optic Nerve",
    # Quadriceps
    "quadriceps":                           "Quadriceps",
    "quad":                                 "Quadriceps",
    # Retina
    "retina":                               "Retina",
    # Skin
    "skin":                                 "Skin",
    "dermis":                               "Skin",
    "epidermis":                            "Skin",
    # Soleus
    "soleus":                               "Soleus",
    # Spleen
    "spleen":                               "Spleen",
    # Thymus
    "thymus":                               "Thymus",
    # Tibialis
    "tibialis":                             "Tibialis",
    # Other — valid tissues not in GeneLab vocabulary
    "pancreas":                             "Other",
    "pancreatic islets":                    "Other",
    "small intestine":                      "Other",
    "duodenum":                             "Other",
    "jejunum":                              "Other",
    "ileum":                                "Other",
    "spinal cord":                          "Other",
    "testis":                               "Other",
    "testes":                               "Other",
    "ovary":                                "Other",
    "uterus":                               "Other",
    "thyroid":                              "Other",
    "lymph nodes":                          "Other",
    "lymph node":                           "Other",
    "blood":                                "Other",
    "peripheral blood":                     "Other",
    "aorta":                                "Other",
    "bladder":                              "Other",
    "stomach":                              "Other",
    "esophagus":                            "Other",
    "prostate":                             "Other",
    "placenta":                             "Other",
    "diaphragm":                            "Other",
    "tongue":                               "Other",
    "eye":                                  "Other",
    "inner ear":                            "Other",
    "cochlea":                              "Other",
    "dorsal root ganglion":                 "Other",
    "dorsal root ganglion (drg) neurons":   "Other",
    "whole embryo":                         "Other",
    "embryo":                               "Other",
}

# Substrings for partial matching (checked if exact match fails)
TISSUE_CONTAINS = [
    ("hippocamp",    "Hippocampus"),
    ("cerebell",     "Cerebellum"),
    ("cortex",       "Brain"),
    ("striatum",     "Brain"),
    ("hypothalam",   "Brain"),
    ("bone marrow",  "Bone Marrow"),
    ("adipose",      "Adipose"),
    ("mammary",      "Mammary Gland"),
    ("gastrocnem",   "Gastrocnemius"),
    ("quadricep",    "Quadriceps"),
    ("tibialis",     "Tibialis"),
    ("soleus",       "Soleus"),
    ("liver",        "Liver"),
    ("kidney",       "Kidney"),
    ("spleen",       "Spleen"),
    ("thymus",       "Thymus"),
    ("retina",       "Retina"),
    ("lung",         "Lung"),
    ("heart",        "Heart"),
    ("colon",        "Colon"),
    ("skin",         "Skin"),
    ("brain",        "Brain"),
    ("muscle",       "Gastrocnemius"),
    ("pancrea",      "Other"),
    ("intestin",     "Other"),
    ("spinal",       "Other"),
    ("testis",       "Other"),
    ("testes",       "Other"),
    ("lymph",        "Other"),
    ("embryo",       "Other"),
    ("blood",        "Other"),
]


def harmonize_tissue(raw_tissue_str):
    """
    Map a raw ARCHS4 tissue string to a GeneLab tissue category.

    Returns:
        str: GeneLab tissue category, or "Unknown" if no match
    """
    if not raw_tissue_str:
        return "Unknown"

    t = raw_tissue_str.lower().strip()

    # exact match first
    if t in TISSUE_MAP:
        return TISSUE_MAP[t]

    # partial match
    for substr, category in TISSUE_CONTAINS:
        if substr in t:
            return category

    return "Unknown"


def parse_tissue_from_source(source_str):
    """
    Use source_name_ch1 directly as tissue label.
    This field often contains a clean tissue name like 'Liver' or 'FrontalLobe'.
    Returns the raw string (lowercased) for harmonization.
    """
    s = source_str.strip().lower()
    # exclude obvious non-tissue source names
    exclude = {"unknown", "", "n/a", "na", "none", "mouse", "mus musculus",
               "whole body", "whole animal", "whole organism"}
    if s in exclude or len(s) < 2:
        return ""
    return s


def parse_tissue_from_characteristics(char_str):
    """Parse tissue from characteristics_ch1 key:value pairs."""
    for part in char_str.split(","):
        part = part.strip()
        if ":" in part:
            key, _, val = part.partition(":")
            key = key.strip().lower()
            if key in ("tissue", "tissue type", "organ", "tissue source",
                       "tissue/cell type", "cell type", "body site"):
                return val.strip().lower()
    return ""


def get_tissue(source_str, char_str):
    """
    Try source_name_ch1 first, fall back to characteristics_ch1.
    Returns harmonized GeneLab tissue category or "Unknown".
    """
    # try source_name_ch1 first
    raw = parse_tissue_from_source(source_str)
    if raw:
        harmonized = harmonize_tissue(raw)
        if harmonized != "Unknown":
            return harmonized

    # fall back to characteristics_ch1
    raw = parse_tissue_from_characteristics(char_str)
    if raw:
        return harmonize_tissue(raw)

    return "Unknown"


# ---------------------------------------------------------------------------
# Gene alignment
# ---------------------------------------------------------------------------

def get_genelab_gene_set(genelab_h5_path):
    with h5py.File(genelab_h5_path, "r") as f:
        ensembl = np.array([decode(v) for v in f["meta/genes/ensembl_id"][:]])
        symbol  = np.array([decode(v) for v in f["meta/genes/symbol"][:]])
    print(f"GeneLab gene set: {len(ensembl):,} genes")
    return ensembl, symbol


def align_genes(archs4_ensembl, genelab_ensembl):
    archs4_map  = {g: i for i, g in enumerate(archs4_ensembl)}
    archs4_idx  = []
    genelab_idx = []
    for j, g in enumerate(genelab_ensembl):
        if g in archs4_map:
            genelab_idx.append(j)
            archs4_idx.append(archs4_map[g])
    n = len(archs4_idx)
    print(f"Genes matched:  {n:,} / {len(genelab_ensembl):,}")
    print(f"Zero-filled:    {len(genelab_ensembl) - n:,} unmatched genes")
    return np.array(archs4_idx), np.array(genelab_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Specific (gene_start, col_start, col_end) regions that are unreadable
# in mouse_bulk_gene_v2.2.h5 — identified by exhaustive chunk scan.
# These combinations will be skipped and zero-filled in the output.
CORRUPT_REGIONS = [
    (2000,  4000,  900000, 997515),   # genes 2000-4000,  cols 900000-997515
    (12000, 14000, 300000, 400000),   # genes 12000-14000, cols 300000-400000
]

def is_corrupt(gene_start, gene_end, col_start, col_end):
    """Return True if this read overlaps a known corrupted region."""
    for cg_s, cg_e, cc_s, cc_e in CORRUPT_REGIONS:
        if gene_start < cg_e and gene_end > cg_s:   # gene overlap
            if col_start < cc_e and col_end > cc_s:  # col overlap
                return True
    return False


def filter_archs4(archs4_path, genelab_path, output_path,
                  sc_prob_threshold=0.5, seed=42):

    np.random.seed(seed)
    print("=== ARCHS4 Pretraining Data Preparation (v4) ===\n")

    genelab_ensembl, genelab_symbols = get_genelab_gene_set(genelab_path)
    n_model_genes = len(genelab_ensembl)

    with h5py.File(archs4_path, "r") as h5:

        # gene alignment
        print("\nAligning genes...")
        archs4_ensembl   = np.array([decode(v)
                                     for v in h5["meta"]["genes"]["ensembl_gene"][:]])
        archs4_idx, genelab_idx = align_genes(archs4_ensembl, genelab_ensembl)

        archs4_expr      = h5["data"]["expression"]
        n_archs4_genes   = archs4_expr.shape[0]
        n_archs4_samples = archs4_expr.shape[1]
        chunk_size       = archs4_expr.chunks[0]
        print(f"ARCHS4: {n_archs4_genes:,} genes x {n_archs4_samples:,} samples")

        # load filter fields
        print("\nLoading filter fields into RAM...")
        sc_prob = h5["meta"]["samples"]["singlecellprobability"][:]
        lib_src = np.array([decode(v)
                            for v in h5["meta"]["samples"]["library_source"][:]])
        print(f"  Loaded singlecellprobability and library_source")

        # load metadata for tissue parsing — try source_name_ch1 first
        print("  Loading source_name_ch1...")
        has_source = "source_name_ch1" in h5["meta"]["samples"]
        if has_source:
            sources = [decode(v) for v in h5["meta"]["samples"]["source_name_ch1"][:]]
            print(f"  Loaded {len(sources):,} source_name_ch1 values")
        else:
            sources = [""] * n_archs4_samples
            print("  source_name_ch1 not found — will use characteristics_ch1 only")

        print("  Loading characteristics_ch1 for tissue fallback...")
        chars = [decode(v) for v in h5["meta"]["samples"]["characteristics_ch1"][:]]
        print(f"  Loaded {len(chars):,} characteristics strings")

        # apply quality filters
        print("\nApplying quality filters...")
        mask_src = lib_src == "transcriptomic"
        mask_sc  = sc_prob < sc_prob_threshold
        mask_all = mask_src & mask_sc
        valid_indices = np.where(mask_all)[0]
        n_valid = len(valid_indices)

        print(f"  library_source == transcriptomic: {mask_src.sum():,}")
        print(f"  singlecellprobability < {sc_prob_threshold}: {mask_sc.sum():,}")
        print(f"  Both filters: {n_valid:,} / {n_archs4_samples:,} "
              f"({n_valid/n_archs4_samples:.1%})")

        # parse and harmonize tissue labels
        # source_name_ch1 is tried first, characteristics_ch1 as fallback
        print("\nHarmonizing tissue labels...")
        tissue_labels  = []
        n_from_source  = 0
        n_from_chars   = 0
        n_has_tissue   = 0

        for i in valid_indices:
            source_raw = sources[i] if has_source else ""
            char_raw   = chars[i]

            # try source_name_ch1 first
            raw_s = parse_tissue_from_source(source_raw)
            harmonized_s = harmonize_tissue(raw_s) if raw_s else "Unknown"

            if harmonized_s != "Unknown":
                tissue_labels.append(harmonized_s)
                n_from_source += 1
                n_has_tissue  += 1
            else:
                # fall back to characteristics_ch1
                raw_c = parse_tissue_from_characteristics(char_raw)
                harmonized_c = harmonize_tissue(raw_c) if raw_c else "Unknown"
                tissue_labels.append(harmonized_c)
                if harmonized_c != "Unknown":
                    n_from_chars += 1
                    n_has_tissue += 1

        tissue_counts = Counter(tissue_labels)
        n_final = len(valid_indices)
        print(f"  Samples with mapped tissue: {n_has_tissue:,} / {n_final:,} "
              f"({n_has_tissue/n_final:.1%})")
        print(f"    from source_name_ch1:     {n_from_source:,}")
        print(f"    from characteristics_ch1: {n_from_chars:,}")
        print(f"  Tissue distribution (top 30):")
        for t, c in tissue_counts.most_common(30):
            print(f"    {c:8,}  {t}")

        # sort indices for sequential H5 access
        sort_order    = np.argsort(valid_indices)
        sorted_idx    = valid_indices[sort_order]
        restore_order = np.argsort(sort_order)

        # Sequential gene-chunk x column-batch reads.
        # Reading archs4_expr[rows, sorted_idx] fails because HDF5
        # cannot handle non-contiguous fancy column indexing on this
        # chunk layout. Instead we read contiguous column slices and
        # select our samples in memory.
        #
        # Each sub-read: chunk_size genes x COL_BATCH samples
        #   = 2000 x 100000 x 4 bytes = ~0.8 GB — well within RAM.
        # Total reads: n_gene_chunks x n_col_batches
        #   = 27 x ceil(997515/100000) = 27 x 11 = 297 sequential reads.

        COL_BATCH = 100000
        n_col_batches = int(np.ceil(n_archs4_samples / COL_BATCH))
        n_chunks      = int(np.ceil(n_archs4_genes / chunk_size))

        print(f"\nReading expression data...")
        print(f"  Gene chunks:   {n_chunks}")
        print(f"  Column batches: {n_col_batches} x {COL_BATCH:,} samples")
        print(f"  Sub-read size: ~{chunk_size * COL_BATCH * 4 / 1e9:.2f} GB each")
        print(f"  Total reads:   {n_chunks * n_col_batches}")

        expr_model = np.zeros((len(archs4_idx), n_final), dtype=np.float32)

        for chunk_i, gene_start in enumerate(range(0, n_archs4_genes, chunk_size)):
            gene_end = min(gene_start + chunk_size, n_archs4_genes)
            mask     = (archs4_idx >= gene_start) & (archs4_idx < gene_end)
            if not mask.any():
                print(f"  Gene chunk {chunk_i+1:2d}/{n_chunks} "
                      f"[{gene_start:,}-{gene_end:,}] — no model genes, skip")
                continue

            local_gene_idx  = archs4_idx[mask] - gene_start
            output_gene_idx = np.where(mask)[0]
            n_model_in_chunk = mask.sum()

            # accumulate across column batches
            chunk_result = np.zeros((n_model_in_chunk, n_final), dtype=np.float32)

            for col_start in range(0, n_archs4_samples, COL_BATCH):
                # skip corrupted region entirely — never read past CORRUPT_START
                col_end = min(col_start + COL_BATCH, n_archs4_samples)

                # which of our sorted valid indices fall in this col batch?
                in_batch = (sorted_idx >= col_start) & (sorted_idx < col_end)
                if not in_batch.any():
                    continue

                # skip known corrupted regions — zero-fill affected genes/samples
                if is_corrupt(gene_start, gene_end, col_start, col_end):
                    print(f"    Skipping corrupt region: "
                          f"genes {gene_start:,}-{gene_end:,} "
                          f"cols {col_start:,}-{col_end:,} (zero-filled)")
                    continue

                # contiguous read — no fancy indexing
                sub = archs4_expr[gene_start:gene_end,
                                   col_start:col_end]        # (chunk_size, COL_BATCH)

                # select model genes and our samples in memory
                local_col_idx = sorted_idx[in_batch] - col_start
                sub_selected  = sub[local_gene_idx, :][:, local_col_idx]

                # place into output at correct restored positions
                out_positions = restore_order[in_batch]
                chunk_result[:, out_positions] = sub_selected.astype(np.float32)

            expr_model[output_gene_idx, :] = chunk_result

            print(f"  Gene chunk {chunk_i+1:2d}/{n_chunks} "
                  f"[{gene_start:,}-{gene_end:,}] "
                  f"— {n_model_in_chunk} model genes extracted")

        print(f"\nExpression read complete. Shape: {expr_model.shape}")

    # write output H5
    print(f"\nWriting {output_path}...")
    with h5py.File(output_path, "w") as out:

        expr_out = out.create_dataset(
            "data/expression",
            shape=(n_model_genes, n_final),
            dtype=np.float32,
            chunks=(n_model_genes, 1),
            compression=None,
        )
        expr_out[genelab_idx, :] = expr_model
        del expr_model
        print("  Expression written.")

        # gene metadata
        out.create_dataset("meta/genes/ensembl_id",
                           data=genelab_ensembl.astype("S32"))
        out.create_dataset("meta/genes/symbol",
                           data=genelab_symbols.astype("S32"))

        # sample metadata
        out.create_dataset("meta/samples/spaceflight",
                           data=np.full(n_final, -1, dtype=np.int8))
        out.create_dataset("meta/samples/tissue",
                           data=np.array(tissue_labels).astype("S32"))

        print("  Metadata written.")

    print(f"\n=== Done ===")
    print(f"Output:      {output_path}")
    print(f"Samples:     {n_final:,}")
    print(f"Genes:       {n_model_genes:,}")
    print(f"With tissue: {n_has_tissue:,} ({n_has_tissue/n_final:.1%})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter ARCHS4 for cVAE pretraining with tissue harmonization"
    )
    parser.add_argument("--archs4",       type=str, required=True)
    parser.add_argument("--genelab",      type=str, required=True)
    parser.add_argument("--output",       type=str, default="archs4_pretrain_v4.h5")
    parser.add_argument("--sc_threshold", type=float, default=0.5)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    filter_archs4(
        archs4_path=args.archs4,
        genelab_path=args.genelab,
        output_path=args.output,
        sc_prob_threshold=args.sc_threshold,
        seed=args.seed,
    )
