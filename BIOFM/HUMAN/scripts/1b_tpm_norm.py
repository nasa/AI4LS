#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Compute canonical per-gene exon lengths from GENCODE (v49).

What this notebook does:
- Load GENCODE GTF, keep only exon features.
- Merge overlapping exons per gene (genomic union) to avoid double-counting.
- Sum merged intervals to get total exon_length_bp per gene.
- Save a CSV for downstream use.

Reasons:
- Exon lengths are required for TPM normalization (reads per kilobase) in the training/eval pipeline.
- Provides a consistent, canonical gene set and length reference across splits.

Inputs:  data/gencode/gencode.v49.basic.annotation.gtf.gz
Outputs: data/gencode/gencode_v49_gene_exon_lengths.csv
"""

import pandas as pd
import pyranges as pr

# gtf_path = "data/gencode/gencode.v36.annotation.gtf.gz"
gtf_path ="../BIOFM_PY/data/gencode/gencode.v49.basic.annotation.gtf.gz"

# Load GTF
gtf = pr.read_gtf(gtf_path)

# Keep only exons
exons = gtf[gtf.Feature == "exon"]

# Convert to dataframe
df = exons.as_df()[["Chromosome", "Start", "End", "gene_name"]]

exon_lengths = {}

for gene, subdf in df.groupby("gene_name"):
    gr = pr.PyRanges(subdf)
    merged = gr.merge()

    # merged.length may be int OR Series, handle both
    length = merged.length
    if hasattr(length, "sum"):
        total_len = int(length.sum())
    else:
        total_len = int(length)

    exon_lengths[gene] = total_len

# Convert to DataFrame
exon_len_df = pd.DataFrame(
    list(exon_lengths.items()),
    columns=["gene_name", "exon_length_bp"]
)

exon_len_df.to_csv("./data/gencode/gencode_v49_gene_exon_lengths.csv", index=False)

print("Saved exon lengths for", len(exon_len_df), "genes")


# In[10]:


#exon_len_df.to_csv("./data/gencode/gencode_v49_gene_exon_lengths.csv", index=False)


# In[13]:


import pandas as pd
import numpy as np

# -------------------------------------------------------------
# 1️⃣ Load safe protein list (≤ 3000 aa)
# -------------------------------------------------------------
safe_protein_path = "./data/ensembl/filtered/safe_sequences.csv"
safe_df = pd.read_csv(safe_protein_path)

print("Safe protein file columns:", safe_df.columns.tolist())
# expected: ["gene_symbol", "seq_len"]

safe_df = safe_df.drop_duplicates("gene_symbol")
safe_genes = safe_df["gene_symbol"].tolist()

print(f"Loaded {len(safe_genes)} safe-length protein-coding genes")


# -------------------------------------------------------------
# 2️⃣ Load GENCODE exon lengths
# -------------------------------------------------------------
# exon_len_df = pd.read_csv("gencode_v36_gene_exon_lengths.csv")
print("Exon length columns:", exon_len_df.columns.tolist())
# expected: ["gene_name", "exon_length_bp"]

exon_len_df = exon_len_df.rename(columns={"gene_name": "gene_symbol"})


# -------------------------------------------------------------
# 3️⃣ Merge safe proteins with exon lengths
# -------------------------------------------------------------
merged = safe_df.merge(exon_len_df, on="gene_symbol", how="left")

print("\nInitial merge preview:")
print(merged.head())

missing_mask = merged["exon_length_bp"].isna()
num_missing = missing_mask.sum()

print(f"\nMatched {len(merged) - num_missing} / {len(merged)} genes")
print(f"Missing genes: {num_missing}")


# -------------------------------------------------------------
# 4️⃣ Build nearest-neighbor imputers
# -------------------------------------------------------------
# Build lookup: AA lengths for all genes
aa_len_dict = dict(zip(merged["gene_symbol"], merged["seq_len"]))

# Known exon lengths and their AA lengths
known_df = merged[~missing_mask][["gene_symbol", "seq_len", "exon_length_bp"]].copy()

# For efficient nearest-neighbor search
known_aa = known_df["seq_len"].values
known_exon = known_df["exon_length_bp"].values
known_symbols = known_df["gene_symbol"].values

def impute_exon_length(gene_symbol):
    """Impute exon length using nearest AA length neighbor."""
    aa_len = aa_len_dict[gene_symbol]

    # compute absolute difference in AA length to all known ones
    diffs = np.abs(known_aa - aa_len)

    # find closest gene
    idx = np.argmin(diffs)
    return known_exon[idx], known_symbols[idx]


# -------------------------------------------------------------
# 5️⃣ Perform imputation for missing genes
# -------------------------------------------------------------
missing_genes = merged.loc[missing_mask, "gene_symbol"].tolist()

print("\nImputing missing genes using nearest AA-length neighbor:")
imputed_rows = []

for g in missing_genes:
    imputed_exon_len, donor = impute_exon_length(g)
    imputed_rows.append((g, imputed_exon_len, donor))
    print(f"  {g:12s}  ← nearest AA match: {donor:12s} "
          f"(exon_len={imputed_exon_len})")

# Apply imputations
for g, imputed_len, donor in imputed_rows:
    merged.loc[merged["gene_symbol"] == g, "exon_length_bp"] = imputed_len

# Validate
assert merged["exon_length_bp"].isna().sum() == 0, "Still missing values after imputation!"


# -------------------------------------------------------------
# 6️⃣ Build gene length dictionary
# -------------------------------------------------------------
gene_length_dict = dict(
    zip(merged["gene_symbol"], merged["exon_length_bp"])
)

print("\nFinal gene_length_dict size:", len(gene_length_dict))
print("Example:", list(gene_length_dict.items())[:5])


# -------------------------------------------------------------
# 7️⃣ Canonical gene order (BulkFormer input order)
# -------------------------------------------------------------
canonical_order = merged["gene_symbol"].tolist()

print("\nCanonical gene order length:", len(canonical_order))
print("First 10 genes:", canonical_order[:10])


# -------------------------------------------------------------
# 8️⃣ Save merged + imputed result
# -------------------------------------------------------------
out_path = "./data/gencode/canonical_genes_with_exon_lengths_safe_sequences.csv"
merged.to_csv(out_path, index=False)

print(f"\nSaved final gene file → {out_path}")


# In[ ]:




