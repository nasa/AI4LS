import pandas as pd

counts  = pd.read_csv("DATA/nasa_common_expr.csv")
meta    = pd.read_csv("DATA/nasa_common_meta.csv")

print("=== COUNTS ===")
print(counts.shape)
print(counts.iloc[:5, :5])          # first 5 rows, 5 cols

print("\n=== METADATA ===")
print(meta.shape)
print(meta.columns.tolist())
print(meta.head(3))

print("\n=== SAMPLE ID OVERLAP ===")
# adjust 'sample_id' to your actual column names
count_samples = set(counts.columns)       # sample IDs are column headers in counts
meta_samples  = set(meta["sample_id"])
print(f"Counts samples:   {len(count_samples)}")
print(f"Metadata samples: {len(meta_samples)}")
print(f"Overlap:          {len(count_samples & meta_samples)}")
print(f"\nExample counts cols:  {list(counts.columns)[:5]}")
print(f"Example meta IDs:     {meta['sample_id'].tolist()[:5]}")

print(meta["spaceflight"].value_counts())
print(meta["tissue"].value_counts())
print(meta["study_id"].value_counts())
