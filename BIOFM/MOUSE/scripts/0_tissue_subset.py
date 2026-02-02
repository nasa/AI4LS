import os, re, pandas as pd
import archs4py as a4

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# Download from https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.5.h5
file = "./data/archs4/mouse_gene_v2.5.h5"
output_dir = "./data/archs4"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

pattern="skin|muscle|liver|mammary|breast|retina|kidney|flight|radiation"

meta=a4.meta.meta('data/archs4/mouse_gene_v2.5.h5', pattern, meta_fields=['geo_accession', 'characteristics_ch1', 'source_name_ch1'], remove_sc=True)

print(f"✅ Retrieved {len(meta):,} total samples matching tissue-related keywords.")

# Create a text field to help detect tissue type keywords
meta["text"] = (meta["characteristics_ch1"].fillna("") + " " +
                meta["source_name_ch1"].fillna("")).str.lower()

# Quick preview
meta.head()

# -------------------------------------------------------------
# ASSIGN tissue type labels
# -------------------------------------------------------------

tissue_map = {
            # breast
                "BREAST": r"breast|mammary",

            # kidney
                "KIDNEY": r"kidney",

            # retina
                "RETINA": r"retina",

            # liver
                "LIVER": r"liver",

            # flight
                "FLIGHT": r"flight",

            # radiation
                "RADIATION": r"radiation",

            # skin
                "SKIN": r"skin",

            # muscle
                "MUSCLE": r"muscle"
}

print(f"📊 Total labeled samples: {len(meta):,}")

def assign_tissue_type(text: str) -> str:
    text = str(text).lower()
    for code, pattern in tissue_map.items():
        if re.search(pattern, text):
            return code
    return "UNKNOWN"

meta["tissue_label"] = meta["text"].apply(assign_tissue_type)

# Drop unknowns
meta = meta[meta["tissue_label"] != "UNKNOWN"].reset_index(drop=True)

print(meta["tissue_label"].value_counts().head())

meta.to_csv(os.path.join(output_dir, "archs4_tissue_metadata_labeled.csv"), index=False)
print(f"🧬 Saved labeled metadata: {meta.shape}")

# -------------------------------------------------------------
# 1️⃣ TRAIN–TEST SPLIT (80/20) stratified by tissue type
# -------------------------------------------------------------
train_df, temp_df = train_test_split(
    meta,
    test_size=0.2,
    stratify=meta["tissue_label"],
    random_state=42
)

# -------------------------------------------------------------
# 2️⃣ VALIDATION–TEST SPLIT (from that 20%)
# -------------------------------------------------------------
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,             # 10% val, 10% test overall
    stratify=temp_df["tissue_label"],
    random_state=42
)

# -------------------------------------------------------------
# 3️⃣ VERIFY SPLIT PROPORTIONS
# -------------------------------------------------------------
def summarize_split(df, name):
    counts = df["tissue_label"].value_counts(normalize=True) * 100
    print(f"\n{name} set ({len(df):,} samples)")
    print(counts.round(2).head(10))

summarize_split(train_df, "TRAIN")
summarize_split(val_df, "VAL")
summarize_split(test_df, "TEST")

# -------------------------------------------------------------
# 4️⃣ SAVE SPLITS FOR LATER USE
# -------------------------------------------------------------
output_dir = "./data/archs4/splits"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(f"{output_dir}/train_metadata.csv", index=False)
val_df.to_csv(f"{output_dir}/val_metadata.csv", index=False)
test_df.to_csv(f"{output_dir}/test_metadata.csv", index=False)

print("\n✅ Saved stratified train/val/test metadata splits!")

