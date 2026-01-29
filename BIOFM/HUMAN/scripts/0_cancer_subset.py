#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Notebook: ARCHS4 Cancer Subset & Stratified Splits

Summary:
- Extract cancer-related bulk RNA-seq samples from ARCHS4 using keyword regex.
- Assign unified TCGA-style cancer type labels from free-text metadata.
- Remove samples without a mapped label.
- Produce stratified train/val/test metadata splits for BulkFormer.

Why (BulkFormer):
- Ensures downstream expression matrices align with consistent cancer labels.
- Stratified splits preserve class proportions → stable training & evaluation.

Inputs:
- ARCHS4 HDF5: ./data/archs4/human_gene_v2.5.h5 (metadata + expression)

Outputs:
- Labeled metadata: ./data/archs4/archs4_cancer_metadata_labeled.csv
- Stratified splits (train/val/test): ./data/archs4/splits/*.csv

Steps:
1) Load metadata filtered by cancer-related keywords.
2) Build combined text field; regex-match TCGA cancer types.
3) Drop UNKNOWN labels.
4) Stratified 80/10/10 split (train/val/test).
5) Save labeled master file and split CSVs.

Notes:
- Expression loading can follow this to build gene × sample matrices.
- Adjust regex map if expanding cancer type coverage.
"""

import os, re, pandas as pd
import archs4py as a4

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
# Download from https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.5.h5
file = "../BIOFM_PY/data/archs4/human_gene_v2.5.h5"
output_dir = "./data/archs4/"
os.makedirs(output_dir, exist_ok=True)

# Define search pattern: all cancer-related samples
pattern = "cancer|tumor|carcinoma|leukemia|lymphoma|melanoma|glioma"

# -------------------------------------------------------------
# 1️⃣ LOAD METADATA
# -------------------------------------------------------------
print("📄 Loading metadata from ARCHS4...")

meta = a4.meta.meta(
    file,
    pattern,
    meta_fields=["geo_accession", "characteristics_ch1", "source_name_ch1"], 
    remove_sc=True
)

print(f"✅ Retrieved {len(meta):,} total samples matching cancer-related keywords.")

# Create a text field to help detect cancer type keywords
meta["text"] = (meta["characteristics_ch1"].fillna("") + " " +
                meta["source_name_ch1"].fillna("")).str.lower()

# Quick preview
meta.head()


# In[23]:


# -------------------------------------------------------------
# 2️⃣ ASSIGN CANCER TYPE LABELS
# -------------------------------------------------------------
print("🏷️  Assigning cancer type labels...")

# Define regex patterns for each cancer type
# -------------------------------------------------------------
# 🧬 Unified TCGA-style regex map (cleaned + consolidated)
# -------------------------------------------------------------
tcga_map = {
    # Adrenal
    "ACC":   r"adrenal",

    # Bladder
    "BLCA":  r"bladder|urothelial",

    # Breast
    "BRCA":  r"breast|mda[-]?mb|sum149|ductal",

    # Cervix
    "CESC":  r"cervix|cervical",

    # Bile duct / cholangiocarcinoma
    "CHOL":  r"bile|cholangio",

    # Colon / rectum
    "COAD":  r"colon|colorectal|sigmoid|ht29",
    "READ":  r"rectum|rectal",

    # Lymphoid malignancies
    "DLBC":  r"lymphoma|dlbcl|b[- ]cell|t[- ]cell|cll|plasma cell|myeloma",

    # Esophagus
    "ESCA":  r"esophagus|esophageal",

    # Brain / glioma
    "GBM":   r"glioblastoma|gbm|g477",
    "LGG":   r"glioma|astrocyt|oligodendro|meningioma|brain tumor",

    # Head & neck
    "HNSC":  r"head|neck|oral|tongue|pharynx",

    # Kidney
    "KICH":  r"chromophobe",
    "KIRC":  r"clear cell",
    "KIRP":  r"papillary kidney",

    # Leukemia
    "LAML":  r"leukemia|acute myeloid|aml|cml|k562|bcr[-]?abl|hl60|mll",

    # Liver
    "LIHC":  r"liver|hcc|hepatocellular|liver tumor",

    # Lung
    "LUAD":  r"lung|a549|h1299|h1975|nsclc|adenocarcinoma lung|luad",
    "LUSC":  r"squamous lung|lusc|squamous cell carcinoma|epidermoid|a431",

    # Mesothelioma
    "MESO":  r"mesothelioma",

    # Ovary
    "OV":    r"ovary|ovarian|hgsc|aocs1|serous carcinoma",

    # Pancreas
    "PAAD":  r"pancreas|pancreatic",

    # Pheochromocytoma / Paraganglioma
    "PCPG":  r"pheochromocytoma|paraganglioma",

    # Prostate
    "PRAD":  r"prostate",

    # Sarcoma
    "SARC":  r"sarcoma|rhabdoid|schwannoma|fibro|leiomyo",

    # Skin / Melanoma
    "SKCM":  r"melanoma|skin",

    # Stomach
    "STAD":  r"stomach|gastric",

    # Testis / Germ cell
    "TGCT":  r"testicular|germ cell",

    # Thyroid
    "THCA":  r"thyroid",

    # Thymus
    "THYM":  r"thymus|thymoma",

    # Uterine / Endometrial
    "UCEC":  r"endometrial|uterine corpus",
    "UCS":   r"uterine carcinosarcoma",

    # Uveal / Ocular
    "UVM":   r"uveal|ocular|eye",
}

def assign_tcga_type(text: str) -> str:
    text = str(text).lower()
    for code, pattern in tcga_map.items():
        if re.search(pattern, text):
            return code
    return "UNKNOWN"

meta["tcga_label"] = meta["text"].apply(assign_tcga_type)

# Drop unknowns
meta = meta[meta["tcga_label"] != "UNKNOWN"].reset_index(drop=True)

print(meta["tcga_label"].value_counts())


meta.head()

meta.to_csv(os.path.join(output_dir, "archs4_cancer_metadata_labeled.csv"), index=False)
print(f"🧬 Saved labeled metadata: {meta.shape}")



# In[24]:


meta.head()


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you already have:
# meta with columns: ["geo_accession", "text", "tcga_label", ...]
# and you’ve removed UNKNOWNs

print(f"📊 Total labeled samples: {len(meta):,}")
print(meta["tcga_label"].value_counts().head())

# -------------------------------------------------------------
# 1️⃣ TRAIN–TEST SPLIT (80/20) stratified by cancer type
# -------------------------------------------------------------
train_df, temp_df = train_test_split(
    meta,
    test_size=0.2,
    stratify=meta["tcga_label"],
    random_state=42
)

# -------------------------------------------------------------
# 2️⃣ VALIDATION–TEST SPLIT (from that 20%)
# -------------------------------------------------------------
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,             # 10% val, 10% test overall
    stratify=temp_df["tcga_label"],
    random_state=42
)

# -------------------------------------------------------------
# 3️⃣ VERIFY SPLIT PROPORTIONS
# -------------------------------------------------------------
def summarize_split(df, name):
    counts = df["tcga_label"].value_counts(normalize=True) * 100
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


# In[ ]:




