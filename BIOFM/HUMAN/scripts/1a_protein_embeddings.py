"""
ESM3 Protein Embeddings (Filtered)

Purpose:
- Generate one fixed-size protein language model embedding per gene_symbol using ESM3.
- Filter sequences by length to avoid OOM and keep GPU-safe runs.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from esm.sdk.api import ESMProtein, ESM3InferenceClient
from esm.utils.structure.protein_chain import ProteinChain
from tqdm import tqdm
from Bio import SeqIO

# ============================================================
# CONFIG
# ============================================================
PROTEIN_CSV = "../BIOFM_PY/data/ensembl/protein_coding_genes.csv"
FASTA_PATH = "../BIOFM_PY/data/ensembl/Homo_sapiens.GRCh38.pep.all.fa"
MODEL_NAME = "esm3_sm_open_v1"
#MAX_SAFE_LEN = 3000
MAX_SAFE_LEN = 1000
OUTDIR = "./data/embeddings"
FILTER_DIR = "./data/ensembl/filtered"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FILTER_DIR, exist_ok=True)

BATCH_SIZE = 1  # conservative for GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Changed this line


# ============================================================
# LOAD DATA (from CSV approach)
# ============================================================
df = pd.read_csv(PROTEIN_CSV)
df["seq"] = df["seq"].astype(str).str.upper().str.replace("*", "", regex=False)
df["seq_len"] = df["seq"].str.len()

safe_df = df[df["seq_len"] <= MAX_SAFE_LEN].reset_index(drop=True)
long_df = df[df["seq_len"] > MAX_SAFE_LEN].reset_index(drop=True)

print(f"✅ Loaded {len(df):,} total protein-coding genes")
print(f"✅ {len(safe_df):,} sequences ≤ {MAX_SAFE_LEN} aa (GPU safe)")
print(f"⚠️ {len(long_df):,} sequences > {MAX_SAFE_LEN} aa (skipped)")

safe_path = os.path.join(FILTER_DIR, "safe_sequences.csv")
long_path = os.path.join(FILTER_DIR, "too_long_sequences.csv")

safe_df[["gene_symbol", "seq_len"]].to_csv(safe_path, index=False)
long_df[["gene_symbol", "seq_len"]].to_csv(long_path, index=False)
print(f"💾 Saved safe/long sequence metadata to {FILTER_DIR}")

# ============================================================
# LOAD MODEL (ESM3)
# ============================================================
print(f"\n🔧 Loading ESM3 model: {MODEL_NAME}")
#model = torch.load('esm3_model.pt', map_location='cpu', weights_only=False)
#model.eval()
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3_OPEN_SMALL

# Load model using ESM3's proper loading method
# Option 1: Load from pretrained
model = ESM3.from_pretrained("esm3_sm_open_v1", device=DEVICE)
model = model.float()  # Convert to float32


if torch.cuda.device_count() > 1:
    print(f"🧠 Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model = model.to(DEVICE).eval()
print("✅ ESM3 Model ready")


# ============================================================
# DATASET
# ============================================================
class ProteinDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["seq"], row["gene_symbol"]

dataset = ProteinDataset(safe_df)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# GENERATE EMBEDDINGS (ESM3 - correct API)
# ============================================================
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.utils.encoding import tokenize_sequence
#from esm.tokenization import get_model_tokenizers
from esm.tokenization import get_esm3_model_tokenizers
from esm.sdk.api import ESM3_OPEN_SMALL

# Get tokenizers once
if isinstance(model, torch.nn.DataParallel):
    #tokenizers = get_esm3_model_tokenizers(model.module)
    tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
    actual_model = model.module
else:
    #tokenizers = get_esm3_model_tokenizers(model)
    from esm.sdk.api import ESM3_OPEN_SMALL
    tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
    actual_model = model


embeddings, gene_order = [], []
print("\n🚀 Generating embeddings with ESM3...")

with torch.no_grad():
    for batch in tqdm(data_loader, total=len(data_loader)):
        seqs, gene_symbols = batch

        for seq, gene_symbol in zip(seqs, gene_symbols):
            # Tokenize the sequence
            sequence_tokens = tokenize_sequence(seq, tokenizers.sequence)
            # Add batch dimension
            sequence_tokens = sequence_tokens.unsqueeze(0).to(DEVICE)


            # Create ESMProteinTensor with just sequence tokens
            protein_tensor = ESMProteinTensor(sequence=sequence_tokens.to(DEVICE))

            # Forward pass through model
            #output = actual_model(protein_tensor)
            #output = actual_model(sequence_tokens=sequence_tokens.to(DEVICE))
            output = actual_model(sequence_tokens=sequence_tokens)

            # Extract embeddings from output
            # ESM3 forward returns ForwardOutput with embeddings
            if hasattr(output, 'embeddings'):
                # embeddings shape: [seq_len, embedding_dim]
                seq_embedding = output.embeddings[0].mean(dim=0).cpu().to(torch.float32)
            elif hasattr(output, 'sequence_logits'):
                # If model returns logits, we need the hidden states
                # This shouldn't be the case, but as fallback
                print(f"Warning: got logits instead of embeddings for {gene_symbol}")
                continue
            elif isinstance(output, dict) and 'embeddings' in output:
                seq_embedding = output['embeddings'].mean(dim=0).cpu().to(torch.float32)
            else:
                print(f"Unexpected output for {gene_symbol}: {type(output)}")
                print(f"Available attributes: {dir(output)}")
                continue

            embeddings.append(seq_embedding)
            gene_order.append(gene_symbol)

# ============================================================
# SAVE RESULTS
# ============================================================
E_ESM3 = torch.stack(embeddings)
save_pt = os.path.join(OUTDIR, f"{MODEL_NAME}_gene_embeddings.pt")
save_csv = os.path.join(OUTDIR, f"{MODEL_NAME}_gene_order.csv")

torch.save({"embeddings": E_ESM3, "genes": gene_order, "model": MODEL_NAME}, save_pt)
pd.Series(gene_order, name="gene_symbol").to_csv(save_csv, index=False)

print(f"\n✅ Saved embeddings → {save_pt}")
print(f"✅ Shape: {E_ESM3.shape} [genes × dim={E_ESM3.shape[1]}]")


