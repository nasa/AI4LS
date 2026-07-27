# Spaceflight Transcriptomics cVAE

A conditional Variational Autoencoder (cVAE) for learning latent representations
of bulk RNA-seq data from NASA GeneLab spaceflight experiments. The model learns
to separate spaceflight-induced gene expression changes from biological covariates
(tissue, strain, sex, euthanasia method) and technical batch effects (study ID).

---

## Project Overview

### Scientific Context
- **Data source**: NASA GeneLab (https://genelab.nasa.gov)
- **Organism**: Mus musculus (mouse)
- **Experiment type**: Bulk RNA-seq
- **Biological question**: What transcriptional changes does spaceflight induce
  across tissues, strains, and sexes?
- **Dataset**: 2,080 samples × 18,907 protein-coding genes across 72 studies,
  794 spaceflight / 1,286 ground control

### Model Summary
A conditional multi-task VAE with:
- **Encoder**: Maps expression + condition embeddings → 32-dimensional latent space
- **Decoder**: Reconstructs expression via Negative Binomial likelihood + classifies
  spaceflight condition
- **Covariates**: tissue (25 types), strain (5), sex (3), euthanasia method (6),
  study ID (72), spaceflight label (binary)
- **Training**: β-VAE with KL annealing, cosine LR schedule, early stopping

---

## File Reference

### Core Model Files

#### `model.py`
Defines the full cVAE architecture.

| Class | Description |
|---|---|
| `ConditionEmbedder` | Embeds all metadata covariates into a single 84-dim condition vector |
| `Encoder` | Maps (expression + condition) → (μ, log_var) via 2 hidden layers [256, 128] |
| `Decoder` | Maps (z + condition) → NB params (log_r, p) + flight logit via [128, 256] |
| `BatchDiscriminator` | Optional adversarial batch corrector (disabled by default, α=0) |
| `SpaceflightCVAE` | Full model combining all components |

**Key methods on SpaceflightCVAE:**
```python
model.forward(x, strain, sex, study, tissue, euth, flight)  # training
model.encode(x, strain, sex, study, tissue, euth, flight)   # inference (returns μ)
model.generate(z, strain, sex, study, tissue, euth, flight) # decode z → expression
```

**Architecture dimensions:**
```
Input:   (B, 18907) expression + (B, 84) condition
Encoder: 18991 → 256 → 128 → μ(32), log_var(32)
Latent:  z ~ N(μ, σ²), dim=32
Decoder: 116 → 128 → 256 → log_r(18907), p(18907), flight_logit(1)
Params:  ~14.6M
```

---

#### `dataset.py`
PyTorch Dataset that loads from `subset_final.h5`.

**Key class: `SpaceflightDataset`**
```python
dataset = SpaceflightDataset("subset_final.h5")
```

Attributes exposed:
- `n_samples`, `n_genes`, `n_tissues`, `n_strains`, `n_sexes`, `n_studies`, `n_euths`
- `tissue_enc`, `strain_enc`, `sex_enc`, `study_enc`, `euth_enc` — sklearn LabelEncoders
- `gene_symbols`, `ensembl_ids` — (n_genes,) arrays

Each `__getitem__` returns:
```python
{
    "x":      log1p(CPM-normalized counts),   # (18907,) encoder input
    "x_raw":  raw counts,                      # (18907,) for NB loss
    "strain": encoded strain ID,
    "sex":    encoded sex ID,
    "study":  encoded study ID,
    "tissue": encoded tissue ID,
    "euth":   encoded euthanasia method ID,
    "flight": spaceflight label (0 or 1),
}
```

**Key methods:**
```python
dataset.split(val_frac=0.15, test_frac=0.15)   # stratified train/val/test split
dataset.kfold(n_splits=5)                        # k-fold cross-validation
make_dataloaders(dataset, batch_size=32)         # convenience DataLoader factory
```

---

#### `losses.py`
Loss functions for the cVAE.

| Function/Class | Description |
|---|---|
| `nb_nll_loss(x_raw, log_r, p)` | Negative Binomial NLL (mean over genes) |
| `kl_divergence(mu, log_var)` | KL(N(μ,σ²) ‖ N(0,I)) |
| `classification_loss(logit, flight)` | Binary cross-entropy for spaceflight |
| `adversarial_batch_loss(logit, study)` | CE for batch discriminator (optional) |
| `CVAELoss(beta, lambda_cls, lambda_adv)` | Combined loss: recon + β·KL + λ·BCE |
| `KLAnnealer(beta, n_epochs)` | Linearly anneals KL weight 0 → β |

**Total loss:**
```
L = L_recon + β·KL + λ_cls·BCE(flight)
β = 1.0 (after annealing), λ_cls = 1.0
```

---

### Training

#### `train.py`
Full training loop with early stopping, checkpointing, and wandb logging.

**Usage:**
```bash
python train.py \
    --data subset_final.h5 \
    --output_dir ./checkpoints_v6
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data` | required | Path to subset_final.h5 |
| `--output_dir` | ./checkpoints | Checkpoint save directory |
| `--latent_dim` | 32 | Latent space dimension |
| `--epochs` | 300 | Max training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 5e-4 | Learning rate |
| `--kl_anneal_epochs` | 100 | Epochs to anneal KL weight |
| `--beta` | 1.0 | Final KL weight |
| `--lambda_cls` | 1.0 | Classification loss weight |
| `--lambda_adv` | 0.0 | Adversarial batch loss weight (disabled) |
| `--patience` | 30 | Early stopping patience |
| `--no_wandb` | False | Disable wandb logging |

**Checkpoints** saved to `output_dir/best_model.pt` contain:
- `model_state`, `optimizer_state`, `epoch`, `val_loss`, `args`
- `label_encoders`: dict of strain/sex/study LabelEncoders

**Epoch diagnostics logged every 20 epochs:**
- Tissue predictability from z (want > 0.7 — z encodes biology)
- Within-study flight AUROC (want > 0.8 — z separates flight/ground)
- Study predictability from z (informational — high is expected)

---

### Inference and Evaluation

#### `inference.py`
Test set evaluation, UMAP/t-SNE visualization, gene attribution, and pathway enrichment.

**Usage:**
```bash
python inference.py \
    --checkpoint checkpoints_v6/best_model.pt \
    --data subset_final.h5 \
    --output_dir inference_results/

# offline (skip Enrichr)
python inference.py ... --skip_enrichment

# tune t-SNE
python inference.py ... --tsne_perplexity 50 --tsne_n_iter 2000
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to .pt checkpoint |
| `--data` | required | Path to subset_final.h5 |
| `--output_dir` | results | Output directory |
| `--skip_enrichment` | False | Skip Enrichr API calls |
| `--enrichment_genes` | 100 | Top genes per tissue for enrichment |
| `--enrichment_cutoff` | 0.05 | Adjusted p-value cutoff |
| `--tsne_perplexity` | 30 | t-SNE perplexity |
| `--tsne_n_iter` | 1000 | t-SNE iterations |

**Outputs:**

| File | Description |
|---|---|
| `test_metrics.txt` | AUROC and accuracy on held-out test set |
| `latent_all.npz` | z vectors + metadata for all 2080 samples |
| `umap_{flight,study,strain,sex,tissue,euth}.png` | UMAP plots colored by metadata |
| `tsne_{flight,study,strain,sex,tissue,euth}.png` | t-SNE plots colored by metadata |
| `gene_attribution.csv` | Top 200 genes by spaceflight attribution (all samples) |
| `gene_attribution_clean.csv` | Same, pseudogenes removed |
| `gene_attribution_by_tissue.csv` | Top 100 genes per tissue |
| `by_tissue/<tissue>.csv` | Per-tissue clean gene lists |
| `enrichment_summary.csv` | Summary of pathway enrichment across all tissues |
| `enrichment/<tissue>/<library>.csv` | Significant terms per tissue per library |
| `enrichment/<tissue>/<library>.png` | Bar plots of top enriched pathways |

**Enrichr libraries used:**
- `GO_Biological_Process_2026`
- `KEGG_2019_Mouse`
- `Reactome_Pathways_2024`
- `WikiPathways_2024_Mouse`

---

### What-If Analysis

#### `whatif.py`
Counterfactual and population-level what-if analysis.

**Two modes:**

**COUNTERFACTUAL** — encode real samples, change one condition, decode and compare:
```bash
# what would liver/C57BL6J/Female ground samples look like in space?
python whatif.py \
    --checkpoint checkpoints_v6/best_model.pt \
    --data subset_final.h5 \
    --mode counterfactual \
    --tissue Liver --strain C57BL/6J --sex Female --flight 0 \
    --change_condition flight --change_from 0 --change_to 1

# what if CO2-euthanized samples had been euthanized with isoflurane?
python whatif.py \
    --checkpoint checkpoints_v6/best_model.pt \
    --data subset_final.h5 \
    --mode counterfactual \
    --tissue Liver --euth CO2 \
    --change_condition euth --change_from CO2 --change_to Isoflurane

# what if liver samples were kidney?
python whatif.py \
    --checkpoint checkpoints_v6/best_model.pt \
    --data subset_final.h5 \
    --mode counterfactual \
    --tissue Liver \
    --change_condition tissue --change_from Liver --change_to Kidney
```

**POPULATION** — average z for a condition, decode under flight=1 and flight=0:
```bash
# average Kidney expression: space vs ground
python whatif.py \
    --checkpoint checkpoints_v6/best_model.pt \
    --data subset_final.h5 \
    --mode population \
    --tissue Kidney --strain C57BL/6J --sex Female
```

**Key arguments:**

| Argument | Description |
|---|---|
| `--mode` | `counterfactual` or `population` |
| `--tissue` | Filter by tissue type |
| `--strain` | Filter by strain |
| `--sex` | Filter by sex (Female, Male, Unknown) |
| `--euth` | Filter by euthanasia method |
| `--flight` | Filter by flight status (0 or 1) |
| `--change_condition` | Condition to change: flight, tissue, strain, sex, euth |
| `--change_from` | Original condition value (documents the run label) |
| `--change_to` | New condition value to decode with |
| `--n_top` | Number of top up/down genes to report (default 200) |

**Outputs (saved under `output_dir/<run_label>/`):**

| File | Description |
|---|---|
| `sample_metadata.csv` | Which samples were selected |
| `original_expression.csv` | Real measured expression (samples × genes) |
| `counterfactual_expression.csv` | Model-predicted expression after condition change |
| `delta_expression.csv` | All genes ranked by abs(delta) |
| `top_up_genes.csv` | Top upregulated genes |
| `top_down_genes.csv` | Top downregulated genes |
| `population_flight_expression.csv` | (population mode) predicted under spaceflight |
| `population_ground_expression.csv` | (population mode) predicted under ground control |
| `population_delta.csv` | (population mode) flight - ground ranked |

---

### Latent Space Visualization

#### `visualize_latent.py`
Four complementary latent space visualizations.

**Usage:**
```bash
python visualize_latent.py \
    --checkpoint checkpoints_v6/best_model.pt \
    --data subset_final.h5 \
    --output_dir latent_viz/
```

**Outputs:**

| File | Description |
|---|---|
| `latent_dimension_heatmap.png` | Spearman correlation of each z dimension with metadata. High correlation = that dimension encodes that variable. |
| `latent_dimension_correlations.csv` | Raw correlation values (32 dims × 6 variables) |
| `pca_scree.png` | Variance explained per PCA component |
| `pca_{flight,tissue,strain,sex,euth}.png` | PC1 vs PC2 colored by metadata |
| `pca_coordinates.csv` | PCA coordinates for all samples |
| `tissue_latent_distributions.png` | Violin plots of top 8 most variable z dimensions grouped by tissue |
| `centroid_distances.png` | Bar chart ranking tissues by spaceflight-induced latent shift |
| `centroid_distances.csv` | Euclidean distance + cosine similarity between flight/ground centroids per tissue |

**`--max_dims`** (default 8): number of latent dimensions to show in violin plots.

---

## Data Files

### `subset_final.h5`
The filtered, harmonized dataset. Structure:

```
subset_final.h5
├── data/
│   └── expression          (18907, 2080)  float32 — genes × samples, raw counts
├── meta/
│   ├── genes/
│   │   ├── ensembl_id      (18907,)       bytes   — e.g. ENSMUSG00000000001
│   │   └── symbol          (18907,)       bytes   — e.g. Actb
│   └── samples/
│       ├── spaceflight     (2080,)        int8    — 0=ground, 1=spaceflight
│       ├── tissue          (2080,)        bytes   — harmonized tissue name
│       ├── strain          (2080,)        bytes   — harmonized strain name
│       ├── sex             (2080,)        bytes   — Female/Male/Unknown
│       ├── study_id        (2080,)        bytes   — OSD-XXX accession
│       └── euthanasia      (2080,)        bytes   — harmonized method
```

**Filtering applied:**
- Protein-coding genes only (biotype == "protein_coding")
- Characterized genes only (removed Gm*, *-ps, *Rik)
- Expressed in ≥10% of samples (count > 1)
- Mouse samples with known flight status only
- Tissue harmonized to 25 categories + Other (min 30 samples)
- Strain harmonized to 5 categories
- Euthanasia harmonized to 6 categories

---

## Setup

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn h5py
pip install umap-learn matplotlib
pip install gseapy          # for pathway enrichment
pip install wandb           # for experiment tracking
```

**GPU note:** Requires PyTorch built for CUDA compute capability 7.0+ (V100).
Use `torch==2.4.0+cu118` for V100 on clusters with CUDA 13.x drivers.

---

## Workflow

```
1. Build dataset
   subset_final.py  →  subset_final.h5

2. Train model
   train.py  →  checkpoints_v6/best_model.pt

3. Evaluate and analyze
   inference.py  →  test metrics, UMAPs, gene attribution, pathway enrichment

4. Visualize latent space
   visualize_latent.py  →  dimension heatmap, PCA, violin plots, centroid distances

5. What-if analysis
   whatif.py  →  counterfactual and population expression predictions
```

---

## Model Performance (v6)

| Metric | Value |
|---|---|
| Test AUROC | 1.000 |
| Test Accuracy | 1.000 |
| Val Loss (best) | ~5.5 |
| Tissue predictability from z | ~0.93 |
| Within-study flight AUROC | ~0.95 |
| Training epochs | ~200 (early stopping) |
