# Spaceflight cVAE

Conditional multi-task Variational Autoencoder for NASA GeneLab bulk RNA-seq data.
Learns a latent representation of the mouse transcriptome under spaceflight and
ground control conditions.

---

## Architecture

```
Raw Counts (n_genes)
      │
      ▼
[ConditionEmbedder] ← tissue, strain, study_id, spaceflight, duration
      │
      ▼
[Encoder]  →  μ, log_var  →  z ~ N(μ, σ²)
                                    │
                              [Decoder]
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              NB Recon        Classification    Regression
            (log_r, p)      P(spaceflight)   mission duration
```

**Adversarial batch correction**: a `BatchDiscriminator` predicts study ID from `z`
via gradient reversal, forcing the latent space to be study-agnostic.

---

## File Structure

```
spaceflight_cvae/
├── model.py       # SpaceflightCVAE, Encoder, Decoder, ConditionEmbedder, BatchDiscriminator
├── losses.py      # CVAELoss (NB NLL + KL + BCE + Huber + adversarial), KLAnnealer
├── dataset.py     # SpaceflightDataset, stratified splitting, DataLoader helpers
├── train.py       # Training loop with early stopping and checkpointing
└── inference.py   # Latent extraction, UMAP, flight vector, gene attribution
```

---

## Setup

```bash
pip install torch torchvision numpy pandas scikit-learn
pip install umap-learn matplotlib  # for visualization
pip install wandb                  # optional, for experiment tracking
```

---

## Data Format

Your input CSV should have:

| Column | Type | Description |
|---|---|---|
| `ENSM*` | float | Raw integer gene counts (one column per gene) |
| `tissue` | str | Tissue type (e.g. `liver`, `muscle`) |
| `strain` | str | Mouse strain (e.g. `C57BL/6J`) |
| `study_id` | str | GeneLab accession (e.g. `GLDS-4`) |
| `spaceflight` | int | 0 = ground control, 1 = spaceflight |
| `duration_days` | float | Mission duration in days (0.0 for ground) |

---

## Training

```bash
python train.py \
  --data genelab_samples.csv \
  --gene_prefix ENSM \
  --latent_dim 64 \
  --epochs 300 \
  --batch_size 64 \
  --lr 1e-3 \
  --kl_anneal_epochs 50 \
  --beta 1.0 \
  --lambda_cls 1.0 \
  --lambda_reg 0.5 \
  --lambda_adv 0.1 \
  --output_dir ./checkpoints
```

### Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `latent_dim` | 64 | Try 32–128; smaller = more regularized |
| `beta` | 1.0 | β > 1 encourages disentanglement |
| `kl_anneal_epochs` | 50 | Ramp KL from 0 → β over this many epochs |
| `grl_alpha` | 1.0 | Gradient reversal strength for batch correction |
| `lambda_cls` | 1.0 | Classification loss weight |
| `lambda_reg` | 0.5 | Regression loss weight |
| `lambda_adv` | 0.1 | Adversarial batch correction weight |
| `patience` | 30 | Early stopping patience (epochs) |

---

## Phased Build Recommendation

**Phase 1** — Basic VAE (comment out task heads and batch discriminator)
- Confirm NB reconstruction works; latent space visually separates flight/ground

**Phase 2** — Add condition embeddings + classification head
- Check val AUC > 0.8 before proceeding

**Phase 3** — Add adversarial batch correction
- Verify study ID is no longer predictable from `z` (batch discriminator accuracy ≈ chance)

**Phase 4** — Add regression head, tune loss weights

---

## Inference & Analysis

```python
from inference import load_model, get_latent_representations, compute_flight_vector
from inference import plot_latent_umap, gene_attribution

# Load trained model
model, encoders, args = load_model("checkpoints/best_model.pt")

# Extract latent representations
latent_dict = get_latent_representations(model, test_loader)

# Visualize
plot_latent_umap(latent_dict, color_by="flight", save_path="umap_flight.png")
plot_latent_umap(latent_dict, color_by="study",  save_path="umap_study.png")

# Spaceflight perturbation vector
v_flight = compute_flight_vector(latent_dict)

# Gene attribution for a single sample
attribution = gene_attribution(model, z_sample, tissue, strain, study, flight, duration)
top_genes = np.argsort(attribution)[::-1][:50]   # top 50 genes by attribution
```

---

## Downstream Tasks

| Task | How |
|---|---|
| **Spaceflight classification** | Use `flight_logit` from decoder; report AUROC on test set |
| **Mission duration prediction** | Use `duration_hat`; report R² and MAE |
| **Biomarker discovery** | Gene attribution scores via `inference.gene_attribution()` |
| **Counterfactual generation** | `inference.generate_counterfactual()` — "what would this tissue look like in space?" |
| **Batch effect assessment** | UMAP colored by `study` before/after adversarial correction |
| **Transfer to new studies** | Freeze encoder, fine-tune decoder on new GeneLab study |
