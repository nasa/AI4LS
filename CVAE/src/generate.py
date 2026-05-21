"""
Synthetic Sample Generation for Spaceflight cVAE
=================================================
Generates an arbitrary number of synthetic bulk RNA-seq samples
conditioned on any combination of:
    - tissue type
    - strain
    - sex
    - spaceflight status (0=ground, 1=spaceflight)
    - euthanasia method

Synthesis works by:
    1. Sampling z ~ N(0, I) in latent space
    2. Building a condition vector from the specified metadata
    3. Decoding z through the trained decoder to get NB parameters
    4. Sampling from the NB distribution to get realistic count data

Usage:

    # 100 synthetic spaceflight liver samples, C57BL/6J female, isoflurane
    python generate.py \\
        --checkpoint checkpoints_v6/best_model.pt \\
        --data subset_final.h5 \\
        --n 100 \\
        --tissue Liver \\
        --strain C57BL/6J \\
        --sex Female \\
        --flight 1 \\
        --euth Isoflurane \\
        --output_dir synthetic_samples/

    # 500 ground control kidney samples, any strain, male
    python generate.py \\
        --checkpoint checkpoints_v6/best_model.pt \\
        --data subset_final.h5 \\
        --n 500 \\
        --tissue Kidney \\
        --sex Male \\
        --flight 0 \\
        --output_dir synthetic_samples/

    # generate from multiple conditions in one run using a config file
    python generate.py \\
        --checkpoint checkpoints_v6/best_model.pt \\
        --data subset_final.h5 \\
        --config conditions.csv \\
        --output_dir synthetic_samples/

Config file format (conditions.csv):
    n,tissue,strain,sex,flight,euth
    100,Liver,C57BL/6J,Female,1,Isoflurane
    100,Liver,C57BL/6J,Female,0,Isoflurane
    50,Soleus,C57BL/6J,Male,1,Ketamine_Xylazine

Outputs (saved under output_dir/<run_label>/):
    synthetic_counts.csv        raw NB-sampled integer counts (samples x genes)
    synthetic_expression.csv    log1p CPM-normalized expression (samples x genes)
    synthetic_metadata.csv      condition metadata for each synthetic sample
    latent_vectors.csv          the z vectors used to generate each sample
    generation_summary.txt      summary of generation parameters
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from dataset import SpaceflightDataset
from model import SpaceflightCVAE


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, dataset, device="cpu"):
    ckpt  = torch.load(checkpoint_path, map_location=device)
    args  = ckpt["args"]
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_strains=dataset.n_strains,
        n_sexes=dataset.n_sexes,
        n_studies=dataset.n_studies,
        n_tissues=dataset.n_tissues,
        n_euths=dataset.n_euths,
        latent_dim=args["latent_dim"],
        hidden_dims=[256, 128],
        dropout=0.0,
        grl_alpha=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print("Loaded model from " + checkpoint_path +
          " (epoch " + str(ckpt["epoch"]) +
          ", val_loss=" + f"{ckpt['val_loss']:.4f})")
    return model, args["latent_dim"]


# ---------------------------------------------------------------------------
# Condition resolution
# ---------------------------------------------------------------------------

def resolve_condition(dataset, tissue, strain, sex, flight, euth):
    """
    Convert string condition values to encoded integer tensors.
    If a value is None, randomly sample from the available categories.

    Returns:
        dict of scalar int values for each condition
        dict of string labels for metadata
    """
    def resolve_one(value, encoder, name):
        if value is None:
            # randomly sample a category weighted by frequency in dataset
            idx = int(np.random.choice(len(encoder.classes_)))
            return idx, encoder.classes_[idx]
        if value not in encoder.classes_:
            raise ValueError(
                "Unknown " + name + ": '" + str(value) + "'" +
                "\nAvailable: " + str(list(encoder.classes_))
            )
        idx = int(encoder.transform([value])[0])
        return idx, value

    tissue_idx, tissue_str = resolve_one(tissue, dataset.tissue_enc, "tissue")
    strain_idx, strain_str = resolve_one(strain, dataset.strain_enc, "strain")
    sex_idx,    sex_str    = resolve_one(sex,    dataset.sex_enc,    "sex")
    euth_idx,   euth_str   = resolve_one(euth,   dataset.euth_enc,   "euth")

    flight_val = int(flight) if flight is not None else int(np.random.randint(0, 2))
    flight_str = "spaceflight" if flight_val == 1 else "ground_control"

    # for study, always use the most common study for the selected tissue
    # (study is a nuisance variable — we want a plausible value, not a random one)
    tissue_mask  = dataset.tissue_ids == tissue_idx
    if tissue_mask.sum() > 0:
        study_ids_for_tissue = dataset.study_ids[tissue_mask]
        study_idx = int(np.bincount(study_ids_for_tissue).argmax())
    else:
        study_idx = 0
    study_str = dataset.study_enc.classes_[study_idx]

    encoded = {
        "tissue": tissue_idx,
        "strain": strain_idx,
        "sex":    sex_idx,
        "euth":   euth_idx,
        "flight": flight_val,
        "study":  study_idx,
    }
    labels = {
        "tissue":  tissue_str,
        "strain":  strain_str,
        "sex":     sex_str,
        "euth":    euth_str,
        "flight":  flight_val,
        "study":   study_str,
    }
    return encoded, labels


# ---------------------------------------------------------------------------
# NB sampling
# ---------------------------------------------------------------------------

def sample_nb(log_r, p, size=1):
    """
    Sample from a Negative Binomial distribution.

    NB(r, p): number of failures before r successes, prob of success = p
    Mean = r(1-p)/p

    Uses the Gamma-Poisson mixture representation:
        lambda ~ Gamma(r, p/(1-p))
        x | lambda ~ Poisson(lambda)

    Args:
        log_r: (n_genes,) log dispersion
        p:     (n_genes,) success probability
        size:  number of samples to draw

    Returns:
        counts: (size, n_genes) integer count matrix
    """
    r = np.exp(log_r).clip(min=1e-4, max=1e4)
    p = p.clip(min=1e-6, max=1 - 1e-6)

    # scale parameter for Gamma: (1-p)/p
    scale = (1 - p) / p

    all_counts = []
    for _ in range(size):
        # sample lambda from Gamma(r, scale)
        lam = np.random.gamma(shape=r, scale=scale)
        # sample counts from Poisson(lambda)
        counts = np.random.poisson(lam)
        all_counts.append(counts)

    return np.stack(all_counts).astype(np.float32)   # (size, n_genes)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_samples(
    model,
    dataset,
    latent_dim,
    n,
    tissue=None,
    strain=None,
    sex=None,
    flight=None,
    euth=None,
    device="cpu",
    batch_size=256,
    seed=None,
):
    """
    Generate n synthetic samples conditioned on the specified metadata.

    Args:
        model:      trained SpaceflightCVAE
        dataset:    SpaceflightDataset (for encoders and gene info)
        latent_dim: latent space dimension
        n:          number of samples to generate
        tissue:     tissue type string, or None to sample randomly
        strain:     strain string, or None to sample randomly
        sex:        sex string, or None to sample randomly
        flight:     0 or 1, or None to sample randomly
        euth:       euthanasia method string, or None to sample randomly
        batch_size: decode this many samples at a time
        seed:       random seed for reproducibility

    Returns:
        counts:     (n, n_genes) raw NB-sampled integer counts
        expression: (n, n_genes) log1p CPM-normalized expression
        z_vectors:  (n, latent_dim) sampled latent vectors
        metadata:   DataFrame with condition labels for each sample
        labels:     dict of condition string labels
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # resolve condition
    encoded, labels = resolve_condition(
        dataset, tissue, strain, sex, flight, euth
    )

    print("Generating " + str(n) + " synthetic samples with conditions:")
    for k, v in labels.items():
        print("  " + k + ": " + str(v))

    model.eval()
    all_log_r = []
    all_p     = []
    all_z     = []

    n_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            batch_n = min(batch_size, n - i * batch_size)

            # sample z from prior N(0, I)
            z = torch.randn(batch_n, latent_dim).to(device)

            # build condition tensors (same condition for all samples in batch)
            strain_t = torch.full((batch_n,), encoded["strain"],
                                  dtype=torch.long, device=device)
            sex_t    = torch.full((batch_n,), encoded["sex"],
                                  dtype=torch.long, device=device)
            study_t  = torch.full((batch_n,), encoded["study"],
                                  dtype=torch.long, device=device)
            tissue_t = torch.full((batch_n,), encoded["tissue"],
                                  dtype=torch.long, device=device)
            euth_t   = torch.full((batch_n,), encoded["euth"],
                                  dtype=torch.long, device=device)
            flight_t = torch.full((batch_n,), encoded["flight"],
                                  dtype=torch.long, device=device)

            log_r, p = model.generate(z, strain_t, sex_t, study_t,
                                      tissue_t, euth_t, flight_t)

            all_log_r.append(log_r.cpu().numpy())
            all_p.append(p.cpu().numpy())
            all_z.append(z.cpu().numpy())

    log_r_all = np.vstack(all_log_r)   # (n, n_genes)
    p_all     = np.vstack(all_p)       # (n, n_genes)
    z_all     = np.vstack(all_z)       # (n, latent_dim)

    # sample counts from NB distribution
    print("Sampling from Negative Binomial distribution...")
    counts = np.zeros((n, dataset.n_genes), dtype=np.float32)
    for i in range(n):
        counts[i] = sample_nb(log_r_all[i], p_all[i], size=1)[0]

    # compute log1p CPM-normalized expression
    lib_sizes  = counts.sum(axis=1, keepdims=True)
    lib_sizes  = np.maximum(lib_sizes, 1.0)
    expression = np.log1p(counts / lib_sizes * 1e4)

    # build metadata DataFrame
    metadata = pd.DataFrame({
        "sample_id": ["synthetic_" + str(i) for i in range(n)],
        "tissue":    labels["tissue"],
        "strain":    labels["strain"],
        "sex":       labels["sex"],
        "flight":    labels["flight"],
        "euth":      labels["euth"],
        "study":     labels["study"],
    })

    return counts, expression, z_all, metadata, labels


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(counts, expression, z_all, metadata, labels,
                 gene_symbols, ensembl_ids, out_dir):
    """Save all generation outputs to CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = metadata["sample_id"].tolist()

    # raw counts
    counts_df = pd.DataFrame(counts, index=sample_ids, columns=gene_symbols)
    counts_df.index.name = "sample_id"
    counts_df.to_csv(out_dir / "synthetic_counts.csv")
    print("  Saved: " + str(out_dir / "synthetic_counts.csv") +
          " (" + str(counts.shape[0]) + " samples x " +
          str(counts.shape[1]) + " genes)")

    # normalized expression
    expr_df = pd.DataFrame(expression, index=sample_ids, columns=gene_symbols)
    expr_df.index.name = "sample_id"
    expr_df.to_csv(out_dir / "synthetic_expression.csv")
    print("  Saved: " + str(out_dir / "synthetic_expression.csv"))

    # latent vectors
    z_df = pd.DataFrame(
        z_all,
        index=sample_ids,
        columns=["z" + str(i) for i in range(z_all.shape[1])]
    )
    z_df.index.name = "sample_id"
    z_df.to_csv(out_dir / "latent_vectors.csv")
    print("  Saved: " + str(out_dir / "latent_vectors.csv"))

    # metadata
    metadata.to_csv(out_dir / "synthetic_metadata.csv", index=False)
    print("  Saved: " + str(out_dir / "synthetic_metadata.csv"))

    # summary
    with open(out_dir / "generation_summary.txt", "w") as f:
        f.write("=== Synthetic Sample Generation Summary ===\n\n")
        f.write("N samples:  " + str(counts.shape[0]) + "\n")
        f.write("N genes:    " + str(counts.shape[1]) + "\n\n")
        f.write("Conditions:\n")
        for k, v in labels.items():
            f.write("  " + k + ": " + str(v) + "\n")
        f.write("\nExpression statistics:\n")
        f.write("  Mean counts per gene:   " +
                f"{counts.mean():.2f}" + "\n")
        f.write("  Median library size:    " +
                f"{counts.sum(axis=1).median() if hasattr(counts.sum(axis=1), 'median') else np.median(counts.sum(axis=1)):.0f}" + "\n")
        f.write("  Fraction zero counts:   " +
                f"{(counts == 0).mean():.3f}" + "\n")
    print("  Saved: " + str(out_dir / "generation_summary.txt"))


# ---------------------------------------------------------------------------
# Multi-condition batch generation from config file
# ---------------------------------------------------------------------------

def generate_from_config(config_path, model, dataset, latent_dim,
                         out_dir, device, seed=None):
    """
    Generate samples for multiple conditions defined in a CSV config file.

    Config format:
        n,tissue,strain,sex,flight,euth
        100,Liver,C57BL/6J,Female,1,Isoflurane
        100,Liver,C57BL/6J,Female,0,Isoflurane

    Each row generates a separate set of synthetic samples saved
    to its own subdirectory.
    """
    config_df = pd.read_csv(config_path)
    required  = {"n"}
    if not required.issubset(config_df.columns):
        raise ValueError("Config file must have at least an 'n' column")

    optional = ["tissue", "strain", "sex", "flight", "euth"]
    print("Generating from config: " + str(config_path))
    print("  " + str(len(config_df)) + " conditions to generate")

    all_metadata = []
    all_counts   = []
    all_expr     = []

    for row_idx, row in config_df.iterrows():
        n = int(row["n"])
        kwargs = {col: (str(row[col]) if pd.notna(row.get(col)) else None)
                  for col in optional if col in config_df.columns}
        if "flight" in kwargs and kwargs["flight"] is not None:
            kwargs["flight"] = int(float(kwargs["flight"]))

        row_seed = None if seed is None else seed + row_idx

        counts, expression, z_all, metadata, labels = generate_samples(
            model, dataset, latent_dim, n,
            device=device, seed=row_seed, **kwargs
        )

        # save per-condition outputs
        label_parts = []
        for col in optional:
            if col in kwargs and kwargs[col] is not None:
                label_parts.append(str(kwargs[col]).replace("/", "-").replace(" ", "_"))
        label_parts.append("n" + str(n))
        run_label = "_".join(label_parts) if label_parts else "condition_" + str(row_idx)

        condition_dir = out_dir / run_label
        save_outputs(counts, expression, z_all, metadata, labels,
                     dataset.gene_symbols, dataset.ensembl_ids, condition_dir)

        all_metadata.append(metadata)
        all_counts.append(counts)
        all_expr.append(expression)

    # save combined outputs
    print("\nSaving combined outputs...")
    combined_meta   = pd.concat(all_metadata, ignore_index=True)
    combined_counts = np.vstack(all_counts)
    combined_expr   = np.vstack(all_expr)
    sample_ids      = combined_meta["sample_id"].tolist()

    combined_counts_df = pd.DataFrame(
        combined_counts, index=sample_ids, columns=dataset.gene_symbols
    )
    combined_counts_df.index.name = "sample_id"
    combined_counts_df.to_csv(out_dir / "all_synthetic_counts.csv")

    combined_expr_df = pd.DataFrame(
        combined_expr, index=sample_ids, columns=dataset.gene_symbols
    )
    combined_expr_df.index.name = "sample_id"
    combined_expr_df.to_csv(out_dir / "all_synthetic_expression.csv")

    combined_meta.to_csv(out_dir / "all_synthetic_metadata.csv", index=False)
    print("  Saved: " + str(out_dir / "all_synthetic_counts.csv") +
          " (" + str(combined_counts.shape[0]) + " total samples)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    print("Loading dataset...")
    dataset = SpaceflightDataset(args.data)
    model, latent_dim = load_model(args.checkpoint, dataset, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        # multi-condition generation from config file
        generate_from_config(
            args.config, model, dataset, latent_dim,
            out_dir, device, seed=args.seed
        )
    else:
        # single condition generation from CLI args
        flight = int(args.flight) if args.flight is not None else None

        counts, expression, z_all, metadata, labels = generate_samples(
            model=model,
            dataset=dataset,
            latent_dim=latent_dim,
            n=args.n,
            tissue=args.tissue,
            strain=args.strain,
            sex=args.sex,
            flight=flight,
            euth=args.euth,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # build output subdirectory name from conditions
        parts = []
        for val, name in [(args.tissue, "tissue"), (args.strain, "strain"),
                          (args.sex, "sex"), (str(flight), "flt"),
                          (args.euth, "euth")]:
            if val is not None:
                parts.append(str(val).replace("/", "-").replace(" ", "_"))
        parts.append("n" + str(args.n))
        run_label = "_".join(parts) if parts else "synthetic"
        run_dir   = out_dir / run_label

        print("\nSaving outputs to: " + str(run_dir))
        save_outputs(
            counts, expression, z_all, metadata, labels,
            dataset.gene_symbols, dataset.ensembl_ids, run_dir
        )

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic bulk RNA-seq samples from Spaceflight cVAE"
    )

    # required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data",       type=str, required=True,
                        help="Path to subset_final.h5")

    # generation mode
    parser.add_argument("--n",      type=int, default=100,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--config", type=str, default=None,
                        help="CSV config file for multi-condition generation")

    # condition filters (all optional — None = sample randomly)
    parser.add_argument("--tissue", type=str, default=None,
                        help="Tissue type, e.g. 'Liver'")
    parser.add_argument("--strain", type=str, default=None,
                        help="Mouse strain, e.g. 'C57BL/6J'")
    parser.add_argument("--sex",    type=str, default=None,
                        help="Sex: Female, Male, or Unknown")
    parser.add_argument("--flight", type=int, default=None,
                        choices=[0, 1],
                        help="Spaceflight status: 0=ground, 1=spaceflight")
    parser.add_argument("--euth",   type=str, default=None,
                        help="Euthanasia method, e.g. 'Isoflurane'")

    # generation parameters
    parser.add_argument("--batch_size",  type=int, default=256,
                        help="Decode this many samples at once (default 256)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir",  type=str, default="synthetic_samples",
                        help="Output directory")

    args = parser.parse_args()
    run(args)
