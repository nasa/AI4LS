"""
What-If Analysis for Spaceflight cVAE
======================================
Two analysis modes:

  COUNTERFACTUAL
    Take real samples matching a filter, encode them, change one
    condition in the decoder, compare predicted vs original expression.
    Example: "What would these ground-control liver samples look like
              under spaceflight?"

  POPULATION
    Average the latent vectors for a condition combination, then decode
    twice — once with flight=1, once with flight=0 — and compare.
    Example: "What does average Liver/C57BL6J/Female expression look
              like in space vs on the ground?"

Usage:

    # counterfactual: flip ground liver/C57BL6J/Female to spaceflight
    python whatif.py \\
        --checkpoint checkpoints_v5/best_model.pt \\
        --data subset_final.h5 \\
        --output_dir whatif_results/ \\
        --mode counterfactual \\
        --tissue Liver --strain C57BL/6J --sex Female \\
        --change_condition flight --change_from 0 --change_to 1

    # population: compare flight vs ground for Kidney/BALB/c/Male
    python whatif.py \\
        --checkpoint checkpoints_v5/best_model.pt \\
        --data subset_final.h5 \\
        --output_dir whatif_results/ \\
        --mode population \\
        --tissue Kidney --strain BALB/c --sex Male

    # change tissue: what if liver samples were kidney?
    python whatif.py \\
        --checkpoint checkpoints_v5/best_model.pt \\
        --data subset_final.h5 \\
        --output_dir whatif_results/ \\
        --mode counterfactual \\
        --tissue Liver --flight 0 \\
        --change_condition tissue \\
        --change_from Liver --change_to Kidney

Outputs (saved under output_dir/<run_label>/):

  Both modes:
    sample_metadata.csv              which samples were selected
    original_expression.csv          real log1p-normalized expression (samples x genes)
    counterfactual_expression.csv    model-predicted expression (samples x genes)
    delta_expression.csv             per-gene difference ranked by abs(delta)
    top_up_genes.csv                 top upregulated genes (delta > 0)
    top_down_genes.csv               top downregulated genes (delta < 0)

  Population mode also:
    population_flight_expression.csv  predicted expression under spaceflight
    population_ground_expression.csv  predicted expression under ground control
    population_delta.csv              flight - ground ranked by abs(delta)
    population_top_up_genes.csv
    population_top_down_genes.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

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
    return model


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(dataset, tissue=None, strain=None, sex=None,
                   euth=None, flight=None):
    """
    Return indices of samples matching ALL specified filters.
    Pass None to match any value for that covariate.

    Returns:
        indices: np.ndarray of matching sample indices
        meta_df: DataFrame describing the selected samples
    """
    mask = np.ones(dataset.n_samples, dtype=bool)

    def apply_filter(values, encoder, label, value):
        if value is None:
            return values
        if value not in encoder.classes_:
            raise ValueError(
                "Unknown " + label + ": '" + str(value) + "'" +
                "\nAvailable: " + str(list(encoder.classes_))
            )
        idx = encoder.transform([value])[0]
        return values & (getattr(dataset, label + "_ids") == idx)

    mask = apply_filter(mask, dataset.tissue_enc, "tissue", tissue)
    mask = apply_filter(mask, dataset.strain_enc, "strain", strain)
    mask = apply_filter(mask, dataset.sex_enc,    "sex",    sex)
    mask = apply_filter(mask, dataset.euth_enc,   "euth",   euth)

    if flight is not None:
        mask &= (dataset.flight == int(flight))

    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(
            "No samples match the specified filters. "
            "Try relaxing tissue, strain, sex, or flight."
        )

    meta_df = pd.DataFrame({
        "sample_idx": indices,
        "tissue":  dataset.tissue_enc.inverse_transform(dataset.tissue_ids[indices]),
        "strain":  dataset.strain_enc.inverse_transform(dataset.strain_ids[indices]),
        "sex":     dataset.sex_enc.inverse_transform(dataset.sex_ids[indices]),
        "study":   dataset.study_enc.inverse_transform(dataset.study_ids[indices]),
        "euth":    dataset.euth_enc.inverse_transform(dataset.euth_ids[indices]),
        "flight":  dataset.flight[indices],
    })
    print("Selected " + str(len(indices)) + " samples:")
    summary = (meta_df.groupby(["tissue", "strain", "sex", "flight"])
               .size().reset_index(name="n"))
    print(summary.to_string(index=False))
    return indices, meta_df


# ---------------------------------------------------------------------------
# NB mean: decoder outputs -> expected expression
# ---------------------------------------------------------------------------

def nb_mean_expr(log_r, p):
    """
    Convert NB decoder outputs to expected expression (count space).
    NB mean = r * (1 - p) / p
    """
    r = torch.exp(log_r).clamp(min=1e-4)
    p = p.clamp(min=1e-6, max=1 - 1e-6)
    return r * (1 - p) / p


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode_samples(model, dataset, indices, device="cpu"):
    """
    Encode selected samples, return (mu tensor, list of batch dicts).
    Uses mu (no sampling) for deterministic counterfactuals.
    """
    loader   = DataLoader(Subset(dataset, indices),
                          batch_size=64, shuffle=False, num_workers=2)
    all_mu   = []
    all_batches = []

    with torch.no_grad():
        for batch in loader:
            x      = batch["x"].to(device)
            strain = batch["strain"].to(device)
            sex    = batch["sex"].to(device)
            study  = batch["study"].to(device)
            tissue = batch["tissue"].to(device)
            euth   = batch["euth"].to(device)
            flight = batch["flight"].to(device)
            mu     = model.encode(x, strain, sex, study, tissue, euth, flight)
            all_mu.append(mu)
            all_batches.append({k: v.to(device) for k, v in batch.items()})

    return torch.cat(all_mu, dim=0), all_batches


# ---------------------------------------------------------------------------
# Decode with modified condition
# ---------------------------------------------------------------------------

def decode_modified(model, dataset, mu_tensor, batches,
                    change_condition, change_to, device="cpu"):
    """
    Decode latent vectors with one condition replaced.

    Args:
        mu_tensor:        (N, latent_dim)
        batches:          list of batch dicts from encode_samples
        change_condition: 'flight', 'tissue', 'strain', 'sex', or 'euth'
        change_to:        new value — int for flight, str for others

    Returns:
        expr: (N, n_genes) numpy array of predicted NB mean expression
    """
    # resolve the new encoded integer value
    if change_condition == "flight":
        new_val = int(change_to)
    elif change_condition == "tissue":
        if change_to not in dataset.tissue_enc.classes_:
            raise ValueError("Unknown tissue: " + str(change_to))
        new_val = int(dataset.tissue_enc.transform([change_to])[0])
    elif change_condition == "strain":
        if change_to not in dataset.strain_enc.classes_:
            raise ValueError("Unknown strain: " + str(change_to))
        new_val = int(dataset.strain_enc.transform([change_to])[0])
    elif change_condition == "sex":
        if change_to not in dataset.sex_enc.classes_:
            raise ValueError("Unknown sex: " + str(change_to))
        new_val = int(dataset.sex_enc.transform([change_to])[0])
    elif change_condition == "euth":
        if change_to not in dataset.euth_enc.classes_:
            raise ValueError("Unknown euthanasia method: " + str(change_to) +
                             "\nAvailable: " + str(list(dataset.euth_enc.classes_)))
        new_val = int(dataset.euth_enc.transform([change_to])[0])
    else:
        raise ValueError("change_condition must be one of: "
                         "flight, tissue, strain, sex, euth")

    all_expr = []
    offset   = 0

    with torch.no_grad():
        for batch in batches:
            n      = batch["x"].shape[0]
            z      = mu_tensor[offset:offset + n]
            strain = batch["strain"].clone()
            sex    = batch["sex"].clone()
            study  = batch["study"].clone()
            tissue = batch["tissue"].clone()
            euth   = batch["euth"].clone()
            flight = batch["flight"].clone()

            # apply the change
            # create a batch-sized tensor filled with new_val
            # (dtype=long, same device as other tensors)
            new_t = torch.full(
                (z.shape[0],), new_val, dtype=torch.long, device=device
            )
            if change_condition == "flight":
                flight = new_t
            elif change_condition == "tissue":
                tissue = new_t
            elif change_condition == "strain":
                strain = new_t
            elif change_condition == "sex":
                sex = new_t
            elif change_condition == "euth":
                euth = new_t

            log_r, p = model.generate(z, strain, sex, study, tissue, euth, flight)
            expr     = nb_mean_expr(log_r, p).cpu().numpy()
            all_expr.append(expr)
            offset  += n

    return np.vstack(all_expr)   # (N, n_genes)


# ---------------------------------------------------------------------------
# Delta analysis: rank genes by expression change
# ---------------------------------------------------------------------------

def compute_delta(original_expr, counterfactual_expr,
                  gene_symbols, ensembl_ids, n_top=200):
    """
    Compute per-gene mean expression change and rank by abs(delta).

    Args:
        original_expr:        (N, G) array — original expression
        counterfactual_expr:  (N, G) array — counterfactual expression
        gene_symbols:         (G,) array
        ensembl_ids:          (G,) array
        n_top:                number of top genes to return in up/down lists

    Returns:
        delta_df:    (G,) DataFrame sorted by abs(delta), all genes
        top_up_df:   top upregulated genes (counterfactual > original)
        top_down_df: top downregulated genes (counterfactual < original)
    """
    mean_orig  = original_expr.mean(axis=0)        # (G,)
    mean_cf    = counterfactual_expr.mean(axis=0)  # (G,)
    delta      = mean_cf - mean_orig
    abs_delta  = np.abs(delta)

    delta_df = pd.DataFrame({
        "ensembl_id":        ensembl_ids,
        "symbol":            gene_symbols,
        "mean_original":     mean_orig,
        "mean_counterfactual": mean_cf,
        "delta":             delta,
        "abs_delta":         abs_delta,
    }).sort_values("abs_delta", ascending=False).reset_index(drop=True)
    delta_df.insert(0, "rank", range(1, len(delta_df) + 1))

    top_up_df   = (delta_df[delta_df["delta"] > 0]
                   .head(n_top).reset_index(drop=True))
    top_down_df = (delta_df[delta_df["delta"] < 0]
                   .sort_values("delta").head(n_top).reset_index(drop=True))

    return delta_df, top_up_df, top_down_df


# ---------------------------------------------------------------------------
# Expression to DataFrame helper
# ---------------------------------------------------------------------------

def expr_to_df(expr, gene_symbols, ensembl_ids, sample_ids=None):
    """
    Convert expression matrix to a DataFrame with gene columns.
    Rows = samples, columns = genes.

    Args:
        expr:        (N, G) numpy array
        sample_ids:  optional list of sample identifiers for row index
    """
    df = pd.DataFrame(
        expr,
        columns=gene_symbols,
        index=sample_ids if sample_ids is not None else range(len(expr)),
    )
    df.index.name = "sample"
    return df


# ---------------------------------------------------------------------------
# Counterfactual mode
# ---------------------------------------------------------------------------

def run_counterfactual(args, model, dataset, out_dir, device):
    """
    Encode real samples, decode with one condition changed,
    save original + counterfactual expression and delta.
    """
    print("\n=== Counterfactual Analysis ===")
    print("Changing " + args.change_condition +
          " from " + str(args.change_from) +
          " to " + str(args.change_to))

    # parse flight filter
    flight_filter = None
    if args.flight is not None:
        flight_filter = int(args.flight)

    # select samples
    indices, meta_df = select_samples(
        dataset,
        tissue=args.tissue,
        strain=args.strain,
        sex=args.sex,
        euth=args.euth,
        flight=flight_filter,
    )
    meta_df.to_csv(out_dir / "sample_metadata.csv", index=False)

    # encode
    print("\nEncoding " + str(len(indices)) + " samples...")
    mu, batches = encode_samples(model, dataset, indices, device)

    # original expression (log1p normalized from dataset)
    orig_expr = dataset.x[indices]   # (N, G)

    # counterfactual expression
    print("Decoding with " + args.change_condition +
          " = " + str(args.change_to) + "...")
    cf_expr = decode_modified(
        model, dataset, mu, batches,
        change_condition=args.change_condition,
        change_to=args.change_to,
        device=device,
    )

    # save full expression matrices
    sample_ids = ["sample_" + str(i) for i in indices]
    expr_to_df(orig_expr, dataset.gene_symbols,
               dataset.ensembl_ids, sample_ids).to_csv(
        out_dir / "original_expression.csv"
    )
    expr_to_df(cf_expr, dataset.gene_symbols,
               dataset.ensembl_ids, sample_ids).to_csv(
        out_dir / "counterfactual_expression.csv"
    )
    print("  Saved: original_expression.csv")
    print("  Saved: counterfactual_expression.csv")

    # delta analysis
    delta_df, top_up, top_down = compute_delta(
        orig_expr, cf_expr,
        dataset.gene_symbols, dataset.ensembl_ids,
        n_top=args.n_top,
    )
    delta_df.to_csv(out_dir / "delta_expression.csv", index=False)
    top_up.to_csv(out_dir / "top_up_genes.csv", index=False)
    top_down.to_csv(out_dir / "top_down_genes.csv", index=False)
    print("  Saved: delta_expression.csv")
    print("  Saved: top_up_genes.csv (" + str(len(top_up)) + " genes)")
    print("  Saved: top_down_genes.csv (" + str(len(top_down)) + " genes)")

    print("\nTop 10 upregulated genes:")
    print(top_up[["rank", "symbol", "mean_original",
                  "mean_counterfactual", "delta"]].head(10).to_string(index=False))
    print("\nTop 10 downregulated genes:")
    print(top_down[["rank", "symbol", "mean_original",
                    "mean_counterfactual", "delta"]].head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Population mode
# ---------------------------------------------------------------------------

def run_population(args, model, dataset, out_dir, device):
    """
    Average z vectors for the selected condition combination,
    decode under both flight=1 and flight=0, compare.
    """
    print("\n=== Population-Level Analysis ===")

    # select samples (any flight status for averaging)
    flight_filter = None
    if args.flight is not None:
        flight_filter = int(args.flight)

    indices, meta_df = select_samples(
        dataset,
        tissue=args.tissue,
        strain=args.strain,
        sex=args.sex,
        euth=args.euth,
        flight=flight_filter,
    )
    meta_df.to_csv(out_dir / "sample_metadata.csv", index=False)

    # encode and average z
    print("\nEncoding " + str(len(indices)) + " samples...")
    mu, batches = encode_samples(model, dataset, indices, device)
    mu_mean     = mu.mean(dim=0, keepdim=True)   # (1, latent_dim)
    print("  Mean latent vector shape: " + str(tuple(mu_mean.shape)))

    # build a single "representative" batch using the first sample's
    # covariates (study embedding) but with averaged z
    ref_batch = batches[0]
    ref_single = {k: v[:1] for k, v in ref_batch.items()}

    # decode under spaceflight (flight=1)
    print("Decoding under spaceflight (flight=1)...")
    cf_flight = decode_modified(
        model, dataset, mu_mean, [ref_single],
        change_condition="flight", change_to=1, device=device,
    )

    # decode under ground control (flight=0)
    print("Decoding under ground control (flight=0)...")
    cf_ground = decode_modified(
        model, dataset, mu_mean, [ref_single],
        change_condition="flight", change_to=0, device=device,
    )

    # original expression (mean of selected samples)
    orig_expr  = dataset.x[indices]
    orig_mean  = orig_expr.mean(axis=0, keepdims=True)

    # save full expression matrices (single row = population mean)
    expr_to_df(orig_mean,  dataset.gene_symbols,
               dataset.ensembl_ids, ["population_mean"]).to_csv(
        out_dir / "original_expression.csv"
    )
    expr_to_df(cf_flight, dataset.gene_symbols,
               dataset.ensembl_ids, ["population_spaceflight"]).to_csv(
        out_dir / "population_flight_expression.csv"
    )
    expr_to_df(cf_ground, dataset.gene_symbols,
               dataset.ensembl_ids, ["population_ground_expression"]).to_csv(
        out_dir / "population_ground_expression.csv"
    )
    print("  Saved: original_expression.csv")
    print("  Saved: population_flight_expression.csv")
    print("  Saved: population_ground_expression.csv")

    # delta: flight vs ground
    delta_df, top_up, top_down = compute_delta(
        cf_ground, cf_flight,
        dataset.gene_symbols, dataset.ensembl_ids,
        n_top=args.n_top,
    )
    delta_df.to_csv(out_dir / "population_delta.csv", index=False)
    top_up.to_csv(out_dir / "population_top_up_genes.csv", index=False)
    top_down.to_csv(out_dir / "population_top_down_genes.csv", index=False)
    print("  Saved: population_delta.csv")
    print("  Saved: population_top_up_genes.csv (" +
          str(len(top_up)) + " genes)")
    print("  Saved: population_top_down_genes.csv (" +
          str(len(top_down)) + " genes)")

    print("\nTop 10 upregulated in spaceflight:")
    print(top_up[["rank", "symbol", "mean_original",
                  "mean_counterfactual", "delta"]].head(10).to_string(index=False))
    print("\nTop 10 downregulated in spaceflight:")
    print(top_down[["rank", "symbol", "mean_original",
                    "mean_counterfactual", "delta"]].head(10).to_string(index=False))



# ---------------------------------------------------------------------------
# Interaction analysis: strain x flight (or any covariate x flight)
# ---------------------------------------------------------------------------

def run_interaction(args, model, dataset, out_dir, device):
    """
    Compute the interaction effect between flight and a second covariate.

    For each pair of values of covariate_b, computes:
        delta_A = flight_effect(tissue, strain_A, sex, euth)
        delta_B = flight_effect(tissue, strain_B, sex, euth)
        interaction = delta_A - delta_B

    Genes with large interaction are those where the spaceflight response
    differs between the two covariate values.

    Example:
        Does spaceflight affect C57BL/6J liver differently than BALB/c liver?

    Usage:
        python whatif.py \
            --mode interaction \
            --tissue Liver \
            --interact_condition strain \
            --interact_value_a C57BL/6J \
            --interact_value_b BALB/c
    """
    print("\n=== Interaction Analysis ===")
    print("Tissue: " + str(args.tissue))
    print("Interact condition: " + args.interact_condition)
    print("Value A: " + args.interact_value_a)
    print("Value B: " + args.interact_value_b)

    def get_flight_delta(condition_val):
        """
        For a given value of interact_condition, compute the population-level
        flight effect: mean(z | condition) decoded under flight=1 minus flight=0.
        """
        # build filter kwargs — set interact_condition to condition_val
        filter_kwargs = {
            "tissue": args.tissue,
            "strain": args.strain,
            "sex":    args.sex,
            "euth":   args.euth,
            "flight": None,   # include both conditions for averaging
        }
        # override the interact condition with specific value
        filter_kwargs[args.interact_condition] = condition_val

        indices, meta_df = select_samples(dataset, **filter_kwargs)
        print("  " + condition_val + ": " + str(len(indices)) + " samples")

        mu, batches = encode_samples(model, dataset, indices, device)
        mu_mean = mu.mean(dim=0, keepdim=True)
        ref_single = {k: v[:1] for k, v in batches[0].items()}

        # decode under flight=1
        cf_flight = decode_modified(
            model, dataset, mu_mean, [ref_single],
            change_condition="flight", change_to=1, device=device,
        )
        # decode under flight=0
        cf_ground = decode_modified(
            model, dataset, mu_mean, [ref_single],
            change_condition="flight", change_to=0, device=device,
        )
        return cf_flight[0] - cf_ground[0]   # (n_genes,) flight effect

    print("\nComputing flight effect for " + args.interact_value_a + "...")
    delta_a = get_flight_delta(args.interact_value_a)

    print("Computing flight effect for " + args.interact_value_b + "...")
    delta_b = get_flight_delta(args.interact_value_b)

    # interaction = difference in flight effects
    interaction = delta_a - delta_b
    abs_interaction = np.abs(interaction)

    df = pd.DataFrame({
        "ensembl_id":    dataset.ensembl_ids,
        "symbol":        dataset.gene_symbols,
        "flight_delta_" + args.interact_value_a.replace("/", "-"): delta_a,
        "flight_delta_" + args.interact_value_b.replace("/", "-"): delta_b,
        "interaction":   interaction,
        "abs_interaction": abs_interaction,
    }).sort_values("abs_interaction", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    # split into genes where A responds more vs B responds more
    top_a = df[df["interaction"] > 0].head(args.n_top).reset_index(drop=True)
    top_b = df[df["interaction"] < 0].head(args.n_top).reset_index(drop=True)

    df.to_csv(out_dir / "interaction_all_genes.csv", index=False)
    top_a.to_csv(out_dir / ("stronger_in_" + args.interact_value_a.replace("/", "-") + ".csv"), index=False)
    top_b.to_csv(out_dir / ("stronger_in_" + args.interact_value_b.replace("/", "-") + ".csv"), index=False)

    print("\nSaved: interaction_all_genes.csv")
    print("Saved: stronger_in_" + args.interact_value_a.replace("/", "-") + ".csv")
    print("Saved: stronger_in_" + args.interact_value_b.replace("/", "-") + ".csv")

    print("\nTop 10 genes with stronger spaceflight response in " + args.interact_value_a + ":")
    print(top_a[["rank", "symbol",
                  "flight_delta_" + args.interact_value_a.replace("/", "-"),
                  "flight_delta_" + args.interact_value_b.replace("/", "-"),
                  "interaction"]].head(10).to_string(index=False))

    print("\nTop 10 genes with stronger spaceflight response in " + args.interact_value_b + ":")
    print(top_b[["rank", "symbol",
                  "flight_delta_" + args.interact_value_a.replace("/", "-"),
                  "flight_delta_" + args.interact_value_b.replace("/", "-"),
                  "interaction"]].head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Euthanasia artifact quantification
# ---------------------------------------------------------------------------

def run_artifact(args, model, dataset, out_dir, device):
    """
    Quantify how much of the observed spaceflight signal is confounded
    by euthanasia method.

    For a given tissue, computes the predicted expression under spaceflight
    for each euthanasia method, then identifies genes where the euthanasia
    method changes the apparent spaceflight response.

    This answers: "Which spaceflight DE genes are actually euthanasia artifacts?"

    Usage:
        python whatif.py \
            --mode artifact \
            --tissue Liver \
            --euth_a Isoflurane \
            --euth_b CO2
    """
    print("\n=== Euthanasia Artifact Analysis ===")
    print("Tissue:    " + str(args.tissue))
    print("Euth A:    " + args.euth_a)
    print("Euth B:    " + args.euth_b)

    def get_spaceflight_effect(euth_val):
        """
        For a given euthanasia method, compute population-level
        spaceflight effect: decoded flight=1 minus decoded flight=0.
        """
        filter_kwargs = {
            "tissue": args.tissue,
            "strain": args.strain,
            "sex":    args.sex,
            "euth":   euth_val,
            "flight": None,
        }
        indices, meta_df = select_samples(dataset, **filter_kwargs)
        n_f = int((dataset.flight[indices] == 1).sum())
        n_g = int((dataset.flight[indices] == 0).sum())
        print("  " + euth_val + ": " + str(len(indices)) +
              " samples (" + str(n_f) + "f/" + str(n_g) + "g)")

        mu, batches = encode_samples(model, dataset, indices, device)
        mu_mean  = mu.mean(dim=0, keepdim=True)
        ref_single = {k: v[:1] for k, v in batches[0].items()}

        cf_flight = decode_modified(
            model, dataset, mu_mean, [ref_single],
            change_condition="flight", change_to=1, device=device,
        )
        cf_ground = decode_modified(
            model, dataset, mu_mean, [ref_single],
            change_condition="flight", change_to=0, device=device,
        )
        return cf_flight[0] - cf_ground[0]   # (n_genes,) flight effect

    print("\nComputing spaceflight effect under " + args.euth_a + "...")
    delta_a = get_spaceflight_effect(args.euth_a)

    print("Computing spaceflight effect under " + args.euth_b + "...")
    delta_b = get_spaceflight_effect(args.euth_b)

    # artifact = genes where euthanasia method changes the flight signal
    artifact    = delta_a - delta_b
    abs_artifact = np.abs(artifact)

    # concordant = genes robust across both euthanasia methods
    # (same sign and both large)
    concordant = np.sign(delta_a) == np.sign(delta_b)
    min_effect  = np.minimum(np.abs(delta_a), np.abs(delta_b))

    df = pd.DataFrame({
        "ensembl_id":             dataset.ensembl_ids,
        "symbol":                 dataset.gene_symbols,
        "flight_effect_" + args.euth_a: delta_a,
        "flight_effect_" + args.euth_b: delta_b,
        "artifact_delta":         artifact,
        "abs_artifact":           abs_artifact,
        "concordant":             concordant,
        "min_effect_both_euths":  min_effect,
    })

    # Robust genes: concordant direction AND meaningful effect in both methods
    robust = df[df["concordant"] & (df["min_effect_both_euths"] > 0.1)].copy()
    robust = robust.sort_values("min_effect_both_euths", ascending=False).reset_index(drop=True)
    robust.insert(0, "rank", range(1, len(robust) + 1))

    # Artifact genes: large discordance between euthanasia methods
    artifacts = df.sort_values("abs_artifact", ascending=False).head(args.n_top).reset_index(drop=True)
    artifacts.insert(0, "rank", range(1, len(artifacts) + 1))

    df.sort_values("abs_artifact", ascending=False).reset_index(drop=True).to_csv(
        out_dir / "all_genes_artifact_analysis.csv", index=False
    )
    robust.to_csv(out_dir / "robust_spaceflight_genes.csv", index=False)
    artifacts.to_csv(out_dir / "potential_artifact_genes.csv", index=False)

    print("\nSaved: all_genes_artifact_analysis.csv")
    print("Saved: robust_spaceflight_genes.csv  (" + str(len(robust)) + " genes)")
    print("Saved: potential_artifact_genes.csv  (" + str(len(artifacts)) + " genes)")

    print("\nTop 10 robust spaceflight genes (consistent across both euthanasia methods):")
    cols = ["rank", "symbol",
            "flight_effect_" + args.euth_a,
            "flight_effect_" + args.euth_b,
            "min_effect_both_euths"]
    print(robust[cols].head(10).to_string(index=False))

    print("\nTop 10 potential artifact genes (diverge by euthanasia method):")
    art_cols = ["rank", "symbol",
                "flight_effect_" + args.euth_a,
                "flight_effect_" + args.euth_b,
                "artifact_delta"]
    print(artifacts[art_cols].head(10).to_string(index=False))

    # Summary stats
    n_robust    = len(robust)
    n_artifact  = int((abs_artifact > 0.1).sum())
    print("\n=== Summary ===")
    print("Robust spaceflight genes (concordant, |effect|>0.1 in both): " + str(n_robust))
    print("Potential artifacts (|artifact_delta|>0.1):                  " + str(n_artifact))
    print("Artifact fraction of active genes:                           " +
          f"{n_artifact / max(n_robust + n_artifact, 1):.1%}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    # load dataset and model
    print("Loading dataset...")
    dataset = SpaceflightDataset(args.data)
    model   = load_model(args.checkpoint, dataset, device)

    # build output label from filters
    parts = []
    if args.tissue:  parts.append(args.tissue.replace(" ", "_"))
    if args.strain:  parts.append(args.strain.replace("/", "-"))
    if args.sex:     parts.append(args.sex)
    if args.flight is not None: parts.append("flt" + str(args.flight))
    if args.mode == "counterfactual":
        parts.append(args.change_condition + "_" +
                     str(args.change_from) + "_to_" + str(args.change_to))
    parts.append(args.mode)
    run_label = "_".join(parts) if parts else args.mode

    out_dir = Path(args.output_dir) / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory: " + str(out_dir))

    if args.mode == "counterfactual":
        if not args.change_condition:
            raise ValueError("--change_condition required for counterfactual mode")
        if args.change_from is None or args.change_to is None:
            raise ValueError("--change_from and --change_to required for counterfactual mode")
        run_counterfactual(args, model, dataset, out_dir, device)
    elif args.mode == "population":
        run_population(args, model, dataset, out_dir, device)
    elif args.mode == "interaction":
        if not args.interact_condition:
            raise ValueError("--interact_condition required for interaction mode")
        if not args.interact_value_a or not args.interact_value_b:
            raise ValueError("--interact_value_a and --interact_value_b required for interaction mode")
        run_interaction(args, model, dataset, out_dir, device)
    elif args.mode == "artifact":
        if not args.euth_a or not args.euth_b:
            raise ValueError("--euth_a and --euth_b required for artifact mode")
        run_artifact(args, model, dataset, out_dir, device)
    else:
        raise ValueError("--mode must be counterfactual, population, interaction, or artifact")

    print("\nDone. Results saved to: " + str(out_dir))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="What-if analysis for Spaceflight cVAE"
    )

    # required
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data",       type=str, required=True)
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["counterfactual", "population", "interaction", "artifact"],
                        help="counterfactual, population, interaction, or artifact")

    # sample filters (all optional — combine to narrow selection)
    parser.add_argument("--tissue", type=str, default=None,
                        help="Tissue to select, e.g. 'Liver'")
    parser.add_argument("--strain", type=str, default=None,
                        help="Strain to select, e.g. 'C57BL/6J'")
    parser.add_argument("--sex",    type=str, default=None,
                        help="Sex to select: Female, Male, or Unknown")
    parser.add_argument("--flight", type=int, default=None,
                        choices=[0, 1],
                        help="Flight status: 0=ground, 1=spaceflight")
    parser.add_argument("--euth",   type=str, default=None,
                        help="Euthanasia method filter, e.g. 'Isoflurane', 'CO2'")

    # counterfactual-specific
    parser.add_argument("--change_condition", type=str, default=None,
                        choices=["flight", "tissue", "strain", "sex", "euth"],
                        help="Which condition to change")
    parser.add_argument("--change_from", type=str, default=None,
                        help="Original value of the condition")
    parser.add_argument("--change_to",   type=str, default=None,
                        help="New value of the condition")

    # interaction mode
    parser.add_argument("--interact_condition", type=str, default=None,
                        choices=["strain", "tissue", "sex", "euth"],
                        help="Covariate to compare flight effects across (interaction mode)")
    parser.add_argument("--interact_value_a",   type=str, default=None,
                        help="First value of interact_condition, e.g. 'C57BL/6J'")
    parser.add_argument("--interact_value_b",   type=str, default=None,
                        help="Second value of interact_condition, e.g. 'BALB/c'")

    # artifact mode
    parser.add_argument("--euth_a", type=str, default=None,
                        help="First euthanasia method for artifact analysis, e.g. 'Isoflurane'")
    parser.add_argument("--euth_b", type=str, default=None,
                        help="Second euthanasia method for artifact analysis, e.g. 'CO2'")

    # output
    parser.add_argument("--output_dir", type=str, default="whatif_results")
    parser.add_argument("--n_top",      type=int, default=200,
                        help="Number of top up/down genes to report")

    args = parser.parse_args()
    run(args)
