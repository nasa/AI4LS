"""
K-Nearest Neighbors in cVAE Latent Space
==========================================
Given a query sample (real or synthetic), finds the k most similar
samples in the GeneLab dataset by Euclidean distance in latent space z.

This is useful for:
  - Finding real samples most similar to a counterfactual prediction
  - Identifying which studies/tissues are most similar to a query
  - Validating synthetic samples by checking their nearest real neighbors
  - Exploring the latent space neighborhood of any sample

Two query modes:
  --mode sample   Query is a real sample from subset_final.h5 (by index or metadata)
  --mode file     Query is a CSV/TSV of raw counts for a new sample

Usage:

    # find 10 nearest neighbors to sample index 42
    python knn_latent.py \\
        --checkpoint checkpoints_v9/best_model.pt \\
        --data subset_final.h5 \\
        --mode sample \\
        --sample_idx 42 \\
        --k 10

    # find nearest neighbors to a specific study/tissue/flight combination
    python knn_latent.py \\
        --checkpoint checkpoints_v9/best_model.pt \\
        --data subset_final.h5 \\
        --mode sample \\
        --tissue Liver \\
        --strain C57BL/6J \\
        --flight 1 \\
        --k 20

    # find nearest neighbors to a new external sample (raw counts CSV)
    python knn_latent.py \\
        --checkpoint checkpoints_v9/best_model.pt \\
        --data subset_final.h5 \\
        --mode file \\
        --query_file my_sample.csv \\
        --query_tissue Liver \\
        --query_strain C57BL/6J \\
        --query_sex Female \\
        --query_euth Isoflurane \\
        --k 10

Outputs (saved to output_dir/):
    knn_results.csv      k nearest neighbors with metadata and distances
    query_info.txt       description of the query sample
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import SpaceflightDataset
from model import SpaceflightCVAE


# ---------------------------------------------------------------------------
# Model and latent space loading
# ---------------------------------------------------------------------------


def _arch_from_state_dict(ckpt):
    """
    Read true model architecture from state dict weights.
    More reliable than ckpt["args"] which may be stale or incorrect.
    """
    sd   = ckpt["model_state"]

    # which condition embeddings actually exist in the weights
    conditions = [
        c for c in ["tissue", "strain", "sex", "study", "euth"]
        if f"embedder.embeddings.{c}.weight" in sd
    ]
    if not conditions:
        conditions = ckpt.get("args", {}).get(
            "conditions", ["tissue","strain","sex","study","euth"])

    # latent dim from μ head output size
    latent_dim = sd["encoder.mu.weight"].shape[0]

    # embedding dims from weight shapes
    def emb_dim(name, default):
        key = f"embedder.embeddings.{name}.weight"
        return int(sd[key].shape[1]) if key in sd else default

    return dict(
        conditions=conditions,
        latent_dim=latent_dim,
        tissue_emb_dim=emb_dim("tissue", 32),
        strain_emb_dim=emb_dim("strain", 16),
        sex_emb_dim=emb_dim("sex",     4),
        study_emb_dim=emb_dim("study",  16),
        euth_emb_dim=emb_dim("euth",    8),
    )

def load_model(checkpoint_path, dataset, device="cpu"):
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch  = _arch_from_state_dict(ckpt)
    model = SpaceflightCVAE(
        n_genes=dataset.n_genes,
        n_strains=dataset.n_strains,
        n_sexes=dataset.n_sexes,
        n_studies=dataset.n_studies,
        n_tissues=dataset.n_tissues,
        n_euths=dataset.n_euths,
        **arch,
        hidden_dims=[256, 128],
        dropout=0.0,
        grl_alpha=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded {checkpoint_path} | epoch={ckpt['epoch']} "
          f"val_loss={ckpt['val_loss']:.4f} | "
          f"conditions={arch['conditions']} latent_dim={arch['latent_dim']}")
    return model, arch["latent_dim"]


def encode_all_samples(model, dataset, device, batch_size=256):
    """
    Encode all samples in the dataset to latent space.
    Returns z (n_samples, latent_dim) and metadata arrays.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_z  = []

    with torch.no_grad():
        for batch in loader:
            mu = model.encode(
                batch["x"].to(device),
                batch["strain"].to(device),
                batch["sex"].to(device),
                batch["study"].to(device),
                batch["tissue"].to(device),
                batch["euth"].to(device),
                batch["flight"].to(device),
            )
            all_z.append(mu.cpu().numpy())

    z_all = np.concatenate(all_z, axis=0)   # (n_samples, latent_dim)
    print(f"Encoded {len(z_all):,} samples → latent dim {z_all.shape[1]}")
    return z_all


# ---------------------------------------------------------------------------
# KNN search
# ---------------------------------------------------------------------------

def knn_search(query_z, reference_z, k, metric="euclidean"):
    """
    Find k nearest neighbors to query_z in reference_z.

    Args:
        query_z:     (latent_dim,) query vector
        reference_z: (n_samples, latent_dim) reference matrix
        k:           number of neighbors to return
        metric:      'euclidean' or 'cosine'

    Returns:
        indices:   (k,) indices into reference_z
        distances: (k,) distances to each neighbor
    """
    if metric == "euclidean":
        # efficient squared euclidean: ||a-b||² = ||a||² + ||b||² - 2a·b
        q = query_z.reshape(1, -1)
        dists = np.sqrt(
            np.sum((reference_z - q) ** 2, axis=1)
        )
    elif metric == "cosine":
        q_norm   = query_z / (np.linalg.norm(query_z) + 1e-8)
        ref_norm = reference_z / (np.linalg.norm(reference_z, axis=1, keepdims=True) + 1e-8)
        cos_sim  = ref_norm @ q_norm
        dists    = 1.0 - cos_sim   # cosine distance
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'euclidean' or 'cosine'.")

    # sort by distance
    sorted_idx = np.argsort(dists)
    top_k_idx  = sorted_idx[:k]
    top_k_dist = dists[top_k_idx]

    return top_k_idx, top_k_dist


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------

def encode_sample_query(model, dataset, device, args):
    """
    Encode a query from a real sample in the dataset.
    Can specify by index or by metadata filters.
    """
    if args.sample_idx is not None:
        idx = args.sample_idx
        print(f"Query: sample index {idx}")
    else:
        # find samples matching metadata filters
        mask = np.ones(dataset.n_samples, dtype=bool)
        desc = []

        if args.tissue:
            t_idx = dataset.tissue_enc.transform([args.tissue])[0]
            mask &= dataset.tissue_ids == t_idx
            desc.append(f"tissue={args.tissue}")
        if args.strain:
            s_idx = dataset.strain_enc.transform([args.strain])[0]
            mask &= dataset.strain_ids == s_idx
            desc.append(f"strain={args.strain}")
        if args.sex:
            sx_idx = dataset.sex_enc.transform([args.sex])[0]
            mask &= dataset.sex_ids == sx_idx
            desc.append(f"sex={args.sex}")
        if args.euth:
            e_idx = dataset.euth_enc.transform([args.euth])[0]
            mask &= dataset.euth_ids == e_idx
            desc.append(f"euth={args.euth}")
        if args.flight is not None:
            mask &= dataset.flight == args.flight
            desc.append(f"flight={args.flight}")

        matching = np.where(mask)[0]
        if len(matching) == 0:
            raise ValueError("No samples match the specified filters: " +
                             ", ".join(desc))

        print(f"Query filters: {', '.join(desc)}")
        print(f"  {len(matching)} matching samples — using mean latent vector")

        # encode all matching samples and average their z vectors
        batch_x      = torch.from_numpy(dataset.x[matching]).float()
        batch_strain = torch.from_numpy(dataset.strain_ids[matching]).long()
        batch_sex    = torch.from_numpy(dataset.sex_ids[matching]).long()
        batch_study  = torch.from_numpy(dataset.study_ids[matching]).long()
        batch_tissue = torch.from_numpy(dataset.tissue_ids[matching]).long()
        batch_euth   = torch.from_numpy(dataset.euth_ids[matching]).long()
        batch_flight = torch.from_numpy(dataset.flight[matching]).long()

        with torch.no_grad():
            mu = model.encode(
                batch_x.to(device),
                batch_strain.to(device),
                batch_sex.to(device),
                batch_study.to(device),
                batch_tissue.to(device),
                batch_euth.to(device),
                batch_flight.to(device),
            )
        query_z = mu.cpu().numpy().mean(axis=0)

        query_info = {
            "mode":     "sample (metadata filter)",
            "filters":  ", ".join(desc),
            "n_matching": len(matching),
            "method":   "mean of matching sample z vectors",
        }
        return query_z, query_info, matching

    # single sample by index
    item = dataset[idx]
    with torch.no_grad():
        mu = model.encode(
            item["x"].unsqueeze(0).to(device),
            item["strain"].unsqueeze(0).to(device),
            item["sex"].unsqueeze(0).to(device),
            item["study"].unsqueeze(0).to(device),
            item["tissue"].unsqueeze(0).to(device),
            item["euth"].unsqueeze(0).to(device),
            item["flight"].unsqueeze(0).to(device),
        )
    query_z = mu.cpu().numpy()[0]

    tissue  = dataset.tissue_enc.classes_[dataset.tissue_ids[idx]]
    strain  = dataset.strain_enc.classes_[dataset.strain_ids[idx]]
    sex     = dataset.sex_enc.classes_[dataset.sex_ids[idx]]
    study   = dataset.study_enc.classes_[dataset.study_ids[idx]]
    euth    = dataset.euth_enc.classes_[dataset.euth_ids[idx]]
    flight  = "spaceflight" if dataset.flight[idx] == 1 else "ground"

    query_info = {
        "mode":    "sample (by index)",
        "index":   idx,
        "tissue":  tissue,
        "strain":  strain,
        "sex":     sex,
        "study":   study,
        "euth":    euth,
        "flight":  flight,
    }
    return query_z, query_info, np.array([idx])


def encode_file_query(model, dataset, device, args):
    """
    Encode a query from an external CSV file of raw counts.

    CSV format: one row, columns = Ensembl IDs or gene symbols.
    Or: rows = genes, one column of counts.
    """
    print(f"Loading query from file: {args.query_file}")
    df = pd.read_csv(args.query_file, index_col=0)

    # figure out orientation (genes as columns or rows)
    if df.shape[0] == 1:
        # one row — genes are columns
        counts = df.iloc[0].values.astype(np.float32)
        gene_ids = df.columns.tolist()
    elif df.shape[1] == 1:
        # one column — genes are rows
        counts = df.iloc[:, 0].values.astype(np.float32)
        gene_ids = df.index.tolist()
    else:
        raise ValueError(
            "Query file must have exactly one sample. "
            f"Got shape {df.shape}. Expected (1, n_genes) or (n_genes, 1)."
        )

    # align to model gene set
    gene_id_set = set(gene_ids)
    expr = np.zeros(dataset.n_genes, dtype=np.float32)

    # try matching by ensembl ID first, then by symbol
    n_matched = 0
    for j, (ens, sym) in enumerate(zip(dataset.ensembl_ids, dataset.gene_symbols)):
        if ens in gene_id_set:
            idx = gene_ids.index(ens)
            expr[j] = counts[idx]
            n_matched += 1
        elif sym in gene_id_set:
            idx = gene_ids.index(sym)
            expr[j] = counts[idx]
            n_matched += 1

    print(f"  Matched {n_matched:,} / {dataset.n_genes:,} model genes")
    if n_matched < dataset.n_genes * 0.5:
        print("  WARNING: fewer than 50% of model genes matched. "
              "Check that gene IDs in your file match Ensembl IDs or symbols.")

    # normalize: log1p(CPM)
    lib_size  = max(expr.sum(), 1.0)
    x_norm    = np.log1p(expr / lib_size * 1e4)

    # resolve condition metadata
    def resolve(val, encoder, name):
        if val is None:
            return 0   # default to first category
        if val not in encoder.classes_:
            raise ValueError(f"Unknown {name}: '{val}'. "
                             f"Available: {list(encoder.classes_)}")
        return int(encoder.transform([val])[0])

    tissue_idx = resolve(args.query_tissue, dataset.tissue_enc, "tissue")
    strain_idx = resolve(args.query_strain, dataset.strain_enc, "strain")
    sex_idx    = resolve(args.query_sex,    dataset.sex_enc,    "sex")
    euth_idx   = resolve(args.query_euth,   dataset.euth_enc,   "euth")
    study_idx  = 0   # default study for external samples
    flight_val = int(args.query_flight) if args.query_flight is not None else 0

    with torch.no_grad():
        mu = model.encode(
            torch.tensor(x_norm).unsqueeze(0).float().to(device),
            torch.tensor([strain_idx]).long().to(device),
            torch.tensor([sex_idx]).long().to(device),
            torch.tensor([study_idx]).long().to(device),
            torch.tensor([tissue_idx]).long().to(device),
            torch.tensor([euth_idx]).long().to(device),
            torch.tensor([flight_val]).long().to(device),
        )
    query_z = mu.cpu().numpy()[0]

    query_info = {
        "mode":        "file",
        "file":        args.query_file,
        "n_matched":   n_matched,
        "query_tissue": args.query_tissue or "default",
        "query_strain": args.query_strain or "default",
        "query_sex":    args.query_sex    or "default",
        "query_euth":   args.query_euth   or "default",
        "query_flight": flight_val,
    }
    return query_z, query_info, np.array([])


# ---------------------------------------------------------------------------
# Build results DataFrame
# ---------------------------------------------------------------------------

def build_results(indices, distances, dataset, query_sample_indices, metric):
    """
    Build a DataFrame of KNN results with full metadata.
    Excludes the query sample itself if it's in the reference set.
    """
    rows = []
    query_set = set(query_sample_indices.tolist())

    rank = 1
    for idx, dist in zip(indices, distances):
        if idx in query_set:
            continue   # skip the query sample itself

        tissue = dataset.tissue_enc.classes_[dataset.tissue_ids[idx]]
        strain = dataset.strain_enc.classes_[dataset.strain_ids[idx]]
        sex    = dataset.sex_enc.classes_[dataset.sex_ids[idx]]
        study  = dataset.study_enc.classes_[dataset.study_ids[idx]]
        euth   = dataset.euth_enc.classes_[dataset.euth_ids[idx]]
        flight = "spaceflight" if dataset.flight[idx] == 1 else "ground"

        rows.append({
            "rank":     rank,
            "sample_idx": int(idx),
            f"{metric}_distance": float(dist),
            "tissue":   tissue,
            "strain":   strain,
            "sex":      sex,
            "study":    study,
            "euth":     euth,
            "flight":   flight,
        })
        rank += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load dataset and model
    print("\nLoading dataset...")
    dataset = SpaceflightDataset(args.data)

    print("\nLoading model...")
    model, latent_dim = load_model(args.checkpoint, dataset, device)

    # encode all samples in the reference dataset
    print("\nEncoding all reference samples...")
    z_all = encode_all_samples(model, dataset, device, batch_size=256)

    # encode query
    print("\nEncoding query...")
    if args.mode == "sample":
        query_z, query_info, query_sample_indices = encode_sample_query(
            model, dataset, device, args
        )
    else:
        query_z, query_info, query_sample_indices = encode_file_query(
            model, dataset, device, args
        )

    print(f"Query z norm: {np.linalg.norm(query_z):.3f}")

    # KNN search — retrieve k + len(query_sample_indices) so we can
    # exclude the query sample itself if it's in the reference
    k_retrieve = args.k + len(query_sample_indices) + 1
    print(f"\nSearching for {args.k} nearest neighbors "
          f"(metric: {args.metric})...")
    indices, distances = knn_search(query_z, z_all, k_retrieve, args.metric)

    # build results
    results = build_results(indices, distances, dataset,
                            query_sample_indices, args.metric)
    results = results.head(args.k)

    # print summary
    print(f"\n=== Top {args.k} Nearest Neighbors ===")
    print(results.to_string(index=False))

    # tissue distribution of neighbors
    print(f"\nTissue distribution of {args.k} neighbors:")
    for tissue, count in results["tissue"].value_counts().items():
        print(f"  {count:3d}  {tissue}")

    print(f"\nFlight status of {args.k} neighbors:")
    for status, count in results["flight"].value_counts().items():
        print(f"  {count:3d}  {status}")

    # save outputs
    results.to_csv(out_dir / "knn_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'knn_results.csv'}")

    with open(out_dir / "query_info.txt", "w") as f:
        f.write("=== Query Information ===\n\n")
        for k, v in query_info.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nk: {args.k}\n")
        f.write(f"metric: {args.metric}\n")
        f.write(f"latent_dim: {latent_dim}\n")
    print(f"Saved: {out_dir / 'query_info.txt'}")

    # optionally save the query z vector
    np.save(out_dir / "query_z.npy", query_z)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-Nearest Neighbors in cVAE latent space"
    )

    # required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data",       type=str, required=True,
                        help="Path to subset_final.h5")

    # query mode
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["sample", "file"],
                        help="'sample': query from dataset; 'file': query from CSV")

    # sample mode args
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Index of query sample in dataset (sample mode)")
    parser.add_argument("--tissue",     type=str, default=None,
                        help="Filter by tissue (sample mode)")
    parser.add_argument("--strain",     type=str, default=None,
                        help="Filter by strain (sample mode)")
    parser.add_argument("--sex",        type=str, default=None,
                        help="Filter by sex (sample mode)")
    parser.add_argument("--euth",       type=str, default=None,
                        help="Filter by euthanasia method (sample mode)")
    parser.add_argument("--flight",     type=int, default=None,
                        choices=[0, 1],
                        help="Filter by flight status 0/1 (sample mode)")

    # file mode args
    parser.add_argument("--query_file",   type=str, default=None,
                        help="CSV file of raw counts for query sample (file mode)")
    parser.add_argument("--query_tissue", type=str, default=None,
                        help="Tissue type for query sample (file mode)")
    parser.add_argument("--query_strain", type=str, default=None,
                        help="Strain for query sample (file mode)")
    parser.add_argument("--query_sex",    type=str, default=None,
                        help="Sex for query sample (file mode)")
    parser.add_argument("--query_euth",   type=str, default=None,
                        help="Euthanasia method for query sample (file mode)")
    parser.add_argument("--query_flight", type=int, default=None,
                        choices=[0, 1],
                        help="Flight status for query sample (file mode)")

    # KNN parameters
    parser.add_argument("--k",      type=int,   default=10,
                        help="Number of nearest neighbors (default 10)")
    parser.add_argument("--metric", type=str,   default="euclidean",
                        choices=["euclidean", "cosine"],
                        help="Distance metric (default euclidean)")

    # output
    parser.add_argument("--output_dir", type=str, default="knn_results",
                        help="Output directory")

    args = parser.parse_args()
    run(args)
