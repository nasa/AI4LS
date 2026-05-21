"""
Inference & Analysis for Spaceflight cVAE
==========================================
Usage:
    python inference.py \
        --checkpoint checkpoints_v5/best_model.pt \
        --data subset_final.h5 \
        --output_dir inference_results_v5/

    # offline (skip Enrichr)
    python inference.py ... --skip_enrichment

    # t-SNE tuning
    python inference.py ... --tsne_perplexity 50 --tsne_n_iter 2000

Outputs:
    test_metrics.txt
    latent_all.npz
    umap_{flight,study,strain,sex,tissue}.png
    tsne_{flight,study,strain,sex,tissue}.png
    gene_attribution.csv
    gene_attribution_clean.csv
    gene_attribution_by_tissue.csv
    by_tissue/<tissue>.csv
    enrichment_summary.csv
    enrichment/<tissue>/<library>.csv + .png
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, Subset

from dataset import SpaceflightDataset, make_dataloaders
from model import SpaceflightCVAE


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, dataset, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]
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
    print("Loaded model from " + str(checkpoint_path) +
          " (epoch " + str(ckpt["epoch"]) +
          ", val_loss=" + f"{ckpt['val_loss']:.4f})")
    return model


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

def get_latent_representations(model, dataloader, device="cpu"):
    """Returns dict: z, logits, probs, flight, study, strain, sex, tissue."""
    model.eval()
    zs, logits_list = [], []
    flights, studies, strains, sexes, tissues, euths = [], [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x      = batch["x"].to(device)
            strain = batch["strain"].to(device)
            sex    = batch["sex"].to(device)
            study  = batch["study"].to(device)
            tissue = batch["tissue"].to(device)
            euth   = batch["euth"].to(device)
            flight = batch["flight"].to(device)

            outputs = model(x, strain, sex, study, tissue, euth, flight)
            zs.append(outputs["mu"].cpu().numpy())
            logits_list.append(
                outputs["flight_logit"].squeeze(-1).cpu().numpy()
            )
            flights.append(flight.cpu().numpy())
            studies.append(study.cpu().numpy())
            strains.append(strain.cpu().numpy())
            sexes.append(sex.cpu().numpy())
            euths.append(batch["euth"].to(device).cpu().numpy())
            tissues.append(tissue.cpu().numpy())

    logits_arr = np.concatenate(logits_list)
    return {
        "z":       np.concatenate(zs),
        "logits":  logits_arr,
        "probs":   1 / (1 + np.exp(-logits_arr)),
        "flight":  np.concatenate(flights),
        "study":   np.concatenate(studies),
        "strain":  np.concatenate(strains),
        "sex":     np.concatenate(sexes),
        "euth":    np.concatenate(euths),
        "tissue":  np.concatenate(tissues),
    }


# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(model, test_loader, device="cpu"):
    results  = get_latent_representations(model, test_loader, device)
    auroc    = roc_auc_score(results["flight"], results["probs"])
    preds    = (results["probs"] > 0.5).astype(int)
    accuracy = accuracy_score(results["flight"], preds)
    n_flight = int((results["flight"] == 1).sum())
    n_ground = int((results["flight"] == 0).sum())

    print("\n=== Test Set Evaluation ===")
    print("  Samples:  " + str(len(results["flight"])) +
          " (" + str(n_flight) + " flight / " + str(n_ground) + " ground)")
    print(f"  AUROC:    {auroc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    return {"auroc": auroc, "accuracy": accuracy,
            "n_flight": n_flight, "n_ground": n_ground,
            "n_total": len(results["flight"])}


# ---------------------------------------------------------------------------
# Shared scatter plot helper
# ---------------------------------------------------------------------------

def _scatter_embedding(embedding, values, color_by, label_encoder,
                       xlabel, ylabel, title, save_path,
                       categorical=False):
    """
    Draw and save a 2D scatter plot colored by metadata values.
    Set categorical=True to force discrete coloring even for
    variables with many unique values (e.g. study with 72 classes).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    unique_vals = np.unique(values)
    n_unique    = len(unique_vals)

    fig, ax = plt.subplots(figsize=(9, 7))

    if categorical or n_unique <= 20:
        # assign each unique value a color from a qualitative palette
        # use tab20 for <= 20, HSV-spread for larger sets
        if n_unique <= 20:
            palette = [cm.get_cmap("tab20")(i / 20) for i in range(n_unique)]
        else:
            palette = [cm.get_cmap("hsv")(i / n_unique) for i in range(n_unique)]

        val_to_color = {v: palette[i] for i, v in enumerate(unique_vals)}
        colors       = [val_to_color[v] for v in values]

        ax.scatter(embedding[:, 0], embedding[:, 1],
                   c=colors, s=10, alpha=0.7)

        # legend patches — skip legend for > 40 classes (too crowded)
        if n_unique <= 40:
            from matplotlib.patches import Patch
            if label_encoder is not None:
                labels = label_encoder.classes_
            else:
                labels = [str(v) for v in unique_vals]
            handles = [Patch(color=palette[i], label=labels[i])
                       for i in range(n_unique)]
            legend_cols = max(1, n_unique // 20)
            ax.legend(handles=handles, title=color_by,
                      fontsize=6, title_fontsize=7,
                      loc="upper right", ncol=legend_cols,
                      framealpha=0.6)
        else:
            # too many classes for a legend — just note the count
            ax.set_title(title + " (" + str(n_unique) + " classes)")
    else:
        # continuous colormap (e.g. for a truly numeric variable)
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=values, cmap="viridis", s=10, alpha=0.7)
        plt.colorbar(sc, ax=ax, label=color_by)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print("  Saved: " + str(save_path))
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# UMAP visualization
# ---------------------------------------------------------------------------

def plot_latent_umap(latent_dict, color_by, label_encoder=None,
                     save_path=None, title=None, categorical=False):
    try:
        import umap
    except ImportError:
        raise ImportError("pip install umap-learn")

    print("Fitting UMAP (color_by=" + color_by + ")...")
    reducer   = umap.UMAP(n_components=2, random_state=42,
                          n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent_dict["z"])
    _scatter_embedding(
        embedding, latent_dict[color_by], color_by, label_encoder,
        xlabel="UMAP 1", ylabel="UMAP 2",
        title=title or ("Latent Space (UMAP) - " + color_by),
        save_path=save_path,
        categorical=categorical,
    )
    return embedding


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------

def plot_latent_tsne(latent_dict, color_by, label_encoder=None,
                     save_path=None, title=None,
                     perplexity=30, n_iter=1000, categorical=False):
    from sklearn.manifold import TSNE

    print("Fitting t-SNE (color_by=" + color_by + ")...")
    tsne      = TSNE(n_components=2, perplexity=perplexity,
                     n_iter=n_iter, random_state=42, n_jobs=-1)
    embedding = tsne.fit_transform(latent_dict["z"])
    _scatter_embedding(
        embedding, latent_dict[color_by], color_by, label_encoder,
        xlabel="t-SNE 1", ylabel="t-SNE 2",
        title=title or ("Latent Space (t-SNE) - " + color_by),
        save_path=save_path,
        categorical=categorical,
    )
    return embedding


# ---------------------------------------------------------------------------
# Gene attribution
# ---------------------------------------------------------------------------

def _is_characterized(symbol):
    """Return True if gene is characterized (not Gm*, *-ps, *Rik)."""
    if symbol.startswith("Gm") and symbol[2:].split("-")[0].isdigit():
        return False
    if "-ps" in symbol:
        return False
    if "Rik" in symbol:
        return False
    return True


def _compute_attribution(model, dataloader, device):
    """Gradient of flight_logit w.r.t. input, averaged over samples."""
    model.eval()
    all_grads = []
    for batch in dataloader:
        x      = batch["x"].to(device).requires_grad_(True)
        strain = batch["strain"].to(device)
        sex    = batch["sex"].to(device)
        study  = batch["study"].to(device)
        tissue = batch["tissue"].to(device)
        flight = batch["flight"].to(device)
        euth = batch["euth"].to(device)

        outputs = model(x, strain, sex, study, tissue, euth, flight)
        outputs["flight_logit"].sum().backward()
        all_grads.append(x.grad.abs().mean(dim=0).detach().cpu().numpy())
        model.zero_grad()
    return np.stack(all_grads).mean(axis=0)


def gene_attribution(model, dataloader, gene_symbols, ensembl_ids,
                     device="cpu", n_top=200, tissue_label=None):
    mean_attr = _compute_attribution(model, dataloader, device)
    top_idx   = np.argsort(mean_attr)[::-1][:n_top]
    df = pd.DataFrame({
        "rank":              range(1, n_top + 1),
        "ensembl_id":        ensembl_ids[top_idx],
        "symbol":            gene_symbols[top_idx],
        "attribution_score": mean_attr[top_idx],
    })
    if tissue_label is not None:
        df.insert(0, "tissue", tissue_label)
    return df


def clean_attribution(df):
    """Remove pseudogenes and uncharacterized genes, re-rank."""
    df_clean = df[df["symbol"].apply(_is_characterized)].copy()
    df_clean = df_clean.reset_index(drop=True)
    df_clean["rank"] = range(1, len(df_clean) + 1)
    return df_clean


def tissue_stratified_attribution(model, dataset, device="cpu",
                                  n_top=100, min_flight_samples=5):
    """Run gene attribution separately per tissue type."""
    all_dfs = []
    for tissue_idx, tissue_name in enumerate(dataset.tissue_enc.classes_):
        mask     = dataset.tissue_ids == tissue_idx
        n_flight = int((dataset.flight[mask] == 1).sum())
        n_ground = int((dataset.flight[mask] == 0).sum())
        if n_flight < min_flight_samples or n_ground < min_flight_samples:
            print("  Skipping " + tissue_name +
                  " (" + str(n_flight) + " flight / " + str(n_ground) + " ground)")
            continue
        indices = np.where(mask)[0]
        loader  = DataLoader(Subset(dataset, indices),
                             batch_size=32, shuffle=False, num_workers=2)
        print("  " + tissue_name + ": " + str(n_flight) +
              " flight / " + str(n_ground) + " ground")
        df = gene_attribution(
            model, loader,
            gene_symbols=dataset.gene_symbols,
            ensembl_ids=dataset.ensembl_ids,
            device=device,
            n_top=n_top,
            tissue_label=tissue_name,
        )
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)



# ---------------------------------------------------------------------------
# Tissue-stratified differential expression
# ---------------------------------------------------------------------------

def tissue_de_analysis(
    dataset,
    out_dir,
    min_samples: int = 5,
    n_top: int = 100,
):
    """
    For each tissue compute mean log fold change between spaceflight and
    ground control samples directly from measured expression data.

    This is the correct approach for tissue-specific gene discovery.
    Gradient attribution (tissue_stratified_attribution) finds genes that
    drive the global spaceflight classifier — this function finds genes
    that are actually differentially expressed within each tissue.

    Method:
      - log fold change = mean(log1p CPM flight) - mean(log1p CPM ground)
      - Welch t-test per gene for ranking
      - Filter genes expressed in < 20% of either group
      - Rank by abs(lfc), save top n_top per tissue

    Outputs:
      out_dir/de_by_tissue/<tissue>_de.csv   per-tissue DE genes
      out_dir/de_by_tissue/all_tissues_de.csv  combined
    """
    from scipy import stats

    de_dir = Path(out_dir) / "de_by_tissue"
    de_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for tissue_idx, tissue_name in enumerate(dataset.tissue_enc.classes_):
        mask        = dataset.tissue_ids == tissue_idx
        flight_mask = mask & (dataset.flight == 1)
        ground_mask = mask & (dataset.flight == 0)

        n_flight = int(flight_mask.sum())
        n_ground = int(ground_mask.sum())

        if n_flight < min_samples or n_ground < min_samples:
            print("  Skipping " + tissue_name +
                  " (" + str(n_flight) + "f/" + str(n_ground) + "g)")
            continue

        x_flight = dataset.x[flight_mask]   # (n_flight, n_genes)
        x_ground = dataset.x[ground_mask]   # (n_ground, n_genes)

        mean_flight = x_flight.mean(axis=0)
        mean_ground = x_ground.mean(axis=0)
        lfc         = mean_flight - mean_ground

        # Welch t-test per gene
        t_stats, pvals = stats.ttest_ind(
            x_flight, x_ground, axis=0, equal_var=False
        )

        # expression filter: expressed in >= 20% of either group
        expressed = (
            (x_flight > 0).mean(axis=0) >= 0.2
        ) | (
            (x_ground > 0).mean(axis=0) >= 0.2
        )

        df = pd.DataFrame({
            "ensembl_id":   dataset.ensembl_ids,
            "symbol":       dataset.gene_symbols,
            "mean_flight":  mean_flight,
            "mean_ground":  mean_ground,
            "lfc":          lfc,
            "abs_lfc":      np.abs(lfc),
            "pvalue":       pvals,
        })

        # filter low-expression genes and sort by abs lfc
        df = df[expressed].sort_values("abs_lfc", ascending=False)
        df.insert(0, "tissue", tissue_name)
        df.insert(0, "rank", range(1, len(df) + 1))

        top_df = df.head(n_top)
        fname  = tissue_name.lower().replace(" ", "_") + "_de.csv"
        top_df.to_csv(de_dir / fname, index=False)
        all_rows.append(top_df)

        top_gene = df.iloc[0]
        print("  " + tissue_name + " (" + str(n_flight) + "f/" +
              str(n_ground) + "g)  top: " + top_gene["symbol"] +
              " lfc=" + f"{top_gene['lfc']:.3f}" +
              " p=" + f"{top_gene['pvalue']:.2e}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(de_dir / "all_tissues_de.csv", index=False)
        print("  Saved: " + str(de_dir / "all_tissues_de.csv"))
    else:
        print("  No tissues had enough samples for DE analysis.")

# ---------------------------------------------------------------------------
# Pathway enrichment
# ---------------------------------------------------------------------------

ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2026",
    "KEGG_2019_Mouse",
    "Reactome_Pathways_2024",
    "WikiPathways_2024_Mouse",
]


def pathway_enrichment(gene_list, tissue_label, out_dir,
                       libraries=None, cutoff=0.05):
    """Run Enrichr on a gene list. Requires internet access."""
    if libraries is None:
        libraries = ENRICHR_LIBRARIES
    try:
        import gseapy as gp
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Install gseapy: pip install gseapy")
        return {}

    tissue_slug = tissue_label.lower().replace(" ", "_")
    enrich_dir  = Path(out_dir) / "enrichment" / tissue_slug
    enrich_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for lib in libraries:
        print("  Enrichr: " + tissue_label + " - " + lib)
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=lib,
                organism="mouse",
                outdir=None,
                verbose=False,
            )
            df = enr.results
            if df is None or df.empty:
                print("    No results returned")
                continue

            df = df[df["Adjusted P-value"] < cutoff].copy()
            df = df.sort_values("Adjusted P-value")
            if df.empty:
                print("    No significant terms (adj p < " + str(cutoff) + ")")
                continue

            lib_slug = lib.replace(" ", "_").replace("/", "_")
            df.to_csv(enrich_dir / (lib_slug + ".csv"), index=False)

            # bar plot — top 15 terms
            plot_df = df.head(15).copy()
            plot_df["neg_log10_p"] = -np.log10(
                plot_df["Adjusted P-value"].clip(lower=1e-10)
            )
            plot_df["Term"] = plot_df["Term"].str[:60]

            fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
            ax.barh(plot_df["Term"][::-1],
                    plot_df["neg_log10_p"][::-1],
                    color="steelblue", edgecolor="white")
            ax.axvline(x=-np.log10(cutoff), color="red",
                       linestyle="--", linewidth=0.8,
                       label="p=" + str(cutoff))
            ax.set_xlabel("-log10(adjusted p-value)")
            ax.set_title(tissue_label + " - " + lib +
                         " (top " + str(len(plot_df)) + " terms)")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(enrich_dir / (lib_slug + ".png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

            print("    " + str(len(df)) + " significant terms")
            results[lib] = df

        except Exception as e:
            print("    Failed: " + str(e))
            continue

    return results


def run_pathway_enrichment_all_tissues(tissue_attr_df, out_dir,
                                       n_genes=100, cutoff=0.05):
    """Run pathway enrichment for every tissue. Saves summary CSV."""
    tissues = tissue_attr_df["tissue"].unique()
    print("\nPathway enrichment: " + str(len(tissues)) +
          " tissues, top " + str(n_genes) +
          " genes, adj p < " + str(cutoff))

    summary_rows = []
    for tissue_name in sorted(tissues):
        grp = tissue_attr_df[tissue_attr_df["tissue"] == tissue_name]
        # support both attribution (attribution_score) and DE (abs_lfc) ranking
        if "abs_lfc" in grp.columns:
            grp_sorted = grp.sort_values("abs_lfc", ascending=False)
        else:
            grp_sorted = grp.sort_values("attribution_score", ascending=False)
        grp_clean = clean_attribution(grp_sorted.drop(columns="tissue", errors="ignore"))
        gene_list = grp_clean["symbol"].head(n_genes).tolist()

        if len(gene_list) < 10:
            print("  Skipping " + tissue_name +
                  " - fewer than 10 characterized genes")
            continue

        enr_results = pathway_enrichment(
            gene_list=gene_list,
            tissue_label=tissue_name,
            out_dir=out_dir,
            cutoff=cutoff,
        )
        for lib, df in enr_results.items():
            summary_rows.append({
                "tissue":         tissue_name,
                "library":        lib,
                "n_sig_terms":    len(df),
                "top_term":       df.iloc[0]["Term"] if not df.empty else "",
                "top_term_adj_p": (df.iloc[0]["Adjusted P-value"]
                                   if not df.empty else None),
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(Path(out_dir) / "enrichment_summary.csv", index=False)
        print("\nEnrichment summary saved to: " +
              str(out_dir) + "/enrichment_summary.csv")
        print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset and model ---
    print("Loading dataset...")
    dataset       = SpaceflightDataset(args.data)
    _, _, test_loader = make_dataloaders(dataset, batch_size=64, num_workers=4)
    full_loader   = DataLoader(dataset, batch_size=64,
                               shuffle=False, num_workers=4)
    model         = load_model(args.checkpoint, dataset, device)

    # --- Test set evaluation ---
    metrics = evaluate_test_set(model, test_loader, device)
    with open(out_dir / "test_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(str(k) + ": " + str(v) + "\n")
    print("  Saved: " + str(out_dir / "test_metrics.txt"))

    # --- Extract latents ---
    print("\nExtracting latent representations...")
    latent_dict = get_latent_representations(model, full_loader, device)
    np.savez(out_dir / "latent_all.npz", **latent_dict)
    print("  Saved: " + str(out_dir / "latent_all.npz"))

    # --- UMAP and t-SNE plots ---
    # (color_by, label_encoder, title, categorical)
    plot_configs = [
        ("flight", None,                "Spaceflight vs Ground Control", True),
        ("study",  dataset.study_enc,   "Study ID",                      True),
        ("strain", dataset.strain_enc,  "Mouse Strain",                  True),
        ("sex",    dataset.sex_enc,     "Sex",                           True),
        ("tissue", dataset.tissue_enc,  "Tissue Type",                   True),
        ("euth",   dataset.euth_enc,    "Euthanasia Method",             True),
    ]

    print("\nGenerating UMAP plots...")
    for color_by, enc, title, cat in plot_configs:
        plot_latent_umap(
            latent_dict, color_by=color_by,
            label_encoder=enc,
            save_path=str(out_dir / ("umap_" + color_by + ".png")),
            title="Latent Space (UMAP) - " + title,
            categorical=cat,
        )

    print("\nGenerating t-SNE plots...")
    for color_by, enc, title, cat in plot_configs:
        plot_latent_tsne(
            latent_dict, color_by=color_by,
            label_encoder=enc,
            save_path=str(out_dir / ("tsne_" + color_by + ".png")),
            title="Latent Space (t-SNE) - " + title,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_n_iter,
            categorical=cat,
        )


    # --- Global gene attribution ---
    print("\nComputing global gene attribution...")
    attr_df = gene_attribution(
        model, full_loader,
        gene_symbols=dataset.gene_symbols,
        ensembl_ids=dataset.ensembl_ids,
        device=device, n_top=200,
    )
    attr_df.to_csv(out_dir / "gene_attribution.csv", index=False)

    attr_clean = clean_attribution(attr_df)
    attr_clean.to_csv(out_dir / "gene_attribution_clean.csv", index=False)
    print("  Saved: " + str(out_dir / "gene_attribution_clean.csv"))
    print("\nTop 20 spaceflight-associated genes (characterized only):")
    print(attr_clean.head(20).to_string(index=False))

    # --- Tissue-stratified attribution ---
    print("\nComputing tissue-stratified gene attribution...")
    tissue_attr_df = tissue_stratified_attribution(
        model, dataset, device=device, n_top=100, min_flight_samples=5
    )
    tissue_attr_df.to_csv(out_dir / "gene_attribution_by_tissue.csv",
                          index=False)
    print("  Saved: " + str(out_dir / "gene_attribution_by_tissue.csv"))

    tissue_out_dir = out_dir / "by_tissue"
    tissue_out_dir.mkdir(exist_ok=True)
    for tissue_name, grp in tissue_attr_df.groupby("tissue"):
        clean = clean_attribution(grp.drop(columns="tissue"))
        fname = tissue_name.lower().replace(" ", "_") + ".csv"
        clean.to_csv(tissue_out_dir / fname, index=False)
    print("  Per-tissue CSVs saved to: " + str(tissue_out_dir))

    # --- Tissue-stratified DE analysis ---
    print("\nComputing tissue-stratified differential expression...")
    tissue_de_analysis(
        dataset=dataset,
        out_dir=out_dir,
        min_samples=5,
        n_top=args.n_de_genes,
    )

    # --- Pathway enrichment (uses DE genes, not attribution) ---
    if not args.skip_enrichment:
        print("\nRunning pathway enrichment on DE genes...")
        de_dir = out_dir / "de_by_tissue"
        de_rows = []
        for tissue_name in dataset.tissue_enc.classes_:
            fname = de_dir / (tissue_name.lower().replace(" ", "_") + "_de.csv")
            if fname.exists():
                df = pd.read_csv(fname)
                #df.insert(0, "tissue", tissue_name)
                de_rows.append(df)
        if de_rows:
            de_combined = pd.concat(de_rows, ignore_index=True)
            run_pathway_enrichment_all_tissues(
                tissue_attr_df=de_combined,
                out_dir=out_dir,
                n_genes=args.enrichment_genes,
                cutoff=args.enrichment_cutoff,
            )
        else:
            print("  No DE results found — skipping enrichment")
    else:
        print("\nSkipping pathway enrichment (--skip_enrichment)")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and analyze Spaceflight cVAE"
    )
    parser.add_argument("--checkpoint",        type=str,   required=True)
    parser.add_argument("--data",              type=str,   required=True)
    parser.add_argument("--output_dir",        type=str,   default="results")
    parser.add_argument("--skip_enrichment",   action="store_true",
                        help="Skip Enrichr (use when offline)")
    parser.add_argument("--n_de_genes",        type=int,   default=200,
                        help="Top DE genes per tissue to save")
    parser.add_argument("--enrichment_genes",  type=int,   default=100,
                        help="Top genes per tissue for enrichment")
    parser.add_argument("--enrichment_cutoff", type=float, default=0.05,
                        help="Adjusted p-value cutoff")
    parser.add_argument("--tsne_perplexity",   type=float, default=30.0,
                        help="t-SNE perplexity (default 30)")
    parser.add_argument("--tsne_n_iter",       type=int,   default=1000,
                        help="t-SNE iterations (default 1000)")
    args = parser.parse_args()
    run_analysis(args)
